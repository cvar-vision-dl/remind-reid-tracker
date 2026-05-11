from __future__ import annotations

import os
import sys
import gc
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
import json
from contextlib import nullcontext

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None

try:
    import psutil
except Exception:
    psutil = None


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
PROJECT_DIR = SRC_DIR.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from testing.common.generic_tracking_metrics import TrackingOnlyEvaluator
from testing.common.generic_tracking_reporting import (
    build_generic_console_report,
    write_csv,
    write_json,
    write_text,
)
from testing.davis_gt import DavisGroundTruthLoader
from config.config_loader import Config
from utils.io import list_image_files, parse_frame_id
from utils.logging import default_run_artifact_dir
from utils.scannetpp_tar import resolve_scene_annotations_tar_path, resolve_scene_tar_path
import detection.davis_segmenter as davis_segmenter_module
from detection.davis_segmenter import DavisSegmenter
from testing import davis_gt as davis_gt_module

import tarfile
from contextlib import contextmanager


@dataclass
class SimpleMaskDetection:
    detection_id: int
    mask: np.ndarray
    bbox: tuple[int, int, int, int] | None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _normalize_tar_member_name(name: str) -> str:
    return str(name or "").strip().lstrip("./").rstrip("/")


def _normalize_mask_variant(mask_variant: str) -> str:
    variant = str(mask_variant or "").strip().lower() or "benchmark"
    return "benchmark_instance" if variant == "benchmark" else variant


def _normalize_davis_variant(mask_variant: str) -> str:
    return "bench" if _normalize_mask_variant(mask_variant) == "benchmark_instance" else "raw"


def _scene_tar_path(root: Path, scene_id: str) -> Path:
    return (root / f"{scene_id}.tar").resolve()


def _build_tar_member_index(tar_path: Path) -> dict[str, str]:
    members_by_rel: dict[str, str] = {}
    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        top_levels = {
            Path(_normalize_tar_member_name(member.name)).parts[0]
            for member in members
            if _normalize_tar_member_name(member.name)
        }
        root_prefix = next(iter(top_levels)) if len(top_levels) == 1 else None

        for member in members:
            raw_name = _normalize_tar_member_name(member.name)
            if not raw_name:
                continue
            parts = Path(raw_name).parts
            if root_prefix and parts and parts[0] == root_prefix:
                parts = parts[1:]
            if not parts:
                continue
            rel_name = "/".join(parts)
            members_by_rel[rel_name] = str(member.name)
    return members_by_rel


@dataclass
class TarSceneBundle:
    scene_id: str
    data_tar_path: Path
    annotations_tar_path: Path
    image_subdir: str
    mask_variant: str
    meta_rel_path: str
    meta: dict[str, Any]
    frame_names: list[str]
    data_members_by_rel: dict[str, str]
    annotation_members_by_rel: dict[str, str]
    _data_tar: tarfile.TarFile | None = None
    _annotations_tar: tarfile.TarFile | None = None

    def data_member_rel(self, frame_name: str) -> str:
        return f"{self.image_subdir}/{frame_name}".strip("/")

    def annotation_member_rel(self, frame_id: int) -> str:
        return f"annotations/{self.mask_variant}/frame_{int(frame_id):06d}.png"

    def raw_meta_rel_path(self) -> str:
        return "meta_raw.json"

    def _get_data_tar(self) -> tarfile.TarFile:
        if self._data_tar is None:
            self._data_tar = tarfile.open(self.data_tar_path, "r:*")
        return self._data_tar

    def _get_annotations_tar(self) -> tarfile.TarFile:
        if self._annotations_tar is None:
            self._annotations_tar = tarfile.open(self.annotations_tar_path, "r:*")
        return self._annotations_tar

    def read_data_member_bytes(self, rel_path: str) -> bytes:
        rel = str(rel_path or "").strip().strip("/")
        member_name = self.data_members_by_rel.get(rel, None)
        if member_name is None:
            raise FileNotFoundError(f"{rel} does not exist in {self.data_tar_path}")
        extracted = self._get_data_tar().extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"No se pudo abrir {rel} en {self.data_tar_path}")
        return extracted.read()

    def read_annotations_member_bytes(self, rel_path: str) -> bytes:
        rel = str(rel_path or "").strip().strip("/")
        member_name = self.annotation_members_by_rel.get(rel, None)
        if member_name is None:
            raise FileNotFoundError(f"{rel} does not exist in {self.annotations_tar_path}")
        extracted = self._get_annotations_tar().extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"No se pudo abrir {rel} en {self.annotations_tar_path}")
        return extracted.read()

    def read_annotations_json(self, rel_path: str) -> dict[str, Any]:
        payload = self.read_annotations_member_bytes(rel_path)
        data = json.loads(payload.decode("utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"JSON invalido en {self.annotations_tar_path}:{rel_path}")
        return data

    def close(self) -> None:
        if self._data_tar is not None:
            self._data_tar.close()
            self._data_tar = None
        if self._annotations_tar is not None:
            self._annotations_tar.close()
            self._annotations_tar = None


def _build_scene_bundle(
    *,
    scene_id: str,
    data_tar_root: Path,
    annotations_tar_root: Path,
    mask_variant: str,
    image_subdir: str,
) -> TarSceneBundle:
    normalized_variant = _normalize_mask_variant(mask_variant)
    data_tar_path = _scene_tar_path(data_tar_root, scene_id)
    annotations_tar_path = _scene_tar_path(annotations_tar_root, scene_id)
    if not data_tar_path.is_file():
        raise FileNotFoundError(f"Missing data tar for {scene_id}: {data_tar_path}")
    if not annotations_tar_path.is_file():
        raise FileNotFoundError(f"Missing annotations tar for {scene_id}: {annotations_tar_path}")

    data_members_by_rel = _build_tar_member_index(data_tar_path)
    annotation_members_by_rel = _build_tar_member_index(annotations_tar_path)

    meta_rel_path = f"meta_{normalized_variant}.json"
    if meta_rel_path not in annotation_members_by_rel:
        raise FileNotFoundError(f"Falta {meta_rel_path} en {annotations_tar_path}")

    with tarfile.open(annotations_tar_path, "r:*") as tf:
        extracted = tf.extractfile(annotation_members_by_rel[meta_rel_path])
        if extracted is None:
            raise FileNotFoundError(f"No se pudo abrir {meta_rel_path} en {annotations_tar_path}")
        meta = json.loads(extracted.read().decode("utf-8")) or {}
    if not isinstance(meta, dict):
        raise ValueError(f"Meta invalido en {annotations_tar_path}:{meta_rel_path}")

    raw_frame_names = meta.get("frame_names", None)
    if isinstance(raw_frame_names, list) and raw_frame_names:
        frame_names = [str(name).strip() for name in raw_frame_names if str(name).strip()]
    else:
        prefix = f"{str(image_subdir).strip().strip('/')}/"
        frame_names = sorted(
            rel_name[len(prefix):]
            for rel_name in data_members_by_rel.keys()
            if rel_name.startswith(prefix)
            and Path(rel_name).suffix.lower() in IMAGE_EXTS
            and "/" not in rel_name[len(prefix):]
        )
    if not frame_names:
        raise RuntimeError(f"No se resolvieron frames en {data_tar_path}")

    missing_frames = [
        frame_name
        for frame_name in frame_names
        if f"{str(image_subdir).strip().strip('/')}/{frame_name}" not in data_members_by_rel
    ]
    if missing_frames:
        preview = ", ".join(missing_frames[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_frames)} frames listed in {data_tar_path}. Examples: {preview}"
        )

    annotation_prefix = f"annotations/{normalized_variant}/"
    missing_masks = [
        frame_idx
        for frame_idx in range(len(frame_names))
        if f"{annotation_prefix}frame_{int(frame_idx):06d}.png" not in annotation_members_by_rel
    ]
    if missing_masks:
        preview = ", ".join(str(idx) for idx in missing_masks[:5])
        raise FileNotFoundError(
            f"Missing {len(missing_masks)} masks in {annotations_tar_path}. frame_id examples: {preview}"
        )

    return TarSceneBundle(
        scene_id=str(scene_id),
        data_tar_path=data_tar_path,
        annotations_tar_path=annotations_tar_path,
        image_subdir=str(image_subdir).strip().strip("/"),
        mask_variant=normalized_variant,
        meta_rel_path=meta_rel_path,
        meta=meta,
        frame_names=frame_names,
        data_members_by_rel=data_members_by_rel,
        annotation_members_by_rel=annotation_members_by_rel,
    )


class TarFrameSource:
    def __init__(self, bundle: TarSceneBundle):
        self.bundle = bundle

    def read_bgr(self, frame_name: str) -> np.ndarray | None:
        payload = self.bundle.read_data_member_bytes(self.bundle.data_member_rel(frame_name))
        arr = np.frombuffer(payload, dtype=np.uint8)
        if arr.size <= 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)


class TarDavisSegmenter(DavisSegmenter):
    def resolve_tar_bundle(self) -> TarSceneBundle:
        bundle = self.davis_cfg.get("tar_scene_bundle", None)
        if not isinstance(bundle, TarSceneBundle):
            raise RuntimeError("TarDavisSegmenter requires config['davis']['tar_scene_bundle'].")
        return bundle

    def load_model(self) -> None:
        bundle = self.resolve_tar_bundle()
        meta = dict(bundle.meta or {})
        self.meta_path = Path(f"{bundle.annotations_tar_path}!{bundle.meta_rel_path}")
        self.sequence_name_resolved = str(meta.get("scene_id", bundle.scene_id) or bundle.scene_id)
        self.annotations_dir = Path(f"{bundle.annotations_tar_path}!annotations/{bundle.mask_variant}")
        self.instance_id_to_label = self.build_instance_id_to_label(meta)

        if bundle.mask_variant == "benchmark_instance" and bundle.raw_meta_rel_path() in bundle.annotation_members_by_rel:
            raw_meta = bundle.read_annotations_json(bundle.raw_meta_rel_path())
            self.instance_id_to_original_label = self.build_instance_id_to_label(raw_meta)
        else:
            self.instance_id_to_original_label = dict(self.instance_id_to_label)

        self.instance_id_to_original_class_name = {
            int(instance_id): str(class_name)
            for instance_id, class_name in (
                (
                    int(instance_id),
                    self.extract_class_name(label),
                )
                for instance_id, label in (self.instance_id_to_original_label or {}).items()
            )
            if class_name is not None
        }

        class_names = sorted(
            {
                self.extract_class_name(label)
                for label in self.instance_id_to_label.values()
                if self.extract_class_name(label) is not None
            }
        )
        self.class_id_to_name = {idx: name for idx, name in enumerate(class_names)}
        self.class_name_to_id = {name: idx for idx, name in self.class_id_to_name.items()}

        self.instance_id_to_class_id = {}
        for instance_id, label in self.instance_id_to_label.items():
            class_name = self.extract_class_name(label)
            if class_name is None:
                continue
            class_id = self.class_name_to_id.get(class_name, None)
            if class_id is not None:
                self.instance_id_to_class_id[int(instance_id)] = int(class_id)

        self.prefetch_enabled = False
        self._prefetch_executor = None
        self._prefetch_future = None
        self._prefetch_frame_id = None

    def schedule_prefetch(self, frame_id: int) -> None:
        return

    def read_annotation_mask_from_path(self, path: Path) -> np.ndarray | None:
        return None

    def read_annotation_mask(self, frame_id: int) -> np.ndarray | None:
        bundle = self.resolve_tar_bundle()
        rel_path = bundle.annotation_member_rel(int(frame_id))
        try:
            payload = bundle.read_annotations_member_bytes(rel_path)
        except FileNotFoundError:
            return None
        arr = np.frombuffer(payload, dtype=np.uint8)
        if arr.size <= 0:
            return None
        mask = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask


@contextmanager
def _patched_tar_davis_segmenter():
    original_detector_cls = davis_segmenter_module.DavisSegmenter
    original_gt_cls = davis_gt_module.DavisSegmenter
    davis_segmenter_module.DavisSegmenter = TarDavisSegmenter
    davis_gt_module.DavisSegmenter = TarDavisSegmenter
    try:
        yield
    finally:
        davis_segmenter_module.DavisSegmenter = original_detector_cls
        davis_gt_module.DavisSegmenter = original_gt_cls


def _sync_cuda() -> None:
    if torch is not None and bool(torch.cuda.is_available()):
        torch.cuda.synchronize()


def _inference_context():
    if torch is None:
        return nullcontext()
    return torch.inference_mode()


def _cuda_autocast_context():
    if torch is None:
        return nullcontext()
    return torch.cuda.amp.autocast(enabled=bool(torch.cuda.is_available()))


def make_process_handle():
    if psutil is None:
        return None
    try:
        return psutil.Process(os.getpid())
    except Exception:
        return None


def read_process_rss_bytes(process) -> int | None:
    if process is None:
        return None
    try:
        return int(process.memory_info().rss)
    except Exception:
        return None


def reset_cuda_peak_memory_stats() -> None:
    if torch is None or not bool(torch.cuda.is_available()):
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def capture_cuda_memory_stats() -> dict[str, int | None]:
    if torch is None or not bool(torch.cuda.is_available()):
        return {}
    try:
        return {
            "mem_gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
            "mem_gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
            "mem_gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "mem_gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return {}


def release_cuda_scene_resources() -> None:
    if torch is None or not bool(torch.cuda.is_available()):
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass


def build_runtime_memory_telemetry(
    *,
    rss_before: int | None,
    rss_after_read: int | None,
    rss_after_pipeline: int | None,
    rss_after_eval: int | None,
    gpu_after_pipeline: dict[str, int | None] | None = None,
    gpu_after_eval: dict[str, int | None] | None = None,
) -> dict[str, int | None]:
    rss_samples = [x for x in [rss_before, rss_after_read, rss_after_pipeline, rss_after_eval] if x is not None]
    pipeline_gpu = dict(gpu_after_pipeline or {})
    eval_gpu = dict(gpu_after_eval or {})
    return {
        "mem_process_rss_before_bytes": None if rss_before is None else int(rss_before),
        "mem_process_rss_after_read_bytes": None if rss_after_read is None else int(rss_after_read),
        "mem_process_rss_after_pipeline_bytes": None if rss_after_pipeline is None else int(rss_after_pipeline),
        "mem_process_rss_after_eval_bytes": None if rss_after_eval is None else int(rss_after_eval),
        "mem_process_rss_peak_approx_bytes": None if not rss_samples else int(max(rss_samples)),
        "mem_process_rss_delta_bytes": (
            None if rss_before is None or rss_after_eval is None else int(rss_after_eval - rss_before)
        ),
        "mem_gpu_allocated_after_pipeline_bytes": (
            None
            if pipeline_gpu.get("mem_gpu_allocated_bytes", None) is None
            else int(pipeline_gpu["mem_gpu_allocated_bytes"])
        ),
        "mem_gpu_reserved_after_pipeline_bytes": (
            None
            if pipeline_gpu.get("mem_gpu_reserved_bytes", None) is None
            else int(pipeline_gpu["mem_gpu_reserved_bytes"])
        ),
        "mem_gpu_allocated_after_eval_bytes": (
            None if eval_gpu.get("mem_gpu_allocated_bytes", None) is None else int(eval_gpu["mem_gpu_allocated_bytes"])
        ),
        "mem_gpu_reserved_after_eval_bytes": (
            None if eval_gpu.get("mem_gpu_reserved_bytes", None) is None else int(eval_gpu["mem_gpu_reserved_bytes"])
        ),
        "mem_gpu_peak_allocated_bytes": (
            None if eval_gpu.get("mem_gpu_peak_allocated_bytes", None) is None else int(eval_gpu["mem_gpu_peak_allocated_bytes"])
        ),
        "mem_gpu_peak_reserved_bytes": (
            None if eval_gpu.get("mem_gpu_peak_reserved_bytes", None) is None else int(eval_gpu["mem_gpu_peak_reserved_bytes"])
        ),
    }


def sanitize_name_for_path(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown_scene"
    import re
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "unknown_scene"


def _build_tar_input_source(
    *,
    scene_id: str,
    images_root_base: Path,
    mask_variant: str,
    image_subdir: str,
) -> dict[str, str] | None:
    data_tar_path = resolve_scene_tar_path(images_root_base=images_root_base, scene_id=scene_id)
    annotations_tar_path = resolve_scene_annotations_tar_path(images_root_base=images_root_base, scene_id=scene_id)
    if data_tar_path is None or annotations_tar_path is None:
        return None
    return {
        "mode": "external_scannetpp_tar",
        "sequence_name": str(scene_id),
        "image_subdir": str(image_subdir),
        "mask_variant": str(_normalize_mask_variant(mask_variant)),
        "data_tar_path": str(data_tar_path),
        "annotations_tar_path": str(annotations_tar_path),
    }


def resolve_testing_input_source(project_dir: str) -> dict[str, str]:
    project_path = Path(project_dir).resolve()

    explicit_frames_dir = os.environ.get("REMIND_INPUT_FRAMES_DIR", "").strip()
    explicit_meta_path = os.environ.get("REMIND_DAVIS_META_PATH", "").strip()
    explicit_annotations_dir = os.environ.get("REMIND_DAVIS_ANNOTATIONS_DIR", "").strip()
    explicit_sequence_name = os.environ.get("REMIND_DAVIS_SEQUENCE_NAME", "").strip()

    if explicit_frames_dir:
        frames_dir = Path(explicit_frames_dir).expanduser().resolve()
        return {
            "mode": "explicit_env",
            "frames_dir": str(frames_dir),
            "sequence_name": explicit_sequence_name or frames_dir.name,
            "davis_meta_path": explicit_meta_path,
            "davis_annotations_dir": explicit_annotations_dir,
            "image_subdir": "",
        }

    local_scannetpp_root = SRC_DIR / "data" / "scannetpp_data"
    external_masks_root_base = Path(
        os.environ.get("REMIND_SCANNETPP_MASKS_ROOT", str(local_scannetpp_root))
    ).expanduser().resolve()
    external_images_root_base = Path(
        os.environ.get("REMIND_SCANNETPP_IMAGES_ROOT", str(local_scannetpp_root))
    ).expanduser().resolve()
    scene_id = os.environ.get("REMIND_SCENE_ID", "00a231a370").strip() or "00a231a370"
    mask_variant = os.environ.get("REMIND_MASK_VARIANT", "benchmark").strip().lower() or "benchmark"
    image_subdir = os.environ.get("REMIND_IMAGE_SUBDIR", "dslr/resized_images").strip() or "dslr/resized_images"
    if mask_variant == "benchmark":
        mask_variant = "benchmark_instance"

    external_masks_root = (external_masks_root_base / "2Dmasks" / scene_id).resolve()
    external_meta_path = (external_masks_root / f"meta_{mask_variant}.json").resolve()
    external_annotations_dir = (external_masks_root / "annotations" / mask_variant).resolve()
    external_frames_dir = (external_images_root_base / "data" / scene_id / image_subdir).resolve()
    external_ready = external_frames_dir.is_dir() and external_meta_path.is_file() and external_annotations_dir.is_dir()
    prefer_external = str(os.environ.get("REMIND_PREFER_EXTERNAL_SCENE", "1")).strip().lower() in {"1", "true", "yes", "on"}
    if prefer_external and external_ready:
        return {
            "mode": "external_scannetpp",
            "frames_dir": str(external_frames_dir),
            "sequence_name": scene_id,
            "davis_meta_path": str(external_meta_path),
            "davis_annotations_dir": str(external_annotations_dir),
            "image_subdir": str(image_subdir),
        }

    if prefer_external:
        tar_source = _build_tar_input_source(
            scene_id=scene_id,
            images_root_base=external_images_root_base,
            mask_variant=mask_variant,
            image_subdir=image_subdir,
        )
        if tar_source is not None:
            return tar_source

    local_frames_dir = (SRC_DIR / "data" / "framesCOMPLETO1").resolve()
    return {
        "mode": "local_fallback",
        "frames_dir": str(local_frames_dir),
        "sequence_name": local_frames_dir.name,
        "davis_meta_path": "",
        "davis_annotations_dir": "",
        "image_subdir": "",
    }


def resolve_frame_files_for_testing(frames_dir: str, *, davis_meta_path: str = "") -> tuple[list[str], bool]:
    meta_path = Path(str(davis_meta_path).strip()).expanduser() if str(davis_meta_path).strip() else None
    frames_root = Path(frames_dir).expanduser().resolve()

    if meta_path is not None and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) or {}
        except Exception:
            meta = {}
        frame_names = meta.get("frame_names", None)
        if isinstance(frame_names, list) and frame_names:
            out: list[str] = []
            missing: list[str] = []
            for raw_name in frame_names:
                name = str(raw_name).strip()
                if not name:
                    continue
                p = (frames_root / name).resolve()
                if p.is_file():
                    out.append(str(p))
                else:
                    missing.append(name)
            if missing:
                preview = ", ".join(missing[:5])
                raise FileNotFoundError(
                    f"Missing {len(missing)} meta frames in {frames_root}. Examples: {preview}"
                )
            if out:
                return out, True

    return list_image_files(str(frames_root)), False


def _ensure_d4sm_import_path() -> Path:
    d4sm_root = (PROJECT_DIR / "third_party" / "d4sm").resolve()
    if str(d4sm_root) not in sys.path:
        sys.path.insert(0, str(d4sm_root))
    return d4sm_root


def resolve_d4sm_runtime_config() -> dict[str, Any]:
    d4sm_root = _ensure_d4sm_import_path()
    model_size = os.environ.get("REMIND_D4SM_MODEL_SIZE", "large").strip().lower() or "large"
    checkpoint_dir = os.environ.get("REMIND_D4SM_CHECKPOINT_DIR", "").strip()
    if checkpoint_dir:
        checkpoint_dir = str(Path(checkpoint_dir).expanduser().resolve())
    else:
        checkpoint_dir = str((d4sm_root / "checkpoints").resolve())
    offload_state_to_cpu = str(os.environ.get("REMIND_D4SM_OFFLOAD_STATE_TO_CPU", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return {
        "d4sm_root": d4sm_root,
        "model_size": model_size,
        "checkpoint_dir": checkpoint_dir,
        "offload_state_to_cpu": offload_state_to_cpu,
    }


def create_d4sm_tracker(*, runtime_config: dict[str, Any] | None = None):
    if torch is None:
        raise RuntimeError("d4sm requires torch, but the current interpreter cannot import it.")
    if not bool(torch.cuda.is_available()):
        cuda_visible_devices = str(os.environ.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
        suffix = ""
        if cuda_visible_devices:
            suffix = f" CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}."
        raise RuntimeError(
            "d4sm requires a visible CUDA GPU to load SAM2/D4SM, "
            "but torch.cuda.is_available() == False."
            f"{suffix} Check `nvidia-smi`, the driver, the conda environment, and the selected GPU."
        )
    cfg = dict(runtime_config or resolve_d4sm_runtime_config())
    from tracking_wrapper_mot import DAM4SAMMOT

    return DAM4SAMMOT(
        model_size=str(cfg["model_size"]),
        checkpoint_dir=str(cfg["checkpoint_dir"]),
        offload_state_to_cpu=bool(cfg["offload_state_to_cpu"]),
    )


def reset_d4sm_tracker_scene_state(tracker: Any) -> None:
    tracker.img_width = None
    tracker.img_height = None
    tracker.frame_index = 0
    tracker.n_frames = None
    tracker.maskmem_pos_enc = None
    tracker.output_dict = {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
        "maskmem_pos_enc": None,
        "per_obj_dict": {},
    }
    tracker.mask_inputs_per_obj = {}
    tracker.output_dict_per_obj = {}
    tracker.temp_output_dict_per_obj = {}
    tracker.consolidated_frame_inds = {
        "cond_frame_outputs": set(),
        "non_cond_frame_outputs": set(),
    }
    tracker.obj_id_to_idx = OrderedDict()
    tracker.obj_idx_to_id = OrderedDict()
    tracker.obj_ids = []
    tracker.per_object_outputs_all = {}
    tracker.per_object_obj_ptr = {}
    tracker.next_obj_id = 1
    tracker.all_obj_ids = []
    tracker.object_sizes = []
    tracker.last_added = []
    tracker.add_to_drm_next = {}


def _compute_bbox_xyxy(mask_bool: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask_bool)
    if len(xs) <= 0 or len(ys) <= 0:
        return None
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _gt_object_to_full_mask(gt_obj: Any, frame_shape: tuple[int, int]) -> np.ndarray:
    full_mask = np.zeros(frame_shape[:2], dtype=bool)
    bbox = getattr(gt_obj, "bbox_xyxy", None)
    mask_local = getattr(gt_obj, "mask", None)
    if bbox is None or mask_local is None:
        return full_mask
    x1, y1, x2, y2 = [int(v) for v in bbox]
    full_mask[y1:y2, x1:x2] = np.asarray(mask_local).astype(bool, copy=False)
    return full_mask


def _build_detections(
    *,
    tracker_pred_masks: list[np.ndarray],
    tracker_pred_ids: list[int],
    appended_new_masks: list[np.ndarray],
    appended_new_pred_ids: list[int],
) -> tuple[list[SimpleMaskDetection], dict[int, int]]:
    detections: list[SimpleMaskDetection] = []
    det_to_pred_id: dict[int, int] = {}
    next_det_id = 1

    for pred_id, mask in zip(tracker_pred_ids, tracker_pred_masks):
        mask_bool = np.asarray(mask).astype(bool, copy=False)
        bbox = _compute_bbox_xyxy(mask_bool)
        if bbox is None:
            continue
        detections.append(SimpleMaskDetection(detection_id=int(next_det_id), mask=mask_bool, bbox=bbox))
        det_to_pred_id[int(next_det_id)] = int(pred_id)
        next_det_id += 1

    for pred_id, mask in zip(appended_new_pred_ids, appended_new_masks):
        mask_bool = np.asarray(mask).astype(bool, copy=False)
        bbox = _compute_bbox_xyxy(mask_bool)
        if bbox is None:
            continue
        detections.append(SimpleMaskDetection(detection_id=int(next_det_id), mask=mask_bool, bbox=bbox))
        det_to_pred_id[int(next_det_id)] = int(pred_id)
        next_det_id += 1

    return detections, det_to_pred_id


def _choose_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    suffix = 1
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _build_scene_output_dir(*, scene_tag: str) -> Path:
    outputs_root = (PROJECT_DIR / "outputs" / "d4sm").resolve()
    out_dir = Path(
        default_run_artifact_dir(
            str(outputs_root),
            group="testing",
            prefix=f"tracking_eval_d4sm_{scene_tag}",
        )
    )
    out_dir = _choose_unique_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def evaluate_scene(
    *,
    project_dir: Path,
    config_path: Path,
    input_source: dict[str, str],
    stable_min_frames: int,
    max_frames: int | None = None,
    tracker: Any | None = None,
) -> tuple[dict[str, Any], str]:
    if torch is None:
        raise RuntimeError("d4sm requires torch, but the current interpreter cannot import it.")
    _ = project_dir
    from hydra.core.global_hydra import GlobalHydra

    cfg = Config(default_config_path=config_path)
    config = cfg.to_dict()
    is_tar_source = str(input_source.get("mode", "") or "") == "external_scannetpp_tar"
    scene_bundle = None
    tar_frame_source = None
    tar_patch_ctx = nullcontext()

    if is_tar_source:
        scene_bundle = _build_scene_bundle(
            scene_id=str(input_source["sequence_name"]),
            data_tar_root=Path(input_source["data_tar_path"]).expanduser().resolve().parent,
            annotations_tar_root=Path(input_source["annotations_tar_path"]).expanduser().resolve().parent,
            mask_variant=str(input_source.get("mask_variant", "benchmark_instance")),
            image_subdir=str(input_source.get("image_subdir", "dslr/resized_images") or "dslr/resized_images"),
        )
        tar_frame_source = TarFrameSource(scene_bundle)
        davis_cfg = config.setdefault("davis", {})
        davis_cfg["sequence_name"] = str(scene_bundle.scene_id)
        davis_cfg["variant"] = _normalize_davis_variant(scene_bundle.mask_variant)
        davis_cfg["tar_scene_bundle"] = scene_bundle
        davis_cfg["prefetch_annotations"] = False
        frame_files = list(scene_bundle.frame_names)
        if max_frames is not None:
            frame_files = frame_files[: max(0, int(max_frames))]
        if not frame_files:
            raise RuntimeError(f"No frames to evaluate for {scene_bundle.scene_id}")
        use_sequential_frame_ids = True
        tar_patch_ctx = _patched_tar_davis_segmenter()
    else:
        config.setdefault("input", {})["frames_dir"] = input_source["frames_dir"]
        davis_cfg = config.setdefault("davis", {})
        davis_cfg["sequence_name"] = input_source["sequence_name"]
        if input_source.get("davis_meta_path"):
            davis_cfg["meta_path"] = input_source["davis_meta_path"]
        if input_source.get("davis_annotations_dir"):
            davis_cfg["annotations_dir"] = input_source["davis_annotations_dir"]
        frame_files, use_sequential_frame_ids = resolve_frame_files_for_testing(
            input_source["frames_dir"],
            davis_meta_path=input_source.get("davis_meta_path", ""),
        )
        if not frame_files:
            raise RuntimeError(f"No images found in {input_source['frames_dir']}")
        if max_frames is not None:
            frame_files = frame_files[: max(0, int(max_frames))]
    total_frames = int(len(frame_files))
    progress_every = 20

    print(
        f"[D4SM][scene={input_source['sequence_name']}] "
        f"start | frames={total_frames} | "
        f"image_subdir={input_source.get('image_subdir', '') or '-'} | "
        f"source_mode={input_source.get('mode', '') or '-'}"
    )

    tracker_owner = tracker is None
    if tracker_owner:
        tracker = create_d4sm_tracker()
    else:
        reset_d4sm_tracker_scene_state(tracker)
    tracker.n_frames = int(len(frame_files))

    dataset_gt_by_tracker_id: dict[int, int] = {}
    seen_gt_ids: set[int] = set()
    total_read_ms = 0.0
    total_pipeline_ms = 0.0
    total_gt_ms = 0.0
    total_eval_ms = 0.0
    total_post_ms = 0.0
    total_loop_ms = 0.0
    per_frame_timing_by_frame_id: dict[int, dict[str, float]] = {}
    per_frame_runtime_memory_by_frame_id: dict[int, dict[str, int | None]] = {}
    process = make_process_handle()

    gt_loader = None
    evaluator = None
    try:
        tar_patch_ctx.__enter__()
        gt_loader = DavisGroundTruthLoader(config)
        evaluator = TrackingOnlyEvaluator(stable_min_frames=stable_min_frames)
        for idx, frame_path in enumerate(frame_files):
            loop_t0 = perf_counter()
            frame_id = parse_frame_id(frame_path)
            if use_sequential_frame_ids:
                frame_id = int(idx)
            else:
                frame_id = int(idx) if frame_id is None else int(frame_id)

            rss_before = read_process_rss_bytes(process)
            read_t0 = perf_counter()
            if is_tar_source:
                frame_bgr = tar_frame_source.read_bgr(str(frame_path))
                if frame_bgr is None:
                    raise RuntimeError(
                        f"Could not read frame {frame_path} from {scene_bundle.data_tar_path}"
                    )
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame_rgb)
            else:
                image = Image.open(frame_path).convert("RGB")
                frame_rgb = np.array(image)
            read_ms = (perf_counter() - read_t0) * 1000.0
            rss_after_read = read_process_rss_bytes(process)

            gt_t0 = perf_counter()
            gt_objects = gt_loader.load_frame(frame_id=frame_id, target_shape=frame_rgb.shape[:2])
            gt_ms = (perf_counter() - gt_t0) * 1000.0

            reset_cuda_peak_memory_stats()
            pipeline_t0 = perf_counter()
            appended_new_masks: list[np.ndarray] = []
            appended_new_pred_ids: list[int] = []

            if idx == 0:
                init_gt_ids = sorted(int(gt_id) for gt_id in gt_objects.keys())
                init_regions = []
                for gt_id in init_gt_ids:
                    full_mask = _gt_object_to_full_mask(gt_objects[int(gt_id)], frame_rgb.shape)
                    init_regions.append({"name": f"obj_{int(gt_id)}", "mask": full_mask.astype(np.uint8)})
                with _inference_context():
                    with _cuda_autocast_context():
                        tracker.initialize(image, init_regions)
                current_tracker_pred_masks = [np.asarray(region["mask"]).astype(np.uint8, copy=False) for region in init_regions]
                current_tracker_pred_ids = [int(x) for x in tracker.all_obj_ids]
                if len(current_tracker_pred_ids) != len(init_gt_ids):
                    raise RuntimeError("d4sm did not return the same number of internal IDs after initialization.")
                for tracker_id, gt_id in zip(current_tracker_pred_ids, init_gt_ids):
                    dataset_gt_by_tracker_id[int(tracker_id)] = int(gt_id)
                    seen_gt_ids.add(int(gt_id))
            else:
                with _inference_context():
                    with _cuda_autocast_context():
                        _sync_cuda()
                        outputs = tracker.track(image)
                        _sync_cuda()
                current_tracker_pred_masks = [np.asarray(mask).astype(np.uint8, copy=False) for mask in (outputs.get("masks", []) or [])]
                current_tracker_pred_ids = [int(x) for x in tracker.all_obj_ids]

                new_gt_ids = sorted(int(gt_id) for gt_id in gt_objects.keys() if int(gt_id) not in seen_gt_ids)
                if new_gt_ids:
                    new_regions = []
                    for gt_id in new_gt_ids:
                        full_mask = _gt_object_to_full_mask(gt_objects[int(gt_id)], frame_rgb.shape)
                        new_regions.append({"name": f"obj_{int(gt_id)}", "mask": full_mask.astype(np.uint8)})
                        appended_new_masks.append(full_mask.astype(np.uint8, copy=False))
                    prev_len = int(len(tracker.all_obj_ids))
                    with _inference_context():
                        with _cuda_autocast_context():
                            tracker.add_objects(image, new_regions)
                    new_tracker_ids = [int(x) for x in tracker.all_obj_ids[prev_len:]]
                    if len(new_tracker_ids) != len(new_gt_ids):
                        raise RuntimeError("d4sm did not return the same number of internal IDs after add_objects.")
                    appended_new_pred_ids.extend(new_tracker_ids)
                    for tracker_id, gt_id in zip(new_tracker_ids, new_gt_ids):
                        dataset_gt_by_tracker_id[int(tracker_id)] = int(gt_id)
                        seen_gt_ids.add(int(gt_id))

            pipeline_ms = (perf_counter() - pipeline_t0) * 1000.0
            rss_after_pipeline = read_process_rss_bytes(process)
            gpu_after_pipeline = capture_cuda_memory_stats()

            detections, det_to_pred_id = _build_detections(
                tracker_pred_masks=current_tracker_pred_masks,
                tracker_pred_ids=current_tracker_pred_ids,
                appended_new_masks=appended_new_masks,
                appended_new_pred_ids=appended_new_pred_ids,
            )
            pred_info_by_id: dict[int, dict[str, Any]] = {}
            for tracker_id, gt_id in dataset_gt_by_tracker_id.items():
                gt_obj = gt_objects.get(int(gt_id), None)
                gt_class_name = None if gt_obj is None else getattr(gt_obj, "class_name", None)
                pred_info_by_id[int(tracker_id)] = {
                    "instance_label": f"d4sm_track_{int(tracker_id):04d}",
                    "class_name": gt_class_name,
                }

            eval_t0 = perf_counter()
            evaluator.ingest_frame(
                frame_id=int(frame_id),
                detections=detections,
                gt_objects=gt_objects,
                det_to_pred_id=det_to_pred_id,
                pred_info_by_id=pred_info_by_id,
                frame_shape=frame_rgb.shape[:2],
            )
            eval_ms = (perf_counter() - eval_t0) * 1000.0
            per_frame_runtime_memory_by_frame_id[int(frame_id)] = build_runtime_memory_telemetry(
                rss_before=rss_before,
                rss_after_read=rss_after_read,
                rss_after_pipeline=rss_after_pipeline,
                rss_after_eval=read_process_rss_bytes(process),
                gpu_after_pipeline=gpu_after_pipeline,
                gpu_after_eval=capture_cuda_memory_stats(),
            )
            post_ms = float(gt_ms + eval_ms)
            loop_ms = (perf_counter() - loop_t0) * 1000.0
            total_read_ms += float(read_ms)
            total_pipeline_ms += float(pipeline_ms)
            total_gt_ms += float(gt_ms)
            total_eval_ms += float(eval_ms)
            total_post_ms += float(post_ms)
            total_loop_ms += float(loop_ms)
            per_frame_timing_by_frame_id[int(frame_id)] = {
                "read_ms": float(read_ms),
                "pipeline_ms": float(pipeline_ms),
                "gt_ms": float(gt_ms),
                "eval_ms": float(eval_ms),
                "post_ms": float(post_ms),
                "loop_ms": float(loop_ms),
            }
            processed_frames = int(idx + 1)
            should_log_progress = (
                processed_frames == 1
                or processed_frames % progress_every == 0
                or processed_frames == total_frames
            )
            if should_log_progress:
                avg_loop_ms = float(total_loop_ms / max(1, processed_frames))
                print(
                    f"[D4SM][scene={input_source['sequence_name']}] "
                    f"progress {processed_frames}/{total_frames} "
                    f"(frame_id={frame_id}, file={Path(frame_path).name}) | "
                    f"read={read_ms:.2f} ms | "
                    f"pipeline={pipeline_ms:.2f} ms | "
                    f"gt={gt_ms:.2f} ms | "
                    f"eval={eval_ms:.2f} ms | "
                    f"loop={loop_ms:.2f} ms | "
                    f"avg_loop={avg_loop_ms:.2f} ms"
                )

        results = evaluator.finalize()
    finally:
        if tracker_owner:
            tracker = None
        gt_loader = None
        evaluator = None
        process = None
        gc.collect()
        if tracker_owner and GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        if scene_bundle is not None:
            try:
                scene_bundle.close()
            except Exception:
                pass
        try:
            tar_patch_ctx.__exit__(None, None, None)
        except Exception:
            pass
        release_cuda_scene_resources()

    n_processed_frames = int(len(frame_files))
    avg_divisor = float(max(1, n_processed_frames))
    timing_summary = {
        "n_processed_frames": n_processed_frames,
        "total_read_ms": float(total_read_ms),
        "avg_read_ms": float(total_read_ms / avg_divisor),
        "total_pipeline_ms": float(total_pipeline_ms),
        "avg_pipeline_ms": float(total_pipeline_ms / avg_divisor),
        "total_gt_ms": float(total_gt_ms),
        "avg_gt_ms": float(total_gt_ms / avg_divisor),
        "total_eval_ms": float(total_eval_ms),
        "avg_eval_ms": float(total_eval_ms / avg_divisor),
        "total_post_ms": float(total_post_ms),
        "avg_post_ms": float(total_post_ms / avg_divisor),
        "total_loop_ms": float(total_loop_ms),
        "avg_loop_ms": float(total_loop_ms / avg_divisor),
        "total_runtime_seconds": float(total_loop_ms / 1000.0),
        "avg_runtime_seconds": float((total_loop_ms / avg_divisor) / 1000.0),
    }
    results["timing_summary"] = timing_summary
    summary = results.setdefault("summary", {})
    summary.update(timing_summary)

    for row in (results.get("per_frame", []) or []):
        frame_id = int(row.get("frame_id", -1))
        frame_timing = per_frame_timing_by_frame_id.get(frame_id, {})
        row["read_ms"] = float(frame_timing.get("read_ms", 0.0))
        row["pipeline_ms"] = float(frame_timing.get("pipeline_ms", 0.0))
        row["gt_ms"] = float(frame_timing.get("gt_ms", 0.0))
        row["eval_ms"] = float(frame_timing.get("eval_ms", 0.0))
        row["post_ms"] = float(frame_timing.get("post_ms", 0.0))
        row["loop_ms"] = float(frame_timing.get("loop_ms", 0.0))
        row.update(per_frame_runtime_memory_by_frame_id.get(frame_id, {}))

    report = build_generic_console_report(results)
    return results, report


def main() -> None:
    project_dir = PROJECT_DIR
    config_path = SRC_DIR / "config" / "default_config.yaml"
    input_source = resolve_testing_input_source(str(project_dir))
    stable_min_frames = int(os.environ.get("REMIND_D4SM_STABLE_MIN_FRAMES", "3").strip() or "3")
    max_frames_raw = os.environ.get("REMIND_D4SM_MAX_FRAMES", "").strip()
    max_frames = None if not max_frames_raw else int(max_frames_raw)

    results, report = evaluate_scene(
        project_dir=project_dir,
        config_path=config_path,
        input_source=input_source,
        stable_min_frames=stable_min_frames,
        max_frames=max_frames,
    )

    scene_tag = sanitize_name_for_path(str(input_source.get("sequence_name", "unknown_scene")))
    out_dir = _build_scene_output_dir(scene_tag=scene_tag)
    write_json(out_dir / "tracking_eval.json", results)
    write_csv(out_dir / "tracking_eval_per_object.csv", results.get("per_object", []) or [])
    write_csv(out_dir / "tracking_eval_per_class.csv", results.get("per_class", []) or [])
    write_csv(out_dir / "tracking_eval_per_case.csv", results.get("per_case", []) or [])
    write_csv(out_dir / "tracking_eval_per_frame.csv", results.get("per_frame", []) or [])
    write_csv(out_dir / "tracking_eval_per_pred_track.csv", results.get("per_pred_track", []) or [])
    write_text(out_dir / "tracking_eval_report.txt", report + "\n")

    print(f"[D4SM] Input source -> {input_source['mode']}")
    print(f"[D4SM] Input sequence -> {input_source['sequence_name']}")
    print(f"[D4SM] Frames dir -> {input_source.get('frames_dir', '')}")
    if input_source.get("data_tar_path"):
        print(f"[D4SM] Data tar -> {input_source['data_tar_path']}")
    if input_source.get("annotations_tar_path"):
        print(f"[D4SM] Annotations tar -> {input_source['annotations_tar_path']}")
    print(report)
    print(f"\n[D4SM] Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
