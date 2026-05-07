from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tarfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
TESTING_DIR = os.path.abspath(CURRENT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if TESTING_DIR not in sys.path:
    sys.path.insert(0, TESTING_DIR)

import detection.davis_segmenter as davis_segmenter_module
from config.config_loader import Config
from detection.davis_segmenter import DavisSegmenter
from pipeline.initialization import initialize_system
from pipeline.reid_pipeline import ReIDPipeline
from testing import davis_gt as davis_gt_module
from testing.davis_gt import DavisGroundTruthLoader
from testing import run_tracking_batch as base_batch
from testing.run_tracking_test import (
    build_det_to_object_id,
    build_runtime_memory_telemetry,
    capture_cuda_memory_stats,
    make_process_handle,
    read_process_rss_bytes,
    reset_cuda_peak_memory_stats,
    resolve_aligned_shape,
)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PROJECT_DIR = Path(SRC_DIR).resolve().parent


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
    _data_tar: tarfile.TarFile | None = field(default=None, init=False, repr=False)
    _annotations_tar: tarfile.TarFile | None = field(default=None, init=False, repr=False)

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
            raise FileNotFoundError(f"No existe {rel} en {self.data_tar_path}")
        extracted = self._get_data_tar().extractfile(member_name)
        if extracted is None:
            raise FileNotFoundError(f"No se pudo abrir {rel} en {self.data_tar_path}")
        return extracted.read()

    def read_annotations_member_bytes(self, rel_path: str) -> bytes:
        rel = str(rel_path or "").strip().strip("/")
        member_name = self.annotation_members_by_rel.get(rel, None)
        if member_name is None:
            raise FileNotFoundError(f"No existe {rel} en {self.annotations_tar_path}")
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
        raise FileNotFoundError(f"Falta data tar para {scene_id}: {data_tar_path}")
    if not annotations_tar_path.is_file():
        raise FileNotFoundError(f"Falta annotations tar para {scene_id}: {annotations_tar_path}")

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
    frame_names: list[str]
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
            f"Faltan {len(missing_frames)} frames listados en {data_tar_path}. Ejemplos: {preview}"
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
            f"Faltan {len(missing_masks)} mascaras en {annotations_tar_path}. Ejemplos de frame_id: {preview}"
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
            raise RuntimeError("TarDavisSegmenter requiere config['davis']['tar_scene_bundle'].")
        return bundle

    def load_model(self) -> None:
        bundle = self.resolve_tar_bundle()
        meta = dict(bundle.meta or {})
        self.meta_path = Path(f"{bundle.annotations_tar_path}!{bundle.meta_rel_path}")
        self.sequence_name_resolved = str(meta.get("scene_id", bundle.scene_id) or bundle.scene_id)
        self.annotations_dir = Path(
            f"{bundle.annotations_tar_path}!annotations/{bundle.mask_variant}"
        )
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


def _env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name, None)
    if raw is None:
        return str(default)
    return str(raw).strip()


def _discover_tar_scene_ids(*, data_tar_root: Path, annotations_tar_root: Path) -> list[str]:
    data_ids = {
        tar_path.stem
        for tar_path in data_tar_root.glob("*.tar")
        if tar_path.is_file()
    }
    annotation_ids = {
        tar_path.stem
        for tar_path in annotations_tar_root.glob("*.tar")
        if tar_path.is_file()
    }
    return sorted(data_ids & annotation_ids)


def _resolve_tar_scene_ids(*, data_tar_root: Path, annotations_tar_root: Path) -> list[str]:
    scenes_env = _env_str("APP2_BATCH_TAR_SCENES", "")
    scenes_file = _env_str("APP2_BATCH_TAR_SCENES_FILE", "")
    single_scene = _env_str("APP2_SCENE_ID", "")

    if scenes_file:
        return base_batch.read_scene_ids_from_file(scenes_file)
    if scenes_env:
        parts = re.split(r"[\s,;]+", scenes_env)
        return base_batch.unique_preserve_order(parts)
    if single_scene:
        return [single_scene]
    return _discover_tar_scene_ids(
        data_tar_root=data_tar_root,
        annotations_tar_root=annotations_tar_root,
    )


def _evaluate_scene_tar(
    *,
    project_dir: Path,
    config_path: Path,
    scene_bundle: TarSceneBundle,
    stable_min_frames: int,
    max_frames: int | None,
    force_detector_backend: str,
) -> tuple[dict[str, Any], str]:
    config = Config(default_config_path=config_path).to_dict()
    config.setdefault("detector", {})["backend"] = str(force_detector_backend)
    davis_cfg = config.setdefault("davis", {})
    davis_cfg["sequence_name"] = str(scene_bundle.scene_id)
    davis_cfg["variant"] = _normalize_davis_variant(scene_bundle.mask_variant)
    davis_cfg["tar_scene_bundle"] = scene_bundle
    davis_cfg["prefetch_annotations"] = False
    timing_cfg = config.setdefault("timing", {})
    timing_cfg["enabled"] = False
    timing_cfg["table"] = False
    timing_cfg["detail_keys"] = []
    trace_cfg = config.setdefault("debug", {}).setdefault("association_trace", {})
    trace_cfg["enabled"] = False
    trace_cfg["mode"] = "off"

    frame_names = list(scene_bundle.frame_names)
    if max_frames is not None:
        frame_names = frame_names[: max(0, int(max_frames))]
    if not frame_names:
        raise RuntimeError(f"No hay frames a evaluar para {scene_bundle.scene_id}")

    frame_source = TarFrameSource(scene_bundle)
    process = make_process_handle()
    progress_every = 20

    try:
        with _patched_tar_davis_segmenter():
            ctx = initialize_system(config)
            pipeline = ReIDPipeline(ctx)
            gt_loader = DavisGroundTruthLoader(config)
            evaluator = base_batch.TrackingEvaluator(
                stable_min_frames=stable_min_frames,
                config=config,
            )

            total_read_ms = 0.0
            total_pipeline_ms = 0.0
            total_gt_ms = 0.0
            total_eval_ms = 0.0
            total_post_ms = 0.0
            total_loop_ms = 0.0
            per_frame_timing_by_frame_id: dict[int, dict[str, float]] = {}
            per_frame_runtime_memory_by_frame_id: dict[int, dict[str, int | None]] = {}
            total_frames = int(len(frame_names))

            print(
                f"[BATCH-TAR][scene={scene_bundle.scene_id}] "
                f"start | frames={total_frames} | "
                f"image_subdir={scene_bundle.image_subdir} | "
                f"mask_variant={scene_bundle.mask_variant}"
            )

            for idx, frame_name in enumerate(frame_names):
                loop_t0 = perf_counter()
                frame_id = int(idx)
                rss_before = read_process_rss_bytes(process)
                read_t0 = perf_counter()
                frame = frame_source.read_bgr(frame_name)
                read_ms = (perf_counter() - read_t0) * 1000.0
                if frame is None:
                    raise RuntimeError(
                        f"No se pudo leer frame {frame_name} desde {scene_bundle.data_tar_path}"
                    )
                rss_after_read = read_process_rss_bytes(process)

                timestamp = float(frame_id)
                reset_cuda_peak_memory_stats()
                t0 = perf_counter()
                p_out, a_out, u_out = pipeline.process_frame(
                    frame=frame,
                    frame_id=frame_id,
                    timestamp=timestamp,
                )
                pipeline_ms = (perf_counter() - t0) * 1000.0
                rss_after_pipeline = read_process_rss_bytes(process)
                gpu_after_pipeline = capture_cuda_memory_stats()

                gt_t0 = perf_counter()
                aligned_shape = resolve_aligned_shape(p_out)
                gt_objects = gt_loader.load_frame(frame_id=frame_id, target_shape=aligned_shape)
                gt_ms = (perf_counter() - gt_t0) * 1000.0

                det_to_object_id = build_det_to_object_id(u_out)
                eval_t0 = perf_counter()
                evaluator.ingest_frame(
                    frame_id=frame_id,
                    detections=p_out.detections,
                    gt_objects=gt_objects,
                    det_to_object_id=det_to_object_id,
                    memory_store=ctx.memory,
                    association_output=a_out,
                    update_output=u_out,
                    frame_shape=aligned_shape,
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
                        f"[BATCH-TAR][scene={scene_bundle.scene_id}] "
                        f"progress {processed_frames}/{total_frames} "
                        f"(frame_id={frame_id}, file={frame_name}) | "
                        f"read={read_ms:.2f} ms | "
                        f"pipeline={pipeline_ms:.2f} ms | "
                        f"gt={gt_ms:.2f} ms | "
                        f"eval={eval_ms:.2f} ms | "
                        f"loop={loop_ms:.2f} ms | "
                        f"avg_loop={avg_loop_ms:.2f} ms"
                    )

            results = evaluator.finalize()
            n_processed_frames = int(len(frame_names))
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

            summary.update(base_batch.build_memory_summary(results.get("per_frame", []) or []))
            report = base_batch.build_console_report(results)
            return results, report
    finally:
        scene_bundle.close()


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir.parent
    project_dir = src_dir.parent.parent
    config_path = src_dir / "config" / "default_config.yaml"

    dataset_root = Path(
        _env_str(
            "APP2_SCANNETPP_TAR_ROOT",
            "/mnt/a/alejodosr/qsync/2026_tracker_reid/datasets/scannetpp_data",
        )
    ).expanduser().resolve()
    data_tar_root = Path(
        _env_str("APP2_SCANNETPP_DATA_TAR_ROOT", str(dataset_root / "data"))
    ).expanduser().resolve()
    annotations_tar_root = Path(
        _env_str("APP2_SCANNETPP_ANNOTATIONS_TAR_ROOT", str(dataset_root / "annotations"))
    ).expanduser().resolve()
    image_subdir = _env_str("APP2_IMAGE_SUBDIR", "dslr/resized_images") or "dslr/resized_images"
    mask_variant = _env_str("APP2_MASK_VARIANT", "benchmark") or "benchmark"
    normalized_mask_variant = _normalize_mask_variant(mask_variant)

    scene_ids = _resolve_tar_scene_ids(
        data_tar_root=data_tar_root,
        annotations_tar_root=annotations_tar_root,
    )
    if not scene_ids:
        raise RuntimeError("No se resolvieron escenas .tar para el batch.")

    run_id = "our_pipeline_tar"
    output_root = (project_dir / "outputs" / "tfm" / "testing_batch_tar").resolve()
    batch_dir = output_root
    scenes_root = batch_dir / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)

    max_scenes = base_batch._env_int("APP2_BATCH_TAR_MAX_SCENES", None)
    batch_size = base_batch._env_int("APP2_BATCH_TAR_SIZE", max_scenes)
    (
        scene_ids,
        registered_scene_ids,
        selection_mode,
        existing_manifest_rows,
        existing_per_scene_rows,
    ) = base_batch.resolve_scene_schedule(
        candidate_scene_ids=base_batch.unique_preserve_order([str(scene_id) for scene_id in scene_ids]),
        batch_dir=batch_dir,
        batch_size=batch_size,
    )

    stable_min_frames = int(base_batch._env_int("APP2_BATCH_TAR_STABLE_MIN_FRAMES", 3) or 3)
    max_frames = base_batch._env_int("APP2_BATCH_TAR_MAX_FRAMES", None)
    force_detector_backend = _env_str("APP2_BATCH_TAR_DETECTOR_BACKEND", "davis") or "davis"

    run_config_row = base_batch.build_run_config_row(
        run_id=run_id,
        batch_name=run_id,
        batch_dir=batch_dir,
        masks_root_base=annotations_tar_root,
        images_root_base=data_tar_root,
        image_subdir=image_subdir,
        mask_variant=normalized_mask_variant,
        stable_min_frames=stable_min_frames,
        max_frames=max_frames,
        max_scenes=max_scenes,
        batch_size=batch_size,
        selection_mode=selection_mode,
        force_detector_backend=force_detector_backend,
        selected_scene_ids=scene_ids,
        registered_scene_ids=registered_scene_ids,
    )
    base_batch.write_single_row_csv(batch_dir / "run_config.csv", run_config_row)

    scene_name_by_id = base_batch.merge_scene_name_index(
        base_scene_name_by_id={str(scene_id): str(scene_id) for scene_id in registered_scene_ids},
        manifest_rows=existing_manifest_rows,
        per_scene_rows=existing_per_scene_rows,
    )
    failed_scene_errors: dict[str, str] = {
        str(row.get("scene_id", "") or "").strip(): str(row.get("error_message", "") or "")
        for row in existing_manifest_rows
        if str(row.get("scene_id", "") or "").strip()
        and str(row.get("status", "") or "").strip() == "failed"
        and str(row.get("error_message", "") or "").strip()
    }
    base_batch.rebuild_batch_outputs(
        batch_dir=batch_dir,
        run_id=run_id,
        selected_scene_ids=scene_ids,
        registered_scene_ids=registered_scene_ids,
        scene_name_by_id=scene_name_by_id,
        failed_scene_errors=failed_scene_errors,
    )
    if not scene_ids:
        print("[BATCH-TAR] No hay escenas pendientes para esta ejecucion.")
        return

    for scene_id in scene_ids:
        scene_key = base_batch.sanitize_name_for_path(str(scene_id))
        final_scene_dir = scenes_root / scene_key
        if base_batch.scene_dir_is_complete(final_scene_dir):
            print(f"[BATCH-TAR] Skip completed scene -> {scene_id}")
            continue
        if final_scene_dir.exists():
            incomplete_backup_dir = base_batch.reserve_incomplete_scene_backup_dir(final_scene_dir)
            final_scene_dir.rename(incomplete_backup_dir)
            print(
                f"[BATCH-TAR] Incomplete final output moved -> {scene_id} "
                f"({final_scene_dir} -> {incomplete_backup_dir})"
            )

        temp_scene_dir = scenes_root / f".tmp_{scene_key}"
        if temp_scene_dir.exists():
            shutil.rmtree(temp_scene_dir)

        scene_started_at = base_batch._now_iso()
        try:
            scene_bundle = _build_scene_bundle(
                scene_id=str(scene_id),
                data_tar_root=data_tar_root,
                annotations_tar_root=annotations_tar_root,
                mask_variant=normalized_mask_variant,
                image_subdir=image_subdir,
            )
            print(f"[BATCH-TAR] Scene start -> {scene_id}")
            print(f"[BATCH-TAR] Data tar -> {scene_bundle.data_tar_path}")
            print(f"[BATCH-TAR] Annotations tar -> {scene_bundle.annotations_tar_path}")
            results, scene_report = _evaluate_scene_tar(
                project_dir=project_dir,
                config_path=config_path,
                scene_bundle=scene_bundle,
                stable_min_frames=stable_min_frames,
                max_frames=max_frames,
                force_detector_backend=force_detector_backend,
            )
            scene_name = str(scene_bundle.scene_id)
            scene_name_by_id[str(scene_id)] = str(scene_name)
            failed_scene_errors.pop(str(scene_id), None)
            base_batch.write_scene_outputs(
                temp_scene_dir=temp_scene_dir,
                final_scene_dir=final_scene_dir,
                run_id=run_id,
                scene_id=str(scene_id),
                scene_name=str(scene_name),
                results=results,
                scene_report=scene_report,
                scene_started_at=scene_started_at,
                scene_finished_at=base_batch._now_iso(),
                stable_min_frames=stable_min_frames,
                max_frames=max_frames,
                force_detector_backend=force_detector_backend,
                mask_variant=normalized_mask_variant,
                image_subdir=image_subdir,
            )
            if final_scene_dir.exists():
                raise RuntimeError(
                    f"Ya existe output final para {scene_id}: {final_scene_dir}. No se sobreescribe automaticamente."
                )
            temp_scene_dir.rename(final_scene_dir)
            base_batch.rebuild_batch_outputs(
                batch_dir=batch_dir,
                run_id=run_id,
                selected_scene_ids=scene_ids,
                registered_scene_ids=registered_scene_ids,
                scene_name_by_id=scene_name_by_id,
                failed_scene_errors=failed_scene_errors,
            )
            print(f"[BATCH-TAR] Scene completed -> {scene_id} ({scene_started_at} -> {base_batch._now_iso()})")
        except Exception as exc:
            failed_scene_errors[str(scene_id)] = str(exc)
            if temp_scene_dir.exists():
                shutil.rmtree(temp_scene_dir, ignore_errors=True)
            base_batch.rebuild_batch_outputs(
                batch_dir=batch_dir,
                run_id=run_id,
                selected_scene_ids=scene_ids,
                registered_scene_ids=registered_scene_ids,
                scene_name_by_id=scene_name_by_id,
                failed_scene_errors=failed_scene_errors,
            )
            print(f"[BATCH-TAR][ERROR] Scene failed -> {scene_id}: {exc}")
            continue

    print("[BATCH-TAR] Done.")


if __name__ == "__main__":
    main()
