from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
TESTING_DIR = CURRENT_DIR.parent
SRC_DIR = TESTING_DIR.parent
WORKSPACE_DIR = SRC_DIR.parent
MASA_ROOT = (WORKSPACE_DIR / "third_party" / "masa").resolve()
for import_root in (SRC_DIR, TESTING_DIR, MASA_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

try:
    import torch
except Exception as exc:  # pragma: no cover - produces a clearer runtime error
    raise RuntimeError("MASA evaluation requires a working PyTorch installation.") from exc

from config.config_loader import Config
from scripts.generate_recovery_reappearance_csvs import generate_outputs as generate_recovery_outputs
from testing import run_tracking_batch as batch
from testing import run_tracking_batch_tar as tar_batch
from testing.common.generic_tracking_metrics import TrackingOnlyEvaluator
from testing.common.generic_tracking_reporting import build_generic_console_report
from testing.davis_gt import DavisGroundTruthLoader
from testing.reporting import write_csv, write_json, write_text
from testing.run_tracking_test import (
    build_runtime_memory_telemetry,
    capture_cuda_memory_stats,
    make_process_handle,
    read_process_rss_bytes,
    reset_cuda_peak_memory_stats,
    resolve_frame_files_for_testing,
)
from testing.tracking_metrics import build_memory_summary
from utils.io import parse_frame_id, read_bgr


DEFAULT_CONFIG = SRC_DIR / "config" / "default_config.yaml"
DEFAULT_MASA_CONFIG = MASA_ROOT / "configs" / "masa-one" / "masa_r50_plug_and_play.py"
DEFAULT_MASA_CHECKPOINT = MASA_ROOT / "saved_models" / "masa_models" / "masa_r50.pth"


@dataclass
class MaskDetection:
    detection_id: int
    mask: np.ndarray
    bbox: tuple[float, float, float, float]
    class_id: int
    class_name: str | None
    confidence: float
    geom: dict[str, Any]


@dataclass(frozen=True)
class CustomScene:
    scene_id: str
    frames_dir: Path
    davis_meta_path: Path
    davis_annotations_dir: Path


class YoloMaskProvider:
    def __init__(
        self,
        *,
        model_path: Path,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        device: str | None,
        mask_erosion_px: int,
        mask_erosion_iters: int,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "YOLO mode requires ultralytics in masaenv. Install it with: "
                "pip install ultralytics"
            ) from exc

        if not model_path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.model_path = model_path
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.max_det = int(max_det)
        self.device = None if not device else str(device)
        self.mask_erosion_px = max(0, int(mask_erosion_px))
        self.mask_erosion_iters = max(1, int(mask_erosion_iters))

        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, dict):
            self.class_id_to_name = {int(key): str(value) for key, value in names.items()}
        else:
            self.class_id_to_name = {
                int(idx): str(value) for idx, value in enumerate(list(names))
            }

    def detect(self, frame_bgr: np.ndarray, *, frame_id: int) -> list[MaskDetection]:
        height, width = frame_bgr.shape[:2]
        kwargs: dict[str, Any] = {
            "task": "segment",
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_det,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        predictions = self.model.predict(frame_bgr, **kwargs)
        if not predictions:
            return []
        result = predictions[0]
        masks_obj = getattr(result, "masks", None)
        boxes_obj = getattr(result, "boxes", None)
        if masks_obj is None or masks_obj.data is None or boxes_obj is None:
            return []

        masks = masks_obj.data.detach().cpu().numpy()
        boxes = boxes_obj.xyxy.detach().cpu().numpy()
        labels = boxes_obj.cls.detach().cpu().numpy().astype(int)
        scores = boxes_obj.conf.detach().cpu().numpy()
        kernel = None
        if self.mask_erosion_px > 0:
            size = 2 * self.mask_erosion_px + 1
            kernel = np.ones((size, size), dtype=np.uint8)

        detections: list[MaskDetection] = []
        for idx in range(len(masks)):
            mask = masks[idx]
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            mask_bool = mask >= 0.5
            if kernel is not None:
                mask_bool = cv2.erode(
                    mask_bool.astype(np.uint8),
                    kernel,
                    iterations=self.mask_erosion_iters,
                ).astype(bool)
            area = int(mask_bool.sum())
            if area <= 0:
                continue
            class_id = int(labels[idx])
            bbox = tuple(float(value) for value in boxes[idx].tolist())
            detections.append(
                MaskDetection(
                    detection_id=int(idx),
                    mask=mask_bool,
                    bbox=bbox,
                    class_id=class_id,
                    class_name=self.class_id_to_name.get(class_id, f"class_{class_id}"),
                    confidence=float(scores[idx]),
                    geom={"area": area},
                )
            )
        return detections


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default) or default).strip()


def _resolve(value: Any, env_name: str, default: str) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()
    return _env(env_name, default)


def _resolve_optional(value: Any, env_name: str) -> str | None:
    resolved = _resolve(value, env_name, "")
    return resolved or None


def _resolve_int(value: int | None, env_name: str, default: int | None) -> int | None:
    if value is not None:
        return int(value)
    raw = _env(env_name)
    return int(raw) if raw else default


def _resolve_float(value: float | None, env_name: str, default: float) -> float:
    if value is not None:
        return float(value)
    raw = _env(env_name)
    return float(raw) if raw else float(default)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref:"):
        ref_path = repo / ".git" / text.split(":", 1)[1].strip()
        if ref_path.is_file():
            return ref_path.read_text(encoding="utf-8").strip()
        return None
    return text or None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _infer_masa_variant(config_path: Path) -> str:
    config_name = config_path.as_posix().lower()
    if "masa-gdino" in config_name or "gdino" in config_name:
        return "gdino_swinb"
    if "masa-sam" in config_name or "sam_" in config_name:
        return "sam_vith" if "vith" in config_name else "sam_vitb"
    if "masa-detic" in config_name or "detic" in config_name:
        return "detic_swinb"
    return "r50"


def _masa_tracker_defaults(config_path: Path) -> dict[str, Any]:
    from mmengine.config import Config as MMEngineConfig

    config = MMEngineConfig.fromfile(str(config_path))
    tracker = config.model.get("tracker", {})
    return dict(tracker)


def _run_id_value(value: Any) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return format(number, ".8g").replace("-", "NEG").replace(".", "p")


def _automatic_run_id(
    *,
    dataset: str,
    detections: str,
    masa_variant: str,
    mask_variant: str,
    class_aware: bool,
    max_distance: float,
    memo_tracklet_frames: int,
    match_score_thr: float,
    memo_momentum: float,
) -> str:
    dataset_name = "SCANNET" if dataset == "scannet-tar" else "CUSTOM"
    parts = [dataset_name]
    if dataset == "scannet-tar":
        mask_name = "BENCH" if mask_variant == "benchmark_instance" else mask_variant.upper()
        parts.append(mask_name)
    parts.extend(
        [
            detections.upper(),
            "MASA",
            masa_variant.upper(),
            f"CLASS_AWARE_{'ON' if class_aware else 'OFF'}",
            "DISTANCE_OFF"
            if float(max_distance) == -1.0
            else f"DISTANCE_{_run_id_value(max_distance)}",
            f"MEMORY_{int(memo_tracklet_frames)}",
            f"MATCH_{_run_id_value(match_score_thr)}",
            f"MOMENTUM_{_run_id_value(memo_momentum)}",
        ]
    )
    return "-".join(parts)


def _full_gt_mask(gt_obj: Any, frame_shape: tuple[int, int]) -> np.ndarray:
    full_mask = np.zeros(frame_shape, dtype=bool)
    bbox = getattr(gt_obj, "bbox_xyxy", None)
    local_mask = getattr(gt_obj, "mask", None)
    if bbox is None or local_mask is None:
        return full_mask
    x1, y1, x2, y2 = [int(value) for value in bbox]
    full_mask[y1:y2, x1:x2] = np.asarray(local_mask).astype(bool, copy=False)
    return full_mask


def _gt_detections(
    gt_objects: dict[int, Any],
    *,
    frame_shape: tuple[int, int],
    class_id_by_name: dict[str, int],
) -> list[MaskDetection]:
    detections: list[MaskDetection] = []
    for detection_id, gt_id in enumerate(sorted(int(key) for key in gt_objects.keys())):
        gt_obj = gt_objects[gt_id]
        bbox = getattr(gt_obj, "bbox_xyxy", None)
        if bbox is None:
            continue
        class_name = getattr(gt_obj, "class_name", None)
        class_key = str(class_name or "unknown")
        if class_key not in class_id_by_name:
            class_id_by_name[class_key] = len(class_id_by_name)
        mask = _full_gt_mask(gt_obj, frame_shape)
        area = int(mask.sum())
        if area <= 0:
            continue
        detections.append(
            MaskDetection(
                detection_id=int(detection_id),
                mask=mask,
                bbox=tuple(float(value) for value in bbox),
                class_id=int(class_id_by_name[class_key]),
                class_name=None if class_name is None else str(class_name),
                confidence=1.0,
                geom={"area": area, "gt_instance_id": int(gt_id)},
            )
        )
    return detections


def _bbox_iou_matrix(
    output_boxes: np.ndarray,
    input_detections: list[MaskDetection],
) -> np.ndarray:
    if len(output_boxes) == 0 or not input_detections:
        return np.zeros((len(output_boxes), len(input_detections)), dtype=np.float64)
    inputs = np.asarray([det.bbox for det in input_detections], dtype=np.float64)
    outputs = np.asarray(output_boxes, dtype=np.float64)
    matrix = np.zeros((len(outputs), len(inputs)), dtype=np.float64)
    for row, box_a in enumerate(outputs):
        x1 = np.maximum(box_a[0], inputs[:, 0])
        y1 = np.maximum(box_a[1], inputs[:, 1])
        x2 = np.minimum(box_a[2], inputs[:, 2])
        y2 = np.minimum(box_a[3], inputs[:, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = np.maximum(0.0, inputs[:, 2] - inputs[:, 0]) * np.maximum(
            0.0, inputs[:, 3] - inputs[:, 1]
        )
        union = area_a + area_b - intersection
        valid = union > 0.0
        matrix[row, valid] = intersection[valid] / union[valid]
    return matrix


def _map_masa_outputs(
    *,
    input_detections: list[MaskDetection],
    track_instances: Any,
    min_bbox_iou: float = 0.99,
) -> tuple[list[MaskDetection], dict[int, int], dict[int, dict[str, Any]]]:
    if len(track_instances.instances_id) == 0 or not input_detections:
        return [], {}, {}
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("MASA output mapping requires scipy.") from exc

    output_boxes = track_instances.bboxes.detach().cpu().numpy()
    output_ids = track_instances.instances_id.detach().cpu().numpy().astype(int)
    iou = _bbox_iou_matrix(output_boxes, input_detections)
    output_indices, input_indices = linear_sum_assignment(-iou)

    detections: list[MaskDetection] = []
    det_to_pred_id: dict[int, int] = {}
    pred_info_by_id: dict[int, dict[str, Any]] = {}
    for output_idx, input_idx in zip(output_indices.tolist(), input_indices.tolist()):
        if float(iou[output_idx, input_idx]) < float(min_bbox_iou):
            continue
        source = input_detections[input_idx]
        pred_id = int(output_ids[output_idx])
        eval_det_id = len(detections)
        detections.append(
            MaskDetection(
                detection_id=eval_det_id,
                mask=source.mask,
                bbox=source.bbox,
                class_id=source.class_id,
                class_name=source.class_name,
                confidence=source.confidence,
                geom=dict(source.geom),
            )
        )
        det_to_pred_id[eval_det_id] = pred_id
        pred_info_by_id.setdefault(
            pred_id,
            {
                "instance_label": f"masa_track_{pred_id:04d}",
                "class_name": source.class_name,
            },
        )
    return detections, det_to_pred_id, pred_info_by_id


def _run_masa_frame(
    *,
    model: Any,
    test_pipeline: Any,
    frame_bgr: np.ndarray,
    frame_id: int,
    video_len: int,
    detections: list[MaskDetection],
    device: str,
    fp16: bool,
) -> tuple[Any, float]:
    from masa.apis import inference_masa

    boxes = torch.empty((0, 5), dtype=torch.float32, device=device)
    labels = torch.empty((0,), dtype=torch.long, device=device)
    if detections:
        boxes = torch.tensor(
            [list(det.bbox) + [float(det.confidence)] for det in detections],
            dtype=torch.float32,
            device=device,
        )
        labels = torch.tensor(
            [int(det.class_id) for det in detections],
            dtype=torch.long,
            device=device,
        )

    _cuda_sync()
    started = perf_counter()
    with torch.inference_mode():
        result = inference_masa(
            model,
            frame_bgr,
            frame_id=int(frame_id),
            video_len=int(video_len),
            test_pipeline=test_pipeline,
            det_bboxes=boxes,
            det_labels=labels,
            fp16=bool(fp16),
        )
    _cuda_sync()
    return result[0].pred_track_instances, (perf_counter() - started) * 1000.0


def _build_config(
    config_path: Path,
    *,
    sequence_name: str,
    custom_scene: CustomScene | None = None,
    tar_bundle: tar_batch.TarSceneBundle | None = None,
) -> dict[str, Any]:
    config = Config(default_config_path=str(config_path)).to_dict()
    davis = config.setdefault("davis", {})
    davis["sequence_name"] = str(sequence_name)
    if custom_scene is not None:
        config.setdefault("input", {})["frames_dir"] = str(custom_scene.frames_dir)
        davis["meta_path"] = str(custom_scene.davis_meta_path)
        davis["annotations_dir"] = str(custom_scene.davis_annotations_dir)
    if tar_bundle is not None:
        davis["variant"] = tar_batch._normalize_davis_variant(tar_bundle.mask_variant)
        davis["tar_scene_bundle"] = tar_bundle
        davis["prefetch_annotations"] = False
    return config


def _load_custom_scenes(args: argparse.Namespace) -> list[CustomScene]:
    dataset_root_raw = _resolve_optional(
        args.custom_dataset_root or args.dataset_root, "REMIND_CUSTOM_DATASET_ROOT"
    )
    dataset_root = Path(dataset_root_raw).expanduser().resolve() if dataset_root_raw else None

    def resolve_path(raw: str) -> Path:
        path = Path(raw).expanduser()
        if not path.is_absolute() and dataset_root is not None:
            path = dataset_root / path
        return path.resolve()

    scenes_file = _resolve_optional(args.scenes_def_file, "REMIND_CUSTOM_SCENES_DEF_FILE")
    raw_scenes: list[dict[str, Any]]
    if scenes_file:
        payload = json.loads(Path(scenes_file).expanduser().read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--scenes-def-file must contain a JSON array.")
        raw_scenes = [dict(item) for item in payload]
    else:
        frames = _resolve_optional(args.frames_dir, "REMIND_CUSTOM_FRAMES_DIR")
        meta = _resolve_optional(args.davis_meta_path, "REMIND_CUSTOM_DAVIS_META_PATH")
        annotations = _resolve_optional(
            args.davis_annotations_dir, "REMIND_CUSTOM_DAVIS_ANNOTATIONS_DIR"
        )
        if not frames or not meta or not annotations:
            raise ValueError(
                "Custom mode requires --frames-dir, --davis-meta-path and "
                "--davis-annotations-dir, or --scenes-def-file."
            )
        raw_scenes = [
            {
                "scene_id": args.scene_id or "CUSTOMVIDEO",
                "frames_dir": frames,
                "davis_meta_path": meta,
                "davis_annotations_dir": annotations,
            }
        ]

    scenes: list[CustomScene] = []
    for idx, item in enumerate(raw_scenes):
        try:
            scene = CustomScene(
                scene_id=str(item.get("scene_id") or f"custom_{idx:03d}"),
                frames_dir=resolve_path(str(item["frames_dir"])),
                davis_meta_path=resolve_path(str(item["davis_meta_path"])),
                davis_annotations_dir=resolve_path(str(item["davis_annotations_dir"])),
            )
        except KeyError as exc:
            raise ValueError(f"Custom scene {idx} is missing {exc.args[0]}.") from exc
        if not scene.frames_dir.is_dir():
            raise FileNotFoundError(f"Frames directory not found: {scene.frames_dir}")
        if not scene.davis_meta_path.is_file():
            raise FileNotFoundError(f"DAVIS metadata not found: {scene.davis_meta_path}")
        if not scene.davis_annotations_dir.is_dir():
            raise FileNotFoundError(
                f"DAVIS annotations not found: {scene.davis_annotations_dir}"
            )
        scenes.append(scene)
    return scenes


def _event_rows(
    *, run_id: str, scene_id: str, scene_name: str, results: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    event_specs = {
        "swap": ("gt_a", "gt_b", "pred_a_prev", "pred_b_prev"),
        "theft_with_new_id": (
            "thief_gt",
            "victim_gt",
            "stolen_pred_id",
            "victim_new_pred_id",
        ),
        "theft_with_displacement": (
            "thief_gt",
            "victim_gt",
            "stolen_pred_id",
            "victim_pred_id_after",
        ),
    }
    events = dict(results.get("events", {}) or {})
    for event_type, fields in event_specs.items():
        for event in events.get(event_type, []) or []:
            rows.append(
                {
                    "run_id": run_id,
                    "scene_id": scene_id,
                    "scene_name": scene_name,
                    "frame_id": int(event.get("frame_id", -1)),
                    "event_type": event_type,
                    "gt_a": event.get(fields[0]),
                    "gt_b": event.get(fields[1]),
                    "pred_id_main": event.get(fields[2]),
                    "pred_id_aux": event.get(fields[3]),
                    "detail": str(event),
                }
            )
    return rows


def _failure_reason(error_message: str) -> str:
    message = str(error_message or "").lower()
    if any(token in message for token in ("cuda out of memory", "outofmemoryerror", "cublas_status_alloc_failed")):
        return "cuda_out_of_memory"
    if any(token in message for token in ("memoryerror", "cannot allocate memory", "std::bad_alloc")):
        return "system_out_of_memory"
    if any(token in message for token in ("filenotfounderror", "no such file", "not found")):
        return "missing_input"
    if any(token in message for token in ("tarreaderror", "bad tar", "invalid header", "unexpected end of data")):
        return "invalid_archive"
    return "other_error"


def _write_failed_scene_lists(output_dir: Path, failed: dict[str, str]) -> None:
    rows = []
    lines = []
    for scene_id in sorted(failed):
        error_message = str(failed[scene_id] or "")
        exception_type = error_message.partition(":")[0].strip()
        if not exception_type or " " in exception_type:
            exception_type = "UnknownError"
        reason = _failure_reason(error_message)
        rows.append(
            {
                "scene_id": str(scene_id),
                "reason": reason,
                "exception_type": exception_type,
                "error_message": error_message,
            }
        )
        lines.append(f"{scene_id}\t{reason}\t{error_message}")
    write_csv(output_dir / "failed_scenes.csv", rows)
    write_text(output_dir / "failed_scenes.txt", "\n".join(lines) + ("\n" if lines else ""))


def _scene_summary(
    *,
    run_id: str,
    scene_id: str,
    scene_name: str,
    results: dict[str, Any],
    output_dir: Path,
    started_at: str,
    finished_at: str,
    stable_min_frames: int,
    max_frames: int | None,
    dataset: str,
    detections: str,
    masa_variant: str,
) -> dict[str, Any]:
    summary = dict(results.get("summary", {}) or {})
    identity = dict(results.get("tracking_identity_metrics", {}) or {})
    cases = list(results.get("per_case", []) or [])
    orphans = list(results.get("per_orphan_pred", []) or [])
    row: dict[str, Any] = {
        "run_id": run_id,
        "scene_id": scene_id,
        "scene_name": scene_name,
        "status": "completed",
        "started_at": started_at,
        "finished_at": finished_at,
        "output_dir": str(output_dir.resolve()),
        "tracker_family": "masa",
        "model_size": masa_variant,
        "model_variant": masa_variant,
        "dataset": dataset,
        "detector_backend": detections,
        "stable_min_frames": stable_min_frames,
        "max_frames": max_frames,
        "n_frames": int(summary.get("n_frames", 0) or 0),
        "n_gt_objects": int(summary.get("n_objects", 0) or 0),
        "n_cases": int(len(cases)),
        "n_visible_gt_observations": int(
            summary.get("n_visible_gt_observations", 0) or 0
        ),
        "n_matched_gt_observations": int(
            summary.get("n_matched_gt_observations", 0) or 0
        ),
        "n_pred_observations_total": int(
            summary.get("n_pred_observations_total", 0) or 0
        ),
        "n_orphan_pred_observations": int(
            summary.get("n_orphan_pred_observations", 0) or 0
        ),
        "n_unique_orphan_pred_ids": int(summary.get("n_unique_orphan_pred_ids", 0) or 0),
        "n_orphan_only_pred_ids": int(summary.get("n_orphan_only_pred_ids", 0) or 0),
        "orphan_pred_rate": summary.get("orphan_pred_rate"),
        "mean_orphan_pred_area_frac": summary.get("mean_orphan_pred_area_frac"),
        "orphan_pred_area_frac_sum": float(
            sum(float(item.get("pred_area_frac", 0.0) or 0.0) for item in orphans)
        ),
        "tracking_iou_sum": float(
            sum(float(item.get("tracking_iou", 0.0) or 0.0) for item in cases)
        ),
        "tracking_iou_sum_iou40": float(
            sum(float(item.get("tracking_iou_iou40", 0.0) or 0.0) for item in cases)
        ),
    }
    for key in (
        "recovery_attempts_total",
        "recovery_success_reference_total",
        "recovery_success_own_identity_total",
        "recovery_success_duplicate_id_total",
        "recovery_success_foreign_id_total",
        "recovery_rate_reference",
        "recovery_rate_own_identity",
        "recovery_rate_duplicate_id",
        "recovery_rate_foreign_id",
        "n_unique_real_pred_tracks",
        "n_total_pred_tracks_including_orphan_only",
        "n_pred_tracks_with_orphan_observations",
        "n_orphan_only_pred_tracks",
        "pred_track_inflation_factor",
        "n_mt_objects",
        "n_pt_objects",
        "n_ml_objects",
        "mt",
        "pt",
        "ml",
        "objects_fragmented",
        "objects_with_foreign_id_use",
        "swap_events_total",
        "theft_with_new_id_total",
        "theft_with_displacement_total",
        "total_runtime_seconds",
        "avg_runtime_seconds",
        "total_loop_ms",
        "avg_loop_ms",
    ):
        row[key] = summary.get(key)
    for key in (
        "idtp",
        "idfp",
        "idfn",
        "idf1",
        "idp",
        "idr",
        "idsw",
        "frag",
        "tracking_recall",
        "mean_tracking_iou",
        "deta",
        "assa",
        "hota",
        "n_matched_gt_observations_iou40",
        "idtp_iou40",
        "idfp_iou40",
        "idfn_iou40",
        "idf1_iou40",
        "idp_iou40",
        "idr_iou40",
        "tracking_recall_iou40",
        "mean_tracking_iou_iou40",
        "deta_iou40",
        "assa_iou40",
        "hota_iou40",
    ):
        row[key] = identity.get(key)
    for key, value in summary.items():
        if str(key).startswith("mem_"):
            row[key] = value
    return row


def _write_scene_outputs(
    *,
    temp_dir: Path,
    final_dir: Path,
    run_id: str,
    scene_id: str,
    scene_name: str,
    results: dict[str, Any],
    report: str,
    started_at: str,
    finished_at: str,
    stable_min_frames: int,
    max_frames: int | None,
    dataset: str,
    detections: str,
    masa_variant: str,
) -> None:
    temp_dir.mkdir(parents=True, exist_ok=True)
    annotate = lambda rows: batch.annotate_rows(
        rows, run_id=run_id, scene_id=scene_id, scene_name=scene_name
    )
    compatible_case_rows = []
    for raw_row in results.get("per_case", []) or []:
        row = dict(raw_row)
        row["collapsed_global_correct"] = bool(row.get("strict_global_correct", False))
        row["collapsed_pred_object_id"] = row.get("pred_object_id")
        compatible_case_rows.append(row)
    results["per_case"] = compatible_case_rows
    summary_row = _scene_summary(
        run_id=run_id,
        scene_id=scene_id,
        scene_name=scene_name,
        results=results,
        output_dir=final_dir,
        started_at=started_at,
        finished_at=finished_at,
        stable_min_frames=stable_min_frames,
        max_frames=max_frames,
        dataset=dataset,
        detections=detections,
        masa_variant=masa_variant,
    )
    write_csv(temp_dir / batch.SCENE_TABLE_FILES["per_scene"], [summary_row])
    write_csv(temp_dir / batch.SCENE_TABLE_FILES["per_class"], annotate(results.get("per_class", []) or []))
    write_csv(temp_dir / batch.SCENE_TABLE_FILES["per_object"], annotate(results.get("per_object", []) or []))
    write_csv(temp_dir / batch.SCENE_TABLE_FILES["per_case"], annotate(compatible_case_rows))
    write_csv(temp_dir / batch.SCENE_TABLE_FILES["per_case_modules"], [])
    write_csv(temp_dir / batch.SCENE_TABLE_FILES["per_frame"], annotate(results.get("per_frame", []) or []))
    write_csv(temp_dir / batch.SCENE_TABLE_FILES["per_pred_track"], annotate(results.get("per_pred_track", []) or []))
    write_csv(
        temp_dir / batch.SCENE_TABLE_FILES["per_event"],
        _event_rows(run_id=run_id, scene_id=scene_id, scene_name=scene_name, results=results),
    )
    write_csv(temp_dir / "per_orphan_pred.csv", annotate(results.get("per_orphan_pred", []) or []))
    write_json(temp_dir / "tracking_eval.json", results)
    write_text(temp_dir / "report.txt", report.rstrip() + "\n")


def _evaluate_frames(
    *,
    scene_id: str,
    model: Any,
    test_pipeline: Any,
    gt_loader: DavisGroundTruthLoader,
    frames: Iterable[tuple[int, str, np.ndarray, float]],
    total_frames: int,
    detection_mode: str,
    yolo: YoloMaskProvider | None,
    stable_min_frames: int,
    device: str,
    fp16: bool,
    progress_every: int,
) -> tuple[dict[str, Any], str]:
    evaluator = TrackingOnlyEvaluator(stable_min_frames=stable_min_frames)
    class_id_by_name: dict[str, int] = {}
    process = make_process_handle()
    totals = {key: 0.0 for key in ("read", "gt", "detector", "masa", "eval", "loop")}
    timings: dict[int, dict[str, float]] = {}
    memories: dict[int, dict[str, Any]] = {}

    for idx, (frame_id, frame_name, frame_bgr, read_ms) in enumerate(frames):
        loop_started = perf_counter()
        rss_before = read_process_rss_bytes(process)
        rss_after_read = read_process_rss_bytes(process)

        gt_started = perf_counter()
        frame_shape = tuple(int(value) for value in frame_bgr.shape[:2])
        gt_objects = gt_loader.load_frame(frame_id=int(frame_id), target_shape=frame_shape)
        gt_ms = (perf_counter() - gt_started) * 1000.0

        detector_started = perf_counter()
        if detection_mode == "gt":
            input_detections = _gt_detections(
                gt_objects,
                frame_shape=frame_shape,
                class_id_by_name=class_id_by_name,
            )
        else:
            if yolo is None:
                raise RuntimeError("YOLO provider was not initialized.")
            input_detections = yolo.detect(frame_bgr, frame_id=int(frame_id))
        detector_ms = (perf_counter() - detector_started) * 1000.0

        reset_cuda_peak_memory_stats()
        tracks, masa_ms = _run_masa_frame(
            model=model,
            test_pipeline=test_pipeline,
            frame_bgr=frame_bgr,
            frame_id=int(frame_id),
            video_len=total_frames,
            detections=input_detections,
            device=device,
            fp16=fp16,
        )
        rss_after_pipeline = read_process_rss_bytes(process)
        gpu_after_pipeline = capture_cuda_memory_stats()
        eval_detections, det_to_pred_id, pred_info = _map_masa_outputs(
            input_detections=input_detections,
            track_instances=tracks,
        )

        eval_started = perf_counter()
        evaluator.ingest_frame(
            frame_id=int(frame_id),
            detections=eval_detections,
            gt_objects=gt_objects,
            det_to_pred_id=det_to_pred_id,
            pred_info_by_id=pred_info,
            frame_shape=frame_shape,
            frame_telemetry={
                "detector_ms": detector_ms,
                "masa_ms": masa_ms,
                "pipeline_ms": detector_ms + masa_ms,
            },
        )
        eval_ms = (perf_counter() - eval_started) * 1000.0
        loop_ms = read_ms + (perf_counter() - loop_started) * 1000.0
        timings[int(frame_id)] = {
            "read_ms": read_ms,
            "gt_ms": gt_ms,
            "detector_ms": detector_ms,
            "masa_ms": masa_ms,
            "pipeline_ms": detector_ms + masa_ms,
            "eval_ms": eval_ms,
            "post_ms": gt_ms + eval_ms,
            "loop_ms": loop_ms,
        }
        memories[int(frame_id)] = build_runtime_memory_telemetry(
            rss_before=rss_before,
            rss_after_read=rss_after_read,
            rss_after_pipeline=rss_after_pipeline,
            rss_after_eval=read_process_rss_bytes(process),
            gpu_after_pipeline=gpu_after_pipeline,
            gpu_after_eval=capture_cuda_memory_stats(),
        )
        totals["read"] += read_ms
        totals["gt"] += gt_ms
        totals["detector"] += detector_ms
        totals["masa"] += masa_ms
        totals["eval"] += eval_ms
        totals["loop"] += loop_ms

        processed = idx + 1
        if processed == total_frames or processed % progress_every == 0:
            print(f"[MASA][FRAMES][scene={scene_id}] {processed}/{total_frames}")

    results = evaluator.finalize()
    divisor = float(max(1, total_frames))
    timing_summary = {
        "n_processed_frames": int(total_frames),
        "total_read_ms": totals["read"],
        "avg_read_ms": totals["read"] / divisor,
        "total_detector_ms": totals["detector"],
        "avg_detector_ms": totals["detector"] / divisor,
        "total_masa_ms": totals["masa"],
        "avg_masa_ms": totals["masa"] / divisor,
        "total_pipeline_ms": totals["detector"] + totals["masa"],
        "avg_pipeline_ms": (totals["detector"] + totals["masa"]) / divisor,
        "total_gt_ms": totals["gt"],
        "avg_gt_ms": totals["gt"] / divisor,
        "total_eval_ms": totals["eval"],
        "avg_eval_ms": totals["eval"] / divisor,
        "total_loop_ms": totals["loop"],
        "avg_loop_ms": totals["loop"] / divisor,
        "total_runtime_seconds": totals["loop"] / 1000.0,
        "avg_runtime_seconds": totals["loop"] / divisor / 1000.0,
        "detector_mode": detection_mode,
    }
    results["timing_summary"] = timing_summary
    results.setdefault("summary", {}).update(timing_summary)
    for row in results.get("per_frame", []) or []:
        frame_id = int(row.get("frame_id", -1))
        row.update(timings.get(frame_id, {}))
        row.update(memories.get(frame_id, {}))
    results["summary"].update(build_memory_summary(results.get("per_frame", []) or []))
    return results, build_generic_console_report(results)


def _evaluate_custom_scene(
    *,
    scene: CustomScene,
    config_path: Path,
    model: Any,
    test_pipeline: Any,
    detection_mode: str,
    yolo: YoloMaskProvider | None,
    stable_min_frames: int,
    max_frames: int | None,
    device: str,
    fp16: bool,
    progress_every: int,
) -> tuple[dict[str, Any], str]:
    config = _build_config(config_path, sequence_name=scene.scene_id, custom_scene=scene)
    gt_loader = DavisGroundTruthLoader(config)
    frame_files, sequential_ids = resolve_frame_files_for_testing(
        str(scene.frames_dir), davis_meta_path=str(scene.davis_meta_path)
    )
    if max_frames is not None:
        frame_files = frame_files[: max(0, int(max_frames))]
    if not frame_files:
        raise RuntimeError(f"No frames found in {scene.frames_dir}")

    def frames() -> Iterable[tuple[int, str, np.ndarray, float]]:
        for idx, frame_path in enumerate(frame_files):
            frame_id = parse_frame_id(frame_path)
            if sequential_ids or frame_id is None:
                frame_id = idx
            read_started = perf_counter()
            frame = read_bgr(frame_path)
            read_ms = (perf_counter() - read_started) * 1000.0
            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_path}")
            yield int(frame_id), Path(frame_path).name, frame, read_ms

    return _evaluate_frames(
        scene_id=scene.scene_id,
        model=model,
        test_pipeline=test_pipeline,
        gt_loader=gt_loader,
        frames=frames(),
        total_frames=len(frame_files),
        detection_mode=detection_mode,
        yolo=yolo,
        stable_min_frames=stable_min_frames,
        device=device,
        fp16=fp16,
        progress_every=progress_every,
    )


def _evaluate_tar_scene(
    *,
    bundle: tar_batch.TarSceneBundle,
    config_path: Path,
    model: Any,
    test_pipeline: Any,
    detection_mode: str,
    yolo: YoloMaskProvider | None,
    stable_min_frames: int,
    max_frames: int | None,
    device: str,
    fp16: bool,
    progress_every: int,
) -> tuple[dict[str, Any], str]:
    config = _build_config(config_path, sequence_name=bundle.scene_id, tar_bundle=bundle)
    frame_names = list(bundle.frame_names)
    if max_frames is not None:
        frame_names = frame_names[: max(0, int(max_frames))]
    if not frame_names:
        raise RuntimeError(f"No frames found for {bundle.scene_id}")
    frame_source = tar_batch.TarFrameSource(bundle)

    with tar_batch._patched_tar_davis_segmenter():
        gt_loader = DavisGroundTruthLoader(config)

        def frames() -> Iterable[tuple[int, str, np.ndarray, float]]:
            for frame_id, frame_name in enumerate(frame_names):
                read_started = perf_counter()
                frame = frame_source.read_bgr(frame_name)
                read_ms = (perf_counter() - read_started) * 1000.0
                if frame is None:
                    raise RuntimeError(
                        f"Could not read {frame_name} from {bundle.data_tar_path}"
                    )
                yield int(frame_id), str(frame_name), frame, read_ms

        return _evaluate_frames(
            scene_id=bundle.scene_id,
            model=model,
            test_pipeline=test_pipeline,
            gt_loader=gt_loader,
            frames=frames(),
            total_frames=len(frame_names),
            detection_mode=detection_mode,
            yolo=yolo,
            stable_min_frames=stable_min_frames,
            device=device,
            fp16=fp16,
            progress_every=progress_every,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the external MASA baseline with GT or YOLO detections on "
            "Custom DAVIS data or ScanNet++ TAR scenes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=("custom", "scannet-tar"), required=True)
    parser.add_argument("--detections", choices=("gt", "yolo"), required=True)
    parser.add_argument("--config-path", help="REMIND YAML used for GT filtering and DAVIS paths")

    custom = parser.add_argument_group("Custom dataset")
    custom.add_argument("--custom-dataset-root")
    custom.add_argument("--frames-dir")
    custom.add_argument("--davis-meta-path")
    custom.add_argument("--davis-annotations-dir")
    custom.add_argument("--scenes-def-file")

    tar = parser.add_argument_group("ScanNet++ TAR dataset")
    tar.add_argument("--dataset-root", help="Top-level ScanNet++ TAR directory")
    tar.add_argument("--data-tar-root", help="Directory containing scene data TARs")
    tar.add_argument("--annotations-tar-root", help="Directory containing annotation TARs")
    tar.add_argument("--image-subdir", default=None)
    tar.add_argument("--mask-variant", default=None)
    tar.add_argument("--scenes", "--scene-ids", nargs="+")
    tar.add_argument("--scenes-file")
    tar.add_argument("--scene-id")
    tar.add_argument("--exclude-scenes-file")

    masa = parser.add_argument_group("MASA")
    masa.add_argument("--masa-config")
    masa.add_argument("--masa-checkpoint")
    masa.add_argument("--device", default="cuda:0")
    masa.add_argument("--fp16", action="store_true")
    masa.add_argument(
        "--class-aware",
        action="store_true",
        help="Only associate detections with MASA tracks of the same class",
    )
    masa.add_argument("--match-score-thr", type=float)
    masa.add_argument("--memo-tracklet-frames", type=int)
    masa.add_argument("--memo-momentum", type=float)
    masa.add_argument("--max-distance", type=float)
    masa.add_argument("--tracker-fps", type=float)

    yolo = parser.add_argument_group("YOLO segmentation")
    yolo.add_argument("--yolo-model")
    yolo.add_argument("--yolo-conf", type=float)
    yolo.add_argument("--yolo-iou", type=float)
    yolo.add_argument("--yolo-imgsz", type=int)
    yolo.add_argument("--yolo-max-det", type=int)
    yolo.add_argument("--yolo-device")
    yolo.add_argument("--mask-erosion-px", type=int)
    yolo.add_argument("--mask-erosion-iters", type=int)

    control = parser.add_argument_group("Batch and outputs")
    control.add_argument("--output-dir")
    control.add_argument("--run-id")
    control.add_argument("--max-scenes", type=int)
    control.add_argument("--batch-size", type=int)
    control.add_argument("--max-frames", type=int)
    control.add_argument("--stable-min-frames", type=int)
    control.add_argument("--progress-every", type=int, default=20)
    control.add_argument("--scene-progress-every", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("CUDA is not available in the current environment.")

    config_path = Path(args.config_path or DEFAULT_CONFIG).expanduser().resolve()
    masa_config = Path(args.masa_config or DEFAULT_MASA_CONFIG).expanduser().resolve()
    checkpoint = Path(args.masa_checkpoint or DEFAULT_MASA_CHECKPOINT).expanduser().resolve()
    masa_variant = _infer_masa_variant(masa_config)
    for path, label in ((config_path, "REMIND config"), (masa_config, "MASA config"), (checkpoint, "MASA checkpoint")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    stable_min_frames = int(
        _resolve_int(args.stable_min_frames, "REMIND_MASA_STABLE_MIN_FRAMES", 3) or 3
    )
    evaluation_config = Config(default_config_path=str(config_path)).to_dict()
    detector_config = evaluation_config.get("detector", {}) or {}
    davis_config = evaluation_config.get("davis", {}) or {}
    ignored_classes = [
        str(name).strip().lower()
        for name in (detector_config.get("ignored_classes", []) or [])
        if str(name).strip()
    ]
    allowed_davis_classes = davis_config.get("classes", None)
    max_frames = _resolve_int(args.max_frames, "REMIND_MASA_MAX_FRAMES", None)
    max_scenes = _resolve_int(args.max_scenes, "REMIND_MASA_MAX_SCENES", None)
    batch_size = _resolve_int(args.batch_size, "REMIND_MASA_BATCH_SIZE", max_scenes)
    progress_every = max(1, int(args.progress_every))
    scene_progress_every = max(1, int(args.scene_progress_every))

    custom_scenes: dict[str, CustomScene] = {}
    data_tar_root: Path | None = None
    annotations_tar_root: Path | None = None
    image_subdir = _resolve(args.image_subdir, "REMIND_IMAGE_SUBDIR", "dslr/resized_images")
    mask_variant = tar_batch._normalize_mask_variant(
        _resolve(args.mask_variant, "REMIND_MASK_VARIANT", "benchmark")
    )
    if args.dataset == "custom":
        custom_scenes = {scene.scene_id: scene for scene in _load_custom_scenes(args)}
        candidate_scene_ids = list(custom_scenes.keys())
    else:
        dataset_root = Path(
            _resolve(
                args.dataset_root,
                "REMIND_SCANNETPP_TAR_ROOT",
                str(WORKSPACE_DIR / "data" / "scannetpp_data"),
            )
        ).expanduser().resolve()
        data_tar_root = Path(
            _resolve(
                args.data_tar_root,
                "REMIND_SCANNETPP_DATA_TAR_ROOT",
                str(dataset_root / "data"),
            )
        ).expanduser().resolve()
        annotations_tar_root = Path(
            _resolve(
                args.annotations_tar_root,
                "REMIND_SCANNETPP_ANNOTATIONS_TAR_ROOT",
                str(dataset_root / "annotations"),
            )
        ).expanduser().resolve()
        exclude_file = _resolve_optional(
            args.exclude_scenes_file, "REMIND_BATCH_TAR_EXCLUDE_SCENES_FILE"
        )
        excluded = tar_batch._read_exclude_scene_ids(exclude_file) if exclude_file else set()
        candidate_scene_ids = tar_batch._resolve_tar_scene_ids(
            data_tar_root=data_tar_root,
            annotations_tar_root=annotations_tar_root,
            exclude_scenes=excluded,
            scenes_file=_resolve_optional(args.scenes_file, "REMIND_BATCH_TAR_SCENES_FILE") or "",
            scenes_list=(" ".join(args.scenes) if args.scenes else _env("REMIND_BATCH_TAR_SCENES")),
            single_scene=_resolve_optional(args.scene_id, "REMIND_SCENE_ID") or "",
        )
    if not candidate_scene_ids:
        raise RuntimeError("No scenes were resolved.")

    tracker_defaults = _masa_tracker_defaults(masa_config)
    effective_class_aware = bool(
        args.class_aware or tracker_defaults.get("with_cats", False)
    )
    effective_match_score_thr = float(
        args.match_score_thr
        if args.match_score_thr is not None
        else tracker_defaults.get("match_score_thr", 0.5)
    )
    effective_memo_tracklet_frames = int(
        args.memo_tracklet_frames
        if args.memo_tracklet_frames is not None
        else tracker_defaults.get("memo_tracklet_frames", 10)
    )
    effective_memo_momentum = float(
        args.memo_momentum
        if args.memo_momentum is not None
        else tracker_defaults.get("memo_momentum", 0.8)
    )
    effective_max_distance = float(
        args.max_distance
        if args.max_distance is not None
        else tracker_defaults.get("max_distance", -1)
    )
    automatic_run_id = _automatic_run_id(
        dataset=args.dataset,
        detections=args.detections,
        masa_variant=masa_variant,
        mask_variant=mask_variant,
        class_aware=effective_class_aware,
        max_distance=effective_max_distance,
        memo_tracklet_frames=effective_memo_tracklet_frames,
        match_score_thr=effective_match_score_thr,
        memo_momentum=effective_memo_momentum,
    )
    run_id = _resolve(
        args.run_id,
        "REMIND_MASA_RUN_ID",
        automatic_run_id,
    )
    output_dir = Path(
        _resolve(
            args.output_dir,
            "REMIND_MASA_OUTPUT_DIR",
            str(WORKSPACE_DIR / "Resultados" / "masa" / run_id),
        )
    ).expanduser().resolve()
    scenes_root = output_dir / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)

    scene_ids, registered_scene_ids, selection_mode, manifest_rows, per_scene_rows = batch.resolve_scene_schedule(
        candidate_scene_ids=batch.unique_preserve_order(candidate_scene_ids),
        batch_dir=output_dir,
        batch_size=batch_size,
    )

    yolo_provider = None
    yolo_model_path = _resolve_optional(args.yolo_model, "REMIND_YOLO_MODEL_PATH")
    yolo_conf = _resolve_float(args.yolo_conf, "REMIND_YOLO_CONF", 0.25)
    yolo_iou = _resolve_float(args.yolo_iou, "REMIND_YOLO_IOU", 0.7)
    yolo_imgsz = int(_resolve_int(args.yolo_imgsz, "REMIND_YOLO_IMGSZ", 640) or 640)
    yolo_max_det = int(_resolve_int(args.yolo_max_det, "REMIND_YOLO_MAX_DET", 100) or 100)
    mask_erosion_px = int(
        _resolve_int(args.mask_erosion_px, "REMIND_YOLO_MASK_EROSION_PX", 0) or 0
    )
    mask_erosion_iters = int(
        _resolve_int(args.mask_erosion_iters, "REMIND_YOLO_MASK_EROSION_ITERS", 1) or 1
    )
    if args.detections == "yolo":
        if not yolo_model_path:
            raise ValueError("YOLO mode requires --yolo-model or REMIND_YOLO_MODEL_PATH.")
        yolo_provider = YoloMaskProvider(
            model_path=Path(yolo_model_path).expanduser().resolve(),
            conf=yolo_conf,
            iou=yolo_iou,
            imgsz=yolo_imgsz,
            max_det=yolo_max_det,
            device=_resolve_optional(args.yolo_device, "REMIND_YOLO_DEVICE"),
            mask_erosion_px=mask_erosion_px,
            mask_erosion_iters=mask_erosion_iters,
        )

    from masa.apis import build_test_pipeline, init_masa

    model = init_masa(str(masa_config), str(checkpoint), device=args.device)
    tracker = model.tracker
    if args.class_aware:
        tracker.with_cats = True
    overrides = {
        "match_score_thr": args.match_score_thr,
        "memo_tracklet_frames": args.memo_tracklet_frames,
        "memo_momentum": args.memo_momentum,
        "max_distance": args.max_distance,
        "fps": args.tracker_fps,
    }
    for name, value in overrides.items():
        if value is not None:
            setattr(tracker, name, value)
    if args.tracker_fps is not None:
        tracker.growth_factor = float(tracker.fps) / 6.0
        tracker.distance_smoothing_factor = 100.0 / float(tracker.fps)
    test_pipeline = build_test_pipeline(model.cfg)
    run_config = {
        "run_id": run_id,
        "automatic_run_id": automatic_run_id,
        "created_at": batch._now_iso(),
        "tracker_family": "masa",
        "masa_variant": masa_variant,
        "dataset": args.dataset,
        "detection_source": args.detections,
        "gt_used_for_identity": False,
        "masa_root": str(MASA_ROOT),
        "masa_commit": _git_commit(MASA_ROOT),
        "masa_config": str(masa_config),
        "masa_checkpoint": str(checkpoint),
        "masa_checkpoint_sha256": _sha256(checkpoint),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "mmcv_version": _package_version("mmcv"),
        "mmengine_version": _package_version("mmengine"),
        "mmdet_version": _package_version("mmdet"),
        "ultralytics_version": _package_version("ultralytics"),
        "device": args.device,
        "fp16": bool(args.fp16),
        "class_aware": bool(tracker.with_cats),
        "class_aware_requested": bool(args.class_aware),
        "init_score_thr": float(tracker.init_score_thr),
        "obj_score_thr": float(tracker.obj_score_thr),
        "match_score_thr": float(tracker.match_score_thr),
        "memo_tracklet_frames": int(tracker.memo_tracklet_frames),
        "memo_momentum": float(tracker.memo_momentum),
        "distractor_score_thr": float(tracker.distractor_score_thr),
        "distractor_nms_thr": float(tracker.distractor_nms_thr),
        "with_cats": bool(tracker.with_cats),
        "max_distance": float(tracker.max_distance),
        "tracker_fps": float(tracker.fps),
        "postprocessing": False,
        "recovery_reappearance_outputs": True,
        "config_path": str(config_path),
        "ignored_classes": ignored_classes,
        "allowed_davis_classes": allowed_davis_classes,
        "stable_min_frames": stable_min_frames,
        "max_frames": max_frames,
        "selection_mode": selection_mode,
        "selected_scene_ids": scene_ids,
        "registered_scene_ids": registered_scene_ids,
        "data_tar_root": None if data_tar_root is None else str(data_tar_root),
        "annotations_tar_root": None if annotations_tar_root is None else str(annotations_tar_root),
        "image_subdir": image_subdir if args.dataset == "scannet-tar" else None,
        "mask_variant": mask_variant if args.dataset == "scannet-tar" else None,
        "yolo_model": yolo_model_path,
        "yolo_conf": yolo_conf if args.detections == "yolo" else None,
        "yolo_iou": yolo_iou if args.detections == "yolo" else None,
        "yolo_imgsz": yolo_imgsz if args.detections == "yolo" else None,
        "yolo_max_det": yolo_max_det if args.detections == "yolo" else None,
        "mask_erosion_px": mask_erosion_px if args.detections == "yolo" else None,
        "mask_erosion_iters": mask_erosion_iters if args.detections == "yolo" else None,
    }
    write_csv(output_dir / "run_config.csv", [run_config])
    write_json(output_dir / "run_config.json", run_config)

    scene_name_by_id = batch.merge_scene_name_index(
        base_scene_name_by_id={scene_id: scene_id for scene_id in registered_scene_ids},
        manifest_rows=manifest_rows,
        per_scene_rows=per_scene_rows,
    )
    failed = {
        str(row.get("scene_id")): str(row.get("error_message") or "")
        for row in manifest_rows
        if str(row.get("status") or "") == "failed"
    }
    batch.rebuild_batch_outputs(
        batch_dir=output_dir,
        run_id=run_id,
        selected_scene_ids=scene_ids,
        registered_scene_ids=registered_scene_ids,
        scene_name_by_id=scene_name_by_id,
        failed_scene_errors=failed,
    )
    _write_failed_scene_lists(output_dir, failed)

    scene_counts = {"completed": 0, "failed": 0, "skipped": 0}
    total_scenes = len(scene_ids)
    for scene_index, scene_id in enumerate(scene_ids, start=1):
        scene_key = batch.sanitize_name_for_path(scene_id)
        final_dir = scenes_root / scene_key
        if batch.scene_dir_is_complete(final_dir):
            scene_counts["skipped"] += 1
            if scene_index == total_scenes or scene_index % scene_progress_every == 0:
                print(
                    f"[MASA][SCENES] {scene_index}/{total_scenes} | "
                    f"completed={scene_counts['completed']} failed={scene_counts['failed']} "
                    f"skipped={scene_counts['skipped']}"
                )
            continue
        if final_dir.exists():
            backup = batch.reserve_incomplete_scene_backup_dir(final_dir)
            final_dir.rename(backup)
        temp_dir = scenes_root / f".tmp_{scene_key}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=False)

        started_at = batch._now_iso()
        bundle = None
        try:
            model.tracker.reset()
            if args.dataset == "custom":
                results, report = _evaluate_custom_scene(
                    scene=custom_scenes[scene_id],
                    config_path=config_path,
                    model=model,
                    test_pipeline=test_pipeline,
                    detection_mode=args.detections,
                    yolo=yolo_provider,
                    stable_min_frames=stable_min_frames,
                    max_frames=max_frames,
                    device=args.device,
                    fp16=args.fp16,
                    progress_every=progress_every,
                )
            else:
                assert data_tar_root is not None and annotations_tar_root is not None
                bundle = tar_batch._build_scene_bundle(
                    scene_id=scene_id,
                    data_tar_root=data_tar_root,
                    annotations_tar_root=annotations_tar_root,
                    mask_variant=mask_variant,
                    image_subdir=image_subdir,
                )
                results, report = _evaluate_tar_scene(
                    bundle=bundle,
                    config_path=config_path,
                    model=model,
                    test_pipeline=test_pipeline,
                    detection_mode=args.detections,
                    yolo=yolo_provider,
                    stable_min_frames=stable_min_frames,
                    max_frames=max_frames,
                    device=args.device,
                    fp16=args.fp16,
                    progress_every=progress_every,
                )

            finished_at = batch._now_iso()
            _write_scene_outputs(
                temp_dir=temp_dir,
                final_dir=final_dir,
                run_id=run_id,
                scene_id=scene_id,
                scene_name=scene_id,
                results=results,
                report=report,
                started_at=started_at,
                finished_at=finished_at,
                stable_min_frames=stable_min_frames,
                max_frames=max_frames,
                dataset=args.dataset,
                detections=args.detections,
                masa_variant=masa_variant,
            )
            temp_dir.rename(final_dir)
            failed.pop(scene_id, None)
            scene_name_by_id[scene_id] = scene_id
            scene_counts["completed"] += 1
        except Exception as exc:
            failed[scene_id] = f"{type(exc).__name__}: {exc}"
            scene_counts["failed"] += 1
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            print(f"[MASA][ERROR] Scene failed: {scene_id}: {exc}")
        finally:
            if bundle is not None:
                bundle.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch.rebuild_batch_outputs(
                batch_dir=output_dir,
                run_id=run_id,
                selected_scene_ids=scene_ids,
                registered_scene_ids=registered_scene_ids,
                scene_name_by_id=scene_name_by_id,
                failed_scene_errors=failed,
            )
            _write_failed_scene_lists(output_dir, failed)
        if scene_index == total_scenes or scene_index % scene_progress_every == 0:
            print(
                f"[MASA][SCENES] {scene_index}/{total_scenes} | "
                f"completed={scene_counts['completed']} failed={scene_counts['failed']} "
                f"skipped={scene_counts['skipped']}"
            )

    generate_recovery_outputs(
        output_dir,
        model="MASA",
        dataset="scannetpp" if args.dataset == "scannet-tar" else "custom",
    )


if __name__ == "__main__":
    main()
