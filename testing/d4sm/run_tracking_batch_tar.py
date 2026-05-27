from __future__ import annotations

import argparse
import gc
import os
import re
import shutil
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import torch
except Exception:
    torch = None


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
PROJECT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.config_loader import Config
from testing.common.generic_tracking_metrics import TrackingOnlyEvaluator
from testing.common.generic_tracking_reporting import build_generic_console_report, write_csv, write_json, write_text
from testing.davis_gt import DavisGroundTruthLoader
from testing.d4sm import run_tracking_batch as d4sm_batch
from testing.d4sm.run_tracking_test import (
    TarFrameSource,
    TarSceneBundle,
    _build_detections,
    _build_scene_bundle,
    _cuda_autocast_context,
    _gt_object_to_full_mask,
    _inference_context,
    _normalize_mask_variant,
    _patched_tar_davis_segmenter,
    _sync_cuda,
    build_runtime_memory_telemetry,
    capture_cuda_memory_stats,
    make_process_handle,
    read_process_rss_bytes,
    release_cuda_scene_resources,
    reset_cuda_peak_memory_stats,
    reset_d4sm_tracker_scene_state,
)


@dataclass
class YoloInitDetection:
    mask: np.ndarray
    class_id: int
    class_name: str
    confidence: float


class YoloInitMaskProvider:
    def __init__(
        self,
        *,
        model_path: str,
        conf: float,
        iou: float,
        imgsz: int,
        device: str | None,
    ) -> None:
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "YOLO init mode requires the `ultralytics` package. "
                "Install it with: pip install ultralytics"
            ) from exc

        path = Path(str(model_path)).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"YOLO model not found: {path}")

        self.model = YOLO(str(path))
        self.model_path = str(path)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = None if not device else str(device)

        names = getattr(self.model, "names", {}) or {}
        if isinstance(names, dict):
            self.class_id_to_name = {int(k): str(v) for k, v in names.items()}
        else:
            self.class_id_to_name = {idx: str(name) for idx, name in enumerate(list(names))}

    def segment(self, frame_bgr: np.ndarray) -> list[YoloInitDetection]:
        h, w = frame_bgr.shape[:2]
        kwargs: dict[str, Any] = {
            "task": "segment",
            "conf": float(self.conf),
            "iou": float(self.iou),
            "imgsz": int(self.imgsz),
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device

        results = self.model(frame_bgr, **kwargs)
        if not results or len(results) <= 0:
            return []
        result = results[0]
        if getattr(result, "masks", None) is None or result.masks.data is None:
            return []

        masks_np = result.masks.data.cpu().numpy()
        cls_np = result.boxes.cls.cpu().numpy().astype(int)
        conf_np = result.boxes.conf.cpu().numpy()
        out: list[YoloInitDetection] = []
        for idx in range(int(len(masks_np))):
            mask = masks_np[idx]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            class_id = int(cls_np[idx])
            out.append(
                YoloInitDetection(
                    mask=(mask > 0.5),
                    class_id=class_id,
                    class_name=self.class_id_to_name.get(class_id, f"class_{class_id}"),
                    confidence=float(conf_np[idx]),
                )
            )
        return out


def _env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name, None)
    if raw is None:
        return str(default)
    return str(raw).strip()


def _resolve(value: Any, env_name: str, default: str) -> str:
    if value is not None and str(value).strip():
        return str(value).strip()
    env_value = _env_str(env_name, "")
    if env_value:
        return env_value
    return str(default)


def _resolve_optional(value: Any, env_name: str) -> str | None:
    if value is not None and str(value).strip():
        return str(value).strip()
    env_value = _env_str(env_name, "")
    return env_value or None


def _resolve_int(value: Any, env_name: str, default: int | None = None) -> int | None:
    raw = _resolve_optional(value, env_name)
    if raw is None:
        return default
    return int(raw)


def _resolve_bool(value: Any, env_name: str, default: bool = False) -> bool:
    if value is not None:
        return bool(value)
    env_value = _env_str(env_name, "")
    if env_value:
        return env_value.lower() in {"1", "true", "yes", "on"}
    return default


def _normalize_class_name(name: Any) -> str:
    return str(name or "").strip().lower().replace("_", " ")


def _mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = np.asarray(mask_a).astype(bool, copy=False)
    b = np.asarray(mask_b).astype(bool, copy=False)
    if a.shape != b.shape:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    if inter <= 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return float(inter) / float(union) if union > 0 else 0.0


def _read_exclude_scene_ids(file_path: str | Path) -> set[str]:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Exclude-scenes file not found: {path}")
    ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        for token in re.split(r"[,;\s]+", line):
            cleaned = token.strip().strip("`'\"").strip()
            if cleaned:
                ids.add(cleaned)
    return ids


def _discover_tar_scene_ids(*, data_tar_root: Path, annotations_tar_root: Path) -> list[str]:
    data_ids = {p.stem for p in data_tar_root.glob("*.tar") if p.is_file()}
    annotation_ids = {p.stem for p in annotations_tar_root.glob("*.tar") if p.is_file()}
    return sorted(data_ids & annotation_ids)


def _resolve_tar_scene_ids(
    *,
    data_tar_root: Path,
    annotations_tar_root: Path,
    exclude_scenes: set[str] | None,
    scenes_file: str,
    scenes_list: str,
    single_scene: str,
) -> list[str]:
    if scenes_file:
        scene_ids = d4sm_batch.read_scene_ids_from_file(scenes_file)
    elif scenes_list:
        scene_ids = d4sm_batch.unique_preserve_order(re.split(r"[\s,;]+", scenes_list))
    elif single_scene:
        scene_ids = [single_scene]
    else:
        scene_ids = _discover_tar_scene_ids(data_tar_root=data_tar_root, annotations_tar_root=annotations_tar_root)

    if exclude_scenes:
        before = len(scene_ids)
        scene_ids = [sid for sid in scene_ids if sid not in exclude_scenes]
        if before != len(scene_ids):
            print(f"[D4SM-YOLO-INIT][BATCH] Excluded {before - len(scene_ids)} scene(s).")
    return scene_ids


def _ensure_d4sm_import_path() -> Path:
    d4sm_root = (PROJECT_DIR / "third_party" / "d4sm").resolve()
    if str(d4sm_root) not in sys.path:
        sys.path.insert(0, str(d4sm_root))
    return d4sm_root


def resolve_d4sm_runtime_config(
    *,
    checkpoint_dir_override: str | None = None,
    model_size_override: str | None = None,
    offload_state_to_cpu_override: bool | None = None,
) -> dict[str, Any]:
    d4sm_root = _ensure_d4sm_import_path()

    if model_size_override:
        model_size = str(model_size_override).strip().lower()
    else:
        model_size = _env_str("REMIND_D4SM_MODEL_SIZE", "large").lower() or "large"

    if checkpoint_dir_override:
        checkpoint_dir = str(Path(checkpoint_dir_override).expanduser().resolve())
    else:
        checkpoint_dir = _env_str("REMIND_D4SM_CHECKPOINT_DIR", "")
        if checkpoint_dir:
            checkpoint_dir = str(Path(checkpoint_dir).expanduser().resolve())
        else:
            checkpoint_dir = str((d4sm_root / "checkpoints").resolve())

    if offload_state_to_cpu_override is not None:
        offload_state_to_cpu = bool(offload_state_to_cpu_override)
    else:
        offload_state_to_cpu = _env_str("REMIND_D4SM_OFFLOAD_STATE_TO_CPU", "").lower() in {
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
        cuda_visible_devices = _env_str("CUDA_VISIBLE_DEVICES", "")
        suffix = f" CUDA_VISIBLE_DEVICES={cuda_visible_devices!r}." if cuda_visible_devices else ""
        raise RuntimeError(
            "d4sm requires a visible CUDA GPU to load SAM2/D4SM, "
            f"but torch.cuda.is_available() == False.{suffix}"
        )
    cfg = dict(runtime_config or resolve_d4sm_runtime_config())
    from tracking_wrapper_mot import DAM4SAMMOT

    return DAM4SAMMOT(
        model_size=str(cfg["model_size"]),
        checkpoint_dir=str(cfg["checkpoint_dir"]),
        offload_state_to_cpu=bool(cfg["offload_state_to_cpu"]),
    )


def _select_yolo_init_regions(
    *,
    gt_objects: dict[int, Any],
    pending_gt_ids: list[int],
    yolo_detections: list[YoloInitDetection],
    frame_shape: tuple[int, int],
    min_iou: float,
) -> tuple[list[dict[str, np.ndarray | str]], list[int], list[dict[str, Any]]]:
    regions: list[dict[str, np.ndarray | str]] = []
    matched_gt_ids: list[int] = []
    rows: list[dict[str, Any]] = []
    used_det_indices: set[int] = set()

    for gt_id in sorted(int(x) for x in pending_gt_ids):
        gt_obj = gt_objects.get(int(gt_id), None)
        gt_class = _normalize_class_name(getattr(gt_obj, "class_name", None))
        gt_mask = _gt_object_to_full_mask(gt_obj, frame_shape)
        best_idx: int | None = None
        best_iou = 0.0
        best_conf = 0.0
        same_class_count = 0

        for det_idx, det in enumerate(yolo_detections):
            if int(det_idx) in used_det_indices:
                continue
            if _normalize_class_name(det.class_name) != gt_class:
                continue
            same_class_count += 1
            score = _mask_iou(det.mask, gt_mask)
            if score > best_iou:
                best_idx = int(det_idx)
                best_iou = float(score)
                best_conf = float(det.confidence)

        accepted = bool(best_idx is not None and best_iou >= float(min_iou))
        rows.append(
            {
                "gt_instance_id": int(gt_id),
                "gt_class_name": getattr(gt_obj, "class_name", None),
                "same_class_yolo_candidates": int(same_class_count),
                "best_yolo_idx": best_idx,
                "best_iou": float(best_iou),
                "best_confidence": float(best_conf),
                "accepted": bool(accepted),
            }
        )
        if not accepted or best_idx is None:
            continue

        used_det_indices.add(int(best_idx))
        matched_gt_ids.append(int(gt_id))
        regions.append(
            {
                "name": f"obj_{int(gt_id)}",
                "mask": np.asarray(yolo_detections[best_idx].mask).astype(np.uint8, copy=False),
            }
        )

    return regions, matched_gt_ids, rows


def evaluate_scene_tar_yolo_init(
    *,
    config_path: Path,
    scene_bundle: TarSceneBundle,
    stable_min_frames: int,
    max_frames: int | None,
    tracker: Any | None,
    yolo_provider: YoloInitMaskProvider,
    yolo_init_min_iou: float,
) -> tuple[dict[str, Any], str]:
    if torch is None:
        raise RuntimeError("d4sm requires torch, but the current interpreter cannot import it.")
    from hydra.core.global_hydra import GlobalHydra

    config = Config(default_config_path=str(config_path)).to_dict()
    davis_cfg = config.setdefault("davis", {})
    davis_cfg["sequence_name"] = str(scene_bundle.scene_id)
    davis_cfg["variant"] = "bench" if _normalize_mask_variant(scene_bundle.mask_variant) == "benchmark_instance" else "raw"
    davis_cfg["tar_scene_bundle"] = scene_bundle
    davis_cfg["prefetch_annotations"] = False

    timing_cfg = config.setdefault("timing", {})
    timing_cfg["enabled"] = False
    timing_cfg["table"] = False
    timing_cfg["detail_keys"] = []

    frame_names = list(scene_bundle.frame_names)
    if max_frames is not None:
        frame_names = frame_names[: max(0, int(max_frames))]
    if not frame_names:
        raise RuntimeError(f"No frames to evaluate for {scene_bundle.scene_id}")

    print(
        f"[D4SM-YOLO-INIT][scene={scene_bundle.scene_id}] "
        f"start | frames={len(frame_names)} | image_subdir={scene_bundle.image_subdir} | "
        f"mask_variant={scene_bundle.mask_variant} | recovery=on"
    )

    tracker_owner = tracker is None
    if tracker_owner:
        tracker = create_d4sm_tracker()
    else:
        reset_d4sm_tracker_scene_state(tracker)
    tracker.n_frames = int(len(frame_names))

    frame_source = TarFrameSource(scene_bundle)
    process = make_process_handle()
    dataset_gt_by_tracker_id: dict[int, int] = {}
    dataset_class_by_tracker_id: dict[int, str | None] = {}
    initialized_gt_ids: set[int] = set()
    yolo_init_attempt_rows: list[dict[str, Any]] = []

    total_read_ms = 0.0
    total_pipeline_ms = 0.0
    total_gt_ms = 0.0
    total_eval_ms = 0.0
    total_loop_ms = 0.0
    per_frame_timing_by_frame_id: dict[int, dict[str, float]] = {}
    per_frame_runtime_memory_by_frame_id: dict[int, dict[str, int | None]] = {}

    gt_loader = None
    evaluator = None
    tar_patch_ctx = _patched_tar_davis_segmenter()
    try:
        tar_patch_ctx.__enter__()
        gt_loader = DavisGroundTruthLoader(config)
        evaluator = TrackingOnlyEvaluator(stable_min_frames=stable_min_frames)

        for idx, frame_name in enumerate(frame_names):
            loop_t0 = perf_counter()
            frame_id = int(idx)
            rss_before = read_process_rss_bytes(process)

            read_t0 = perf_counter()
            frame_bgr = frame_source.read_bgr(frame_name)
            if frame_bgr is None:
                raise RuntimeError(f"Could not read frame {frame_name} from {scene_bundle.data_tar_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            read_ms = (perf_counter() - read_t0) * 1000.0
            rss_after_read = read_process_rss_bytes(process)

            gt_t0 = perf_counter()
            gt_objects = gt_loader.load_frame(frame_id=frame_id, target_shape=frame_rgb.shape[:2])
            gt_ms = (perf_counter() - gt_t0) * 1000.0

            reset_cuda_peak_memory_stats()
            pipeline_t0 = perf_counter()
            appended_new_masks: list[np.ndarray] = []
            appended_new_pred_ids: list[int] = []
            tracker_initialized = bool(len(getattr(tracker, "all_obj_ids", []) or []) > 0)

            if tracker_initialized:
                with _inference_context():
                    with _cuda_autocast_context():
                        _sync_cuda()
                        outputs = tracker.track(image)
                        _sync_cuda()
                current_tracker_pred_masks = [
                    np.asarray(mask).astype(np.uint8, copy=False)
                    for mask in (outputs.get("masks", []) or [])
                ]
                current_tracker_pred_ids = [int(x) for x in tracker.all_obj_ids]
            else:
                current_tracker_pred_masks = []
                current_tracker_pred_ids = []

            pending_gt_ids = sorted(int(gt_id) for gt_id in gt_objects.keys() if int(gt_id) not in initialized_gt_ids)
            if pending_gt_ids:
                yolo_dets = yolo_provider.segment(frame_bgr)
                init_regions, matched_gt_ids, attempt_rows = _select_yolo_init_regions(
                    gt_objects=gt_objects,
                    pending_gt_ids=pending_gt_ids,
                    yolo_detections=yolo_dets,
                    frame_shape=frame_rgb.shape[:2],
                    min_iou=float(yolo_init_min_iou),
                )
                for row in attempt_rows:
                    item = dict(row)
                    item["frame_id"] = int(frame_id)
                    item["n_yolo_detections"] = int(len(yolo_dets))
                    yolo_init_attempt_rows.append(item)

                if init_regions:
                    prev_len = int(len(tracker.all_obj_ids))
                    if tracker_initialized:
                        with _inference_context():
                            with _cuda_autocast_context():
                                tracker.add_objects(image, init_regions)
                        new_tracker_ids = [int(x) for x in tracker.all_obj_ids[prev_len:]]
                        if len(new_tracker_ids) != len(matched_gt_ids):
                            raise RuntimeError("d4sm did not return the same number of internal IDs after add_objects.")
                        appended_new_pred_ids.extend(new_tracker_ids)
                        appended_new_masks.extend(
                            np.asarray(region["mask"]).astype(np.uint8, copy=False)
                            for region in init_regions
                        )
                    else:
                        with _inference_context():
                            with _cuda_autocast_context():
                                tracker.initialize(image, init_regions)
                        new_tracker_ids = [int(x) for x in tracker.all_obj_ids]
                        if len(new_tracker_ids) != len(matched_gt_ids):
                            raise RuntimeError("d4sm did not return the same number of internal IDs after initialization.")
                        current_tracker_pred_ids = list(new_tracker_ids)
                        current_tracker_pred_masks = [
                            np.asarray(region["mask"]).astype(np.uint8, copy=False)
                            for region in init_regions
                        ]

                    for tracker_id, gt_id in zip(new_tracker_ids, matched_gt_ids):
                        gt_obj = gt_objects.get(int(gt_id), None)
                        dataset_gt_by_tracker_id[int(tracker_id)] = int(gt_id)
                        dataset_class_by_tracker_id[int(tracker_id)] = getattr(gt_obj, "class_name", None)
                        initialized_gt_ids.add(int(gt_id))

            pipeline_ms = (perf_counter() - pipeline_t0) * 1000.0
            rss_after_pipeline = read_process_rss_bytes(process)
            gpu_after_pipeline = capture_cuda_memory_stats()

            detections, det_to_pred_id = _build_detections(
                tracker_pred_masks=current_tracker_pred_masks,
                tracker_pred_ids=current_tracker_pred_ids,
                appended_new_masks=appended_new_masks,
                appended_new_pred_ids=appended_new_pred_ids,
            )
            pred_info_by_id = {
                int(tracker_id): {
                    "instance_label": f"d4sm_track_{int(tracker_id):04d}",
                    "class_name": dataset_class_by_tracker_id.get(int(tracker_id), None),
                }
                for tracker_id in dataset_gt_by_tracker_id.keys()
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
            loop_ms = (perf_counter() - loop_t0) * 1000.0
            total_read_ms += float(read_ms)
            total_pipeline_ms += float(pipeline_ms)
            total_gt_ms += float(gt_ms)
            total_eval_ms += float(eval_ms)
            total_loop_ms += float(loop_ms)
            per_frame_timing_by_frame_id[int(frame_id)] = {
                "read_ms": float(read_ms),
                "pipeline_ms": float(pipeline_ms),
                "gt_ms": float(gt_ms),
                "eval_ms": float(eval_ms),
                "post_ms": float(gt_ms + eval_ms),
                "loop_ms": float(loop_ms),
            }

            processed_frames = int(idx + 1)
            if processed_frames == 1 or processed_frames % 20 == 0 or processed_frames == len(frame_names):
                print(
                    f"[D4SM-YOLO-INIT][scene={scene_bundle.scene_id}] "
                    f"progress {processed_frames}/{len(frame_names)} | frame_id={frame_id} | "
                    f"pending={len(pending_gt_ids)} | initialized={len(initialized_gt_ids)} | "
                    f"pipeline={pipeline_ms:.2f} ms | gt={gt_ms:.2f} ms | eval={eval_ms:.2f} ms"
                )

        results = evaluator.finalize()
    finally:
        if tracker_owner:
            tracker = None
        gt_loader = None
        evaluator = None
        process = None
        gc.collect()
        try:
            if tracker_owner:
                from hydra.core.global_hydra import GlobalHydra

                if GlobalHydra.instance().is_initialized():
                    GlobalHydra.instance().clear()
        except Exception:
            pass
        try:
            tar_patch_ctx.__exit__(None, None, None)
        except Exception:
            pass
        try:
            scene_bundle.close()
        except Exception:
            pass
        release_cuda_scene_resources()

    n_processed_frames = int(len(frame_names))
    avg_divisor = float(max(1, n_processed_frames))
    yolo_attempts = int(len(yolo_init_attempt_rows))
    yolo_successes = int(sum(1 for row in yolo_init_attempt_rows if bool(row.get("accepted", False))))
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
        "total_loop_ms": float(total_loop_ms),
        "avg_loop_ms": float(total_loop_ms / avg_divisor),
        "total_runtime_seconds": float(total_loop_ms / 1000.0),
        "avg_runtime_seconds": float((total_loop_ms / avg_divisor) / 1000.0),
    }
    results["timing_summary"] = timing_summary
    results["yolo_init_attempts"] = list(yolo_init_attempt_rows)
    summary = results.setdefault("summary", {})
    summary.update(timing_summary)
    summary.update(
        {
            "init_source": "yolo",
            "init_recovery_enabled": True,
            "n_initialized_gt_ids": int(len(initialized_gt_ids)),
            "yolo_model_path": str(yolo_provider.model_path),
            "yolo_conf": float(yolo_provider.conf),
            "yolo_iou": float(yolo_provider.iou),
            "yolo_imgsz": int(yolo_provider.imgsz),
            "yolo_device": yolo_provider.device or "auto",
            "yolo_init_min_iou": float(yolo_init_min_iou),
            "yolo_init_attempts": int(yolo_attempts),
            "yolo_init_successes": int(yolo_successes),
            "yolo_init_failures": int(yolo_attempts - yolo_successes),
            "yolo_init_success_rate": (float(yolo_successes) / float(yolo_attempts)) if yolo_attempts else None,
        }
    )

    for row in (results.get("per_frame", []) or []):
        frame_id = int(row.get("frame_id", -1))
        row.update(per_frame_timing_by_frame_id.get(frame_id, {}))
        row.update(per_frame_runtime_memory_by_frame_id.get(frame_id, {}))

    return results, build_generic_console_report(results)


def write_scene_outputs_yolo_init(
    *,
    temp_scene_dir: Path,
    final_scene_dir: Path,
    run_id: str,
    scene_id: str,
    scene_name: str,
    results: dict[str, Any],
    scene_report: str,
    scene_started_at: str,
    scene_finished_at: str,
    stable_min_frames: int,
    max_frames: int | None,
    model_size: str,
) -> None:
    d4sm_batch.write_scene_outputs(
        temp_scene_dir=temp_scene_dir,
        final_scene_dir=final_scene_dir,
        run_id=run_id,
        scene_id=scene_id,
        scene_name=scene_name,
        results=results,
        scene_report=scene_report,
        scene_started_at=scene_started_at,
        scene_finished_at=scene_finished_at,
        stable_min_frames=stable_min_frames,
        max_frames=max_frames,
        model_size=model_size,
    )
    summary = dict(results.get("summary", {}) or {})
    scene_summary_row = d4sm_batch.build_scene_summary_row(
        run_id=run_id,
        scene_id=scene_id,
        scene_name=scene_name,
        results=results,
        scene_started_at=scene_started_at,
        scene_finished_at=scene_finished_at,
        output_dir=final_scene_dir,
        stable_min_frames=stable_min_frames,
        max_frames=max_frames,
        model_size=model_size,
    )
    scene_summary_row.update(
        {
            "init_source": summary.get("init_source", None),
            "init_recovery_enabled": summary.get("init_recovery_enabled", None),
            "n_initialized_gt_ids": summary.get("n_initialized_gt_ids", None),
            "yolo_model_path": summary.get("yolo_model_path", None),
            "yolo_conf": summary.get("yolo_conf", None),
            "yolo_iou": summary.get("yolo_iou", None),
            "yolo_imgsz": summary.get("yolo_imgsz", None),
            "yolo_device": summary.get("yolo_device", None),
            "yolo_init_min_iou": summary.get("yolo_init_min_iou", None),
            "yolo_init_attempts": summary.get("yolo_init_attempts", None),
            "yolo_init_successes": summary.get("yolo_init_successes", None),
            "yolo_init_failures": summary.get("yolo_init_failures", None),
            "yolo_init_success_rate": summary.get("yolo_init_success_rate", None),
        }
    )
    d4sm_batch.write_single_row_csv(temp_scene_dir / d4sm_batch.SCENE_TABLE_FILES["per_scene"], scene_summary_row)
    write_json(temp_scene_dir / "tracking_eval.json", results)
    write_csv(
        temp_scene_dir / "yolo_init_attempts.csv",
        d4sm_batch.annotate_rows(
            results.get("yolo_init_attempts", []) or [],
            run_id=run_id,
            scene_id=scene_id,
            scene_name=scene_name,
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run D4SM over ScanNet++ .tar scenes using YOLO masks for object "
            "initialization. GT is used only to decide which visible objects "
            "are still pending and to select the same-class YOLO mask with "
            "maximum IoU. If YOLO misses an object, there is no GT fallback; "
            "the object is retried on later visible frames."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    paths = p.add_argument_group("data paths")
    paths.add_argument("--dataset-root", metavar="DIR", help="Top-level dataset directory. [env: REMIND_SCANNETPP_TAR_ROOT]")
    paths.add_argument("--data-tar-root", metavar="DIR", help="Directory containing per-scene data .tar files. [env: REMIND_SCANNETPP_DATA_TAR_ROOT, default: <dataset-root>/data]")
    paths.add_argument("--annotations-tar-root", metavar="DIR", help="Directory containing per-scene annotation .tar files. [env: REMIND_SCANNETPP_ANNOTATIONS_TAR_ROOT, default: <dataset-root>/annotations]")
    paths.add_argument("--config-path", metavar="FILE", help="Path to the YAML config file (default: REMIND/config/default_config.yaml).")
    paths.add_argument("--image-subdir", metavar="PATH", help="Relative image sub-directory inside each data tar. [env: REMIND_IMAGE_SUBDIR, default: dslr/resized_images]")
    paths.add_argument("--mask-variant", metavar="NAME", help="Mask variant name (e.g. benchmark, raw). [env: REMIND_MASK_VARIANT, default: benchmark]")

    scenes = p.add_argument_group("scene selection")
    scenes.add_argument("--scenes", "--scene-ids", metavar="ID", nargs="+", help="Explicit list of scene IDs. [env: REMIND_D4SM_YOLO_INIT_BATCH_SCENES]")
    scenes.add_argument("--scenes-file", metavar="FILE", help="Text file listing scene IDs. [env: REMIND_D4SM_YOLO_INIT_BATCH_SCENES_FILE]")
    scenes.add_argument("--scene-id", metavar="ID", help="Evaluate a single scene. [env: REMIND_SCENE_ID]")
    scenes.add_argument("--exclude-scenes-file", metavar="FILE", help="Text file listing scene IDs to exclude. [env: REMIND_D4SM_YOLO_INIT_EXCLUDE_SCENES_FILE]")

    model = p.add_argument_group("D4SM / SAM model")
    model.add_argument("--checkpoint-dir", metavar="DIR", help="Directory containing SAM2/D4SM checkpoints. [env: REMIND_D4SM_CHECKPOINT_DIR, default: <project>/third_party/d4sm/checkpoints]")
    model.add_argument("--model-size", metavar="SIZE", help="D4SM model size, e.g. large, base_plus, small, tiny. [env: REMIND_D4SM_MODEL_SIZE, default: large]")
    model.add_argument("--offload-state-to-cpu", action="store_true", default=None, help="Offload D4SM tracker state to CPU between frames. [env: REMIND_D4SM_OFFLOAD_STATE_TO_CPU]")

    batch = p.add_argument_group("batch control")
    batch.add_argument("--output-dir", metavar="DIR", help="Root directory for batch results. [env: REMIND_D4SM_YOLO_INIT_OUTPUT_DIR, default: <project>/outputs/d4sm/testing_batch_tar_yolo_init]")
    batch.add_argument("--run-id", metavar="NAME", help="Identifier for this run. [env: REMIND_D4SM_YOLO_INIT_RUN_ID, default: d4sm_yolo_init_tar]")
    batch.add_argument("--max-scenes", type=int, metavar="N", help="Maximum number of scenes. [env: REMIND_D4SM_YOLO_INIT_MAX_SCENES]")
    batch.add_argument("--batch-size", type=int, metavar="N", help="Batch size. [env: REMIND_D4SM_YOLO_INIT_BATCH_SIZE]")
    batch.add_argument("--stable-min-frames", type=int, metavar="N", help="Stable frames threshold. [env: REMIND_D4SM_YOLO_INIT_STABLE_MIN_FRAMES, default: 3]")
    batch.add_argument("--max-frames", type=int, metavar="N", help="Maximum frames per scene. [env: REMIND_D4SM_YOLO_INIT_MAX_FRAMES]")

    yolo = p.add_argument_group("YOLO initialization")
    yolo.add_argument("--yolo-model", metavar="FILE", help="Ultralytics YOLO segmentation .pt model. [env: REMIND_YOLO_MODEL_PATH]")
    yolo.add_argument("--yolo-conf", type=float, metavar="F", help="YOLO confidence threshold. [env: REMIND_YOLO_CONF, default: 0.25]")
    yolo.add_argument("--yolo-iou", type=float, metavar="F", help="YOLO NMS IoU threshold. [env: REMIND_YOLO_IOU, default: 0.7]")
    yolo.add_argument("--yolo-imgsz", type=int, metavar="PX", help="YOLO inference image size. [env: REMIND_YOLO_IMGSZ, default: 640]")
    yolo.add_argument("--yolo-device", metavar="DEV", help="YOLO device, e.g. cuda:0 or cpu. [env: REMIND_YOLO_DEVICE, default: auto]")
    yolo.add_argument("--yolo-init-min-iou", type=float, metavar="F", help="Minimum GT-vs-YOLO IoU to accept an init mask. [env: REMIND_D4SM_YOLO_INIT_MIN_IOU, default: 0.1]")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else SRC_DIR / "config" / "default_config.yaml"
    dataset_root = Path(
        _resolve(args.dataset_root, "REMIND_SCANNETPP_TAR_ROOT", str(PROJECT_DIR / "data" / "scannetpp_data"))
    ).expanduser().resolve()
    data_tar_root = Path(
        _resolve(args.data_tar_root, "REMIND_SCANNETPP_DATA_TAR_ROOT", str(dataset_root / "data"))
    ).expanduser().resolve()
    annotations_tar_root = Path(
        _resolve(args.annotations_tar_root, "REMIND_SCANNETPP_ANNOTATIONS_TAR_ROOT", str(dataset_root / "annotations"))
    ).expanduser().resolve()
    image_subdir = _resolve(args.image_subdir, "REMIND_IMAGE_SUBDIR", "dslr/resized_images")
    mask_variant = _normalize_mask_variant(_resolve(args.mask_variant, "REMIND_MASK_VARIANT", "benchmark"))

    yolo_model_path = _resolve_optional(args.yolo_model, "REMIND_YOLO_MODEL_PATH")
    if not yolo_model_path:
        raise ValueError("--yolo-model or REMIND_YOLO_MODEL_PATH is required.")
    yolo_conf = float(_resolve(args.yolo_conf, "REMIND_YOLO_CONF", "0.25"))
    yolo_iou = float(_resolve(args.yolo_iou, "REMIND_YOLO_IOU", "0.7"))
    yolo_imgsz = int(_resolve(args.yolo_imgsz, "REMIND_YOLO_IMGSZ", "640"))
    yolo_device = _resolve_optional(args.yolo_device, "REMIND_YOLO_DEVICE")
    yolo_init_min_iou = float(_resolve(args.yolo_init_min_iou, "REMIND_D4SM_YOLO_INIT_MIN_IOU", "0.1"))

    # --- D4SM / SAM checkpoint resolution (CLI > env > default) ---
    checkpoint_dir_override = _resolve_optional(args.checkpoint_dir, "")  # env handled inside resolve_d4sm_runtime_config
    model_size_override = _resolve_optional(args.model_size, "")
    offload_override = args.offload_state_to_cpu  # None when not passed, True when flag present

    exclude_file = _resolve_optional(args.exclude_scenes_file, "REMIND_D4SM_YOLO_INIT_EXCLUDE_SCENES_FILE")
    exclude_scenes = _read_exclude_scene_ids(exclude_file) if exclude_file else set()
    scenes_list = ",".join(args.scenes) if args.scenes else _env_str("REMIND_D4SM_YOLO_INIT_BATCH_SCENES", "")
    scene_ids = _resolve_tar_scene_ids(
        data_tar_root=data_tar_root,
        annotations_tar_root=annotations_tar_root,
        exclude_scenes=exclude_scenes or None,
        scenes_file=_resolve_optional(args.scenes_file, "REMIND_D4SM_YOLO_INIT_BATCH_SCENES_FILE") or "",
        scenes_list=scenes_list,
        single_scene=_resolve_optional(args.scene_id, "REMIND_SCENE_ID") or "",
    )
    if not scene_ids:
        raise RuntimeError("No .tar scenes were resolved for the batch.")

    run_id = _resolve(args.run_id, "REMIND_D4SM_YOLO_INIT_RUN_ID", "d4sm_yolo_init_tar")
    output_root = Path(
        _resolve(
            args.output_dir,
            "REMIND_D4SM_YOLO_INIT_OUTPUT_DIR",
            str(PROJECT_DIR / "outputs" / "d4sm" / "testing_batch_tar_yolo_init"),
        )
    ).expanduser().resolve()
    batch_dir = output_root
    scenes_root = batch_dir / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)

    max_scenes = _resolve_int(args.max_scenes, "REMIND_D4SM_YOLO_INIT_MAX_SCENES", None)
    batch_size = _resolve_int(args.batch_size, "REMIND_D4SM_YOLO_INIT_BATCH_SIZE", max_scenes)
    scene_ids, registered_scene_ids, selection_mode, existing_manifest_rows, existing_per_scene_rows = d4sm_batch.resolve_scene_schedule(
        candidate_scene_ids=d4sm_batch.unique_preserve_order([str(scene_id) for scene_id in scene_ids]),
        batch_dir=batch_dir,
        batch_size=batch_size,
    )

    stable_min_frames = _resolve_int(args.stable_min_frames, "REMIND_D4SM_YOLO_INIT_STABLE_MIN_FRAMES", 3) or 3
    max_frames = _resolve_int(args.max_frames, "REMIND_D4SM_YOLO_INIT_MAX_FRAMES", None)

    # Build runtime config with CLI overrides
    runtime_config = resolve_d4sm_runtime_config(
        checkpoint_dir_override=checkpoint_dir_override,
        model_size_override=model_size_override,
        offload_state_to_cpu_override=offload_override,
    )
    model_size = str(runtime_config["model_size"])

    d4sm_batch.write_single_row_csv(
        batch_dir / "run_config.csv",
        {
            "run_id": str(run_id),
            "batch_name": str(run_id),
            "batch_dir": str(batch_dir.resolve()),
            "created_at": d4sm_batch._now_iso(),
            "tracker_family": "d4sm",
            "init_source": "yolo",
            "init_recovery_enabled": True,
            "model_size": str(model_size),
            "checkpoint_dir": str(runtime_config["checkpoint_dir"]),
            "offload_state_to_cpu": bool(runtime_config["offload_state_to_cpu"]),
            "stable_min_frames": int(stable_min_frames),
            "max_frames": None if max_frames is None else int(max_frames),
            "max_scenes": None if max_scenes is None else int(max_scenes),
            "batch_size": None if batch_size is None else int(batch_size),
            "selection_mode": str(selection_mode),
            "mask_variant": str(mask_variant),
            "image_subdir": str(image_subdir),
            "data_tar_root": str(data_tar_root.resolve()),
            "annotations_tar_root": str(annotations_tar_root.resolve()),
            "yolo_model_path": str(Path(yolo_model_path).expanduser().resolve()),
            "yolo_conf": float(yolo_conf),
            "yolo_iou": float(yolo_iou),
            "yolo_imgsz": int(yolo_imgsz),
            "yolo_device": yolo_device or "auto",
            "yolo_init_min_iou": float(yolo_init_min_iou),
            "selected_scene_count": int(len(scene_ids)),
            "selected_scene_ids": [str(scene_id) for scene_id in scene_ids],
            "registered_scene_count": int(len(registered_scene_ids)),
            "registered_scene_ids": [str(scene_id) for scene_id in registered_scene_ids],
        },
    )

    scene_name_by_id = d4sm_batch.merge_scene_name_index(
        base_scene_name_by_id={str(scene_id): str(scene_id) for scene_id in registered_scene_ids},
        manifest_rows=existing_manifest_rows,
        per_scene_rows=existing_per_scene_rows,
    )
    failed_scene_errors = {
        str(row.get("scene_id", "") or "").strip(): str(row.get("error_message", "") or "")
        for row in existing_manifest_rows
        if str(row.get("scene_id", "") or "").strip()
        and str(row.get("status", "") or "").strip() == "failed"
        and str(row.get("error_message", "") or "").strip()
    }

    d4sm_batch.rebuild_batch_outputs(
        batch_dir=batch_dir,
        run_id=run_id,
        selected_scene_ids=scene_ids,
        registered_scene_ids=registered_scene_ids,
        scene_name_by_id=scene_name_by_id,
        failed_scene_errors=failed_scene_errors,
    )
    if not scene_ids:
        print("[D4SM-YOLO-INIT][BATCH] No pending scenes for this run.")
        return

    print("[D4SM-YOLO-INIT][BATCH] Config:")
    print(f"[D4SM-YOLO-INIT][BATCH] Output dir -> {batch_dir}")
    print(f"[D4SM-YOLO-INIT][BATCH] D4SM model size -> {runtime_config['model_size']}")
    print(f"[D4SM-YOLO-INIT][BATCH] D4SM checkpoint dir -> {runtime_config['checkpoint_dir']}")
    print(f"[D4SM-YOLO-INIT][BATCH] D4SM offload to CPU -> {runtime_config['offload_state_to_cpu']}")
    print(f"[D4SM-YOLO-INIT][BATCH] YOLO model -> {yolo_model_path}")
    print(f"[D4SM-YOLO-INIT][BATCH] YOLO conf={yolo_conf} iou={yolo_iou} imgsz={yolo_imgsz} device={yolo_device or 'auto'}")
    print(f"[D4SM-YOLO-INIT][BATCH] YOLO init min IoU -> {yolo_init_min_iou}")
    print("[D4SM-YOLO-INIT][BATCH] Loading models once...")
    shared_tracker = create_d4sm_tracker(runtime_config=runtime_config)
    yolo_provider = YoloInitMaskProvider(
        model_path=str(yolo_model_path),
        conf=float(yolo_conf),
        iou=float(yolo_iou),
        imgsz=int(yolo_imgsz),
        device=yolo_device,
    )
    print("[D4SM-YOLO-INIT][BATCH] Models ready.")

    try:
        for scene_id in scene_ids:
            scene_key = d4sm_batch.sanitize_name_for_path(str(scene_id))
            final_scene_dir = scenes_root / scene_key
            if d4sm_batch.scene_dir_is_complete(final_scene_dir):
                print(f"[D4SM-YOLO-INIT][BATCH] Skip completed scene -> {scene_id}")
                continue
            if final_scene_dir.exists():
                backup_dir = d4sm_batch.reserve_incomplete_scene_backup_dir(final_scene_dir)
                final_scene_dir.rename(backup_dir)
                print(f"[D4SM-YOLO-INIT][BATCH] Incomplete output moved -> {scene_id}: {backup_dir}")

            temp_scene_dir = scenes_root / f".tmp_{scene_key}"
            if temp_scene_dir.exists():
                shutil.rmtree(temp_scene_dir)

            scene_started_at = d4sm_batch._now_iso()
            scene_bundle = None
            try:
                scene_bundle = _build_scene_bundle(
                    scene_id=str(scene_id),
                    data_tar_root=data_tar_root,
                    annotations_tar_root=annotations_tar_root,
                    mask_variant=mask_variant,
                    image_subdir=image_subdir,
                )
                print(f"[D4SM-YOLO-INIT][BATCH] Scene start -> {scene_id}")
                print(f"[D4SM-YOLO-INIT][BATCH] Data tar -> {scene_bundle.data_tar_path}")
                print(f"[D4SM-YOLO-INIT][BATCH] Annotations tar -> {scene_bundle.annotations_tar_path}")
                results, scene_report = evaluate_scene_tar_yolo_init(
                    config_path=config_path,
                    scene_bundle=scene_bundle,
                    stable_min_frames=stable_min_frames,
                    max_frames=max_frames,
                    tracker=shared_tracker,
                    yolo_provider=yolo_provider,
                    yolo_init_min_iou=yolo_init_min_iou,
                )
                scene_name = str(scene_bundle.scene_id)
                scene_name_by_id[str(scene_id)] = scene_name
                failed_scene_errors.pop(str(scene_id), None)
                write_scene_outputs_yolo_init(
                    temp_scene_dir=temp_scene_dir,
                    final_scene_dir=final_scene_dir,
                    run_id=run_id,
                    scene_id=str(scene_id),
                    scene_name=scene_name,
                    results=results,
                    scene_report=scene_report,
                    scene_started_at=scene_started_at,
                    scene_finished_at=d4sm_batch._now_iso(),
                    stable_min_frames=stable_min_frames,
                    max_frames=max_frames,
                    model_size=model_size,
                )
                if final_scene_dir.exists():
                    raise RuntimeError(f"Final output already exists for {scene_id}: {final_scene_dir}")
                temp_scene_dir.rename(final_scene_dir)
                d4sm_batch.rebuild_batch_outputs(
                    batch_dir=batch_dir,
                    run_id=run_id,
                    selected_scene_ids=scene_ids,
                    registered_scene_ids=registered_scene_ids,
                    scene_name_by_id=scene_name_by_id,
                    failed_scene_errors=failed_scene_errors,
                )
                print(f"[D4SM-YOLO-INIT][BATCH] Scene completed -> {scene_id}")
            except Exception as exc:
                failed_scene_errors[str(scene_id)] = str(exc)
                if temp_scene_dir.exists():
                    shutil.rmtree(temp_scene_dir, ignore_errors=True)
                d4sm_batch.rebuild_batch_outputs(
                    batch_dir=batch_dir,
                    run_id=run_id,
                    selected_scene_ids=scene_ids,
                    registered_scene_ids=registered_scene_ids,
                    scene_name_by_id=scene_name_by_id,
                    failed_scene_errors=failed_scene_errors,
                )
                print(f"[D4SM-YOLO-INIT][BATCH][ERROR] Scene failed -> {scene_id}: {exc}")
            finally:
                if scene_bundle is not None:
                    try:
                        scene_bundle.close()
                    except Exception:
                        pass
    finally:
        shared_tracker = None
        yolo_provider = None
        gc.collect()
        release_cuda_scene_resources()

    write_text(batch_dir / "notes.txt", "D4SM YOLO-init with GT oracle for pending-object selection and recovery enabled.\n")
    print("[D4SM-YOLO-INIT][BATCH] Done.")


if __name__ == "__main__":
    main()