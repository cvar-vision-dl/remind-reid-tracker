from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import sys
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
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.config_loader import Config
from testing.common.generic_tracking_metrics import TrackingOnlyEvaluator
from testing.common.generic_tracking_reporting import build_generic_console_report, write_text
from testing.d4sm import run_tracking_batch as d4sm_batch
from testing.d4sm.run_tracking_batch_tar import (
    PROJECT_DIR,
    YoloInitMaskProvider,
    _build_detections,
    _cuda_autocast_context,
    _env_str,
    _gt_object_to_full_mask,
    _inference_context,
    _resolve,
    _resolve_int,
    _resolve_optional,
    _select_yolo_init_regions,
    _sync_cuda,
    build_runtime_memory_telemetry,
    capture_cuda_memory_stats,
    create_d4sm_tracker,
    make_process_handle,
    read_process_rss_bytes,
    release_cuda_scene_resources,
    reset_cuda_peak_memory_stats,
    reset_d4sm_tracker_scene_state,
    resolve_d4sm_runtime_config,
    write_scene_outputs_yolo_init,
)
from testing.d4sm.run_tracking_test import resolve_frame_files_for_testing
from testing.davis_gt import DavisGroundTruthLoader
from utils.io import parse_frame_id


def _read_scene_defs(path: str | Path) -> list[dict[str, str]]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Scenes definition file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Scenes definition file must be a JSON array: {p}")
    out: list[dict[str, str]] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {idx} in scenes definition file is not an object.")
        scene_id = str(item.get("scene_id", "") or "").strip()
        frames_dir = str(item.get("frames_dir", "") or "").strip()
        davis_meta_path = str(item.get("davis_meta_path", "") or "").strip()
        davis_annotations_dir = str(item.get("davis_annotations_dir", "") or "").strip()
        if not scene_id:
            raise ValueError(f"Entry {idx} is missing scene_id.")
        out.append(
            {
                "scene_id": scene_id,
                "frames_dir": frames_dir,
                "davis_meta_path": davis_meta_path,
                "davis_annotations_dir": davis_annotations_dir,
            }
        )
    return out


def _build_single_scene_def(args: argparse.Namespace) -> dict[str, str]:
    dataset_root_raw = _resolve_optional(args.dataset_root, "REMIND_CUSTOM_DATASET_ROOT")
    dataset_root = Path(dataset_root_raw).expanduser().resolve() if dataset_root_raw else None

    def resolve_custom_path(value: Any, env_name: str) -> str:
        raw = _resolve_optional(value, env_name)
        if not raw:
            return ""
        path = Path(raw).expanduser()
        if not path.is_absolute() and dataset_root is not None:
            path = dataset_root / path
        return str(path.resolve())

    frames_dir = resolve_custom_path(args.frames_dir, "REMIND_CUSTOM_FRAMES_DIR")
    davis_meta_path = resolve_custom_path(args.davis_meta_path, "REMIND_CUSTOM_DAVIS_META_PATH")
    davis_annotations_dir = resolve_custom_path(args.davis_annotations_dir, "REMIND_CUSTOM_DAVIS_ANNOTATIONS_DIR")
    scene_id = _resolve_optional(args.scene_id, "REMIND_SCENE_ID") or ""
    if not scene_id and frames_dir:
        scene_id = Path(frames_dir).name
    return {
        "scene_id": str(scene_id),
        "frames_dir": str(frames_dir),
        "davis_meta_path": str(davis_meta_path),
        "davis_annotations_dir": str(davis_annotations_dir),
    }


def _validate_scene_def(scene_def: dict[str, str]) -> dict[str, str]:
    scene_id = str(scene_def.get("scene_id", "") or "").strip()
    frames_dir = Path(str(scene_def.get("frames_dir", "") or "")).expanduser().resolve()
    davis_meta_path = Path(str(scene_def.get("davis_meta_path", "") or "")).expanduser().resolve()
    davis_annotations_dir = Path(str(scene_def.get("davis_annotations_dir", "") or "")).expanduser().resolve()
    issues: list[str] = []
    if not scene_id:
        issues.append("missing_scene_id")
    if not frames_dir.is_dir():
        issues.append(f"missing_frames_dir:{frames_dir}")
    if not davis_meta_path.is_file():
        issues.append(f"missing_meta:{davis_meta_path}")
    if not davis_annotations_dir.is_dir():
        issues.append(f"missing_annotations_dir:{davis_annotations_dir}")
    if issues:
        raise FileNotFoundError(f"Custom scene '{scene_id or '-'}' has invalid inputs: {', '.join(issues)}")
    return {
        "scene_id": scene_id,
        "frames_dir": str(frames_dir),
        "davis_meta_path": str(davis_meta_path),
        "davis_annotations_dir": str(davis_annotations_dir),
    }


def _resolve_scene_defs(args: argparse.Namespace) -> list[dict[str, str]]:
    scenes_def_file = _resolve_optional(args.scenes_def_file, "REMIND_CUSTOM_SCENES_DEF_FILE")
    if scenes_def_file:
        return [_validate_scene_def(scene_def) for scene_def in _read_scene_defs(scenes_def_file)]
    return [_validate_scene_def(_build_single_scene_def(args))]


def evaluate_scene_custom_yolo_init(
    *,
    config_path: Path,
    scene_def: dict[str, str],
    stable_min_frames: int,
    max_frames: int | None,
    tracker: Any | None,
    yolo_provider: YoloInitMaskProvider,
    yolo_init_min_iou: float,
) -> tuple[dict[str, Any], str]:
    if torch is None:
        raise RuntimeError("d4sm requires torch, but the current interpreter cannot import it.")

    scene_id = str(scene_def["scene_id"])
    frames_dir = str(scene_def["frames_dir"])
    davis_meta_path = str(scene_def["davis_meta_path"])
    davis_annotations_dir = str(scene_def["davis_annotations_dir"])

    config = Config(default_config_path=str(config_path)).to_dict()
    config.setdefault("input", {})["frames_dir"] = frames_dir
    davis_cfg = config.setdefault("davis", {})
    davis_cfg["sequence_name"] = scene_id
    davis_cfg["meta_path"] = davis_meta_path
    davis_cfg["annotations_dir"] = davis_annotations_dir
    timing_cfg = config.setdefault("timing", {})
    timing_cfg["enabled"] = False
    timing_cfg["table"] = False
    timing_cfg["detail_keys"] = []

    frame_files, use_sequential_frame_ids = resolve_frame_files_for_testing(
        frames_dir,
        davis_meta_path=davis_meta_path,
    )
    if max_frames is not None:
        frame_files = frame_files[: max(0, int(max_frames))]
    if not frame_files:
        raise RuntimeError(f"No images found in {frames_dir}")

    print(
        f"[D4SM-YOLO-INIT-CUSTOM][scene={scene_id}] "
        f"start | frames={len(frame_files)} | recovery=on"
    )

    tracker_owner = tracker is None
    if tracker_owner:
        tracker = create_d4sm_tracker()
    else:
        reset_d4sm_tracker_scene_state(tracker)
    tracker.n_frames = int(len(frame_files))

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
    try:
        gt_loader = DavisGroundTruthLoader(config)
        evaluator = TrackingOnlyEvaluator(stable_min_frames=stable_min_frames)

        for idx, frame_path in enumerate(frame_files):
            loop_t0 = perf_counter()
            parsed_frame_id = parse_frame_id(frame_path)
            if use_sequential_frame_ids:
                frame_id = int(idx)
            else:
                frame_id = int(idx) if parsed_frame_id is None else int(parsed_frame_id)
            rss_before = read_process_rss_bytes(process)

            read_t0 = perf_counter()
            image = Image.open(frame_path).convert("RGB")
            frame_rgb = np.array(image)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
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
                    item["frame_path"] = str(frame_path)
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
            if processed_frames == 1 or processed_frames % 20 == 0 or processed_frames == len(frame_files):
                print(
                    f"[D4SM-YOLO-INIT-CUSTOM][scene={scene_id}] "
                    f"progress {processed_frames}/{len(frame_files)} | frame_id={frame_id} | "
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
        release_cuda_scene_resources()

    n_processed_frames = int(len(frame_files))
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
            "input_mode": "custom_davis_dir",
            "frames_dir": frames_dir,
            "davis_meta_path": davis_meta_path,
            "davis_annotations_dir": davis_annotations_dir,
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


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run D4SM over a custom DAVIS-style image directory using YOLO masks "
            "for object initialization. GT is used only to decide which visible "
            "objects are still pending and to select the same-class YOLO mask "
            "with maximum IoU. Recovery is enabled: missed objects are retried "
            "on later visible frames."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    paths = p.add_argument_group("custom data paths")
    paths.add_argument("--dataset-root", metavar="DIR", help="Optional root used to resolve relative custom dataset paths. [env: REMIND_CUSTOM_DATASET_ROOT]")
    paths.add_argument("--frames-dir", metavar="DIR", help="Directory containing scene frames. Required in single-scene mode. [env: REMIND_CUSTOM_FRAMES_DIR]")
    paths.add_argument("--davis-meta-path", metavar="FILE", help="DAVIS metadata JSON. Required in single-scene mode. [env: REMIND_CUSTOM_DAVIS_META_PATH]")
    paths.add_argument("--davis-annotations-dir", metavar="DIR", help="Directory containing DAVIS GT masks. Required in single-scene mode. [env: REMIND_CUSTOM_DAVIS_ANNOTATIONS_DIR]")
    paths.add_argument("--scenes-def-file", metavar="FILE", help="JSON array with scene_id, frames_dir, davis_meta_path and davis_annotations_dir. [env: REMIND_CUSTOM_SCENES_DEF_FILE]")
    paths.add_argument("--config-path", metavar="FILE", help="Path to the YAML config file (default: REMIND/config/default_config.yaml).")

    batch = p.add_argument_group("batch control")
    batch.add_argument("--scene-id", metavar="ID", help="Scene ID for single-scene mode. [env: REMIND_SCENE_ID, default: basename of --frames-dir]")
    batch.add_argument("--output-dir", metavar="DIR", help="Root directory for batch results. [env: REMIND_D4SM_YOLO_INIT_CUSTOM_OUTPUT_DIR]")
    batch.add_argument("--run-id", metavar="NAME", help="Identifier for this run. [env: REMIND_D4SM_YOLO_INIT_CUSTOM_RUN_ID]")
    batch.add_argument("--max-scenes", type=int, metavar="N", help="Maximum number of scenes from --scenes-def-file. [env: REMIND_D4SM_YOLO_INIT_CUSTOM_MAX_SCENES]")
    batch.add_argument("--batch-size", type=int, metavar="N", help="Batch size/resume window. [env: REMIND_D4SM_YOLO_INIT_CUSTOM_BATCH_SIZE]")
    batch.add_argument("--stable-min-frames", type=int, metavar="N", help="Stable frames threshold. [env: REMIND_D4SM_YOLO_INIT_STABLE_MIN_FRAMES, default: 3]")
    batch.add_argument("--max-frames", type=int, metavar="N", help="Maximum frames per scene. [env: REMIND_D4SM_YOLO_INIT_MAX_FRAMES]")

    d4sm = p.add_argument_group("D4SM tracker")
    d4sm.add_argument(
        "--checkpoints-dir", metavar="DIR",
        help="Directory containing D4SM / SAM2 checkpoint files (e.g. sam2.1_hiera_large.pt). [env: REMIND_D4SM_CHECKPOINT_DIR]",
    )
    d4sm.add_argument(
        "--offload-state-to-cpu", action="store_true", default=False,
        help="Offload D4SM internal state to CPU between frames to reduce GPU memory. [env: REMIND_D4SM_OFFLOAD_STATE_TO_CPU]",
    )
    d4sm.add_argument(
        "--model-size", metavar="SIZE",
        help="D4SM / SAM2 model size, e.g. large, base_plus, small, tiny. [env: REMIND_D4SM_MODEL_SIZE, default: large]",
    )

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
    scene_defs = _resolve_scene_defs(args)
    max_scenes = _resolve_int(args.max_scenes, "REMIND_D4SM_YOLO_INIT_CUSTOM_MAX_SCENES", None)
    if max_scenes is not None:
        scene_defs = scene_defs[: max(0, int(max_scenes))]
    scene_ids = [str(scene_def["scene_id"]) for scene_def in scene_defs]
    if not scene_ids:
        raise RuntimeError("No custom scenes were resolved.")

    yolo_model_path = _resolve_optional(args.yolo_model, "REMIND_YOLO_MODEL_PATH")
    if not yolo_model_path:
        raise ValueError("--yolo-model or REMIND_YOLO_MODEL_PATH is required.")
    yolo_conf = float(_resolve(args.yolo_conf, "REMIND_YOLO_CONF", "0.25"))
    yolo_iou = float(_resolve(args.yolo_iou, "REMIND_YOLO_IOU", "0.7"))
    yolo_imgsz = int(_resolve(args.yolo_imgsz, "REMIND_YOLO_IMGSZ", "640"))
    yolo_device = _resolve_optional(args.yolo_device, "REMIND_YOLO_DEVICE")
    yolo_init_min_iou = float(_resolve(args.yolo_init_min_iou, "REMIND_D4SM_YOLO_INIT_MIN_IOU", "0.1"))

    run_id = _resolve(args.run_id, "REMIND_D4SM_YOLO_INIT_CUSTOM_RUN_ID", "d4sm_yolo_init_custom")
    output_root = Path(
        _resolve(
            args.output_dir,
            "REMIND_D4SM_YOLO_INIT_CUSTOM_OUTPUT_DIR",
            str(PROJECT_DIR / "outputs" / "d4sm" / "testing_batch_custom_yolo_init"),
        )
    ).expanduser().resolve()
    batch_dir = output_root
    scenes_root = batch_dir / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)

    batch_size = _resolve_int(args.batch_size, "REMIND_D4SM_YOLO_INIT_CUSTOM_BATCH_SIZE", max_scenes)
    scene_ids, registered_scene_ids, selection_mode, existing_manifest_rows, existing_per_scene_rows = d4sm_batch.resolve_scene_schedule(
        candidate_scene_ids=d4sm_batch.unique_preserve_order(scene_ids),
        batch_dir=batch_dir,
        batch_size=batch_size,
    )
    scene_defs_by_id = {str(scene_def["scene_id"]): scene_def for scene_def in scene_defs}
    stable_min_frames = _resolve_int(args.stable_min_frames, "REMIND_D4SM_YOLO_INIT_STABLE_MIN_FRAMES", 3) or 3
    max_frames = _resolve_int(args.max_frames, "REMIND_D4SM_YOLO_INIT_MAX_FRAMES", None)

    # ---- D4SM runtime config with CLI overrides ----
    # Resolve --model-size from CLI / env (before resolve_d4sm_runtime_config which
    # reads REMIND_D4SM_MODEL_SIZE internally).  We set the env var so the downstream
    # helper picks it up transparently.
    model_size_cli = _resolve_optional(args.model_size, "REMIND_D4SM_MODEL_SIZE")
    if model_size_cli:
        os.environ["REMIND_D4SM_MODEL_SIZE"] = model_size_cli
    model_size = _env_str("REMIND_D4SM_MODEL_SIZE", "large").lower() or "large"

    runtime_config = resolve_d4sm_runtime_config()

    # --checkpoints-dir override
    checkpoints_dir_raw = _resolve_optional(args.checkpoints_dir, "REMIND_D4SM_CHECKPOINT_DIR")
    if checkpoints_dir_raw:
        runtime_config["checkpoint_dir"] = str(Path(checkpoints_dir_raw).expanduser().resolve())

    # --offload-state-to-cpu
    offload_state_to_cpu = bool(
        args.offload_state_to_cpu
        or os.environ.get("REMIND_D4SM_OFFLOAD_STATE_TO_CPU", "").strip().lower() in ("1", "true", "yes")
    )
    runtime_config["offload_state_to_cpu"] = offload_state_to_cpu

    d4sm_batch.write_single_row_csv(
        batch_dir / "run_config.csv",
        {
            "run_id": str(run_id),
            "batch_name": str(run_id),
            "batch_dir": str(batch_dir.resolve()),
            "created_at": d4sm_batch._now_iso(),
            "tracker_family": "d4sm",
            "input_mode": "custom_davis_dir",
            "init_source": "yolo",
            "init_recovery_enabled": True,
            "model_size": str(model_size),
            "checkpoint_dir": str(runtime_config["checkpoint_dir"]),
            "offload_state_to_cpu": bool(offload_state_to_cpu),
            "stable_min_frames": int(stable_min_frames),
            "max_frames": None if max_frames is None else int(max_frames),
            "max_scenes": None if max_scenes is None else int(max_scenes),
            "batch_size": None if batch_size is None else int(batch_size),
            "selection_mode": str(selection_mode),
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
        print("[D4SM-YOLO-INIT-CUSTOM][BATCH] No pending scenes for this run.")
        return

    print("[D4SM-YOLO-INIT-CUSTOM][BATCH] Config:")
    print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] Output dir -> {batch_dir}")
    print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] D4SM model size -> {runtime_config['model_size']}")
    print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] D4SM checkpoint dir -> {runtime_config['checkpoint_dir']}")
    print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] D4SM offload state to CPU -> {offload_state_to_cpu}")
    print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] YOLO model -> {yolo_model_path}")
    print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] YOLO conf={yolo_conf} iou={yolo_iou} imgsz={yolo_imgsz} device={yolo_device or 'auto'}")
    print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] YOLO init min IoU -> {yolo_init_min_iou}")
    print("[D4SM-YOLO-INIT-CUSTOM][BATCH] Loading models once...")
    shared_tracker = create_d4sm_tracker(runtime_config=runtime_config)
    yolo_provider = YoloInitMaskProvider(
        model_path=str(yolo_model_path),
        conf=float(yolo_conf),
        iou=float(yolo_iou),
        imgsz=int(yolo_imgsz),
        device=yolo_device,
    )
    print("[D4SM-YOLO-INIT-CUSTOM][BATCH] Models ready.")

    try:
        for scene_id in scene_ids:
            scene_def = scene_defs_by_id[str(scene_id)]
            scene_key = d4sm_batch.sanitize_name_for_path(str(scene_id))
            final_scene_dir = scenes_root / scene_key
            if d4sm_batch.scene_dir_is_complete(final_scene_dir):
                print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] Skip completed scene -> {scene_id}")
                continue
            if final_scene_dir.exists():
                backup_dir = d4sm_batch.reserve_incomplete_scene_backup_dir(final_scene_dir)
                final_scene_dir.rename(backup_dir)
                print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] Incomplete output moved -> {scene_id}: {backup_dir}")

            temp_scene_dir = scenes_root / f".tmp_{scene_key}"
            if temp_scene_dir.exists():
                shutil.rmtree(temp_scene_dir)

            scene_started_at = d4sm_batch._now_iso()
            try:
                print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] Scene start -> {scene_id}")
                print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] Frames dir -> {scene_def['frames_dir']}")
                print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] DAVIS meta -> {scene_def['davis_meta_path']}")
                print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] DAVIS annotations -> {scene_def['davis_annotations_dir']}")
                results, scene_report = evaluate_scene_custom_yolo_init(
                    config_path=config_path,
                    scene_def=scene_def,
                    stable_min_frames=stable_min_frames,
                    max_frames=max_frames,
                    tracker=shared_tracker,
                    yolo_provider=yolo_provider,
                    yolo_init_min_iou=yolo_init_min_iou,
                )
                scene_name = str(scene_id)
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
                print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH] Scene completed -> {scene_id}")
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
                print(f"[D4SM-YOLO-INIT-CUSTOM][BATCH][ERROR] Scene failed -> {scene_id}: {exc}")
    finally:
        shared_tracker = None
        yolo_provider = None
        gc.collect()
        release_cuda_scene_resources()

    write_text(batch_dir / "notes.txt", "D4SM custom DAVIS YOLO-init with GT oracle for pending-object selection and recovery enabled.\n")
    print("[D4SM-YOLO-INIT-CUSTOM][BATCH] Done.")


if __name__ == "__main__":
    main()