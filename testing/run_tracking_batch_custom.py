from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
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

from config.config_loader import Config
from pipeline.initialization import initialize_system
from pipeline.reid_pipeline import ReIDPipeline
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
    resolve_frame_files_for_testing,
)

PROJECT_DIR = Path(SRC_DIR).resolve().parent


# ---------------------------------------------------------------------------
# CLI argument resolution helpers
# ---------------------------------------------------------------------------

def _env_str(name: str, default: str = "") -> str:
    raw = os.environ.get(name, None)
    if raw is None:
        return str(default)
    return str(raw).strip()


def _resolve(cli_val: str | int | float | None, env_name: str, default: str) -> str:
    if cli_val is not None and str(cli_val).strip():
        return str(cli_val).strip()
    return _env_str(env_name, default) or default


def _resolve_optional(cli_val: str | None, env_name: str) -> str | None:
    if cli_val is not None and str(cli_val).strip():
        return str(cli_val).strip()
    v = _env_str(env_name, "")
    return v if v else None


def _resolve_int(cli_val: int | None, env_name: str, default: int | None) -> int | None:
    if cli_val is not None:
        return int(cli_val)
    raw = _env_str(env_name, "")
    if raw:
        return int(raw)
    return default


# ---------------------------------------------------------------------------
# Scene definition helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Evaluation entry-point (single scene)
# ---------------------------------------------------------------------------

def _evaluate_scene_custom(
    *,
    config_path: Path,
    override_config_path: Path | None,
    scene_def: dict[str, str],
    stable_min_frames: int,
    max_frames: int | None,
    force_detector_backend: str,
) -> tuple[dict[str, Any], str]:
    scene_id = str(scene_def["scene_id"])
    frames_dir = str(scene_def["frames_dir"])
    davis_meta_path = str(scene_def["davis_meta_path"])
    davis_annotations_dir = str(scene_def["davis_annotations_dir"])

    config = Config(
        default_config_path=str(config_path),
        override_config_path=str(override_config_path) if override_config_path else None,
    ).to_dict()
    config.setdefault("detector", {})["backend"] = str(force_detector_backend)
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

    total_frames = int(len(frame_files))
    progress_every = 20
    process = make_process_handle()

    print(
        f"[REMIND-CUSTOM][scene={scene_id}] "
        f"start | frames={total_frames}"
    )

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

    from utils.io import parse_frame_id as _parse_frame_id

    for idx, frame_path in enumerate(frame_files):
        loop_t0 = perf_counter()
        parsed_frame_id = _parse_frame_id(frame_path)
        if use_sequential_frame_ids:
            frame_id = int(idx)
        else:
            frame_id = int(idx) if parsed_frame_id is None else int(parsed_frame_id)
        rss_before = read_process_rss_bytes(process)

        read_t0 = perf_counter()
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not read frame: {frame_path}")
        read_ms = (perf_counter() - read_t0) * 1000.0
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
        should_log = (
            processed_frames == 1
            or processed_frames % progress_every == 0
            or processed_frames == total_frames
        )
        if should_log:
            avg_loop_ms = float(total_loop_ms / max(1, processed_frames))
            print(
                f"[REMIND-CUSTOM][scene={scene_id}] "
                f"progress {processed_frames}/{total_frames} "
                f"(frame_id={frame_id}) | "
                f"read={read_ms:.2f} ms | "
                f"pipeline={pipeline_ms:.2f} ms | "
                f"gt={gt_ms:.2f} ms | "
                f"eval={eval_ms:.2f} ms | "
                f"loop={loop_ms:.2f} ms | "
                f"avg_loop={avg_loop_ms:.2f} ms"
            )

    results = evaluator.finalize()
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
        "detector_mode": str(force_detector_backend),
    }
    results["timing_summary"] = timing_summary
    summary = results.setdefault("summary", {})
    summary.update(timing_summary)
    summary.update(
        {
            "input_mode": "custom_davis_dir",
            "frames_dir": frames_dir,
            "davis_meta_path": davis_meta_path,
            "davis_annotations_dir": davis_annotations_dir,
        }
    )

    for row in (results.get("per_frame", []) or []):
        fid = int(row.get("frame_id", -1))
        frame_timing = per_frame_timing_by_frame_id.get(fid, {})
        row["read_ms"] = float(frame_timing.get("read_ms", 0.0))
        row["pipeline_ms"] = float(frame_timing.get("pipeline_ms", 0.0))
        row["gt_ms"] = float(frame_timing.get("gt_ms", 0.0))
        row["eval_ms"] = float(frame_timing.get("eval_ms", 0.0))
        row["post_ms"] = float(frame_timing.get("post_ms", 0.0))
        row["loop_ms"] = float(frame_timing.get("loop_ms", 0.0))
        row.update(per_frame_runtime_memory_by_frame_id.get(fid, {}))

    summary.update(base_batch.build_memory_summary(results.get("per_frame", []) or []))
    report = base_batch.build_console_report(results)
    return results, report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Run REMIND tracker over custom DAVIS-style image directories. "
            "GT annotation masks are used as the detector input (same as "
            "the ScanNet++ benchmark scripts).  Supports single-scene mode "
            "via --frames-dir / --davis-meta-path / --davis-annotations-dir, "
            "or multi-scene mode via --scenes-def-file (JSON array)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    paths = p.add_argument_group("custom data paths")
    paths.add_argument("--dataset-root", metavar="DIR", help="Optional root used to resolve relative custom dataset paths. [env: REMIND_CUSTOM_DATASET_ROOT]")
    paths.add_argument("--frames-dir", metavar="DIR", help="Directory containing scene frames. Required in single-scene mode. [env: REMIND_CUSTOM_FRAMES_DIR]")
    paths.add_argument("--davis-meta-path", metavar="FILE", help="DAVIS metadata JSON. Required in single-scene mode. [env: REMIND_CUSTOM_DAVIS_META_PATH]")
    paths.add_argument("--davis-annotations-dir", metavar="DIR", help="Directory containing DAVIS GT masks. Required in single-scene mode. [env: REMIND_CUSTOM_DAVIS_ANNOTATIONS_DIR]")
    paths.add_argument("--scenes-def-file", metavar="FILE", help="JSON array with scene_id, frames_dir, davis_meta_path and davis_annotations_dir. [env: REMIND_CUSTOM_SCENES_DEF_FILE]")
    paths.add_argument("--config-path", metavar="FILE", help="Path to the YAML config file (default: src/config/default_config.yaml).")
    paths.add_argument("--override-config", metavar="FILE", help="Path to a YAML override config file (merged on top of --config-path).")

    batch = p.add_argument_group("batch control")
    batch.add_argument("--scene-id", metavar="ID", help="Scene ID for single-scene mode. [env: REMIND_SCENE_ID, default: basename of --frames-dir]")
    batch.add_argument("--output-dir", metavar="DIR", help="Root directory for batch results. [env: REMIND_CUSTOM_OUTPUT_DIR]")
    batch.add_argument("--run-id", metavar="NAME", help="Identifier for this run. [env: REMIND_CUSTOM_RUN_ID, default: remind_custom]")
    batch.add_argument("--max-scenes", type=int, metavar="N", help="Maximum number of scenes from --scenes-def-file. [env: REMIND_CUSTOM_MAX_SCENES]")
    batch.add_argument("--batch-size", type=int, metavar="N", help="Batch size/resume window. [env: REMIND_CUSTOM_BATCH_SIZE]")
    batch.add_argument("--stable-min-frames", type=int, metavar="N", help="Stable frames threshold. [env: REMIND_CUSTOM_STABLE_MIN_FRAMES, default: 3]")
    batch.add_argument("--max-frames", type=int, metavar="N", help="Maximum frames per scene. [env: REMIND_CUSTOM_MAX_FRAMES]")
    batch.add_argument("--detector-backend", metavar="NAME", help="Force detector backend name. [env: REMIND_CUSTOM_DETECTOR_BACKEND, default: davis]")
    return p


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    src_dir = Path(SRC_DIR).resolve()
    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else src_dir / "config" / "default_config.yaml"
    override_config_path: Path | None = None
    if args.override_config:
        override_config_path = Path(args.override_config).expanduser().resolve()

    scene_defs = _resolve_scene_defs(args)
    max_scenes = _resolve_int(args.max_scenes, "REMIND_CUSTOM_MAX_SCENES", None)
    if max_scenes is not None:
        scene_defs = scene_defs[: max(0, int(max_scenes))]
    scene_ids = [str(sd["scene_id"]) for sd in scene_defs]
    if not scene_ids:
        raise RuntimeError("No custom scenes were resolved.")

    run_id = _resolve(args.run_id, "REMIND_CUSTOM_RUN_ID", "remind_custom")
    output_root = Path(
        _resolve(
            args.output_dir,
            "REMIND_CUSTOM_OUTPUT_DIR",
            str(PROJECT_DIR / "outputs" / "tfm" / "testing_batch_custom"),
        )
    ).expanduser().resolve()
    batch_dir = output_root
    scenes_root = batch_dir / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)

    batch_size = _resolve_int(args.batch_size, "REMIND_CUSTOM_BATCH_SIZE", max_scenes)
    (
        scene_ids,
        registered_scene_ids,
        selection_mode,
        existing_manifest_rows,
        existing_per_scene_rows,
    ) = base_batch.resolve_scene_schedule(
        candidate_scene_ids=base_batch.unique_preserve_order(scene_ids),
        batch_dir=batch_dir,
        batch_size=batch_size,
    )

    stable_min_frames = _resolve_int(args.stable_min_frames, "REMIND_CUSTOM_STABLE_MIN_FRAMES", 3) or 3
    max_frames = _resolve_int(args.max_frames, "REMIND_CUSTOM_MAX_FRAMES", None)
    force_detector_backend = _resolve(args.detector_backend, "REMIND_CUSTOM_DETECTOR_BACKEND", "davis")

    scene_defs_by_id = {str(sd["scene_id"]): sd for sd in scene_defs}

    base_batch.write_single_row_csv(
        batch_dir / "run_config.csv",
        {
            "run_id": str(run_id),
            "batch_name": str(run_id),
            "batch_dir": str(batch_dir.resolve()),
            "created_at": base_batch._now_iso(),
            "tracker_family": "remind",
            "input_mode": "custom_davis_dir",
            "detector_backend": str(force_detector_backend),
            "config_path": str(config_path),
            "override_config_path": str(override_config_path) if override_config_path else None,
            "stable_min_frames": int(stable_min_frames),
            "max_frames": None if max_frames is None else int(max_frames),
            "max_scenes": None if max_scenes is None else int(max_scenes),
            "batch_size": None if batch_size is None else int(batch_size),
            "selection_mode": str(selection_mode),
            "selected_scene_count": int(len(scene_ids)),
            "selected_scene_ids": [str(sid) for sid in scene_ids],
            "registered_scene_count": int(len(registered_scene_ids)),
            "registered_scene_ids": [str(sid) for sid in registered_scene_ids],
        },
    )

    scene_name_by_id = base_batch.merge_scene_name_index(
        base_scene_name_by_id={str(sid): str(sid) for sid in registered_scene_ids},
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
        print("[REMIND-CUSTOM][BATCH] No pending scenes for this run.")
        return

    print("[REMIND-CUSTOM][BATCH] Config:")
    print(f"[REMIND-CUSTOM][BATCH]   config_path       -> {config_path}")
    if override_config_path:
        print(f"[REMIND-CUSTOM][BATCH]   override_config   -> {override_config_path}")
    print(f"[REMIND-CUSTOM][BATCH]   output_dir        -> {batch_dir}")
    print(f"[REMIND-CUSTOM][BATCH]   detector_backend  -> {force_detector_backend}")
    print(f"[REMIND-CUSTOM][BATCH]   scenes            -> {len(scene_ids)}")

    for scene_id in scene_ids:
        scene_def = scene_defs_by_id.get(str(scene_id))
        if scene_def is None:
            print(f"[REMIND-CUSTOM][BATCH][ERROR] Scene def not found -> {scene_id}")
            continue

        scene_key = base_batch.sanitize_name_for_path(str(scene_id))
        final_scene_dir = scenes_root / scene_key
        if base_batch.scene_dir_is_complete(final_scene_dir):
            print(f"[REMIND-CUSTOM][BATCH] Skip completed scene -> {scene_id}")
            continue
        if final_scene_dir.exists():
            incomplete_backup_dir = base_batch.reserve_incomplete_scene_backup_dir(final_scene_dir)
            final_scene_dir.rename(incomplete_backup_dir)
            print(f"[REMIND-CUSTOM][BATCH] Incomplete output moved -> {scene_id}: {incomplete_backup_dir}")

        temp_scene_dir = scenes_root / f".tmp_{scene_key}"
        if temp_scene_dir.exists():
            shutil.rmtree(temp_scene_dir)

        scene_started_at = base_batch._now_iso()
        try:
            print(f"[REMIND-CUSTOM][BATCH] Scene start -> {scene_id}")
            print(f"[REMIND-CUSTOM][BATCH]   frames_dir        -> {scene_def['frames_dir']}")
            print(f"[REMIND-CUSTOM][BATCH]   davis_meta         -> {scene_def['davis_meta_path']}")
            print(f"[REMIND-CUSTOM][BATCH]   davis_annotations  -> {scene_def['davis_annotations_dir']}")
            results, scene_report = _evaluate_scene_custom(
                config_path=config_path,
                override_config_path=override_config_path,
                scene_def=scene_def,
                stable_min_frames=stable_min_frames,
                max_frames=max_frames,
                force_detector_backend=force_detector_backend,
            )
            scene_name = str(scene_id)
            scene_name_by_id[str(scene_id)] = scene_name
            failed_scene_errors.pop(str(scene_id), None)
            base_batch.write_scene_outputs(
                temp_scene_dir=temp_scene_dir,
                final_scene_dir=final_scene_dir,
                run_id=run_id,
                scene_id=str(scene_id),
                scene_name=scene_name,
                results=results,
                scene_report=scene_report,
                scene_started_at=scene_started_at,
                scene_finished_at=base_batch._now_iso(),
                stable_min_frames=stable_min_frames,
                max_frames=max_frames,
                force_detector_backend=force_detector_backend,
            )
            if final_scene_dir.exists():
                raise RuntimeError(f"Final output already exists for {scene_id}: {final_scene_dir}")
            temp_scene_dir.rename(final_scene_dir)
            base_batch.rebuild_batch_outputs(
                batch_dir=batch_dir,
                run_id=run_id,
                selected_scene_ids=scene_ids,
                registered_scene_ids=registered_scene_ids,
                scene_name_by_id=scene_name_by_id,
                failed_scene_errors=failed_scene_errors,
            )
            print(f"[REMIND-CUSTOM][BATCH] Scene completed -> {scene_id}")
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
            print(f"[REMIND-CUSTOM][BATCH][ERROR] Scene failed -> {scene_id}: {exc}")
            continue

    print("[REMIND-CUSTOM][BATCH] Done.")


if __name__ == "__main__":
    main()
