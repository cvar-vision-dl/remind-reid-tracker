from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config.config_loader import Config
from pipeline.initialization import initialize_system
from pipeline.reid_pipeline import ReIDPipeline
from davis_gt import DavisGroundTruthLoader
from reporting import build_console_report, write_csv, write_json, write_text
from tracking_metrics import TrackingEvaluator, build_memory_summary
from utils.io import list_image_files, read_bgr
from utils.io import parse_frame_id
from utils.logging import default_run_artifact_dir
from utils.scannetpp_tar import resolve_prepared_scene_from_tar


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


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


def cuda_memory_enabled() -> bool:
    try:
        return bool(torch is not None and torch.cuda.is_available())
    except Exception:
        return False


def reset_cuda_peak_memory_stats() -> None:
    if not cuda_memory_enabled():
        return
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def capture_cuda_memory_stats() -> dict[str, int | None]:
    if not cuda_memory_enabled():
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
    telemetry: dict[str, int | None] = {
        "mem_process_rss_before_bytes": None if rss_before is None else int(rss_before),
        "mem_process_rss_after_read_bytes": None if rss_after_read is None else int(rss_after_read),
        "mem_process_rss_after_pipeline_bytes": None if rss_after_pipeline is None else int(rss_after_pipeline),
        "mem_process_rss_after_eval_bytes": None if rss_after_eval is None else int(rss_after_eval),
        "mem_process_rss_peak_approx_bytes": None if not rss_samples else int(max(rss_samples)),
        "mem_process_rss_delta_bytes": (
            None if rss_before is None or rss_after_eval is None else int(rss_after_eval - rss_before)
        ),
    }
    pipeline_gpu = dict(gpu_after_pipeline or {})
    eval_gpu = dict(gpu_after_eval or {})
    telemetry["mem_gpu_allocated_after_pipeline_bytes"] = (
        None if pipeline_gpu.get("mem_gpu_allocated_bytes", None) is None else int(pipeline_gpu["mem_gpu_allocated_bytes"])
    )
    telemetry["mem_gpu_reserved_after_pipeline_bytes"] = (
        None if pipeline_gpu.get("mem_gpu_reserved_bytes", None) is None else int(pipeline_gpu["mem_gpu_reserved_bytes"])
    )
    telemetry["mem_gpu_allocated_after_eval_bytes"] = (
        None if eval_gpu.get("mem_gpu_allocated_bytes", None) is None else int(eval_gpu["mem_gpu_allocated_bytes"])
    )
    telemetry["mem_gpu_reserved_after_eval_bytes"] = (
        None if eval_gpu.get("mem_gpu_reserved_bytes", None) is None else int(eval_gpu["mem_gpu_reserved_bytes"])
    )
    telemetry["mem_gpu_peak_allocated_bytes"] = (
        None if eval_gpu.get("mem_gpu_peak_allocated_bytes", None) is None else int(eval_gpu["mem_gpu_peak_allocated_bytes"])
    )
    telemetry["mem_gpu_peak_reserved_bytes"] = (
        None if eval_gpu.get("mem_gpu_peak_reserved_bytes", None) is None else int(eval_gpu["mem_gpu_peak_reserved_bytes"])
    )
    return telemetry


def resolve_testing_input_source(
    project_dir: str,
    *,
    frames_dir: str = "",
    davis_meta_path: str = "",
    davis_annotations_dir: str = "",
    sequence_name: str = "",
    masks_root: str = "",
    images_root: str = "",
    scene_id: str = "00a231a370",
    mask_variant: str = "benchmark",
    image_subdir: str = "dslr/resized_images",
    masks_subdir: str = "2Dmasks",
    prefer_external: bool = True,
) -> dict[str, str]:
    project_path = Path(project_dir).resolve()

    if frames_dir:
        frames_path = Path(frames_dir).expanduser().resolve()
        return {
            "mode": "explicit_env",
            "frames_dir": str(frames_path),
            "sequence_name": sequence_name or frames_path.name,
            "davis_meta_path": davis_meta_path,
            "davis_annotations_dir": davis_annotations_dir,
            "image_subdir": "",
        }

    _default_data_root = "/media/pablo/LINUX/Qsync/2026_tracker_reid/datasets/scannetpp_data"
    external_masks_root_base = Path(masks_root or _default_data_root).expanduser().resolve()
    external_images_root_base = Path(images_root or _default_data_root).expanduser().resolve()
    scene_id = scene_id or "00a231a370"
    mask_variant = (mask_variant or "benchmark").strip().lower() or "benchmark"
    image_subdir = image_subdir or "dslr/resized_images"
    masks_subdir = masks_subdir or "2Dmasks"

    if mask_variant == "benchmark":
        mask_variant = "benchmark_instance"

    external_masks_root = (external_masks_root_base / masks_subdir / scene_id).resolve()
    external_meta_path = (external_masks_root / f"meta_{mask_variant}.json").resolve()
    external_annotations_dir_path = (external_masks_root / "annotations" / mask_variant).resolve()
    external_frames_dir = (external_images_root_base / "data" / scene_id / image_subdir).resolve()

    external_ready = (
        external_frames_dir.is_dir()
        and external_meta_path.is_file()
        and external_annotations_dir_path.is_dir()
    )
    if prefer_external and external_ready:
        return {
            "mode": "external_scannetpp",
            "frames_dir": str(external_frames_dir),
            "sequence_name": scene_id,
            "davis_meta_path": str(external_meta_path),
            "davis_annotations_dir": str(external_annotations_dir_path),
            "image_subdir": str(image_subdir),
        }

    if prefer_external:
        tar_source = resolve_prepared_scene_from_tar(
            project_dir=project_dir,
            images_root_base=external_images_root_base,
            scene_id=scene_id,
            mask_variant=mask_variant,
            image_subdir=image_subdir,
        )
        if tar_source is not None:
            return tar_source

    local_frames_dir = (project_path / "data" / "framesCOMPLETO1").resolve()
    return {
        "mode": "local_fallback",
        "frames_dir": str(local_frames_dir),
        "sequence_name": local_frames_dir.name,
        "davis_meta_path": "",
        "davis_annotations_dir": "",
        "image_subdir": "",
    }


def resolve_frame_files_for_testing(frames_dir: str, *, davis_meta_path: str = "") -> tuple[list[str], bool]:
    """
    Resuelve la lista de frames y si el frame_id debe ser secuencial.
    """
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
                    f"Faltan {len(missing)} frames del meta en {frames_root}. "
                    f"Ejemplos: {preview}"
                )

            if out:
                return out, True

    return list_image_files(str(frames_root)), False


def build_det_to_object_id(update_output) -> dict[int, int]:
    out: dict[int, int] = {}
    for row in getattr(update_output, "matches", []) or []:
        if "det_id" in row and "object_id" in row:
            out[int(row["det_id"])] = int(row["object_id"])
    for row in getattr(update_output, "created", []) or []:
        if "det_id" in row and "object_id" in row:
            out[int(row["det_id"])] = int(row["object_id"])
    return out


def resolve_aligned_shape(perception_output) -> tuple[int, int]:
    transforms = getattr(perception_output, "transforms", {}) or {}
    aligned_shape = transforms.get("aligned_shape", None)
    if isinstance(aligned_shape, (tuple, list)) and len(aligned_shape) == 2:
        return (int(aligned_shape[0]), int(aligned_shape[1]))

    frame_features = getattr(perception_output, "frame_features", {}) or {}
    frame_shape = frame_features.get("frame_shape", None)
    if isinstance(frame_shape, (tuple, list)) and len(frame_shape) == 2:
        return (int(frame_shape[0]), int(frame_shape[1]))

    aligned = (getattr(perception_output, "debug", {}) or {}).get("frame_aligned_bgr", None)
    if aligned is not None and getattr(aligned, "shape", None) is not None and len(aligned.shape) >= 2:
        return (int(aligned.shape[0]), int(aligned.shape[1]))

    raise RuntimeError(
        "Testing runner esperaba aligned_shape en p_out.transforms, "
        "o frame_shape en p_out.frame_features."
    )


def build_testing_file_names() -> list[str]:
    return [
        "tracking_eval.json",
        "tracking_eval_per_object.csv",
        "tracking_eval_per_case.csv",
        "tracking_eval_per_frame.csv",
        "tracking_eval_per_pred_track.csv",
        "tracking_eval_report.txt",
    ]


def sanitize_name_for_path(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown_scene"

    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "unknown_scene"


def choose_unique_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir

    suffix = 1
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def format_testing_timestamp_from_paths(paths: list[Path]) -> str:
    if not paths:
        raise ValueError("Se esperaba al menos un path para calcular el timestamp del run.")

    latest_mtime = max(float(path.stat().st_mtime) for path in paths)
    return datetime.fromtimestamp(latest_mtime).strftime("%Y%m%d_%H%M%S")


def migrate_legacy_testing_outputs(testing_root: Path, scene_tag: str | None = None) -> Path | None:
    legacy_files = [testing_root / name for name in build_testing_file_names() if (testing_root / name).exists()]
    if not legacy_files:
        return None

    legacy_ts = format_testing_timestamp_from_paths(legacy_files)
    legacy_prefix = "tracking_eval_legacy"
    if scene_tag:
        legacy_prefix = f"{legacy_prefix}_{sanitize_name_for_path(scene_tag)}"
    legacy_dir = Path(
        default_run_artifact_dir(
            str(testing_root.parent),
            group="testing",
            prefix=legacy_prefix,
            timestamp=legacy_ts,
        )
    )
    legacy_dir = choose_unique_dir(legacy_dir)
    legacy_dir.mkdir(parents=True, exist_ok=True)

    for path in legacy_files:
        path.rename(legacy_dir / path.name)

    return legacy_dir


def build_event_rows(*, run_id: str, scene_id: str, scene_name: str, results: dict) -> list[dict]:
    out: list[dict] = []
    events = dict(results.get("events", {}) or {})
    for ev in (events.get("swap", []) or []):
        out.append(
            {
                "run_id": str(run_id),
                "scene_id": str(scene_id),
                "scene_name": str(scene_name),
                "frame_id": int(ev.get("frame_id", -1)),
                "event_type": "swap",
                "gt_a": ev.get("gt_a", None),
                "gt_b": ev.get("gt_b", None),
                "pred_id_main": ev.get("pred_a_prev", None),
                "pred_id_aux": ev.get("pred_b_prev", None),
                "class_name": None,
                "detail": f"{ev.get('pred_a_prev')}<->{ev.get('pred_b_prev')}",
            }
        )
    for ev in (events.get("theft_with_new_id", []) or []):
        out.append(
            {
                "run_id": str(run_id),
                "scene_id": str(scene_id),
                "scene_name": str(scene_name),
                "frame_id": int(ev.get("frame_id", -1)),
                "event_type": "theft_with_new_id",
                "gt_a": ev.get("thief_gt", None),
                "gt_b": ev.get("victim_gt", None),
                "pred_id_main": ev.get("stolen_pred_id", None),
                "pred_id_aux": ev.get("victim_new_pred_id", None),
                "class_name": None,
                "detail": f"stolen={ev.get('stolen_pred_id')} victim_new={ev.get('victim_new_pred_id')}",
            }
        )
    for ev in (events.get("theft_with_displacement", []) or []):
        out.append(
            {
                "run_id": str(run_id),
                "scene_id": str(scene_id),
                "scene_name": str(scene_name),
                "frame_id": int(ev.get("frame_id", -1)),
                "event_type": "theft_with_displacement",
                "gt_a": ev.get("thief_gt", None),
                "gt_b": ev.get("victim_gt", None),
                "pred_id_main": ev.get("stolen_pred_id", None),
                "pred_id_aux": ev.get("victim_pred_id_after", None),
                "class_name": None,
                "detail": f"stolen={ev.get('stolen_pred_id')} victim_after={ev.get('victim_pred_id_after')}",
            }
        )
    for row in (results.get("per_case", []) or []):
        if str(row.get("real_state")) == "existing" and str(row.get("collapsed_kind")) == "new":
            out.append(
                {
                    "run_id": str(run_id),
                    "scene_id": str(scene_id),
                    "scene_name": str(scene_name),
                    "frame_id": int(row.get("frame_id", -1)),
                    "event_type": "reopen_existing_as_new",
                    "gt_a": int(row.get("gt_instance_id", -1)),
                    "gt_b": None,
                    "pred_id_main": row.get("collapsed_pred_object_id", None),
                    "pred_id_aux": None,
                    "class_name": row.get("gt_class_name", None),
                    "detail": str(row.get("final_reason", "") or "reopened_as_new"),
                }
            )
    return out


def main():
    parser = argparse.ArgumentParser(description="Run tracking evaluation on a single scene.")
    parser.add_argument("--frames-dir", default="", help="Explicit frames directory (env: INPUT_FRAMES_DIR)")
    parser.add_argument("--sequence-name", default="", help="Sequence name override (env: DAVIS_SEQUENCE_NAME)")
    parser.add_argument("--davis-meta-path", default="", help="DAVIS meta JSON path (env: DAVIS_META_PATH)")
    parser.add_argument("--davis-annotations-dir", default="", help="DAVIS annotations directory (env: DAVIS_ANNOTATIONS_DIR)")
    parser.add_argument("--scene-id", default="", help="ScanNet++ scene ID (env: SCENE_ID, default: 00a231a370)")
    parser.add_argument("--mask-variant", default="", help="Mask variant (env: MASK_VARIANT, default: benchmark)")
    parser.add_argument("--image-subdir", default="", help="Image subdirectory (env: IMAGE_SUBDIR)")
    parser.add_argument("--masks-root", default="", help="ScanNet++ masks root directory (env: SCANNETPP_MASKS_ROOT)")
    parser.add_argument("--images-root", default="", help="ScanNet++ images root directory (env: SCANNETPP_IMAGES_ROOT)")
    parser.add_argument("--masks-subdir", default="", help="Masks subdirectory name (env: SCANNETPP_MASKS_SUBDIR)")
    parser.add_argument("--output-dir", default="", help="Output root directory (env: OUTPUT_DIR)")
    parser.add_argument("--config", default="", help="Config YAML path override")
    parser.add_argument("--stable-min-frames", type=int, default=3, help="Stable min frames threshold (default: 3)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--detector-backend", default="davis", help="Detector backend (default: davis)")
    parser.add_argument("--no-prefer-external", action="store_true", help="Do not prefer external scene data")
    args = parser.parse_args()

    def _e(arg_val: str, env_name: str, default: str = "") -> str:
        return (arg_val or os.environ.get(env_name, default) or default).strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(base_dir, ".."))
    project_dir = os.path.abspath(os.path.join(src_dir, ".."))

    config_path = _e(args.config, "CONFIG_PATH") or os.path.join(src_dir, "config", "default_config.yaml")

    if args.no_prefer_external:
        prefer_external = False
    else:
        prefer_external = _env_flag("PREFER_EXTERNAL_SCENE", default=True)

    input_source = resolve_testing_input_source(
        project_dir,
        frames_dir=_e(args.frames_dir, "INPUT_FRAMES_DIR"),
        davis_meta_path=_e(args.davis_meta_path, "DAVIS_META_PATH"),
        davis_annotations_dir=_e(args.davis_annotations_dir, "DAVIS_ANNOTATIONS_DIR"),
        sequence_name=_e(args.sequence_name, "DAVIS_SEQUENCE_NAME"),
        masks_root=_e(args.masks_root, "SCANNETPP_MASKS_ROOT"),
        images_root=_e(args.images_root, "SCANNETPP_IMAGES_ROOT"),
        scene_id=_e(args.scene_id, "SCENE_ID", "00a231a370"),
        mask_variant=_e(args.mask_variant, "MASK_VARIANT", "benchmark"),
        image_subdir=_e(args.image_subdir, "IMAGE_SUBDIR", "dslr/resized_images"),
        masks_subdir=_e(args.masks_subdir, "SCANNETPP_MASKS_SUBDIR", "2Dmasks"),
        prefer_external=prefer_external,
    )
    frames_dir = input_source["frames_dir"]
    sequence_name = input_source["sequence_name"]
    stable_min_frames = args.stable_min_frames
    max_frames = args.max_frames
    force_detector_backend = args.detector_backend

    cfg = Config(default_config_path=config_path)
    config = cfg.to_dict()
    input_cfg = config.setdefault("input", {})
    input_cfg["frames_dir"] = frames_dir
    if force_detector_backend is not None:
        config.setdefault("detector", {})["backend"] = str(force_detector_backend)
    davis_cfg = config.setdefault("davis", {})
    davis_cfg["sequence_name"] = sequence_name
    if input_source.get("davis_meta_path"):
        davis_cfg["meta_path"] = input_source["davis_meta_path"]
    if input_source.get("davis_annotations_dir"):
        davis_cfg["annotations_dir"] = input_source["davis_annotations_dir"]
    davis_variant = str(davis_cfg.get("variant", "bench") or "bench")
    timing_cfg = config.setdefault("timing", {})
    timing_cfg["enabled"] = False
    timing_cfg["table"] = False
    timing_cfg["detail_keys"] = []
    trace_cfg = config.setdefault("debug", {}).setdefault("association_trace", {})
    trace_cfg["enabled"] = False
    trace_cfg["mode"] = "off"

    ctx = initialize_system(config)
    pipeline = ReIDPipeline(ctx)
    gt_loader = DavisGroundTruthLoader(config)
    evaluator = TrackingEvaluator(stable_min_frames=stable_min_frames, config=config)
    scene_tag = sanitize_name_for_path(gt_loader.sequence_name)

    _output_dir_arg = _e(args.output_dir, "OUTPUT_DIR")
    testing_output_root = (
        Path(_output_dir_arg).expanduser().resolve()
        if _output_dir_arg
        else Path(project_dir) / "outputs" / "tfm"
    )
    testing_root = testing_output_root / "testing"
    testing_root.mkdir(parents=True, exist_ok=True)
    migrated_legacy_dir = migrate_legacy_testing_outputs(testing_root, scene_tag=scene_tag)
    out_dir = Path(
        default_run_artifact_dir(
            str(testing_output_root),
            group="testing",
            prefix=f"tracking_eval_{scene_tag}",
        )
    )
    out_dir = choose_unique_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_files, use_sequential_frame_ids = resolve_frame_files_for_testing(
        frames_dir,
        davis_meta_path=input_source.get("davis_meta_path", ""),
    )
    if not frame_files:
        raise RuntimeError(f"No images found in {frames_dir}")
    if max_frames is not None:
        frame_files = frame_files[: max(0, int(max_frames))]

    print(f"[TEST] Input source -> {input_source['mode']}")
    print(f"[TEST] Input sequence -> {sequence_name}")
    print(f"[TEST] Frames dir -> {frames_dir}")
    if input_source.get("image_subdir"):
        print(f"[TEST] Image subdir -> {input_source['image_subdir']}")
    print(f"[TEST] Resolved frames -> {len(frame_files)}")
    print(f"[TEST] DAVIS variant -> {davis_variant}")
    print(f"[TEST] GT sequence resolved -> {gt_loader.sequence_name}")
    if input_source.get("davis_meta_path"):
        print(f"[TEST] DAVIS meta -> {input_source['davis_meta_path']}")
    if input_source.get("davis_annotations_dir"):
        print(f"[TEST] DAVIS annotations -> {input_source['davis_annotations_dir']}")

    total_read_ms = 0.0
    total_pipeline_ms = 0.0
    total_gt_ms = 0.0
    total_eval_ms = 0.0
    total_post_ms = 0.0
    total_loop_ms = 0.0
    per_frame_timing_by_frame_id: dict[int, dict[str, float]] = {}
    per_frame_runtime_memory_by_frame_id: dict[int, dict[str, int | None]] = {}
    process = make_process_handle()

    for idx, frame_path in enumerate(frame_files):
        loop_t0 = perf_counter()
        frame_id = parse_frame_id(frame_path)
        if use_sequential_frame_ids:
            frame_id = int(idx)
        else:
            frame_id = int(idx) if frame_id is None else int(frame_id)

        rss_before = read_process_rss_bytes(process)
        read_t0 = perf_counter()
        frame = read_bgr(frame_path)
        read_ms = (perf_counter() - read_t0) * 1000.0
        if frame is None:
            raise RuntimeError(f"Failed to read frame {frame_path}")
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
        print(
            f"[TEST][frame={frame_id}] "
            f"read={read_ms:.2f} ms | "
            f"pipeline={pipeline_ms:.2f} ms | "
            f"gt={gt_ms:.2f} ms | "
            f"eval={eval_ms:.2f} ms | "
            f"post={post_ms:.2f} ms | "
            f"loop={loop_ms:.2f} ms"
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
    }
    results["timing_summary"] = timing_summary
    summary = results.setdefault("summary", {})
    summary.update(timing_summary)

    per_frame_rows = results.get("per_frame", []) or []
    if per_frame_rows:
        for row in per_frame_rows:
            frame_id = int(row.get("frame_id", -1))
            frame_timing = per_frame_timing_by_frame_id.get(frame_id, {})
            row["read_ms"] = float(frame_timing.get("read_ms", 0.0))
            row["pipeline_ms"] = float(frame_timing.get("pipeline_ms", 0.0))
            row["gt_ms"] = float(frame_timing.get("gt_ms", 0.0))
            row["eval_ms"] = float(frame_timing.get("eval_ms", 0.0))
            row["post_ms"] = float(frame_timing.get("post_ms", 0.0))
            row["loop_ms"] = float(frame_timing.get("loop_ms", 0.0))
            row.update(per_frame_runtime_memory_by_frame_id.get(frame_id, {}))
    summary.update(build_memory_summary(per_frame_rows))

    write_json(out_dir / "tracking_eval.json", results)

    per_object_rows = []
    for row in results.get("per_object", []) or []:
        x = dict(row)
        x.pop("segments", None)
        x.pop("pred_ids_timeline", None)
        x.pop("frames_timeline", None)
        per_object_rows.append(x)
    write_csv(out_dir / "tracking_eval_per_object.csv", per_object_rows)
    write_csv(out_dir / "tracking_eval_per_class.csv", results.get("per_class", []) or [])
    write_csv(out_dir / "tracking_eval_per_case.csv", results.get("per_case", []) or [])
    write_csv(out_dir / "tracking_eval_per_frame.csv", results.get("per_frame", []) or [])
    write_csv(out_dir / "tracking_eval_per_pred_track.csv", results.get("per_pred_track", []) or [])
    write_csv(out_dir / "per_case_modules.csv", results.get("per_case_modules", []) or [])
    write_csv(
        out_dir / "per_event.csv",
        build_event_rows(run_id="single_scene", scene_id=scene_tag, scene_name=scene_tag, results=results),
    )

    report = build_console_report(results)
    write_text(out_dir / "tracking_eval_report.txt", report + "\n")
    write_text(out_dir / "report.txt", report + "\n")
    print(report)
    if migrated_legacy_dir is not None:
        print(f"[TEST] Legacy testing outputs moved to {migrated_legacy_dir}")
    print(f"\n[TEST] Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
