# main.py

import os
import shutil
import json
from pathlib import Path

import cv2
import numpy as np

from config.config_loader import Config
from pipeline.initialization import initialize_system
from pipeline.reid_pipeline import ReIDPipeline
from utils.time import ExecutionTimer, format_timing_table
from utils.debug.association.debug_assoc import (
    print_assoc_diagnostics_table,
    print_assoc_similarity_details_table,
    print_assoc_table,
    print_known_set_distance_disambiguation_table,
    print_neighbor_sets_table,
    print_postcreate_temporal_table,
)
from utils.debug.update.debug_update import (
    print_memory_table,
    print_proto_update_table,
    print_update_summary,
    print_neighbor_distance_graph,
)
from utils.io import (
    basename,
    decode_action,
    list_image_files,
    parse_frame_id,
    read_bgr,
)
from utils.image import ImageHistory
from utils.logging import TeeStdout, default_run_log_path
from utils.scannetpp_tar import resolve_prepared_scene_from_tar
from utils.visualization import (
    overlay_header,
)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name, None)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def resolve_main_input_source(project_dir: str) -> dict[str, str]:
    """
    Resolve the input scene for interactive mode.

    Priority:
    1. Explicit environment variables.
    2. Uncompressed ScanNet++ scene when available.
    3. Fallback local dentro del repo.
    """
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
        }

    local_scannetpp_root = project_path / "data" / "scannetpp_data"
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

    prefer_external = _env_flag("REMIND_PREFER_EXTERNAL_SCENE", default=True)
    external_ready = (
        external_frames_dir.is_dir()
        and external_meta_path.is_file()
        and external_annotations_dir.is_dir()
    )
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


def resolve_frame_files_for_main(frames_dir: str, *, davis_meta_path: str = "") -> tuple[list[str], bool]:
    """
    Resolve the input frame list.

    If DAVIS meta provides `frame_names`, use that list to map
    dense `frame_000XYZ.png` masks against the original names
    (`DSC....JPG`) and process exactly the annotated frames.

    Returns:
    - lista de paths de frame en el orden correcto
    - bool indicating whether `frame_id` should be sequential (0..N-1)
    """
    meta_path = Path(str(davis_meta_path).strip()).expanduser() if str(davis_meta_path).strip() else None
    frames_root = Path(frames_dir).expanduser().resolve()

    if meta_path is not None and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid DAVIS meta at {meta_path}: malformed JSON.") from e
        if not isinstance(meta, dict):
            raise ValueError(f"Invalid DAVIS meta at {meta_path}: expected a JSON object.")

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
                    f"Missing {len(missing)} meta frames in {frames_root}. "
                    f"Ejemplos: {preview}"
                )

            if out:
                return out, True

    return list_image_files(str(frames_root)), False


def build_debug_header(*, view_idx: int, total: int, frame_id: int | None, name: str, mode: str, processed_until: int) -> str:
    frame_txt = "NA" if frame_id is None else str(int(frame_id))
    total_last = max(0, int(total) - 1)
    return (
        f"idx={int(view_idx)}/{total_last} | frame_id={frame_txt} | {str(name)} | "
        f"mode={str(mode)} | processed_idx={int(processed_until)}"
    )


FRAME_TIMING_ORDER = [
    "read_frame",
    "pipeline",
    "debug_association",
    "debug_update",
    "history_put",
    "ui_show",
    "ui_wait",
]


def _apply_main_input_to_config(config: dict, *, input_source: dict, frames_dir: str, sequence_name: str) -> None:
    input_cfg = config.setdefault("input", {})
    input_cfg["frames_dir"] = frames_dir

    davis_cfg = config.setdefault("davis", {})
    davis_cfg["sequence_name"] = sequence_name
    if input_source.get("davis_meta_path"):
        davis_cfg["meta_path"] = input_source["davis_meta_path"]
    if input_source.get("davis_annotations_dir"):
        davis_cfg["annotations_dir"] = input_source["davis_annotations_dir"]


def _print_main_input_summary(*, input_source: dict, frames_dir: str, sequence_name: str, frame_count: int) -> None:
    print(f"[INFO] Input source -> {input_source['mode']}")
    print(f"[INFO] Input sequence -> {sequence_name}")
    print(f"[INFO] Frames dir -> {frames_dir}")
    if input_source.get("image_subdir"):
        print(f"[INFO] Image subdir -> {input_source['image_subdir']}")
    print(f"[INFO] Resolved frames -> {int(frame_count)}")
    if input_source.get("davis_meta_path"):
        print(f"[INFO] DAVIS meta -> {input_source['davis_meta_path']}")
    if input_source.get("davis_annotations_dir"):
        print(f"[INFO] DAVIS annotations -> {input_source['davis_annotations_dir']}")


def _resolve_visualization_runtime(config: dict) -> dict[str, int | bool]:
    dbg_cfg = (config.get("debug", {}) or {})
    viz_cfg = (dbg_cfg.get("visualization", {}) or {})

    viz_enabled = bool(viz_cfg.get("enabled", True))

    cache_frames = int(viz_cfg.get("cache_frames", 300))
    cache_frames = max(1, cache_frames)

    mode_auto = bool(viz_cfg.get("auto_start", True))
    auto_delay_ms = int(viz_cfg.get("auto_delay_ms", 30))
    auto_delay_ms = max(0, auto_delay_ms)

    win_w = max(320, int(viz_cfg.get("window_width", 2000)))
    win_h = max(240, int(viz_cfg.get("window_height", 600)))

    return {
        "viz_enabled": bool(viz_enabled),
        "cache_frames": int(cache_frames),
        "mode_auto": bool(mode_auto),
        "auto_delay_ms": int(auto_delay_ms),
        "window_width": int(win_w),
        "window_height": int(win_h),
    }


def _prepare_viz_dir(output_dir: Path) -> str:
    viz_dir = os.path.join(str(output_dir), "debug_viz")
    if os.path.isdir(viz_dir):
        shutil.rmtree(viz_dir)
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir


def _print_controls(*, cache_frames: int, viz_dir: str, mode_auto: bool, auto_delay_ms: int) -> None:
    print("[INFO] Controls:")
    print("  SPACE = toggle AUTO/MANUAL")
    print("  ← / a = prev (MANUAL)")
    print("  → / d = next (MANUAL; in AUTO advances automatically)")
    print("  c = catch up (jump to last processed)")
    print("  q / ESC = quit")
    print(f"[INFO] Hybrid cache: ram={int(cache_frames)} frames | disk={viz_dir}")
    print(f"[INFO] Start mode: {'AUTO' if mode_auto else 'MANUAL'}  (delay_ms={int(auto_delay_ms)})")


def _show_debug_window(viz_show: np.ndarray, header: str) -> None:
    cv2.imshow("debug", viz_show)
    cv2.setWindowTitle("debug", header)


def _render_review_frame(
    *,
    history: ImageHistory,
    frame_files: list[str],
    view_idx: int,
    total: int,
    processed_until: int,
    mode_auto: bool,
    frame_meta_by_idx: dict[int, dict],
    use_sequential_frame_ids: bool,
) -> None:
    viz = history.get(view_idx)
    if viz is None:
        viz = read_bgr(frame_files[view_idx])

    if viz is None:
        raise FileNotFoundError(f"Could not load image for review: {frame_files[view_idx]}")

    mode = f"REVIEW_{'AUTO' if mode_auto else 'MANUAL'}"
    meta = frame_meta_by_idx.get(int(view_idx), {}) or {}
    name = str(meta.get("name", basename(frame_files[view_idx])))
    fallback_frame_id = int(view_idx) if use_sequential_frame_ids else parse_frame_id(frame_files[view_idx])
    frame_id = meta.get("frame_id", fallback_frame_id)
    frame_id = None if frame_id is None else int(frame_id)
    header = build_debug_header(
        view_idx=int(view_idx),
        total=int(total),
        frame_id=frame_id,
        name=name,
        mode=mode,
        processed_until=int(processed_until),
    )
    viz_show = overlay_header(viz.copy(), header)
    _show_debug_window(viz_show, header)


def _resolve_frame_id_for_path(*, frame_path: str, view_idx: int, use_sequential_frame_ids: bool) -> int:
    parsed_frame_id = parse_frame_id(frame_path)
    if use_sequential_frame_ids:
        return int(view_idx)
    return int(view_idx) if parsed_frame_id is None else int(parsed_frame_id)


def _build_det_id_to_local(detections: list) -> dict[int, int]:
    out = {}
    for idx, det in enumerate(detections or []):
        det_id = getattr(det, "detection_id", None)
        if det_id is not None:
            out[int(det_id)] = int(idx)
    return out


def _print_association_debug_tables(*, config: dict, frame_id: int, assoc_output, memory_store, det_id_to_local: dict[int, int]) -> None:
    print_assoc_table(config, frame_id, assoc_output, memory_store, det_id_to_local=det_id_to_local)
    print_assoc_diagnostics_table(config, frame_id, assoc_output, memory_store, det_id_to_local=det_id_to_local)
    print_assoc_similarity_details_table(config, frame_id, assoc_output, memory_store, det_id_to_local=det_id_to_local)
    print_neighbor_sets_table(config, frame_id, assoc_output, memory_store, det_id_to_local=det_id_to_local)
    print_known_set_distance_disambiguation_table(config, frame_id, assoc_output, memory_store, det_id_to_local=det_id_to_local)
    print_postcreate_temporal_table(config, frame_id, assoc_output, memory_store, det_id_to_local=det_id_to_local)


def _print_update_debug_tables(*, config: dict, frame_id: int, update_output, memory_store, det_id_to_local: dict[int, int]) -> None:
    print_update_summary(config, frame_id, update_output, memory_store, det_id_to_local=det_id_to_local)
    print_neighbor_distance_graph(config, frame_id, update_output, memory_store, object_ids=None)
    print_proto_update_table(config, frame_id, update_output)
    print_memory_table(config, frame_id, memory_store)


def _print_frame_timing(
    *,
    frame_total_timer: ExecutionTimer | None,
    frame_total_id: int | None,
    config: dict,
    ui_wait_seconds: float,
) -> None:
    if frame_total_timer is None or frame_total_id is None:
        return
    if not bool((config.get("timing", {}) or {}).get("enabled", True)):
        return

    frame_total_timer.add("ui_wait", float(ui_wait_seconds))
    print(
        format_timing_table(
            frame_total_timer.snapshot_seconds(),
            order=list(FRAME_TIMING_ORDER),
            precision=int((config.get("timing", {}) or {}).get("precision", 2)),
            total_seconds=frame_total_timer.total_seconds(),
            title=f"[TIME_FRAME][frame={int(frame_total_id)}]",
        )
    )



def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, ".."))

    config_path = os.path.join(base_dir, "config", "default_config.yaml")
    input_source = resolve_main_input_source(project_dir)
    frames_dir = input_source["frames_dir"]
    sequence_name = input_source["sequence_name"]

    cfg = Config(default_config_path=config_path)
    config = cfg.to_dict()
    _apply_main_input_to_config(
        config,
        input_source=input_source,
        frames_dir=frames_dir,
        sequence_name=sequence_name,
    )
    ctx = initialize_system(config)

    log_path = default_run_log_path(str(ctx.output_dir), prefix="reid_console")
    tee = TeeStdout(log_path)
    tee.install()
    print(f"[INFO] Console log -> {log_path}")

    pipeline = ReIDPipeline(ctx)

    frame_files, use_sequential_frame_ids = resolve_frame_files_for_main(
        frames_dir,
        davis_meta_path=input_source.get("davis_meta_path", ""),
    )
    if not frame_files:
        print(f"[ERROR] No images found in: {frames_dir}")
        tee.close()
        return

    _print_main_input_summary(
        input_source=input_source,
        frames_dir=frames_dir,
        sequence_name=sequence_name,
        frame_count=len(frame_files),
    )

    viz_runtime = _resolve_visualization_runtime(ctx.config)
    viz_enabled = bool(viz_runtime["viz_enabled"])
    cache_frames = int(viz_runtime["cache_frames"])
    mode_auto = bool(viz_runtime["mode_auto"])
    auto_delay_ms = int(viz_runtime["auto_delay_ms"])
    win_w = int(viz_runtime["window_width"])
    win_h = int(viz_runtime["window_height"])

    viz_dir = _prepare_viz_dir(ctx.output_dir)

    history = ImageHistory(disk_dir=viz_dir, ram_max_items=cache_frames)
    frame_meta_by_idx = {}

    processed_until = -1
    view_idx = 0
    total = len(frame_files)

    _print_controls(
        cache_frames=cache_frames,
        viz_dir=viz_dir,
        mode_auto=mode_auto,
        auto_delay_ms=auto_delay_ms,
    )

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug", win_w, win_h)

    while True:
        frame_total_timer = None
        frame_total_id = None
        if view_idx <= processed_until:
            _render_review_frame(
                history=history,
                frame_files=frame_files,
                view_idx=int(view_idx),
                total=int(total),
                processed_until=int(processed_until),
                mode_auto=bool(mode_auto),
                frame_meta_by_idx=frame_meta_by_idx,
                use_sequential_frame_ids=bool(use_sequential_frame_ids),
            )

        else:
            if view_idx != processed_until + 1:
                view_idx = processed_until + 1
                continue

            frame_path = frame_files[view_idx]
            name = basename(frame_path)
            frame_total_timer = ExecutionTimer()
            frame = frame_total_timer.run("read_frame", read_bgr, frame_path)

            if frame is None:
                print(f"[WARN] Failed to read: {frame_path}")
                processed_until = view_idx
                if view_idx < total - 1:
                    view_idx += 1
                continue

            frame_id = _resolve_frame_id_for_path(
                frame_path=frame_path,
                view_idx=int(view_idx),
                use_sequential_frame_ids=bool(use_sequential_frame_ids),
            )
            timestamp = float(frame_id)
            frame_total_id = int(frame_id)

            p_out, a_out, u_out = frame_total_timer.run(
                "pipeline",
                pipeline.process_frame,
                frame=frame,
                frame_id=frame_id,
                timestamp=timestamp,
            )

            det_id_to_local = _build_det_id_to_local(p_out.detections or [])

            frame_total_timer.run(
                "debug_association",
                _print_association_debug_tables,
                config=ctx.config,
                frame_id=int(frame_id),
                assoc_output=a_out,
                memory_store=ctx.memory,
                det_id_to_local=det_id_to_local,
            )
            frame_total_timer.run(
                "debug_update",
                _print_update_debug_tables,
                config=ctx.config,
                frame_id=int(frame_id),
                update_output=u_out,
                memory_store=ctx.memory,
                det_id_to_local=det_id_to_local,
            )

            viz = frame

            mode = f"PLAY_{'AUTO' if mode_auto else 'MANUAL'}"
            processed_until = view_idx
            frame_meta_by_idx[int(view_idx)] = {
                "frame_id": int(frame_id),
                "name": str(name),
            }
            header = build_debug_header(
                view_idx=int(view_idx),
                total=int(total),
                frame_id=int(frame_id),
                name=name,
                mode=mode,
                processed_until=int(processed_until),
            )
            viz_show = overlay_header(viz.copy(), header)

            frame_total_timer.run("history_put", history.put, view_idx, viz, save_disk=True)
            frame_total_timer.run("ui_show", _show_debug_window, viz_show, header)

        key_wait_t0 = cv2.getTickCount()
        key = cv2.waitKeyEx(auto_delay_ms if mode_auto else 0)
        key_wait_dt = (cv2.getTickCount() - key_wait_t0) / cv2.getTickFrequency()
        _print_frame_timing(
            frame_total_timer=frame_total_timer,
            frame_total_id=frame_total_id,
            config=ctx.config,
            ui_wait_seconds=float(key_wait_dt),
        )
        action = decode_action(key)

        if action == "toggle_auto":
            mode_auto = not mode_auto
            continue

        if action == "quit":
            break

        if action == "catchup":
            view_idx = processed_until if processed_until >= 0 else 0
            continue

        if mode_auto and action == "none":
            if view_idx < total - 1:
                view_idx += 1
            continue

        if action == "left":
            view_idx = max(0, view_idx - 1)
            continue

        if action == "right":
            if view_idx < total - 1:
                view_idx += 1
            continue

    cv2.destroyAllWindows()
    print("[INFO] Finished processing frames.")
    tee.close()


if __name__ == "__main__":
    main()
