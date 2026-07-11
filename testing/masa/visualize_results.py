#!/usr/bin/env python3
"""Visualize MASA tracking results together with their input DAVIS boxes."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
TESTING_DIR = CURRENT_DIR.parent
SRC_DIR = TESTING_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.config_loader import Config
from testing import run_tracking_batch_tar as tar_batch
from testing.davis_gt import DavisGroundTruthLoader
from utils.io import list_image_files, parse_frame_id, read_bgr


WINDOW_NAME = "MASA results"
RIGHT_KEYS = {83, 2555904, ord("d"), ord("D"), ord(".")}
LEFT_KEYS = {81, 2424832, ord("a"), ord("A"), ord(",")}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Draw the exact DAVIS boxes supplied to MASA, their masks, and "
            "the predicted track IDs stored in per_case.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="MASA run directory or scene directory containing per_case.csv",
    )
    parser.add_argument("--dataset", choices=("custom", "scannet-tar"), default="custom")
    parser.add_argument("--scene-id", help="Scene to select when --results-dir is a batch run")
    parser.add_argument("--frames-dir")
    parser.add_argument("--davis-meta-path")
    parser.add_argument("--davis-annotations-dir")
    parser.add_argument("--dataset-root", help="Top-level ScanNet++ directory with data/ and annotations/")
    parser.add_argument("--data-tar-root", help="Directory containing ScanNet++ scene data TARs or dirs")
    parser.add_argument("--annotations-tar-root", help="Directory containing ScanNet++ annotation TARs or dirs")
    parser.add_argument("--image-subdir", default="dslr/resized_images")
    parser.add_argument("--mask-variant", default="benchmark")
    parser.add_argument(
        "--config-path",
        default=str(SRC_DIR / "config" / "default_config.yaml"),
        help="REMIND config used by the MASA evaluation",
    )
    parser.add_argument("--output", help="Write an MP4 instead of opening an interactive window")
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--mask-alpha", type=float, default=0.22)
    parser.add_argument("--no-masks", action="store_true")
    parser.add_argument("--hide-coordinates", action="store_true")
    parser.add_argument(
        "--comparison-style",
        action="store_true",
        help="Match the existing REMIND/D4SM video style (masks, contours and compact labels).",
    )
    parser.add_argument(
        "--allow-non-gt-matched-geometry",
        action="store_true",
        help=(
            "For non-GT runs, draw predicted identities on matched DAVIS geometry. "
            "This does not reconstruct the original detector masks."
        ),
    )
    parser.add_argument("--display-max-width", type=int, default=1600)
    parser.add_argument("--display-max-height", type=int, default=900)
    parser.add_argument("--output-width", type=int, help="Resize rendered frames to this width.")
    parser.add_argument("--output-height", type=int, help="Resize rendered frames to this height.")
    return parser


def _resolve_scene_results(results_dir: Path, scene_id: str | None) -> Path:
    results_dir = results_dir.expanduser().resolve()
    if (results_dir / "per_case.csv").is_file():
        return results_dir
    if scene_id and (results_dir / "scenes" / scene_id / "per_case.csv").is_file():
        return (results_dir / "scenes" / scene_id).resolve()
    candidates = sorted((results_dir / "scenes").glob("*/per_case.csv"))
    if len(candidates) == 1:
        return candidates[0].parent.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    raise FileNotFoundError(
        "Could not resolve a scene per_case.csv. Pass the scene output directory "
        "or add --scene-id."
    )


def _read_cases(path: Path) -> dict[int, dict[int, dict[str, str]]]:
    by_frame: dict[int, dict[int, dict[str, str]]] = defaultdict(dict)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            frame_id = int(row["frame_id"])
            gt_id = int(row["gt_instance_id"])
            by_frame[frame_id][gt_id] = dict(row)
    return dict(by_frame)


def _read_run_config(scene_results: Path) -> dict[str, str]:
    for directory in (scene_results, scene_results.parent, scene_results.parent.parent):
        path = directory / "run_config.csv"
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            return next(csv.DictReader(handle), {})
    return {}


def _as_optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _colour_for_track(track_id: int | None) -> tuple[int, int, int]:
    if track_id is None:
        return (160, 160, 160)
    hue = int((int(track_id) * 137 + 23) % 180)
    hsv = np.uint8([[[hue, 210, 245]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(value) for value in bgr)


def _comparison_colour(track_id: int | None, *, brighten: bool = False) -> tuple[int, int, int]:
    idx = -1 if track_id is None else int(track_id)
    hsv = np.uint8([[[(idx * 47) % 180, 200, 230]]])
    colour = tuple(int(value) for value in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])
    if not brighten:
        return colour
    return tuple(min(255, value + 45) for value in colour)


def _comparison_label(image: np.ndarray, text: str, anchor: tuple[int, int], colour: tuple[int, int, int]) -> None:
    height, width = image.shape[:2]
    font, scale, thickness, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2, 6
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = int(anchor[0]), int(anchor[1]) - 10
    if y - text_height - baseline - 2 * pad < 0:
        y = int(anchor[1]) + text_height + baseline + 10
    x0 = max(0, min(width - 1, x))
    y0 = max(0, min(height - 1, y - text_height - baseline - 2 * pad))
    x1 = max(0, min(width - 1, x0 + text_width + 2 * pad))
    y1 = max(0, min(height - 1, y0 + text_height + baseline + 2 * pad))
    x0, y0 = max(0, x1 - text_width - 2 * pad), max(0, y1 - text_height - baseline - 2 * pad)
    cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), colour, 1)
    cv2.putText(image, text, (x0 + pad, y1 - baseline - pad), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)


def _render_comparison_frame(
    *, frame: np.ndarray, gt_objects: dict[int, Any], cases: dict[int, dict[str, str]],
    frame_id: int, frame_index: int, total_frames: int, sequence_name: str,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    items = []
    for gt_id, gt_obj in sorted(gt_objects.items()):
        case = cases.get(int(gt_id))
        track_id = _as_optional_int(None if case is None else case.get("pred_object_id"))
        if gt_obj.bbox_xyxy is None:
            continue
        x1, y1, x2, y2 = (int(value) for value in gt_obj.bbox_xyxy)
        mask = np.asarray(gt_obj.mask, dtype=bool)
        full_mask = np.zeros((height, width), dtype=bool)
        if mask.shape == (height, width):
            full_mask = mask
        elif output[y1:y2, x1:x2].shape[:2] == mask.shape:
            full_mask[y1:y2, x1:x2] = mask
        items.append((int(full_mask.sum()), gt_obj, track_id, full_mask))

    for _, _, track_id, mask in sorted(items, key=lambda item: item[0], reverse=True):
        colour = _comparison_colour(track_id, brighten=True)
        overlay = output.copy()
        overlay[mask] = colour
        output = cv2.addWeighted(overlay, 0.45, output, 0.55, 0.0)
        contours, _ = cv2.findContours(mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output, contours, -1, colour, 2, lineType=cv2.LINE_AA)

    for _, gt_obj, track_id, _ in sorted(items, key=lambda item: item[0]):
        x1, y1, _, _ = (int(value) for value in gt_obj.bbox_xyxy)
        class_name = str(gt_obj.class_name or "obj").strip().upper()
        label = f"{class_name}_{'-' if track_id is None else track_id}"
        _comparison_label(output, label, (x1, y1), _comparison_colour(track_id))

    return output


def _fit_for_display(frame: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_width) / width, float(max_height) / height)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _draw_label(
    image: np.ndarray,
    *,
    text: str,
    box: tuple[int, int, int, int],
    colour: tuple[int, int, int],
) -> None:
    height, width = image.shape[:2]
    x1, y1, _, _ = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.48
    thickness = 1
    pad = 4
    reserved_top = 120
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    left = max(0, min(width - text_width - 2 * pad, x1))
    top = y1 - text_height - baseline - 2 * pad
    if top < reserved_top:
        top = min(height - text_height - baseline - 2 * pad, max(reserved_top, y1 + 3))
    right = min(width - 1, left + text_width + 2 * pad)
    bottom = min(height - 1, top + text_height + baseline + 2 * pad)
    cv2.rectangle(image, (left, top), (right, bottom), (18, 18, 18), -1)
    cv2.putText(
        image,
        text,
        (left + pad, bottom - baseline - pad),
        font,
        font_scale,
        colour,
        thickness,
        cv2.LINE_AA,
    )


def _draw_hud(
    image: np.ndarray,
    *,
    frame_id: int,
    frame_index: int,
    total_frames: int,
    visible: int,
    run_id: str,
) -> None:
    lines = [
        f"MASA | {run_id}",
        f"frame_id={frame_id}  frame={frame_index + 1}/{total_frames}",
        f"input boxes={visible}",
        "bbox and mask colour = MASA track ID",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    panel_width = min(image.shape[1] - 16, 720)
    panel_height = 16 + 24 * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.78, image, 0.22, 0, image)
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (18, 30 + idx * 24),
            font,
            0.55,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )


def _render_frame(
    *,
    frame: np.ndarray,
    gt_objects: dict[int, Any],
    cases: dict[int, dict[str, str]],
    frame_id: int,
    frame_index: int,
    total_frames: int,
    run_id: str,
    mask_alpha: float,
    show_masks: bool,
    show_coordinates: bool,
) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    rendered: list[tuple[Any, dict[str, str] | None, int | None, tuple[int, int, int]]] = []

    for gt_id, gt_obj in sorted(gt_objects.items()):
        case = cases.get(int(gt_id))
        pred_id = _as_optional_int(None if case is None else case.get("pred_object_id"))
        colour = _colour_for_track(pred_id)
        rendered.append((gt_obj, case, pred_id, colour))
        if not show_masks:
            continue
        bbox = gt_obj.bbox_xyxy
        if bbox is None:
            continue
        x1, y1, x2, y2 = (int(value) for value in bbox)
        local_mask = np.asarray(gt_obj.mask, dtype=bool)
        region = overlay[y1:y2, x1:x2]
        if region.shape[:2] == local_mask.shape:
            region[local_mask] = colour

    if show_masks and rendered:
        alpha = max(0.0, min(1.0, float(mask_alpha)))
        cv2.addWeighted(overlay, alpha, output, 1.0 - alpha, 0, output)

    for gt_obj, case, pred_id, colour in rendered:
        if gt_obj.bbox_xyxy is None:
            continue
        box = tuple(int(value) for value in gt_obj.bbox_xyxy)
        x1, y1, x2, y2 = box
        cv2.rectangle(output, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), colour, 3)
        class_name = str(gt_obj.class_name or "unknown")
        pred_text = "-" if pred_id is None else str(pred_id)
        text = f"{class_name} DAVIS={gt_obj.instance_id} MASA={pred_text}"
        if show_coordinates:
            text += f" IN=[{x1},{y1},{x2},{y2}]"
        _draw_label(output, text=text, box=box, colour=colour)

    _draw_hud(
        output,
        frame_id=frame_id,
        frame_index=frame_index,
        total_frames=total_frames,
        visible=len(rendered),
        run_id=run_id,
    )
    return output


def _build_custom_gt_loader(args: argparse.Namespace, sequence_name: str) -> DavisGroundTruthLoader:
    if not args.frames_dir or not args.davis_meta_path or not args.davis_annotations_dir:
        raise ValueError(
            "Custom visualization requires --frames-dir, --davis-meta-path and "
            "--davis-annotations-dir."
        )
    config = Config(default_config_path=str(Path(args.config_path).expanduser().resolve())).to_dict()
    config.setdefault("input", {})["frames_dir"] = str(Path(args.frames_dir).expanduser().resolve())
    davis = config.setdefault("davis", {})
    davis["sequence_name"] = sequence_name
    davis["meta_path"] = str(Path(args.davis_meta_path).expanduser().resolve())
    davis["annotations_dir"] = str(Path(args.davis_annotations_dir).expanduser().resolve())
    return DavisGroundTruthLoader(config)


def _build_tar_gt_loader(
    args: argparse.Namespace,
    *,
    bundle: tar_batch.TarSceneBundle,
) -> DavisGroundTruthLoader:
    config = Config(default_config_path=str(Path(args.config_path).expanduser().resolve())).to_dict()
    config.setdefault("input", {})["frames_dir"] = str(bundle.data_tar_path)
    davis = config.setdefault("davis", {})
    davis["sequence_name"] = bundle.scene_id
    davis["variant"] = tar_batch._normalize_davis_variant(bundle.mask_variant)
    davis["tar_scene_bundle"] = bundle
    davis["prefetch_annotations"] = False
    return DavisGroundTruthLoader(config)


def _resolve_scannet_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.dataset_root:
        root = Path(args.dataset_root).expanduser().resolve()
        data_root = Path(args.data_tar_root).expanduser().resolve() if args.data_tar_root else root / "data"
        annotations_root = (
            Path(args.annotations_tar_root).expanduser().resolve()
            if args.annotations_tar_root
            else root / "annotations"
        )
        return data_root, annotations_root
    if not args.data_tar_root or not args.annotations_tar_root:
        raise ValueError(
            "ScanNet visualization requires --dataset-root or both "
            "--data-tar-root and --annotations-tar-root."
        )
    return (
        Path(args.data_tar_root).expanduser().resolve(),
        Path(args.annotations_tar_root).expanduser().resolve(),
    )


def _resolve_scene_id(scene_results: Path, requested_scene_id: str | None) -> str:
    if requested_scene_id:
        return str(requested_scene_id)
    if scene_results.parent.name == "scenes":
        return scene_results.name
    raise ValueError("--scene-id is required when visualizing ScanNet from a batch results dir")


def main() -> None:
    args = _parser().parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")

    scene_results = _resolve_scene_results(Path(args.results_dir), args.scene_id)
    run_config = _read_run_config(scene_results)
    detection_source = str(run_config.get("detection_source") or "gt").strip().lower()
    if detection_source != "gt" and not args.allow_non_gt_matched_geometry:
        raise ValueError(
            "This visualizer reconstructs exact DAVIS input boxes for GT runs. "
            "YOLO input boxes are not stored in the current result format."
        )
    cases_by_frame = _read_cases(scene_results / "per_case.csv")
    sequence_name = args.scene_id or scene_results.name
    first_case = next((row for rows in cases_by_frame.values() for row in rows.values()), {})
    run_id = str(run_config.get("run_id") or first_case.get("run_id") or scene_results.parent.parent.name)
    tar_bundle: tar_batch.TarSceneBundle | None = None

    if args.dataset == "custom":
        if not args.frames_dir:
            raise ValueError("Custom visualization requires --frames-dir.")
        frame_paths = [
            Path(path)
            for path in list_image_files(str(Path(args.frames_dir).expanduser().resolve()))
        ]
        if not frame_paths:
            raise FileNotFoundError(f"No frames found in {args.frames_dir}")

        selected: list[tuple[int, str]] = []
        for index, frame_path in enumerate(frame_paths):
            parsed = parse_frame_id(str(frame_path))
            frame_id = index if parsed is None else int(parsed)
            if args.start_frame is not None and frame_id < int(args.start_frame):
                continue
            selected.append((frame_id, str(frame_path)))
        gt_loader = _build_custom_gt_loader(args, sequence_name)

        def read_selected_frame(item: tuple[int, str]) -> np.ndarray:
            return read_bgr(item[1])

    else:
        scene_id = _resolve_scene_id(scene_results, args.scene_id)
        data_root, annotations_root = _resolve_scannet_roots(args)
        tar_bundle = tar_batch._build_scene_bundle(
            scene_id=scene_id,
            data_tar_root=data_root,
            annotations_tar_root=annotations_root,
            mask_variant=args.mask_variant,
            image_subdir=str(args.image_subdir or "dslr/resized_images"),
        )
        frame_source = tar_batch.TarFrameSource(tar_bundle)
        selected = []
        for frame_id, frame_name in enumerate(tar_bundle.frame_names):
            if args.start_frame is not None and frame_id < int(args.start_frame):
                continue
            selected.append((frame_id, frame_name))
        with tar_batch._patched_tar_davis_segmenter():
            gt_loader = _build_tar_gt_loader(args, bundle=tar_bundle)

        def read_selected_frame(item: tuple[int, str]) -> np.ndarray:
            frame = frame_source.read_bgr(item[1])
            if frame is None:
                raise RuntimeError(f"Could not read {item[1]} from {tar_bundle.data_tar_path}")
            return frame

    if args.max_frames is not None:
        selected = selected[: max(0, int(args.max_frames))]
    if not selected:
        raise ValueError("No frames remain after applying the requested range")

    def render(index: int) -> np.ndarray:
        frame_id, frame_ref = selected[index]
        frame = read_selected_frame((frame_id, frame_ref))
        gt_objects = gt_loader.load_frame(frame_id=frame_id, target_shape=frame.shape[:2])
        if args.comparison_style:
            rendered = _render_comparison_frame(
                frame=frame, gt_objects=gt_objects, cases=cases_by_frame.get(frame_id, {}),
                frame_id=frame_id, frame_index=index, total_frames=len(selected),
                sequence_name=sequence_name,
            )
        else:
            rendered = _render_frame(
                frame=frame,
                gt_objects=gt_objects,
                cases=cases_by_frame.get(frame_id, {}),
                frame_id=frame_id,
                frame_index=index,
                total_frames=len(selected),
                run_id=run_id,
                mask_alpha=args.mask_alpha,
                show_masks=not args.no_masks,
                show_coordinates=not args.hide_coordinates,
            )
        if args.output_width and args.output_height:
            rendered = cv2.resize(
                rendered, (int(args.output_width), int(args.output_height)),
                interpolation=cv2.INTER_AREA,
            )
        return rendered

    try:
        if args.dataset == "scannet-tar":
            tar_context = tar_batch._patched_tar_davis_segmenter()
        else:
            tar_context = None
        if tar_context is not None:
            tar_context.__enter__()
        try:
            if args.output:
                output_path = Path(args.output).expanduser().resolve()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                writer: cv2.VideoWriter | None = None
                try:
                    for index in range(len(selected)):
                        visual = render(index)
                        if writer is None:
                            height, width = visual.shape[:2]
                            writer = cv2.VideoWriter(
                                str(output_path),
                                cv2.VideoWriter_fourcc(*str(args.codec)),
                                float(args.fps),
                                (width, height),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(f"Could not open video writer: {output_path}")
                        writer.write(visual)
                        if index == 0 or (index + 1) % 100 == 0 or index + 1 == len(selected):
                            print(f"[MASA VIS] {index + 1}/{len(selected)}")
                finally:
                    if writer is not None:
                        writer.release()
                print(f"[MASA VIS] Video written to {output_path}")
                return

            index = 0
            playing = False
            delay_play = max(1, int(round(1000.0 / float(args.fps))))
            print("Controls: left/a previous, right/d next, space play/pause, q/esc quit")
            while True:
                visual = _fit_for_display(
                    render(index), int(args.display_max_width), int(args.display_max_height)
                )
                cv2.imshow(WINDOW_NAME, visual)
                key = cv2.waitKeyEx(delay_play if playing else 0)
                if key in {27, ord("q"), ord("Q")}:
                    break
                if key == 32:
                    playing = not playing
                    continue
                if key in RIGHT_KEYS:
                    index = min(len(selected) - 1, index + 1)
                    playing = False
                    continue
                if key in LEFT_KEYS:
                    index = max(0, index - 1)
                    playing = False
                    continue
                if playing:
                    if index + 1 >= len(selected):
                        playing = False
                    else:
                        index += 1
            cv2.destroyAllWindows()
        finally:
            if tar_context is not None:
                tar_context.__exit__(None, None, None)
    finally:
        if tar_bundle is not None:
            tar_bundle.close()


if __name__ == "__main__":
    main()
