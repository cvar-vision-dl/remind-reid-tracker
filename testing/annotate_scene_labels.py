from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config.config_loader import Config
from davis_gt import DavisGroundTruthLoader
from utils.io import decode_action, list_image_files, parse_frame_id, read_bgr
from utils.visualization import overlay_mask_bgr


WINDOW_NAME = "APP2 Scene Labeler"
DEFAULT_OUTPUT_CSV = Path(CURRENT_DIR) / "scene_labels.csv"
DEFAULT_AUTOPLAY_FPS = 12.0
DEFAULT_DISPLAY_MAX_W = 1600
DEFAULT_DISPLAY_MAX_H = 900

SCENE_TYPE_OPTIONS = {
    ord("h"): "home",
    ord("H"): "home",
    ord("o"): "office",
    ord("O"): "office",
    ord("i"): "industrial",
    ord("I"): "industrial",
    ord("m"): "mixed",
    ord("M"): "mixed",
}

LEVEL_OPTIONS = {
    ord("l"): "low",
    ord("L"): "low",
    ord("m"): "medium",
    ord("M"): "medium",
    ord("h"): "high",
    ord("H"): "high",
}

OCCLUSION_ERROR_OPTIONS = {
    ord("l"): "none",
    ord("L"): "none",
    ord("m"): "some",
    ord("M"): "some",
    ord("h"): "many",
    ord("H"): "many",
}

CAMERA_DISTANCE_OPTIONS = {
    ord("n"): "near",
    ord("N"): "near",
    ord("m"): "medium",
    ord("M"): "medium",
    ord("f"): "far",
    ord("F"): "far",
}

SAVE_KEYS = {13, 10, ord("s"), ord("S")}
NEXT_SCENE_KEYS = {ord("n"), ord("N")}
PREV_SCENE_KEYS = {ord("b"), ord("B")}
CLEAR_LABEL_KEYS = {ord("u"), ord("U")}
HELP_KEYS = {ord("?"), ord("/")}
TARGET_SCENE_TYPE_KEYS = {ord("t"), ord("T")}
TARGET_SYMMETRY_KEYS = {ord("r"), ord("R")}
TARGET_DENSITY_KEYS = {ord("v"), ord("V")}
TARGET_CLASS_REPETITION_KEYS = {ord("c"), ord("C")}
TARGET_OCCLUSION_ERRORS_KEYS = {ord("e"), ord("E")}
TARGET_CAMERA_DISTANCE_KEYS = {ord("p"), ord("P")}
TOGGLE_MASKS_KEYS = {ord("g"), ord("G")}

FIELD_LABELS = {
    "scene_type": "scene_type",
    "symmetry_repetition": "symmetry_repetition",
    "visible_density": "visible_density",
    "class_repetition": "class_repetition",
    "mask_errors": "mask_errors",
    "camera_distance": "camera_distance",
}

FIELD_TITLES = {
    "scene_type": "Scene type",
    "symmetry_repetition": "Symmetry",
    "visible_density": "Density",
    "class_repetition": "Class repetition",
    "mask_errors": "Mask errors",
    "camera_distance": "Camera distance",
}


@dataclass
class SceneEntry:
    scene_id: str
    frames_dir: str
    frame_paths: list[str]
    frame_names: list[str]
    frame_ids: list[int]
    davis_meta_path: str
    davis_annotations_dir: str


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def normalize_mask_errors(value: str | None) -> str:
    text = str(value or "").strip().lower()
    legacy_map = {
        "rare": "some",
        "occasional": "some",
        "frequent": "many",
    }
    if text in legacy_map:
        return legacy_map[text]
    return text


def normalize_camera_distance(value: str | None) -> str:
    text = str(value or "").strip().lower()
    legacy_map = {
        "near+medium": "medium",
        "near+medium+far": "far",
    }
    if text in legacy_map:
        return legacy_map[text]
    return text


def read_scene_ids_from_file(path: str) -> list[str]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"No existe APP2_BATCH_SCENES_FILE: {p}")
    values: list[str] = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        text = str(raw).strip()
        if not text or text.startswith("#"):
            continue
        if "," in text:
            text = text.split(",", 1)[0].strip()
        values.append(text)
    return unique_preserve_order(values)


def resolve_masks_parent(base_path: Path) -> Path | None:
    base = base_path.expanduser().resolve()
    if base.name == "annotations" and base.is_dir():
        return base
    candidate = (base / "annotations").resolve()
    if candidate.is_dir():
        return candidate
    return None


def resolve_data_parent(base_path: Path) -> Path | None:
    base = base_path.expanduser().resolve()
    if base.name == "data" and base.is_dir():
        return base
    candidate = (base / "data").resolve()
    if candidate.is_dir():
        return candidate
    return None


def resolve_scene_frames_dir(
    *,
    data_parent: Path,
    scene_id: str,
    preferred_subdir: str,
) -> Path | None:
    candidates = [
        str(preferred_subdir or "").strip(),
        "dslr/resized_undistorted_images",
        "dslr/resized_images",
        "iphone/rgb",
    ]
    seen: set[str] = set()
    for rel in candidates:
        rel_clean = str(rel or "").strip().strip("/")
        if not rel_clean or rel_clean in seen:
            continue
        seen.add(rel_clean)
        candidate = (data_parent / scene_id / rel_clean).resolve()
        if candidate.is_dir():
            return candidate
    return None


def resolve_batch_scene_ids(
    *,
    masks_parent: Path,
    data_parent: Path,
    mask_variant: str,
    image_subdir: str,
) -> list[str]:
    scenes_env = os.environ.get("APP2_BATCH_SCENES", "").strip()
    scenes_file = os.environ.get("APP2_BATCH_SCENES_FILE", "").strip()
    single_scene = os.environ.get("APP2_SCENE_ID", "").strip()

    if scenes_file:
        return read_scene_ids_from_file(scenes_file)
    if scenes_env:
        return unique_preserve_order(re.split(r"[\s,;]+", scenes_env))
    if single_scene:
        return [single_scene]

    scene_ids: list[str] = []
    for child in sorted(masks_parent.iterdir()):
        if not child.is_dir():
            continue
        scene_id = str(child.name)
        meta_path = (child / f"meta_{mask_variant}.json").resolve()
        ann_dir = (child / "annotations" / mask_variant).resolve()
        frames_dir = resolve_scene_frames_dir(
            data_parent=data_parent,
            scene_id=scene_id,
            preferred_subdir=image_subdir,
        )
        if meta_path.is_file() and ann_dir.is_dir() and frames_dir is not None:
            scene_ids.append(scene_id)
    return unique_preserve_order(scene_ids)


def resolve_frame_files(frames_dir: Path, *, davis_meta_path: Path | None) -> tuple[list[str], list[int]]:
    if davis_meta_path is not None and davis_meta_path.is_file():
        try:
            meta = json.loads(davis_meta_path.read_text(encoding="utf-8")) or {}
        except Exception:
            meta = {}
        frame_names = meta.get("frame_names", None)
        if isinstance(frame_names, list) and frame_names:
            ordered_paths: list[str] = []
            for raw_name in frame_names:
                name = str(raw_name).strip()
                if not name:
                    continue
                path = (frames_dir / name).resolve()
                if path.is_file():
                    ordered_paths.append(str(path))
            if ordered_paths:
                return ordered_paths, [int(idx) for idx in range(len(ordered_paths))]

    paths = list_image_files(str(frames_dir))
    frame_ids: list[int] = []
    for idx, path in enumerate(paths):
        parsed = parse_frame_id(path)
        frame_ids.append(int(idx) if parsed is None else int(parsed))
    return paths, frame_ids


def build_scene_entries() -> list[SceneEntry]:
    default_scannetpp_root = str(Path(SRC_DIR) / "data" / "scannetpp_data")
    masks_root_base = Path(
        os.environ.get(
            "APP2_SCANNETPP_MASKS_ROOT",
            default_scannetpp_root,
        )
    ).expanduser().resolve()
    images_root_base = Path(
        os.environ.get(
            "APP2_SCANNETPP_IMAGES_ROOT",
            default_scannetpp_root,
        )
    ).expanduser().resolve()
    mask_variant = os.environ.get("APP2_MASK_VARIANT", "benchmark").strip().lower() or "benchmark"
    image_subdir = os.environ.get("APP2_IMAGE_SUBDIR", "dslr/resized_images").strip() or "dslr/resized_images"
    if mask_variant == "benchmark":
        mask_variant = "benchmark_instance"

    masks_parent = resolve_masks_parent(masks_root_base)
    data_parent = resolve_data_parent(images_root_base)
    if masks_parent is None:
        return []
    if data_parent is None:
        return []

    scene_ids = resolve_batch_scene_ids(
        masks_parent=masks_parent,
        data_parent=data_parent,
        mask_variant=mask_variant,
        image_subdir=image_subdir,
    )
    entries: list[SceneEntry] = []
    for scene_id in scene_ids:
        masks_root = (masks_parent / scene_id).resolve()
        davis_meta_path = (masks_root / f"meta_{mask_variant}.json").resolve()
        annotations_dir = (masks_root / "annotations" / mask_variant).resolve()
        frames_dir = resolve_scene_frames_dir(
            data_parent=data_parent,
            scene_id=str(scene_id),
            preferred_subdir=image_subdir,
        )
        if frames_dir is None or not davis_meta_path.is_file() or not annotations_dir.is_dir():
            continue
        frame_paths, frame_ids = resolve_frame_files(
            frames_dir=frames_dir,
            davis_meta_path=davis_meta_path,
        )
        if not frame_paths:
            continue
        entries.append(
            SceneEntry(
                scene_id=str(scene_id),
                frames_dir=str(frames_dir),
                frame_paths=frame_paths,
                frame_names=[Path(path).name for path in frame_paths],
                frame_ids=frame_ids,
                davis_meta_path=str(davis_meta_path),
                davis_annotations_dir=str(annotations_dir),
            )
        )
    return entries


def load_existing_labels(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = str(row.get("scene_id", "")).strip()
            if not scene_id:
                continue
            normalized_row = {str(k): str(v) for k, v in row.items()}
            old_mask_errors = normalized_row.get("occlusion_mask_errors", "")
            if "mask_errors" in normalized_row:
                normalized_row["mask_errors"] = normalize_mask_errors(
                    normalized_row.get("mask_errors", "")
                )
            elif str(old_mask_errors).strip():
                normalized_row["mask_errors"] = normalize_mask_errors(old_mask_errors)
            if "camera_distance" in normalized_row:
                normalized_row["camera_distance"] = normalize_camera_distance(
                    normalized_row.get("camera_distance", "")
                )
            out[scene_id] = normalized_row
    return out


def write_labels_csv(path: Path, rows_by_scene: dict[str, dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "scene_id",
        "scene_type",
        "symmetry_repetition",
        "visible_density",
        "class_repetition",
        "mask_errors",
        "camera_distance",
        "n_frames",
        "frames_dir",
        "annotated_at",
    ]
    ordered_scene_ids = sorted(rows_by_scene.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for scene_id in ordered_scene_ids:
            row = dict(rows_by_scene[scene_id])
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def fit_for_display(frame, *, max_w: int, max_h: int):
    h, w = frame.shape[:2]
    scale = min(float(max_w) / float(w), float(max_h) / float(h), 1.0)
    if scale >= 0.999:
        return frame
    new_w = max(1, int(round(float(w) * scale)))
    new_h = max(1, int(round(float(h) * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def color_for_instance(instance_id: int) -> tuple[int, int, int]:
    hue = int((int(instance_id) * 47) % 180)
    hsv = np.uint8([[[hue, 230, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def overlay_gt_objects(frame_bgr, gt_objects: dict[int, Any], *, alpha: float = 0.68):
    out = frame_bgr.copy()
    for instance_id in sorted(int(x) for x in gt_objects.keys()):
        gt_obj = gt_objects[int(instance_id)]
        bbox = gt_obj.bbox_xyxy
        mask_local = gt_obj.mask
        if bbox is None or mask_local is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            continue
        full_mask = np.zeros(out.shape[:2], dtype=bool)
        full_mask[y1:y2, x1:x2] = np.asarray(mask_local).astype(bool, copy=False)
        color = color_for_instance(int(instance_id))
        out = overlay_mask_bgr(out, full_mask, color, float(alpha))

        mask_u8 = full_mask.astype(np.uint8, copy=False)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(out, contours, -1, color, 2, cv2.LINE_AA)
            main_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(main_contour)
            if moments["m00"] > 0.0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                bx, by, bw, bh = cv2.boundingRect(main_contour)
                cx = int(bx + bw // 2)
                cy = int(by + bh // 2)

            label_text = str(getattr(gt_obj, "label", "") or getattr(gt_obj, "class_name", "") or f"id_{instance_id}")
            (text_w, text_h), baseline = cv2.getTextSize(
                label_text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                1,
            )
            text_x = int(np.clip(cx - text_w // 2, 6, max(6, out.shape[1] - text_w - 6)))
            text_y = int(np.clip(cy, text_h + 8, max(text_h + 8, out.shape[0] - baseline - 6)))
            box_x1 = max(0, text_x - 4)
            box_y1 = max(0, text_y - text_h - 6)
            box_x2 = min(out.shape[1] - 1, text_x + text_w + 4)
            box_y2 = min(out.shape[0] - 1, text_y + baseline + 4)
            cv2.rectangle(out, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
            cv2.putText(
                out,
                label_text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    return out


def build_gt_loader(scene: SceneEntry) -> DavisGroundTruthLoader:
    config_path = os.path.join(SRC_DIR, "config", "default_config.yaml")
    cfg = Config(default_config_path=config_path)
    config = cfg.to_dict()
    config.setdefault("input", {})["frames_dir"] = str(scene.frames_dir)
    config.setdefault("detector", {})["backend"] = "davis"
    davis_cfg = config.setdefault("davis", {})
    davis_cfg["sequence_name"] = str(scene.scene_id)
    davis_cfg["meta_path"] = str(scene.davis_meta_path)
    davis_cfg["annotations_dir"] = str(scene.davis_annotations_dir)
    loader = DavisGroundTruthLoader(config)
    loader.load_frame  # keep linter calm about import usage
    return loader


def draw_overlay(
    frame,
    *,
    scene_index: int,
    n_scenes: int,
    scene: SceneEntry,
    frame_index: int,
    autoplay: bool,
    autoplay_fps: float,
    scene_type: str | None,
    symmetry_repetition: str | None,
    visible_density: str | None,
    class_repetition: str | None,
    mask_errors: str | None,
    camera_distance: str | None,
    n_gt_visible: int | None,
    show_masks: bool,
    active_field: str,
    show_help: bool,
):
    canvas = frame.copy()
    active_field_label = FIELD_LABELS.get(active_field, active_field)
    ready_to_save = bool(
        scene_type
        and symmetry_repetition
        and visible_density
        and class_repetition
        and mask_errors
        and camera_distance
    )
    header_lines = [
        f"Scene {scene_index + 1}/{n_scenes}: {scene.scene_id} | Frame {frame_index + 1}/{len(scene.frame_paths)}: {scene.frame_names[frame_index]}",
        f"Auto {'ON' if autoplay else 'OFF'} {autoplay_fps:.1f}fps | Masks {'ON' if show_masks else 'OFF'} | Visible GT {n_gt_visible if n_gt_visible is not None else '-'} | Active {active_field_label}",
    ]
    header_line_h = 24
    header_pad = 12
    header_h = header_pad * 2 + header_line_h * len(header_lines)
    header_w = max(520, min(int(canvas.shape[1] * 0.66), canvas.shape[1] - 32))
    header_overlay = canvas.copy()
    cv2.rectangle(header_overlay, (12, 12), (12 + header_w, 12 + header_h), (15, 15, 15), -1)
    canvas = cv2.addWeighted(header_overlay, 0.60, canvas, 0.40, 0.0)

    y = 12 + header_pad + 16
    for idx, text in enumerate(header_lines):
        color = (255, 255, 255) if idx == 0 else (120, 200, 255)
        cv2.putText(
            canvas,
            text,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
        y += header_line_h

    control_lines = [
        "Move: Left/Right or A/D | Space: auto | G: masks | T H/O/I/M: type | R L/M/H: symmetry | V L/M/H: density | C L/M/H: class repetition",
        "E L/M/H: mask errors | P N/M/F: camera distance | Enter/S: save next | B: previous scene | U: clear labels | Q/Esc: quit | ?: extra help",
    ]
    if show_help:
        control_lines.append("Tip: set labels while watching frames. Progress is saved immediately and resumed from CSV.")

    footer_line_h = 22
    footer_pad = 10
    footer_h = footer_pad * 2 + footer_line_h * len(control_lines)
    footer_y1 = max(12, canvas.shape[0] - footer_h - 12)
    footer_overlay = canvas.copy()
    cv2.rectangle(
        footer_overlay,
        (12, footer_y1),
        (canvas.shape[1] - 12, canvas.shape[0] - 12),
        (15, 15, 15),
        -1,
    )
    canvas = cv2.addWeighted(footer_overlay, 0.58, canvas, 0.42, 0.0)

    y = footer_y1 + footer_pad + 15
    for idx, text in enumerate(control_lines):
        color = (220, 220, 220) if idx == 0 else (200, 200, 200)
        if idx == len(control_lines) - 1 and show_help:
            color = (120, 200, 255)
        cv2.putText(
            canvas,
            text,
            (24, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            color,
            1,
            cv2.LINE_AA,
        )
        y += footer_line_h

    summary_w = min(420, max(290, int(canvas.shape[1] * 0.28)))
    summary_x2 = canvas.shape[1] - 16
    summary_x1 = max(16, summary_x2 - summary_w)
    summary_y1 = 16
    summary_y2 = summary_y1 + 264
    summary_overlay = canvas.copy()
    cv2.rectangle(summary_overlay, (summary_x1, summary_y1), (summary_x2, summary_y2), (20, 20, 20), -1)
    canvas = cv2.addWeighted(summary_overlay, 0.72, canvas, 0.28, 0.0)

    cv2.putText(
        canvas,
        "Current labels",
        (summary_x1 + 14, summary_y1 + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    summary_items = [
        ("scene_type", scene_type),
        ("symmetry_repetition", symmetry_repetition),
        ("visible_density", visible_density),
        ("class_repetition", class_repetition),
        ("mask_errors", mask_errors),
        ("camera_distance", camera_distance),
    ]
    y = summary_y1 + 62
    for field_name, value in summary_items:
        is_active = field_name == active_field
        title = FIELD_TITLES.get(field_name, field_name)
        shown_value = "-" if not value else str(value)
        title_color = (120, 200, 255) if is_active else (220, 220, 220)
        value_color = (100, 220, 100) if value else (180, 180, 180)
        if is_active and not value:
            value_color = (120, 200, 255)
        cv2.putText(
            canvas,
            f"{title}:",
            (summary_x1 + 14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            title_color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            shown_value,
            (summary_x1 + 168, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            value_color,
            2,
            cv2.LINE_AA,
        )
        y += 32

    status_text = "Ready to save" if ready_to_save else "Pending labels"
    status_color = (100, 220, 100) if ready_to_save else (80, 180, 255)
    cv2.putText(
        canvas,
        status_text,
        (summary_x1 + 14, summary_y2 - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )
    return canvas


def read_frame_cached(scene: SceneEntry, frame_index: int, cache: dict[tuple[str, int], Any]):
    key = (scene.scene_id, int(frame_index))
    frame = cache.get(key, None)
    if frame is not None:
        return frame
    frame = read_bgr(scene.frame_paths[frame_index])
    if frame is None:
        raise RuntimeError(f"No se pudo leer frame: {scene.frame_paths[frame_index]}")
    cache[key] = frame
    if len(cache) > 12:
        oldest_key = next(iter(cache.keys()))
        cache.pop(oldest_key, None)
    return frame


def make_annotation_row(
    *,
    scene: SceneEntry,
    scene_type: str,
    symmetry_repetition: str,
    visible_density: str,
    class_repetition: str,
    mask_errors: str,
    camera_distance: str,
) -> dict[str, str]:
    return {
        "scene_id": str(scene.scene_id),
        "scene_type": str(scene_type),
        "symmetry_repetition": str(symmetry_repetition),
        "visible_density": str(visible_density),
        "class_repetition": str(class_repetition),
        "mask_errors": str(mask_errors),
        "camera_distance": str(camera_distance),
        "n_frames": str(len(scene.frame_paths)),
        "frames_dir": str(scene.frames_dir),
        "annotated_at": datetime.now().isoformat(timespec="seconds"),
    }


def row_has_required_labels(row: dict[str, str] | None) -> bool:
    if row is None:
        return False
    required_fields = (
        "scene_type",
        "symmetry_repetition",
        "visible_density",
        "class_repetition",
        "mask_errors",
        "camera_distance",
    )
    return all(str(row.get(field, "")).strip() for field in required_fields)


def print_startup_help(output_csv: Path, total_scenes: int, pending_scenes: int) -> None:
    print(f"Output CSV: {output_csv}")
    print(f"Scenes found: {total_scenes} | Pending: {pending_scenes}")
    print("Controles:")
    print("  Left/Right o A/D: mover frame")
    print("  Space: autoplay on/off")
    print("  G: show/hide DAVIS mask overlay")
    print("  T + H/O/I/M: scene_type = home/office/industrial/mixed")
    print("  R + L/M/H: symmetry_repetition = low/medium/high")
    print("  V + L/M/H: visible_density = low/medium/high")
    print("  C + L/M/H: class_repetition = low/medium/high")
    print("  E + L/M/H: mask_errors = none/some/many")
    print("  P + N/M/F: camera_distance = near/medium/far")
    print("  Enter or S: save labels and move to the next scene")
    print("  B: go back to the previous scene")
    print("  U: limpiar labels actuales")
    print("  ?: mostrar/ocultar ayuda extendida")
    print("  Q o Esc: salir")


def default_active_field(
    *,
    scene_type: str | None,
    symmetry_repetition: str | None,
    visible_density: str | None,
    class_repetition: str | None,
    mask_errors: str | None,
    camera_distance: str | None,
) -> str:
    if not scene_type:
        return "scene_type"
    if not symmetry_repetition:
        return "symmetry_repetition"
    if not visible_density:
        return "visible_density"
    if not class_repetition:
        return "class_repetition"
    if not mask_errors:
        return "mask_errors"
    return "camera_distance"


def main() -> int:
    output_csv = Path(os.environ.get("APP2_SCENE_LABELS_CSV", str(DEFAULT_OUTPUT_CSV))).expanduser().resolve()
    autoplay_fps = float(os.environ.get("APP2_SCENE_LABELS_FPS", str(DEFAULT_AUTOPLAY_FPS)))
    max_w = int(os.environ.get("APP2_SCENE_LABELS_MAX_W", str(DEFAULT_DISPLAY_MAX_W)))
    max_h = int(os.environ.get("APP2_SCENE_LABELS_MAX_H", str(DEFAULT_DISPLAY_MAX_H)))

    scenes = build_scene_entries()
    if not scenes:
        print("No scenes found to annotate.", file=sys.stderr)
        return 1

    saved_rows = load_existing_labels(output_csv)
    pending_indices = [
        idx
        for idx, scene in enumerate(scenes)
        if not row_has_required_labels(saved_rows.get(scene.scene_id, None))
    ]
    if not pending_indices:
        print(f"All scenes are already annotated in {output_csv}")
        return 0

    print_startup_help(output_csv=output_csv, total_scenes=len(scenes), pending_scenes=len(pending_indices))

    scene_index = int(pending_indices[0])
    frame_index = 0
    autoplay = False
    show_masks = True
    show_help = True
    cache: dict[tuple[str, int], any] = {}
    gt_cache: dict[tuple[str, int], dict[int, Any]] = {}
    gt_loader_by_scene: dict[str, DavisGroundTruthLoader] = {}

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        scene = scenes[scene_index]
        previous_row = saved_rows.get(scene.scene_id, None)
        scene_type = None if previous_row is None else str(previous_row.get("scene_type", "")).strip() or None
        symmetry_repetition = None if previous_row is None else str(previous_row.get("symmetry_repetition", "")).strip() or None
        visible_density = None if previous_row is None else str(previous_row.get("visible_density", "")).strip() or None
        class_repetition = None if previous_row is None else str(previous_row.get("class_repetition", "")).strip() or None
        mask_errors = None if previous_row is None else str(previous_row.get("mask_errors", "")).strip() or None
        camera_distance = None if previous_row is None else str(previous_row.get("camera_distance", "")).strip() or None
        active_field = default_active_field(
            scene_type=scene_type,
            symmetry_repetition=symmetry_repetition,
            visible_density=visible_density,
            class_repetition=class_repetition,
            mask_errors=mask_errors,
            camera_distance=camera_distance,
        )

        while True:
            frame_index = max(0, min(frame_index, len(scene.frame_paths) - 1))
            base_frame = read_frame_cached(scene=scene, frame_index=frame_index, cache=cache)
            gt_objects: dict[int, Any] = {}
            gt_key = (scene.scene_id, int(frame_index))
            if gt_key in gt_cache:
                gt_objects = gt_cache[gt_key]
            else:
                gt_loader = gt_loader_by_scene.get(scene.scene_id, None)
                if gt_loader is None:
                    gt_loader = build_gt_loader(scene)
                    gt_loader_by_scene[scene.scene_id] = gt_loader
                frame_id = int(scene.frame_ids[frame_index])
                gt_objects = gt_loader.load_frame(frame_id=frame_id, target_shape=base_frame.shape[:2])
                gt_cache[gt_key] = gt_objects

            composed_frame = overlay_gt_objects(base_frame, gt_objects) if show_masks else base_frame
            display = fit_for_display(composed_frame, max_w=max_w, max_h=max_h)
            composed = draw_overlay(
                display,
                scene_index=scene_index,
                n_scenes=len(scenes),
                scene=scene,
                frame_index=frame_index,
                autoplay=autoplay,
                autoplay_fps=autoplay_fps,
                scene_type=scene_type,
                symmetry_repetition=symmetry_repetition,
                visible_density=visible_density,
                class_repetition=class_repetition,
                mask_errors=mask_errors,
                camera_distance=camera_distance,
                n_gt_visible=len(gt_objects),
                show_masks=show_masks,
                active_field=active_field,
                show_help=show_help,
            )
            cv2.imshow(WINDOW_NAME, composed)

            delay_ms = max(1, int(round(1000.0 / max(0.1, autoplay_fps)))) if autoplay else 0
            key = cv2.waitKeyEx(delay_ms)
            action = decode_action(key)

            if autoplay and key == -1:
                frame_index = min(frame_index + 1, len(scene.frame_paths) - 1)
                continue

            if action == "quit":
                cv2.destroyAllWindows()
                return 0
            if action == "toggle_auto":
                autoplay = not autoplay
                continue
            if action == "left":
                autoplay = False
                frame_index = max(0, frame_index - 1)
                continue
            if action == "right":
                autoplay = False
                frame_index = min(len(scene.frame_paths) - 1, frame_index + 1)
                continue

            if key in HELP_KEYS:
                show_help = not show_help
                continue
            if key in TOGGLE_MASKS_KEYS:
                show_masks = not show_masks
                continue
            if key in TARGET_SCENE_TYPE_KEYS:
                active_field = "scene_type"
                continue
            if key in TARGET_SYMMETRY_KEYS:
                active_field = "symmetry_repetition"
                continue
            if key in TARGET_DENSITY_KEYS:
                active_field = "visible_density"
                continue
            if key in TARGET_CLASS_REPETITION_KEYS:
                active_field = "class_repetition"
                continue
            if key in TARGET_OCCLUSION_ERRORS_KEYS:
                active_field = "mask_errors"
                continue
            if key in TARGET_CAMERA_DISTANCE_KEYS:
                active_field = "camera_distance"
                continue
            if key in CLEAR_LABEL_KEYS:
                scene_type = None
                symmetry_repetition = None
                visible_density = None
                class_repetition = None
                mask_errors = None
                camera_distance = None
                active_field = "scene_type"
                continue
            if active_field == "scene_type" and key in SCENE_TYPE_OPTIONS:
                scene_type = str(SCENE_TYPE_OPTIONS[key])
                active_field = default_active_field(
                    scene_type=scene_type,
                    symmetry_repetition=symmetry_repetition,
                    visible_density=visible_density,
                    class_repetition=class_repetition,
                    mask_errors=mask_errors,
                    camera_distance=camera_distance,
                )
                continue
            if active_field == "symmetry_repetition" and key in LEVEL_OPTIONS:
                symmetry_repetition = str(LEVEL_OPTIONS[key])
                active_field = default_active_field(
                    scene_type=scene_type,
                    symmetry_repetition=symmetry_repetition,
                    visible_density=visible_density,
                    class_repetition=class_repetition,
                    mask_errors=mask_errors,
                    camera_distance=camera_distance,
                )
                continue
            if active_field == "visible_density" and key in LEVEL_OPTIONS:
                visible_density = str(LEVEL_OPTIONS[key])
                active_field = default_active_field(
                    scene_type=scene_type,
                    symmetry_repetition=symmetry_repetition,
                    visible_density=visible_density,
                    class_repetition=class_repetition,
                    mask_errors=mask_errors,
                    camera_distance=camera_distance,
                )
                continue
            if active_field == "class_repetition" and key in LEVEL_OPTIONS:
                class_repetition = str(LEVEL_OPTIONS[key])
                active_field = default_active_field(
                    scene_type=scene_type,
                    symmetry_repetition=symmetry_repetition,
                    visible_density=visible_density,
                    class_repetition=class_repetition,
                    mask_errors=mask_errors,
                    camera_distance=camera_distance,
                )
                continue
            if active_field == "mask_errors" and key in OCCLUSION_ERROR_OPTIONS:
                mask_errors = str(OCCLUSION_ERROR_OPTIONS[key])
                active_field = default_active_field(
                    scene_type=scene_type,
                    symmetry_repetition=symmetry_repetition,
                    visible_density=visible_density,
                    class_repetition=class_repetition,
                    mask_errors=mask_errors,
                    camera_distance=camera_distance,
                )
                continue
            if active_field == "camera_distance" and key in CAMERA_DISTANCE_OPTIONS:
                camera_distance = str(CAMERA_DISTANCE_OPTIONS[key])
                active_field = default_active_field(
                    scene_type=scene_type,
                    symmetry_repetition=symmetry_repetition,
                    visible_density=visible_density,
                    class_repetition=class_repetition,
                    mask_errors=mask_errors,
                    camera_distance=camera_distance,
                )
                continue

            if key in SAVE_KEYS:
                if not (
                    scene_type
                    and symmetry_repetition
                    and visible_density
                    and class_repetition
                    and mask_errors
                    and camera_distance
                ):
                    print(
                        f"[{scene.scene_id}] Missing labels: "
                        f"scene_type={scene_type}, symmetry_repetition={symmetry_repetition}, "
                        f"visible_density={visible_density}, class_repetition={class_repetition}, "
                        f"mask_errors={mask_errors}, camera_distance={camera_distance}"
                    )
                    continue
                saved_rows[scene.scene_id] = make_annotation_row(
                    scene=scene,
                    scene_type=scene_type,
                    symmetry_repetition=symmetry_repetition,
                    visible_density=visible_density,
                    class_repetition=class_repetition,
                    mask_errors=mask_errors,
                    camera_distance=camera_distance,
                )
                write_labels_csv(output_csv, saved_rows)
                print(
                    f"[saved] {scene.scene_id}: "
                    f"{scene_type}, {symmetry_repetition}, {visible_density}, "
                    f"{class_repetition}, {mask_errors}, {camera_distance}"
                )

                next_index = None
                for idx in range(scene_index + 1, len(scenes)):
                    if not row_has_required_labels(saved_rows.get(scenes[idx].scene_id, None)):
                        next_index = idx
                        break
                if next_index is None:
                    cv2.destroyAllWindows()
                    print(f"Anotacion completada. CSV guardado en {output_csv}")
                    return 0
                scene_index = int(next_index)
                frame_index = 0
                autoplay = False
                break

            if key in NEXT_SCENE_KEYS:
                next_index = None
                for idx in range(scene_index + 1, len(scenes)):
                    if not row_has_required_labels(saved_rows.get(scenes[idx].scene_id, None)):
                        next_index = idx
                        break
                if next_index is not None:
                    scene_index = int(next_index)
                    frame_index = 0
                    autoplay = False
                    break
                continue

            if key in PREV_SCENE_KEYS:
                prev_index = None
                for idx in range(scene_index - 1, -1, -1):
                    prev_index = idx
                    break
                if prev_index is not None:
                    scene_index = int(prev_index)
                    frame_index = 0
                    autoplay = False
                    break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
