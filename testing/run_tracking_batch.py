from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config.config_loader import Config
from pipeline.initialization import initialize_system
from pipeline.reid_pipeline import ReIDPipeline
from davis_gt import DavisGroundTruthLoader
from reporting import build_console_report, fmt_pct, render_table, write_csv, write_text
from tracking_metrics import TrackingEvaluator, build_memory_summary, safe_pct
from run_tracking_test import (
    build_event_rows,
    build_runtime_memory_telemetry,
    build_det_to_object_id,
    capture_cuda_memory_stats,
    make_process_handle,
    read_process_rss_bytes,
    reset_cuda_peak_memory_stats,
    resolve_aligned_shape,
    resolve_frame_files_for_testing,
    sanitize_name_for_path,
)
from utils.io import read_bgr, parse_frame_id
from utils.scannetpp_tar import (
    resolve_prepared_scene_from_tar,
    resolve_scene_tar_path,
    resolve_scannetpp_data_parent,
)


PROJECT_DIR = Path(SRC_DIR).resolve().parent


SCENE_TABLE_FILES = {
    "per_scene": "scene_summary.csv",
    "per_class": "per_class.csv",
    "per_object": "per_object.csv",
    "per_case": "per_case.csv",
    "per_case_modules": "per_case_modules.csv",
    "per_frame": "per_frame.csv",
    "per_pred_track": "per_pred_track.csv",
    "per_event": "per_event.csv",
}


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for raw in values:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def read_scene_ids_from_file(path: str) -> list[str]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Scenes file not found: {p}")
    lines = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        text = str(raw).strip()
        if not text or text.startswith("#"):
            continue
        if "," in text:
            text = text.split(",", 1)[0].strip()
        lines.append(text)
    return unique_preserve_order(lines)


def collect_scene_input_issues(
    *,
    scene_id: str,
    masks_root_base: Path,
    images_root_base: Path,
    mask_variant: str,
    image_subdir: str,
    masks_subdir: str = "2Dmasks",
) -> list[str]:
    if mask_variant == "benchmark":
        mask_variant = "benchmark_instance"

    masks_root = (masks_root_base / masks_subdir / scene_id).resolve()
    meta_path = (masks_root / f"meta_{mask_variant}.json").resolve()
    annotations_dir = (masks_root / "annotations" / mask_variant).resolve()
    frames_dir = (images_root_base / "data" / scene_id / image_subdir).resolve()

    issues: list[str] = []
    if not frames_dir.is_dir():
        issues.append(f"missing_frames_dir:{frames_dir}")
    if not meta_path.is_file():
        issues.append(f"missing_meta:{meta_path}")
    if not annotations_dir.is_dir():
        issues.append(f"missing_annotations_dir:{annotations_dir}")
    if not issues:
        return issues

    try:
        tar_source = resolve_prepared_scene_from_tar(
            project_dir=PROJECT_DIR,
            images_root_base=images_root_base,
            scene_id=scene_id,
            mask_variant=mask_variant,
            image_subdir=image_subdir,
        )
    except FileNotFoundError as exc:
        tar_path = resolve_scene_tar_path(images_root_base=images_root_base, scene_id=scene_id)
        if tar_path is not None:
            return [str(exc)]
        return issues

    if tar_source is not None:
        return []
    return issues


def discover_tar_scene_ids(*, images_root_base: Path) -> list[str]:
    data_parent = resolve_scannetpp_data_parent(images_root_base)
    if data_parent is None or not data_parent.is_dir():
        return []
    return sorted(
        tar_path.stem
        for tar_path in data_parent.glob("*.tar")
        if tar_path.is_file()
    )


def load_custom_scene_definitions(path: str) -> list[dict[str, str]]:
    """Load a JSON file listing custom scene definitions.

    Each entry must be an object with at least:
      - scene_id          (str)
      - frames_dir        (str, path to images)
      - davis_meta_path   (str, path to meta JSON)
      - davis_annotations_dir (str, path to annotations dir)

    Example::

        [
          {
            "scene_id": "my_scene",
            "frames_dir": "/data/my_scene/frames",
            "davis_meta_path": "/data/my_scene/meta.json",
            "davis_annotations_dir": "/data/my_scene/annotations"
          }
        ]
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Scenes definition file not found: {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Scenes definition file must be a JSON array: {p}")
    defs: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {i} in scenes def file is not a dict: {item!r}")
        scene_id = str(item.get("scene_id", "") or "").strip()
        if not scene_id:
            raise ValueError(f"Entry {i} in scenes def file is missing 'scene_id'")
        defs.append(
            {
                "scene_id": scene_id,
                "frames_dir": str(item.get("frames_dir", "") or "").strip(),
                "davis_meta_path": str(item.get("davis_meta_path", "") or "").strip(),
                "davis_annotations_dir": str(item.get("davis_annotations_dir", "") or "").strip(),
            }
        )
    return defs


def build_custom_scene_input_source(scene_def: dict[str, str]) -> dict[str, str]:
    """Build an input_source dict from a custom scene definition, with validation."""
    scene_id = str(scene_def.get("scene_id", "") or "").strip()
    frames_dir = str(scene_def.get("frames_dir", "") or "").strip()
    davis_meta_path = str(scene_def.get("davis_meta_path", "") or "").strip()
    davis_annotations_dir = str(scene_def.get("davis_annotations_dir", "") or "").strip()

    issues: list[str] = []
    if not frames_dir or not Path(frames_dir).expanduser().is_dir():
        issues.append(f"missing_frames_dir:{frames_dir}")
    if not davis_meta_path or not Path(davis_meta_path).expanduser().is_file():
        issues.append(f"missing_meta:{davis_meta_path}")
    if not davis_annotations_dir or not Path(davis_annotations_dir).expanduser().is_dir():
        issues.append(f"missing_annotations_dir:{davis_annotations_dir}")
    if issues:
        raise FileNotFoundError(
            f"Custom scene '{scene_id}' has missing inputs: {', '.join(issues)}"
        )

    return {
        "mode": "custom_davis_def",
        "frames_dir": str(Path(frames_dir).expanduser().resolve()),
        "sequence_name": scene_id,
        "davis_meta_path": str(Path(davis_meta_path).expanduser().resolve()),
        "davis_annotations_dir": str(Path(davis_annotations_dir).expanduser().resolve()),
        "image_subdir": "",
    }


def discover_scene_ids(
    *,
    masks_root_base: Path,
    images_root_base: Path,
    mask_variant: str,
    image_subdir: str,
    masks_subdir: str = "2Dmasks",
) -> list[str]:
    masks_parent = (masks_root_base / masks_subdir).resolve()
    candidate_scene_ids: list[str]
    if masks_parent.is_dir():
        candidate_scene_ids = [
            str(child.name)
            for child in sorted(masks_parent.iterdir())
            if child.is_dir()
        ]
        if not candidate_scene_ids:
            # masks_parent exists but contains no subdirectories (e.g. only .tar files)
            candidate_scene_ids = discover_tar_scene_ids(images_root_base=images_root_base)
    else:
        candidate_scene_ids = discover_tar_scene_ids(images_root_base=images_root_base)
    if not candidate_scene_ids:
        return []

    scene_ids: list[str] = []
    skipped_examples: list[str] = []
    skipped_count = 0
    for scene_id in candidate_scene_ids:
        issues = collect_scene_input_issues(
            scene_id=scene_id,
            masks_root_base=masks_root_base,
            images_root_base=images_root_base,
            mask_variant=mask_variant,
            image_subdir=image_subdir,
            masks_subdir=masks_subdir,
        )
        if not issues:
            scene_ids.append(scene_id)
            continue
        skipped_count += 1
        if len(skipped_examples) < 5:
            skipped_examples.append(f"{scene_id} ({', '.join(issues)})")
    if skipped_count > 0:
        suffix = ""
        if skipped_examples:
            suffix = f" | ejemplos: {'; '.join(skipped_examples)}"
        print(
            f"[BATCH] Scene discovery skipped {skipped_count} escenas sin inputs listos "
            f"(variant={mask_variant}, image_subdir={image_subdir}){suffix}"
        )
    return unique_preserve_order(scene_ids)


def resolve_batch_scene_ids(
    *,
    masks_root_base: Path | None,
    images_root_base: Path | None,
    mask_variant: str,
    image_subdir: str,
    masks_subdir: str = "2Dmasks",
    custom_scene_defs: list[dict[str, str]] | None = None,
    scenes: str = "",
    scenes_file: str = "",
    scene_id: str = "",
) -> list[str]:
    # Custom DAVIS/generic mode: scene list is fully determined by the def file.
    if custom_scene_defs is not None:
        return unique_preserve_order([str(d["scene_id"]) for d in custom_scene_defs])

    if scenes_file:
        return read_scene_ids_from_file(scenes_file)
    if scenes:
        parts = re.split(r"[\s,;]+", scenes)
        return unique_preserve_order(parts)
    if scene_id:
        return [scene_id]

    if masks_root_base is None or images_root_base is None:
        raise RuntimeError(
            "Cannot auto-discover scenes: set --scenes-def-file for custom scenes, "
            "or set --masks-root / --images-root for ScanNet++ mode."
        )
    return discover_scene_ids(
        masks_root_base=masks_root_base,
        images_root_base=images_root_base,
        mask_variant=mask_variant,
        image_subdir=image_subdir,
        masks_subdir=masks_subdir,
    )


def build_scene_input_source(
    *,
    scene_id: str,
    masks_root_base: Path,
    images_root_base: Path,
    mask_variant: str,
    image_subdir: str,
    masks_subdir: str = "2Dmasks",
) -> dict[str, str]:
    if mask_variant == "benchmark":
        mask_variant = "benchmark_instance"

    masks_root = (masks_root_base / masks_subdir / scene_id).resolve()
    meta_path = (masks_root / f"meta_{mask_variant}.json").resolve()
    annotations_dir = (masks_root / "annotations" / mask_variant).resolve()
    frames_dir = (images_root_base / "data" / scene_id / image_subdir).resolve()

    # If all pre-extracted paths exist, use them directly.
    if frames_dir.is_dir() and meta_path.is_file() and annotations_dir.is_dir():
        return {
            "mode": "external_scannetpp_batch",
            "frames_dir": str(frames_dir),
            "sequence_name": str(scene_id),
            "davis_meta_path": str(meta_path),
            "davis_annotations_dir": str(annotations_dir),
            "image_subdir": str(image_subdir),
        }

    # Otherwise try tar resolution (extracts to project cache).
    tar_source = resolve_prepared_scene_from_tar(
        project_dir=PROJECT_DIR,
        images_root_base=images_root_base,
        scene_id=scene_id,
        mask_variant=mask_variant,
        image_subdir=image_subdir,
    )
    if tar_source is not None:
        return tar_source

    issues: list[str] = []
    if not frames_dir.is_dir():
        issues.append(f"missing_frames_dir:{frames_dir}")
    if not meta_path.is_file():
        issues.append(f"missing_meta:{meta_path}")
    if not annotations_dir.is_dir():
        issues.append(f"missing_annotations_dir:{annotations_dir}")
    raise FileNotFoundError(
        f"Escena {scene_id} sin inputs de testing preparados: {', '.join(issues)}"
    )


def coerce_csv_value(raw: str):
    text = "" if raw is None else str(raw).strip()
    if text == "":
        return None
    low = text.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        if "." not in text and "e" not in low:
            return int(text)
        return float(text)
    except ValueError:
        pass
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("{") and text.endswith("}")):
        try:
            return ast.literal_eval(text)
        except Exception:
            return text
    return text


def read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{str(k): coerce_csv_value(v) for k, v in row.items()} for row in reader]


def write_single_row_csv(path: Path, row: dict) -> None:
    write_csv(path, [dict(row)])


def scene_dir_is_complete(scene_dir: Path) -> bool:
    return all((scene_dir / filename).is_file() for filename in SCENE_TABLE_FILES.values())


def iter_scene_dirs(batch_dir: Path) -> list[Path]:
    scenes_root = batch_dir / "scenes"
    if not scenes_root.is_dir():
        return []
    return sorted(
        child
        for child in scenes_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )


def read_completed_scene_rows_from_dirs(batch_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for scene_dir in iter_scene_dirs(batch_dir):
        if not scene_dir_is_complete(scene_dir):
            continue
        rows.extend(read_csv_rows(scene_dir / SCENE_TABLE_FILES["per_scene"]))
    return rows


def reserve_incomplete_scene_backup_dir(scene_dir: Path) -> Path:
    parent = scene_dir.parent
    base_name = f".incomplete_{scene_dir.name}"
    candidate = parent / base_name
    suffix = 1
    while candidate.exists():
        candidate = parent / f"{base_name}_{suffix:02d}"
        suffix += 1
    return candidate


def read_manifest_rows(batch_dir: Path) -> list[dict]:
    return read_csv_rows(batch_dir / "manifest.csv")


def read_batch_per_scene_rows(batch_dir: Path) -> list[dict]:
    return read_csv_rows(batch_dir / "per_scene.csv")


def resolve_scene_schedule(
    *,
    candidate_scene_ids: list[str],
    batch_dir: Path,
    batch_size: int | None,
) -> tuple[list[str], list[str], str, list[dict], list[dict]]:
    existing_manifest_rows = read_manifest_rows(batch_dir)
    existing_per_scene_rows = read_completed_scene_rows_from_dirs(batch_dir)
    existing_registered_scene_ids = unique_preserve_order(
        [str(row.get("scene_id", "") or "").strip() for row in existing_manifest_rows]
    )
    completed_scene_ids = {
        str(row.get("scene_id", "") or "").strip()
        for row in existing_per_scene_rows
        if str(row.get("scene_id", "") or "").strip()
    }

    pending_scene_ids = [
        scene_id
        for scene_id in candidate_scene_ids
        if scene_id not in completed_scene_ids
    ]
    selected_scene_ids = list(pending_scene_ids)
    if batch_size is not None:
        selected_scene_ids = selected_scene_ids[: max(0, int(batch_size))]
    selection_mode = "automatic_progress"
    registered_scene_ids = unique_preserve_order(existing_registered_scene_ids + selected_scene_ids)
    return (
        selected_scene_ids,
        registered_scene_ids,
        selection_mode,
        existing_manifest_rows,
        existing_per_scene_rows,
    )


def merge_scene_name_index(
    *,
    base_scene_name_by_id: dict[str, str],
    manifest_rows: list[dict],
    per_scene_rows: list[dict],
) -> dict[str, str]:
    out = {str(k): str(v) for k, v in (base_scene_name_by_id or {}).items()}
    for row in list(manifest_rows or []) + list(per_scene_rows or []):
        scene_id = str(row.get("scene_id", "") or "").strip()
        scene_name = str(row.get("scene_name", "") or "").strip()
        if scene_id and scene_name:
            out[scene_id] = scene_name
    return out


def annotate_rows(rows: list[dict], *, run_id: str, scene_id: str, scene_name: str) -> list[dict]:
    out: list[dict] = []
    for row in rows or []:
        item = dict(row)
        item["run_id"] = str(run_id)
        item["scene_id"] = str(scene_id)
        item["scene_name"] = str(scene_name)
        out.append(item)
    return out


def build_global_per_class_rows(
    *,
    run_id: str,
    per_object_rows: list[dict],
    per_pred_track_rows: list[dict],
    per_case_rows: list[dict],
) -> list[dict]:
    gt_objects_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in per_object_rows:
        gt_objects_by_class[str(row.get("gt_class_name") or "unknown")].append(row)

    pred_tracks_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in per_pred_track_rows:
        pred_tracks_by_class[str(row.get("pred_class_name") or "unknown")].append(row)

    case_rows_by_class: dict[str, list[dict]] = defaultdict(list)
    reopened_rows_by_class: dict[str, list[dict]] = defaultdict(list)
    for row in per_case_rows:
        class_name = str(row.get("gt_class_name") or "unknown")
        case_rows_by_class[class_name].append(row)
        if str(row.get("real_state")) == "existing" and str(row.get("collapsed_kind")) == "new":
            reopened_rows_by_class[class_name].append(row)

    out: list[dict] = []
    for class_name in sorted(set(gt_objects_by_class.keys()) | set(pred_tracks_by_class.keys())):
        class_gt_rows = list(gt_objects_by_class.get(str(class_name), []))
        class_pred_rows = list(pred_tracks_by_class.get(str(class_name), []))
        class_case_rows = list(case_rows_by_class.get(str(class_name), []))
        class_reopen_rows = list(reopened_rows_by_class.get(str(class_name), []))

        n_gt_class = int(len(class_gt_rows))
        total_frames_class = int(sum(int(row.get("n_frames", 0) or 0) for row in class_gt_rows))
        duplicate_id_frame_count = int(sum(int(row.get("duplicate_id_frame_count", 0) or 0) for row in class_gt_rows))
        foreign_id_frame_count = int(sum(int(row.get("foreign_id_frame_count", 0) or 0) for row in class_gt_rows))
        recovery_attempts_total = int(sum(int(row.get("recovery_attempts", 0) or 0) for row in class_gt_rows))
        recovery_success_reference_total = int(
            sum(int(row.get("recovery_success_reference", 0) or 0) for row in class_gt_rows)
        )
        recovery_success_own_identity_total = int(
            sum(int(row.get("recovery_success_own_identity", 0) or 0) for row in class_gt_rows)
        )
        recovery_success_duplicate_id_total = int(
            sum(int(row.get("recovery_success_duplicate_id", 0) or 0) for row in class_gt_rows)
        )
        recovery_success_foreign_id_total = int(
            sum(int(row.get("recovery_success_foreign_id", 0) or 0) for row in class_gt_rows)
        )
        n_class_cases = int(len(class_case_rows))
        n_class_matches = int(
            sum(
                1
                for row in class_case_rows
                if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0
            )
        )
        n_class_correct = int(sum(1 for row in class_case_rows if bool(row.get("collapsed_global_correct", False))))

        out.append(
            {
                "class_name": str(class_name),
                "n_gt_objects": n_gt_class,
                "n_real_pred_tracks": int(len(class_pred_rows)),
                "pred_track_surplus_vs_gt": int(len(class_pred_rows) - n_gt_class),
                "pred_track_inflation_factor": (float(len(class_pred_rows)) / float(n_gt_class) if n_gt_class > 0 else None),
                "weighted_strict_accuracy": (
                    float(
                        sum(
                            float(row.get("strict_accuracy", 0.0) or 0.0) * float(row.get("n_frames", 0) or 0)
                            for row in class_gt_rows
                        )
                    )
                    / float(total_frames_class)
                    if total_frames_class > 0
                    else None
                ),
                "weighted_permissive_accuracy": (
                    float(
                        sum(
                            float(row.get("permissive_accuracy", 0.0) or 0.0) * float(row.get("n_frames", 0) or 0)
                            for row in class_gt_rows
                        )
                    )
                    / float(total_frames_class)
                    if total_frames_class > 0
                    else None
                ),
                "mean_pred_ids_per_gt": (
                    float(sum(float(row.get("n_unique_pred_ids", 0) or 0) for row in class_gt_rows)) / float(len(class_gt_rows))
                    if class_gt_rows
                    else None
                ),
                "mean_id_changes_per_gt": (
                    float(sum(float(row.get("id_changes", 0) or 0) for row in class_gt_rows)) / float(len(class_gt_rows))
                    if class_gt_rows
                    else None
                ),
                "duplicate_id_segment_count": int(
                    sum(int(row.get("duplicate_id_segment_count", 0) or 0) for row in class_gt_rows)
                ),
                "foreign_id_segment_count": int(
                    sum(int(row.get("foreign_id_segment_count", 0) or 0) for row in class_gt_rows)
                ),
                "duplicate_id_frame_count": duplicate_id_frame_count,
                "foreign_id_frame_count": foreign_id_frame_count,
                "duplicate_id_rate_visible": safe_pct(duplicate_id_frame_count, n_class_cases),
                "foreign_id_rate_visible": safe_pct(foreign_id_frame_count, n_class_cases),
                "recovery_attempts_total": recovery_attempts_total,
                "recovery_success_reference_total": recovery_success_reference_total,
                "recovery_success_own_identity_total": recovery_success_own_identity_total,
                "recovery_success_duplicate_id_total": recovery_success_duplicate_id_total,
                "recovery_success_foreign_id_total": recovery_success_foreign_id_total,
                "recovery_rate_reference": safe_pct(recovery_success_reference_total, recovery_attempts_total),
                "recovery_rate_own_identity": safe_pct(recovery_success_own_identity_total, recovery_attempts_total),
                "recovery_rate_duplicate_id": safe_pct(recovery_success_duplicate_id_total, recovery_attempts_total),
                "recovery_rate_foreign_id": safe_pct(recovery_success_foreign_id_total, recovery_attempts_total),
                "tracking_recall": safe_pct(n_class_correct, n_class_cases),
                "mean_tracking_iou": (
                    float(sum(float(row.get("tracking_iou", 0.0) or 0.0) for row in class_case_rows)) / float(n_class_cases)
                    if n_class_cases > 0
                    else None
                ),
                "deta": safe_pct(n_class_matches, n_class_cases),
                "assa": safe_pct(n_class_correct, n_class_matches),
                "hota": (
                    float((float(safe_pct(n_class_matches, n_class_cases) or 0.0) * float(safe_pct(n_class_correct, n_class_matches) or 0.0)) ** 0.5)
                    if n_class_cases > 0
                    else None
                ),
                "accuracy_existing_vs_new_collapsed": safe_pct(
                    sum(1 for row in class_case_rows if bool(row.get("collapsed_existing_vs_new_correct", False))),
                    n_class_cases,
                ),
                "accuracy_parent_collapsed": safe_pct(
                    sum(
                        1
                        for row in class_case_rows
                        if str(row.get("real_state")) == "existing" and bool(row.get("collapsed_parent_correct", False))
                    ),
                    sum(1 for row in class_case_rows if str(row.get("real_state")) == "existing"),
                ),
                "new_detection_accuracy_collapsed": safe_pct(
                    sum(
                        1
                        for row in class_case_rows
                        if str(row.get("real_state")) == "new" and str(row.get("collapsed_kind")) == "new"
                    ),
                    sum(1 for row in class_case_rows if str(row.get("real_state")) == "new"),
                ),
                "uncertain_rate": safe_pct(
                    sum(
                        1
                        for row in class_case_rows
                        if str(row.get("final_decision")) in {"AMBIGUOUS_TRACK", "PROVISIONAL_PARENT", "PROVISIONAL_NEW"}
                    ),
                    n_class_cases,
                ),
                "hypothesis_recall_uncertain": safe_pct(
                    sum(1 for row in class_case_rows if bool(row.get("ambiguous_set_hit", False)))
                    + sum(1 for row in class_case_rows if bool(row.get("provisional_parent_hit", False)))
                    + sum(
                        1
                        for row in class_case_rows
                        if str(row.get("final_decision")) == "PROVISIONAL_NEW" and str(row.get("real_state")) == "new"
                    ),
                    sum(
                        1
                        for row in class_case_rows
                        if str(row.get("final_decision")) in {"AMBIGUOUS_TRACK", "PROVISIONAL_PARENT", "PROVISIONAL_NEW"}
                    ),
                ),
                "reopen_rate_existing": safe_pct(
                    len(class_reopen_rows),
                    sum(1 for row in class_case_rows if str(row.get("real_state")) == "existing"),
                ),
                "distance_usage_rate": safe_pct(
                    sum(1 for row in class_case_rows if bool(row.get("distance_used", False))),
                    n_class_cases,
                ),
                "distance_disambiguation_accuracy": safe_pct(
                    sum(1 for row in class_case_rows if bool(row.get("distance_correct", False))),
                    sum(1 for row in class_case_rows if bool(row.get("distance_resolved", False))),
                ),
                "context_intervention_rate": safe_pct(
                    sum(1 for row in class_case_rows if bool(row.get("context_intervened", False))),
                    n_class_cases,
                ),
                "context_intervention_accuracy": safe_pct(
                    sum(1 for row in class_case_rows if bool(row.get("context_change_correct", False))),
                    sum(1 for row in class_case_rows if bool(row.get("context_intervened", False))),
                ),
                "gt_objects_with_foreign_id_use": int(
                    sum(1 for row in class_gt_rows if int(row.get("n_foreign_pred_ids", 0) or 0) > 0)
                ),
                "existing_gt_reopened_as_new_rows": int(len(class_reopen_rows)),
                "existing_gt_reopened_as_new_ids": int(
                    len({int(row["gt_instance_id"]) for row in class_reopen_rows if row.get("gt_instance_id", None) is not None})
                ),
                "run_id": str(run_id),
                "aggregate_scope": "global",
            }
        )

    return out


def normalize_per_case_rows(per_case_rows: list[dict]) -> list[dict]:
    normalized_rows: list[dict] = []
    for row in per_case_rows:
        item = dict(row)
        final_decision = str(item.get("final_decision", "") or "")
        real_state = str(item.get("real_state", "") or "")
        firm_global_correct = bool(item.get("firm_global_correct", False))
        item.setdefault("final_decision_is_existing_assignment", bool(final_decision == "MATCH"))
        item.setdefault("final_decision_is_new_assignment", bool(final_decision == "NEW"))
        item.setdefault(
            "final_decision_parent_correct",
            bool(final_decision == "MATCH" and real_state == "existing" and firm_global_correct),
        )
        item.setdefault("final_decision_global_correct", bool(firm_global_correct))
        normalized_rows.append(item)
    return normalized_rows


def build_run_config_row(
    *,
    run_id: str,
    batch_name: str,
    batch_dir: Path,
    masks_root_base: Path | None,
    images_root_base: Path | None,
    image_subdir: str,
    mask_variant: str,
    stable_min_frames: int,
    max_frames: int | None,
    max_scenes: int | None,
    batch_size: int | None,
    selection_mode: str,
    force_detector_backend: str,
    selected_scene_ids: list[str],
    registered_scene_ids: list[str],
    scenes_def_file: str | None = None,
) -> dict:
    return {
        "run_id": str(run_id),
        "batch_name": str(batch_name),
        "batch_dir": str(batch_dir.resolve()),
        "created_at": _now_iso(),
        "detector_backend": str(force_detector_backend),
        "stable_min_frames": int(stable_min_frames),
        "max_frames": None if max_frames is None else int(max_frames),
        "max_scenes": None if max_scenes is None else int(max_scenes),
        "batch_size": None if batch_size is None else int(batch_size),
        "selection_mode": str(selection_mode),
        "mask_variant": str(mask_variant),
        "image_subdir": str(image_subdir),
        "masks_root_base": None if masks_root_base is None else str(masks_root_base.resolve()),
        "images_root_base": None if images_root_base is None else str(images_root_base.resolve()),
        "scenes_def_file": scenes_def_file,
        "selected_scene_count": int(len(selected_scene_ids)),
        "selected_scene_ids": [str(scene_id) for scene_id in selected_scene_ids],
        "registered_scene_count": int(len(registered_scene_ids)),
        "registered_scene_ids": [str(scene_id) for scene_id in registered_scene_ids],
    }


def build_scene_summary_row(
    *,
    run_id: str,
    scene_id: str,
    scene_name: str,
    results: dict,
    scene_started_at: str | None = None,
    scene_finished_at: str | None = None,
    output_dir: Path | None = None,
    stable_min_frames: int | None = None,
    max_frames: int | None = None,
    force_detector_backend: str | None = None,
    mask_variant: str | None = None,
    image_subdir: str | None = None,
) -> dict:
    summary = dict(results.get("summary", {}) or {})
    collapsed = dict(results.get("collapsed_metrics", {}) or {})
    collapsed_identity = dict(results.get("collapsed_identity_metrics", {}) or {})
    uncertainty = dict(results.get("uncertainty_metrics", {}) or {})
    per_case = list(results.get("per_case", []) or [])

    gt_area_values = [
        float(row["gt_area_frac"])
        for row in per_case
        if row.get("gt_area_frac", None) is not None
    ]
    total_distractors_sum = float(sum(float(row.get("n_total_distractors", 0) or 0) for row in per_case))
    same_class_distractors_sum = float(sum(float(row.get("n_same_class_distractors", 0) or 0) for row in per_case))
    tracking_iou_sum = float(sum(float(row.get("tracking_iou", 0.0) or 0.0) for row in per_case))
    n_cases = int(collapsed.get("n_cases", len(per_case)) or 0)
    n_frames = int(summary.get("n_frames", 0) or 0)
    n_gt_objects = int(summary.get("n_objects", 0) or 0)

    row = {
        "run_id": str(run_id),
        "scene_id": str(scene_id),
        "scene_name": str(scene_name),
        "status": "completed",
        "started_at": scene_started_at,
        "finished_at": scene_finished_at,
        "output_dir": None if output_dir is None else str(output_dir.resolve()),
        "detector_backend": force_detector_backend,
        "stable_min_frames": stable_min_frames,
        "max_frames": max_frames,
        "mask_variant": mask_variant,
        "image_subdir": image_subdir,
        "n_frames": int(n_frames),
        "n_gt_objects": int(n_gt_objects),
        "n_cases": int(n_cases),
        "n_visible_gt_observations": int(summary.get("n_visible_gt_observations", n_cases) or 0),
        "n_matched_gt_observations": int(summary.get("n_matched_gt_observations", 0) or 0),
        "mean_gt_per_frame": (float(summary.get("n_visible_gt_observations", n_cases) or 0) / float(n_frames)) if n_frames > 0 else None,
        "total_distractors_sum": float(total_distractors_sum),
        "same_class_distractors_sum": float(same_class_distractors_sum),
        "mean_total_distractors": (float(total_distractors_sum) / float(n_cases)) if n_cases > 0 else None,
        "mean_same_class_distractors": (float(same_class_distractors_sum) / float(n_cases)) if n_cases > 0 else None,
        "gt_area_frac_sum": float(sum(gt_area_values)),
        "gt_area_frac_count": int(len(gt_area_values)),
        "mean_gt_area_frac": (float(sum(gt_area_values)) / float(len(gt_area_values))) if gt_area_values else None,
        "n_new_gt": int(collapsed.get("n_new_gt", 0) or 0),
        "n_existing_gt": int(collapsed.get("n_existing_gt", 0) or 0),
        "new_object_rate": safe_pct(int(collapsed.get("n_new_gt", 0) or 0), n_cases),
        "idtp": int(collapsed_identity.get("idtp", 0) or 0),
        "idfp": int(collapsed_identity.get("idfp", 0) or 0),
        "idfn": int(collapsed_identity.get("idfn", 0) or 0),
        "idf1": collapsed_identity.get("idf1", None),
        "idp": collapsed_identity.get("idp", None),
        "idr": collapsed_identity.get("idr", None),
        "idsw": int(collapsed_identity.get("idsw", 0) or 0),
        "frag": int(collapsed_identity.get("frag", 0) or 0),
        "duplicate_id_segment_count": int(summary.get("duplicate_id_segment_count", 0) or 0),
        "foreign_id_segment_count": int(summary.get("foreign_id_segment_count", 0) or 0),
        "duplicate_id_frame_count": int(summary.get("duplicate_id_frame_count", 0) or 0),
        "foreign_id_frame_count": int(summary.get("foreign_id_frame_count", 0) or 0),
        "duplicate_id_rate_visible": summary.get("duplicate_id_rate_visible", None),
        "foreign_id_rate_visible": summary.get("foreign_id_rate_visible", None),
        "recovery_attempts_total": int(summary.get("recovery_attempts_total", 0) or 0),
        "recovery_success_reference_total": int(summary.get("recovery_success_reference_total", 0) or 0),
        "recovery_success_own_identity_total": int(summary.get("recovery_success_own_identity_total", 0) or 0),
        "recovery_success_duplicate_id_total": int(summary.get("recovery_success_duplicate_id_total", 0) or 0),
        "recovery_success_foreign_id_total": int(summary.get("recovery_success_foreign_id_total", 0) or 0),
        "recovery_rate_reference": summary.get("recovery_rate_reference", None),
        "recovery_rate_own_identity": summary.get("recovery_rate_own_identity", None),
        "recovery_rate_duplicate_id": summary.get("recovery_rate_duplicate_id", None),
        "recovery_rate_foreign_id": summary.get("recovery_rate_foreign_id", None),
        "tracking_recall": collapsed_identity.get("tracking_recall", None),
        "tracking_iou_sum": float(tracking_iou_sum),
        "mean_tracking_iou": collapsed_identity.get("mean_tracking_iou", None),
        "deta": collapsed_identity.get("deta", None),
        "assa": collapsed_identity.get("assa", None),
        "hota": collapsed_identity.get("hota", None),
        "n_mt_objects": int(summary.get("n_mt_objects", 0) or 0),
        "n_pt_objects": int(summary.get("n_pt_objects", 0) or 0),
        "n_ml_objects": int(summary.get("n_ml_objects", 0) or 0),
        "mt": summary.get("mt", None),
        "pt": summary.get("pt", None),
        "ml": summary.get("ml", None),
        "accuracy_global_collapsed": collapsed.get("accuracy_global_collapsed", None),
        "accuracy_existing_vs_new_collapsed": collapsed.get("accuracy_existing_vs_new_collapsed", None),
        "accuracy_parent_collapsed": collapsed.get("accuracy_parent_collapsed", None),
        "new_detection_accuracy_collapsed": collapsed.get("new_detection_accuracy_collapsed", None),
        "coverage_firm": uncertainty.get("coverage_firm", None),
        "firm_accuracy": uncertainty.get("firm_accuracy", None),
        "uncertain_rate": uncertainty.get("uncertain_rate", None),
        "hypothesis_recall_uncertain": uncertainty.get("hypothesis_recall_uncertain", None),
        "n_unique_real_pred_tracks": int(summary.get("n_unique_real_pred_tracks", 0) or 0),
        "pred_track_inflation_factor": summary.get("pred_track_inflation_factor", None),
        "reopen_rate_existing": summary.get("reopen_rate_existing", None),
        "n_existing_gt_reopened_as_new_rows": int(summary.get("n_existing_gt_reopened_as_new_rows", 0) or 0),
        "n_existing_gt_reopened_as_new_ids": int(summary.get("n_existing_gt_reopened_as_new_ids", 0) or 0),
        "distance_used_count": int(summary.get("distance_used_count", 0) or 0),
        "distance_resolved_count": int(summary.get("distance_resolved_count", 0) or 0),
        "distance_correct_count": int(summary.get("distance_correct_count", 0) or 0),
        "distance_usage_rate": summary.get("distance_usage_rate", None),
        "distance_resolution_rate": summary.get("distance_resolution_rate", None),
        "distance_disambiguation_accuracy": summary.get("distance_disambiguation_accuracy", None),
        "distance_unresolved_rate": summary.get("distance_unresolved_rate", None),
        "neighbor_sets_available_count": int(summary.get("neighbor_sets_available_count", 0) or 0),
        "neighbor_sets_available_rate": summary.get("neighbor_sets_available_rate", None),
        "context_intervened_count": int(summary.get("context_intervened_count", 0) or 0),
        "context_correct_count": int(summary.get("context_correct_count", 0) or 0),
        "context_rescue_count": int(summary.get("context_rescue_count", 0) or 0),
        "context_veto_case_count": int(summary.get("context_veto_case_count", 0) or 0),
        "context_intervention_rate": summary.get("context_intervention_rate", None),
        "context_intervention_accuracy": summary.get("context_intervention_accuracy", None),
        "context_rescue_rate": summary.get("context_rescue_rate", None),
        "context_veto_rate": summary.get("context_veto_rate", None),
        "context_net_gain": summary.get("context_net_gain", None),
        "objects_fragmented": int(summary.get("objects_fragmented", 0) or 0),
        "objects_with_foreign_id_use": int(summary.get("objects_with_foreign_id_use", 0) or 0),
        "swap_events_total": int(summary.get("swap_events_total", 0) or 0),
        "theft_with_new_id_total": int(summary.get("theft_with_new_id_total", 0) or 0),
        "theft_with_displacement_total": int(summary.get("theft_with_displacement_total", 0) or 0),
        "total_runtime_seconds": summary.get("total_runtime_seconds", None),
        "avg_runtime_seconds": summary.get("avg_runtime_seconds", None),
        "total_loop_ms": summary.get("total_loop_ms", None),
        "avg_loop_ms": summary.get("avg_loop_ms", None),
    }
    for key, value in summary.items():
        if str(key).startswith("mem_"):
            row[str(key)] = value
    return row


def evaluate_scene(
    *,
    project_dir: Path,
    config_path: Path,
    input_source: dict[str, str],
    stable_min_frames: int,
    max_frames: int | None,
    force_detector_backend: str,
) -> tuple[dict, str]:
    frames_dir = input_source["frames_dir"]
    sequence_name = input_source["sequence_name"]

    cfg = Config(default_config_path=config_path)
    config = cfg.to_dict()
    config.setdefault("input", {})["frames_dir"] = frames_dir
    config.setdefault("detector", {})["backend"] = str(force_detector_backend)
    davis_cfg = config.setdefault("davis", {})
    davis_cfg["sequence_name"] = sequence_name
    if input_source.get("davis_meta_path"):
        davis_cfg["meta_path"] = input_source["davis_meta_path"]
    if input_source.get("davis_annotations_dir"):
        davis_cfg["annotations_dir"] = input_source["davis_annotations_dir"]
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

    frame_files, use_sequential_frame_ids = resolve_frame_files_for_testing(
        frames_dir,
        davis_meta_path=input_source.get("davis_meta_path", ""),
    )
    if not frame_files:
        raise RuntimeError(f"No images found in {frames_dir}")
    if max_frames is not None:
        frame_files = frame_files[: max(0, int(max_frames))]

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
            f"[BATCH][scene={sequence_name}][frame={frame_id}] "
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

    summary.update(build_memory_summary(results.get("per_frame", []) or []))

    report = build_console_report(results)
    return results, report


def aggregate_global_summary(
    *,
    run_id: str,
    per_scene_rows: list[dict],
    per_frame_rows: list[dict],
    per_case_rows: list[dict],
    per_object_rows: list[dict],
    per_event_rows: list[dict],
) -> dict:
    def _mean_metric(rows: list[dict], key: str) -> float | None:
        values: list[float] = []
        for row in rows:
            raw = row.get(key, None)
            if raw in (None, ""):
                continue
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                continue
        if not values:
            return None
        return float(sum(values) / float(len(values)))

    n_scenes_completed = int(len(per_scene_rows))
    n_frames = int(sum(int(row.get("n_frames", 0) or 0) for row in per_scene_rows))
    n_gt_objects = int(sum(int(row.get("n_gt_objects", 0) or 0) for row in per_scene_rows))
    n_cases = int(sum(int(row.get("n_cases", 0) or 0) for row in per_scene_rows))
    idtp = int(sum(int(row.get("idtp", 0) or 0) for row in per_scene_rows))
    idfp = int(sum(int(row.get("idfp", 0) or 0) for row in per_scene_rows))
    idfn = int(sum(int(row.get("idfn", 0) or 0) for row in per_scene_rows))
    n_visible_gt_observations = int(sum(int(row.get("n_visible_gt_observations", 0) or 0) for row in per_scene_rows))
    n_matched_gt_observations = int(sum(int(row.get("n_matched_gt_observations", 0) or 0) for row in per_scene_rows))
    tracking_iou_sum = float(sum(float(row.get("tracking_iou_sum", 0.0) or 0.0) for row in per_scene_rows))
    n_mt_objects = int(sum(int(row.get("n_mt_objects", 0) or 0) for row in per_scene_rows))
    n_pt_objects = int(sum(int(row.get("n_pt_objects", 0) or 0) for row in per_scene_rows))
    n_ml_objects = int(sum(int(row.get("n_ml_objects", 0) or 0) for row in per_scene_rows))
    distance_used_count = int(sum(int(row.get("distance_used_count", 0) or 0) for row in per_scene_rows))
    distance_resolved_count = int(sum(int(row.get("distance_resolved_count", 0) or 0) for row in per_scene_rows))
    distance_correct_count = int(sum(int(row.get("distance_correct_count", 0) or 0) for row in per_scene_rows))
    neighbor_sets_available_count = int(sum(int(row.get("neighbor_sets_available_count", 0) or 0) for row in per_scene_rows))
    context_intervened_count = int(sum(int(row.get("context_intervened_count", 0) or 0) for row in per_scene_rows))
    context_correct_count = int(sum(int(row.get("context_correct_count", 0) or 0) for row in per_scene_rows))
    context_rescue_count = int(sum(int(row.get("context_rescue_count", 0) or 0) for row in per_scene_rows))
    context_veto_case_count = int(sum(int(row.get("context_veto_case_count", 0) or 0) for row in per_scene_rows))
    total_runtime_seconds = float(sum(float(row.get("total_runtime_seconds", 0.0) or 0.0) for row in per_scene_rows))
    total_loop_ms = float(sum(float(row.get("total_loop_ms", 0.0) or 0.0) for row in per_scene_rows))
    objects_fragmented = int(sum(int(row.get("objects_fragmented", 0) or 0) for row in per_scene_rows))
    objects_with_foreign_id_use = int(sum(int(row.get("objects_with_foreign_id_use", 0) or 0) for row in per_scene_rows))
    duplicate_id_segment_count = int(sum(int(row.get("duplicate_id_segment_count", 0) or 0) for row in per_scene_rows))
    foreign_id_segment_count = int(sum(int(row.get("foreign_id_segment_count", 0) or 0) for row in per_scene_rows))
    duplicate_id_frame_count = int(sum(int(row.get("duplicate_id_frame_count", 0) or 0) for row in per_scene_rows))
    foreign_id_frame_count = int(sum(int(row.get("foreign_id_frame_count", 0) or 0) for row in per_scene_rows))
    recovery_attempts_total = int(sum(int(row.get("recovery_attempts_total", 0) or 0) for row in per_scene_rows))
    recovery_success_reference_total = int(sum(int(row.get("recovery_success_reference_total", 0) or 0) for row in per_scene_rows))
    recovery_success_own_identity_total = int(sum(int(row.get("recovery_success_own_identity_total", 0) or 0) for row in per_scene_rows))
    recovery_success_duplicate_id_total = int(sum(int(row.get("recovery_success_duplicate_id_total", 0) or 0) for row in per_scene_rows))
    recovery_success_foreign_id_total = int(sum(int(row.get("recovery_success_foreign_id_total", 0) or 0) for row in per_scene_rows))
    swap_events_total = int(sum(1 for row in per_event_rows if str(row.get("event_type", "")) == "swap"))
    theft_with_new_id_total = int(sum(1 for row in per_event_rows if str(row.get("event_type", "")) == "theft_with_new_id"))
    theft_with_displacement_total = int(sum(1 for row in per_event_rows if str(row.get("event_type", "")) == "theft_with_displacement"))
    n_tracking_correct_cases = int(sum(1 for row in per_case_rows if bool(row.get("collapsed_global_correct", False))))
    total_distractors_sum = float(sum(float(row.get("n_total_distractors", 0) or 0) for row in per_case_rows))
    same_class_distractors_sum = float(sum(float(row.get("n_same_class_distractors", 0) or 0) for row in per_case_rows))
    gt_area_values = [float(row["gt_area_frac"]) for row in per_case_rows if row.get("gt_area_frac", None) is not None]
    uncertainty_cases = int(
        sum(
            1
            for row in per_case_rows
            if str(row.get("final_decision", "")) in {"AMBIGUOUS_TRACK", "PROVISIONAL_PARENT", "PROVISIONAL_NEW"}
        )
    )
    firm_cases = int(sum(1 for row in per_case_rows if str(row.get("final_decision", "")) in {"MATCH", "NEW"}))
    firm_correct = int(sum(1 for row in per_case_rows if str(row.get("final_decision", "")) in {"MATCH", "NEW"} and bool(row.get("firm_global_correct", False))))
    ambiguous_hit_count = int(sum(1 for row in per_case_rows if bool(row.get("ambiguous_set_hit", False))))
    provisional_hit_count = int(sum(1 for row in per_case_rows if bool(row.get("provisional_parent_hit", False))))
    n_ambiguous_cases = int(sum(1 for row in per_case_rows if str(row.get("final_decision", "")) == "AMBIGUOUS_TRACK"))
    n_provisional_parent = int(sum(1 for row in per_case_rows if str(row.get("final_decision", "")) == "PROVISIONAL_PARENT"))
    n_provisional_new = int(sum(1 for row in per_case_rows if str(row.get("final_decision", "")) == "PROVISIONAL_NEW"))
    provisional_new_hit_count = int(
        sum(
            1
            for row in per_case_rows
            if str(row.get("final_decision", "")) == "PROVISIONAL_NEW"
            and str(row.get("real_state", "")) == "new"
        )
    )
    n_existing_gt = int(sum(1 for row in per_case_rows if str(row.get("real_state", "")) == "existing"))
    n_new_gt = int(sum(1 for row in per_case_rows if str(row.get("real_state", "")) == "new"))
    reopened_existing_rows = int(
        sum(
            1
            for row in per_case_rows
            if str(row.get("real_state", "")) == "existing" and str(row.get("collapsed_kind", "")) == "new"
        )
    )

    summary_row = {
        "run_id": str(run_id),
        "n_scenes_completed": int(n_scenes_completed),
        "n_frames": int(n_frames),
        "n_gt_objects": int(n_gt_objects),
        "n_cases": int(n_cases),
        "n_visible_gt_observations": int(n_visible_gt_observations),
        "n_matched_gt_observations": int(n_matched_gt_observations),
        "idtp": int(idtp),
        "idfp": int(idfp),
        "idfn": int(idfn),
        "idf1": safe_pct(2 * idtp, 2 * idtp + idfp + idfn),
        "idp": safe_pct(idtp, idtp + idfp),
        "idr": safe_pct(idtp, idtp + idfn),
        "idsw": int(sum(int(row.get("idsw", 0) or 0) for row in per_scene_rows)),
        "frag": int(sum(int(row.get("frag", 0) or 0) for row in per_scene_rows)),
        "duplicate_id_segment_count": int(duplicate_id_segment_count),
        "foreign_id_segment_count": int(foreign_id_segment_count),
        "duplicate_id_frame_count": int(duplicate_id_frame_count),
        "foreign_id_frame_count": int(foreign_id_frame_count),
        "duplicate_id_rate_visible": safe_pct(duplicate_id_frame_count, n_visible_gt_observations),
        "foreign_id_rate_visible": safe_pct(foreign_id_frame_count, n_visible_gt_observations),
        "recovery_attempts_total": int(recovery_attempts_total),
        "recovery_success_reference_total": int(recovery_success_reference_total),
        "recovery_success_own_identity_total": int(recovery_success_own_identity_total),
        "recovery_success_duplicate_id_total": int(recovery_success_duplicate_id_total),
        "recovery_success_foreign_id_total": int(recovery_success_foreign_id_total),
        "recovery_rate_reference": safe_pct(recovery_success_reference_total, recovery_attempts_total),
        "recovery_rate_own_identity": safe_pct(recovery_success_own_identity_total, recovery_attempts_total),
        "recovery_rate_duplicate_id": safe_pct(recovery_success_duplicate_id_total, recovery_attempts_total),
        "recovery_rate_foreign_id": safe_pct(recovery_success_foreign_id_total, recovery_attempts_total),
        "tracking_recall": safe_pct(n_tracking_correct_cases, n_visible_gt_observations),
        "mean_tracking_iou": (float(tracking_iou_sum) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else None,
        "deta": safe_pct(n_matched_gt_observations, n_visible_gt_observations),
        "assa": safe_pct(n_tracking_correct_cases, n_matched_gt_observations),
        "hota": (
            float(
                (safe_pct(n_matched_gt_observations, n_visible_gt_observations) or 0.0)
                * (safe_pct(n_tracking_correct_cases, n_matched_gt_observations) or 0.0)
            ) ** 0.5
            if n_visible_gt_observations > 0 and n_matched_gt_observations > 0
            else None
        ),
        "n_mt_objects": int(n_mt_objects),
        "n_pt_objects": int(n_pt_objects),
        "n_ml_objects": int(n_ml_objects),
        "mt": safe_pct(n_mt_objects, n_gt_objects),
        "pt": safe_pct(n_pt_objects, n_gt_objects),
        "ml": safe_pct(n_ml_objects, n_gt_objects),
        "accuracy_global_collapsed": safe_pct(n_tracking_correct_cases, n_cases),
        "accuracy_existing_vs_new_collapsed": safe_pct(
            sum(1 for row in per_case_rows if bool(row.get("collapsed_existing_vs_new_correct", False))),
            n_cases,
        ),
        "accuracy_parent_collapsed": safe_pct(
            sum(1 for row in per_case_rows if str(row.get("real_state", "")) == "existing" and bool(row.get("collapsed_parent_correct", False))),
            n_existing_gt,
        ),
        "new_detection_accuracy_collapsed": safe_pct(
            sum(1 for row in per_case_rows if str(row.get("real_state", "")) == "new" and str(row.get("collapsed_kind", "")) == "new"),
            n_new_gt,
        ),
        "coverage_firm": safe_pct(firm_cases, n_cases),
        "firm_accuracy": safe_pct(firm_correct, firm_cases),
        "uncertain_rate": safe_pct(uncertainty_cases, n_cases),
        "hypothesis_recall_uncertain": safe_pct(
            ambiguous_hit_count + provisional_hit_count + provisional_new_hit_count,
            n_ambiguous_cases + n_provisional_parent + n_provisional_new,
        ),
        "n_unique_real_pred_tracks": int(sum(int(row.get("n_unique_real_pred_tracks", 0) or 0) for row in per_scene_rows)),
        "pred_track_inflation_factor": (
            float(sum(int(row.get("n_unique_real_pred_tracks", 0) or 0) for row in per_scene_rows)) / float(n_gt_objects)
            if n_gt_objects > 0
            else None
        ),
        "reopen_rate_existing": safe_pct(reopened_existing_rows, n_existing_gt),
        "mean_gt_per_frame": (float(n_visible_gt_observations) / float(n_frames)) if n_frames > 0 else None,
        "mean_total_distractors": (float(total_distractors_sum) / float(n_cases)) if n_cases > 0 else None,
        "mean_same_class_distractors": (float(same_class_distractors_sum) / float(n_cases)) if n_cases > 0 else None,
        "mean_gt_area_frac": (float(sum(gt_area_values)) / float(len(gt_area_values))) if gt_area_values else None,
        "distance_used_count": int(distance_used_count),
        "distance_resolved_count": int(distance_resolved_count),
        "distance_correct_count": int(distance_correct_count),
        "distance_usage_rate": safe_pct(distance_used_count, n_cases),
        "distance_resolution_rate": safe_pct(distance_resolved_count, distance_used_count),
        "distance_disambiguation_accuracy": safe_pct(distance_correct_count, distance_resolved_count),
        "neighbor_sets_available_count": int(neighbor_sets_available_count),
        "neighbor_sets_available_rate": safe_pct(neighbor_sets_available_count, n_cases),
        "context_intervened_count": int(context_intervened_count),
        "context_correct_count": int(context_correct_count),
        "context_rescue_count": int(context_rescue_count),
        "context_veto_case_count": int(context_veto_case_count),
        "context_intervention_rate": safe_pct(context_intervened_count, n_cases),
        "context_intervention_accuracy": safe_pct(context_correct_count, context_intervened_count),
        "context_rescue_rate": safe_pct(context_rescue_count, n_cases),
        "context_veto_rate": safe_pct(context_veto_case_count, n_cases),
        "context_net_gain": (
            float(context_correct_count - max(0, context_intervened_count - context_correct_count)) / float(n_cases)
            if n_cases > 0
            else None
        ),
        "objects_fragmented": int(objects_fragmented),
        "objects_with_foreign_id_use": int(objects_with_foreign_id_use),
        "swap_events_total": int(swap_events_total),
        "theft_with_new_id_total": int(theft_with_new_id_total),
        "theft_with_displacement_total": int(theft_with_displacement_total),
        "total_runtime_seconds": float(total_runtime_seconds),
        "avg_runtime_seconds": (float(total_runtime_seconds) / float(n_scenes_completed)) if n_scenes_completed > 0 else None,
        "total_loop_ms": float(total_loop_ms),
        "avg_loop_ms": (float(total_loop_ms) / float(n_frames)) if n_frames > 0 else None,
        # Current primary metrics are observation-weighted/global aggregates.
        "obs_weighted_idf1": safe_pct(2 * idtp, 2 * idtp + idfp + idfn),
        "obs_weighted_hota": (
            float(
                (safe_pct(n_matched_gt_observations, n_visible_gt_observations) or 0.0)
                * (safe_pct(n_tracking_correct_cases, n_matched_gt_observations) or 0.0)
            ) ** 0.5
            if n_visible_gt_observations > 0 and n_matched_gt_observations > 0
            else None
        ),
        "obs_weighted_tracking_recall": safe_pct(n_tracking_correct_cases, n_visible_gt_observations),
        # Macro summaries help compare scene-to-scene and object-to-object fairness.
        "scene_macro_idf1": _mean_metric(per_scene_rows, "idf1"),
        "scene_macro_hota": _mean_metric(per_scene_rows, "hota"),
        "scene_macro_tracking_recall": _mean_metric(per_scene_rows, "tracking_recall"),
        "scene_macro_accuracy_existing_vs_new": _mean_metric(per_scene_rows, "accuracy_existing_vs_new_collapsed"),
        "object_macro_strict_accuracy": _mean_metric(per_object_rows, "strict_accuracy"),
        "object_macro_permissive_accuracy": _mean_metric(per_object_rows, "permissive_accuracy"),
        "object_macro_tracking_recall": _mean_metric(per_object_rows, "tracking_recall_object"),
        "object_macro_tracking_iou": _mean_metric(per_object_rows, "mean_tracking_iou_object"),
    }
    summary_row.update(build_memory_summary(per_frame_rows))
    return summary_row


def build_batch_report(*, summary_row: dict, per_scene_rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("[BATCH][Summary]")
    for key in [
        "run_id",
        "n_scenes_completed",
        "n_frames",
        "n_gt_objects",
        "n_cases",
        "idf1",
        "idp",
        "idr",
        "idsw",
        "frag",
        "mt",
        "pt",
        "ml",
        "tracking_recall",
        "mean_tracking_iou",
        "hota",
        "deta",
        "assa",
        "accuracy_global_collapsed",
        "coverage_firm",
        "firm_accuracy",
        "uncertain_rate",
        "hypothesis_recall_uncertain",
        "pred_track_inflation_factor",
        "reopen_rate_existing",
        "distance_usage_rate",
        "distance_resolution_rate",
        "distance_disambiguation_accuracy",
        "context_intervention_rate",
        "context_intervention_accuracy",
        "obs_weighted_idf1",
        "obs_weighted_hota",
        "obs_weighted_tracking_recall",
        "scene_macro_idf1",
        "scene_macro_hota",
        "scene_macro_tracking_recall",
        "object_macro_strict_accuracy",
        "object_macro_permissive_accuracy",
        "object_macro_tracking_recall",
        "object_macro_tracking_iou",
        "total_runtime_seconds",
        "avg_loop_ms",
    ]:
        value = summary_row.get(key, None)
        if key in {
            "idf1", "idp", "idr", "mt", "pt", "ml", "tracking_recall",
            "hota", "deta", "assa", "accuracy_global_collapsed",
            "coverage_firm", "firm_accuracy", "uncertain_rate",
            "hypothesis_recall_uncertain", "reopen_rate_existing",
            "distance_usage_rate", "distance_resolution_rate",
            "distance_disambiguation_accuracy", "context_intervention_rate",
            "context_intervention_accuracy", "obs_weighted_idf1",
            "obs_weighted_hota", "obs_weighted_tracking_recall",
            "scene_macro_idf1", "scene_macro_hota",
            "scene_macro_tracking_recall", "object_macro_strict_accuracy",
            "object_macro_permissive_accuracy", "object_macro_tracking_recall",
        }:
            value = fmt_pct(value)
        elif key in {"mean_tracking_iou", "object_macro_tracking_iou"} and value is not None:
            value = f"{float(value):.4f}"
        elif key == "pred_track_inflation_factor" and value is not None:
            value = f"{float(value):.3f}x"
        elif key == "avg_loop_ms" and value is not None:
            value = f"{float(value):.2f}"
        lines.append(f"  {key}: {value}")

    sorted_by_idf1 = sorted(
        per_scene_rows,
        key=lambda row: float(row.get("idf1", -1.0) if row.get("idf1", None) is not None else -1.0),
    )[:5]
    lines.append("")
    lines.append("[BATCH][WorstScenesByIDF1]")
    if sorted_by_idf1:
        lines.append(
            render_table(
                [
                    {
                        "scene": row.get("scene_name", row.get("scene_id", "")),
                        "idf1": fmt_pct(row.get("idf1", None)),
                        "trkrec": fmt_pct(row.get("tracking_recall", None)),
                        "reopen": fmt_pct(row.get("reopen_rate_existing", None)),
                        "infl": "-" if row.get("pred_track_inflation_factor", None) is None else f"{float(row.get('pred_track_inflation_factor')):.2f}x",
                    }
                    for row in sorted_by_idf1
                ],
                [
                    ("scene", "Scene"),
                    ("idf1", "IDF1"),
                    ("trkrec", "TrackRec"),
                    ("reopen", "Reopen"),
                    ("infl", "Infl"),
                ],
            )
        )
    else:
        lines.append("No completed scenes.")

    sorted_by_reopen = sorted(
        per_scene_rows,
        key=lambda row: float(row.get("reopen_rate_existing", -1.0) if row.get("reopen_rate_existing", None) is not None else -1.0),
        reverse=True,
    )[:5]
    lines.append("")
    lines.append("[BATCH][WorstScenesByReopen]")
    if sorted_by_reopen:
        lines.append(
            render_table(
                [
                    {
                        "scene": row.get("scene_name", row.get("scene_id", "")),
                        "reopen": fmt_pct(row.get("reopen_rate_existing", None)),
                        "idf1": fmt_pct(row.get("idf1", None)),
                        "dist": fmt_pct(row.get("distance_disambiguation_accuracy", None)),
                        "ctx": fmt_pct(row.get("context_intervention_accuracy", None)),
                    }
                    for row in sorted_by_reopen
                ],
                [
                    ("scene", "Scene"),
                    ("reopen", "Reopen"),
                    ("idf1", "IDF1"),
                    ("dist", "DistAcc"),
                    ("ctx", "CtxAcc"),
                ],
            )
        )
    else:
        lines.append("No completed scenes.")

    detector_backends = {
        str(row.get("detector_backend", "") or "").strip().lower()
        for row in per_scene_rows
        if str(row.get("detector_backend", "") or "").strip()
    }
    lines.append("")
    lines.append("[BATCH][Notes]")
    lines.append("  per_class.csv concatena filas por clase y por escena; usa per_class_global.csv para el agregado global real por clase.")
    lines.append("  En per_case.csv, final_decision=MATCH significa 'asignado a un track existente', no 'identidad correcta'; usa final_decision_parent_correct/final_decision_global_correct o firm_global_correct.")
    if detector_backends == {"davis"}:
        lines.append("  Con detector_backend=davis y mascaras GT exactas, mean_tracking_iou coincide con tracking_recall; no lo interpretes como una metrica geometrica independiente.")

    return "\n".join(lines)


def rebuild_batch_outputs(
    *,
    batch_dir: Path,
    run_id: str,
    selected_scene_ids: list[str],
    registered_scene_ids: list[str],
    scene_name_by_id: dict[str, str],
    failed_scene_errors: dict[str, str] | None = None,
) -> None:
    scenes_root = batch_dir / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)
    previous_manifest_rows = read_manifest_rows(batch_dir)
    previous_manifest_by_scene_id = {
        str(row.get("scene_id", "") or "").strip(): dict(row)
        for row in previous_manifest_rows
        if str(row.get("scene_id", "") or "").strip()
    }

    aggregated_rows: dict[str, list[dict]] = {key: [] for key in SCENE_TABLE_FILES.keys()}
    for scene_dir in iter_scene_dirs(batch_dir):
        if not scene_dir_is_complete(scene_dir):
            continue
        for table_key, filename in SCENE_TABLE_FILES.items():
            aggregated_rows[table_key].extend(read_csv_rows(scene_dir / filename))

    per_scene_rows = list(aggregated_rows["per_scene"])
    per_case_rows = normalize_per_case_rows(list(aggregated_rows["per_case"]))
    per_object_rows = list(aggregated_rows["per_object"])
    per_pred_track_rows = list(aggregated_rows["per_pred_track"])
    per_event_rows = list(aggregated_rows["per_event"])
    per_class_global_rows = build_global_per_class_rows(
        run_id=run_id,
        per_object_rows=per_object_rows,
        per_pred_track_rows=per_pred_track_rows,
        per_case_rows=per_case_rows,
    )
    selected_scene_id_set = {str(scene_id) for scene_id in selected_scene_ids}
    selected_per_scene_rows = [
        row for row in per_scene_rows if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_frame_rows = [
        row for row in aggregated_rows["per_frame"] if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_case_rows = [
        row for row in per_case_rows if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_object_rows = [
        row for row in per_object_rows if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_pred_track_rows = [
        row for row in per_pred_track_rows if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_event_rows = [
        row for row in per_event_rows if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_class_scene_rows = [
        row for row in aggregated_rows["per_class"] if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_class_global_rows = build_global_per_class_rows(
        run_id=run_id,
        per_object_rows=selected_per_object_rows,
        per_pred_track_rows=selected_per_pred_track_rows,
        per_case_rows=selected_per_case_rows,
    )

    summary_row = aggregate_global_summary(
        run_id=run_id,
        per_scene_rows=per_scene_rows,
        per_frame_rows=list(aggregated_rows["per_frame"]),
        per_case_rows=per_case_rows,
        per_object_rows=per_object_rows,
        per_event_rows=per_event_rows,
    )
    selected_summary_row = aggregate_global_summary(
        run_id=run_id,
        per_scene_rows=selected_per_scene_rows,
        per_frame_rows=selected_per_frame_rows,
        per_case_rows=selected_per_case_rows,
        per_object_rows=selected_per_object_rows,
        per_event_rows=selected_per_event_rows,
    )

    write_single_row_csv(batch_dir / "summary_global.csv", summary_row)
    write_single_row_csv(batch_dir / "summary_current_batch.csv", selected_summary_row)
    write_csv(batch_dir / "per_scene.csv", per_scene_rows)
    write_csv(batch_dir / "per_scene_current_batch.csv", selected_per_scene_rows)
    write_csv(batch_dir / "per_class.csv", aggregated_rows["per_class"])
    write_csv(batch_dir / "per_class_scene.csv", aggregated_rows["per_class"])
    write_csv(batch_dir / "per_class_scene_current_batch.csv", selected_per_class_scene_rows)
    write_csv(batch_dir / "per_class_global.csv", per_class_global_rows)
    write_csv(batch_dir / "per_class_global_current_batch.csv", selected_per_class_global_rows)
    write_csv(batch_dir / "per_object.csv", per_object_rows)
    write_csv(batch_dir / "per_case.csv", per_case_rows)
    write_csv(batch_dir / "per_case_modules.csv", aggregated_rows["per_case_modules"])
    write_csv(batch_dir / "per_frame.csv", aggregated_rows["per_frame"])
    write_csv(batch_dir / "per_pred_track.csv", per_pred_track_rows)
    write_csv(batch_dir / "per_event.csv", per_event_rows)
    write_text(batch_dir / "report.txt", build_batch_report(summary_row=summary_row, per_scene_rows=per_scene_rows) + "\n")
    write_text(
        batch_dir / "report_current_batch.txt",
        build_batch_report(summary_row=selected_summary_row, per_scene_rows=selected_per_scene_rows) + "\n",
    )

    failed_scene_errors = dict(failed_scene_errors or {})
    manifest_rows = []
    completed_by_scene_id = {
        str(row.get("scene_id", "")): row
        for row in per_scene_rows
        if row.get("scene_id", None) is not None
    }
    for scene_id in registered_scene_ids:
        scene_id = str(scene_id)
        previous_row = previous_manifest_by_scene_id.get(scene_id, {})
        output_dir = str((scenes_root / sanitize_name_for_path(scene_id)).resolve())
        interrupted = (scenes_root / f".tmp_{sanitize_name_for_path(scene_id)}").exists()
        row = completed_by_scene_id.get(scene_id, None)
        if row is not None:
            manifest_rows.append(
                {
                    "run_id": str(run_id),
                    "scene_id": scene_id,
                    "scene_name": str(row.get("scene_name", scene_name_by_id.get(scene_id, scene_id))),
                    "status": "completed",
                    "started_at": row.get("started_at", None),
                    "finished_at": row.get("finished_at", None),
                    "n_frames_expected": row.get("n_frames", None),
                    "n_frames_processed": row.get("n_frames", None),
                    "output_dir": str(row.get("output_dir", output_dir)),
                    "error_message": "",
                    "selected_in_current_run": bool(scene_id in selected_scene_id_set),
                }
            )
            continue

        status = "pending"
        error_message = ""
        if interrupted:
            status = "interrupted"
        elif scene_id in failed_scene_errors:
            status = "failed"
            error_message = str(failed_scene_errors.get(scene_id, "") or "")
        elif str(previous_row.get("status", "") or "").strip():
            status = str(previous_row.get("status", "") or "").strip()
            if status == "failed":
                error_message = str(previous_row.get("error_message", "") or "")

        manifest_rows.append(
            {
                "run_id": str(run_id),
                "scene_id": scene_id,
                "scene_name": str(scene_name_by_id.get(scene_id, scene_id)),
                "status": status,
                "started_at": previous_row.get("started_at", None),
                "finished_at": previous_row.get("finished_at", None),
                "n_frames_expected": previous_row.get("n_frames_expected", None),
                "n_frames_processed": previous_row.get("n_frames_processed", None),
                "output_dir": str(previous_row.get("output_dir", output_dir)),
                "error_message": error_message,
                "selected_in_current_run": bool(scene_id in selected_scene_id_set),
            }
        )
    write_csv(batch_dir / "manifest.csv", manifest_rows)
    write_csv(
        batch_dir / "manifest_current_batch.csv",
        [row for row in manifest_rows if bool(row.get("selected_in_current_run", False))],
    )


def write_scene_outputs(
    *,
    temp_scene_dir: Path,
    final_scene_dir: Path,
    run_id: str,
    scene_id: str,
    scene_name: str,
    results: dict,
    scene_report: str,
    scene_started_at: str | None = None,
    scene_finished_at: str | None = None,
    stable_min_frames: int | None = None,
    max_frames: int | None = None,
    force_detector_backend: str | None = None,
    mask_variant: str | None = None,
    image_subdir: str | None = None,
) -> None:
    temp_scene_dir.mkdir(parents=True, exist_ok=True)
    scene_summary_row = build_scene_summary_row(
        run_id=run_id,
        scene_id=scene_id,
        scene_name=scene_name,
        results=results,
        scene_started_at=scene_started_at,
        scene_finished_at=scene_finished_at,
        output_dir=final_scene_dir,
        stable_min_frames=stable_min_frames,
        max_frames=max_frames,
        force_detector_backend=force_detector_backend,
        mask_variant=mask_variant,
        image_subdir=image_subdir,
    )

    write_single_row_csv(temp_scene_dir / SCENE_TABLE_FILES["per_scene"], scene_summary_row)
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_class"], annotate_rows(results.get("per_class", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_object"], annotate_rows(results.get("per_object", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_case"], annotate_rows(results.get("per_case", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_case_modules"], annotate_rows(results.get("per_case_modules", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_frame"], annotate_rows(results.get("per_frame", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_pred_track"], annotate_rows(results.get("per_pred_track", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_event"], build_event_rows(run_id=run_id, scene_id=scene_id, scene_name=scene_name, results=results))
    write_text(temp_scene_dir / "report.txt", str(scene_report).rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tracking evaluation on a batch of scenes.")
    parser.add_argument("--scenes-def-file", default="", help="JSON file with custom scene definitions (env: BATCH_SCENES_DEF_FILE)")
    parser.add_argument("--scenes", default="", help="Comma/space-separated scene IDs (env: BATCH_SCENES)")
    parser.add_argument("--scenes-file", default="", help="Text file with scene IDs, one per line (env: BATCH_SCENES_FILE)")
    parser.add_argument("--scene-id", default="", help="Single scene ID to process (env: SCENE_ID)")
    parser.add_argument("--mask-variant", default="", help="Mask variant (env: MASK_VARIANT, default: benchmark)")
    parser.add_argument("--image-subdir", default="", help="Image subdirectory (env: IMAGE_SUBDIR)")
    parser.add_argument("--masks-root", default="", help="ScanNet++ masks root directory (env: SCANNETPP_MASKS_ROOT)")
    parser.add_argument("--images-root", default="", help="ScanNet++ images root directory (env: SCANNETPP_IMAGES_ROOT)")
    parser.add_argument("--masks-subdir", default="", help="Masks subdirectory name (env: SCANNETPP_MASKS_SUBDIR)")
    parser.add_argument("--output-dir", default="", help="Output root directory (env: OUTPUT_DIR)")
    parser.add_argument("--max-scenes", type=int, default=None, help="Maximum number of scenes to process (env: BATCH_MAX_SCENES)")
    parser.add_argument("--batch-size", type=int, default=None, help="Scenes per run (env: BATCH_SIZE, defaults to --max-scenes)")
    parser.add_argument("--stable-min-frames", type=int, default=None, help="Stable min frames threshold (env: BATCH_STABLE_MIN_FRAMES, default: 3)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per scene (env: BATCH_MAX_FRAMES)")
    parser.add_argument("--detector-backend", default="", help="Detector backend (env: BATCH_DETECTOR_BACKEND, default: davis)")
    args = parser.parse_args()

    def _e(arg_val: str, env_name: str, default: str = "") -> str:
        return (arg_val or os.environ.get(env_name, default) or default).strip()

    def _ei(arg_val: int | None, env_name: str, default: int | None = None) -> int | None:
        if arg_val is not None:
            return arg_val
        return _env_int(env_name, default)

    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir.parent
    project_dir = src_dir.parent
    config_path = src_dir / "config" / "default_config.yaml"

    scenes_def_file = _e(args.scenes_def_file, "BATCH_SCENES_DEF_FILE")
    custom_scene_defs: list[dict[str, str]] | None = None
    masks_root_base: Path | None = None
    images_root_base: Path | None = None

    image_subdir = _e(args.image_subdir, "IMAGE_SUBDIR", "dslr/resized_images")
    mask_variant = _e(args.mask_variant, "MASK_VARIANT", "benchmark").lower() or "benchmark"
    if mask_variant == "benchmark":
        mask_variant = "benchmark_instance"
    masks_subdir = _e(args.masks_subdir, "SCANNETPP_MASKS_SUBDIR", "2Dmasks")

    if scenes_def_file:
        custom_scene_defs = load_custom_scene_definitions(scenes_def_file)
        print(f"[BATCH] Mode -> custom_davis_def ({len(custom_scene_defs)} scenes from {scenes_def_file})")
    else:
        masks_root_env = _e(args.masks_root, "SCANNETPP_MASKS_ROOT")
        images_root_env = _e(args.images_root, "SCANNETPP_IMAGES_ROOT")
        if not masks_root_env:
            masks_root_env = "/media/pablo/LINUX/Qsync/2026_tracker_reid/datasets/scannetpp_data"
        if not images_root_env:
            images_root_env = "/media/pablo/LINUX/Qsync/2026_tracker_reid/datasets/scannetpp_data"
        masks_root_base = Path(masks_root_env).expanduser().resolve()
        images_root_base = Path(images_root_env).expanduser().resolve()
        print(f"[BATCH] Mode -> scannetpp (masks={masks_root_base}, images={images_root_base})")

    scene_ids = resolve_batch_scene_ids(
        masks_root_base=masks_root_base,
        images_root_base=images_root_base,
        mask_variant=mask_variant,
        image_subdir=image_subdir,
        masks_subdir=masks_subdir,
        custom_scene_defs=custom_scene_defs,
        scenes=_e(args.scenes, "BATCH_SCENES"),
        scenes_file=_e(args.scenes_file, "BATCH_SCENES_FILE"),
        scene_id=_e(args.scene_id, "SCENE_ID"),
    )
    if not scene_ids:
        raise RuntimeError("No scenes resolved for the batch.")

    run_id = "our_pipeline"
    _output_dir_arg = _e(args.output_dir, "OUTPUT_DIR")
    output_root = (
        Path(_output_dir_arg).expanduser().resolve() / "testing_batch"
        if _output_dir_arg
        else (project_dir / "outputs" / "tfm" / "testing_batch").resolve()
    )
    batch_dir = output_root
    scenes_root = batch_dir / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)

    max_scenes = _ei(args.max_scenes, "BATCH_MAX_SCENES", None)
    batch_size_arg = _ei(args.batch_size, "BATCH_SIZE", None)
    batch_size = batch_size_arg if batch_size_arg is not None else max_scenes
    (
        scene_ids,
        registered_scene_ids,
        selection_mode,
        existing_manifest_rows,
        existing_per_scene_rows,
    ) = resolve_scene_schedule(
        candidate_scene_ids=unique_preserve_order([str(scene_id) for scene_id in scene_ids]),
        batch_dir=batch_dir,
        batch_size=batch_size,
    )

    stable_min_frames = int(_ei(args.stable_min_frames, "BATCH_STABLE_MIN_FRAMES", 3) or 3)
    max_frames = _ei(args.max_frames, "BATCH_MAX_FRAMES", None)
    force_detector_backend = _e(args.detector_backend, "BATCH_DETECTOR_BACKEND", "davis")

    # Build a lookup for custom mode so each scene loop can find its def quickly.
    custom_defs_by_id: dict[str, dict[str, str]] = (
        {d["scene_id"]: d for d in custom_scene_defs} if custom_scene_defs is not None else {}
    )

    run_config_row = build_run_config_row(
        run_id=run_id,
        batch_name=run_id,
        batch_dir=batch_dir,
        masks_root_base=masks_root_base,
        images_root_base=images_root_base,
        image_subdir=image_subdir,
        mask_variant=mask_variant,
        stable_min_frames=stable_min_frames,
        max_frames=max_frames,
        max_scenes=max_scenes,
        batch_size=batch_size,
        selection_mode=selection_mode,
        force_detector_backend=force_detector_backend,
        selected_scene_ids=scene_ids,
        registered_scene_ids=registered_scene_ids,
        scenes_def_file=scenes_def_file or None,
    )
    write_single_row_csv(batch_dir / "run_config.csv", run_config_row)

    scene_name_by_id = merge_scene_name_index(
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
    rebuild_batch_outputs(
        batch_dir=batch_dir,
        run_id=run_id,
        selected_scene_ids=scene_ids,
        registered_scene_ids=registered_scene_ids,
        scene_name_by_id=scene_name_by_id,
        failed_scene_errors=failed_scene_errors,
    )
    if not scene_ids:
        print("[BATCH] No hay escenas pendientes para esta ejecucion.")
        return

    for scene_id in scene_ids:
        scene_key = sanitize_name_for_path(str(scene_id))
        final_scene_dir = scenes_root / scene_key
        if scene_dir_is_complete(final_scene_dir):
            print(f"[BATCH] Skip completed scene -> {scene_id}")
            continue
        if final_scene_dir.exists():
            incomplete_backup_dir = reserve_incomplete_scene_backup_dir(final_scene_dir)
            final_scene_dir.rename(incomplete_backup_dir)
            print(
                f"[BATCH] Incomplete final output moved -> {scene_id} "
                f"({final_scene_dir} -> {incomplete_backup_dir})"
            )

        temp_scene_dir = scenes_root / f".tmp_{scene_key}"
        if temp_scene_dir.exists():
            shutil.rmtree(temp_scene_dir)

        scene_started_at = _now_iso()
        try:
            if custom_defs_by_id:
                input_source = build_custom_scene_input_source(custom_defs_by_id[str(scene_id)])
            else:
                input_source = build_scene_input_source(
                    scene_id=str(scene_id),
                    masks_root_base=masks_root_base,
                    images_root_base=images_root_base,
                    mask_variant=mask_variant,
                    image_subdir=image_subdir,
                    masks_subdir=masks_subdir,
                )
            print(f"[BATCH] Scene start -> {scene_id}")
            print(f"[BATCH] Frames dir -> {input_source['frames_dir']}")
            results, scene_report = evaluate_scene(
                project_dir=project_dir,
                config_path=config_path,
                input_source=input_source,
                stable_min_frames=stable_min_frames,
                max_frames=max_frames,
                force_detector_backend=force_detector_backend,
            )
            scene_name = str(input_source.get("sequence_name", scene_id))
            scene_name_by_id[str(scene_id)] = str(scene_name)
            failed_scene_errors.pop(str(scene_id), None)
            write_scene_outputs(
                temp_scene_dir=temp_scene_dir,
                final_scene_dir=final_scene_dir,
                run_id=run_id,
                scene_id=str(scene_id),
                scene_name=str(scene_name),
                results=results,
                scene_report=scene_report,
                scene_started_at=scene_started_at,
                scene_finished_at=_now_iso(),
                stable_min_frames=stable_min_frames,
                max_frames=max_frames,
                force_detector_backend=force_detector_backend,
                mask_variant=mask_variant,
                image_subdir=image_subdir,
            )
            if final_scene_dir.exists():
                raise RuntimeError(
                    f"Ya existe output final para {scene_id}: {final_scene_dir}. No se sobreescribe automaticamente."
                )
            temp_scene_dir.rename(final_scene_dir)
            rebuild_batch_outputs(
                batch_dir=batch_dir,
                run_id=run_id,
                selected_scene_ids=scene_ids,
                registered_scene_ids=registered_scene_ids,
                scene_name_by_id=scene_name_by_id,
                failed_scene_errors=failed_scene_errors,
            )
            print(f"[BATCH] Scene completed -> {scene_id} ({scene_started_at} -> {_now_iso()})")
        except Exception as exc:
            failed_scene_errors[str(scene_id)] = str(exc)
            if temp_scene_dir.exists():
                shutil.rmtree(temp_scene_dir)
            rebuild_batch_outputs(
                batch_dir=batch_dir,
                run_id=run_id,
                selected_scene_ids=scene_ids,
                registered_scene_ids=registered_scene_ids,
                scene_name_by_id=scene_name_by_id,
                failed_scene_errors=failed_scene_errors,
            )
            print(f"[BATCH] Scene failed -> {scene_id}: {exc}")


if __name__ == "__main__":
    main()
