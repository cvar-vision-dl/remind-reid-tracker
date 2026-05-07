from __future__ import annotations

import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import re


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
PROJECT_DIR = SRC_DIR.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from testing.common.generic_tracking_reporting import fmt_pct, render_table, write_csv, write_text
from testing.d4sm.run_tracking_test import create_d4sm_tracker, evaluate_scene, resolve_d4sm_runtime_config
from utils.scannetpp_tar import (
    resolve_scene_annotations_tar_path,
    resolve_scene_tar_path,
    resolve_scannetpp_data_parent,
)


SCENE_TABLE_FILES = {
    "per_scene": "scene_summary.csv",
    "per_class": "per_class.csv",
    "per_object": "per_object.csv",
    "per_case": "per_case.csv",
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


def sanitize_name_for_path(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown_scene"
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "unknown_scene"


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
        raise FileNotFoundError(f"No existe APP2_D4SM_BATCH_SCENES_FILE: {p}")
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
) -> list[str]:
    if mask_variant == "benchmark":
        mask_variant = "benchmark_instance"

    masks_root = (masks_root_base / "2Dmasks" / scene_id).resolve()
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

    tar_path = resolve_scene_tar_path(images_root_base=images_root_base, scene_id=scene_id)
    annotations_tar_path = resolve_scene_annotations_tar_path(images_root_base=images_root_base, scene_id=scene_id)
    if tar_path is not None and annotations_tar_path is not None:
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


def discover_scene_ids(
    *,
    masks_root_base: Path,
    images_root_base: Path,
    mask_variant: str,
    image_subdir: str,
) -> list[str]:
    masks_parent = (masks_root_base / "2Dmasks").resolve()
    candidate_scene_ids: list[str]
    if masks_parent.is_dir():
        candidate_scene_ids = [
            str(child.name)
            for child in sorted(masks_parent.iterdir())
            if child.is_dir()
        ]
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
            f"[D4SM][BATCH] Scene discovery skipped {skipped_count} escenas sin inputs listos "
            f"(variant={mask_variant}, image_subdir={image_subdir}){suffix}"
        )
    return unique_preserve_order(scene_ids)


def resolve_batch_scene_ids(
    *,
    masks_root_base: Path,
    images_root_base: Path,
    mask_variant: str,
    image_subdir: str,
) -> list[str]:
    scenes_env = os.environ.get("APP2_D4SM_BATCH_SCENES", "").strip()
    scenes_file = os.environ.get("APP2_D4SM_BATCH_SCENES_FILE", "").strip()
    single_scene = os.environ.get("APP2_SCENE_ID", "").strip()

    if scenes_file:
        return read_scene_ids_from_file(scenes_file)
    if scenes_env:
        parts = re.split(r"[\s,;]+", scenes_env)
        return unique_preserve_order(parts)
    if single_scene:
        return [single_scene]
    return discover_scene_ids(
        masks_root_base=masks_root_base,
        images_root_base=images_root_base,
        mask_variant=mask_variant,
        image_subdir=image_subdir,
    )


def build_scene_input_source(
    *,
    scene_id: str,
    masks_root_base: Path,
    images_root_base: Path,
    mask_variant: str,
    image_subdir: str,
) -> dict[str, str]:
    if mask_variant == "benchmark":
        mask_variant = "benchmark_instance"
    masks_root = (masks_root_base / "2Dmasks" / scene_id).resolve()
    meta_path = (masks_root / f"meta_{mask_variant}.json").resolve()
    annotations_dir = (masks_root / "annotations" / mask_variant).resolve()
    frames_dir = (images_root_base / "data" / scene_id / image_subdir).resolve()
    external_ready = frames_dir.is_dir() and meta_path.is_file() and annotations_dir.is_dir()
    if not external_ready:
        tar_path = resolve_scene_tar_path(images_root_base=images_root_base, scene_id=scene_id)
        annotations_tar_path = resolve_scene_annotations_tar_path(images_root_base=images_root_base, scene_id=scene_id)
        if tar_path is not None and annotations_tar_path is not None:
            return {
                "mode": "external_scannetpp_tar",
                "sequence_name": str(scene_id),
                "image_subdir": str(image_subdir),
                "mask_variant": str(mask_variant),
                "data_tar_path": str(tar_path),
                "annotations_tar_path": str(annotations_tar_path),
            }
        issues = collect_scene_input_issues(
            scene_id=scene_id,
            masks_root_base=masks_root_base,
            images_root_base=images_root_base,
            mask_variant=mask_variant,
            image_subdir=image_subdir,
        )
        raise FileNotFoundError(
            f"Escena {scene_id} sin inputs de testing preparados: {', '.join(issues)}"
        )
    return {
        "mode": "external_scannetpp_batch",
        "frames_dir": str(frames_dir),
        "sequence_name": str(scene_id),
        "davis_meta_path": str(meta_path),
        "davis_annotations_dir": str(annotations_dir),
        "image_subdir": str(image_subdir),
    }


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
        return text


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{str(k): coerce_csv_value(v) for k, v in row.items()} for row in reader]


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
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


def read_completed_scene_rows_from_dirs(batch_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def read_manifest_rows(batch_dir: Path) -> list[dict[str, Any]]:
    return read_csv_rows(batch_dir / "manifest.csv")


def read_batch_per_scene_rows(batch_dir: Path) -> list[dict[str, Any]]:
    return read_csv_rows(batch_dir / "per_scene.csv")


def resolve_scene_schedule(
    *,
    candidate_scene_ids: list[str],
    batch_dir: Path,
    batch_size: int | None,
) -> tuple[list[str], list[str], str, list[dict[str, Any]], list[dict[str, Any]]]:
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
    manifest_rows: list[dict[str, Any]],
    per_scene_rows: list[dict[str, Any]],
) -> dict[str, str]:
    out = {str(k): str(v) for k, v in (base_scene_name_by_id or {}).items()}
    for row in list(manifest_rows or []) + list(per_scene_rows or []):
        scene_id = str(row.get("scene_id", "") or "").strip()
        scene_name = str(row.get("scene_name", "") or "").strip()
        if scene_id and scene_name:
            out[scene_id] = scene_name
    return out


def annotate_rows(rows: list[dict[str, Any]], *, run_id: str, scene_id: str, scene_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows or []:
        item = dict(row)
        item["run_id"] = str(run_id)
        item["scene_id"] = str(scene_id)
        item["scene_name"] = str(scene_name)
        out.append(item)
    return out


def build_scene_summary_row(
    *,
    run_id: str,
    scene_id: str,
    scene_name: str,
    results: dict[str, Any],
    scene_started_at: str | None = None,
    scene_finished_at: str | None = None,
    output_dir: Path | None = None,
    stable_min_frames: int | None = None,
    max_frames: int | None = None,
    model_size: str | None = None,
) -> dict[str, Any]:
    summary = dict(results.get("summary", {}) or {})
    identity = dict(results.get("tracking_identity_metrics", {}) or {})
    per_case = list(results.get("per_case", []) or [])
    per_orphan_pred = list(results.get("per_orphan_pred", []) or [])
    tracking_iou_sum = float(sum(float(row.get("tracking_iou", 0.0) or 0.0) for row in per_case))
    tracking_iou_sum_iou40 = float(sum(float(row.get("tracking_iou_iou40", 0.0) or 0.0) for row in per_case))
    orphan_pred_area_frac_sum = float(
        sum(float(row.get("pred_area_frac", 0.0) or 0.0) for row in per_orphan_pred if row.get("pred_area_frac", None) is not None)
    )

    return {
        "run_id": str(run_id),
        "scene_id": str(scene_id),
        "scene_name": str(scene_name),
        "status": "completed",
        "started_at": scene_started_at,
        "finished_at": scene_finished_at,
        "output_dir": None if output_dir is None else str(output_dir.resolve()),
        "tracker_family": "d4sm",
        "model_size": model_size,
        "stable_min_frames": stable_min_frames,
        "max_frames": max_frames,
        "n_frames": int(summary.get("n_frames", 0) or 0),
        "n_gt_objects": int(summary.get("n_objects", 0) or 0),
        "n_visible_gt_observations": int(summary.get("n_visible_gt_observations", 0) or 0),
        "n_matched_gt_observations": int(summary.get("n_matched_gt_observations", 0) or 0),
        "n_pred_observations_total": int(summary.get("n_pred_observations_total", 0) or 0),
        "n_orphan_pred_observations": int(summary.get("n_orphan_pred_observations", 0) or 0),
        "n_unique_orphan_pred_ids": int(summary.get("n_unique_orphan_pred_ids", 0) or 0),
        "n_orphan_only_pred_ids": int(summary.get("n_orphan_only_pred_ids", 0) or 0),
        "recovery_attempts_total": int(summary.get("recovery_attempts_total", 0) or 0),
        "recovery_success_reference_total": int(summary.get("recovery_success_reference_total", 0) or 0),
        "recovery_success_own_identity_total": int(summary.get("recovery_success_own_identity_total", 0) or 0),
        "recovery_success_duplicate_id_total": int(summary.get("recovery_success_duplicate_id_total", 0) or 0),
        "recovery_success_foreign_id_total": int(summary.get("recovery_success_foreign_id_total", 0) or 0),
        "recovery_rate_reference": summary.get("recovery_rate_reference", None),
        "recovery_rate_own_identity": summary.get("recovery_rate_own_identity", None),
        "recovery_rate_duplicate_id": summary.get("recovery_rate_duplicate_id", None),
        "recovery_rate_foreign_id": summary.get("recovery_rate_foreign_id", None),
        "orphan_pred_rate": summary.get("orphan_pred_rate", None),
        "mean_orphan_pred_area_frac": summary.get("mean_orphan_pred_area_frac", None),
        "orphan_pred_area_frac_sum": float(orphan_pred_area_frac_sum),
        "idtp": int(identity.get("idtp", 0) or 0),
        "idfp": int(identity.get("idfp", 0) or 0),
        "idfn": int(identity.get("idfn", 0) or 0),
        "idf1": identity.get("idf1", None),
        "idp": identity.get("idp", None),
        "idr": identity.get("idr", None),
        "n_matched_gt_observations_iou40": int(identity.get("n_matched_gt_observations_iou40", 0) or 0),
        "idtp_iou40": int(identity.get("idtp_iou40", 0) or 0),
        "idfp_iou40": int(identity.get("idfp_iou40", 0) or 0),
        "idfn_iou40": int(identity.get("idfn_iou40", 0) or 0),
        "idf1_iou40": identity.get("idf1_iou40", None),
        "idp_iou40": identity.get("idp_iou40", None),
        "idr_iou40": identity.get("idr_iou40", None),
        "idsw": int(identity.get("idsw", 0) or 0),
        "frag": int(identity.get("frag", 0) or 0),
        "tracking_recall": identity.get("tracking_recall", None),
        "tracking_iou_sum": float(tracking_iou_sum),
        "mean_tracking_iou": identity.get("mean_tracking_iou", None),
        "deta": identity.get("deta", None),
        "assa": identity.get("assa", None),
        "hota": identity.get("hota", None),
        "tracking_recall_iou40": identity.get("tracking_recall_iou40", None),
        "tracking_iou_sum_iou40": float(tracking_iou_sum_iou40),
        "mean_tracking_iou_iou40": identity.get("mean_tracking_iou_iou40", None),
        "deta_iou40": identity.get("deta_iou40", None),
        "assa_iou40": identity.get("assa_iou40", None),
        "hota_iou40": identity.get("hota_iou40", None),
        "n_unique_real_pred_tracks": int(summary.get("n_unique_real_pred_tracks", 0) or 0),
        "n_total_pred_tracks_including_orphan_only": int(summary.get("n_total_pred_tracks_including_orphan_only", 0) or 0),
        "n_pred_tracks_with_orphan_observations": int(summary.get("n_pred_tracks_with_orphan_observations", 0) or 0),
        "n_orphan_only_pred_tracks": int(summary.get("n_orphan_only_pred_tracks", 0) or 0),
        "pred_track_inflation_factor": summary.get("pred_track_inflation_factor", None),
        "global_frame_accuracy_strict": summary.get("global_frame_accuracy_strict", None),
        "global_frame_accuracy_permissive": summary.get("global_frame_accuracy_permissive", None),
        "global_object_accuracy_strict": summary.get("global_object_accuracy_strict", None),
        "global_object_accuracy_permissive": summary.get("global_object_accuracy_permissive", None),
        "global_frame_accuracy_strict_iou40": summary.get("global_frame_accuracy_strict_iou40", None),
        "global_frame_accuracy_permissive_iou40": summary.get("global_frame_accuracy_permissive_iou40", None),
        "global_object_accuracy_strict_iou40": summary.get("global_object_accuracy_strict_iou40", None),
        "global_object_accuracy_permissive_iou40": summary.get("global_object_accuracy_permissive_iou40", None),
        "n_mt_objects": int(summary.get("n_mt_objects", 0) or 0),
        "n_pt_objects": int(summary.get("n_pt_objects", 0) or 0),
        "n_ml_objects": int(summary.get("n_ml_objects", 0) or 0),
        "mt": summary.get("mt", None),
        "pt": summary.get("pt", None),
        "ml": summary.get("ml", None),
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


def build_event_rows(*, run_id: str, scene_id: str, scene_name: str, results: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
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
                "detail": f"stolen={ev.get('stolen_pred_id')} victim_after={ev.get('victim_pred_id_after')}",
            }
        )
    return out


def build_batch_report(*, summary_row: dict[str, Any], per_scene_rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("[BATCH][Summary]")
    for key in [
        "run_id",
        "model_size",
        "stable_min_frames",
        "n_scenes_completed",
        "n_frames",
        "n_gt_objects",
        "n_pred_observations_total",
        "n_orphan_pred_observations",
        "n_unique_orphan_pred_ids",
        "n_orphan_only_pred_ids",
        "n_total_pred_tracks_including_orphan_only",
        "recovery_attempts_total",
        "recovery_success_reference_total",
        "recovery_success_own_identity_total",
        "recovery_success_duplicate_id_total",
        "recovery_success_foreign_id_total",
        "idf1",
        "idp",
        "idr",
        "idf1_iou40",
        "idp_iou40",
        "idr_iou40",
        "idsw",
        "frag",
        "tracking_recall",
        "tracking_recall_iou40",
        "mean_tracking_iou",
        "mean_tracking_iou_iou40",
        "hota",
        "hota_iou40",
        "deta",
        "deta_iou40",
        "assa",
        "assa_iou40",
        "obs_weighted_idf1",
        "obs_weighted_idf1_iou40",
        "obs_weighted_hota",
        "obs_weighted_hota_iou40",
        "obs_weighted_tracking_recall",
        "obs_weighted_tracking_recall_iou40",
        "scene_macro_idf1",
        "scene_macro_idf1_iou40",
        "scene_macro_hota",
        "scene_macro_hota_iou40",
        "scene_macro_tracking_recall",
        "scene_macro_tracking_recall_iou40",
        "object_macro_strict_accuracy",
        "object_macro_strict_accuracy_iou40",
        "object_macro_permissive_accuracy",
        "object_macro_permissive_accuracy_iou40",
        "object_macro_tracking_recall",
        "object_macro_tracking_recall_iou40",
        "object_macro_tracking_iou",
        "object_macro_tracking_iou_iou40",
        "recovery_rate_reference",
        "recovery_rate_own_identity",
        "recovery_rate_duplicate_id",
        "recovery_rate_foreign_id",
        "orphan_pred_rate",
        "mean_orphan_pred_area_frac",
        "pred_track_inflation_factor",
        "n_pred_tracks_with_orphan_observations",
        "n_orphan_only_pred_tracks",
        "mt",
        "pt",
        "ml",
        "total_runtime_seconds",
        "avg_loop_ms",
    ]:
        value = summary_row.get(key, None)
        if key in {
            "idf1", "idp", "idr", "idf1_iou40", "idp_iou40", "idr_iou40",
            "tracking_recall", "tracking_recall_iou40",
            "hota", "hota_iou40",
            "deta", "deta_iou40",
            "assa", "assa_iou40",
            "obs_weighted_idf1", "obs_weighted_idf1_iou40",
            "obs_weighted_hota", "obs_weighted_hota_iou40",
            "obs_weighted_tracking_recall", "obs_weighted_tracking_recall_iou40",
            "scene_macro_idf1", "scene_macro_idf1_iou40",
            "scene_macro_hota", "scene_macro_hota_iou40",
            "scene_macro_tracking_recall", "scene_macro_tracking_recall_iou40",
            "object_macro_strict_accuracy", "object_macro_strict_accuracy_iou40",
            "object_macro_permissive_accuracy", "object_macro_permissive_accuracy_iou40",
            "object_macro_tracking_recall", "object_macro_tracking_recall_iou40",
            "recovery_rate_reference", "recovery_rate_own_identity",
            "recovery_rate_duplicate_id", "recovery_rate_foreign_id",
            "orphan_pred_rate",
            "mt", "pt", "ml",
        }:
            value = fmt_pct(value)
        elif key == "pred_track_inflation_factor" and value is not None:
            value = f"{float(value):.3f}x"
        elif key in {
            "mean_tracking_iou", "mean_tracking_iou_iou40",
            "object_macro_tracking_iou", "object_macro_tracking_iou_iou40",
            "mean_orphan_pred_area_frac",
        } and value is not None:
            value = f"{float(value):.4f}"
        elif key == "avg_loop_ms" and value is not None:
            value = f"{float(value):.2f}"
        lines.append(f"  {key}: {value}")

    worst_by_idf1 = sorted(
        per_scene_rows,
        key=lambda row: float(row.get("idf1", -1.0) if row.get("idf1", None) is not None else -1.0),
    )[:5]
    lines.append("")
    lines.append("[BATCH][WorstScenesByIDF1]")
    if worst_by_idf1:
        lines.append(
            render_table(
                [
                    {
                        "scene": row.get("scene_name", row.get("scene_id", "")),
                        "idf1": fmt_pct(row.get("idf1", None)),
                        "idf140": fmt_pct(row.get("idf1_iou40", None)),
                        "trkrec": fmt_pct(row.get("tracking_recall", None)),
                        "trkrec40": fmt_pct(row.get("tracking_recall_iou40", None)),
                        "infl": "-" if row.get("pred_track_inflation_factor", None) is None else f"{float(row.get('pred_track_inflation_factor')):.2f}x",
                    }
                    for row in worst_by_idf1
                ],
                [
                    ("scene", "Scene"),
                    ("idf1", "IDF1"),
                    ("idf140", "IDF1@0.4"),
                    ("trkrec", "TrackRec"),
                    ("trkrec40", "TrackRec@0.4"),
                    ("infl", "Infl"),
                ],
            )
        )
    else:
        lines.append("No completed scenes.")

    worst_by_infl = sorted(
        per_scene_rows,
        key=lambda row: float(row.get("pred_track_inflation_factor", -1.0) if row.get("pred_track_inflation_factor", None) is not None else -1.0),
        reverse=True,
    )[:5]
    lines.append("")
    lines.append("[BATCH][WorstScenesByInflation]")
    if worst_by_infl:
        lines.append(
            render_table(
                [
                    {
                        "scene": row.get("scene_name", row.get("scene_id", "")),
                        "infl": "-" if row.get("pred_track_inflation_factor", None) is None else f"{float(row.get('pred_track_inflation_factor')):.2f}x",
                        "idf1": fmt_pct(row.get("idf1", None)),
                        "frag": row.get("frag", None),
                        "swaps": row.get("swap_events_total", None),
                    }
                    for row in worst_by_infl
                ],
                [
                    ("scene", "Scene"),
                    ("infl", "Infl"),
                    ("idf1", "IDF1"),
                    ("frag", "Frag"),
                    ("swaps", "Swaps"),
                ],
            )
        )
    else:
        lines.append("No completed scenes.")

    return "\n".join(lines)


def aggregate_global_summary(
    *,
    run_id: str,
    per_scene_rows: list[dict[str, Any]],
    per_object_rows: list[dict[str, Any]],
    per_case_rows: list[dict[str, Any]],
    per_event_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def _mean_metric(rows: list[dict[str, Any]], key: str) -> float | None:
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

    def _single_value(rows: list[dict[str, Any]], key: str):
        values = {
            str(row.get(key, "")).strip()
            for row in rows
            if row.get(key, None) not in (None, "")
            and str(row.get(key, "")).strip()
        }
        if not values:
            return None
        if len(values) == 1:
            return next(iter(values))
        return ",".join(sorted(values))

    n_scenes_completed = int(len(per_scene_rows))
    n_frames = int(sum(int(row.get("n_frames", 0) or 0) for row in per_scene_rows))
    n_gt_objects = int(sum(int(row.get("n_gt_objects", 0) or 0) for row in per_scene_rows))
    n_visible_gt_observations = int(sum(int(row.get("n_visible_gt_observations", 0) or 0) for row in per_scene_rows))
    n_matched_gt_observations = int(sum(int(row.get("n_matched_gt_observations", 0) or 0) for row in per_scene_rows))
    n_pred_observations_total = int(sum(int(row.get("n_pred_observations_total", 0) or 0) for row in per_scene_rows))
    n_orphan_pred_observations = int(sum(int(row.get("n_orphan_pred_observations", 0) or 0) for row in per_scene_rows))
    n_unique_orphan_pred_ids = int(sum(int(row.get("n_unique_orphan_pred_ids", 0) or 0) for row in per_scene_rows))
    n_orphan_only_pred_ids = int(sum(int(row.get("n_orphan_only_pred_ids", 0) or 0) for row in per_scene_rows))
    recovery_attempts_total = int(sum(int(row.get("recovery_attempts_total", 0) or 0) for row in per_scene_rows))
    recovery_success_reference_total = int(sum(int(row.get("recovery_success_reference_total", 0) or 0) for row in per_scene_rows))
    recovery_success_own_identity_total = int(sum(int(row.get("recovery_success_own_identity_total", 0) or 0) for row in per_scene_rows))
    recovery_success_duplicate_id_total = int(sum(int(row.get("recovery_success_duplicate_id_total", 0) or 0) for row in per_scene_rows))
    recovery_success_foreign_id_total = int(sum(int(row.get("recovery_success_foreign_id_total", 0) or 0) for row in per_scene_rows))
    orphan_pred_area_frac_sum = float(sum(float(row.get("orphan_pred_area_frac_sum", 0.0) or 0.0) for row in per_scene_rows))
    n_matched_gt_observations_iou40 = int(sum(int(row.get("n_matched_gt_observations_iou40", 0) or 0) for row in per_scene_rows))
    idtp = int(sum(int(row.get("idtp", 0) or 0) for row in per_scene_rows))
    idfp = int(sum(int(row.get("idfp", 0) or 0) for row in per_scene_rows))
    idfn = int(sum(int(row.get("idfn", 0) or 0) for row in per_scene_rows))
    idtp_iou40 = int(sum(int(row.get("idtp_iou40", 0) or 0) for row in per_scene_rows))
    idfp_iou40 = int(sum(int(row.get("idfp_iou40", 0) or 0) for row in per_scene_rows))
    idfn_iou40 = int(sum(int(row.get("idfn_iou40", 0) or 0) for row in per_scene_rows))
    tracking_iou_sum = float(sum(float(row.get("tracking_iou_sum", 0.0) or 0.0) for row in per_scene_rows))
    tracking_iou_sum_iou40 = float(sum(float(row.get("tracking_iou_sum_iou40", 0.0) or 0.0) for row in per_scene_rows))
    n_unique_real_pred_tracks = int(sum(int(row.get("n_unique_real_pred_tracks", 0) or 0) for row in per_scene_rows))
    n_total_pred_tracks_including_orphan_only = int(sum(int(row.get("n_total_pred_tracks_including_orphan_only", 0) or 0) for row in per_scene_rows))
    n_tracking_correct = int(sum(1 for row in per_case_rows if bool(row.get("strict_global_correct", False))))
    n_tracking_correct_iou40 = int(sum(1 for row in per_case_rows if bool(row.get("strict_global_correct_iou40", False))))
    total_runtime_seconds = float(sum(float(row.get("total_runtime_seconds", 0.0) or 0.0) for row in per_scene_rows))
    total_loop_ms = float(sum(float(row.get("total_loop_ms", 0.0) or 0.0) for row in per_scene_rows))

    return {
        "run_id": str(run_id),
        "model_size": _single_value(per_scene_rows, "model_size"),
        "stable_min_frames": (
            None
            if _single_value(per_scene_rows, "stable_min_frames") is None
            else int(_single_value(per_scene_rows, "stable_min_frames"))
        ),
        "n_scenes_completed": int(n_scenes_completed),
        "n_frames": int(n_frames),
        "n_gt_objects": int(n_gt_objects),
        "n_visible_gt_observations": int(n_visible_gt_observations),
        "n_matched_gt_observations": int(n_matched_gt_observations),
        "n_pred_observations_total": int(n_pred_observations_total),
        "n_orphan_pred_observations": int(n_orphan_pred_observations),
        "n_unique_orphan_pred_ids": int(n_unique_orphan_pred_ids),
        "n_orphan_only_pred_ids": int(n_orphan_only_pred_ids),
        "recovery_attempts_total": int(recovery_attempts_total),
        "recovery_success_reference_total": int(recovery_success_reference_total),
        "recovery_success_own_identity_total": int(recovery_success_own_identity_total),
        "recovery_success_duplicate_id_total": int(recovery_success_duplicate_id_total),
        "recovery_success_foreign_id_total": int(recovery_success_foreign_id_total),
        "recovery_rate_reference": (float(recovery_success_reference_total) / float(recovery_attempts_total)) if recovery_attempts_total > 0 else None,
        "recovery_rate_own_identity": (float(recovery_success_own_identity_total) / float(recovery_attempts_total)) if recovery_attempts_total > 0 else None,
        "recovery_rate_duplicate_id": (float(recovery_success_duplicate_id_total) / float(recovery_attempts_total)) if recovery_attempts_total > 0 else None,
        "recovery_rate_foreign_id": (float(recovery_success_foreign_id_total) / float(recovery_attempts_total)) if recovery_attempts_total > 0 else None,
        "orphan_pred_rate": (float(n_orphan_pred_observations) / float(n_pred_observations_total)) if n_pred_observations_total > 0 else None,
        "mean_orphan_pred_area_frac": (float(orphan_pred_area_frac_sum) / float(n_orphan_pred_observations)) if n_orphan_pred_observations > 0 else None,
        "idtp": int(idtp),
        "idfp": int(idfp),
        "idfn": int(idfn),
        "idf1": (2.0 * float(idtp) / float(2 * idtp + idfp + idfn)) if (2 * idtp + idfp + idfn) > 0 else None,
        "idp": (float(idtp) / float(idtp + idfp)) if (idtp + idfp) > 0 else None,
        "idr": (float(idtp) / float(idtp + idfn)) if (idtp + idfn) > 0 else None,
        "n_matched_gt_observations_iou40": int(n_matched_gt_observations_iou40),
        "idtp_iou40": int(idtp_iou40),
        "idfp_iou40": int(idfp_iou40),
        "idfn_iou40": int(idfn_iou40),
        "idf1_iou40": (
            2.0 * float(idtp_iou40) / float(2 * idtp_iou40 + idfp_iou40 + idfn_iou40)
        ) if (2 * idtp_iou40 + idfp_iou40 + idfn_iou40) > 0 else None,
        "idp_iou40": (float(idtp_iou40) / float(idtp_iou40 + idfp_iou40)) if (idtp_iou40 + idfp_iou40) > 0 else None,
        "idr_iou40": (float(idtp_iou40) / float(idtp_iou40 + idfn_iou40)) if (idtp_iou40 + idfn_iou40) > 0 else None,
        "idsw": int(sum(int(row.get("idsw", 0) or 0) for row in per_scene_rows)),
        "frag": int(sum(int(row.get("frag", 0) or 0) for row in per_scene_rows)),
        "tracking_recall": (float(n_tracking_correct) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else None,
        "mean_tracking_iou": (float(tracking_iou_sum) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else None,
        "deta": (float(n_matched_gt_observations) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else None,
        "assa": (float(n_tracking_correct) / float(n_matched_gt_observations)) if n_matched_gt_observations > 0 else None,
        "hota": (
            float(
                (
                    ((float(n_matched_gt_observations) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else 0.0)
                    * ((float(n_tracking_correct) / float(n_matched_gt_observations)) if n_matched_gt_observations > 0 else 0.0)
                ) ** 0.5
            )
            if n_visible_gt_observations > 0 and n_matched_gt_observations > 0
            else None
        ),
        "tracking_recall_iou40": (float(n_tracking_correct_iou40) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else None,
        "mean_tracking_iou_iou40": (float(tracking_iou_sum_iou40) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else None,
        "deta_iou40": (float(n_matched_gt_observations_iou40) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else None,
        "assa_iou40": (float(n_tracking_correct_iou40) / float(n_matched_gt_observations_iou40)) if n_matched_gt_observations_iou40 > 0 else None,
        "hota_iou40": (
            float(
                (
                    ((float(n_matched_gt_observations_iou40) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else 0.0)
                    * ((float(n_tracking_correct_iou40) / float(n_matched_gt_observations_iou40)) if n_matched_gt_observations_iou40 > 0 else 0.0)
                ) ** 0.5
            )
            if n_visible_gt_observations > 0 and n_matched_gt_observations_iou40 > 0
            else None
        ),
        "n_unique_real_pred_tracks": int(n_unique_real_pred_tracks),
        "n_total_pred_tracks_including_orphan_only": int(n_total_pred_tracks_including_orphan_only),
        "n_pred_tracks_with_orphan_observations": int(sum(int(row.get("n_pred_tracks_with_orphan_observations", 0) or 0) for row in per_scene_rows)),
        "n_orphan_only_pred_tracks": int(sum(int(row.get("n_orphan_only_pred_tracks", 0) or 0) for row in per_scene_rows)),
        "pred_track_inflation_factor": (float(n_unique_real_pred_tracks) / float(n_gt_objects)) if n_gt_objects > 0 else None,
        "mt": (float(sum(int(row.get("n_mt_objects", 0) or 0) for row in per_scene_rows)) / float(n_gt_objects)) if n_gt_objects > 0 else None,
        "pt": (float(sum(int(row.get("n_pt_objects", 0) or 0) for row in per_scene_rows)) / float(n_gt_objects)) if n_gt_objects > 0 else None,
        "ml": (float(sum(int(row.get("n_ml_objects", 0) or 0) for row in per_scene_rows)) / float(n_gt_objects)) if n_gt_objects > 0 else None,
        "swap_events_total": int(sum(1 for row in per_event_rows if str(row.get("event_type", "")) == "swap")),
        "theft_with_new_id_total": int(sum(1 for row in per_event_rows if str(row.get("event_type", "")) == "theft_with_new_id")),
        "theft_with_displacement_total": int(sum(1 for row in per_event_rows if str(row.get("event_type", "")) == "theft_with_displacement")),
        "total_runtime_seconds": float(total_runtime_seconds),
        "avg_runtime_seconds": (float(total_runtime_seconds) / float(n_scenes_completed)) if n_scenes_completed > 0 else None,
        "total_loop_ms": float(total_loop_ms),
        "avg_loop_ms": (float(total_loop_ms) / float(n_frames)) if n_frames > 0 else None,
        "obs_weighted_idf1": (2.0 * float(idtp) / float(2 * idtp + idfp + idfn)) if (2 * idtp + idfp + idfn) > 0 else None,
        "obs_weighted_idf1_iou40": (
            2.0 * float(idtp_iou40) / float(2 * idtp_iou40 + idfp_iou40 + idfn_iou40)
        ) if (2 * idtp_iou40 + idfp_iou40 + idfn_iou40) > 0 else None,
        "obs_weighted_hota": (
            float(
                (
                    ((float(n_matched_gt_observations) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else 0.0)
                    * ((float(n_tracking_correct) / float(n_matched_gt_observations)) if n_matched_gt_observations > 0 else 0.0)
                ) ** 0.5
            )
            if n_visible_gt_observations > 0 and n_matched_gt_observations > 0
            else None
        ),
        "obs_weighted_hota_iou40": (
            float(
                (
                    ((float(n_matched_gt_observations_iou40) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else 0.0)
                    * ((float(n_tracking_correct_iou40) / float(n_matched_gt_observations_iou40)) if n_matched_gt_observations_iou40 > 0 else 0.0)
                ) ** 0.5
            )
            if n_visible_gt_observations > 0 and n_matched_gt_observations_iou40 > 0
            else None
        ),
        "obs_weighted_tracking_recall": (float(n_tracking_correct) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else None,
        "obs_weighted_tracking_recall_iou40": (float(n_tracking_correct_iou40) / float(n_visible_gt_observations)) if n_visible_gt_observations > 0 else None,
        "scene_macro_idf1": _mean_metric(per_scene_rows, "idf1"),
        "scene_macro_idf1_iou40": _mean_metric(per_scene_rows, "idf1_iou40"),
        "scene_macro_hota": _mean_metric(per_scene_rows, "hota"),
        "scene_macro_hota_iou40": _mean_metric(per_scene_rows, "hota_iou40"),
        "scene_macro_tracking_recall": _mean_metric(per_scene_rows, "tracking_recall"),
        "scene_macro_tracking_recall_iou40": _mean_metric(per_scene_rows, "tracking_recall_iou40"),
        "object_macro_strict_accuracy": _mean_metric(per_object_rows, "strict_accuracy"),
        "object_macro_strict_accuracy_iou40": _mean_metric(per_object_rows, "strict_accuracy_iou40"),
        "object_macro_permissive_accuracy": _mean_metric(per_object_rows, "permissive_accuracy"),
        "object_macro_permissive_accuracy_iou40": _mean_metric(per_object_rows, "permissive_accuracy_iou40"),
        "object_macro_tracking_recall": _mean_metric(per_object_rows, "tracking_recall_object"),
        "object_macro_tracking_recall_iou40": _mean_metric(per_object_rows, "tracking_recall_object_iou40"),
        "object_macro_tracking_iou": _mean_metric(per_object_rows, "mean_tracking_iou_object"),
        "object_macro_tracking_iou_iou40": _mean_metric(per_object_rows, "mean_tracking_iou_object_iou40"),
    }


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

    aggregated_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in SCENE_TABLE_FILES.keys()}
    for scene_dir in iter_scene_dirs(batch_dir):
        if not scene_dir_is_complete(scene_dir):
            continue
        for table_key, filename in SCENE_TABLE_FILES.items():
            aggregated_rows[table_key].extend(read_csv_rows(scene_dir / filename))

    per_scene_rows = list(aggregated_rows["per_scene"])
    per_object_rows = list(aggregated_rows["per_object"])
    per_case_rows = list(aggregated_rows["per_case"])
    per_event_rows = list(aggregated_rows["per_event"])
    selected_scene_id_set = {str(scene_id) for scene_id in selected_scene_ids}
    selected_per_scene_rows = [
        row for row in per_scene_rows if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_case_rows = [
        row for row in per_case_rows if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_object_rows = [
        row for row in per_object_rows if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    selected_per_event_rows = [
        row for row in per_event_rows if str(row.get("scene_id", "") or "") in selected_scene_id_set
    ]
    summary_row = aggregate_global_summary(
        run_id=run_id,
        per_scene_rows=per_scene_rows,
        per_object_rows=per_object_rows,
        per_case_rows=per_case_rows,
        per_event_rows=per_event_rows,
    )
    selected_summary_row = aggregate_global_summary(
        run_id=run_id,
        per_scene_rows=selected_per_scene_rows,
        per_object_rows=selected_per_object_rows,
        per_case_rows=selected_per_case_rows,
        per_event_rows=selected_per_event_rows,
    )

    write_single_row_csv(batch_dir / "summary_global.csv", summary_row)
    write_single_row_csv(batch_dir / "summary_current_batch.csv", selected_summary_row)
    write_csv(batch_dir / "per_scene.csv", per_scene_rows)
    write_csv(batch_dir / "per_scene_current_batch.csv", selected_per_scene_rows)
    write_csv(batch_dir / "per_class.csv", aggregated_rows["per_class"])
    write_csv(batch_dir / "per_object.csv", aggregated_rows["per_object"])
    write_csv(batch_dir / "per_case.csv", per_case_rows)
    write_csv(batch_dir / "per_frame.csv", aggregated_rows["per_frame"])
    write_csv(batch_dir / "per_pred_track.csv", aggregated_rows["per_pred_track"])
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
    results: dict[str, Any],
    scene_report: str,
    scene_started_at: str | None = None,
    scene_finished_at: str | None = None,
    stable_min_frames: int | None = None,
    max_frames: int | None = None,
    model_size: str | None = None,
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
        model_size=model_size,
    )
    write_single_row_csv(temp_scene_dir / SCENE_TABLE_FILES["per_scene"], scene_summary_row)
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_class"], annotate_rows(results.get("per_class", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_object"], annotate_rows(results.get("per_object", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_case"], annotate_rows(results.get("per_case", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_frame"], annotate_rows(results.get("per_frame", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_pred_track"], annotate_rows(results.get("per_pred_track", []) or [], run_id=run_id, scene_id=scene_id, scene_name=scene_name))
    write_csv(temp_scene_dir / SCENE_TABLE_FILES["per_event"], build_event_rows(run_id=run_id, scene_id=scene_id, scene_name=scene_name, results=results))
    write_text(temp_scene_dir / "report.txt", str(scene_report).rstrip() + "\n")


def main() -> None:
    config_path = SRC_DIR / "config" / "default_config.yaml"
    masks_root_base = Path(
        os.environ.get(
            "APP2_SCANNETPP_MASKS_ROOT",
            "/media/pablo/LINUX/Qsync/2026_tracker_reid/datasets/scannetpp_data",
        )
    ).expanduser().resolve()
    images_root_base = Path(
        os.environ.get(
            "APP2_SCANNETPP_IMAGES_ROOT",
            "/media/pablo/LINUX/Qsync/2026_tracker_reid/datasets/scannetpp_data",
        )
    ).expanduser().resolve()
    image_subdir = os.environ.get("APP2_IMAGE_SUBDIR", "dslr/resized_images").strip() or "dslr/resized_images"
    mask_variant = os.environ.get("APP2_MASK_VARIANT", "benchmark").strip().lower() or "benchmark"
    if mask_variant == "benchmark":
        mask_variant = "benchmark_instance"

    scene_ids = resolve_batch_scene_ids(
        masks_root_base=masks_root_base,
        images_root_base=images_root_base,
        mask_variant=mask_variant,
        image_subdir=image_subdir,
    )
    scene_ids = unique_preserve_order([str(scene_id) for scene_id in scene_ids])
    if not scene_ids:
        raise RuntimeError("No se resolvieron escenas para el batch de d4sm.")

    run_id = "d4sm"
    output_root = (PROJECT_DIR / "outputs" / "d4sm" / "testing_batch").resolve()
    batch_dir = output_root
    scenes_root = batch_dir / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)

    max_scenes = _env_int("APP2_D4SM_BATCH_MAX_SCENES", None)
    batch_size = _env_int("APP2_D4SM_BATCH_SIZE", max_scenes)
    (
        scene_ids,
        registered_scene_ids,
        selection_mode,
        existing_manifest_rows,
        existing_per_scene_rows,
    ) = resolve_scene_schedule(
        candidate_scene_ids=scene_ids,
        batch_dir=batch_dir,
        batch_size=batch_size,
    )

    stable_min_frames = int(_env_int("APP2_D4SM_BATCH_STABLE_MIN_FRAMES", 3) or 3)
    max_frames = _env_int("APP2_D4SM_BATCH_MAX_FRAMES", None)
    model_size = os.environ.get("APP2_D4SM_MODEL_SIZE", "large").strip().lower() or "large"

    write_single_row_csv(
        batch_dir / "run_config.csv",
        {
            "run_id": str(run_id),
            "batch_name": str(run_id),
            "batch_dir": str(batch_dir.resolve()),
            "created_at": _now_iso(),
            "tracker_family": "d4sm",
            "model_size": str(model_size),
            "stable_min_frames": int(stable_min_frames),
            "max_frames": None if max_frames is None else int(max_frames),
            "max_scenes": None if max_scenes is None else int(max_scenes),
            "batch_size": None if batch_size is None else int(batch_size),
            "selection_mode": str(selection_mode),
            "mask_variant": str(mask_variant),
            "image_subdir": str(image_subdir),
            "masks_root_base": str(masks_root_base.resolve()),
            "images_root_base": str(images_root_base.resolve()),
            "selected_scene_count": int(len(scene_ids)),
            "selected_scene_ids": [str(scene_id) for scene_id in scene_ids],
            "registered_scene_count": int(len(registered_scene_ids)),
            "registered_scene_ids": [str(scene_id) for scene_id in registered_scene_ids],
        },
    )

    tracker_runtime_config = resolve_d4sm_runtime_config()
    print("[D4SM][BATCH] Config:")
    print(f"[D4SM][BATCH] Output dir -> {batch_dir}")
    print(f"[D4SM][BATCH] Model size -> {tracker_runtime_config['model_size']}")
    print(f"[D4SM][BATCH] Checkpoint dir -> {tracker_runtime_config['checkpoint_dir']}")
    print(f"[D4SM][BATCH] Stable min frames -> {stable_min_frames}")
    print(f"[D4SM][BATCH] Max scenes -> {max_scenes}")
    print(f"[D4SM][BATCH] Batch size -> {batch_size}")
    print(f"[D4SM][BATCH] Masks root -> {masks_root_base}")
    print(f"[D4SM][BATCH] Images root -> {images_root_base}")
    print("[D4SM][BATCH] Loading tracker weights once...")
    shared_tracker = create_d4sm_tracker(runtime_config=tracker_runtime_config)
    print("[D4SM][BATCH] Tracker ready.")

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
        print("[D4SM][BATCH] No hay escenas pendientes para esta ejecucion.")
        return

    for scene_id in scene_ids:
        scene_key = sanitize_name_for_path(str(scene_id))
        final_scene_dir = scenes_root / scene_key
        if scene_dir_is_complete(final_scene_dir):
            print(f"[D4SM][BATCH] Skip completed scene -> {scene_id}")
            continue
        if final_scene_dir.exists():
            incomplete_backup_dir = reserve_incomplete_scene_backup_dir(final_scene_dir)
            final_scene_dir.rename(incomplete_backup_dir)
            print(
                f"[D4SM][BATCH] Incomplete final output moved -> {scene_id} "
                f"({final_scene_dir} -> {incomplete_backup_dir})"
            )

        temp_scene_dir = scenes_root / f".tmp_{scene_key}"
        if temp_scene_dir.exists():
            shutil.rmtree(temp_scene_dir)

        scene_started_at = _now_iso()
        try:
            input_source = build_scene_input_source(
                scene_id=str(scene_id),
                masks_root_base=masks_root_base,
                images_root_base=images_root_base,
                mask_variant=mask_variant,
                image_subdir=image_subdir,
            )
            print(f"[D4SM][BATCH] Scene start -> {scene_id}")
            print(f"[D4SM][BATCH] Source mode -> {input_source.get('mode', '')}")
            print(f"[D4SM][BATCH] Frames dir -> {input_source.get('frames_dir', '')}")
            if input_source.get("image_subdir"):
                print(f"[D4SM][BATCH] Image subdir -> {input_source['image_subdir']}")
            if input_source.get("data_tar_path"):
                print(f"[D4SM][BATCH] Data tar -> {input_source['data_tar_path']}")
            if input_source.get("annotations_tar_path"):
                print(f"[D4SM][BATCH] Annotations tar -> {input_source['annotations_tar_path']}")
            if input_source.get("davis_meta_path"):
                print(f"[D4SM][BATCH] DAVIS meta -> {input_source['davis_meta_path']}")
            if input_source.get("davis_annotations_dir"):
                print(f"[D4SM][BATCH] DAVIS annotations -> {input_source['davis_annotations_dir']}")
            results, scene_report = evaluate_scene(
                project_dir=PROJECT_DIR,
                config_path=config_path,
                input_source=input_source,
                stable_min_frames=stable_min_frames,
                max_frames=max_frames,
                tracker=shared_tracker,
            )
            scene_name = str(input_source.get("sequence_name", scene_id))
            scene_name_by_id[str(scene_id)] = scene_name
            failed_scene_errors.pop(str(scene_id), None)
            write_scene_outputs(
                temp_scene_dir=temp_scene_dir,
                final_scene_dir=final_scene_dir,
                run_id=run_id,
                scene_id=str(scene_id),
                scene_name=scene_name,
                results=results,
                scene_report=scene_report,
                scene_started_at=scene_started_at,
                scene_finished_at=_now_iso(),
                stable_min_frames=stable_min_frames,
                max_frames=max_frames,
                model_size=model_size,
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
            print(f"[D4SM][BATCH] Scene completed -> {scene_id} ({scene_started_at} -> {_now_iso()})")
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
            print(f"[D4SM][BATCH][ERROR] Scene failed -> {scene_id}: {exc}")

    shared_tracker = None
    print("[D4SM][BATCH] Done.")


if __name__ == "__main__":
    main()
