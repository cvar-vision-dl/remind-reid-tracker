from __future__ import annotations

import ast
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent.parent
PROJECT_DIR = SRC_DIR.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _parse_optional_int(raw: Any) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text or text.lower() == "none":
        return None
    return int(float(text))


def _safe_pct(num: int, den: int) -> float | None:
    if int(den) <= 0:
        return None
    return float(num) / float(den)


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        return [], []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [{str(k): ("" if v is None else str(v)) for k, v in row.items()} for row in reader]
    return fieldnames, rows


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _append_missing_fields(fieldnames: list[str], new_fields: list[str]) -> list[str]:
    out = list(fieldnames)
    for field in new_fields:
        if field not in out:
            out.append(field)
    return out


def _parse_timeline(raw: str) -> list[int | None]:
    text = str(raw or "").strip()
    if not text:
        return []
    value = ast.literal_eval(text)
    if not isinstance(value, list):
        return []
    out: list[int | None] = []
    for item in value:
        if item is None:
            out.append(None)
        else:
            out.append(int(item))
    return out


def _build_canonical_gt_by_pred(per_case_rows: list[dict[str, str]]) -> dict[int, int]:
    owner_votes_by_pred: dict[int, Counter[int]] = defaultdict(Counter)
    for row in per_case_rows:
        pred_id = _parse_optional_int(row.get("pred_object_id"))
        gt_id = _parse_optional_int(row.get("gt_instance_id"))
        if pred_id is None or gt_id is None:
            continue
        owner_votes_by_pred[int(pred_id)][int(gt_id)] += 1

    out: dict[int, int] = {}
    for pred_id, votes in owner_votes_by_pred.items():
        best_gt_id, _count = sorted(votes.items(), key=lambda item: (-int(item[1]), int(item[0])))[0]
        out[int(pred_id)] = int(best_gt_id)
    return out


def _compute_recovery_fields(
    *,
    row: dict[str, str],
    canonical_gt_by_pred: dict[int, int],
) -> dict[str, Any]:
    gt_id = _parse_optional_int(row.get("gt_instance_id"))
    if gt_id is None:
        return {}

    assigned_ref_pred = _parse_optional_int(row.get("reference_pred_id"))
    pred_timeline = _parse_timeline(row.get("pred_ids_timeline", ""))
    frames_timeline = _parse_timeline(row.get("frames_timeline", ""))
    if len(pred_timeline) != len(frames_timeline):
        return {}

    strict_flags = [
        bool(assigned_ref_pred is not None and pred_id is not None and int(pred_id) == int(assigned_ref_pred))
        for pred_id in pred_timeline
    ]

    first_failure_idx = None
    for idx, ok in enumerate(strict_flags):
        if not ok:
            first_failure_idx = int(idx)
            break

    recovery_start_indices: list[int] = []
    in_recovery_run = False
    prev_visible_frame_id = None
    for idx, pred_id in enumerate(pred_timeline):
        frame_id_cur = int(frames_timeline[int(idx)])
        has_visibility_gap = bool(
            prev_visible_frame_id is not None and frame_id_cur > int(prev_visible_frame_id) + 1
        )
        if has_visibility_gap:
            in_recovery_run = False
        cur_tracked = pred_id is not None
        if cur_tracked and not in_recovery_run:
            recovery_start_indices.append(int(idx))
            in_recovery_run = True
        elif not cur_tracked:
            in_recovery_run = False
        prev_visible_frame_id = int(frame_id_cur)

    recovery_only_start_indices = list(recovery_start_indices[1:])
    recovery_attempts = int(len(recovery_only_start_indices))
    recovery_success_reference = 0
    recovery_success_own_identity = 0
    recovery_success_duplicate_id = 0
    recovery_success_foreign_id = 0

    for idx in recovery_only_start_indices:
        pred_id = pred_timeline[int(idx)]
        if pred_id is None:
            continue
        owner = canonical_gt_by_pred.get(int(pred_id), None)
        is_reference = bool(assigned_ref_pred is not None and int(pred_id) == int(assigned_ref_pred))
        is_own_identity = bool(owner == int(gt_id))
        if is_reference:
            recovery_success_reference += 1
            recovery_success_own_identity += 1
        elif is_own_identity:
            recovery_success_own_identity += 1
            recovery_success_duplicate_id += 1
        else:
            recovery_success_foreign_id += 1

    recovered_reference = False
    recovered_own_identity = False
    post_failure_strict_accuracy = None
    if first_failure_idx is not None and first_failure_idx + 1 < len(pred_timeline):
        tail_pred_ids = pred_timeline[first_failure_idx + 1 :]
        tail_strict = [
            bool(assigned_ref_pred is not None and pred_id is not None and int(pred_id) == int(assigned_ref_pred))
            for pred_id in tail_pred_ids
        ]
        tail_own = [
            bool(pred_id is not None and canonical_gt_by_pred.get(int(pred_id), None) == int(gt_id))
            for pred_id in tail_pred_ids
        ]
        recovered_reference = any(tail_strict)
        recovered_own_identity = any(tail_own)
        post_failure_strict_accuracy = float(sum(1 for ok in tail_strict if ok)) / float(len(tail_strict))
    elif first_failure_idx is not None:
        post_failure_strict_accuracy = 0.0

    return {
        "recovered_reference": bool(recovered_reference),
        "recovered_own_identity": bool(recovered_own_identity),
        "recovery_attempts": int(recovery_attempts),
        "recovery_success_reference": int(recovery_success_reference),
        "recovery_success_own_identity": int(recovery_success_own_identity),
        "recovery_success_duplicate_id": int(recovery_success_duplicate_id),
        "recovery_success_foreign_id": int(recovery_success_foreign_id),
        "recovery_rate_reference": _safe_pct(recovery_success_reference, recovery_attempts),
        "recovery_rate_own_identity": _safe_pct(recovery_success_own_identity, recovery_attempts),
        "recovery_rate_duplicate_id": _safe_pct(recovery_success_duplicate_id, recovery_attempts),
        "recovery_rate_foreign_id": _safe_pct(recovery_success_foreign_id, recovery_attempts),
        "post_failure_strict_accuracy": post_failure_strict_accuracy,
    }


def _backfill_scene(scene_dir: Path) -> bool:
    per_object_path = scene_dir / "per_object.csv"
    per_case_path = scene_dir / "per_case.csv"
    scene_summary_path = scene_dir / "scene_summary.csv"
    per_class_path = scene_dir / "per_class.csv"
    if not per_object_path.is_file() or not per_case_path.is_file() or not scene_summary_path.is_file():
        return False

    per_object_fields, per_object_rows = _read_csv_rows(per_object_path)
    _per_case_fields, per_case_rows = _read_csv_rows(per_case_path)
    if not per_object_rows or not per_case_rows:
        return False

    canonical_gt_by_pred = _build_canonical_gt_by_pred(per_case_rows)
    recovery_fields = [
        "recovered_reference",
        "recovered_own_identity",
        "recovery_attempts",
        "recovery_success_reference",
        "recovery_success_own_identity",
        "recovery_success_duplicate_id",
        "recovery_success_foreign_id",
        "recovery_rate_reference",
        "recovery_rate_own_identity",
        "recovery_rate_duplicate_id",
        "recovery_rate_foreign_id",
        "post_failure_strict_accuracy",
    ]

    for row in per_object_rows:
        row.update(_compute_recovery_fields(row=row, canonical_gt_by_pred=canonical_gt_by_pred))
    per_object_fields = _append_missing_fields(per_object_fields, recovery_fields)
    _write_csv_rows(per_object_path, per_object_fields, per_object_rows)

    scene_fields, scene_rows = _read_csv_rows(scene_summary_path)
    if scene_rows:
        scene_row = dict(scene_rows[0])
        recovery_attempts_total = int(sum(int(row.get("recovery_attempts", 0) or 0) for row in per_object_rows))
        recovery_success_reference_total = int(sum(int(row.get("recovery_success_reference", 0) or 0) for row in per_object_rows))
        recovery_success_own_identity_total = int(sum(int(row.get("recovery_success_own_identity", 0) or 0) for row in per_object_rows))
        recovery_success_duplicate_id_total = int(sum(int(row.get("recovery_success_duplicate_id", 0) or 0) for row in per_object_rows))
        recovery_success_foreign_id_total = int(sum(int(row.get("recovery_success_foreign_id", 0) or 0) for row in per_object_rows))
        scene_row.update(
            {
                "recovery_attempts_total": int(recovery_attempts_total),
                "recovery_success_reference_total": int(recovery_success_reference_total),
                "recovery_success_own_identity_total": int(recovery_success_own_identity_total),
                "recovery_success_duplicate_id_total": int(recovery_success_duplicate_id_total),
                "recovery_success_foreign_id_total": int(recovery_success_foreign_id_total),
                "recovery_rate_reference": _safe_pct(recovery_success_reference_total, recovery_attempts_total),
                "recovery_rate_own_identity": _safe_pct(recovery_success_own_identity_total, recovery_attempts_total),
                "recovery_rate_duplicate_id": _safe_pct(recovery_success_duplicate_id_total, recovery_attempts_total),
                "recovery_rate_foreign_id": _safe_pct(recovery_success_foreign_id_total, recovery_attempts_total),
            }
        )
        scene_fields = _append_missing_fields(
            scene_fields,
            [
                "recovery_attempts_total",
                "recovery_success_reference_total",
                "recovery_success_own_identity_total",
                "recovery_success_duplicate_id_total",
                "recovery_success_foreign_id_total",
                "recovery_rate_reference",
                "recovery_rate_own_identity",
                "recovery_rate_duplicate_id",
                "recovery_rate_foreign_id",
            ],
        )
        _write_csv_rows(scene_summary_path, scene_fields, [scene_row])

    class_fields, class_rows = _read_csv_rows(per_class_path)
    if class_rows:
        object_rows_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_object_rows:
            object_rows_by_class[str(row.get("gt_class_name") or "unknown")].append(row)
        for row in class_rows:
            class_name = str(row.get("class_name") or "unknown")
            class_gt_rows = object_rows_by_class.get(class_name, [])
            recovery_attempts_total = int(sum(int(item.get("recovery_attempts", 0) or 0) for item in class_gt_rows))
            recovery_success_reference_total = int(sum(int(item.get("recovery_success_reference", 0) or 0) for item in class_gt_rows))
            recovery_success_own_identity_total = int(sum(int(item.get("recovery_success_own_identity", 0) or 0) for item in class_gt_rows))
            recovery_success_duplicate_id_total = int(sum(int(item.get("recovery_success_duplicate_id", 0) or 0) for item in class_gt_rows))
            recovery_success_foreign_id_total = int(sum(int(item.get("recovery_success_foreign_id", 0) or 0) for item in class_gt_rows))
            row.update(
                {
                    "recovery_attempts_total": int(recovery_attempts_total),
                    "recovery_success_reference_total": int(recovery_success_reference_total),
                    "recovery_success_own_identity_total": int(recovery_success_own_identity_total),
                    "recovery_success_duplicate_id_total": int(recovery_success_duplicate_id_total),
                    "recovery_success_foreign_id_total": int(recovery_success_foreign_id_total),
                    "recovery_rate_reference": _safe_pct(recovery_success_reference_total, recovery_attempts_total),
                    "recovery_rate_own_identity": _safe_pct(recovery_success_own_identity_total, recovery_attempts_total),
                    "recovery_rate_duplicate_id": _safe_pct(recovery_success_duplicate_id_total, recovery_attempts_total),
                    "recovery_rate_foreign_id": _safe_pct(recovery_success_foreign_id_total, recovery_attempts_total),
                }
            )
        class_fields = _append_missing_fields(
            class_fields,
            [
                "recovery_attempts_total",
                "recovery_success_reference_total",
                "recovery_success_own_identity_total",
                "recovery_success_duplicate_id_total",
                "recovery_success_foreign_id_total",
                "recovery_rate_reference",
                "recovery_rate_own_identity",
                "recovery_rate_duplicate_id",
                "recovery_rate_foreign_id",
            ],
        )
        _write_csv_rows(per_class_path, class_fields, class_rows)

    return True


def main() -> None:
    batch_dir = (PROJECT_DIR / "outputs" / "d4sm" / "testing_batch").resolve()
    scenes_root = (batch_dir / "scenes").resolve()
    if not scenes_root.is_dir():
        raise FileNotFoundError(f"No existe scenes root: {scenes_root}")

    processed = 0
    for scene_dir in sorted(child for child in scenes_root.iterdir() if child.is_dir() and not child.name.startswith(".")):
        if _backfill_scene(scene_dir):
            processed += 1

    from testing.d4sm.run_tracking_batch import (
        merge_scene_name_index,
        read_manifest_rows,
        rebuild_batch_outputs,
        unique_preserve_order,
    )

    manifest_rows = read_manifest_rows(batch_dir)
    registered_scene_ids = unique_preserve_order(
        [str(row.get("scene_id", "") or "").strip() for row in manifest_rows]
    )
    selected_scene_ids = [
        str(row.get("scene_id", "") or "").strip()
        for row in manifest_rows
        if bool(row.get("selected_in_current_run", False))
    ]
    failed_scene_errors = {
        str(row.get("scene_id", "") or "").strip(): str(row.get("error_message", "") or "")
        for row in manifest_rows
        if str(row.get("scene_id", "") or "").strip()
        and str(row.get("status", "") or "").strip() == "failed"
        and str(row.get("error_message", "") or "").strip()
    }
    scene_name_by_id = merge_scene_name_index(
        base_scene_name_by_id={scene_id: scene_id for scene_id in registered_scene_ids},
        manifest_rows=manifest_rows,
        per_scene_rows=[],
    )
    rebuild_batch_outputs(
        batch_dir=batch_dir,
        run_id="d4sm",
        selected_scene_ids=selected_scene_ids,
        registered_scene_ids=registered_scene_ids,
        scene_name_by_id=scene_name_by_id,
        failed_scene_errors=failed_scene_errors,
    )
    print(f"[D4SM][BACKFILL] Scenes processed: {processed}")
    print(f"[D4SM][BACKFILL] Batch outputs rebuilt in {batch_dir}")


if __name__ == "__main__":
    main()
