from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def fmt_pct(x) -> str:
    if x is None:
        return "-"
    return f"{100.0 * float(x):.2f}%"


def _truncate(s, n: int) -> str:
    text = "" if s is None else str(s)
    if len(text) <= n:
        return text
    return text[: max(0, n - 1)] + "…"


def render_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    headers = [str(title) for _, title in columns]
    widths = [len(h) for h in headers]

    normalized_rows = []
    for row in rows:
        vals = []
        for idx, (key, _) in enumerate(columns):
            val = row.get(key, "")
            sval = "" if val is None else str(val)
            vals.append(sval)
            widths[idx] = max(widths[idx], len(sval))
        normalized_rows.append(vals)

    sep = "-+-".join("-" * w for w in widths)
    out = [" | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)), sep]
    for vals in normalized_rows:
        out.append(" | ".join(vals[i].ljust(widths[i]) for i in range(len(widths))))
    return "\n".join(out)


def append_metric_block(lines: list[str], title: str, metrics: dict, keys: list[str]) -> None:
    lines.append(title)
    for key in keys:
        val = metrics.get(key, None)
        if (
            key.startswith("accuracy_")
            or key.endswith("_rate")
            or key.endswith("_recall")
            or key.endswith("_precision")
            or key.endswith("_coverage")
            or key in {
                "coverage_firm", "firm_accuracy", "idp", "idr", "idf1",
                "hypothesis_recall_uncertain", "hota", "deta", "assa",
                "mt", "pt", "ml",
            }
        ):
            val = fmt_pct(val)
        elif key in {
            "set_accuracy_ambiguous",
            "ambiguity_rate",
            "provisional_parent_rate",
            "provisional_new_rate",
            "uncertain_rate",
            "parent_hit_rate_provisional",
            "new_detection_accuracy_uncertain",
            "new_detection_accuracy_collapsed",
            "accuracy_existing_vs_new_collapsed",
            "accuracy_parent_collapsed",
            "accuracy_global_collapsed",
            "firm_error_rate_over_all_cases",
        }:
            val = fmt_pct(val)
        elif key in {"mean_tracking_iou", "mean_tracking_iou_object"} and val is not None:
            val = f"{float(val):.4f}"
        elif key == "pred_track_inflation_factor" and val is not None:
            val = f"{float(val):.3f}x"
        lines.append(f"  {key}: {val}")


def build_console_report(results: dict) -> str:
    collapsed_identity = results.get("collapsed_identity_metrics", {}) or {}
    collapsed = results.get("collapsed_metrics", {}) or {}
    uncertainty = results.get("uncertainty_metrics", {}) or {}
    summary = results.get("summary", {}) or {}
    per_class = results.get("per_class", []) or []
    per_object = results.get("per_object", []) or []
    events = results.get("events", {}) or {}

    lines = []
    append_metric_block(
        lines,
        "[TEST][Collapsed]",
        collapsed,
        [
            "n_cases",
            "n_existing_gt",
            "n_new_gt",
            "n_ambiguous_cases",
            "accuracy_global_collapsed",
            "accuracy_existing_vs_new_collapsed",
            "accuracy_parent_collapsed",
            "set_accuracy_ambiguous",
            "new_detection_accuracy_collapsed",
        ],
    )

    lines.append("")
    append_metric_block(
        lines,
        "[TEST][CollapsedIdentity]",
        collapsed_identity,
        [
            "n_gt_observations",
            "n_pred_observations",
            "n_unique_gt_ids",
            "n_unique_existing_pred_ids",
            "n_unique_new_pred_ids",
            "n_unique_pred_ids",
            "idtp",
            "idfp",
            "idfn",
            "idp",
            "idr",
            "idf1",
            "idsw",
            "frag",
            "tracking_recall",
            "mean_tracking_iou",
            "deta",
            "assa",
            "hota",
        ],
    )

    lines.append("")
    append_metric_block(
        lines,
        "[TEST][Uncertainty]",
        uncertainty,
        [
            "n_cases",
            "n_firm",
            "n_ambiguous",
            "n_provisional_parent",
            "n_provisional_new",
            "coverage_firm",
            "firm_accuracy",
            "firm_error_rate_over_all_cases",
            "ambiguity_rate",
            "provisional_parent_rate",
            "provisional_new_rate",
            "uncertain_rate",
            "set_accuracy_ambiguous",
            "parent_hit_rate_provisional",
            "new_detection_accuracy_uncertain",
            "avg_ambiguous_candidates",
            "max_ambiguous_candidates",
            "avg_provisional_parent_candidates",
            "max_provisional_parent_candidates",
            "hypothesis_recall_uncertain",
        ],
    )

    lines.append("")
    lines.append("[TEST][AuxiliaryTracking]")
    for key in [
        "label_summary",
        "reopen_summary",
    ]:
        val = summary.get(key, None)
        lines.append(f"  {key}: {val}")
    for key in [
        "n_frames",
        "n_objects",
        "n_assignments",
        "n_unique_real_pred_tracks",
        "pred_track_surplus_vs_gt",
        "pred_track_inflation_factor",
        "n_mt_objects",
        "n_pt_objects",
        "n_ml_objects",
        "mt",
        "pt",
        "ml",
        "n_existing_gt_reopened_as_new_rows",
        "n_existing_gt_reopened_as_new_ids",
        "reopen_rate_existing",
        "gt_with_reopen_rate",
        "global_frame_accuracy_strict",
        "global_frame_accuracy_permissive",
        "global_object_accuracy_strict",
        "global_object_accuracy_permissive",
        "stable_min_frames",
        "objects_fragmented",
        "objects_with_foreign_id_use",
        "id_changes_total",
        "objects_recovered_reference",
        "objects_recovered_own_identity",
        "stable_foreign_segments_total",
        "stable_own_new_segments_total",
        "swap_events_total",
        "theft_with_new_id_total",
        "theft_with_displacement_total",
        "total_runtime_seconds",
        "avg_runtime_seconds",
        "total_loop_ms",
        "avg_loop_ms",
    ]:
        val = summary.get(key, None)
        if key.endswith("accuracy_strict") or key.endswith("accuracy_permissive"):
            val = fmt_pct(val)
        elif key in {"reopen_rate_existing", "gt_with_reopen_rate", "mt", "pt", "ml"}:
            val = fmt_pct(val)
        elif key == "pred_track_inflation_factor" and val is not None:
            val = f"{float(val):.3f}x"
        lines.append(f"  {key}: {val}")

    lines.append("")
    lines.append("[TEST][PerClass]")
    class_rows = []
    sorted_classes = sorted(
        per_class,
        key=lambda row: (
            -(float(row.get("pred_track_inflation_factor", 0.0) or 0.0)),
            float(row.get("weighted_strict_accuracy", 1.0) or 1.0),
            str(row.get("class_name") or ""),
        ),
    )
    for row in sorted_classes:
        infl = row.get("pred_track_inflation_factor")
        class_rows.append(
            {
                "cls": row.get("class_name"),
                "gts": row.get("n_gt_objects"),
                "preds": row.get("n_real_pred_tracks"),
                "surplus": row.get("pred_track_surplus_vs_gt"),
                "infl": "-" if infl is None else f"{float(infl):.2f}x",
                "strict": fmt_pct(row.get("weighted_strict_accuracy")),
                "perm": fmt_pct(row.get("weighted_permissive_accuracy")),
                "avg_preds": "-" if row.get("mean_pred_ids_per_gt") is None else f"{float(row.get('mean_pred_ids_per_gt')):.2f}",
                "avg_chg": "-" if row.get("mean_id_changes_per_gt") is None else f"{float(row.get('mean_id_changes_per_gt')):.2f}",
                "foreign": row.get("gt_objects_with_foreign_id_use"),
                "reopen_gt": row.get("existing_gt_reopened_as_new_ids"),
                "reopen_rows": row.get("existing_gt_reopened_as_new_rows"),
            }
        )
    lines.append(
        render_table(
            class_rows,
            [
                ("cls", "Class"),
                ("gts", "GTs"),
                ("preds", "RealPreds"),
                ("surplus", "Surplus"),
                ("infl", "Infl"),
                ("strict", "StrictW"),
                ("perm", "PermW"),
                ("avg_preds", "AvgPreds"),
                ("avg_chg", "AvgChg"),
                ("foreign", "ForeignGT"),
                ("reopen_gt", "ReopenGT"),
                ("reopen_rows", "ReopenRows"),
            ],
        )
    )

    lines.append("")
    lines.append("[TEST][PerObject]")
    object_rows = []
    for row in per_object:
        object_rows.append(
            {
                "gt": row.get("gt_label"),
                "frames": row.get("n_frames"),
                "ref": row.get("reference_pred_label") or row.get("reference_pred_id"),
                "strict": fmt_pct(row.get("strict_accuracy")),
                "perm": fmt_pct(row.get("permissive_accuracy")),
                "preds": row.get("n_unique_pred_ids"),
                "own_extra": max(0, int(row.get("n_own_pred_ids", 0)) - 1),
                "foreign": row.get("n_foreign_pred_ids"),
                "changes": row.get("id_changes"),
                "fail_frame": row.get("first_failure_frame"),
                "rec_ref": row.get("recovered_reference"),
                "rec_own": row.get("recovered_own_identity"),
            }
        )
    lines.append(
        render_table(
            object_rows,
            [
                ("gt", "GT"),
                ("frames", "Frames"),
                ("ref", "RefPred"),
                ("strict", "Strict"),
                ("perm", "Perm"),
                ("preds", "PredIDs"),
                ("own_extra", "Own+"),
                ("foreign", "Foreign"),
                ("changes", "Changes"),
                ("fail_frame", "FailAt"),
                ("rec_ref", "RecRef"),
                ("rec_own", "RecOwn"),
            ],
        )
    )

    lines.append("")
    lines.append("[TEST][Events]")
    event_rows = []
    for ev in (events.get("swap", []) or [])[:20]:
        event_rows.append(
            {
                "frame": ev.get("frame_id"),
                "type": "swap",
                "a": ev.get("gt_a"),
                "b": ev.get("gt_b"),
                "detail": f"{ev.get('pred_a_prev')}<->{ev.get('pred_b_prev')}",
            }
        )
    for ev in (events.get("theft_with_new_id", []) or [])[:20]:
        event_rows.append(
            {
                "frame": ev.get("frame_id"),
                "type": "theft_new",
                "a": ev.get("thief_gt"),
                "b": ev.get("victim_gt"),
                "detail": f"stolen={ev.get('stolen_pred_id')} victim_new={ev.get('victim_new_pred_id')}",
            }
        )
    for ev in (events.get("theft_with_displacement", []) or [])[:20]:
        event_rows.append(
            {
                "frame": ev.get("frame_id"),
                "type": "theft_disp",
                "a": ev.get("thief_gt"),
                "b": ev.get("victim_gt"),
                "detail": f"stolen={ev.get('stolen_pred_id')} victim_after={ev.get('victim_pred_id_after')}",
            }
        )

    if event_rows:
        for row in event_rows:
            row["detail"] = _truncate(row["detail"], 28)
        lines.append(
            render_table(
                event_rows,
                [
                    ("frame", "Frame"),
                    ("type", "Type"),
                    ("a", "GT_A"),
                    ("b", "GT_B"),
                    ("detail", "Detail"),
                ],
            )
        )
    else:
        lines.append("No events detected.")

    return "\n".join(lines)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(temp_path, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    if not rows:
        with temp_path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        os.replace(temp_path, path)
        return

    keys = list(rows[0].keys())
    with temp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        f.write(str(text))
    os.replace(temp_path, path)
