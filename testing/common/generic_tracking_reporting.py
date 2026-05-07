from __future__ import annotations

from testing.reporting import fmt_pct, render_table, write_csv, write_json, write_text


def _truncate(text: str | None, n: int) -> str:
    raw = "" if text is None else str(text)
    if len(raw) <= n:
        return raw
    return raw[: max(0, n - 1)] + "…"


def build_generic_console_report(results: dict) -> str:
    identity = results.get("tracking_identity_metrics", {}) or {}
    identity_variants = results.get("tracking_identity_metric_variants", {}) or {}
    summary = results.get("summary", {}) or {}
    summary_variants = results.get("summary_variants", {}) or {}
    per_class = results.get("per_class", []) or []
    per_object = results.get("per_object", []) or []
    events = results.get("events", {}) or {}

    lines: list[str] = []
    lines.append("[TEST][Identity]")
    for key in [
        "n_gt_observations",
        "n_pred_observations",
        "n_pred_observations_total",
        "n_orphan_pred_observations",
        "n_unique_gt_ids",
        "n_unique_pred_ids",
        "n_unique_pred_ids_total",
        "n_unique_orphan_pred_ids",
        "n_orphan_only_pred_ids",
        "idtp",
        "idfp",
        "idfn",
        "idp",
        "idr",
        "idf1",
        "idsw",
        "frag",
        "tracking_recall",
        "orphan_pred_rate",
        "mean_orphan_pred_area_frac",
        "mean_tracking_iou",
        "deta",
        "assa",
        "hota",
    ]:
        value = identity.get(key, None)
        if key in {"idp", "idr", "idf1", "tracking_recall", "orphan_pred_rate", "deta", "assa", "hota"}:
            value = fmt_pct(value)
        lines.append(f"  {key}: {value}")

    for suffix, variant_metrics in sorted(identity_variants.items(), key=lambda item: float(item[1].get("iou_threshold", 0.0) or 0.0)):
        lines.append("")
        lines.append(f"[TEST][Identity][{variant_metrics.get('label', suffix)}]")
        for key in [
            "n_gt_observations",
            "n_pred_observations",
            "n_unique_gt_ids",
            "n_unique_pred_ids",
            "n_matched_gt_observations",
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
        ]:
            value = variant_metrics.get(key, None)
            if key in {"idp", "idr", "idf1", "tracking_recall", "deta", "assa", "hota"}:
                value = fmt_pct(value)
            lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("[TEST][Tracking]")
    for key in [
        "label_summary",
        "n_frames",
        "n_objects",
        "n_visible_gt_observations",
        "n_matched_gt_observations",
        "n_pred_observations_total",
        "n_orphan_pred_observations",
        "n_unique_orphan_pred_ids",
        "n_orphan_only_pred_ids",
        "n_unique_real_pred_tracks",
        "pred_track_inflation_factor",
        "n_pred_tracks_with_orphan_observations",
        "n_orphan_only_pred_tracks",
        "orphan_pred_rate",
        "mean_orphan_pred_area_frac",
        "global_frame_accuracy_strict",
        "global_frame_accuracy_permissive",
        "global_object_accuracy_strict",
        "global_object_accuracy_permissive",
        "mt",
        "pt",
        "ml",
        "objects_fragmented",
        "objects_with_foreign_id_use",
        "id_changes_total",
        "swap_events_total",
        "theft_with_new_id_total",
        "theft_with_displacement_total",
        "total_runtime_seconds",
        "avg_runtime_seconds",
        "total_loop_ms",
        "avg_loop_ms",
    ]:
        value = summary.get(key, None)
        if key in {
            "global_frame_accuracy_strict",
            "global_frame_accuracy_permissive",
            "global_object_accuracy_strict",
            "global_object_accuracy_permissive",
            "orphan_pred_rate",
            "mt",
            "pt",
            "ml",
        }:
            value = fmt_pct(value)
        elif key == "pred_track_inflation_factor" and value is not None:
            value = f"{float(value):.3f}x"
        elif key == "mean_orphan_pred_area_frac" and value is not None:
            value = f"{float(value):.4f}"
        lines.append(f"  {key}: {value}")

    for suffix, variant_summary in sorted(summary_variants.items(), key=lambda item: float(item[1].get("iou_threshold", 0.0) or 0.0)):
        lines.append("")
        lines.append(f"[TEST][Tracking][{variant_summary.get('label', suffix)}]")
        for key in [
            "n_matched_gt_observations",
            "global_frame_accuracy_strict",
            "global_frame_accuracy_permissive",
            "global_object_accuracy_strict",
            "global_object_accuracy_permissive",
            "tracking_recall",
            "mean_tracking_iou",
            "deta",
            "assa",
            "hota",
        ]:
            value = variant_summary.get(key, None)
            if key in {
                "global_frame_accuracy_strict",
                "global_frame_accuracy_permissive",
                "global_object_accuracy_strict",
                "global_object_accuracy_permissive",
                "tracking_recall",
                "deta",
                "assa",
                "hota",
            }:
                value = fmt_pct(value)
            lines.append(f"  {key}: {value}")

    lines.append("")
    lines.append("[TEST][PerClass]")
    class_rows = []
    for row in sorted(
        per_class,
        key=lambda item: (
            -(float(item.get("pred_track_inflation_factor", 0.0) or 0.0)),
            float(item.get("weighted_strict_accuracy", 1.0) or 1.0),
            str(item.get("class_name") or ""),
        ),
    ):
        infl = row.get("pred_track_inflation_factor", None)
        class_rows.append(
            {
                "cls": row.get("class_name", None),
                "gts": row.get("n_gt_objects", None),
                "preds": row.get("n_real_pred_tracks", None),
                "infl": "-" if infl is None else f"{float(infl):.2f}x",
                "strict": fmt_pct(row.get("weighted_strict_accuracy", None)),
                "strict40": fmt_pct(row.get("weighted_strict_accuracy_iou40", None)),
                "perm": fmt_pct(row.get("weighted_permissive_accuracy", None)),
                "perm40": fmt_pct(row.get("weighted_permissive_accuracy_iou40", None)),
                "idf1": fmt_pct(row.get("hota", None)),
                "idf140": fmt_pct(row.get("hota_iou40", None)),
                "avg_preds": (
                    "-"
                    if row.get("mean_pred_ids_per_gt", None) is None
                    else f"{float(row.get('mean_pred_ids_per_gt')):.2f}"
                ),
                "avg_chg": (
                    "-"
                    if row.get("mean_id_changes_per_gt", None) is None
                    else f"{float(row.get('mean_id_changes_per_gt')):.2f}"
                ),
                "foreign": row.get("gt_objects_with_foreign_id_use", None),
            }
        )
    lines.append(
        render_table(
            class_rows,
            [
                ("cls", "Class"),
                ("gts", "GTs"),
                ("preds", "RealPreds"),
                ("infl", "Infl"),
                ("strict", "StrictW"),
                ("strict40", "StrictW@0.4"),
                ("perm", "PermW"),
                ("perm40", "PermW@0.4"),
                ("idf1", "HOTA"),
                ("idf140", "HOTA@0.4"),
                ("avg_preds", "AvgPreds"),
                ("avg_chg", "AvgChg"),
                ("foreign", "ForeignGT"),
            ],
        )
    )

    lines.append("")
    lines.append("[TEST][PerObject]")
    object_rows = []
    for row in per_object:
        object_rows.append(
            {
                "gt": row.get("gt_label", None),
                "frames": row.get("n_frames", None),
                "ref": row.get("reference_pred_label") or row.get("reference_pred_id"),
                "strict": fmt_pct(row.get("strict_accuracy", None)),
                "strict40": fmt_pct(row.get("strict_accuracy_iou40", None)),
                "perm": fmt_pct(row.get("permissive_accuracy", None)),
                "perm40": fmt_pct(row.get("permissive_accuracy_iou40", None)),
                "preds": row.get("n_unique_pred_ids", None),
                "own_extra": max(0, int(row.get("n_own_pred_ids", 0) or 0) - 1),
                "foreign": row.get("n_foreign_pred_ids", None),
                "changes": row.get("id_changes", None),
                "fail_frame": row.get("first_failure_frame", None),
                "rec_ref": row.get("recovered_reference", None),
                "rec_own": row.get("recovered_own_identity", None),
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
                ("strict40", "Strict@0.4"),
                ("perm", "Perm"),
                ("perm40", "Perm@0.4"),
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
                "frame": ev.get("frame_id", None),
                "type": "swap",
                "a": ev.get("gt_a", None),
                "b": ev.get("gt_b", None),
                "detail": f"{ev.get('pred_a_prev')}<->{ev.get('pred_b_prev')}",
            }
        )
    for ev in (events.get("theft_with_new_id", []) or [])[:20]:
        event_rows.append(
            {
                "frame": ev.get("frame_id", None),
                "type": "theft_new",
                "a": ev.get("thief_gt", None),
                "b": ev.get("victim_gt", None),
                "detail": f"stolen={ev.get('stolen_pred_id')} victim_new={ev.get('victim_new_pred_id')}",
            }
        )
    for ev in (events.get("theft_with_displacement", []) or [])[:20]:
        event_rows.append(
            {
                "frame": ev.get("frame_id", None),
                "type": "theft_disp",
                "a": ev.get("thief_gt", None),
                "b": ev.get("victim_gt", None),
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


__all__ = [
    "build_generic_console_report",
    "fmt_pct",
    "render_table",
    "write_csv",
    "write_json",
    "write_text",
]
