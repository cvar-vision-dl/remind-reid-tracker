from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

import numpy as np

from testing.tracking_metrics import (
    SegmentRecord,
    build_memory_summary,
    infer_frame_area_px,
    match_detections_to_gt,
    safe_pct,
)


def _iou_threshold_suffix(threshold: float) -> str:
    return f"iou{int(round(float(threshold) * 100.0)):02d}"


def _mt_pt_ml_label_from_recall(tracking_recall: float | None) -> str | None:
    if tracking_recall is None:
        return None
    if float(tracking_recall) >= 0.80:
        return "MT"
    if float(tracking_recall) <= 0.20:
        return "ML"
    return "PT"


class TrackingOnlyEvaluator:
    """
    Generic evaluator for trackers that only expose trajectories/masks.

    Keeps the same global evaluation philosophy as the internal pipeline:
    - reference GT<->pred mapping by class with Hungarian;
    - global identity metrics;
    - analysis by object, frame, class, and predicted track;
    - swap/theft-like events;
    - without internal pipeline decision, uncertainty, or memory metrics.
    """

    def __init__(
        self,
        stable_min_frames: int = 3,
        additional_iou_thresholds: tuple[float, ...] = (0.4,),
    ):
        self.stable_min_frames = max(1, int(stable_min_frames))
        self.additional_iou_thresholds = tuple(
            sorted(
                {
                    float(threshold)
                    for threshold in tuple(additional_iou_thresholds or ())
                    if float(threshold) > 0.0
                }
            )
        )
        self.frames_seen: list[int] = []
        self.case_records: list[dict[str, Any]] = []
        self.orphan_pred_records: list[dict[str, Any]] = []
        self.frame_telemetry_by_id: dict[int, dict[str, Any]] = {}
        self.gt_meta: dict[int, dict[str, Any]] = {}
        self.pred_meta: dict[int, dict[str, Any]] = {}

    def _iter_iou_variants(self) -> list[dict[str, Any]]:
        return [
            {
                "threshold": float(threshold),
                "suffix": _iou_threshold_suffix(float(threshold)),
                "label": f"IoU>={float(threshold):.2f}",
            }
            for threshold in self.additional_iou_thresholds
        ]

    def ingest_frame(
        self,
        *,
        frame_id: int,
        detections: list[Any],
        gt_objects: dict[int, Any],
        det_to_pred_id: dict[int, int],
        pred_info_by_id: dict[int, dict[str, Any]] | None = None,
        frame_shape: tuple[int, int] | None = None,
        frame_telemetry: dict[str, Any] | None = None,
    ) -> None:
        frame_id = int(frame_id)
        self.frames_seen.append(frame_id)
        self.frame_telemetry_by_id[frame_id] = dict(frame_telemetry or {})

        det_to_gt = match_detections_to_gt(detections=detections, gt_objects=gt_objects)
        gt_to_det = {
            int(gt_id): int(det_id)
            for det_id, (gt_id, _iou) in (det_to_gt or {}).items()
        }
        frame_area_px = int(infer_frame_area_px(frame_shape=frame_shape, detections=detections))
        gt_class_counts = Counter(
            str(getattr(gt_obj, "class_name", None) or "unknown")
            for gt_obj in (gt_objects or {}).values()
        )
        pred_info_by_id = dict(pred_info_by_id or {})

        for gt_id, gt_obj in (gt_objects or {}).items():
            meta = self.gt_meta.setdefault(
                int(gt_id),
                {
                    "label": str(getattr(gt_obj, "label", f"instance_{gt_id}")),
                    "class_name": getattr(gt_obj, "class_name", None),
                    "first_frame": frame_id,
                },
            )
            meta["first_frame"] = min(int(meta.get("first_frame", frame_id)), frame_id)

        for raw_pred_id, pred_info in pred_info_by_id.items():
            pred_id = int(raw_pred_id)
            self.pred_meta.setdefault(
                pred_id,
                {
                    "instance_label": pred_info.get("instance_label", f"track_{pred_id}"),
                    "class_name": pred_info.get("class_name", None),
                    "first_frame": frame_id,
                },
            )

        matched_det_ids = {int(det_id) for det_id in (det_to_gt or {}).keys()}
        for det in detections or []:
            det_id = getattr(det, "detection_id", None)
            if det_id is None:
                continue
            det_id = int(det_id)
            if det_id in matched_det_ids:
                continue

            pred_id = det_to_pred_id.get(int(det_id), None)
            pred_info = {}
            if pred_id is not None:
                pred_id = int(pred_id)
                pred_info = pred_info_by_id.get(int(pred_id), {})
                self.pred_meta.setdefault(
                    int(pred_id),
                    {
                        "instance_label": pred_info.get("instance_label", f"track_{pred_id}"),
                        "class_name": pred_info.get("class_name", getattr(det, "class_name", None)),
                        "first_frame": frame_id,
                    },
                )

            det_area_px = 0
            geom = getattr(det, "geom", None)
            if isinstance(geom, dict):
                det_area_px = int(geom.get("area", 0.0) or 0.0)
            if det_area_px <= 0:
                det_mask = getattr(det, "mask", None)
                if det_mask is not None:
                    det_area_px = int(np.asarray(det_mask).astype(bool, copy=False).sum())

            pred_class_name = getattr(det, "class_name", None)
            pred_instance_label = None
            if pred_id is not None:
                pred_instance_label = self.pred_meta.get(int(pred_id), {}).get("instance_label", None)
                pred_class_name = self.pred_meta.get(int(pred_id), {}).get("class_name", pred_class_name)

            self.orphan_pred_records.append(
                {
                    "frame_id": int(frame_id),
                    "det_id": int(det_id),
                    "pred_object_id": None if pred_id is None else int(pred_id),
                    "pred_instance_label": pred_instance_label,
                    "pred_class_name": pred_class_name,
                    "pred_area_px": int(det_area_px),
                    "pred_area_frac": (
                        float(det_area_px) / float(frame_area_px)
                        if frame_area_px > 0 and det_area_px > 0
                        else None
                    ),
                    "frame_gt_count": int(len(gt_objects or {})),
                    "frame_area_px": int(frame_area_px),
                    "is_orphan_pred_observation": True,
                }
            )

        for gt_id in sorted(int(x) for x in (gt_objects or {}).keys()):
            gt_obj = gt_objects[int(gt_id)]
            gt_meta = self.gt_meta[int(gt_id)]
            det_id = gt_to_det.get(int(gt_id), None)
            pred_id = None if det_id is None else det_to_pred_id.get(int(det_id), None)
            iou = 0.0
            if det_id is not None and int(det_id) in det_to_gt:
                iou = float(det_to_gt[int(det_id)][1])

            if pred_id is not None:
                pred_id = int(pred_id)
                pred_info = pred_info_by_id.get(int(pred_id), {})
                self.pred_meta.setdefault(
                    int(pred_id),
                    {
                        "instance_label": pred_info.get("instance_label", f"track_{pred_id}"),
                        "class_name": pred_info.get("class_name", None),
                        "first_frame": frame_id,
                    },
                )

            self.case_records.append(
                {
                    "frame_id": frame_id,
                    "gt_instance_id": int(gt_id),
                    "gt_label": str(gt_meta["label"]),
                    "gt_class_name": gt_meta.get("class_name", None),
                    "det_id": -1 if det_id is None else int(det_id),
                    "iou": float(iou),
                    "pred_object_id": None if pred_id is None else int(pred_id),
                    "pred_instance_label": (
                        None
                        if pred_id is None
                        else self.pred_meta.get(int(pred_id), {}).get("instance_label", None)
                    ),
                    "pred_class_name": (
                        None
                        if pred_id is None
                        else self.pred_meta.get(int(pred_id), {}).get("class_name", None)
                    ),
                    "gt_area_px": int(getattr(gt_obj, "area", 0) or 0),
                    "frame_gt_count": int(len(gt_objects or {})),
                    "frame_same_class_count": int(gt_class_counts.get(str(gt_meta.get("class_name") or "unknown"), 0)),
                    "frame_area_px": int(frame_area_px),
                    "real_state": (
                        "new"
                        if frame_id == int(self.gt_meta.get(int(gt_id), {}).get("first_frame", frame_id))
                        else "existing"
                    ),
                }
            )

    def finalize(self) -> dict[str, Any]:
        case_rows = sorted(
            [dict(row) for row in self.case_records],
            key=lambda row: (int(row["frame_id"]), int(row["gt_instance_id"])),
        )
        orphan_pred_rows = sorted(
            [dict(row) for row in self.orphan_pred_records],
            key=lambda row: (
                int(row.get("frame_id", -1)),
                int(row.get("pred_object_id", -1) if row.get("pred_object_id", None) is not None else -1),
                int(row.get("det_id", -1)),
            ),
        )
        reference_pred_by_gt = self.build_global_reference_mapping(case_rows=case_rows)
        canonical_gt_by_pred = self.build_canonical_gt_by_pred(case_rows=case_rows)

        for row in case_rows:
            gt_id = int(row["gt_instance_id"])
            pred_id = row.get("pred_object_id", None)
            if pred_id is not None:
                pred_id = int(pred_id)
            ref_pred = reference_pred_by_gt.get(int(gt_id), None)
            strict_global_correct = bool(
                pred_id is not None and ref_pred is not None and int(pred_id) == int(ref_pred)
            )
            permissive_global_correct = bool(
                pred_id is not None
                and (
                    strict_global_correct
                    or canonical_gt_by_pred.get(int(pred_id), None) == int(gt_id)
                )
            )
            row["reference_pred_id"] = None if ref_pred is None else int(ref_pred)
            row["strict_global_correct"] = bool(strict_global_correct)
            row["permissive_global_correct"] = bool(permissive_global_correct)
            row["gt_is_new_this_frame"] = bool(str(row.get("real_state")) == "new")
            row["gt_age_frames"] = int(
                max(
                    0,
                    int(row["frame_id"])
                    - int(self.gt_meta.get(int(gt_id), {}).get("first_frame", row["frame_id"])),
                )
            )
            row["n_gt_visible_in_frame"] = int(row.get("frame_gt_count", 0) or 0)
            row["n_gt_same_class_in_frame"] = int(row.get("frame_same_class_count", 0) or 0)
            row["n_total_distractors"] = int(max(0, int(row.get("frame_gt_count", 0) or 0) - 1))
            row["n_same_class_distractors"] = int(max(0, int(row.get("frame_same_class_count", 0) or 0) - 1))
            frame_area_px = int(row.get("frame_area_px", 0) or 0)
            gt_area_px = int(row.get("gt_area_px", 0) or 0)
            row["gt_area_frac"] = (float(gt_area_px) / float(frame_area_px)) if frame_area_px > 0 else None
            row["is_tracked_visible_case"] = bool(pred_id is not None)
            row["tracking_iou"] = float(row.get("iou", 0.0) or 0.0) if strict_global_correct else 0.0
            for variant in self._iter_iou_variants():
                suffix = str(variant["suffix"])
                threshold = float(variant["threshold"])
                iou_value = float(row.get("iou", 0.0) or 0.0)
                matched_key = f"matched_{suffix}"
                strict_key = f"strict_global_correct_{suffix}"
                permissive_key = f"permissive_global_correct_{suffix}"
                tracking_iou_key = f"tracking_iou_{suffix}"
                det_id_value = row.get("det_id", -1)
                row[matched_key] = bool(
                    int(-1 if det_id_value is None else det_id_value) >= 0 and iou_value >= threshold
                )
                row[strict_key] = bool(strict_global_correct and iou_value >= threshold)
                row[permissive_key] = bool(permissive_global_correct and iou_value >= threshold)
                row[tracking_iou_key] = float(iou_value) if bool(row[strict_key]) else 0.0

        per_object_rows = self.build_per_object_rows(
            case_rows=case_rows,
            reference_pred_by_gt=reference_pred_by_gt,
            canonical_gt_by_pred=canonical_gt_by_pred,
        )
        per_frame_rows = self.build_per_frame_rows(
            case_rows=case_rows,
            orphan_pred_rows=orphan_pred_rows,
            reference_pred_by_gt=reference_pred_by_gt,
            canonical_gt_by_pred=canonical_gt_by_pred,
        )
        pred_rows = self.build_pred_rows(
            case_rows=case_rows,
            orphan_pred_rows=orphan_pred_rows,
            reference_pred_by_gt=reference_pred_by_gt,
            canonical_gt_by_pred=canonical_gt_by_pred,
        )
        events = self.build_events(case_rows=case_rows)
        tracking_identity_metrics = self.compute_tracking_identity_metrics(
            case_rows=case_rows,
            orphan_pred_rows=orphan_pred_rows,
            per_object_rows=per_object_rows,
            reference_pred_by_gt=reference_pred_by_gt,
        )
        summary = self.build_summary(
            case_rows=case_rows,
            orphan_pred_rows=orphan_pred_rows,
            per_object_rows=per_object_rows,
            per_frame_rows=per_frame_rows,
            pred_rows=pred_rows,
            events=events,
            tracking_identity_metrics=tracking_identity_metrics,
        )
        per_class_rows = self.build_per_class_rows(
            case_rows=case_rows,
            per_object_rows=per_object_rows,
            pred_rows=pred_rows,
            tracking_identity_metrics=tracking_identity_metrics,
        )

        per_frame_by_id = {
            int(row["frame_id"]): row
            for row in per_frame_rows
            if row.get("frame_id", None) is not None
        }
        for frame_id in sorted(set(int(x) for x in self.frames_seen) | set(per_frame_by_id.keys())):
            row = per_frame_by_id.get(int(frame_id), None)
            if row is None:
                row = {
                    "frame_id": int(frame_id),
                    "n_objects": 0,
                    "strict_accuracy": 0.0,
                    "permissive_accuracy": 0.0,
                    "strict_correct": 0,
                    "permissive_correct": 0,
                }
                per_frame_rows.append(row)
                per_frame_by_id[int(frame_id)] = row
            row.update(self.frame_telemetry_by_id.get(int(frame_id), {}))

        summary.update(build_memory_summary(per_frame_rows))

        tracking_identity_metric_variants = {
            str(variant["suffix"]): {
                "iou_threshold": float(variant["threshold"]),
                "label": str(variant["label"]),
                "n_gt_observations": tracking_identity_metrics.get("n_gt_observations", None),
                "n_pred_observations": tracking_identity_metrics.get("n_pred_observations", None),
                "n_unique_gt_ids": tracking_identity_metrics.get("n_unique_gt_ids", None),
                "n_unique_pred_ids": tracking_identity_metrics.get("n_unique_pred_ids", None),
                "n_matched_gt_observations": tracking_identity_metrics.get(
                    f"n_matched_gt_observations_{variant['suffix']}",
                    None,
                ),
                "idtp": tracking_identity_metrics.get(f"idtp_{variant['suffix']}", None),
                "idfp": tracking_identity_metrics.get(f"idfp_{variant['suffix']}", None),
                "idfn": tracking_identity_metrics.get(f"idfn_{variant['suffix']}", None),
                "idp": tracking_identity_metrics.get(f"idp_{variant['suffix']}", None),
                "idr": tracking_identity_metrics.get(f"idr_{variant['suffix']}", None),
                "idf1": tracking_identity_metrics.get(f"idf1_{variant['suffix']}", None),
                "idsw": tracking_identity_metrics.get("idsw", None),
                "frag": tracking_identity_metrics.get("frag", None),
                "tracking_recall": tracking_identity_metrics.get(
                    f"tracking_recall_{variant['suffix']}",
                    None,
                ),
                "mean_tracking_iou": tracking_identity_metrics.get(
                    f"mean_tracking_iou_{variant['suffix']}",
                    None,
                ),
                "deta": tracking_identity_metrics.get(f"deta_{variant['suffix']}", None),
                "assa": tracking_identity_metrics.get(f"assa_{variant['suffix']}", None),
                "hota": tracking_identity_metrics.get(f"hota_{variant['suffix']}", None),
            }
            for variant in self._iter_iou_variants()
        }
        summary_variants = {
            str(variant["suffix"]): {
                "iou_threshold": float(variant["threshold"]),
                "label": str(variant["label"]),
                "n_matched_gt_observations": summary.get(
                    f"n_matched_gt_observations_{variant['suffix']}",
                    None,
                ),
                "global_frame_accuracy_strict": summary.get(
                    f"global_frame_accuracy_strict_{variant['suffix']}",
                    None,
                ),
                "global_frame_accuracy_permissive": summary.get(
                    f"global_frame_accuracy_permissive_{variant['suffix']}",
                    None,
                ),
                "global_object_accuracy_strict": summary.get(
                    f"global_object_accuracy_strict_{variant['suffix']}",
                    None,
                ),
                "global_object_accuracy_permissive": summary.get(
                    f"global_object_accuracy_permissive_{variant['suffix']}",
                    None,
                ),
                "tracking_recall": summary.get(f"tracking_recall_{variant['suffix']}", None),
                "mean_tracking_iou": summary.get(f"mean_tracking_iou_{variant['suffix']}", None),
                "deta": summary.get(f"deta_{variant['suffix']}", None),
                "assa": summary.get(f"assa_{variant['suffix']}", None),
                "hota": summary.get(f"hota_{variant['suffix']}", None),
            }
            for variant in self._iter_iou_variants()
        }

        return {
            "tracking_identity_metrics": tracking_identity_metrics,
            "tracking_identity_metric_variants": tracking_identity_metric_variants,
            "summary": summary,
            "summary_variants": summary_variants,
            "per_class": per_class_rows,
            "per_case": case_rows,
            "per_orphan_pred": orphan_pred_rows,
            "per_frame": sorted(per_frame_rows, key=lambda row: int(row["frame_id"])),
            "per_object": sorted(per_object_rows, key=lambda row: int(row["gt_instance_id"])),
            "per_pred_track": sorted(pred_rows, key=lambda row: int(row["pred_object_id"])),
            "events": events,
            "canonical_gt_by_pred": {int(k): int(v) for k, v in canonical_gt_by_pred.items()},
            "reference_pred_by_gt": {int(k): int(v) for k, v in reference_pred_by_gt.items()},
        }

    @staticmethod
    def build_canonical_gt_by_pred(*, case_rows: list[dict[str, Any]]) -> dict[int, int]:
        owner_votes_by_pred: dict[int, Counter] = defaultdict(Counter)
        for row in case_rows:
            pred_id = row.get("pred_object_id", None)
            if pred_id is None:
                continue
            owner_votes_by_pred[int(pred_id)][int(row["gt_instance_id"])] += 1
        out: dict[int, int] = {}
        for pred_id, counter in owner_votes_by_pred.items():
            best_gt, _ = sorted(counter.items(), key=lambda item: (-int(item[1]), int(item[0])))[0]
            out[int(pred_id)] = int(best_gt)
        return out

    def build_per_object_rows(
        self,
        *,
        case_rows: list[dict[str, Any]],
        reference_pred_by_gt: dict[int, int],
        canonical_gt_by_pred: dict[int, int],
    ) -> list[dict[str, Any]]:
        by_gt: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in case_rows:
            by_gt[int(row["gt_instance_id"])].append(row)

        per_object_rows: list[dict[str, Any]] = []
        for gt_id in sorted(by_gt):
            gt_cases = sorted(by_gt[int(gt_id)], key=lambda row: int(row["frame_id"]))
            visible_n_frames = int(len(gt_cases))
            pred_timeline = [
                None if item.get("pred_object_id", None) is None else int(item["pred_object_id"])
                for item in gt_cases
            ]
            assigned_n_frames = int(sum(1 for pred_id in pred_timeline if pred_id is not None))
            assigned_ref_pred = reference_pred_by_gt.get(int(gt_id), None)

            strict_flags = [
                bool(assigned_ref_pred is not None and pred_id is not None and int(pred_id) == int(assigned_ref_pred))
                for pred_id in pred_timeline
            ]
            permissive_flags = [
                bool(
                    pred_id is not None
                    and (
                        strict_flags[idx]
                        or canonical_gt_by_pred.get(int(pred_id), None) == int(gt_id)
                    )
                )
                for idx, pred_id in enumerate(pred_timeline)
            ]
            strict_correct = int(sum(1 for flag in strict_flags if flag))
            permissive_correct = int(sum(1 for flag in permissive_flags if flag))

            unique_pred_ids: list[int] = []
            for pred_id in [int(pid) for pid in pred_timeline if pid is not None]:
                if pred_id not in unique_pred_ids:
                    unique_pred_ids.append(int(pred_id))
            own_pred_ids = [
                int(pred_id)
                for pred_id in unique_pred_ids
                if canonical_gt_by_pred.get(int(pred_id), None) == int(gt_id)
            ]
            foreign_pred_ids = [
                int(pred_id)
                for pred_id in unique_pred_ids
                if canonical_gt_by_pred.get(int(pred_id), None) != int(gt_id)
            ]

            segments: list[SegmentRecord] = []
            seg_start_idx = None
            for idx, pred_id in enumerate(list(pred_timeline) + [None]):
                if seg_start_idx is None:
                    if pred_id is not None:
                        seg_start_idx = int(idx)
                    continue
                prev_pred_id = pred_timeline[int(idx) - 1] if idx > 0 else None
                if idx < len(pred_timeline) and pred_id == prev_pred_id:
                    continue
                if prev_pred_id is not None:
                    owner = int(canonical_gt_by_pred.get(int(prev_pred_id), -1))
                    kind = "reference"
                    if assigned_ref_pred is None or int(prev_pred_id) != int(assigned_ref_pred):
                        kind = "own_new_id" if owner == int(gt_id) else "foreign_id"
                    segments.append(
                        SegmentRecord(
                            pred_object_id=int(prev_pred_id),
                            start_frame=int(gt_cases[int(seg_start_idx)]["frame_id"]),
                            end_frame=int(gt_cases[int(idx) - 1]["frame_id"]),
                            length=int(idx - int(seg_start_idx)),
                            canonical_owner_gt=owner,
                            kind=kind,
                        )
                    )
                seg_start_idx = None
                if idx < len(pred_timeline) and pred_id is not None:
                    seg_start_idx = int(idx)

            idsw_object = 0
            prev_pred_non_null = None
            for pred_id in pred_timeline:
                if pred_id is None:
                    continue
                if prev_pred_non_null is not None and int(pred_id) != int(prev_pred_non_null):
                    idsw_object += 1
                prev_pred_non_null = int(pred_id)

            tracked_runs = 0
            in_tracked_run = False
            for pred_id in pred_timeline:
                cur_tracked = pred_id is not None
                if cur_tracked and not in_tracked_run:
                    tracked_runs += 1
                    in_tracked_run = True
                elif not cur_tracked:
                    in_tracked_run = False
            frag_object = max(0, tracked_runs - 1)

            stable_foreign_segments = [
                segment
                for segment in segments
                if segment.kind == "foreign_id" and int(segment.length) >= self.stable_min_frames
            ]
            stable_own_new_segments = [
                segment
                for segment in segments
                if segment.kind == "own_new_id" and int(segment.length) >= self.stable_min_frames
            ]

            first_failure_idx = None
            for idx, ok in enumerate(strict_flags):
                if not ok:
                    first_failure_idx = idx
                    break

            recovered_reference = False
            recovered_own_identity = False
            post_failure_strict_acc = None
            recovery_start_indices: list[int] = []
            in_recovery_run = False
            prev_visible_frame_id = None
            assigned_frames_timeline = [int(item["frame_id"]) for item in gt_cases]
            for idx, pred_id in enumerate(pred_timeline):
                frame_id_cur = int(assigned_frames_timeline[int(idx)])
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
                post_failure_strict_acc = float(sum(1 for flag in tail_strict if flag)) / float(len(tail_strict))
            elif first_failure_idx is not None:
                post_failure_strict_acc = 0.0

            tracking_correct_count = int(sum(1 for item in gt_cases if bool(item.get("strict_global_correct", False))))
            tracking_iou_sum = float(sum(float(item.get("tracking_iou", 0.0) or 0.0) for item in gt_cases))
            tracking_recall_object = safe_pct(tracking_correct_count, visible_n_frames)
            mean_tracking_iou_object = (
                float(tracking_iou_sum) / float(visible_n_frames) if visible_n_frames > 0 else None
            )
            mt_pt_ml_label = _mt_pt_ml_label_from_recall(tracking_recall_object)

            gt_area_fracs = [
                float(item["gt_area_frac"])
                for item in gt_cases
                if item.get("gt_area_frac", None) is not None
            ]
            out_row = {
                "gt_instance_id": int(gt_id),
                "gt_label": str(self.gt_meta[int(gt_id)]["label"]),
                "gt_class_name": self.gt_meta[int(gt_id)]["class_name"],
                "assigned_n_frames": int(assigned_n_frames),
                "visible_n_frames": int(visible_n_frames),
                "n_frames": int(visible_n_frames),
                "first_frame": int(gt_cases[0]["frame_id"]),
                "last_frame": int(gt_cases[-1]["frame_id"]),
                "reference_pred_id": None if assigned_ref_pred is None else int(assigned_ref_pred),
                "reference_pred_label": (
                    None
                    if assigned_ref_pred is None
                    else self.pred_meta.get(int(assigned_ref_pred), {}).get("instance_label", None)
                ),
                "strict_accuracy": (
                    float(strict_correct) / float(visible_n_frames) if visible_n_frames > 0 else 0.0
                ),
                "permissive_accuracy": (
                    float(permissive_correct) / float(visible_n_frames) if visible_n_frames > 0 else 0.0
                ),
                "perfect_strict": bool(strict_correct == visible_n_frames),
                "perfect_permissive": bool(permissive_correct == visible_n_frames),
                "tracking_recall_object": tracking_recall_object,
                "mean_tracking_iou_object": mean_tracking_iou_object,
                "mt_pt_ml_label": mt_pt_ml_label,
                "n_unique_pred_ids": int(len(unique_pred_ids)),
                "n_own_pred_ids": int(len(own_pred_ids)),
                "n_foreign_pred_ids": int(len(foreign_pred_ids)),
                "id_changes": int(max(0, len(segments) - 1)),
                "stable_foreign_segments": int(len(stable_foreign_segments)),
                "stable_own_new_segments": int(len(stable_own_new_segments)),
                "first_failure_frame": None if first_failure_idx is None else int(gt_cases[int(first_failure_idx)]["frame_id"]),
                "recovered_reference": bool(recovered_reference),
                "recovered_own_identity": bool(recovered_own_identity),
                "recovery_attempts": int(recovery_attempts),
                "recovery_success_reference": int(recovery_success_reference),
                "recovery_success_own_identity": int(recovery_success_own_identity),
                "recovery_success_duplicate_id": int(recovery_success_duplicate_id),
                "recovery_success_foreign_id": int(recovery_success_foreign_id),
                "recovery_rate_reference": safe_pct(recovery_success_reference, recovery_attempts),
                "recovery_rate_own_identity": safe_pct(recovery_success_own_identity, recovery_attempts),
                "recovery_rate_duplicate_id": safe_pct(recovery_success_duplicate_id, recovery_attempts),
                "recovery_rate_foreign_id": safe_pct(recovery_success_foreign_id, recovery_attempts),
                "post_failure_strict_accuracy": post_failure_strict_acc,
                "segments": [asdict(segment) for segment in segments],
                "idsw_object": int(idsw_object),
                "frag_object": int(frag_object),
                "pred_ids_timeline": [
                    None if pred_id is None else int(pred_id)
                    for pred_id in pred_timeline
                ],
                "frames_timeline": [int(item["frame_id"]) for item in gt_cases],
                "mean_visible_gt_in_frame": float(
                    np.mean([float(item.get("n_gt_visible_in_frame", 0) or 0) for item in gt_cases])
                ),
                "mean_total_distractors": float(
                    np.mean([float(item.get("n_total_distractors", 0) or 0) for item in gt_cases])
                ),
                "mean_same_class_distractors": float(
                    np.mean([float(item.get("n_same_class_distractors", 0) or 0) for item in gt_cases])
                ),
                "mean_gt_area_frac": float(np.mean(gt_area_fracs)) if gt_area_fracs else None,
            }
            for variant in self._iter_iou_variants():
                suffix = str(variant["suffix"])
                strict_key = f"strict_global_correct_{suffix}"
                permissive_key = f"permissive_global_correct_{suffix}"
                tracking_iou_key = f"tracking_iou_{suffix}"
                strict_correct_variant = int(sum(1 for item in gt_cases if bool(item.get(strict_key, False))))
                permissive_correct_variant = int(sum(1 for item in gt_cases if bool(item.get(permissive_key, False))))
                tracking_iou_sum_variant = float(
                    sum(float(item.get(tracking_iou_key, 0.0) or 0.0) for item in gt_cases)
                )
                tracking_recall_object_variant = safe_pct(strict_correct_variant, visible_n_frames)
                mean_tracking_iou_object_variant = (
                    float(tracking_iou_sum_variant) / float(visible_n_frames)
                    if visible_n_frames > 0
                    else None
                )
                out_row[f"strict_accuracy_{suffix}"] = (
                    float(strict_correct_variant) / float(visible_n_frames) if visible_n_frames > 0 else 0.0
                )
                out_row[f"permissive_accuracy_{suffix}"] = (
                    float(permissive_correct_variant) / float(visible_n_frames) if visible_n_frames > 0 else 0.0
                )
                out_row[f"perfect_strict_{suffix}"] = bool(strict_correct_variant == visible_n_frames)
                out_row[f"perfect_permissive_{suffix}"] = bool(permissive_correct_variant == visible_n_frames)
                out_row[f"tracking_recall_object_{suffix}"] = tracking_recall_object_variant
                out_row[f"mean_tracking_iou_object_{suffix}"] = mean_tracking_iou_object_variant
                out_row[f"mt_pt_ml_label_{suffix}"] = _mt_pt_ml_label_from_recall(tracking_recall_object_variant)
            per_object_rows.append(out_row)

        return per_object_rows

    def build_per_frame_rows(
        self,
        *,
        case_rows: list[dict[str, Any]],
        orphan_pred_rows: list[dict[str, Any]],
        reference_pred_by_gt: dict[int, int],
        canonical_gt_by_pred: dict[int, int],
    ) -> list[dict[str, Any]]:
        by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in case_rows:
            by_frame[int(row["frame_id"])].append(row)
        orphan_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in orphan_pred_rows:
            orphan_by_frame[int(row["frame_id"])].append(row)

        per_frame_rows: list[dict[str, Any]] = []
        for frame_id in sorted(set(by_frame.keys()) | set(orphan_by_frame.keys())):
            frame_cases = sorted(by_frame[int(frame_id)], key=lambda row: int(row["gt_instance_id"]))
            frame_orphans = sorted(
                orphan_by_frame[int(frame_id)],
                key=lambda row: (
                    int(row.get("pred_object_id", -1) if row.get("pred_object_id", None) is not None else -1),
                    int(row.get("det_id", -1)),
                ),
            )
            visible_n = int(len(frame_cases))
            matched_pred_n = int(sum(1 for row in frame_cases if row.get("pred_object_id", None) is not None))
            orphan_n = int(len(frame_orphans))
            pred_obs_total = int(matched_pred_n + orphan_n)
            orphan_area_fracs = [
                float(row["pred_area_frac"])
                for row in frame_orphans
                if row.get("pred_area_frac", None) is not None
            ]
            strict_correct = int(sum(1 for row in frame_cases if bool(row.get("strict_global_correct", False))))
            permissive_correct = int(sum(1 for row in frame_cases if bool(row.get("permissive_global_correct", False))))
            out_row = {
                "frame_id": int(frame_id),
                "n_objects": int(visible_n),
                "visible_n_objects": int(visible_n),
                "n_pred_observations": int(pred_obs_total),
                "n_orphan_pred_observations": int(orphan_n),
                "orphan_pred_rate": safe_pct(orphan_n, pred_obs_total),
                "mean_orphan_pred_area_frac": float(np.mean(orphan_area_fracs)) if orphan_area_fracs else None,
                "n_classes_visible": int(len({str(row.get("gt_class_name") or "unknown") for row in frame_cases})),
                "strict_correct": int(strict_correct),
                "permissive_correct": int(permissive_correct),
                "strict_accuracy": (float(strict_correct) / float(visible_n)) if visible_n > 0 else 0.0,
                "permissive_accuracy": (float(permissive_correct) / float(visible_n)) if visible_n > 0 else 0.0,
                "tracking_recall_frame": safe_pct(
                    sum(1 for row in frame_cases if bool(row.get("strict_global_correct", False))),
                    visible_n,
                ),
                "mean_tracking_iou_frame": (
                    float(sum(float(row.get("tracking_iou", 0.0) or 0.0) for row in frame_cases)) / float(visible_n)
                    if visible_n > 0
                    else None
                ),
                "n_new_gt": int(sum(1 for row in frame_cases if str(row.get("real_state")) == "new")),
                "n_existing_gt": int(sum(1 for row in frame_cases if str(row.get("real_state")) == "existing")),
            }
            for variant in self._iter_iou_variants():
                suffix = str(variant["suffix"])
                strict_key = f"strict_global_correct_{suffix}"
                permissive_key = f"permissive_global_correct_{suffix}"
                tracking_iou_key = f"tracking_iou_{suffix}"
                matched_key = f"matched_{suffix}"
                strict_correct_variant = int(sum(1 for row in frame_cases if bool(row.get(strict_key, False))))
                permissive_correct_variant = int(sum(1 for row in frame_cases if bool(row.get(permissive_key, False))))
                matched_variant = int(sum(1 for row in frame_cases if bool(row.get(matched_key, False))))
                out_row[f"n_matched_{suffix}"] = int(matched_variant)
                out_row[f"strict_correct_{suffix}"] = int(strict_correct_variant)
                out_row[f"permissive_correct_{suffix}"] = int(permissive_correct_variant)
                out_row[f"strict_accuracy_{suffix}"] = (
                    float(strict_correct_variant) / float(visible_n) if visible_n > 0 else 0.0
                )
                out_row[f"permissive_accuracy_{suffix}"] = (
                    float(permissive_correct_variant) / float(visible_n) if visible_n > 0 else 0.0
                )
                out_row[f"tracking_recall_frame_{suffix}"] = safe_pct(strict_correct_variant, visible_n)
                out_row[f"mean_tracking_iou_frame_{suffix}"] = (
                    float(sum(float(row.get(tracking_iou_key, 0.0) or 0.0) for row in frame_cases)) / float(visible_n)
                    if visible_n > 0
                    else None
                )
            per_frame_rows.append(out_row)

        return per_frame_rows

    def build_pred_rows(
        self,
        *,
        case_rows: list[dict[str, Any]],
        orphan_pred_rows: list[dict[str, Any]],
        reference_pred_by_gt: dict[int, int],
        canonical_gt_by_pred: dict[int, int],
    ) -> list[dict[str, Any]]:
        by_pred: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in case_rows:
            pred_id = row.get("pred_object_id", None)
            if pred_id is None:
                continue
            by_pred[int(pred_id)].append(row)
        orphan_by_pred: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in orphan_pred_rows:
            pred_id = row.get("pred_object_id", None)
            if pred_id is None:
                continue
            orphan_by_pred[int(pred_id)].append(row)

        reverse_reference_gt_by_pred = {
            int(pred_id): int(gt_id)
            for gt_id, pred_id in reference_pred_by_gt.items()
            if pred_id is not None
        }

        pred_rows: list[dict[str, Any]] = []
        for pred_id in sorted(set(by_pred.keys()) | set(orphan_by_pred.keys())):
            pred_cases = sorted(by_pred[int(pred_id)], key=lambda row: (int(row["frame_id"]), int(row["gt_instance_id"])))
            pred_orphans = sorted(
                orphan_by_pred[int(pred_id)],
                key=lambda row: (int(row["frame_id"]), int(row.get("det_id", -1))),
            )
            gt_users = sorted({int(row["gt_instance_id"]) for row in pred_cases})
            pred_frames = sorted(
                {int(row["frame_id"]) for row in pred_cases}
                | {int(row["frame_id"]) for row in pred_orphans}
            )
            orphan_area_fracs = [
                float(row["pred_area_frac"])
                for row in pred_orphans
                if row.get("pred_area_frac", None) is not None
            ]
            total_observations = int(len(pred_cases) + len(pred_orphans))
            pred_rows.append(
                {
                    "pred_object_id": int(pred_id),
                    "pred_instance_label": self.pred_meta.get(int(pred_id), {}).get("instance_label", None),
                    "pred_class_name": self.pred_meta.get(int(pred_id), {}).get("class_name", None),
                    "canonical_gt": int(canonical_gt_by_pred.get(int(pred_id), -1)),
                    "majority_gt": int(canonical_gt_by_pred.get(int(pred_id), -1)),
                    "reference_gt": int(reverse_reference_gt_by_pred.get(int(pred_id), -1)),
                    "gt_users": [int(x) for x in gt_users],
                    "n_gt_users": int(len(gt_users)),
                    "first_frame": None if not pred_frames else int(pred_frames[0]),
                    "last_frame": None if not pred_frames else int(pred_frames[-1]),
                    "n_frames_present": int(len(pred_frames)),
                    "n_observations_total": int(total_observations),
                    "n_gt_overlap_observations": int(len(pred_cases)),
                    "n_orphan_observations": int(len(pred_orphans)),
                    "n_orphan_frames": int(len({int(row["frame_id"]) for row in pred_orphans})),
                    "has_orphan_observations": bool(len(pred_orphans) > 0),
                    "orphan_observation_ratio": safe_pct(len(pred_orphans), total_observations),
                    "mean_orphan_pred_area_frac": float(np.mean(orphan_area_fracs)) if orphan_area_fracs else None,
                    "is_orphan_only_track": bool(len(pred_cases) == 0 and len(pred_orphans) > 0),
                    "is_pure_track": bool(len(gt_users) <= 1),
                    "is_fragment_track": bool(len(gt_users) > 1),
                    "is_foreign_track": bool(
                        int(canonical_gt_by_pred.get(int(pred_id), -1)) != int(reverse_reference_gt_by_pred.get(int(pred_id), -1))
                    ),
                }
            )
        return pred_rows

    @staticmethod
    def build_events(*, case_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        frame_maps: dict[int, dict[int, int]] = defaultdict(dict)
        for row in case_rows:
            pred_id = row.get("pred_object_id", None)
            if pred_id is None:
                continue
            frame_maps[int(row["frame_id"])][int(row["gt_instance_id"])] = int(pred_id)

        swap_events: list[dict[str, Any]] = []
        theft_with_new_id_events: list[dict[str, Any]] = []
        theft_with_displacement_events: list[dict[str, Any]] = []
        sorted_frames = sorted(frame_maps)
        for idx in range(1, len(sorted_frames)):
            prev_f = int(sorted_frames[idx - 1])
            cur_f = int(sorted_frames[idx])
            prev_map = frame_maps[prev_f]
            cur_map = frame_maps[cur_f]
            common_gt = sorted(set(prev_map.keys()) & set(cur_map.keys()))
            prev_pred_to_gt = {int(pid): int(gt) for gt, pid in prev_map.items()}

            seen_pairs: set[tuple[int, int]] = set()
            for gt_id in common_gt:
                prev_pid = int(prev_map[gt_id])
                cur_pid = int(cur_map[gt_id])
                if prev_pid == cur_pid:
                    continue

                owner_prev = prev_pred_to_gt.get(cur_pid, None)
                if owner_prev is None or int(owner_prev) == int(gt_id):
                    continue

                other_gt = int(owner_prev)
                if other_gt not in cur_map:
                    continue

                other_prev = int(prev_map[other_gt])
                other_cur = int(cur_map[other_gt])

                if other_cur == prev_pid:
                    pair = tuple(sorted((int(gt_id), int(other_gt))))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    swap_events.append(
                        {
                            "frame_id": int(cur_f),
                            "gt_a": int(pair[0]),
                            "gt_b": int(pair[1]),
                            "pred_a_prev": int(prev_map[pair[0]]),
                            "pred_b_prev": int(prev_map[pair[1]]),
                        }
                    )
                    continue

                if other_cur not in prev_pred_to_gt:
                    theft_with_new_id_events.append(
                        {
                            "frame_id": int(cur_f),
                            "thief_gt": int(gt_id),
                            "victim_gt": int(other_gt),
                            "stolen_pred_id": int(cur_pid),
                            "victim_new_pred_id": int(other_cur),
                        }
                    )
                else:
                    theft_with_displacement_events.append(
                        {
                            "frame_id": int(cur_f),
                            "thief_gt": int(gt_id),
                            "victim_gt": int(other_gt),
                            "stolen_pred_id": int(cur_pid),
                            "victim_pred_id_after": int(other_cur),
                        }
                    )

        return {
            "swap": swap_events,
            "theft_with_new_id": theft_with_new_id_events,
            "theft_with_displacement": theft_with_displacement_events,
        }

    def build_summary(
        self,
        *,
        case_rows: list[dict[str, Any]],
        orphan_pred_rows: list[dict[str, Any]],
        per_object_rows: list[dict[str, Any]],
        per_frame_rows: list[dict[str, Any]],
        pred_rows: list[dict[str, Any]],
        events: dict[str, list[dict[str, Any]]],
        tracking_identity_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        n_frames_total = int(len({int(frame_id) for frame_id in self.frames_seen}))
        n_objects_total = int(len({int(gt_id) for gt_id in self.gt_meta.keys()}))
        perfect_objects_strict = int(sum(1 for row in per_object_rows if bool(row.get("perfect_strict", False))))
        perfect_objects_permissive = int(sum(1 for row in per_object_rows if bool(row.get("perfect_permissive", False))))
        objects_with_fragmentation = int(sum(1 for row in per_object_rows if int(row.get("n_own_pred_ids", 0) or 0) > 1))
        objects_with_foreign_id_use = int(sum(1 for row in per_object_rows if int(row.get("n_foreign_pred_ids", 0) or 0) > 0))
        objects_recovered_reference = int(sum(1 for row in per_object_rows if bool(row.get("recovered_reference", False))))
        objects_recovered_own_identity = int(sum(1 for row in per_object_rows if bool(row.get("recovered_own_identity", False))))
        total_recovery_attempts = int(sum(int(row.get("recovery_attempts", 0) or 0) for row in per_object_rows))
        total_recovery_success_reference = int(sum(int(row.get("recovery_success_reference", 0) or 0) for row in per_object_rows))
        total_recovery_success_own_identity = int(sum(int(row.get("recovery_success_own_identity", 0) or 0) for row in per_object_rows))
        total_recovery_success_duplicate_id = int(sum(int(row.get("recovery_success_duplicate_id", 0) or 0) for row in per_object_rows))
        total_recovery_success_foreign_id = int(sum(int(row.get("recovery_success_foreign_id", 0) or 0) for row in per_object_rows))
        total_id_changes = int(sum(int(row.get("id_changes", 0) or 0) for row in per_object_rows))
        total_stable_foreign_segments = int(sum(int(row.get("stable_foreign_segments", 0) or 0) for row in per_object_rows))
        total_stable_own_new_segments = int(sum(int(row.get("stable_own_new_segments", 0) or 0) for row in per_object_rows))
        n_mt_objects = int(sum(1 for row in per_object_rows if str(row.get("mt_pt_ml_label", "")) == "MT"))
        n_pt_objects = int(sum(1 for row in per_object_rows if str(row.get("mt_pt_ml_label", "")) == "PT"))
        n_ml_objects = int(sum(1 for row in per_object_rows if str(row.get("mt_pt_ml_label", "")) == "ML"))
        n_visible_gt_observations = int(len(case_rows))
        n_matched_gt_observations = int(
            sum(1 for row in case_rows if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0)
        )
        total_visible_frame_objects = int(sum(int(row.get("n_objects", 0) or 0) for row in per_frame_rows))
        total_strict_correct_visible = int(sum(int(row.get("strict_correct", 0) or 0) for row in per_frame_rows))
        total_permissive_correct_visible = int(sum(int(row.get("permissive_correct", 0) or 0) for row in per_frame_rows))
        n_orphan_pred_observations = int(len(orphan_pred_rows))
        n_pred_observations_total = int(tracking_identity_metrics.get("n_pred_observations_total", 0) or 0)
        n_unique_orphan_pred_ids = int(tracking_identity_metrics.get("n_unique_orphan_pred_ids", 0) or 0)
        n_orphan_only_pred_ids = int(tracking_identity_metrics.get("n_orphan_only_pred_ids", 0) or 0)
        pred_tracks_with_orphan = int(sum(1 for row in pred_rows if bool(row.get("has_orphan_observations", False))))
        orphan_only_tracks = int(sum(1 for row in pred_rows if bool(row.get("is_orphan_only_track", False))))
        n_unique_real_pred_tracks = int(sum(1 for row in pred_rows if not bool(row.get("is_orphan_only_track", False))))
        n_total_pred_tracks_including_orphan_only = int(len(pred_rows))
        out = {
            "n_frames": int(n_frames_total),
            "n_objects": int(n_objects_total),
            "n_visible_gt_observations": int(n_visible_gt_observations),
            "n_matched_gt_observations": int(n_matched_gt_observations),
            "n_pred_observations_total": int(n_pred_observations_total),
            "n_orphan_pred_observations": int(n_orphan_pred_observations),
            "n_unique_orphan_pred_ids": int(n_unique_orphan_pred_ids),
            "n_orphan_only_pred_ids": int(n_orphan_only_pred_ids),
            "orphan_pred_rate": tracking_identity_metrics.get("orphan_pred_rate", None),
            "mean_orphan_pred_area_frac": tracking_identity_metrics.get("mean_orphan_pred_area_frac", None),
            "n_unique_real_pred_tracks": int(n_unique_real_pred_tracks),
            "n_total_pred_tracks_including_orphan_only": int(n_total_pred_tracks_including_orphan_only),
            "pred_track_surplus_vs_gt": int(n_unique_real_pred_tracks - n_objects_total),
            "pred_track_inflation_factor": (
                float(n_unique_real_pred_tracks) / float(n_objects_total) if n_objects_total > 0 else None
            ),
            "n_pred_tracks_with_orphan_observations": int(pred_tracks_with_orphan),
            "n_orphan_only_pred_tracks": int(orphan_only_tracks),
            "label_summary": (
                f"{n_objects_total} GT objects -> {n_unique_real_pred_tracks} real tracker labels "
                f"({n_unique_real_pred_tracks - n_objects_total:+d}, "
                f"{(float(n_unique_real_pred_tracks) / float(n_objects_total)):.3f}x)"
                if n_objects_total > 0
                else None
            ),
            "stable_min_frames": int(self.stable_min_frames),
            "global_frame_accuracy_strict": (
                float(total_strict_correct_visible) / float(total_visible_frame_objects)
                if total_visible_frame_objects > 0
                else 0.0
            ),
            "global_frame_accuracy_permissive": (
                float(total_permissive_correct_visible) / float(total_visible_frame_objects)
                if total_visible_frame_objects > 0
                else 0.0
            ),
            "global_object_accuracy_strict": (
                float(perfect_objects_strict) / float(n_objects_total) if n_objects_total > 0 else 0.0
            ),
            "global_object_accuracy_permissive": (
                float(perfect_objects_permissive) / float(n_objects_total) if n_objects_total > 0 else 0.0
            ),
            "tracking_recall": tracking_identity_metrics.get("tracking_recall", None),
            "mean_tracking_iou": tracking_identity_metrics.get("mean_tracking_iou", None),
            "deta": tracking_identity_metrics.get("deta", None),
            "assa": tracking_identity_metrics.get("assa", None),
            "hota": tracking_identity_metrics.get("hota", None),
            "n_mt_objects": int(n_mt_objects),
            "n_pt_objects": int(n_pt_objects),
            "n_ml_objects": int(n_ml_objects),
            "mt": safe_pct(n_mt_objects, n_objects_total),
            "pt": safe_pct(n_pt_objects, n_objects_total),
            "ml": safe_pct(n_ml_objects, n_objects_total),
            "objects_fragmented": int(objects_with_fragmentation),
            "objects_with_foreign_id_use": int(objects_with_foreign_id_use),
            "objects_recovered_reference": int(objects_recovered_reference),
            "objects_recovered_own_identity": int(objects_recovered_own_identity),
            "recovery_attempts_total": int(total_recovery_attempts),
            "recovery_success_reference_total": int(total_recovery_success_reference),
            "recovery_success_own_identity_total": int(total_recovery_success_own_identity),
            "recovery_success_duplicate_id_total": int(total_recovery_success_duplicate_id),
            "recovery_success_foreign_id_total": int(total_recovery_success_foreign_id),
            "recovery_rate_reference": safe_pct(total_recovery_success_reference, total_recovery_attempts),
            "recovery_rate_own_identity": safe_pct(total_recovery_success_own_identity, total_recovery_attempts),
            "recovery_rate_duplicate_id": safe_pct(total_recovery_success_duplicate_id, total_recovery_attempts),
            "recovery_rate_foreign_id": safe_pct(total_recovery_success_foreign_id, total_recovery_attempts),
            "id_changes_total": int(total_id_changes),
            "stable_foreign_segments_total": int(total_stable_foreign_segments),
            "stable_own_new_segments_total": int(total_stable_own_new_segments),
            "swap_events_total": int(len(events.get("swap", []) or [])),
            "theft_with_new_id_total": int(len(events.get("theft_with_new_id", []) or [])),
            "theft_with_displacement_total": int(len(events.get("theft_with_displacement", []) or [])),
        }
        for variant in self._iter_iou_variants():
            suffix = str(variant["suffix"])
            n_matched_variant = int(
                sum(1 for row in case_rows if bool(row.get(f"matched_{suffix}", False)))
            )
            perfect_objects_strict_variant = int(
                sum(1 for row in per_object_rows if bool(row.get(f"perfect_strict_{suffix}", False)))
            )
            perfect_objects_permissive_variant = int(
                sum(1 for row in per_object_rows if bool(row.get(f"perfect_permissive_{suffix}", False)))
            )
            total_strict_correct_visible_variant = int(
                sum(int(row.get(f"strict_correct_{suffix}", 0) or 0) for row in per_frame_rows)
            )
            total_permissive_correct_visible_variant = int(
                sum(int(row.get(f"permissive_correct_{suffix}", 0) or 0) for row in per_frame_rows)
            )
            out[f"iou_threshold_{suffix}"] = float(variant["threshold"])
            out[f"n_matched_gt_observations_{suffix}"] = int(n_matched_variant)
            out[f"global_frame_accuracy_strict_{suffix}"] = (
                float(total_strict_correct_visible_variant) / float(total_visible_frame_objects)
                if total_visible_frame_objects > 0
                else 0.0
            )
            out[f"global_frame_accuracy_permissive_{suffix}"] = (
                float(total_permissive_correct_visible_variant) / float(total_visible_frame_objects)
                if total_visible_frame_objects > 0
                else 0.0
            )
            out[f"global_object_accuracy_strict_{suffix}"] = (
                float(perfect_objects_strict_variant) / float(n_objects_total) if n_objects_total > 0 else 0.0
            )
            out[f"global_object_accuracy_permissive_{suffix}"] = (
                float(perfect_objects_permissive_variant) / float(n_objects_total) if n_objects_total > 0 else 0.0
            )
            out[f"tracking_recall_{suffix}"] = tracking_identity_metrics.get(
                f"tracking_recall_{suffix}",
                None,
            )
            out[f"mean_tracking_iou_{suffix}"] = tracking_identity_metrics.get(
                f"mean_tracking_iou_{suffix}",
                None,
            )
            out[f"deta_{suffix}"] = tracking_identity_metrics.get(f"deta_{suffix}", None)
            out[f"assa_{suffix}"] = tracking_identity_metrics.get(f"assa_{suffix}", None)
            out[f"hota_{suffix}"] = tracking_identity_metrics.get(f"hota_{suffix}", None)
        return out

    def build_per_class_rows(
        self,
        *,
        case_rows: list[dict[str, Any]],
        per_object_rows: list[dict[str, Any]],
        pred_rows: list[dict[str, Any]],
        tracking_identity_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        gt_objects_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_object_rows:
            gt_objects_by_class[str(row.get("gt_class_name") or "unknown")].append(row)

        pred_tracks_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pred_rows:
            pred_tracks_by_class[str(row.get("pred_class_name") or "unknown")].append(row)

        case_rows_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in case_rows:
            case_rows_by_class[str(row.get("gt_class_name") or "unknown")].append(row)

        out: list[dict[str, Any]] = []
        for class_name in sorted(set(gt_objects_by_class.keys()) | set(pred_tracks_by_class.keys()) | set(case_rows_by_class.keys())):
            class_gt_rows = list(gt_objects_by_class.get(str(class_name), []))
            class_pred_rows = list(pred_tracks_by_class.get(str(class_name), []))
            class_case_rows = list(case_rows_by_class.get(str(class_name), []))
            total_visible = int(sum(int(row.get("n_frames", 0) or 0) for row in class_gt_rows))
            n_gt_objects = int(len(class_gt_rows))
            n_pred_tracks = int(len(class_pred_rows))
            n_tracking_correct = int(sum(1 for row in class_case_rows if bool(row.get("strict_global_correct", False))))
            n_matched = int(
                sum(1 for row in class_case_rows if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0)
            )
            mean_gt_area_vals = [row.get("mean_gt_area_frac", None) for row in class_gt_rows if row.get("mean_gt_area_frac", None) is not None]
            out_row = {
                "class_name": str(class_name),
                "n_gt_objects": int(n_gt_objects),
                "n_real_pred_tracks": int(n_pred_tracks),
                "pred_track_surplus_vs_gt": int(n_pred_tracks - n_gt_objects),
                "pred_track_inflation_factor": (
                    float(n_pred_tracks) / float(n_gt_objects) if n_gt_objects > 0 else None
                ),
                "weighted_strict_accuracy": (
                    float(sum(float(row.get("strict_accuracy", 0.0) or 0.0) * float(row.get("n_frames", 0) or 0) for row in class_gt_rows)) / float(total_visible)
                    if total_visible > 0
                    else 0.0
                ),
                "weighted_permissive_accuracy": (
                    float(sum(float(row.get("permissive_accuracy", 0.0) or 0.0) * float(row.get("n_frames", 0) or 0) for row in class_gt_rows)) / float(total_visible)
                    if total_visible > 0
                    else 0.0
                ),
                "mean_pred_ids_per_gt": (
                    float(np.mean([float(row.get("n_unique_pred_ids", 0) or 0) for row in class_gt_rows]))
                    if class_gt_rows
                    else None
                ),
                "mean_id_changes_per_gt": (
                    float(np.mean([float(row.get("id_changes", 0) or 0) for row in class_gt_rows]))
                    if class_gt_rows
                    else None
                ),
                "tracking_recall": safe_pct(n_tracking_correct, len(class_case_rows)),
                "mean_tracking_iou": (
                    float(sum(float(row.get("tracking_iou", 0.0) or 0.0) for row in class_case_rows)) / float(len(class_case_rows))
                    if class_case_rows
                    else None
                ),
                "deta": safe_pct(n_matched, len(class_case_rows)),
                "assa": safe_pct(n_tracking_correct, n_matched),
                "hota": (
                    float(np.sqrt(max(0.0, float(safe_pct(n_matched, len(class_case_rows)) or 0.0) * float(safe_pct(n_tracking_correct, n_matched) or 0.0))))
                    if class_case_rows
                    else None
                ),
                "n_visible_gt_observations": int(len(class_case_rows)),
                "n_matched_gt_observations": int(n_matched),
                "n_mt_objects": int(sum(1 for row in class_gt_rows if str(row.get("mt_pt_ml_label", "")) == "MT")),
                "n_pt_objects": int(sum(1 for row in class_gt_rows if str(row.get("mt_pt_ml_label", "")) == "PT")),
                "n_ml_objects": int(sum(1 for row in class_gt_rows if str(row.get("mt_pt_ml_label", "")) == "ML")),
                "gt_objects_with_foreign_id_use": int(
                    sum(1 for row in class_gt_rows if int(row.get("n_foreign_pred_ids", 0) or 0) > 0)
                ),
                "recovery_attempts_total": int(
                    sum(int(row.get("recovery_attempts", 0) or 0) for row in class_gt_rows)
                ),
                "recovery_success_reference_total": int(
                    sum(int(row.get("recovery_success_reference", 0) or 0) for row in class_gt_rows)
                ),
                "recovery_success_own_identity_total": int(
                    sum(int(row.get("recovery_success_own_identity", 0) or 0) for row in class_gt_rows)
                ),
                "recovery_success_duplicate_id_total": int(
                    sum(int(row.get("recovery_success_duplicate_id", 0) or 0) for row in class_gt_rows)
                ),
                "recovery_success_foreign_id_total": int(
                    sum(int(row.get("recovery_success_foreign_id", 0) or 0) for row in class_gt_rows)
                ),
                "recovery_rate_reference": safe_pct(
                    sum(int(row.get("recovery_success_reference", 0) or 0) for row in class_gt_rows),
                    sum(int(row.get("recovery_attempts", 0) or 0) for row in class_gt_rows),
                ),
                "recovery_rate_own_identity": safe_pct(
                    sum(int(row.get("recovery_success_own_identity", 0) or 0) for row in class_gt_rows),
                    sum(int(row.get("recovery_attempts", 0) or 0) for row in class_gt_rows),
                ),
                "recovery_rate_duplicate_id": safe_pct(
                    sum(int(row.get("recovery_success_duplicate_id", 0) or 0) for row in class_gt_rows),
                    sum(int(row.get("recovery_attempts", 0) or 0) for row in class_gt_rows),
                ),
                "recovery_rate_foreign_id": safe_pct(
                    sum(int(row.get("recovery_success_foreign_id", 0) or 0) for row in class_gt_rows),
                    sum(int(row.get("recovery_attempts", 0) or 0) for row in class_gt_rows),
                ),
                "mean_gt_area_frac": float(np.mean(mean_gt_area_vals)) if mean_gt_area_vals else None,
            }
            for variant in self._iter_iou_variants():
                suffix = str(variant["suffix"])
                strict_key = f"strict_global_correct_{suffix}"
                tracking_iou_key = f"tracking_iou_{suffix}"
                matched_key = f"matched_{suffix}"
                n_tracking_correct_variant = int(sum(1 for row in class_case_rows if bool(row.get(strict_key, False))))
                n_matched_variant = int(sum(1 for row in class_case_rows if bool(row.get(matched_key, False))))
                out_row[f"weighted_strict_accuracy_{suffix}"] = (
                    float(sum(float(row.get(f"strict_accuracy_{suffix}", 0.0) or 0.0) * float(row.get("n_frames", 0) or 0) for row in class_gt_rows)) / float(total_visible)
                    if total_visible > 0
                    else 0.0
                )
                out_row[f"weighted_permissive_accuracy_{suffix}"] = (
                    float(sum(float(row.get(f"permissive_accuracy_{suffix}", 0.0) or 0.0) * float(row.get("n_frames", 0) or 0) for row in class_gt_rows)) / float(total_visible)
                    if total_visible > 0
                    else 0.0
                )
                out_row[f"tracking_recall_{suffix}"] = safe_pct(n_tracking_correct_variant, len(class_case_rows))
                out_row[f"mean_tracking_iou_{suffix}"] = (
                    float(sum(float(row.get(tracking_iou_key, 0.0) or 0.0) for row in class_case_rows)) / float(len(class_case_rows))
                    if class_case_rows
                    else None
                )
                out_row[f"deta_{suffix}"] = safe_pct(n_matched_variant, len(class_case_rows))
                out_row[f"assa_{suffix}"] = safe_pct(n_tracking_correct_variant, n_matched_variant)
                out_row[f"hota_{suffix}"] = (
                    float(
                        np.sqrt(
                            max(
                                0.0,
                                float(safe_pct(n_matched_variant, len(class_case_rows)) or 0.0)
                                * float(safe_pct(n_tracking_correct_variant, n_matched_variant) or 0.0),
                            )
                        )
                    )
                    if class_case_rows
                    else None
                )
                out_row[f"n_matched_gt_observations_{suffix}"] = int(n_matched_variant)
                out_row[f"n_mt_objects_{suffix}"] = int(
                    sum(1 for row in class_gt_rows if str(row.get(f"mt_pt_ml_label_{suffix}", "")) == "MT")
                )
                out_row[f"n_pt_objects_{suffix}"] = int(
                    sum(1 for row in class_gt_rows if str(row.get(f"mt_pt_ml_label_{suffix}", "")) == "PT")
                )
                out_row[f"n_ml_objects_{suffix}"] = int(
                    sum(1 for row in class_gt_rows if str(row.get(f"mt_pt_ml_label_{suffix}", "")) == "ML")
                )
            out.append(out_row)
        return out

    def compute_tracking_identity_metrics(
        self,
        *,
        case_rows: list[dict[str, Any]],
        orphan_pred_rows: list[dict[str, Any]],
        per_object_rows: list[dict[str, Any]],
        reference_pred_by_gt: dict[int, int],
    ) -> dict[str, Any]:
        n_gt_observations = int(len(case_rows))
        visible_rows = [row for row in case_rows if row.get("pred_object_id", None) is not None]
        n_pred_observations = int(len(visible_rows))
        n_orphan_pred_observations = int(len(orphan_pred_rows))
        n_pred_observations_total = int(n_pred_observations + n_orphan_pred_observations)
        visible_pred_ids = {int(row["pred_object_id"]) for row in visible_rows}
        orphan_pred_ids = {
            int(row["pred_object_id"])
            for row in orphan_pred_rows
            if row.get("pred_object_id", None) is not None
        }
        orphan_area_fracs = [
            float(row["pred_area_frac"])
            for row in orphan_pred_rows
            if row.get("pred_area_frac", None) is not None
        ]
        idtp = int(sum(1 for row in case_rows if bool(row.get("strict_global_correct", False))))
        idfp = int(max(0, n_pred_observations - idtp))
        idfn = int(max(0, n_gt_observations - idtp))
        n_matched = int(
            sum(1 for row in case_rows if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0)
        )
        tracking_iou_sum = float(sum(float(row.get("tracking_iou", 0.0) or 0.0) for row in case_rows))
        out = {
            "n_gt_observations": int(n_gt_observations),
            "n_pred_observations": int(n_pred_observations),
            "n_pred_observations_total": int(n_pred_observations_total),
            "n_orphan_pred_observations": int(n_orphan_pred_observations),
            "n_unique_gt_ids": int(len({int(row["gt_instance_id"]) for row in case_rows})),
            "n_unique_pred_ids": int(len({int(row["pred_object_id"]) for row in visible_rows})),
            "n_unique_pred_ids_total": int(len(visible_pred_ids | orphan_pred_ids)),
            "n_unique_orphan_pred_ids": int(len(orphan_pred_ids)),
            "n_orphan_only_pred_ids": int(len(orphan_pred_ids - visible_pred_ids)),
            "n_matched_gt_observations": int(n_matched),
            "idtp": int(idtp),
            "idfp": int(idfp),
            "idfn": int(idfn),
            "idp": safe_pct(idtp, idtp + idfp),
            "idr": safe_pct(idtp, idtp + idfn),
            "idf1": safe_pct(2 * idtp, 2 * idtp + idfp + idfn),
            "idsw": int(sum(int(row.get("idsw_object", 0) or 0) for row in per_object_rows)),
            "frag": int(sum(int(row.get("frag_object", 0) or 0) for row in per_object_rows)),
            "tracking_recall": safe_pct(idtp, n_gt_observations),
            "orphan_pred_rate": safe_pct(n_orphan_pred_observations, n_pred_observations_total),
            "mean_orphan_pred_area_frac": float(np.mean(orphan_area_fracs)) if orphan_area_fracs else None,
            "mean_tracking_iou": (
                float(tracking_iou_sum) / float(n_gt_observations) if n_gt_observations > 0 else None
            ),
            "deta": safe_pct(n_matched, n_gt_observations),
            "assa": safe_pct(idtp, n_matched),
            "hota": (
                float(np.sqrt(max(0.0, float(safe_pct(n_matched, n_gt_observations) or 0.0) * float(safe_pct(idtp, n_matched) or 0.0))))
                if n_gt_observations > 0 and n_matched > 0
                else None
            ),
            "reference_pred_by_gt": {int(k): int(v) for k, v in reference_pred_by_gt.items()},
        }
        for variant in self._iter_iou_variants():
            suffix = str(variant["suffix"])
            strict_key = f"strict_global_correct_{suffix}"
            matched_key = f"matched_{suffix}"
            tracking_iou_key = f"tracking_iou_{suffix}"
            idtp_variant = int(sum(1 for row in case_rows if bool(row.get(strict_key, False))))
            n_matched_variant = int(sum(1 for row in case_rows if bool(row.get(matched_key, False))))
            tracking_iou_sum_variant = float(
                sum(float(row.get(tracking_iou_key, 0.0) or 0.0) for row in case_rows)
            )
            idfp_variant = int(max(0, n_pred_observations - idtp_variant))
            idfn_variant = int(max(0, n_gt_observations - idtp_variant))
            out[f"iou_threshold_{suffix}"] = float(variant["threshold"])
            out[f"n_matched_gt_observations_{suffix}"] = int(n_matched_variant)
            out[f"idtp_{suffix}"] = int(idtp_variant)
            out[f"idfp_{suffix}"] = int(idfp_variant)
            out[f"idfn_{suffix}"] = int(idfn_variant)
            out[f"idp_{suffix}"] = safe_pct(idtp_variant, idtp_variant + idfp_variant)
            out[f"idr_{suffix}"] = safe_pct(idtp_variant, idtp_variant + idfn_variant)
            out[f"idf1_{suffix}"] = safe_pct(
                2 * idtp_variant,
                2 * idtp_variant + idfp_variant + idfn_variant,
            )
            out[f"tracking_recall_{suffix}"] = safe_pct(idtp_variant, n_gt_observations)
            out[f"mean_tracking_iou_{suffix}"] = (
                float(tracking_iou_sum_variant) / float(n_gt_observations) if n_gt_observations > 0 else None
            )
            out[f"deta_{suffix}"] = safe_pct(n_matched_variant, n_gt_observations)
            out[f"assa_{suffix}"] = safe_pct(idtp_variant, n_matched_variant)
            out[f"hota_{suffix}"] = (
                float(
                    np.sqrt(
                        max(
                            0.0,
                            float(safe_pct(n_matched_variant, n_gt_observations) or 0.0)
                            * float(safe_pct(idtp_variant, n_matched_variant) or 0.0),
                        )
                    )
                )
                if n_gt_observations > 0 and n_matched_variant > 0
                else None
            )
        return out

    def build_global_reference_mapping(self, *, case_rows: list[dict[str, Any]]) -> dict[int, int]:
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as e:
            raise RuntimeError("The generic evaluator requires scipy for global GT<->pred matching.") from e

        overlap_counts: dict[tuple[int, int], int] = defaultdict(int)
        overlap_iou_sum: dict[tuple[int, int], float] = defaultdict(float)
        gt_ids_by_class: dict[str, set[int]] = defaultdict(set)
        pred_ids_by_class: dict[str, set[int]] = defaultdict(set)

        for row in case_rows:
            pred_id = row.get("pred_object_id", None)
            if pred_id is None:
                continue
            gt_id = int(row["gt_instance_id"])
            pred_id = int(pred_id)
            gt_class_name = str(row.get("gt_class_name") or "unknown")
            pred_class_name = str(row.get("pred_class_name") or gt_class_name or "unknown")
            gt_ids_by_class[gt_class_name].add(gt_id)
            pred_ids_by_class[pred_class_name].add(pred_id)
            overlap_counts[(gt_id, pred_id)] += 1
            overlap_iou_sum[(gt_id, pred_id)] += float(row.get("iou", 0.0) or 0.0)

        reference_pred_by_gt: dict[int, int] = {}
        for class_name in sorted(set(gt_ids_by_class.keys()) | set(pred_ids_by_class.keys())):
            gt_ids = sorted(int(x) for x in gt_ids_by_class.get(class_name, set()))
            pred_ids = sorted(int(x) for x in pred_ids_by_class.get(class_name, set()))
            if not gt_ids or not pred_ids:
                continue

            weight = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float64)
            for i, gt_id in enumerate(gt_ids):
                for j, pred_id in enumerate(pred_ids):
                    count = int(overlap_counts.get((int(gt_id), int(pred_id)), 0))
                    iou_sum = float(overlap_iou_sum.get((int(gt_id), int(pred_id)), 0.0))
                    if count <= 0 and iou_sum <= 0.0:
                        continue
                    weight[i, j] = float(count) * 1000.0 + float(iou_sum)

            row_ind, col_ind = linear_sum_assignment(-weight)
            for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
                score = float(weight[int(row_idx), int(col_idx)])
                if score <= 0.0:
                    continue
                reference_pred_by_gt[int(gt_ids[int(row_idx)])] = int(pred_ids[int(col_idx)])

        return reference_pred_by_gt
