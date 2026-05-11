from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


SYNTHETIC_PROVISIONAL_NEW_BASE = 1_000_000_000
SYNTHETIC_CREATED_NEW_BASE = 2_000_000_000


@dataclass
class AssignmentRecord:
    frame_id: int
    gt_instance_id: int
    gt_label: str
    gt_class_name: str | None
    pred_object_id: int
    pred_instance_label: str | None
    pred_class_name: str | None
    det_id: int
    iou: float


@dataclass
class SegmentRecord:
    pred_object_id: int
    start_frame: int
    end_frame: int
    length: int
    canonical_owner_gt: int
    kind: str


@dataclass
class DecisionCaseRecord:
    frame_id: int
    gt_instance_id: int
    gt_label: str
    gt_class_name: str | None
    det_id: int
    iou: float
    real_is_new: bool
    final_decision: str
    final_reason: str
    firm_pred_object_id: int | None
    ambiguous_candidate_ids: list[int]
    provisional_parent_ids: list[int]
    provisional_temp_id: int | None
    created_object_id: int | None
    created_origin_provisional_temp_id: int | None
    created_origin_parent_ids: list[int]
    gt_area_px: int = 0
    frame_gt_count: int = 0
    frame_same_class_count: int = 0
    frame_area_px: int = 0
    best_sim_object_id: int | None = None
    best_sim_score: float = 0.0
    best_final_candidate_object_id: int | None = None
    best_final_candidate_score: float = 0.0
    best_sim_margin: float | None = None
    best_final_margin: float | None = None
    match_source: str = "association"
    distance_used: bool = False
    distance_resolved: bool = False
    neighbor_sets_available: bool = False
    context_intervened: bool = False
    context_rescue_applied: bool = False
    context_veto_candidate_count: int = 0
    selected_candidate_score_sets: float = 0.0
    selected_candidate_quality_sets: float = 0.0


def safe_pct(num: int, den: int) -> float | None:
    if den <= 0:
        return None
    return float(num) / float(den)


def _embedding_nbytes(embedding: Any) -> int:
    if embedding is None:
        return 0
    try:
        return int(np.asarray(embedding).nbytes)
    except Exception:
        return 0


def summarize_descriptor_memory(memory_store) -> dict[str, Any]:
    summary = {
        "mem_active_track_count": 0,
        "mem_obj_work_count": 0,
        "mem_obj_stable_count": 0,
        "mem_obj_count": 0,
        "mem_obj_work_capacity": 0,
        "mem_obj_stable_capacity": 0,
        "mem_obj_capacity": 0,
        "mem_obj_fill_ratio": None,
        "mem_obj_bytes": 0,
        "mem_parts_work_count": 0,
        "mem_parts_stable_count": 0,
        "mem_parts_count": 0,
        "mem_parts_work_capacity": 0,
        "mem_parts_stable_capacity": 0,
        "mem_parts_capacity": 0,
        "mem_parts_fill_ratio": None,
        "mem_parts_bytes": 0,
        "mem_bg_global_work_count": 0,
        "mem_bg_global_stable_count": 0,
        "mem_bg_global_count": 0,
        "mem_bg_global_work_capacity": 0,
        "mem_bg_global_stable_capacity": 0,
        "mem_bg_global_capacity": 0,
        "mem_bg_global_fill_ratio": None,
        "mem_bg_global_bytes": 0,
        "mem_bg_partials_work_count": 0,
        "mem_bg_partials_stable_count": 0,
        "mem_bg_partials_count": 0,
        "mem_bg_partials_work_capacity": 0,
        "mem_bg_partials_stable_capacity": 0,
        "mem_bg_partials_capacity": 0,
        "mem_bg_partials_fill_ratio": None,
        "mem_bg_partials_bytes": 0,
        "mem_descriptor_count": 0,
        "mem_descriptor_capacity": 0,
        "mem_descriptor_fill_ratio": None,
        "mem_descriptor_bytes": 0,
        "mem_obj_saturated": False,
        "mem_parts_saturated": False,
        "mem_bg_global_saturated": False,
        "mem_bg_partials_saturated": False,
        "mem_descriptor_saturated": False,
    }
    if memory_store is None:
        return summary

    objects = list(getattr(memory_store, "all_objects", lambda: [])() or [])
    summary["mem_active_track_count"] = int(len(objects))

    for obj in objects:
        appearance = getattr(obj, "appearance", None)
        if appearance is not None:
            default_work_cap = getattr(appearance, "max_prototypes_per_channel", None)
            default_stable_cap = getattr(appearance, "max_stable_prototypes_per_channel", None)
            for ch_name in list(getattr(appearance, "channel_names", lambda: [])() or []):
                ch = appearance.get_channel(ch_name)
                if ch is None:
                    continue
                work = list(getattr(ch, "work_protos", []) or [])
                stable = list(getattr(ch, "stable_protos", []) or [])
                work_cap = getattr(ch, "max_prototypes", default_work_cap)
                stable_cap = getattr(ch, "max_stable", default_stable_cap)
                summary["mem_obj_work_count"] += int(len(work))
                summary["mem_obj_stable_count"] += int(len(stable))
                summary["mem_obj_work_capacity"] += 0 if work_cap is None else int(work_cap)
                summary["mem_obj_stable_capacity"] += 0 if stable_cap is None else int(stable_cap)
                summary["mem_obj_bytes"] += int(sum(_embedding_nbytes(getattr(p, "embedding", None)) for p in work))
                summary["mem_obj_bytes"] += int(sum(_embedding_nbytes(getattr(p, "embedding", None)) for p in stable))

        parts = getattr(obj, "parts", None)
        if parts is not None:
            default_parts_cap = getattr(parts, "max_prototypes_per_channel", None)
            for ch_name in list(getattr(parts, "channel_names", lambda: [])() or []):
                ch = parts.get_channel(ch_name)
                if ch is None:
                    continue
                work = list(getattr(ch, "work_protos", []) or [])
                stable = list(getattr(ch, "stable_protos", []) or [])
                cap = getattr(ch, "max_prototypes", default_parts_cap)
                summary["mem_parts_work_count"] += int(len(work))
                summary["mem_parts_stable_count"] += int(len(stable))
                summary["mem_parts_work_capacity"] += 0 if cap is None else int(cap)
                summary["mem_parts_stable_capacity"] += 0 if cap is None else int(cap)
                summary["mem_parts_bytes"] += int(sum(_embedding_nbytes(getattr(p, "embedding", None)) for p in work))
                summary["mem_parts_bytes"] += int(sum(_embedding_nbytes(getattr(p, "embedding", None)) for p in stable))

        bg = getattr(obj, "background", None)
        if bg is not None:
            bg_work_banks = [
                ("mem_bg_global_work_count", "mem_bg_global_work_capacity", "mem_bg_global_bytes", getattr(bg, "inner_global_work", None)),
                ("mem_bg_global_work_count", "mem_bg_global_work_capacity", "mem_bg_global_bytes", getattr(bg, "outer_global_work", None)),
                ("mem_bg_partials_work_count", "mem_bg_partials_work_capacity", "mem_bg_partials_bytes", getattr(bg, "inner_partials_work", None)),
                ("mem_bg_partials_work_count", "mem_bg_partials_work_capacity", "mem_bg_partials_bytes", getattr(bg, "outer_partials_work", None)),
            ]
            bg_stable_banks = [
                ("mem_bg_global_stable_count", "mem_bg_global_stable_capacity", "mem_bg_global_bytes", getattr(bg, "inner_global_stable", None)),
                ("mem_bg_global_stable_count", "mem_bg_global_stable_capacity", "mem_bg_global_bytes", getattr(bg, "outer_global_stable", None)),
                ("mem_bg_partials_stable_count", "mem_bg_partials_stable_capacity", "mem_bg_partials_bytes", getattr(bg, "inner_partials_stable", None)),
                ("mem_bg_partials_stable_count", "mem_bg_partials_stable_capacity", "mem_bg_partials_bytes", getattr(bg, "outer_partials_stable", None)),
            ]
            for count_key, cap_key, bytes_key, bank in bg_work_banks + bg_stable_banks:
                if bank is None:
                    continue
                protos = list(getattr(bank, "prototypes", []) or [])
                summary[count_key] += int(len(protos))
                summary[cap_key] += int(getattr(bank, "max_size", 0) or 0)
                summary[bytes_key] += int(sum(_embedding_nbytes(getattr(p, "embedding", None)) for p in protos))

    summary["mem_obj_count"] = int(summary["mem_obj_work_count"] + summary["mem_obj_stable_count"])
    summary["mem_obj_capacity"] = int(summary["mem_obj_work_capacity"] + summary["mem_obj_stable_capacity"])
    summary["mem_parts_count"] = int(summary["mem_parts_work_count"] + summary["mem_parts_stable_count"])
    summary["mem_parts_capacity"] = int(summary["mem_parts_work_capacity"] + summary["mem_parts_stable_capacity"])
    summary["mem_bg_global_count"] = int(summary["mem_bg_global_work_count"] + summary["mem_bg_global_stable_count"])
    summary["mem_bg_global_capacity"] = int(summary["mem_bg_global_work_capacity"] + summary["mem_bg_global_stable_capacity"])
    summary["mem_bg_partials_count"] = int(summary["mem_bg_partials_work_count"] + summary["mem_bg_partials_stable_count"])
    summary["mem_bg_partials_capacity"] = int(summary["mem_bg_partials_work_capacity"] + summary["mem_bg_partials_stable_capacity"])
    summary["mem_descriptor_count"] = int(
        summary["mem_obj_count"]
        + summary["mem_parts_count"]
        + summary["mem_bg_global_count"]
        + summary["mem_bg_partials_count"]
    )
    summary["mem_descriptor_capacity"] = int(
        summary["mem_obj_capacity"]
        + summary["mem_parts_capacity"]
        + summary["mem_bg_global_capacity"]
        + summary["mem_bg_partials_capacity"]
    )
    summary["mem_descriptor_bytes"] = int(
        summary["mem_obj_bytes"]
        + summary["mem_parts_bytes"]
        + summary["mem_bg_global_bytes"]
        + summary["mem_bg_partials_bytes"]
    )

    for prefix in ["mem_obj", "mem_parts", "mem_bg_global", "mem_bg_partials", "mem_descriptor"]:
        count = int(summary.get(f"{prefix}_count", 0) or 0)
        capacity = int(summary.get(f"{prefix}_capacity", 0) or 0)
        summary[f"{prefix}_fill_ratio"] = (float(count) / float(capacity)) if capacity > 0 else None
        summary[f"{prefix}_saturated"] = bool(capacity > 0 and count >= capacity)

    return summary


def summarize_proto_events(proto_events: list[dict[str, Any]] | None) -> dict[str, Any]:
    summary = {
        "mem_evt_total_count": 0,
        "mem_evt_obj_count": 0,
        "mem_evt_parts_count": 0,
        "mem_evt_bg_count": 0,
        "mem_evt_insert_count": 0,
        "mem_evt_merge_insert_count": 0,
        "mem_evt_evict_insert_count": 0,
        "mem_evt_dup_count": 0,
        "mem_evt_stable_count": 0,
        "mem_evt_promote_count": 0,
        "mem_evt_skip_count": 0,
    }
    for event in list(proto_events or []):
        if not isinstance(event, dict):
            continue
        summary["mem_evt_total_count"] += 1
        kind = str(event.get("kind", "") or "").strip().lower()
        action = str(event.get("action", "") or "").strip().upper()
        if kind == "obj":
            summary["mem_evt_obj_count"] += 1
        elif kind == "parts":
            summary["mem_evt_parts_count"] += 1
        elif kind == "bg":
            summary["mem_evt_bg_count"] += 1

        if action == "INSERT":
            summary["mem_evt_insert_count"] += 1
        if action == "MERGE_INSERT":
            summary["mem_evt_merge_insert_count"] += 1
        if action == "EVICT_INSERT":
            summary["mem_evt_evict_insert_count"] += 1
        if action.startswith("DUP_"):
            summary["mem_evt_dup_count"] += 1
        if action.startswith("STABLE_"):
            summary["mem_evt_stable_count"] += 1
        if action.startswith("PROMOTE_"):
            summary["mem_evt_promote_count"] += 1
        if action.startswith("SKIP_"):
            summary["mem_evt_skip_count"] += 1
    return summary


def build_memory_summary(per_frame_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(
        [dict(row) for row in (per_frame_rows or []) if row.get("frame_id", None) is not None],
        key=lambda row: int(row.get("frame_id", -1)),
    )
    if not rows:
        return {}

    summary: dict[str, Any] = {}
    numeric_fields = [
        "mem_active_track_count",
        "mem_obj_count",
        "mem_obj_capacity",
        "mem_obj_fill_ratio",
        "mem_obj_bytes",
        "mem_parts_count",
        "mem_parts_capacity",
        "mem_parts_fill_ratio",
        "mem_parts_bytes",
        "mem_bg_global_count",
        "mem_bg_global_capacity",
        "mem_bg_global_fill_ratio",
        "mem_bg_global_bytes",
        "mem_bg_partials_count",
        "mem_bg_partials_capacity",
        "mem_bg_partials_fill_ratio",
        "mem_bg_partials_bytes",
        "mem_descriptor_count",
        "mem_descriptor_capacity",
        "mem_descriptor_fill_ratio",
        "mem_descriptor_bytes",
        "mem_process_rss_after_eval_bytes",
        "mem_process_rss_peak_approx_bytes",
        "mem_process_rss_delta_bytes",
        "mem_gpu_peak_allocated_bytes",
        "mem_gpu_peak_reserved_bytes",
    ]
    for field in numeric_fields:
        values = []
        last_value = None
        for row in rows:
            raw = row.get(field, None)
            if raw is None:
                continue
            val = float(raw)
            values.append(val)
            last_value = val
        if not values:
            continue
        summary[f"{field}_mean"] = float(sum(values) / float(len(values)))
        summary[f"{field}_max"] = float(max(values))
        summary[f"{field}_final"] = float(last_value) if last_value is not None else None

    saturation_fields = [
        "mem_obj_saturated",
        "mem_parts_saturated",
        "mem_bg_global_saturated",
        "mem_bg_partials_saturated",
        "mem_descriptor_saturated",
    ]
    for field in saturation_fields:
        bool_values = [bool(row.get(field, False)) for row in rows]
        first_frame = next((int(row["frame_id"]) for row in rows if bool(row.get(field, False))), None)
        summary[f"{field}_first_frame"] = first_frame
        summary[f"{field}_frame_fraction"] = (
            float(sum(1 for value in bool_values if value)) / float(len(bool_values))
            if bool_values
            else None
        )

    event_fields = [
        "mem_evt_total_count",
        "mem_evt_obj_count",
        "mem_evt_parts_count",
        "mem_evt_bg_count",
        "mem_evt_insert_count",
        "mem_evt_merge_insert_count",
        "mem_evt_evict_insert_count",
        "mem_evt_dup_count",
        "mem_evt_stable_count",
        "mem_evt_promote_count",
        "mem_evt_skip_count",
    ]
    n_frames = int(len(rows))
    for field in event_fields:
        total = int(sum(int(row.get(field, 0) or 0) for row in rows))
        base = str(field[:-6]) if str(field).endswith("_count") else str(field)
        summary[f"{base}_total"] = int(total)
        summary[f"{base}_rate_per_frame"] = (float(total) / float(n_frames)) if n_frames > 0 else None

    return summary


def normalize_class_name(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum(dtype=np.int64)
    union = np.logical_or(a, b).sum(dtype=np.int64)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def normalize_bbox_xyxy(bbox) -> tuple[int, int, int, int] | None:
    if bbox is None or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    x1_i = int(np.floor(float(x1)))
    y1_i = int(np.floor(float(y1)))
    x2_i = int(np.ceil(float(x2)))
    y2_i = int(np.ceil(float(y2)))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return (x1_i, y1_i, x2_i, y2_i)


def bbox_intersection_xyxy(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if a is None or b is None:
        return None
    x1 = max(int(a[0]), int(b[0]))
    y1 = max(int(a[1]), int(b[1]))
    x2 = min(int(a[2]), int(b[2]))
    y2 = min(int(a[3]), int(b[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def mask_bbox_area(mask: np.ndarray) -> tuple[tuple[int, int, int, int] | None, int]:
    ys, xs = np.nonzero(mask)
    area = int(ys.size)
    if area <= 0:
        return None, 0
    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    ), area


def cropped_mask_iou(
    det_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    det_area: int,
    gt_area: int,
    gt_bbox: tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
) -> float:
    x1, y1, x2, y2 = crop_bbox
    det_crop = det_mask[y1:y2, x1:x2]
    gx1, gy1, _gx2, _gy2 = gt_bbox
    gt_crop = gt_mask[(y1 - gy1):(y2 - gy1), (x1 - gx1):(x2 - gx1)]
    inter = np.logical_and(det_crop, gt_crop).sum(dtype=np.int64)
    if inter <= 0:
        return 0.0
    union = int(det_area) + int(gt_area) - int(inter)
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def match_detections_to_gt(detections: list, gt_objects: dict[int, Any]) -> dict[int, tuple[int, float]]:
    """
    Match det_id -> (gt_instance_id, iou) with 1-to-1 IoU-maximizing matching.

    If class is available in detection and GT, matching is restricted to
    compatible pairs. Candidates are considered only when IoU > 0.
    """
    pairs: list[tuple[float, int, int]] = []
    gt_packs = []
    for gt_id, gt_obj in (gt_objects or {}).items():
        gt_mask = getattr(gt_obj, "mask", None)
        if gt_mask is None:
            continue
        gt_mask_b = gt_mask.astype(bool, copy=False)
        gt_area = int(getattr(gt_obj, "area", 0) or 0)
        gt_bbox = getattr(gt_obj, "bbox_xyxy", None)
        if gt_area <= 0 or gt_bbox is None:
            gt_bbox, gt_area = mask_bbox_area(gt_mask_b)
        if gt_area <= 0 or gt_bbox is None:
            continue
        gt_packs.append(
            {
                "gt_id": int(gt_id),
                "class_name": normalize_class_name(getattr(gt_obj, "class_name", None)),
                "mask": gt_mask_b,
                "area": int(gt_area),
                "bbox": gt_bbox,
            }
        )

    for det in detections or []:
        det_id = getattr(det, "detection_id", None)
        det_mask = getattr(det, "mask", None)
        if det_id is None or det_mask is None:
            continue

        det_mask_b = det_mask.astype(bool, copy=False)
        det_bbox = normalize_bbox_xyxy(getattr(det, "bbox", None))
        det_area = int(getattr(det, "geom", {}).get("area", 0.0) or 0.0) if isinstance(getattr(det, "geom", None), dict) else 0
        if det_bbox is None or det_area <= 0:
            det_bbox, det_area = mask_bbox_area(det_mask_b)
        if det_bbox is None or det_area <= 0:
            continue

        # Match against the active evaluation variant class namespace.
        # `original_class_name` is kept for filtering/diagnostics, but using it
        # here breaks benchmark evaluation because detections may carry raw
        # aliases such as `power_socket` while GT is labeled as `socket`.
        det_class_name = normalize_class_name(
            getattr(det, "class_name", None) or getattr(det, "original_class_name", None)
        )
        for gt_pack in gt_packs:
            gt_class_name = gt_pack["class_name"]
            if det_class_name is not None and gt_class_name is not None and det_class_name != gt_class_name:
                continue
            crop_bbox = bbox_intersection_xyxy(det_bbox, gt_pack["bbox"])
            if crop_bbox is None:
                continue
            iou = cropped_mask_iou(
                det_mask_b,
                gt_pack["mask"],
                det_area=int(det_area),
                gt_area=int(gt_pack["area"]),
                gt_bbox=gt_pack["bbox"],
                crop_bbox=crop_bbox,
            )
            if iou > 0.0:
                pairs.append((float(iou), int(det_id), int(gt_pack["gt_id"])))

    if not pairs:
        return {}

    det_ids = sorted({int(det_id) for _, det_id, _ in pairs})
    gt_ids = sorted({int(gt_id) for _, _, gt_id in pairs})
    if not det_ids or not gt_ids:
        return {}

    det_idx = {int(det_id): idx for idx, det_id in enumerate(det_ids)}
    gt_idx = {int(gt_id): idx for idx, gt_id in enumerate(gt_ids)}
    score = np.zeros((len(det_ids), len(gt_ids)), dtype=np.float64)
    for iou, det_id, gt_id in pairs:
        didx = int(det_idx[int(det_id)])
        gidx = int(gt_idx[int(gt_id)])
        score[didx, gidx] = max(float(score[didx, gidx]), float(iou))

    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(-score)
        out: dict[int, tuple[int, float]] = {}
        for row_idx, col_idx in zip(row_ind.tolist(), col_ind.tolist()):
            iou = float(score[int(row_idx), int(col_idx)])
            if iou <= 0.0:
                continue
            det_id = int(det_ids[int(row_idx)])
            gt_id = int(gt_ids[int(col_idx)])
            out[int(det_id)] = (int(gt_id), float(iou))
        return out
    except Exception:
        pairs.sort(key=lambda x: x[0], reverse=True)
        out: dict[int, tuple[int, float]] = {}
        used_dets: set[int] = set()
        used_gt: set[int] = set()
        for iou, det_id, gt_id in pairs:
            if det_id in used_dets or gt_id in used_gt:
                continue
            used_dets.add(int(det_id))
            used_gt.add(int(gt_id))
            out[int(det_id)] = (int(gt_id), float(iou))
        return out


def sort_candidate_ids(candidate_ids: list[int], candidate_scores: dict[int, float] | None = None) -> list[int]:
    scores = {int(k): float(v) for k, v in ((candidate_scores or {}).items()) if k is not None and v is not None}
    seen: set[int] = set()
    uniq_ids: list[int] = []
    for raw_oid in candidate_ids or []:
        oid = int(raw_oid)
        if oid in seen:
            continue
        seen.add(int(oid))
        uniq_ids.append(int(oid))
    return sorted(
        uniq_ids,
        key=lambda oid: (-float(scores.get(int(oid), 0.0)), int(oid)),
    )


def best_candidate_by_key(candidates: list[dict], key: str) -> dict | None:
    best = None
    best_score = -1e18
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        score = float(candidate.get(key, 0.0) or 0.0)
        if best is None or score > best_score:
            best = candidate
            best_score = score
    return best


def second_best_score(candidates: list[dict], key: str, best_object_id: int | None) -> float | None:
    second = None
    best_oid = None if best_object_id is None else int(best_object_id)
    for candidate in candidates or []:
        if not isinstance(candidate, dict):
            continue
        oid = candidate.get("object_id", None)
        if oid is None:
            continue
        if best_oid is not None and int(oid) == int(best_oid):
            continue
        score = float(candidate.get(key, 0.0) or 0.0)
        if second is None or score > second:
            second = float(score)
    return None if second is None else float(second)


def infer_frame_area_px(*, frame_shape: tuple[int, int] | None = None, detections: list | None = None) -> int:
    if isinstance(frame_shape, (tuple, list)) and len(frame_shape) == 2:
        h = int(frame_shape[0])
        w = int(frame_shape[1])
        if h > 0 and w > 0:
            return int(h * w)

    for det in (detections or []):
        mask = getattr(det, "mask", None)
        if mask is None or getattr(mask, "shape", None) is None or len(mask.shape) < 2:
            continue
        h = int(mask.shape[0])
        w = int(mask.shape[1])
        if h > 0 and w > 0:
            return int(h * w)

    return 0


class TrackingEvaluator:
    def __init__(self, stable_min_frames: int = 3, config: dict | None = None):
        self.stable_min_frames = max(1, int(stable_min_frames))
        self.config = config or {}
        self.collapse_mode = self.resolve_collapse_mode(self.config)

        self.records: list[AssignmentRecord] = []
        self.case_records: list[DecisionCaseRecord] = []
        self.frames_seen: list[int] = []
        self.frame_telemetry_by_id: dict[int, dict[str, Any]] = {}
        self.frame_collapsed_state: dict[int, dict[str, Any]] = {}
        self.resolved_collapsed_by_det: dict[tuple[int, int], tuple[str, int | None]] = {}

        self.gt_meta: dict[int, dict[str, Any]] = {}
        self.pred_meta: dict[int, dict[str, Any]] = {}

    @staticmethod
    def resolve_collapse_mode(config: dict | None) -> str:
        testing_cfg = (config or {}).get("testing", {}) or {}
        collapse_cfg = (testing_cfg.get("collapse", {}) or {})
        mode = str(collapse_cfg.get("method", "hungarian") or "hungarian").strip().lower()
        if mode not in {"greedy", "hungarian"}:
            return "hungarian"
        return mode

    def ingest_frame(
        self,
        frame_id: int,
        detections: list,
        gt_objects: dict[int, Any],
        det_to_object_id: dict[int, int],
        memory_store,
        association_output=None,
        update_output=None,
        frame_shape: tuple[int, int] | None = None,
        frame_telemetry: dict[str, Any] | None = None,
    ) -> None:
        self.frames_seen.append(int(frame_id))
        frame_memory = summarize_descriptor_memory(memory_store)
        frame_events = summarize_proto_events(getattr(update_output, "proto_events", None) or [])
        merged_telemetry = dict(frame_memory)
        merged_telemetry.update(frame_events)
        for key, value in dict(frame_telemetry or {}).items():
            merged_telemetry[str(key)] = value
        self.frame_telemetry_by_id[int(frame_id)] = merged_telemetry
        det_to_gt = match_detections_to_gt(detections=detections, gt_objects=gt_objects)
        gt_to_det = {
            int(gt_id): int(det_id)
            for det_id, (gt_id, _) in (det_to_gt or {}).items()
        }
        frame_gt_ids: set[int] = set(int(gt_id) for gt_id in (gt_objects or {}).keys())
        frame_pred_to_gt_ids: dict[int, set[int]] = defaultdict(set)
        frame_area_px = int(infer_frame_area_px(frame_shape=frame_shape, detections=detections))
        gt_class_counts = Counter(
            str(getattr(gt_obj, "class_name", None) or "unknown")
            for gt_obj in (gt_objects or {}).values()
        )

        for gt_id, gt_obj in (gt_objects or {}).items():
            meta = self.gt_meta.setdefault(
                int(gt_id),
                {
                    "label": str(getattr(gt_obj, "label", f"instance_{gt_id}")),
                    "class_name": getattr(gt_obj, "class_name", None),
                    "first_frame": int(frame_id),
                },
            )
            meta["first_frame"] = min(int(meta.get("first_frame", frame_id)), int(frame_id))

        reports_by_det_id = dict(getattr(association_output, "reports_by_det_id", {}) or {})
        assoc_debug = dict(getattr(association_output, "debug", {}) or {})
        known_set_debug = dict(assoc_debug.get("known_set_distance_disambiguation", {}) or {})
        distance_input_det_ids = {
            int(det_id)
            for item in (known_set_debug.get("passes", []) or [])
            if isinstance(item, dict)
            for det_id in (item.get("input_det_ids", []) or [])
            if det_id is not None
        }
        created_by_det_id = {
            int(row["det_id"]): dict(row)
            for row in (getattr(update_output, "created", []) or [])
            if isinstance(row, dict) and "det_id" in row
        }
        match_by_det_id = {
            int(row["det_id"]): dict(row)
            for row in (getattr(update_output, "matches", []) or [])
            if isinstance(row, dict) and "det_id" in row
        }
        provisional_by_det_id = {
            int(row["det_id"]): dict(row)
            for row in (getattr(update_output, "provisional", []) or [])
            if isinstance(row, dict) and "det_id" in row
        }
        frame_resolution_specs: list[dict[str, Any]] = []

        for det in detections or []:
            det_id = getattr(det, "detection_id", None)
            if det_id is None:
                continue

            pred_object_id = det_to_object_id.get(int(det_id), None)
            obj = memory_store.get(int(pred_object_id)) if (memory_store is not None and pred_object_id is not None) else None

            if pred_object_id is not None:
                self.pred_meta.setdefault(
                    int(pred_object_id),
                    {
                        "instance_label": getattr(obj, "instance_label", None),
                        "class_name": getattr(obj, "class_name", None),
                        "first_frame": int(frame_id),
                    },
                )

            report = reports_by_det_id.get(int(det_id), None)
            created_row = created_by_det_id.get(int(det_id), {}) or {}
            match_row = match_by_det_id.get(int(det_id), {}) or {}
            provisional_row = provisional_by_det_id.get(int(det_id), {}) or {}
            gt_id_for_det = int(det_to_gt[int(det_id)][0]) if int(det_id) in det_to_gt else None
            gt_obj_for_det = (gt_objects or {}).get(int(gt_id_for_det), None) if gt_id_for_det is not None else None

            final_decision = str(getattr(report, "final_decision", "") or "")
            final_reason = str(getattr(report, "final_reason", "") or "")
            if not final_decision:
                if int(det_id) in created_by_det_id:
                    final_decision = "NEW"
                    final_reason = "CREATED_NEW"
                elif int(det_id) in provisional_by_det_id:
                    provisional_knowns = list(provisional_row.get("related_known_ids", []) or [])
                    final_decision = "PROVISIONAL_PARENT" if provisional_knowns else "PROVISIONAL_NEW"
                    final_reason = "INFERRED_FROM_UPDATE"
                elif pred_object_id is not None:
                    final_decision = "MATCH"
                    final_reason = "INFERRED_FROM_UPDATE"
                else:
                    final_decision = "UNASSIGNED"
                    final_reason = "NO_ASSIGNMENT"

            candidates = [
                dict(candidate)
                for candidate in (getattr(report, "candidates", None) or [])
                if isinstance(candidate, dict)
            ]
            best_sim_candidate = best_candidate_by_key(candidates, key="score_sim")
            best_final_candidate = best_candidate_by_key(candidates, key="score_final")
            best_sim_object_id = None if best_sim_candidate is None else int(best_sim_candidate.get("object_id", -1))
            if best_sim_object_id is not None and best_sim_object_id < 0:
                best_sim_object_id = None
            best_final_candidate_object_id = (
                None if best_final_candidate is None else int(best_final_candidate.get("object_id", -1))
            )
            if best_final_candidate_object_id is not None and best_final_candidate_object_id < 0:
                best_final_candidate_object_id = None
            best_sim_score = 0.0 if best_sim_candidate is None else float(best_sim_candidate.get("score_sim", 0.0) or 0.0)
            best_final_candidate_score = (
                0.0 if best_final_candidate is None else float(best_final_candidate.get("score_final", 0.0) or 0.0)
            )
            best_sim_second = second_best_score(candidates, key="score_sim", best_object_id=best_sim_object_id)
            best_final_second = second_best_score(candidates, key="score_final", best_object_id=best_final_candidate_object_id)
            best_sim_margin = None if best_sim_second is None else float(best_sim_score - float(best_sim_second))
            best_final_margin = (
                None if best_final_second is None else float(best_final_candidate_score - float(best_final_second))
            )
            context_veto_candidate_count = 0
            context_rescue_applied = False
            for candidate in candidates:
                policy = dict(candidate.get("policy", {}) or {})
                veto_reason = str(policy.get("veto_reason", "") or "")
                gate_reason = str(policy.get("gate_reason", "") or "")
                if veto_reason:
                    context_veto_candidate_count += 1
                if gate_reason == "SETS_RESCUE":
                    if best_final_candidate_object_id is not None and int(candidate.get("object_id", -1)) == int(best_final_candidate_object_id):
                        context_rescue_applied = True
            match_source = str(match_row.get("source", "association") or "association")
            distance_resolved = bool(str(match_source).startswith("distance_"))
            distance_used = bool(distance_resolved or int(det_id) in distance_input_det_ids)
            neighbor_sets_available = bool(isinstance(getattr(association_output, "neighbor_sets_out", None), dict))
            context_intervened = bool(
                neighbor_sets_available
                and best_sim_object_id is not None
                and best_final_candidate_object_id is not None
                and int(best_sim_object_id) != int(best_final_candidate_object_id)
            )

            ambiguous_candidate_ids = []
            if report is not None:
                ambiguous_candidate_ids = sort_candidate_ids(
                    list(getattr(report, "ambiguous_candidate_ids", []) or []),
                    dict(getattr(report, "ambiguous_candidate_scores", {}) or {}),
                )
            ambiguous_candidate_scores = {
                int(k): float(v)
                for k, v in ((getattr(report, "ambiguous_candidate_scores", {}) or {}).items())
                if k is not None and v is not None
            }

            provisional_parent_ids: list[int] = []
            if report is not None:
                provisional_parent_ids = sort_candidate_ids(
                    list(
                        getattr(report, "provisional_related_known_ids", None)
                        or getattr(report, "provisional_support_ids", None)
                        or []
                    ),
                    dict(
                        getattr(report, "provisional_related_known_scores", None)
                        or getattr(report, "provisional_support_scores", None)
                        or {}
                    ),
                )
            provisional_parent_scores = {
                int(k): float(v)
                for k, v in (
                    (
                        getattr(report, "provisional_related_known_scores", None)
                        or getattr(report, "provisional_support_scores", None)
                        or {}
                    ).items()
                    if report is not None
                    else {}
                )
                if k is not None and v is not None
            }
            if not provisional_parent_ids:
                provisional_parent_ids = sort_candidate_ids(
                    list(provisional_row.get("related_known_ids", provisional_row.get("support_known_ids", [])) or []),
                    dict(provisional_row.get("related_known_scores", provisional_row.get("support_known_scores", {})) or {}),
                )
            if not provisional_parent_scores:
                provisional_parent_scores = {
                    int(k): float(v)
                    for k, v in (
                        (
                            provisional_row.get("related_known_scores", provisional_row.get("support_known_scores", {}))
                            or {}
                        ).items()
                    )
                    if k is not None and v is not None
                }

            provisional_temp_id = (
                int(provisional_row.get("temp_id"))
                if provisional_row.get("temp_id", None) is not None
                else None
            )
            created_object_id = (
                int(created_row.get("object_id"))
                if created_row.get("object_id", None) is not None
                else None
            )
            created_origin_provisional_temp_id = (
                int(created_row.get("origin_provisional_temp_id"))
                if created_row.get("origin_provisional_temp_id", None) is not None
                else None
            )
            frame_resolution_specs.append(
                {
                    "det_id": int(det_id),
                    "final_decision": str(final_decision),
                    "iou": float(det_to_gt[int(det_id)][1]) if int(det_id) in det_to_gt else None,
                    "options": self.build_resolution_options(
                        final_decision=final_decision,
                        firm_pred_object_id=None if pred_object_id is None else int(pred_object_id),
                        ambiguous_candidate_ids=[int(x) for x in ambiguous_candidate_ids],
                        ambiguous_candidate_scores=ambiguous_candidate_scores,
                        provisional_parent_ids=[int(x) for x in provisional_parent_ids],
                        provisional_parent_scores=provisional_parent_scores,
                        provisional_temp_id=provisional_temp_id,
                        created_object_id=created_object_id,
                        created_origin_provisional_temp_id=created_origin_provisional_temp_id,
                    ),
                }
            )

            if int(det_id) not in det_to_gt:
                continue

            gt_id, iou = det_to_gt[int(det_id)]
            gt_meta = self.gt_meta[int(gt_id)]

            if pred_object_id is not None:
                rec = AssignmentRecord(
                    frame_id=int(frame_id),
                    gt_instance_id=int(gt_id),
                    gt_label=str(gt_meta["label"]),
                    gt_class_name=gt_meta["class_name"],
                    pred_object_id=int(pred_object_id),
                    pred_instance_label=getattr(obj, "instance_label", None),
                    pred_class_name=getattr(obj, "class_name", None),
                    det_id=int(det_id),
                    iou=float(iou),
                )
                self.records.append(rec)

            case = DecisionCaseRecord(
                frame_id=int(frame_id),
                gt_instance_id=int(gt_id),
                gt_label=str(gt_meta["label"]),
                gt_class_name=gt_meta["class_name"],
                det_id=int(det_id),
                iou=float(iou),
                real_is_new=(int(frame_id) == int(gt_meta.get("first_frame", frame_id))),
                final_decision=str(final_decision),
                final_reason=str(final_reason),
                firm_pred_object_id=None if pred_object_id is None else int(pred_object_id),
                ambiguous_candidate_ids=[int(x) for x in ambiguous_candidate_ids],
                provisional_parent_ids=[int(x) for x in provisional_parent_ids],
                provisional_temp_id=provisional_temp_id,
                created_object_id=created_object_id,
                created_origin_provisional_temp_id=created_origin_provisional_temp_id,
                created_origin_parent_ids=[
                    int(x) for x in (created_row.get("origin_parent_object_ids", []) or [])
                ],
                gt_area_px=int(getattr(gt_obj_for_det, "area", 0) or 0),
                frame_gt_count=int(len(gt_objects or {})),
                frame_same_class_count=int(gt_class_counts.get(str(gt_meta["class_name"] or "unknown"), 0)),
                frame_area_px=int(frame_area_px),
                best_sim_object_id=best_sim_object_id,
                best_sim_score=float(best_sim_score),
                best_final_candidate_object_id=best_final_candidate_object_id,
                best_final_candidate_score=float(best_final_candidate_score),
                best_sim_margin=best_sim_margin,
                best_final_margin=best_final_margin,
                match_source=str(match_source),
                distance_used=bool(distance_used),
                distance_resolved=bool(distance_resolved),
                neighbor_sets_available=bool(neighbor_sets_available),
                context_intervened=bool(context_intervened),
                context_rescue_applied=bool(context_rescue_applied),
                context_veto_candidate_count=int(context_veto_candidate_count),
                selected_candidate_score_sets=0.0 if best_final_candidate is None else float(best_final_candidate.get("score_sets", 0.0) or 0.0),
                selected_candidate_quality_sets=0.0 if best_final_candidate is None else float(best_final_candidate.get("quality_sets", 0.0) or 0.0),
            )
            self.case_records.append(case)

        resolved_by_det = self.resolve_frame_collapsed_outputs(frame_resolution_specs)
        for det_id, resolved in resolved_by_det.items():
            self.resolved_collapsed_by_det[(int(frame_id), int(det_id))] = resolved
            _resolved_kind, resolved_pred_object_id = resolved
            if resolved_pred_object_id is not None:
                if int(det_id) in det_to_gt:
                    gt_id_for_pred, _ = det_to_gt[int(det_id)]
                    frame_pred_to_gt_ids[int(resolved_pred_object_id)].add(int(gt_id_for_pred))
                else:
                    frame_pred_to_gt_ids.setdefault(int(resolved_pred_object_id), set())

        self.frame_collapsed_state[int(frame_id)] = {
            "gt_ids": sorted(int(x) for x in frame_gt_ids),
            "pred_to_gt_ids": {
                int(pred_id): sorted(int(gt_id) for gt_id in gt_ids)
                for pred_id, gt_ids in frame_pred_to_gt_ids.items()
            },
        }

        missing_gt_ids = sorted(int(gt_id) for gt_id in frame_gt_ids if int(gt_id) not in gt_to_det)
        for gt_id in missing_gt_ids:
            gt_obj = (gt_objects or {}).get(int(gt_id), None)
            gt_meta = self.gt_meta[int(gt_id)]
            self.case_records.append(
                DecisionCaseRecord(
                    frame_id=int(frame_id),
                    gt_instance_id=int(gt_id),
                    gt_label=str(gt_meta["label"]),
                    gt_class_name=gt_meta["class_name"],
                    det_id=-1,
                    iou=0.0,
                    real_is_new=(int(frame_id) == int(gt_meta.get("first_frame", frame_id))),
                    final_decision="NO_DETECTION",
                    final_reason="GT_VISIBLE_WITHOUT_MATCHED_DETECTION",
                    firm_pred_object_id=None,
                    ambiguous_candidate_ids=[],
                    provisional_parent_ids=[],
                    provisional_temp_id=None,
                    created_object_id=None,
                    created_origin_provisional_temp_id=None,
                    created_origin_parent_ids=[],
                    gt_area_px=int(getattr(gt_obj, "area", 0) or 0),
                    frame_gt_count=int(len(gt_objects or {})),
                    frame_same_class_count=int(gt_class_counts.get(str(gt_meta["class_name"] or "unknown"), 0)),
                    frame_area_px=int(frame_area_px),
                )
            )

    def finalize(self) -> dict[str, Any]:
        records = sorted(self.records, key=lambda r: (int(r.frame_id), int(r.gt_instance_id)))

        by_gt: dict[int, list[AssignmentRecord]] = {}
        by_frame: dict[int, list[AssignmentRecord]] = {}

        for rec in records:
            by_gt.setdefault(int(rec.gt_instance_id), []).append(rec)
            by_frame.setdefault(int(rec.frame_id), []).append(rec)

        owner_votes_by_pred: dict[int, Counter] = defaultdict(Counter)
        for rec in records:
            owner_votes_by_pred[int(rec.pred_object_id)][int(rec.gt_instance_id)] += 1
        owner_gt_by_pred_majority: dict[int, int] = {}
        for pred_id, counter in owner_votes_by_pred.items():
            best_gt, _ = sorted(
                counter.items(),
                key=lambda item: (
                    -int(item[1]),
                    int(self.gt_meta.get(int(item[0]), {}).get("first_frame", 10**12)),
                    int(item[0]),
                ),
            )[0]
            owner_gt_by_pred_majority[int(pred_id)] = int(best_gt)
        canonical_gt_by_pred: dict[int, int] = dict(owner_gt_by_pred_majority)
        reference_pred_by_gt = self.build_global_reference_mapping(by_gt=by_gt)

        per_object_rows = []
        total_visible_frames = 0
        total_strict_correct = 0
        total_permissive_correct = 0
        perfect_objects_strict = 0
        perfect_objects_permissive = 0
        objects_with_fragmentation = 0
        objects_with_foreign_id_use = 0
        objects_recovered_reference = 0
        objects_recovered_own_identity = 0
        total_id_changes = 0
        total_stable_foreign_segments = 0
        total_stable_own_new_segments = 0

        for gt_id in sorted(by_gt):
            timeline = sorted(by_gt[gt_id], key=lambda r: int(r.frame_id))
            pred_ids = [int(r.pred_object_id) for r in timeline]
            frames = [int(r.frame_id) for r in timeline]
            ref_pred = reference_pred_by_gt.get(int(gt_id), None)

            strict_flags = [ref_pred is not None and pid == int(ref_pred) for pid in pred_ids]
            # Permissive must be a true relaxation of strict: accept the global
            # reference track or any track canonically owned by the same GT.
            permissive_flags = [
                bool(strict_flags[idx] or canonical_gt_by_pred.get(int(pid)) == int(gt_id))
                for idx, pid in enumerate(pred_ids)
            ]

            strict_correct = int(sum(1 for x in strict_flags if x))
            permissive_correct = int(sum(1 for x in permissive_flags if x))
            n_frames = int(len(timeline))

            total_visible_frames += n_frames
            total_strict_correct += strict_correct
            total_permissive_correct += permissive_correct

            if strict_correct == n_frames:
                perfect_objects_strict += 1
            if permissive_correct == n_frames:
                perfect_objects_permissive += 1

            unique_pred_ids = []
            for pid in pred_ids:
                if pid not in unique_pred_ids:
                    unique_pred_ids.append(pid)

            own_pred_ids = [pid for pid in unique_pred_ids if canonical_gt_by_pred.get(int(pid)) == int(gt_id)]
            foreign_pred_ids = [pid for pid in unique_pred_ids if canonical_gt_by_pred.get(int(pid)) != int(gt_id)]

            if len(own_pred_ids) > 1:
                objects_with_fragmentation += 1
            if len(foreign_pred_ids) > 0:
                objects_with_foreign_id_use += 1

            segments: list[SegmentRecord] = []
            start = 0
            for idx in range(1, len(pred_ids) + 1):
                boundary = idx == len(pred_ids) or pred_ids[idx] != pred_ids[idx - 1]
                if not boundary:
                    continue
                pid = int(pred_ids[start])
                owner = int(canonical_gt_by_pred.get(pid, -1))
                kind = "reference"
                if ref_pred is None or pid != int(ref_pred):
                    kind = "own_new_id" if owner == int(gt_id) else "foreign_id"
                segments.append(
                    SegmentRecord(
                        pred_object_id=pid,
                        start_frame=int(frames[start]),
                        end_frame=int(frames[idx - 1]),
                        length=int(idx - start),
                        canonical_owner_gt=owner,
                        kind=kind,
                    )
                )
                start = idx

            id_changes = max(0, len(segments) - 1)
            total_id_changes += int(id_changes)

            stable_foreign_segments = [
                s for s in segments if s.kind == "foreign_id" and int(s.length) >= self.stable_min_frames
            ]
            stable_own_new_segments = [
                s for s in segments if s.kind == "own_new_id" and int(s.length) >= self.stable_min_frames
            ]
            total_stable_foreign_segments += int(len(stable_foreign_segments))
            total_stable_own_new_segments += int(len(stable_own_new_segments))

            first_failure_idx = None
            for idx, ok in enumerate(strict_flags):
                if not ok:
                    first_failure_idx = idx
                    break

            recovered_reference = False
            recovered_own_identity = False
            post_failure_strict_acc = None
            if first_failure_idx is not None and first_failure_idx + 1 < len(timeline):
                tail_pred_ids = pred_ids[first_failure_idx + 1 :]
                tail_strict = [pid == ref_pred for pid in tail_pred_ids]
                tail_own = [canonical_gt_by_pred.get(int(pid)) == int(gt_id) for pid in tail_pred_ids]
                recovered_reference = any(tail_strict)
                recovered_own_identity = any(tail_own)
                post_failure_strict_acc = float(sum(1 for x in tail_strict if x)) / float(len(tail_strict))
            elif first_failure_idx is not None:
                post_failure_strict_acc = 0.0

            if recovered_reference:
                objects_recovered_reference += 1
            if recovered_own_identity:
                objects_recovered_own_identity += 1

            per_object_rows.append(
                {
                    "gt_instance_id": int(gt_id),
                    "gt_label": str(self.gt_meta[int(gt_id)]["label"]),
                    "gt_class_name": self.gt_meta[int(gt_id)]["class_name"],
                    "n_frames": n_frames,
                    "first_frame": int(frames[0]),
                    "last_frame": int(frames[-1]),
                    "reference_pred_id": None if ref_pred is None else int(ref_pred),
                    "reference_pred_label": None if ref_pred is None else self.pred_meta.get(int(ref_pred), {}).get("instance_label", None),
                    "strict_accuracy": (float(strict_correct) / float(n_frames)) if n_frames > 0 else 0.0,
                    "permissive_accuracy": (float(permissive_correct) / float(n_frames)) if n_frames > 0 else 0.0,
                    "perfect_strict": bool(strict_correct == n_frames),
                    "perfect_permissive": bool(permissive_correct == n_frames),
                    "n_unique_pred_ids": int(len(unique_pred_ids)),
                    "n_own_pred_ids": int(len(own_pred_ids)),
                    "n_foreign_pred_ids": int(len(foreign_pred_ids)),
                    "id_changes": int(id_changes),
                    "stable_foreign_segments": int(len(stable_foreign_segments)),
                    "stable_own_new_segments": int(len(stable_own_new_segments)),
                    "first_failure_frame": None if first_failure_idx is None else int(frames[first_failure_idx]),
                    "recovered_reference": bool(recovered_reference),
                    "recovered_own_identity": bool(recovered_own_identity),
                    "post_failure_strict_accuracy": post_failure_strict_acc,
                    "segments": [asdict(s) for s in segments],
                    "pred_ids_timeline": [int(x) for x in pred_ids],
                    "frames_timeline": [int(x) for x in frames],
                }
            )

        per_frame_rows = []
        for frame_id in sorted(by_frame):
            frame_recs = sorted(by_frame[frame_id], key=lambda r: int(r.gt_instance_id))
            n = int(len(frame_recs))
            strict_correct = 0
            permissive_correct = 0
            for rec in frame_recs:
                gt_id = int(rec.gt_instance_id)
                pid = int(rec.pred_object_id)
                ref_pred = reference_pred_by_gt.get(int(gt_id), None)
                is_strict_correct = bool(ref_pred is not None and pid == int(ref_pred))
                if is_strict_correct:
                    strict_correct += 1
                if is_strict_correct or canonical_gt_by_pred.get(pid) == gt_id:
                    permissive_correct += 1

            per_frame_rows.append(
                {
                    "frame_id": int(frame_id),
                    "n_objects": n,
                    "strict_accuracy": (float(strict_correct) / float(n)) if n > 0 else 0.0,
                    "permissive_accuracy": (float(permissive_correct) / float(n)) if n > 0 else 0.0,
                    "strict_correct": int(strict_correct),
                    "permissive_correct": int(permissive_correct),
                }
            )

        frame_maps: dict[int, dict[int, int]] = {}
        for frame_id, frame_recs in by_frame.items():
            frame_maps[int(frame_id)] = {int(r.gt_instance_id): int(r.pred_object_id) for r in frame_recs}

        swap_events = []
        theft_with_new_id_events = []
        theft_with_displacement_events = []
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

        n_objects = int(len(per_object_rows))
        summary = {
            "n_frames": int(len(sorted(by_frame))),
            "n_objects": n_objects,
            "n_assignments": int(len(records)),
            "stable_min_frames": int(self.stable_min_frames),
            "global_frame_accuracy_strict": (float(total_strict_correct) / float(total_visible_frames))
            if total_visible_frames > 0
            else 0.0,
            "global_frame_accuracy_permissive": (float(total_permissive_correct) / float(total_visible_frames))
            if total_visible_frames > 0
            else 0.0,
            "global_object_accuracy_strict": (float(perfect_objects_strict) / float(n_objects)) if n_objects > 0 else 0.0,
            "global_object_accuracy_permissive": (float(perfect_objects_permissive) / float(n_objects))
            if n_objects > 0
            else 0.0,
            "objects_fragmented": int(objects_with_fragmentation),
            "objects_with_foreign_id_use": int(objects_with_foreign_id_use),
            "objects_recovered_reference": int(objects_recovered_reference),
            "objects_recovered_own_identity": int(objects_recovered_own_identity),
            "id_changes_total": int(total_id_changes),
            "stable_foreign_segments_total": int(total_stable_foreign_segments),
            "stable_own_new_segments_total": int(total_stable_own_new_segments),
            "swap_events_total": int(len(swap_events)),
            "theft_with_new_id_total": int(len(theft_with_new_id_events)),
            "theft_with_displacement_total": int(len(theft_with_displacement_events)),
        }

        pred_rows = []
        pred_to_gts: dict[int, list[int]] = {}
        for gt_row in per_object_rows:
            gt_id = int(gt_row["gt_instance_id"])
            for pid in gt_row["pred_ids_timeline"]:
                if pid is None:
                    continue
                pred_to_gts.setdefault(int(pid), [])
                if gt_id not in pred_to_gts[int(pid)]:
                    pred_to_gts[int(pid)].append(gt_id)

        for pred_id in sorted(pred_to_gts):
            pred_records = [rec for rec in records if int(rec.pred_object_id) == int(pred_id)]
            pred_frames = sorted(int(rec.frame_id) for rec in pred_records)
            pred_rows.append(
                {
                    "pred_object_id": int(pred_id),
                    "pred_instance_label": self.pred_meta.get(int(pred_id), {}).get("instance_label", None),
                    "pred_class_name": self.pred_meta.get(int(pred_id), {}).get("class_name", None),
                    "canonical_gt": int(canonical_gt_by_pred.get(int(pred_id), -1)),
                    "majority_gt": int(owner_gt_by_pred_majority.get(int(pred_id), -1)),
                    "reference_gt": int(
                        next((int(gt_id) for gt_id, pred_id_ref in reference_pred_by_gt.items() if int(pred_id_ref) == int(pred_id)), -1)
                    ),
                    "gt_users": [int(x) for x in pred_to_gts[int(pred_id)]],
                    "n_gt_users": int(len(pred_to_gts[int(pred_id)])),
                    "first_frame": None if not pred_frames else int(pred_frames[0]),
                    "last_frame": None if not pred_frames else int(pred_frames[-1]),
                    "n_frames_present": int(len(pred_frames)),
                    "is_pure_track": bool(len(pred_to_gts[int(pred_id)]) <= 1),
                    "is_fragment_track": bool(len(pred_to_gts[int(pred_id)]) > 1),
                    "is_foreign_track": bool(
                        int(owner_gt_by_pred_majority.get(int(pred_id), -1)) != int(
                            next((int(gt_id) for gt_id, pred_id_ref in reference_pred_by_gt.items() if int(pred_id_ref) == int(pred_id)), -1)
                        )
                    ),
                }
            )

        case_rows = self.build_case_rows(owner_gt_by_pred_majority=owner_gt_by_pred_majority)
        collapsed_identity_metrics = self.compute_collapsed_identity_metrics(case_rows=case_rows)
        collapsed_metrics, uncertainty_metrics = self.compute_decision_metrics(case_rows=case_rows)
        by_gt_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_frame_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
        by_class_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        collapsed_visible_rows = []
        for row in case_rows:
            gt_id = int(row["gt_instance_id"])
            frame_id = int(row["frame_id"])
            class_name = str(row.get("gt_class_name") or "unknown")
            by_gt_case[gt_id].append(row)
            by_frame_case[frame_id].append(row)
            by_class_case[class_name].append(row)
            if (
                str(row.get("collapsed_kind")) in {"existing", "new"}
                and row.get("collapsed_pred_object_id") is not None
            ):
                collapsed_visible_rows.append(row)

        collapsed_owner_votes_by_pred: dict[int, Counter] = defaultdict(Counter)
        for row in collapsed_visible_rows:
            collapsed_owner_votes_by_pred[int(row["collapsed_pred_object_id"])][int(row["gt_instance_id"])] += 1
        collapsed_owner_gt_by_pred_majority: dict[int, int] = {}
        for pred_id, counter in collapsed_owner_votes_by_pred.items():
            best_gt, _ = sorted(
                counter.items(),
                key=lambda item: (
                    -int(item[1]),
                    int(self.gt_meta.get(int(item[0]), {}).get("first_frame", 10**12)),
                    int(item[0]),
                ),
            )[0]
            collapsed_owner_gt_by_pred_majority[int(pred_id)] = int(best_gt)
        collapsed_reference_pred_by_gt = {
            int(k): int(v)
            for k, v in ((collapsed_identity_metrics.get("reference_pred_by_gt", {}) or {}).items())
        }

        per_object_by_gt_id = {
            int(row["gt_instance_id"]): row
            for row in per_object_rows
            if row.get("gt_instance_id", None) is not None
        }
        for gt_id in sorted(int(x) for x in self.gt_meta.keys()):
            gt_cases = sorted(by_gt_case.get(int(gt_id), []), key=lambda row: int(row["frame_id"]))
            visible_n_frames = int(len(gt_cases))
            if visible_n_frames <= 0:
                continue

            row = per_object_by_gt_id.get(int(gt_id), None)
            if row is None:
                ref_pred = reference_pred_by_gt.get(int(gt_id), None)
                row = {
                    "gt_instance_id": int(gt_id),
                    "gt_label": str(self.gt_meta[int(gt_id)]["label"]),
                    "gt_class_name": self.gt_meta[int(gt_id)]["class_name"],
                    "n_frames": 0,
                    "first_frame": int(gt_cases[0]["frame_id"]),
                    "last_frame": int(gt_cases[-1]["frame_id"]),
                    "reference_pred_id": None if ref_pred is None else int(ref_pred),
                    "reference_pred_label": None if ref_pred is None else self.pred_meta.get(int(ref_pred), {}).get("instance_label", None),
                    "strict_accuracy": 0.0,
                    "permissive_accuracy": 0.0,
                    "perfect_strict": False,
                    "perfect_permissive": False,
                    "n_unique_pred_ids": 0,
                    "n_own_pred_ids": 0,
                    "n_foreign_pred_ids": 0,
                    "id_changes": 0,
                    "stable_foreign_segments": 0,
                    "stable_own_new_segments": 0,
                    "duplicate_id_segment_count": 0,
                    "foreign_id_segment_count": 0,
                    "duplicate_id_frame_count": 0,
                    "foreign_id_frame_count": 0,
                    "duplicate_id_rate_visible": 0.0,
                    "foreign_id_rate_visible": 0.0,
                    "first_failure_frame": int(gt_cases[0]["frame_id"]),
                    "recovered_reference": False,
                    "recovered_own_identity": False,
                    "recovery_attempts": 0,
                    "recovery_success_reference": 0,
                    "recovery_success_own_identity": 0,
                    "recovery_success_duplicate_id": 0,
                    "recovery_success_foreign_id": 0,
                    "recovery_rate_reference": None,
                    "recovery_rate_own_identity": None,
                    "recovery_rate_duplicate_id": None,
                    "recovery_rate_foreign_id": None,
                    "post_failure_strict_accuracy": None,
                    "segments": [],
                    "pred_ids_timeline": [],
                    "frames_timeline": [],
                }
                per_object_rows.append(row)
                per_object_by_gt_id[int(gt_id)] = row

            assigned_ref_pred = reference_pred_by_gt.get(int(gt_id), None)
            assigned_pred_timeline = [
                None if item.get("firm_pred_object_id", None) is None else int(item["firm_pred_object_id"])
                for item in gt_cases
            ]
            assigned_frames_timeline = [int(item["frame_id"]) for item in gt_cases]
            assigned_n_frames = int(sum(1 for pred_id in assigned_pred_timeline if pred_id is not None))

            strict_flags_assigned = [
                bool(assigned_ref_pred is not None and pred_id is not None and int(pred_id) == int(assigned_ref_pred))
                for pred_id in assigned_pred_timeline
            ]
            permissive_flags_assigned = [
                bool(
                    pred_id is not None
                    and (
                        strict_flags_assigned[idx]
                        or collapsed_owner_gt_by_pred_majority.get(int(pred_id), None) == int(gt_id)
                    )
                )
                for idx, pred_id in enumerate(assigned_pred_timeline)
            ]
            strict_correct_visible = int(sum(1 for flag in strict_flags_assigned if flag))
            permissive_correct_visible = int(sum(1 for flag in permissive_flags_assigned if flag))

            pred_timeline = [
                None if item.get("collapsed_pred_object_id", None) is None else int(item["collapsed_pred_object_id"])
                for item in gt_cases
            ]
            non_null_pred_timeline = [int(pid) for pid in pred_timeline if pid is not None]
            unique_pred_ids = []
            for pid in non_null_pred_timeline:
                if pid not in unique_pred_ids:
                    unique_pred_ids.append(int(pid))

            own_pred_ids = [
                int(pid)
                for pid in unique_pred_ids
                if collapsed_owner_gt_by_pred_majority.get(int(pid), None) == int(gt_id)
            ]
            foreign_pred_ids = [
                int(pid)
                for pid in unique_pred_ids
                if collapsed_owner_gt_by_pred_majority.get(int(pid), None) != int(gt_id)
            ]

            collapsed_ref_pred = collapsed_reference_pred_by_gt.get(int(gt_id), None)
            collapsed_segments: list[SegmentRecord] = []
            seg_start_idx = None
            for idx, pid in enumerate(list(pred_timeline) + [None]):
                if seg_start_idx is None:
                    if pid is not None:
                        seg_start_idx = int(idx)
                    continue

                prev_pid = pred_timeline[int(idx) - 1] if idx > 0 else None
                if idx < len(pred_timeline) and pid == prev_pid:
                    continue

                if prev_pid is not None:
                    owner = int(collapsed_owner_gt_by_pred_majority.get(int(prev_pid), -1))
                    kind = "reference"
                    if collapsed_ref_pred is None or int(prev_pid) != int(collapsed_ref_pred):
                        kind = "own_new_id" if owner == int(gt_id) else "foreign_id"
                    collapsed_segments.append(
                        SegmentRecord(
                            pred_object_id=int(prev_pid),
                            start_frame=int(gt_cases[int(seg_start_idx)]["frame_id"]),
                            end_frame=int(gt_cases[int(idx) - 1]["frame_id"]),
                            length=int(idx - int(seg_start_idx)),
                            canonical_owner_gt=owner,
                            kind=kind,
                        )
                    )
                seg_start_idx = None
                if idx < len(pred_timeline) and pid is not None:
                    seg_start_idx = int(idx)

            idsw_object = 0
            prev_pred_non_null = None
            for pid in pred_timeline:
                if pid is None:
                    continue
                if prev_pred_non_null is not None and int(pid) != int(prev_pred_non_null):
                    idsw_object += 1
                prev_pred_non_null = int(pid)

            tracked_runs = 0
            in_tracked_run = False
            for pid in pred_timeline:
                cur_tracked = pid is not None
                if cur_tracked and not in_tracked_run:
                    tracked_runs += 1
                    in_tracked_run = True
                elif not cur_tracked:
                    in_tracked_run = False
            frag_object = max(0, tracked_runs - 1)

            stable_foreign_segments = [
                segment
                for segment in collapsed_segments
                if segment.kind == "foreign_id" and int(segment.length) >= self.stable_min_frames
            ]
            stable_own_new_segments = [
                segment
                for segment in collapsed_segments
                if segment.kind == "own_new_id" and int(segment.length) >= self.stable_min_frames
            ]
            duplicate_id_segments = [
                segment
                for segment in collapsed_segments
                if segment.kind == "own_new_id"
            ]
            foreign_id_segments = [
                segment
                for segment in collapsed_segments
                if segment.kind == "foreign_id"
            ]
            id_changes = max(0, len(collapsed_segments) - 1)
            duplicate_id_frame_count = int(
                sum(
                    1
                    for pid in pred_timeline
                    if (
                        pid is not None
                        and collapsed_owner_gt_by_pred_majority.get(int(pid), None) == int(gt_id)
                        and (collapsed_ref_pred is None or int(pid) != int(collapsed_ref_pred))
                    )
                )
            )
            foreign_id_frame_count = int(
                sum(
                    1
                    for pid in pred_timeline
                    if (
                        pid is not None
                        and collapsed_owner_gt_by_pred_majority.get(int(pid), None) != int(gt_id)
                    )
                )
            )
            strict_flags_collapsed = [
                bool(collapsed_ref_pred is not None and pid is not None and int(pid) == int(collapsed_ref_pred))
                for pid in pred_timeline
            ]
            first_failure_idx = None
            for idx, ok in enumerate(strict_flags_collapsed):
                if not ok:
                    first_failure_idx = idx
                    break

            recovered_reference = False
            recovered_own_identity = False
            post_failure_strict_acc = None
            recovery_start_indices: list[int] = []
            in_tracked_run = False
            prev_visible_frame_id = None
            for idx, pid in enumerate(pred_timeline):
                frame_id_cur = int(assigned_frames_timeline[int(idx)])
                has_visibility_gap = bool(
                    prev_visible_frame_id is not None and frame_id_cur > int(prev_visible_frame_id) + 1
                )
                if has_visibility_gap:
                    in_tracked_run = False
                cur_tracked = pid is not None
                if cur_tracked and not in_tracked_run:
                    recovery_start_indices.append(int(idx))
                    in_tracked_run = True
                elif not cur_tracked:
                    in_tracked_run = False
                prev_visible_frame_id = int(frame_id_cur)
            recovery_only_start_indices = list(recovery_start_indices[1:])
            recovery_attempts = int(len(recovery_only_start_indices))
            recovery_success_reference = 0
            recovery_success_own_identity = 0
            recovery_success_duplicate_id = 0
            recovery_success_foreign_id = 0
            for idx in recovery_only_start_indices:
                pid = pred_timeline[int(idx)]
                if pid is None:
                    continue
                owner = collapsed_owner_gt_by_pred_majority.get(int(pid), None)
                is_reference = bool(collapsed_ref_pred is not None and int(pid) == int(collapsed_ref_pred))
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
                    bool(collapsed_ref_pred is not None and pid is not None and int(pid) == int(collapsed_ref_pred))
                    for pid in tail_pred_ids
                ]
                tail_own = [
                    bool(pid is not None and collapsed_owner_gt_by_pred_majority.get(int(pid), None) == int(gt_id))
                    for pid in tail_pred_ids
                ]
                recovered_reference = any(tail_strict)
                recovered_own_identity = any(tail_own)
                post_failure_strict_acc = float(sum(1 for flag in tail_strict if flag)) / float(len(tail_strict))
            elif first_failure_idx is not None:
                post_failure_strict_acc = 0.0

            tracking_correct_count = int(sum(1 for item in gt_cases if bool(item.get("collapsed_global_correct", False))))
            tracking_iou_sum = float(sum(float(item.get("tracking_iou", 0.0) or 0.0) for item in gt_cases))
            tracking_recall_object = safe_pct(tracking_correct_count, visible_n_frames)
            mean_tracking_iou_object = (
                float(tracking_iou_sum) / float(visible_n_frames) if visible_n_frames > 0 else None
            )
            if tracking_recall_object is None:
                mt_pt_ml_label = None
            elif float(tracking_recall_object) >= 0.80:
                mt_pt_ml_label = "MT"
            elif float(tracking_recall_object) <= 0.20:
                mt_pt_ml_label = "ML"
            else:
                mt_pt_ml_label = "PT"

            row["assigned_n_frames"] = int(assigned_n_frames)
            row["visible_n_frames"] = int(visible_n_frames)
            row["n_frames"] = int(visible_n_frames)
            row["reference_pred_id"] = None if assigned_ref_pred is None else int(assigned_ref_pred)
            row["reference_pred_label"] = (
                None
                if assigned_ref_pred is None
                else self.pred_meta.get(int(assigned_ref_pred), {}).get("instance_label", None)
            )
            row["strict_accuracy"] = (
                float(strict_correct_visible) / float(visible_n_frames) if visible_n_frames > 0 else 0.0
            )
            row["permissive_accuracy"] = (
                float(permissive_correct_visible) / float(visible_n_frames) if visible_n_frames > 0 else 0.0
            )
            row["perfect_strict"] = bool(strict_correct_visible == visible_n_frames)
            row["perfect_permissive"] = bool(permissive_correct_visible == visible_n_frames)
            row["tracking_recall_object"] = tracking_recall_object
            row["mean_tracking_iou_object"] = mean_tracking_iou_object
            row["mt_pt_ml_label"] = mt_pt_ml_label
            row["n_unique_pred_ids"] = int(len(unique_pred_ids))
            row["n_own_pred_ids"] = int(len(own_pred_ids))
            row["n_foreign_pred_ids"] = int(len(foreign_pred_ids))
            row["id_changes"] = int(id_changes)
            row["stable_foreign_segments"] = int(len(stable_foreign_segments))
            row["stable_own_new_segments"] = int(len(stable_own_new_segments))
            row["duplicate_id_segment_count"] = int(len(duplicate_id_segments))
            row["foreign_id_segment_count"] = int(len(foreign_id_segments))
            row["duplicate_id_frame_count"] = int(duplicate_id_frame_count)
            row["foreign_id_frame_count"] = int(foreign_id_frame_count)
            row["duplicate_id_rate_visible"] = safe_pct(duplicate_id_frame_count, visible_n_frames)
            row["foreign_id_rate_visible"] = safe_pct(foreign_id_frame_count, visible_n_frames)
            row["first_failure_frame"] = None if first_failure_idx is None else int(gt_cases[int(first_failure_idx)]["frame_id"])
            row["recovered_reference"] = bool(recovered_reference)
            row["recovered_own_identity"] = bool(recovered_own_identity)
            row["recovery_attempts"] = int(recovery_attempts)
            row["recovery_success_reference"] = int(recovery_success_reference)
            row["recovery_success_own_identity"] = int(recovery_success_own_identity)
            row["recovery_success_duplicate_id"] = int(recovery_success_duplicate_id)
            row["recovery_success_foreign_id"] = int(recovery_success_foreign_id)
            row["recovery_rate_reference"] = safe_pct(recovery_success_reference, recovery_attempts)
            row["recovery_rate_own_identity"] = safe_pct(recovery_success_own_identity, recovery_attempts)
            row["recovery_rate_duplicate_id"] = safe_pct(recovery_success_duplicate_id, recovery_attempts)
            row["recovery_rate_foreign_id"] = safe_pct(recovery_success_foreign_id, recovery_attempts)
            row["post_failure_strict_accuracy"] = post_failure_strict_acc
            row["segments"] = [asdict(segment) for segment in collapsed_segments]
            row["idsw_object"] = int(idsw_object)
            row["frag_object"] = int(frag_object)
            row["pred_ids_timeline"] = [
                None if pred_id is None else int(pred_id)
                for pred_id in assigned_pred_timeline
            ]
            row["frames_timeline"] = [int(frame_id) for frame_id in assigned_frames_timeline]
            row["mean_visible_gt_in_frame"] = float(
                np.mean([float(item.get("n_gt_visible_in_frame", 0) or 0) for item in gt_cases])
            )
            row["mean_total_distractors"] = float(
                np.mean([float(item.get("n_total_distractors", 0) or 0) for item in gt_cases])
            )
            row["mean_same_class_distractors"] = float(
                np.mean([float(item.get("n_same_class_distractors", 0) or 0) for item in gt_cases])
            )
            gt_area_fracs = [float(item["gt_area_frac"]) for item in gt_cases if item.get("gt_area_frac", None) is not None]
            row["mean_gt_area_frac"] = float(np.mean(gt_area_fracs)) if gt_area_fracs else None
            row["n_ambiguous_cases"] = int(sum(1 for item in gt_cases if str(item.get("final_decision", "")) == "AMBIGUOUS_TRACK"))
            row["n_provisional_parent_cases"] = int(sum(1 for item in gt_cases if str(item.get("final_decision", "")) == "PROVISIONAL_PARENT"))
            row["n_provisional_new_cases"] = int(sum(1 for item in gt_cases if str(item.get("final_decision", "")) == "PROVISIONAL_NEW"))
            row["distance_used_count"] = int(sum(1 for item in gt_cases if bool(item.get("distance_used", False))))
            row["distance_correct_count"] = int(sum(1 for item in gt_cases if bool(item.get("distance_correct", False))))
            row["context_intervened_count"] = int(sum(1 for item in gt_cases if bool(item.get("context_intervened", False))))
            row["context_correct_count"] = int(sum(1 for item in gt_cases if bool(item.get("context_change_correct", False))))

        per_frame_by_id = {
            int(row["frame_id"]): row
            for row in per_frame_rows
            if row.get("frame_id", None) is not None
        }
        for frame_id in sorted(int(x) for x in set(self.frames_seen) | set(by_frame_case.keys())):
            frame_cases = list(by_frame_case.get(int(frame_id), []))
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

            visible_n = int(len(frame_cases))
            strict_correct = 0
            permissive_correct = 0
            if frame_cases:
                for item in frame_cases:
                    gt_id = int(item["gt_instance_id"])
                    assigned_pred_id = item.get("firm_pred_object_id", None)
                    ref_pred = reference_pred_by_gt.get(int(gt_id), None)
                    is_strict_correct = bool(
                        assigned_pred_id is not None and ref_pred is not None and int(assigned_pred_id) == int(ref_pred)
                    )
                    if is_strict_correct:
                        strict_correct += 1
                    if assigned_pred_id is not None and (
                        is_strict_correct or canonical_gt_by_pred.get(int(assigned_pred_id), None) == int(gt_id)
                    ):
                        permissive_correct += 1

            row["n_objects"] = int(visible_n)
            row["visible_n_objects"] = int(visible_n)
            row["n_classes_visible"] = int(len({str(item.get("gt_class_name") or "unknown") for item in frame_cases}))
            row["strict_correct"] = int(strict_correct)
            row["permissive_correct"] = int(permissive_correct)
            row["strict_accuracy"] = (float(strict_correct) / float(visible_n)) if visible_n > 0 else 0.0
            row["permissive_accuracy"] = (float(permissive_correct) / float(visible_n)) if visible_n > 0 else 0.0
            row["tracking_recall_frame"] = safe_pct(
                sum(1 for item in frame_cases if bool(item.get("collapsed_global_correct", False))),
                visible_n,
            )
            row["mean_tracking_iou_frame"] = (
                float(sum(float(item.get("tracking_iou", 0.0) or 0.0) for item in frame_cases)) / float(visible_n)
                if visible_n > 0
                else None
            )
            row["n_new_gt"] = int(sum(1 for item in frame_cases if str(item.get("real_state")) == "new"))
            row["n_existing_gt"] = int(sum(1 for item in frame_cases if str(item.get("real_state")) == "existing"))
            row["n_firm"] = int(sum(1 for item in frame_cases if str(item.get("final_decision")) in {"MATCH", "NEW"}))
            row["n_ambiguous"] = int(sum(1 for item in frame_cases if str(item.get("final_decision")) == "AMBIGUOUS_TRACK"))
            row["n_provisional_parent"] = int(sum(1 for item in frame_cases if str(item.get("final_decision")) == "PROVISIONAL_PARENT"))
            row["n_provisional_new"] = int(sum(1 for item in frame_cases if str(item.get("final_decision")) == "PROVISIONAL_NEW"))
            row["n_distance_used"] = int(sum(1 for item in frame_cases if bool(item.get("distance_used", False))))
            row["n_context_interventions"] = int(sum(1 for item in frame_cases if bool(item.get("context_intervened", False))))
            row.update(self.frame_telemetry_by_id.get(int(frame_id), {}))

        n_frames_total = int(len({int(frame_id) for frame_id in self.frames_seen}))
        n_objects_total = int(len({int(gt_id) for gt_id in self.gt_meta.keys()}))
        perfect_objects_strict = int(sum(1 for row in per_object_rows if bool(row.get("perfect_strict", False))))
        perfect_objects_permissive = int(sum(1 for row in per_object_rows if bool(row.get("perfect_permissive", False))))
        objects_with_fragmentation = int(sum(1 for row in per_object_rows if int(row.get("n_own_pred_ids", 0) or 0) > 1))
        objects_with_foreign_id_use = int(sum(1 for row in per_object_rows if int(row.get("n_foreign_pred_ids", 0) or 0) > 0))
        objects_recovered_reference = int(sum(1 for row in per_object_rows if bool(row.get("recovered_reference", False))))
        objects_recovered_own_identity = int(sum(1 for row in per_object_rows if bool(row.get("recovered_own_identity", False))))
        total_id_changes = int(sum(int(row.get("id_changes", 0) or 0) for row in per_object_rows))
        total_stable_foreign_segments = int(sum(int(row.get("stable_foreign_segments", 0) or 0) for row in per_object_rows))
        total_stable_own_new_segments = int(sum(int(row.get("stable_own_new_segments", 0) or 0) for row in per_object_rows))
        total_duplicate_id_segments = int(sum(int(row.get("duplicate_id_segment_count", 0) or 0) for row in per_object_rows))
        total_foreign_id_segments = int(sum(int(row.get("foreign_id_segment_count", 0) or 0) for row in per_object_rows))
        total_duplicate_id_frames = int(sum(int(row.get("duplicate_id_frame_count", 0) or 0) for row in per_object_rows))
        total_foreign_id_frames = int(sum(int(row.get("foreign_id_frame_count", 0) or 0) for row in per_object_rows))
        total_recovery_attempts = int(sum(int(row.get("recovery_attempts", 0) or 0) for row in per_object_rows))
        total_recovery_success_reference = int(sum(int(row.get("recovery_success_reference", 0) or 0) for row in per_object_rows))
        total_recovery_success_own_identity = int(sum(int(row.get("recovery_success_own_identity", 0) or 0) for row in per_object_rows))
        total_recovery_success_duplicate_id = int(sum(int(row.get("recovery_success_duplicate_id", 0) or 0) for row in per_object_rows))
        total_recovery_success_foreign_id = int(sum(int(row.get("recovery_success_foreign_id", 0) or 0) for row in per_object_rows))
        n_mt_objects = int(sum(1 for row in per_object_rows if str(row.get("mt_pt_ml_label", "")) == "MT"))
        n_pt_objects = int(sum(1 for row in per_object_rows if str(row.get("mt_pt_ml_label", "")) == "PT"))
        n_ml_objects = int(sum(1 for row in per_object_rows if str(row.get("mt_pt_ml_label", "")) == "ML"))
        n_visible_gt_observations = int(len(case_rows))
        n_matched_gt_observations = int(
            sum(1 for row in case_rows if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0)
        )
        distance_used_count = int(sum(1 for row in case_rows if bool(row.get("distance_used", False))))
        distance_resolved_count = int(sum(1 for row in case_rows if bool(row.get("distance_resolved", False))))
        distance_correct_count = int(sum(1 for row in case_rows if bool(row.get("distance_correct", False))))
        neighbor_sets_available_count = int(sum(1 for row in case_rows if bool(row.get("neighbor_sets_available", False))))
        context_intervened_count = int(sum(1 for row in case_rows if bool(row.get("context_intervened", False))))
        context_correct_count = int(sum(1 for row in case_rows if bool(row.get("context_change_correct", False))))
        context_rescue_count = int(sum(1 for row in case_rows if bool(row.get("context_rescue_applied", False))))
        context_veto_case_count = int(sum(1 for row in case_rows if int(row.get("context_veto_candidate_count", 0) or 0) > 0))

        existing_gt_reopened_as_new_rows = [
            row
            for row in case_rows
            if str(row.get("real_state")) == "existing" and str(row.get("collapsed_kind")) == "new"
        ]
        existing_gt_reopened_as_new_ids = {
            int(row["gt_instance_id"]) for row in existing_gt_reopened_as_new_rows
        }
        n_existing_gt = int(collapsed_metrics.get("n_existing_gt", 0) or 0)
        total_visible_frame_objects = int(sum(int(row.get("n_objects", 0) or 0) for row in per_frame_rows))
        total_strict_correct_visible = int(sum(int(row.get("strict_correct", 0) or 0) for row in per_frame_rows))
        total_permissive_correct_visible = int(sum(int(row.get("permissive_correct", 0) or 0) for row in per_frame_rows))
        summary.update(
            {
                "n_frames": int(n_frames_total),
                "n_objects": int(n_objects_total),
                "n_visible_gt_observations": int(n_visible_gt_observations),
                "n_matched_gt_observations": int(n_matched_gt_observations),
                "n_unique_real_pred_tracks": int(len(pred_rows)),
                "pred_track_surplus_vs_gt": int(len(pred_rows) - n_objects_total),
                "pred_track_inflation_factor": (
                    float(len(pred_rows)) / float(n_objects_total) if n_objects_total > 0 else None
                ),
                "n_existing_gt_reopened_as_new_rows": int(len(existing_gt_reopened_as_new_rows)),
                "n_existing_gt_reopened_as_new_ids": int(len(existing_gt_reopened_as_new_ids)),
                "reopen_rate_existing": (
                    float(len(existing_gt_reopened_as_new_rows)) / float(n_existing_gt)
                    if n_existing_gt > 0
                    else None
                ),
                "gt_with_reopen_rate": (
                    float(len(existing_gt_reopened_as_new_ids)) / float(n_objects_total)
                    if n_objects_total > 0
                    else None
                ),
                "label_summary": (
                    f"{n_objects_total} GT objects -> {len(pred_rows)} real tracker labels "
                    f"({len(pred_rows) - n_objects_total:+d}, "
                    f"{(float(len(pred_rows)) / float(n_objects_total)):.3f}x)"
                    if n_objects_total > 0
                    else None
                ),
                "reopen_summary": (
                    f"{len(existing_gt_reopened_as_new_ids)} GT objects reopened as NEW "
                    f"across {len(existing_gt_reopened_as_new_rows)} rows"
                ),
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
                "duplicate_id_segment_count": int(total_duplicate_id_segments),
                "foreign_id_segment_count": int(total_foreign_id_segments),
                "duplicate_id_frame_count": int(total_duplicate_id_frames),
                "foreign_id_frame_count": int(total_foreign_id_frames),
                "duplicate_id_rate_visible": safe_pct(total_duplicate_id_frames, n_visible_gt_observations),
                "foreign_id_rate_visible": safe_pct(total_foreign_id_frames, n_visible_gt_observations),
                "recovery_attempts_total": int(total_recovery_attempts),
                "recovery_success_reference_total": int(total_recovery_success_reference),
                "recovery_success_own_identity_total": int(total_recovery_success_own_identity),
                "recovery_success_duplicate_id_total": int(total_recovery_success_duplicate_id),
                "recovery_success_foreign_id_total": int(total_recovery_success_foreign_id),
                "recovery_rate_reference": safe_pct(total_recovery_success_reference, total_recovery_attempts),
                "recovery_rate_own_identity": safe_pct(total_recovery_success_own_identity, total_recovery_attempts),
                "recovery_rate_duplicate_id": safe_pct(total_recovery_success_duplicate_id, total_recovery_attempts),
                "recovery_rate_foreign_id": safe_pct(total_recovery_success_foreign_id, total_recovery_attempts),
                "tracking_recall": collapsed_identity_metrics.get("tracking_recall", None),
                "mean_tracking_iou": collapsed_identity_metrics.get("mean_tracking_iou", None),
                "deta": collapsed_identity_metrics.get("deta", None),
                "assa": collapsed_identity_metrics.get("assa", None),
                "hota": collapsed_identity_metrics.get("hota", None),
                "n_mt_objects": int(n_mt_objects),
                "n_pt_objects": int(n_pt_objects),
                "n_ml_objects": int(n_ml_objects),
                "mt": safe_pct(n_mt_objects, n_objects_total),
                "pt": safe_pct(n_pt_objects, n_objects_total),
                "ml": safe_pct(n_ml_objects, n_objects_total),
                "distance_used_count": int(distance_used_count),
                "distance_resolved_count": int(distance_resolved_count),
                "distance_correct_count": int(distance_correct_count),
                "distance_usage_rate": safe_pct(distance_used_count, n_visible_gt_observations),
                "distance_resolution_rate": safe_pct(distance_resolved_count, distance_used_count),
                "distance_disambiguation_accuracy": safe_pct(distance_correct_count, distance_resolved_count),
                "distance_unresolved_rate": safe_pct(
                    max(0, distance_used_count - distance_resolved_count),
                    distance_used_count,
                ),
                "neighbor_sets_available_count": int(neighbor_sets_available_count),
                "neighbor_sets_available_rate": safe_pct(neighbor_sets_available_count, n_visible_gt_observations),
                "context_intervened_count": int(context_intervened_count),
                "context_correct_count": int(context_correct_count),
                "context_rescue_count": int(context_rescue_count),
                "context_veto_case_count": int(context_veto_case_count),
                "context_intervention_rate": safe_pct(context_intervened_count, n_visible_gt_observations),
                "context_intervention_accuracy": safe_pct(context_correct_count, context_intervened_count),
                "context_rescue_rate": safe_pct(context_rescue_count, n_visible_gt_observations),
                "context_veto_rate": safe_pct(context_veto_case_count, n_visible_gt_observations),
                "context_net_gain": (
                    float(context_correct_count - max(0, context_intervened_count - context_correct_count))
                    / float(n_visible_gt_observations)
                    if n_visible_gt_observations > 0
                    else None
                ),
            }
        )
        summary.update(build_memory_summary(per_frame_rows))

        per_class_rows = []
        gt_objects_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in per_object_rows:
            gt_objects_by_class[str(row.get("gt_class_name") or "unknown")].append(row)

        pred_tracks_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in pred_rows:
            pred_tracks_by_class[str(row.get("pred_class_name") or "unknown")].append(row)

        reopened_rows_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in existing_gt_reopened_as_new_rows:
            reopened_rows_by_class[str(row.get("gt_class_name") or "unknown")].append(row)

        for class_name in sorted(set(gt_objects_by_class.keys()) | set(pred_tracks_by_class.keys())):
            class_gt_rows = list(gt_objects_by_class.get(str(class_name), []))
            class_pred_rows = list(pred_tracks_by_class.get(str(class_name), []))
            class_reopen_rows = list(reopened_rows_by_class.get(str(class_name), []))
            class_case_rows = list(by_class_case.get(str(class_name), []))

            n_gt_class = int(len(class_gt_rows))
            total_frames_class = int(sum(int(row.get("n_frames", 0) or 0) for row in class_gt_rows))
            weighted_strict_num = float(
                sum(float(row.get("strict_accuracy", 0.0) or 0.0) * float(row.get("n_frames", 0) or 0) for row in class_gt_rows)
            )
            weighted_perm_num = float(
                sum(
                    float(row.get("permissive_accuracy", 0.0) or 0.0) * float(row.get("n_frames", 0) or 0)
                    for row in class_gt_rows
                )
            )
            per_class_rows.append(
                {
                    "class_name": str(class_name),
                    "n_gt_objects": n_gt_class,
                    "n_real_pred_tracks": int(len(class_pred_rows)),
                    "pred_track_surplus_vs_gt": int(len(class_pred_rows) - n_gt_class),
                    "pred_track_inflation_factor": (
                        float(len(class_pred_rows)) / float(n_gt_class) if n_gt_class > 0 else None
                    ),
                    "weighted_strict_accuracy": (
                        float(weighted_strict_num) / float(total_frames_class) if total_frames_class > 0 else None
                    ),
                    "weighted_permissive_accuracy": (
                        float(weighted_perm_num) / float(total_frames_class) if total_frames_class > 0 else None
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
                    "duplicate_id_segment_count": int(
                        sum(int(row.get("duplicate_id_segment_count", 0) or 0) for row in class_gt_rows)
                    ),
                    "foreign_id_segment_count": int(
                        sum(int(row.get("foreign_id_segment_count", 0) or 0) for row in class_gt_rows)
                    ),
                    "duplicate_id_frame_count": int(
                        sum(int(row.get("duplicate_id_frame_count", 0) or 0) for row in class_gt_rows)
                    ),
                    "foreign_id_frame_count": int(
                        sum(int(row.get("foreign_id_frame_count", 0) or 0) for row in class_gt_rows)
                    ),
                    "duplicate_id_rate_visible": safe_pct(
                        sum(int(row.get("duplicate_id_frame_count", 0) or 0) for row in class_gt_rows),
                        len(class_case_rows),
                    ),
                    "foreign_id_rate_visible": safe_pct(
                        sum(int(row.get("foreign_id_frame_count", 0) or 0) for row in class_gt_rows),
                        len(class_case_rows),
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
                    "tracking_recall": safe_pct(
                        sum(1 for row in class_case_rows if bool(row.get("collapsed_global_correct", False))),
                        len(class_case_rows),
                    ),
                    "mean_tracking_iou": (
                        float(sum(float(row.get("tracking_iou", 0.0) or 0.0) for row in class_case_rows)) / float(len(class_case_rows))
                        if class_case_rows
                        else None
                    ),
                    "deta": safe_pct(
                        sum(1 for row in class_case_rows if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0),
                        len(class_case_rows),
                    ),
                    "assa": safe_pct(
                        sum(1 for row in class_case_rows if bool(row.get("collapsed_global_correct", False))),
                        sum(1 for row in class_case_rows if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0),
                    ),
                    "hota": (
                        float(
                            np.sqrt(
                                max(
                                    0.0,
                                    float(
                                        safe_pct(
                                            sum(1 for row in class_case_rows if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0),
                                            len(class_case_rows),
                                        ) or 0.0
                                    )
                                    * float(
                                        safe_pct(
                                            sum(1 for row in class_case_rows if bool(row.get("collapsed_global_correct", False))),
                                            sum(1 for row in class_case_rows if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0),
                                        ) or 0.0
                                    ),
                                )
                            )
                        )
                        if class_case_rows
                        else None
                    ),
                    "accuracy_existing_vs_new_collapsed": safe_pct(
                        sum(1 for row in class_case_rows if bool(row.get("collapsed_existing_vs_new_correct", False))),
                        len(class_case_rows),
                    ),
                    "accuracy_parent_collapsed": safe_pct(
                        sum(1 for row in class_case_rows if str(row.get("real_state")) == "existing" and bool(row.get("collapsed_parent_correct", False))),
                        sum(1 for row in class_case_rows if str(row.get("real_state")) == "existing"),
                    ),
                    "new_detection_accuracy_collapsed": safe_pct(
                        sum(1 for row in class_case_rows if str(row.get("real_state")) == "new" and str(row.get("collapsed_kind")) == "new"),
                        sum(1 for row in class_case_rows if str(row.get("real_state")) == "new"),
                    ),
                    "uncertain_rate": safe_pct(
                        sum(
                            1
                            for row in class_case_rows
                            if str(row.get("final_decision")) in {"AMBIGUOUS_TRACK", "PROVISIONAL_PARENT", "PROVISIONAL_NEW"}
                        ),
                        len(class_case_rows),
                    ),
                    "hypothesis_recall_uncertain": safe_pct(
                        sum(1 for row in class_case_rows if bool(row.get("ambiguous_set_hit", False)))
                        + sum(1 for row in class_case_rows if bool(row.get("provisional_parent_hit", False)))
                        + sum(
                            1
                            for row in class_case_rows
                            if str(row.get("final_decision")) == "PROVISIONAL_NEW"
                            and str(row.get("real_state")) == "new"
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
                        len(class_case_rows),
                    ),
                    "distance_disambiguation_accuracy": safe_pct(
                        sum(1 for row in class_case_rows if bool(row.get("distance_correct", False))),
                        sum(1 for row in class_case_rows if bool(row.get("distance_resolved", False))),
                    ),
                    "context_intervention_rate": safe_pct(
                        sum(1 for row in class_case_rows if bool(row.get("context_intervened", False))),
                        len(class_case_rows),
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
                        len({int(row["gt_instance_id"]) for row in class_reopen_rows})
                    ),
                }
            )

        return {
            "collapsed_identity_metrics": collapsed_identity_metrics,
            "collapsed_metrics": collapsed_metrics,
            "uncertainty_metrics": uncertainty_metrics,
            "summary": summary,
            "per_class": per_class_rows,
            "per_case": case_rows,
            "per_case_modules": [
                {
                    "frame_id": int(row["frame_id"]),
                    "gt_instance_id": int(row["gt_instance_id"]),
                    "gt_label": str(row["gt_label"]),
                    "gt_class_name": row.get("gt_class_name", None),
                    "det_id": int(row["det_id"]),
                    "final_decision": str(row.get("final_decision", "")),
                    "final_reason": str(row.get("final_reason", "")),
                    "match_source": str(row.get("match_source", "")),
                    "distance_used": bool(row.get("distance_used", False)),
                    "distance_resolved": bool(row.get("distance_resolved", False)),
                    "distance_correct": bool(row.get("distance_correct", False)),
                    "neighbor_sets_available": bool(row.get("neighbor_sets_available", False)),
                    "context_intervened": bool(row.get("context_intervened", False)),
                    "context_change_correct": bool(row.get("context_change_correct", False)),
                    "context_rescue_applied": bool(row.get("context_rescue_applied", False)),
                    "context_veto_candidate_count": int(row.get("context_veto_candidate_count", 0) or 0),
                    "best_sim_object_id": row.get("best_sim_object_id", None),
                    "best_sim_score": row.get("best_sim_score", None),
                    "best_final_candidate_object_id": row.get("best_final_candidate_object_id", None),
                    "best_final_candidate_score": row.get("best_final_candidate_score", None),
                    "best_sim_margin": row.get("best_sim_margin", None),
                    "best_final_margin": row.get("best_final_margin", None),
                    "selected_candidate_score_sets": row.get("selected_candidate_score_sets", None),
                    "selected_candidate_quality_sets": row.get("selected_candidate_quality_sets", None),
                }
                for row in case_rows
            ],
            "per_frame": per_frame_rows,
            "per_object": per_object_rows,
            "per_pred_track": pred_rows,
            "events": {
                "swap": swap_events,
                "theft_with_new_id": theft_with_new_id_events,
                "theft_with_displacement": theft_with_displacement_events,
            },
            "canonical_gt_by_pred": {int(k): int(v) for k, v in canonical_gt_by_pred.items()},
            "reference_pred_by_gt": {int(k): int(v) for k, v in reference_pred_by_gt.items()},
            "owner_gt_by_pred_majority": {int(k): int(v) for k, v in owner_gt_by_pred_majority.items()},
        }

    def build_global_reference_mapping(self, *, by_gt: dict[int, list[AssignmentRecord]]) -> dict[int, int]:
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as e:
            raise RuntimeError("The offline evaluator requires scipy for global GT<->pred matching.") from e

        overlap_counts: dict[tuple[int, int], int] = defaultdict(int)
        overlap_iou_sum: dict[tuple[int, int], float] = defaultdict(float)
        gt_ids_by_class: dict[str, set[int]] = defaultdict(set)
        pred_ids_by_class: dict[str, set[int]] = defaultdict(set)

        for gt_id, gt_records in (by_gt or {}).items():
            gt_class_name = str(
                self.gt_meta.get(int(gt_id), {}).get(
                    "class_name",
                    gt_records[0].gt_class_name if gt_records else "unknown",
                )
            )
            gt_ids_by_class[str(gt_class_name)].add(int(gt_id))
            for rec in gt_records or []:
                pred_id = int(rec.pred_object_id)
                overlap_counts[(int(gt_id), int(pred_id))] += 1
                overlap_iou_sum[(int(gt_id), int(pred_id))] += float(rec.iou)
                pred_class_name = self.pred_meta.get(int(pred_id), {}).get("class_name", rec.pred_class_name)
                pred_ids_by_class[str(pred_class_name)].add(int(pred_id))

        reference_pred_by_gt: dict[int, int] = {}

        for class_name in sorted(set(gt_ids_by_class.keys()) | set(pred_ids_by_class.keys())):
            gt_ids = sorted(int(x) for x in gt_ids_by_class.get(str(class_name), set()))
            pred_ids = sorted(int(x) for x in pred_ids_by_class.get(str(class_name), set()))
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

            if not np.any(weight > 0.0):
                continue

            cost = -weight
            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind.tolist(), col_ind.tolist()):
                if float(weight[int(i), int(j)]) <= 0.0:
                    continue
                reference_pred_by_gt[int(gt_ids[int(i)])] = int(pred_ids[int(j)])

        return reference_pred_by_gt

    def build_case_rows(self, *, owner_gt_by_pred_majority: dict[int, int]) -> list[dict]:
        rows: list[dict] = []
        for case in sorted(
            self.case_records,
            key=lambda r: (
                int(r.frame_id),
                int(r.gt_instance_id),
                int(r.det_id if r.det_id is not None else -1),
            ),
        ):
            collapsed_kind, collapsed_pred_object_id = self.resolved_collapsed_by_det.get(
                (int(case.frame_id), int(case.det_id)),
                self.collapse_case_decision(case=case),
            )
            collapsed_pred_class_name = self.resolve_collapsed_pred_class_name(
                case=case,
                collapsed_kind=collapsed_kind,
                collapsed_pred_object_id=collapsed_pred_object_id,
            )

            is_existing_gt = bool(not case.real_is_new)
            is_new_gt = bool(case.real_is_new)
            collapsed_existing_vs_new_correct = (
                (collapsed_kind == "new" and is_new_gt)
                or (collapsed_kind == "existing" and is_existing_gt)
            )
            collapsed_parent_correct = (
                bool(is_existing_gt)
                and collapsed_kind == "existing"
                and collapsed_pred_object_id is not None
                and owner_gt_by_pred_majority.get(int(collapsed_pred_object_id), None) == int(case.gt_instance_id)
            )
            collapsed_global_correct = (
                bool(is_new_gt and collapsed_kind == "new")
                or bool(is_existing_gt and collapsed_parent_correct)
            )

            firm_kind = self.firm_kind(case.final_decision)
            firm_pred_object_id = case.firm_pred_object_id
            firm_parent_correct = (
                case.final_decision == "MATCH"
                and firm_pred_object_id is not None
                and owner_gt_by_pred_majority.get(int(firm_pred_object_id), None) == int(case.gt_instance_id)
            )
            firm_global_correct = (
                bool(case.final_decision == "NEW" and is_new_gt)
                or bool(case.final_decision == "MATCH" and is_existing_gt and firm_parent_correct)
            )

            ambiguous_set_hit = False
            if case.final_decision == "AMBIGUOUS_TRACK":
                ambiguous_set_hit = any(
                    owner_gt_by_pred_majority.get(int(oid), None) == int(case.gt_instance_id)
                    for oid in (case.ambiguous_candidate_ids or [])
                )

            provisional_parent_hit = False
            if case.final_decision == "PROVISIONAL_PARENT":
                provisional_parent_hit = any(
                    owner_gt_by_pred_majority.get(int(parent_id), None) == int(case.gt_instance_id)
                    for parent_id in (case.provisional_parent_ids or [])
                )

            novelty_detected_uncertain = bool(
                is_new_gt and case.final_decision in {"PROVISIONAL_NEW", "NEW"}
            )
            gt_area_frac = (
                float(case.gt_area_px) / float(case.frame_area_px)
                if int(case.frame_area_px) > 0 and int(case.gt_area_px) > 0
                else None
            )
            n_total_distractors = max(0, int(case.frame_gt_count) - 1)
            n_same_class_distractors = max(0, int(case.frame_same_class_count) - 1)
            tracking_iou = float(case.iou) if bool(collapsed_global_correct) else 0.0
            distance_correct = bool(case.distance_resolved and collapsed_global_correct)
            context_change_correct = bool(case.context_intervened and collapsed_global_correct)

            rows.append(
                {
                    "frame_id": int(case.frame_id),
                    "gt_instance_id": int(case.gt_instance_id),
                    "gt_label": str(case.gt_label),
                    "gt_class_name": case.gt_class_name,
                    "det_id": int(case.det_id),
                    "iou": float(case.iou),
                    "real_state": "new" if bool(case.real_is_new) else "existing",
                    "final_decision": str(case.final_decision),
                    "final_reason": str(case.final_reason),
                    "final_decision_is_existing_assignment": bool(case.final_decision == "MATCH"),
                    "final_decision_is_new_assignment": bool(case.final_decision == "NEW"),
                    "final_decision_parent_correct": bool(firm_parent_correct),
                    "final_decision_global_correct": bool(firm_global_correct),
                    "firm_kind": str(firm_kind),
                    "firm_pred_object_id": None if firm_pred_object_id is None else int(firm_pred_object_id),
                    "ambiguous_candidate_ids": [int(x) for x in (case.ambiguous_candidate_ids or [])],
                    "provisional_parent_ids": [int(x) for x in (case.provisional_parent_ids or [])],
                    "provisional_temp_id": None if case.provisional_temp_id is None else int(case.provisional_temp_id),
                    "created_object_id": None if case.created_object_id is None else int(case.created_object_id),
                    "created_origin_provisional_temp_id": (
                        None
                        if case.created_origin_provisional_temp_id is None
                        else int(case.created_origin_provisional_temp_id)
                    ),
                    "collapsed_kind": str(collapsed_kind),
                    "collapsed_pred_object_id": (
                        None if collapsed_pred_object_id is None else int(collapsed_pred_object_id)
                    ),
                    "collapsed_pred_class_name": collapsed_pred_class_name,
                    "collapsed_existing_vs_new_correct": bool(collapsed_existing_vs_new_correct),
                    "collapsed_parent_correct": bool(collapsed_parent_correct),
                    "collapsed_global_correct": bool(collapsed_global_correct),
                    "firm_global_correct": bool(firm_global_correct),
                    "ambiguous_set_hit": bool(ambiguous_set_hit),
                    "provisional_parent_hit": bool(provisional_parent_hit),
                    "novelty_detected_uncertain": bool(novelty_detected_uncertain),
                    "gt_area_px": int(case.gt_area_px),
                    "frame_gt_count": int(case.frame_gt_count),
                    "frame_same_class_count": int(case.frame_same_class_count),
                    "frame_area_px": int(case.frame_area_px),
                    "gt_area_frac": gt_area_frac,
                    "n_gt_visible_in_frame": int(case.frame_gt_count),
                    "n_gt_same_class_in_frame": int(case.frame_same_class_count),
                    "n_total_distractors": int(n_total_distractors),
                    "n_same_class_distractors": int(n_same_class_distractors),
                    "gt_is_new_this_frame": bool(case.real_is_new),
                    "gt_age_frames": int(max(0, int(case.frame_id) - int(self.gt_meta.get(int(case.gt_instance_id), {}).get("first_frame", case.frame_id)))),
                    "is_tracked_visible_case": bool(collapsed_kind in {"existing", "new"} and collapsed_pred_object_id is not None),
                    "tracking_iou": float(tracking_iou),
                    "best_sim_object_id": None if case.best_sim_object_id is None else int(case.best_sim_object_id),
                    "best_sim_score": float(case.best_sim_score),
                    "best_final_candidate_object_id": (
                        None if case.best_final_candidate_object_id is None else int(case.best_final_candidate_object_id)
                    ),
                    "best_final_candidate_score": float(case.best_final_candidate_score),
                    "best_sim_margin": None if case.best_sim_margin is None else float(case.best_sim_margin),
                    "best_final_margin": None if case.best_final_margin is None else float(case.best_final_margin),
                    "match_source": str(case.match_source),
                    "distance_used": bool(case.distance_used),
                    "distance_resolved": bool(case.distance_resolved),
                    "distance_correct": bool(distance_correct),
                    "neighbor_sets_available": bool(case.neighbor_sets_available),
                    "context_intervened": bool(case.context_intervened),
                    "context_change_correct": bool(context_change_correct),
                    "context_rescue_applied": bool(case.context_rescue_applied),
                    "context_veto_candidate_count": int(case.context_veto_candidate_count),
                    "selected_candidate_score_sets": float(case.selected_candidate_score_sets),
                    "selected_candidate_quality_sets": float(case.selected_candidate_quality_sets),
                }
            )
        return rows

    def compute_decision_metrics(self, *, case_rows: list[dict]) -> tuple[dict, dict]:
        n_cases = int(len(case_rows))
        n_existing_gt = int(sum(1 for row in case_rows if row["real_state"] == "existing"))
        n_new_gt = int(sum(1 for row in case_rows if row["real_state"] == "new"))

        ambiguous_rows = [row for row in case_rows if row["final_decision"] == "AMBIGUOUS_TRACK"]
        provisional_parent_rows = [row for row in case_rows if row["final_decision"] == "PROVISIONAL_PARENT"]
        provisional_new_rows = [row for row in case_rows if row["final_decision"] == "PROVISIONAL_NEW"]
        firm_rows = [row for row in case_rows if row["final_decision"] in {"MATCH", "NEW"}]

        ambiguous_candidate_sizes = [
            int(len(row["ambiguous_candidate_ids"]))
            for row in ambiguous_rows
        ]
        provisional_parent_sizes = [
            int(len(row["provisional_parent_ids"]))
            for row in provisional_parent_rows
        ]

        collapsed = {
            "n_cases": int(n_cases),
            "n_existing_gt": int(n_existing_gt),
            "n_new_gt": int(n_new_gt),
            "n_ambiguous_cases": int(len(ambiguous_rows)),
            "accuracy_global_collapsed": safe_pct(
                sum(1 for row in case_rows if bool(row["collapsed_global_correct"])),
                n_cases,
            ),
            "accuracy_existing_vs_new_collapsed": safe_pct(
                sum(1 for row in case_rows if bool(row["collapsed_existing_vs_new_correct"])),
                n_cases,
            ),
            "accuracy_parent_collapsed": safe_pct(
                sum(1 for row in case_rows if row["real_state"] == "existing" and bool(row["collapsed_parent_correct"])),
                n_existing_gt,
            ),
            "set_accuracy_ambiguous": safe_pct(
                sum(1 for row in ambiguous_rows if bool(row["ambiguous_set_hit"])),
                len(ambiguous_rows),
            ),
            "new_detection_accuracy_collapsed": safe_pct(
                sum(
                    1
                    for row in case_rows
                    if row["real_state"] == "new" and str(row["collapsed_kind"]) == "new"
                ),
                n_new_gt,
            ),
        }

        uncertainty = {
            "n_cases": int(n_cases),
            "n_firm": int(len(firm_rows)),
            "n_ambiguous": int(len(ambiguous_rows)),
            "n_provisional_parent": int(len(provisional_parent_rows)),
            "n_provisional_new": int(len(provisional_new_rows)),
            "coverage_firm": safe_pct(len(firm_rows), n_cases),
            "firm_accuracy": safe_pct(
                sum(1 for row in firm_rows if bool(row["firm_global_correct"])),
                len(firm_rows),
            ),
            "firm_error_rate_over_all_cases": safe_pct(
                sum(1 for row in firm_rows if not bool(row["firm_global_correct"])),
                n_cases,
            ),
            "ambiguity_rate": safe_pct(len(ambiguous_rows), n_cases),
            "provisional_parent_rate": safe_pct(len(provisional_parent_rows), n_cases),
            "provisional_new_rate": safe_pct(len(provisional_new_rows), n_cases),
            "uncertain_rate": safe_pct(
                len(ambiguous_rows) + len(provisional_parent_rows) + len(provisional_new_rows),
                n_cases,
            ),
            "set_accuracy_ambiguous": safe_pct(
                sum(1 for row in ambiguous_rows if bool(row["ambiguous_set_hit"])),
                len(ambiguous_rows),
            ),
            "parent_hit_rate_provisional": safe_pct(
                sum(1 for row in provisional_parent_rows if bool(row["provisional_parent_hit"])),
                len(provisional_parent_rows),
            ),
            "new_detection_accuracy_uncertain": safe_pct(
                sum(1 for row in case_rows if bool(row["novelty_detected_uncertain"])),
                n_new_gt,
            ),
            "avg_ambiguous_candidates": (
                float(np.mean(ambiguous_candidate_sizes)) if ambiguous_candidate_sizes else None
            ),
            "max_ambiguous_candidates": (
                int(max(ambiguous_candidate_sizes)) if ambiguous_candidate_sizes else None
            ),
            "avg_provisional_parent_candidates": (
                float(np.mean(provisional_parent_sizes)) if provisional_parent_sizes else None
            ),
            "max_provisional_parent_candidates": (
                int(max(provisional_parent_sizes)) if provisional_parent_sizes else None
            ),
            "hypothesis_recall_uncertain": safe_pct(
                sum(1 for row in ambiguous_rows if bool(row["ambiguous_set_hit"]))
                + sum(1 for row in provisional_parent_rows if bool(row["provisional_parent_hit"]))
                + sum(
                    1
                    for row in provisional_new_rows
                    if str(row.get("real_state")) == "new"
                ),
                len(ambiguous_rows) + len(provisional_parent_rows) + len(provisional_new_rows),
            ),
        }

        return collapsed, uncertainty

    def compute_collapsed_reference_mapping(self, *, case_rows: list[dict]) -> dict[int, int]:
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as e:
            raise RuntimeError("The offline evaluator requires scipy to compute IDF1/IDP/IDR.") from e

        visible_rows = [
            row for row in case_rows
            if str(row["collapsed_kind"]) in {"existing", "new"}
            and row["collapsed_pred_object_id"] is not None
        ]

        if not visible_rows:
            return {}

        gt_ids_by_class: dict[str, set[int]] = defaultdict(set)
        pred_ids_by_class: dict[str, set[int]] = defaultdict(set)
        overlap_counts: dict[tuple[int, int], int] = defaultdict(int)
        overlap_iou_sum: dict[tuple[int, int], float] = defaultdict(float)

        for row in visible_rows:
            gt_id = int(row["gt_instance_id"])
            pred_id = int(row["collapsed_pred_object_id"])
            gt_class_name = str(row["gt_class_name"])
            pred_class_name = str(row["collapsed_pred_class_name"])
            gt_ids_by_class[gt_class_name].add(gt_id)
            pred_ids_by_class[pred_class_name].add(pred_id)
            overlap_counts[(gt_id, pred_id)] += 1
            overlap_iou_sum[(gt_id, pred_id)] += float(row["iou"])

        reference_pred_by_gt: dict[int, int] = {}

        for class_name in sorted(set(gt_ids_by_class.keys()) | set(pred_ids_by_class.keys())):
            cls_gt_ids = sorted(int(x) for x in gt_ids_by_class.get(str(class_name), set()))
            cls_pred_ids = sorted(int(x) for x in pred_ids_by_class.get(str(class_name), set()))
            if not cls_gt_ids or not cls_pred_ids:
                continue

            weight = np.zeros((len(cls_gt_ids), len(cls_pred_ids)), dtype=np.float64)
            for i, gt_id in enumerate(cls_gt_ids):
                for j, pred_id in enumerate(cls_pred_ids):
                    count = int(overlap_counts.get((gt_id, pred_id), 0))
                    iou_sum = float(overlap_iou_sum.get((gt_id, pred_id), 0.0))
                    if count <= 0 and iou_sum <= 0.0:
                        continue
                    weight[i, j] = float(count) * 1000.0 + float(iou_sum)

            if not np.any(weight > 0.0):
                continue

            row_ind, col_ind = linear_sum_assignment(-weight)
            for i, j in zip(row_ind.tolist(), col_ind.tolist()):
                if float(weight[int(i), int(j)]) <= 0.0:
                    continue
                reference_pred_by_gt[int(cls_gt_ids[int(i)])] = int(cls_pred_ids[int(j)])

        return reference_pred_by_gt

    def compute_collapsed_identity_metrics(self, *, case_rows: list[dict]) -> dict[str, Any]:
        if not self.frame_collapsed_state:
            return {
                "n_gt_observations": 0,
                "n_matched_gt_observations": 0,
                "n_pred_observations": 0,
                "n_unique_gt_ids": 0,
                "n_unique_existing_pred_ids": 0,
                "n_unique_new_pred_ids": 0,
                "n_unique_pred_ids": 0,
                "idtp": 0,
                "idfp": 0,
                "idfn": 0,
                "idp": None,
                "idr": None,
                "idf1": None,
                "idsw": 0,
                "frag": 0,
                "tracking_recall": None,
                "mean_tracking_iou": None,
                "deta": None,
                "assa": None,
                "hota": None,
                "reference_pred_by_gt": {},
            }

        reference_pred_by_gt = self.compute_collapsed_reference_mapping(case_rows=case_rows)
        pred_owner_gt = {int(pred_id): int(gt_id) for gt_id, pred_id in reference_pred_by_gt.items()}
        visible_rows = [
            row
            for row in case_rows
            if str(row.get("collapsed_kind")) in {"existing", "new"}
            and row.get("collapsed_pred_object_id") is not None
        ]
        gt_ids = sorted(
            {
                int(gt_id)
                for state in self.frame_collapsed_state.values()
                for gt_id in (state.get("gt_ids", []) or [])
            }
        )
        existing_pred_ids = sorted(
            {
                int(row["collapsed_pred_object_id"])
                for row in visible_rows
                if str(row.get("collapsed_kind")) == "existing"
            }
        )
        new_pred_ids = sorted(
            {
                int(row["collapsed_pred_object_id"])
                for row in visible_rows
                if str(row.get("collapsed_kind")) == "new"
            }
        )
        pred_ids = sorted(set(existing_pred_ids) | set(new_pred_ids))
        n_visible_cases = int(len(case_rows))
        n_matched_gt_observations = int(
            sum(1 for row in case_rows if int(row.get("det_id", -1)) >= 0 and float(row.get("iou", 0.0) or 0.0) > 0.0)
        )
        n_tracking_correct = int(sum(1 for row in case_rows if bool(row.get("collapsed_global_correct", False))))
        tracking_iou_sum = float(sum(float(row.get("tracking_iou", 0.0) or 0.0) for row in case_rows))
        deta = safe_pct(n_matched_gt_observations, n_visible_cases)
        assa = safe_pct(n_tracking_correct, n_matched_gt_observations)
        hota = None
        if deta is not None and assa is not None:
            hota = float(np.sqrt(max(0.0, float(deta) * float(assa))))

        idtp = 0
        idfn = 0
        idfp = 0

        by_gt_timeline: dict[int, list[dict[str, Any]]] = defaultdict(list)
        n_gt_observations = 0
        n_pred_observations = 0

        for frame_id in sorted(self.frame_collapsed_state):
            state = self.frame_collapsed_state[int(frame_id)] or {}
            gt_ids_in_frame = [int(x) for x in (state.get("gt_ids", []) or [])]
            pred_to_gt_ids = {
                int(pred_id): [int(x) for x in (gt_ids_for_pred or [])]
                for pred_id, gt_ids_for_pred in (state.get("pred_to_gt_ids", {}) or {}).items()
            }
            gt_to_pred: dict[int, int] = {}
            for pred_id, gt_ids_for_pred in pred_to_gt_ids.items():
                if len(gt_ids_for_pred) == 1 and int(gt_ids_for_pred[0]) not in gt_to_pred:
                    gt_to_pred[int(gt_ids_for_pred[0])] = int(pred_id)

            n_gt_observations += int(len(gt_ids_in_frame))
            n_pred_observations += int(len(pred_to_gt_ids))

            for gt_id in gt_ids_in_frame:
                assigned_pred_id = gt_to_pred.get(int(gt_id), None)
                ref_pred = reference_pred_by_gt.get(int(gt_id), None)

                if ref_pred is not None and assigned_pred_id == ref_pred:
                    idtp += 1
                else:
                    idfn += 1

                by_gt_timeline[int(gt_id)].append(
                    {
                        "frame_id": int(frame_id),
                        "pred_id": None if assigned_pred_id is None else int(assigned_pred_id),
                        "is_correct": bool(ref_pred is not None and assigned_pred_id == ref_pred),
                        "is_tracked": bool(assigned_pred_id is not None),
                    }
                )

            for pred_id, gt_ids_for_pred in pred_to_gt_ids.items():
                owner_gt = pred_owner_gt.get(int(pred_id), None)
                if owner_gt is None:
                    idfp += 1
                    continue
                if len(gt_ids_for_pred) != 1 or int(gt_ids_for_pred[0]) != int(owner_gt):
                    idfp += 1

        idsw = 0
        frag = 0

        for gt_id in sorted(by_gt_timeline):
            seq = sorted(by_gt_timeline[gt_id], key=lambda x: int(x["frame_id"]))

            prev_pred_non_null = None
            for item in seq:
                cur_pred = item["pred_id"]
                if cur_pred is None:
                    continue
                if prev_pred_non_null is not None and int(cur_pred) != int(prev_pred_non_null):
                    idsw += 1
                prev_pred_non_null = int(cur_pred)

            tracked_runs = 0
            in_tracked_run = False
            for item in seq:
                cur_tracked = bool(item["is_tracked"])
                if cur_tracked and not in_tracked_run:
                    tracked_runs += 1
                    in_tracked_run = True
                elif not cur_tracked:
                    in_tracked_run = False
            frag += max(0, tracked_runs - 1)

        return {
            "n_gt_observations": int(n_gt_observations),
            "n_matched_gt_observations": int(n_matched_gt_observations),
            "n_pred_observations": int(n_pred_observations),
            "n_unique_gt_ids": int(len(gt_ids)),
            "n_unique_existing_pred_ids": int(len(existing_pred_ids)),
            "n_unique_new_pred_ids": int(len(new_pred_ids)),
            "n_unique_pred_ids": int(len(pred_ids)),
            "idtp": int(idtp),
            "idfp": int(idfp),
            "idfn": int(idfn),
            "idp": safe_pct(idtp, idtp + idfp),
            "idr": safe_pct(idtp, idtp + idfn),
            "idf1": safe_pct(2 * idtp, 2 * idtp + idfp + idfn),
            "idsw": int(idsw),
            "frag": int(frag),
            "tracking_recall": safe_pct(n_tracking_correct, n_visible_cases),
            "mean_tracking_iou": (float(tracking_iou_sum) / float(n_visible_cases)) if n_visible_cases > 0 else None,
            "deta": deta,
            "assa": assa,
            "hota": hota,
            "reference_pred_by_gt": {int(k): int(v) for k, v in reference_pred_by_gt.items()},
        }

    @staticmethod
    def firm_kind(final_decision: str) -> str:
        decision = str(final_decision or "").upper()
        if decision == "MATCH":
            return "existing"
        if decision == "NEW":
            return "new"
        return "uncertain"

    def collapse_case_decision(self, *, case: DecisionCaseRecord) -> tuple[str, int | None]:
        return self.collapse_detection_output(
            final_decision=case.final_decision,
            firm_pred_object_id=case.firm_pred_object_id,
            ambiguous_candidate_ids=case.ambiguous_candidate_ids,
            provisional_parent_ids=case.provisional_parent_ids,
            provisional_temp_id=case.provisional_temp_id,
            created_object_id=case.created_object_id,
            created_origin_provisional_temp_id=case.created_origin_provisional_temp_id,
        )

    def build_resolution_options(
        self,
        *,
        final_decision: str,
        firm_pred_object_id: int | None,
        ambiguous_candidate_ids: list[int],
        ambiguous_candidate_scores: dict[int, float],
        provisional_parent_ids: list[int],
        provisional_parent_scores: dict[int, float],
        provisional_temp_id: int | None,
        created_object_id: int | None,
        created_origin_provisional_temp_id: int | None,
    ) -> list[dict[str, Any]]:
        decision = str(final_decision or "").upper().strip()
        options: list[dict[str, Any]] = []
        seen: set[tuple[str, int | None]] = set()

        def add_option(kind: str, pred_id: int | None, score: float, source: str) -> None:
            key = (str(kind), None if pred_id is None else int(pred_id))
            if key in seen:
                return
            seen.add(key)
            options.append(
                {
                    "kind": str(kind),
                    "pred_id": None if pred_id is None else int(pred_id),
                    "score": float(score),
                    "source": str(source),
                }
            )

        if decision == "MATCH":
            add_option("existing", firm_pred_object_id, 1.0e12, "firm")
        elif decision == "NEW":
            synthetic_id = self.synthetic_new_id_from_fields(
                provisional_temp_id=provisional_temp_id,
                created_object_id=created_object_id,
                created_origin_provisional_temp_id=created_origin_provisional_temp_id,
            )
            add_option("new", synthetic_id, 1.0e12, "new")
        elif decision == "AMBIGUOUS_TRACK":
            for rank, pred_id in enumerate(ambiguous_candidate_ids or []):
                score = float(ambiguous_candidate_scores.get(int(pred_id), 0.0))
                add_option("existing", int(pred_id), score - (1e-6 * float(rank)), "ambiguous")
        elif decision == "PROVISIONAL_PARENT":
            for rank, pred_id in enumerate(provisional_parent_ids or []):
                score = float(provisional_parent_scores.get(int(pred_id), 0.0))
                add_option("existing", int(pred_id), score - (1e-6 * float(rank)), "provisional_parent")
        elif decision == "PROVISIONAL_NEW":
            synthetic_id = self.synthetic_new_id_from_fields(
                provisional_temp_id=provisional_temp_id,
                created_object_id=created_object_id,
                created_origin_provisional_temp_id=created_origin_provisional_temp_id,
            )
            add_option("new", synthetic_id, 1.0e12, "provisional_new")

        add_option("none", None, -1.0e12, "fallback_none")
        return options

    def resolve_frame_collapsed_outputs(self, specs: list[dict[str, Any]]) -> dict[int, tuple[str, int | None]]:
        if str(self.collapse_mode) == "greedy":
            return self.resolve_frame_collapsed_outputs_greedy(specs)
        return self.resolve_frame_collapsed_outputs_hungarian(specs)

    def resolve_frame_collapsed_outputs_greedy(self, specs: list[dict[str, Any]]) -> dict[int, tuple[str, int | None]]:
        if not specs:
            return {}

        mutable_specs: list[dict[str, Any]] = []
        for raw in specs:
            options = list(raw.get("options", []) or [])
            if not options:
                options = [{"kind": "none", "pred_id": None, "score": -1.0e12, "source": "fallback_none"}]
            mutable_specs.append(
                {
                    "det_id": int(raw["det_id"]),
                    "iou": raw.get("iou", None),
                    "choice_idx": 0,
                    "options": options,
                }
            )

        while True:
            pred_to_specs: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for spec in mutable_specs:
                choice = spec["options"][int(spec["choice_idx"])]
                pred_id = choice.get("pred_id", None)
                if pred_id is None:
                    continue
                pred_to_specs[int(pred_id)].append(spec)

            changed = False
            for pred_id, colliders in pred_to_specs.items():
                if len(colliders) <= 1:
                    continue

                winner = sorted(
                    colliders,
                    key=lambda spec: (
                        -float(spec["options"][int(spec["choice_idx"])]["score"]),
                        -float(spec.get("iou", -1.0) if spec.get("iou", None) is not None else -1.0),
                        int(spec["det_id"]),
                    ),
                )[0]

                for spec in colliders:
                    if spec is winner:
                        continue
                    advanced = self.advance_resolution_choice(spec=spec, blocked_pred_id=int(pred_id))
                    changed = bool(changed or advanced)

            if not changed:
                break

        resolved: dict[int, tuple[str, int | None]] = {}
        for spec in mutable_specs:
            choice = spec["options"][int(spec["choice_idx"])]
            resolved[int(spec["det_id"])] = (
                str(choice.get("kind", "none")),
                None if choice.get("pred_id", None) is None else int(choice["pred_id"]),
            )
        return resolved

    def resolve_frame_collapsed_outputs_hungarian(self, specs: list[dict[str, Any]]) -> dict[int, tuple[str, int | None]]:
        if not specs:
            return {}

        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as e:
            raise RuntimeError(
                "Collapse mode 'hungarian' requires scipy to resolve ambiguous conflicts."
            ) from e

        resolved: dict[int, tuple[str, int | None]] = {}
        occupied_pred_ids: set[int] = set()
        ambiguous_specs: list[dict[str, Any]] = []

        for raw in specs:
            det_id = int(raw["det_id"])
            options = list(raw.get("options", []) or [])
            if not options:
                options = [{"kind": "none", "pred_id": None, "score": -1.0e12, "source": "fallback_none"}]

            final_decision = str(raw.get("final_decision", "") or "").upper().strip()
            if final_decision != "AMBIGUOUS_TRACK":
                choice = options[0]
                pred_id = choice.get("pred_id", None)
                if pred_id is not None:
                    occupied_pred_ids.add(int(pred_id))
                resolved[det_id] = (
                    str(choice.get("kind", "none")),
                    None if pred_id is None else int(pred_id),
                )
                continue

            ambiguous_specs.append(
                {
                    "det_id": det_id,
                    "options": options,
                }
            )

        if not ambiguous_specs:
            return resolved

        available_pred_ids: list[int] = sorted(
            {
                int(option["pred_id"])
                for spec in ambiguous_specs
                for option in (spec.get("options", []) or [])
                if option.get("pred_id", None) is not None
                and int(option["pred_id"]) not in occupied_pred_ids
            }
        )
        pred_col_by_id = {int(pred_id): idx for idx, pred_id in enumerate(available_pred_ids)}
        n_specs = int(len(ambiguous_specs))
        n_pred_cols = int(len(available_pred_ids))

        # One dedicated dummy column per ambiguous spec preserves the possibility
        # of leaving the case unresolved without creating cross-spec conflicts.
        score = np.full((n_specs, n_pred_cols + n_specs), -1.0e12, dtype=np.float64)

        for row_idx, spec in enumerate(ambiguous_specs):
            dummy_score = -1.0e12
            for option in (spec.get("options", []) or []):
                pred_id = option.get("pred_id", None)
                opt_score = float(option.get("score", -1.0e12) or -1.0e12)
                if pred_id is None:
                    dummy_score = max(float(dummy_score), float(opt_score))
                    continue
                pred_id = int(pred_id)
                if pred_id in occupied_pred_ids:
                    continue
                col_idx = pred_col_by_id.get(pred_id, None)
                if col_idx is None:
                    continue
                score[int(row_idx), int(col_idx)] = max(float(score[int(row_idx), int(col_idx)]), float(opt_score))

            score[int(row_idx), int(n_pred_cols + row_idx)] = float(dummy_score)

        row_ind, col_ind = linear_sum_assignment(-score)
        chosen_col_by_row = {int(row): int(col) for row, col in zip(row_ind.tolist(), col_ind.tolist())}

        for row_idx, spec in enumerate(ambiguous_specs):
            det_id = int(spec["det_id"])
            chosen_col = chosen_col_by_row.get(int(row_idx), None)
            chosen_pred_id = None
            if chosen_col is not None and int(chosen_col) < int(n_pred_cols):
                chosen_pred_id = int(available_pred_ids[int(chosen_col)])

            if chosen_pred_id is None:
                choice = next(
                    (option for option in (spec.get("options", []) or []) if option.get("pred_id", None) is None),
                    {"kind": "none", "pred_id": None},
                )
            else:
                choice = next(
                    (
                        option
                        for option in (spec.get("options", []) or [])
                        if option.get("pred_id", None) is not None
                        and int(option["pred_id"]) == int(chosen_pred_id)
                    ),
                    {"kind": "existing", "pred_id": int(chosen_pred_id)},
                )

            resolved[det_id] = (
                str(choice.get("kind", "none")),
                None if choice.get("pred_id", None) is None else int(choice["pred_id"]),
            )

        return resolved

    @staticmethod
    def advance_resolution_choice(spec: dict[str, Any], blocked_pred_id: int) -> bool:
        current_idx = int(spec["choice_idx"])
        options = list(spec.get("options", []) or [])
        for idx in range(current_idx + 1, len(options)):
            pred_id = options[idx].get("pred_id", None)
            if pred_id is not None and int(pred_id) == int(blocked_pred_id):
                continue
            spec["choice_idx"] = int(idx)
            return True
        return False

    @staticmethod
    def collapse_detection_output(
        *,
        final_decision: str,
        firm_pred_object_id: int | None,
        ambiguous_candidate_ids: list[int],
        provisional_parent_ids: list[int],
        provisional_temp_id: int | None,
        created_object_id: int | None,
        created_origin_provisional_temp_id: int | None,
    ) -> tuple[str, int | None]:
        decision = str(final_decision or "").upper()

        if decision == "MATCH":
            return ("existing", None if firm_pred_object_id is None else int(firm_pred_object_id))

        if decision == "NEW":
            synthetic_id = TrackingEvaluator.synthetic_new_id_from_fields(
                provisional_temp_id=provisional_temp_id,
                created_object_id=created_object_id,
                created_origin_provisional_temp_id=created_origin_provisional_temp_id,
            )
            return ("new", synthetic_id)

        if decision == "AMBIGUOUS_TRACK":
            best_parent = ambiguous_candidate_ids[0] if ambiguous_candidate_ids else None
            if best_parent is None:
                return ("none", None)
            return ("existing", int(best_parent))

        if decision == "PROVISIONAL_PARENT":
            best_parent = provisional_parent_ids[0] if provisional_parent_ids else None
            if best_parent is None:
                return ("none", None)
            return ("existing", int(best_parent))

        if decision == "PROVISIONAL_NEW":
            synthetic_id = TrackingEvaluator.synthetic_new_id_from_fields(
                provisional_temp_id=provisional_temp_id,
                created_object_id=created_object_id,
                created_origin_provisional_temp_id=created_origin_provisional_temp_id,
            )
            if synthetic_id is None:
                return ("none", None)
            return ("new", synthetic_id)

        return ("none", None)

    def resolve_collapsed_pred_class_name(
        self,
        *,
        case: DecisionCaseRecord,
        collapsed_kind: str,
        collapsed_pred_object_id: int | None,
    ) -> str | None:
        if collapsed_kind == "existing" and collapsed_pred_object_id is not None:
            return self.pred_meta.get(int(collapsed_pred_object_id), {}).get("class_name", case.gt_class_name)
        if collapsed_kind == "new":
            return case.gt_class_name
        return None

    @staticmethod
    def synthetic_new_id_for_case(*, case: DecisionCaseRecord) -> int | None:
        group_key = TrackingEvaluator.synthetic_new_group_key(case=case)
        if group_key is None:
            return None
        group_kind, group_value = group_key
        if group_kind == "provisional":
            return int(SYNTHETIC_PROVISIONAL_NEW_BASE + int(group_value))
        if group_kind == "created":
            return int(SYNTHETIC_CREATED_NEW_BASE + int(group_value))
        return None

    @staticmethod
    def synthetic_new_id_from_fields(
        *,
        provisional_temp_id: int | None,
        created_object_id: int | None,
        created_origin_provisional_temp_id: int | None,
    ) -> int | None:
        group_key = TrackingEvaluator.synthetic_new_group_key_from_fields(
            provisional_temp_id=provisional_temp_id,
            created_object_id=created_object_id,
            created_origin_provisional_temp_id=created_origin_provisional_temp_id,
        )
        if group_key is None:
            return None
        group_kind, group_value = group_key
        if group_kind == "provisional":
            return int(SYNTHETIC_PROVISIONAL_NEW_BASE + int(group_value))
        if group_kind == "created":
            return int(SYNTHETIC_CREATED_NEW_BASE + int(group_value))
        return None

    @staticmethod
    def synthetic_new_group_key(*, case: DecisionCaseRecord) -> tuple[str, int] | None:
        return TrackingEvaluator.synthetic_new_group_key_from_fields(
            provisional_temp_id=case.provisional_temp_id,
            created_object_id=case.created_object_id,
            created_origin_provisional_temp_id=case.created_origin_provisional_temp_id,
        )

    @staticmethod
    def synthetic_new_group_key_from_fields(
        *,
        provisional_temp_id: int | None,
        created_object_id: int | None,
        created_origin_provisional_temp_id: int | None,
    ) -> tuple[str, int] | None:
        # Prefer the committed created object id when available. The originating
        # provisional temp id is not guaranteed to be globally unique across the
        # whole sequence, so using it here can incorrectly merge different GTs.
        if created_object_id is not None:
            return ("created", int(created_object_id))
        if provisional_temp_id is not None:
            return ("provisional", int(provisional_temp_id))
        if created_origin_provisional_temp_id is not None:
            return ("provisional", int(created_origin_provisional_temp_id))
        return None
