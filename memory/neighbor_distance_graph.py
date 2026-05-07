from __future__ import annotations

import math

import cv2
import numpy as np


def mask_touches_border(mask) -> bool:
    if mask is None:
        return False
    arr = np.asarray(mask).astype(bool, copy=False)
    if arr.ndim != 2 or arr.size == 0:
        return False
    return bool(arr[0, :].any() or arr[-1, :].any() or arr[:, 0].any() or arr[:, -1].any())


def _bbox_gap_px(bbox_a, bbox_b) -> float:
    if bbox_a is None or bbox_b is None or len(bbox_a) < 4 or len(bbox_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in bbox_a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in bbox_b[:4]]
    dx = max(0.0, max(ax1 - bx2, bx1 - ax2))
    dy = max(0.0, max(ay1 - by2, by1 - ay2))
    if dx <= 0.0 and dy <= 0.0:
        overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        if overlap_w > 0.0 and overlap_h > 0.0:
            return -math.sqrt(overlap_w * overlap_h)
        return 0.0
    return math.sqrt(dx * dx + dy * dy)


def _bbox_axis_gap_px(a1: float, a2: float, b1: float, b2: float) -> float:
    if a2 < b1:
        return float(b1 - a2)
    if b2 < a1:
        return float(a1 - b2)
    overlap = float(min(a2, b2) - max(a1, b1))
    return float(-max(0.0, overlap))


def _bbox_projected_gaps_px(bbox_a, bbox_b) -> tuple[float | None, float | None]:
    if bbox_a is None or bbox_b is None or len(bbox_a) < 4 or len(bbox_b) < 4:
        return None, None
    ax1, ay1, ax2, ay2 = [float(x) for x in bbox_a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in bbox_b[:4]]
    return (
        float(_bbox_axis_gap_px(ax1, ax2, bx1, bx2)),
        float(_bbox_axis_gap_px(ay1, ay2, by1, by2)),
    )


def _bbox_intersection_area(bbox_a, bbox_b) -> float:
    if bbox_a is None or bbox_b is None or len(bbox_a) < 4 or len(bbox_b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = [float(x) for x in bbox_a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in bbox_b[:4]]
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return float(iw * ih)


def _bbox_contains_point(bbox, x: float, y: float) -> bool:
    if bbox is None or len(bbox) < 4:
        return False
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return bool(x1 <= float(x) <= x2 and y1 <= float(y) <= y2)


def prepare_relation_mask_runtime(mask, *, compute_bbox: bool = True) -> dict | None:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim != 2 or arr.size == 0:
        return None
    if not arr.any():
        return None
    bbox = None
    if bool(compute_bbox):
        ys, xs = np.nonzero(arr)
        y1 = int(np.min(ys))
        y2 = int(np.max(ys)) + 1
        x1 = int(np.min(xs))
        x2 = int(np.max(xs)) + 1
        bbox = (x1, y1, x2, y2)
    return {
        "mask": arr,
        "mask_bool": arr if arr.dtype == bool else None,
        "touches_border": bool(arr[0, :].any() or arr[-1, :].any() or arr[:, 0].any() or arr[:, -1].any()),
        "bbox": bbox,
        "dt": None,
    }


def _runtime_mask_bool(runtime: dict | None):
    if not isinstance(runtime, dict):
        return None
    arr_bool = runtime.get("mask_bool", None)
    if arr_bool is not None:
        arr_bool = np.asarray(arr_bool).astype(bool, copy=False)
        if arr_bool.ndim == 2 and arr_bool.size > 0:
            runtime["mask_bool"] = arr_bool
            return arr_bool
    arr = np.asarray(runtime.get("mask", None))
    if arr.ndim != 2 or arr.size == 0:
        return None
    arr_bool = arr.astype(bool, copy=False)
    if arr_bool.ndim != 2 or arr_bool.size == 0:
        return None
    runtime["mask_bool"] = arr_bool
    return arr_bool


def _runtime_bbox(runtime: dict | None) -> tuple[int, int, int, int] | None:
    if not isinstance(runtime, dict):
        return None
    bbox = runtime.get("bbox", None)
    if isinstance(bbox, tuple) and len(bbox) == 4:
        return (
            int(bbox[0]),
            int(bbox[1]),
            int(bbox[2]),
            int(bbox[3]),
        )
    mask = np.asarray(runtime.get("mask", None))
    if mask.ndim != 2 or mask.size == 0 or not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    y1 = int(np.min(ys))
    y2 = int(np.max(ys)) + 1
    x1 = int(np.min(xs))
    x2 = int(np.max(xs)) + 1
    bbox = (x1, y1, x2, y2)
    runtime["bbox"] = bbox
    return bbox


def _signed_mask_gap_px(mask_a, mask_b, *, runtime_a: dict | None = None, runtime_b: dict | None = None) -> tuple[float | None, bool]:
    ra = runtime_a if isinstance(runtime_a, dict) else prepare_relation_mask_runtime(mask_a)
    rb = runtime_b if isinstance(runtime_b, dict) else prepare_relation_mask_runtime(mask_b)
    if ra is None or rb is None:
        return None, False

    ma = _runtime_mask_bool(ra)
    mb = _runtime_mask_bool(rb)
    if ma is None or mb is None or ma.ndim != 2 or mb.ndim != 2 or ma.shape != mb.shape or ma.size == 0:
        return None, False

    bbox_a = _runtime_bbox(ra)
    bbox_b = _runtime_bbox(rb)
    if (
        isinstance(bbox_a, tuple)
        and len(bbox_a) == 4
        and isinstance(bbox_b, tuple)
        and len(bbox_b) == 4
    ):
        x1 = max(0, min(int(bbox_a[0]), int(bbox_b[0])))
        y1 = max(0, min(int(bbox_a[1]), int(bbox_b[1])))
        x2 = min(ma.shape[1], max(int(bbox_a[2]), int(bbox_b[2])))
        y2 = min(ma.shape[0], max(int(bbox_a[3]), int(bbox_b[3])))
    else:
        x1, y1 = 0, 0
        y2, x2 = ma.shape

    if x1 >= x2 or y1 >= y2:
        return None, False

    ma_view = ma[y1:y2, x1:x2]
    mb_view = mb[y1:y2, x1:x2]

    inter = np.logical_and(ma_view, mb_view)
    if inter.any():
        overlap_px = float(np.count_nonzero(inter))
        return -math.sqrt(max(0.0, overlap_px)), True

    # Exact gap only depends on the support pixels of `ma`, all of which are
    # contained in bbox_a and therefore inside the ROI union [x1:x2, y1:y2].
    # Running the distance transform on the cropped ROI is exact here and much
    # cheaper than transforming the full frame for every close pair.
    dt_view = cv2.distanceTransform((~ma_view).astype(np.uint8), cv2.DIST_L2, 3)
    vals = dt_view[mb_view]
    if vals.size <= 0:
        return None, False
    gap = float(vals.min())
    return gap, True


def compute_relation_observation(
    geom_a: dict | None,
    geom_b: dict | None,
    *,
    scale_min: float,
    contact_margin_px: float = 2.0,
    near_thresh_n: float = 1.25,
    exact_gap_max_n: float = 1.75,
    mask_runtime_a: dict | None = None,
    mask_runtime_b: dict | None = None,
) -> dict | None:
    if not isinstance(geom_a, dict) or not isinstance(geom_b, dict):
        return None

    c_a = geom_a.get("center", None)
    c_b = geom_b.get("center", None)
    area_a = geom_a.get("area", None)
    area_b = geom_b.get("area", None)
    if c_a is None or c_b is None or area_a is None or area_b is None:
        return None

    ax, ay = float(c_a[0]), float(c_a[1])
    bx, by = float(c_b[0]), float(c_b[1])
    dx = float(ax - bx)
    dy = float(ay - by)
    d_center = float(math.sqrt(dx * dx + dy * dy))

    sa = math.sqrt(max(0.0, float(area_a)))
    sb = math.sqrt(max(0.0, float(area_b)))
    scale = float(max(float(scale_min), 0.5 * (sa + sb)))
    center_dist_n = float(d_center / max(1e-6, scale))

    bbox_a = geom_a.get("bbox", None)
    bbox_b = geom_b.get("bbox", None)
    mask_a = geom_a.get("mask", None)
    mask_b = geom_b.get("mask", None)
    border_touch = bool(
        (bool(mask_runtime_a.get("touches_border", False)) if isinstance(mask_runtime_a, dict) else mask_touches_border(mask_a))
        or (bool(mask_runtime_b.get("touches_border", False)) if isinstance(mask_runtime_b, dict) else mask_touches_border(mask_b))
    )
    bbox_metrics_inline = (
        isinstance(bbox_a, (tuple, list))
        and len(bbox_a) >= 4
        and isinstance(bbox_b, (tuple, list))
        and len(bbox_b) >= 4
    )
    if bbox_metrics_inline:
        ax1, ay1, ax2, ay2 = [float(v) for v in bbox_a[:4]]
        bx1, by1, bx2, by2 = [float(v) for v in bbox_b[:4]]

        sep_x = max(0.0, max(ax1 - bx2, bx1 - ax2))
        sep_y = max(0.0, max(ay1 - by2, by1 - ay2))
        overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))

        if sep_x <= 0.0 and sep_y <= 0.0:
            if overlap_w > 0.0 and overlap_h > 0.0:
                bbox_gap_px = -math.sqrt(overlap_w * overlap_h)
            else:
                bbox_gap_px = 0.0
        else:
            bbox_gap_px = math.sqrt(sep_x * sep_x + sep_y * sep_y)

        proj_gap_x_px = float(_bbox_axis_gap_px(ax1, ax2, bx1, bx2))
        proj_gap_y_px = float(_bbox_axis_gap_px(ay1, ay2, by1, by2))
        inter_bbox = float(overlap_w * overlap_h)
        center_a_in_b = bool(bx1 <= ax <= bx2 and by1 <= ay <= by2)
        center_b_in_a = bool(ax1 <= bx <= ax2 and ay1 <= by <= ay2)
    else:
        bbox_gap_px = _bbox_gap_px(bbox_a, bbox_b)
        proj_gap_x_px, proj_gap_y_px = _bbox_projected_gaps_px(bbox_a, bbox_b)
        inter_bbox = float(_bbox_intersection_area(bbox_a, bbox_b))
        center_a_in_b = bool(_bbox_contains_point(bbox_b, ax, ay))
        center_b_in_a = bool(_bbox_contains_point(bbox_a, bx, by))

    bbox_gap_n = float(bbox_gap_px / max(1e-6, scale))
    proj_gap_x_n = float(proj_gap_x_px / max(1e-6, scale)) if proj_gap_x_px is not None else 0.0
    proj_gap_y_n = float(proj_gap_y_px / max(1e-6, scale)) if proj_gap_y_px is not None else 0.0

    # Use the true mask-to-mask distance whenever both masks are available,
    # except when the boxes are already clearly separated beyond the exact-gap
    # operating range. In that regime, bbox gap is the intended approximation
    # controlled by `exact_gap_max_n`.
    use_exact_gap = bool(mask_a is not None and mask_b is not None)
    skipped_exact_due_far = False
    if use_exact_gap and bbox_gap_px > 0.0 and bbox_gap_n > float(exact_gap_max_n):
        use_exact_gap = False
        skipped_exact_due_far = True

    gap_px = None
    used_mask = False
    if use_exact_gap:
        gap_px, used_mask = _signed_mask_gap_px(
            mask_a,
            mask_b,
            runtime_a=mask_runtime_a,
            runtime_b=mask_runtime_b,
        )
    if gap_px is None:
        gap_px = bbox_gap_px
        used_mask = False
    mask_gap_n = float(gap_px / max(1e-6, scale))

    if used_mask or skipped_exact_due_far:
        gap_quality = 1.0
    else:
        gap_quality = 0.5 if not border_touch else 0.0

    if gap_px < 0.0:
        contact_state = "overlap"
    elif gap_px <= float(contact_margin_px):
        contact_state = "touch"
    elif mask_gap_n <= float(near_thresh_n):
        contact_state = "near"
    else:
        contact_state = "separate"

    area_big = float(max(float(area_a), float(area_b), 1e-6))
    area_small = float(max(1e-6, min(float(area_a), float(area_b))))
    overlap_small = float(inter_bbox / area_small)
    overlap_big = float(inter_bbox / area_big)
    center_inside = bool(center_a_in_b or center_b_in_a)
    size_ratio = float(area_small / area_big)
    support_like = 0.0
    if size_ratio <= 0.6:
        support_like = max(
            float(overlap_small),
            0.75 if center_inside and contact_state in ("overlap", "touch", "near") else 0.0,
            0.60 if overlap_small >= 0.5 and contact_state in ("overlap", "touch", "near") else 0.0,
        )
    support_like = float(max(0.0, min(1.0, support_like)))

    return {
        "center_dist_n": float(center_dist_n),
        "mask_gap_n": float(mask_gap_n),
        "proj_gap_x_n": float(proj_gap_x_n),
        "proj_gap_y_n": float(proj_gap_y_n),
        "gap_valid": bool(gap_quality > 0.0),
        "gap_quality": float(gap_quality),
        "contact_state": str(contact_state),
        "truncation_risk": float(1.0 if border_touch else 0.0),
        "bbox_overlap_small": float(max(0.0, min(1.0, overlap_small))),
        "bbox_overlap_big": float(max(0.0, min(1.0, overlap_big))),
        "center_inside": bool(center_inside),
        "support_like": float(support_like),
    }


def relation_distance_metric(obs: dict | None) -> float:
    if not isinstance(obs, dict):
        return float("inf")
    if bool(obs.get("gap_valid", False)):
        return float(obs.get("mask_gap_n", float("inf")))
    return float(obs.get("center_dist_n", float("inf")))


class RunningStat:
    def __init__(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, x: float) -> None:
        x = float(x)
        self.count += 1
        delta = x - self.mean
        self.mean += delta / float(self.count)
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def variance(self, floor: float = 0.0) -> float:
        if self.count <= 1:
            return float(max(0.0, floor))
        return float(max(floor, self.m2 / float(max(1, self.count - 1))))

    def std(self, floor: float = 0.0) -> float:
        return float(math.sqrt(max(0.0, self.variance(floor))))


class PairOrderStat:
    def __init__(self, first_id: int, second_id: int) -> None:
        self.first_id = int(first_id)
        self.second_id = int(second_id)
        self.count = 0
        self.first_before_count = 0
        self.margin_stat = RunningStat()
        self.rank_gap_stat = RunningStat()

    def update(self, *, first_before_second: bool, margin: float, rank_gap: float) -> None:
        self.count += 1
        if bool(first_before_second):
            self.first_before_count += 1
        self.margin_stat.update(max(0.0, float(margin)))
        self.rank_gap_stat.update(max(0.0, float(rank_gap)))

    def probability(self, *, first_before_second: bool) -> float:
        if self.count <= 0:
            return 0.0
        p_first = float(self.first_before_count) / float(max(1, self.count))
        return float(p_first if bool(first_before_second) else (1.0 - p_first))

    def consistency(self) -> float:
        if self.count <= 0:
            return 0.0
        p_first = float(self.first_before_count) / float(max(1, self.count))
        return float(max(p_first, 1.0 - p_first))

    def mean_margin(self) -> float:
        if self.margin_stat.count <= 0:
            return 0.0
        return float(self.margin_stat.mean)

    def margin_std(self) -> float:
        return float(self.margin_stat.std())

    def mean_rank_gap(self) -> float:
        if self.rank_gap_stat.count <= 0:
            return 0.0
        return float(self.rank_gap_stat.mean)

    def reliability(self) -> float:
        return float(min(1.0, math.sqrt(float(self.count) / 6.0)))


class RelationEdge:
    def __init__(self, dst_id: int):
        self.dst_id = int(dst_id)
        self.cooccurrence_count = 0
        self.last_seen_episode = -1
        self.last_seen_ts = 0.0
        self.gap_stat = RunningStat()
        self.center_stat = RunningStat()
        self.rank_stat = RunningStat()
        self.support_like_stat = RunningStat()
        self.contact_counts = {"overlap": 0, "touch": 0, "near": 0, "separate": 0}

    def total_count(self) -> int:
        return int(self.cooccurrence_count)

    def mean_gap(self) -> float | None:
        return float(self.gap_stat.mean) if self.gap_stat.count > 0 else None

    def mean_center(self) -> float | None:
        return float(self.center_stat.mean) if self.center_stat.count > 0 else None

    def mean_rank(self) -> float | None:
        return float(self.rank_stat.mean) if self.rank_stat.count > 0 else None

    def mean_support_like(self) -> float:
        if self.support_like_stat.count <= 0:
            return 0.0
        return float(self.support_like_stat.mean)

    def primary_distance(self) -> float | None:
        gap = self.mean_gap()
        if gap is not None:
            return float(gap)
        center = self.mean_center()
        return None if center is None else float(center)

    def contact_probability(self, state: str) -> float:
        total = int(sum(int(v) for v in (self.contact_counts or {}).values()))
        if total <= 0:
            return 0.0
        key = str(state or "separate").lower()
        if key not in self.contact_counts:
            key = "separate"
        return float(self.contact_counts.get(key, 0) / float(total))

    def reliability(self) -> float:
        return float(min(1.0, math.sqrt(float(self.cooccurrence_count) / 6.0)))

    def informativeness(self) -> float:
        support_like = float(self.mean_support_like())
        return float(max(0.10, 1.0 - (0.80 * support_like)))


class NeighborDistanceGraph:
    """
    Memoria relacional simple orientada a desambiguación.

    No guarda vistas ni modos geométricos complejos. Consolida por episodios:
      - gap mínimo entre máscaras/cajas
      - distancia entre centros
      - ranking de cercanía dentro del contexto estable
      - patrón de contacto
      - soporte/inside-like
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.var_floor = max(1e-8, float(cfg.get("var_floor", 0.0225)))
        self.scale_min = max(1.0, float(cfg.get("scale_min", 40.0)))
        self.contact_margin_px = max(0.0, float(cfg.get("contact_margin_px", 2.0)))
        self.near_thresh_n = max(0.0, float(cfg.get("near_thresh_n", 1.25)))
        self.exact_gap_max_n = max(0.0, float(cfg.get("exact_gap_max_n", 1.75)))

        self.edges: dict[int, RelationEdge] = {}
        self.pending_obs: dict[int, dict[str, list]] = {}
        self.pair_order_stats: dict[tuple[int, int], PairOrderStat] = {}

    def clear_pending(self) -> None:
        self.pending_obs = {}

    def pair_key(self, first_id: int, second_id: int) -> tuple[int, int]:
        a_id = int(first_id)
        b_id = int(second_id)
        return (a_id, b_id) if a_id <= b_id else (b_id, a_id)

    def get_pair_order_stat(self, first_id: int, second_id: int) -> PairOrderStat | None:
        return self.pair_order_stats.get(self.pair_key(first_id, second_id), None)

    def add_pending_observation(self, other_id: int, obs: dict | None) -> None:
        if not isinstance(obs, dict):
            return
        oid = int(other_id)
        entry = self.pending_obs.setdefault(
            oid,
            {
                "obs": [],
            },
        )
        entry["obs"].append(dict(obs))

    def observe_frame(self, self_id: int, visible_object_ids: list[int], geom_by_object_id: dict) -> None:
        if not self.enabled:
            return

        sid = int(self_id)
        geom_self = geom_by_object_id.get(sid)
        if not isinstance(geom_self, dict):
            return

        for oid in visible_object_ids or []:
            oid = int(oid)
            if oid == sid:
                continue
            geom_other = geom_by_object_id.get(oid)
            obs = compute_relation_observation(
                geom_self,
                geom_other,
                scale_min=float(self.scale_min),
                contact_margin_px=float(self.contact_margin_px),
                near_thresh_n=float(self.near_thresh_n),
                exact_gap_max_n=float(self.exact_gap_max_n),
            )
            self.add_pending_observation(oid, obs)

    def accept_episode(
        self,
        stable_context: set[int],
        timestamp: float,
        episode_idx: int,
        view_id: int | None = None,
    ) -> None:
        if not self.enabled:
            self.clear_pending()
            return

        del view_id

        per_other = {}
        for oid in stable_context or set():
            oid = int(oid)
            entry = self.pending_obs.get(oid, None)
            if not isinstance(entry, dict):
                continue
            obs_list = [dict(x) for x in (entry.get("obs", []) or []) if isinstance(x, dict)]
            if not obs_list:
                continue

            gap_vals = [float(x.get("mask_gap_n", 0.0)) for x in obs_list if bool(x.get("gap_valid", False))]
            center_vals = [float(x.get("center_dist_n", 0.0)) for x in obs_list]
            support_vals = [float(x.get("support_like", 0.0)) for x in obs_list]
            metric_vals = [relation_distance_metric(x) for x in obs_list]
            metric_vals = [float(x) for x in metric_vals if math.isfinite(float(x))]
            if not center_vals or not metric_vals:
                continue

            contact_counts = {}
            for obs in obs_list:
                state = str(obs.get("contact_state", "separate")).lower()
                if state not in ("overlap", "touch", "near", "separate"):
                    state = "separate"
                contact_counts[state] = int(contact_counts.get(state, 0)) + 1

            per_other[int(oid)] = {
                "gap": None if not gap_vals else float(np.median(np.asarray(gap_vals, dtype=np.float32))),
                "center": float(np.median(np.asarray(center_vals, dtype=np.float32))),
                "support_like": float(np.mean(np.asarray(support_vals, dtype=np.float32))) if support_vals else 0.0,
                "metric": float(np.median(np.asarray(metric_vals, dtype=np.float32))),
                "contact_state": self.majority_state(contact_counts),
            }

        if not per_other:
            self.clear_pending()
            return

        ranked = sorted(
            ((float(item["metric"]), int(oid)) for oid, item in per_other.items()),
            key=lambda kv: (float(kv[0]), int(kv[1])),
        )
        rank_by_oid = {int(oid): int(rank) for rank, (_, oid) in enumerate(ranked, start=1)}

        ranked_metrics = {int(oid): float(item["metric"]) for oid, item in per_other.items()}
        other_ids = sorted(int(oid) for oid in per_other.keys())
        for idx, oid_a in enumerate(other_ids):
            for oid_b in other_ids[idx + 1 :]:
                metric_a = float(ranked_metrics.get(int(oid_a), float("inf")))
                metric_b = float(ranked_metrics.get(int(oid_b), float("inf")))
                if not math.isfinite(metric_a) or not math.isfinite(metric_b):
                    continue
                key = self.pair_key(int(oid_a), int(oid_b))
                stat = self.pair_order_stats.get(key, None)
                if stat is None:
                    stat = PairOrderStat(first_id=int(key[0]), second_id=int(key[1]))
                    self.pair_order_stats[key] = stat
                first_before_second = bool(
                    float(metric_a) < float(metric_b)
                    if key[0] == int(oid_a)
                    else float(metric_b) < float(metric_a)
                )
                margin = float(abs(float(metric_a) - float(metric_b)))
                rank_gap = float(abs(int(rank_by_oid.get(int(oid_a), 0)) - int(rank_by_oid.get(int(oid_b), 0))))
                stat.update(
                    first_before_second=bool(first_before_second),
                    margin=float(margin),
                    rank_gap=float(rank_gap),
                )

        ep = int(episode_idx)
        ts = float(timestamp)
        for oid, item in per_other.items():
            edge = self.edges.get(int(oid), None)
            if edge is None:
                edge = RelationEdge(dst_id=int(oid))
                self.edges[int(oid)] = edge
            self.update_edge(
                edge=edge,
                gap=float(item["gap"]) if item["gap"] is not None else None,
                center=float(item["center"]),
                rank=float(rank_by_oid.get(int(oid), 0)),
                contact_state=str(item["contact_state"]),
                support_like=float(item["support_like"]),
                episode_idx=ep,
                timestamp=ts,
            )

        self.clear_pending()

    def get_edge(self, dst_id: int, view_id: int | None = None) -> RelationEdge | None:
        del view_id
        return self.edges.get(int(dst_id), None)

    def majority_state(self, counts: dict[str, int]) -> str:
        if not counts:
            return "separate"
        items = sorted(counts.items(), key=lambda kv: (int(kv[1]), str(kv[0])), reverse=True)
        return str(items[0][0]) if items else "separate"

    def update_edge(
        self,
        *,
        edge: RelationEdge,
        gap: float | None,
        center: float,
        rank: float,
        contact_state: str,
        support_like: float,
        episode_idx: int,
        timestamp: float,
    ) -> None:
        if gap is not None:
            edge.gap_stat.update(float(gap))
        edge.center_stat.update(float(center))
        if float(rank) > 0.0:
            edge.rank_stat.update(float(rank))
        edge.support_like_stat.update(float(max(0.0, min(1.0, support_like))))

        state = str(contact_state or "separate").lower()
        if state not in edge.contact_counts:
            state = "separate"
        edge.contact_counts[state] = int(edge.contact_counts.get(state, 0)) + 1

        edge.cooccurrence_count += 1
        edge.last_seen_episode = int(episode_idx)
        edge.last_seen_ts = float(timestamp)
