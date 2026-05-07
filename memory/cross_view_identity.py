from __future__ import annotations

import math


def zone_key_from_support_bbox(
    support_bbox,
    center_xy,
    *,
    rows: int = 3,
    cols: int = 3,
) -> str | None:
    if support_bbox is None or center_xy is None or len(support_bbox) < 4:
        return None
    rows = max(1, int(rows))
    cols = max(1, int(cols))

    x1, y1, x2, y2 = [float(v) for v in support_bbox[:4]]
    cx, cy = float(center_xy[0]), float(center_xy[1])
    w = max(1e-6, float(x2 - x1))
    h = max(1e-6, float(y2 - y1))

    u = max(0.0, min(0.999999, float((cx - x1) / w)))
    v = max(0.0, min(0.999999, float((cy - y1) / h)))
    col = int(min(cols - 1, max(0, math.floor(u * cols))))
    row = int(min(rows - 1, max(0, math.floor(v * rows))))
    return f"r{row}c{col}"


def relation_affinity(obs: dict | None) -> float:
    if not isinstance(obs, dict):
        return 0.0
    support_like = float(max(0.0, min(1.0, obs.get("support_like", 0.0))))
    center_dist = float(max(0.0, obs.get("center_dist_n", 0.0)))
    gap = float(max(0.0, obs.get("mask_gap_n", 0.0)))
    contact_state = str(obs.get("contact_state", "separate"))

    center_score = float(math.exp(-0.5 * min(4.0, center_dist)))
    gap_score = float(math.exp(-0.6 * min(4.0, gap)))
    contact_score = {
        "overlap": 1.0,
        "touch": 0.90,
        "near": 0.70,
        "separate": 0.35,
    }.get(contact_state, 0.35)
    return float(
        max(
            support_like,
            0.35 * center_score + 0.40 * gap_score + 0.25 * contact_score,
        )
    )


class CrossViewIdentity:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.grid_rows = max(1, int(cfg.get("grid_rows", 3)))
        self.grid_cols = max(1, int(cfg.get("grid_cols", 3)))
        self.max_anchor_rank = max(1, int(cfg.get("max_anchor_rank", 3)))
        self.max_support_neighbor_rank = max(1, int(cfg.get("max_support_neighbor_rank", 3)))
        self.min_support_like = max(0.0, min(1.0, float(cfg.get("min_support_like", 0.35))))
        self.conf_tau = max(1e-6, float(cfg.get("conf_tau", 4.0)))

        self.support_hist: dict[int, float] = {}
        self.support_oid: int | None = None
        self.support_conf: float = 0.0

        self.zone_hist: dict[str, float] = {}
        self.anchor_rank_hist: dict[int, dict[int, float]] = {}
        self.anchor_near_hist: dict[int, float] = {}
        self.support_neighbor_rank_hist: dict[int, dict[int, float]] = {}

        self.on_support_sum: float = 0.0
        self.on_support_count: int = 0
        self.inside_support_sum: float = 0.0
        self.inside_support_count: int = 0

        self.obs_count: int = 0
        self.last_episode: int = -1

    def _bump_hist(self, hist: dict, key, weight: float) -> None:
        if key is None or float(weight) <= 0.0:
            return
        hist[key] = float(hist.get(key, 0.0) + float(weight))

    def _bump_rank_hist(self, hist: dict[int, dict[int, float]], oid: int, rank: int, weight: float) -> None:
        if oid is None or rank <= 0 or float(weight) <= 0.0:
            return
        per_oid = hist.setdefault(int(oid), {})
        per_oid[int(rank)] = float(per_oid.get(int(rank), 0.0) + float(weight))

    def _update_support_summary(self) -> None:
        if not self.support_hist:
            self.support_oid = None
            self.support_conf = 0.0
            return
        total = float(sum(float(v) for v in self.support_hist.values()))
        if total <= 1e-12:
            self.support_oid = None
            self.support_conf = 0.0
            return
        oid, mass = max(self.support_hist.items(), key=lambda kv: float(kv[1]))
        self.support_oid = int(oid)
        self.support_conf = float(mass / total)

    def mean_on_support(self) -> float:
        if self.on_support_count <= 0:
            return 0.0
        return float(self.on_support_sum / float(self.on_support_count))

    def mean_inside_support(self) -> float:
        if self.inside_support_count <= 0:
            return 0.0
        return float(self.inside_support_sum / float(self.inside_support_count))

    def observe(
        self,
        *,
        support_oid: int | None,
        support_like: float,
        zone_key: str | None,
        anchor_order: list[int] | None,
        support_neighbor_order: list[int] | None,
        on_support_like: float,
        inside_support_like: float,
        episode_idx: int,
    ) -> None:
        if not self.enabled:
            return

        support_like = float(max(0.0, min(1.0, support_like)))
        on_support_like = float(max(0.0, min(1.0, on_support_like)))
        inside_support_like = float(max(0.0, min(1.0, inside_support_like)))

        if support_oid is not None and support_like >= float(self.min_support_like):
            self._bump_hist(self.support_hist, int(support_oid), float(support_like))
            self.on_support_sum += float(on_support_like)
            self.on_support_count += 1
            self.inside_support_sum += float(inside_support_like)
            self.inside_support_count += 1
            if zone_key:
                self._bump_hist(self.zone_hist, str(zone_key), float(support_like))

        for rank, oid in enumerate(list(anchor_order or [])[: self.max_anchor_rank], start=1):
            w = float(1.0 / float(rank))
            self._bump_rank_hist(self.anchor_rank_hist, int(oid), int(rank), w)
            self._bump_hist(self.anchor_near_hist, int(oid), w)

        for rank, oid in enumerate(list(support_neighbor_order or [])[: self.max_support_neighbor_rank], start=1):
            w = float(1.0 / float(rank))
            self._bump_rank_hist(self.support_neighbor_rank_hist, int(oid), int(rank), w)

        self.obs_count += 1
        self.last_episode = int(episode_idx)
        self._update_support_summary()

    def _rank_match(self, rank_hist: dict[int, dict[int, float]], observed_order: list[int] | None, max_rank: int) -> tuple[float, float]:
        order = [int(x) for x in (observed_order or [])[: max_rank] if x is not None]
        if not order:
            return 0.5, 0.0

        vals = []
        hits = 0
        for rank, oid in enumerate(order, start=1):
            per_oid = rank_hist.get(int(oid), None)
            if not isinstance(per_oid, dict) or not per_oid:
                continue
            den = float(sum(float(v) for v in per_oid.values()))
            if den <= 1e-12:
                continue
            vals.append(float(per_oid.get(int(rank), 0.0) / den))
            hits += 1
        if not vals:
            return 0.5, 0.0
        coverage = float(hits / max(1, len(order)))
        return float(sum(vals) / len(vals)), float(coverage)

    def score_observation(
        self,
        *,
        support_oid: int | None,
        zone_key: str | None,
        anchor_order: list[int] | None,
        support_neighbor_order: list[int] | None,
        on_support_like: float,
        inside_support_like: float,
    ) -> dict:
        if not self.enabled or int(self.obs_count) <= 0:
            return {
                "score": 0.0,
                "reliability": 0.0,
                "support": 0.0,
                "zone": 0.0,
                "anchors": 0.0,
                "neighbors": 0.0,
                "topology": 0.0,
            }

        comps = []
        comp_cov = []

        if self.support_oid is not None and support_oid is not None:
            support_score = 1.0 if int(self.support_oid) == int(support_oid) else 0.0
            comps.append((float(support_score), 0.20))
            comp_cov.append(1.0)
        else:
            support_score = 0.5

        zone_total = float(sum(float(v) for v in self.zone_hist.values()))
        if zone_total > 1e-12 and zone_key is not None:
            zone_score = float(self.zone_hist.get(str(zone_key), 0.0) / zone_total)
            comps.append((float(zone_score), 0.35))
            comp_cov.append(1.0)
        else:
            zone_score = 0.5

        anchor_score, anchor_cov = self._rank_match(self.anchor_rank_hist, anchor_order, self.max_anchor_rank)
        comps.append((float(anchor_score), 0.20))
        comp_cov.append(float(anchor_cov))

        neighbor_score, neighbor_cov = self._rank_match(
            self.support_neighbor_rank_hist,
            support_neighbor_order,
            self.max_support_neighbor_rank,
        )
        comps.append((float(neighbor_score), 0.20))
        comp_cov.append(float(neighbor_cov))

        topo_vals = []
        if self.on_support_count > 0:
            topo_vals.append(1.0 - abs(float(on_support_like) - float(self.mean_on_support())))
        if self.inside_support_count > 0:
            topo_vals.append(1.0 - abs(float(inside_support_like) - float(self.mean_inside_support())))
        topology_score = float(sum(topo_vals) / len(topo_vals)) if topo_vals else 0.5
        comps.append((float(topology_score), 0.05))
        comp_cov.append(1.0 if topo_vals else 0.0)

        num = float(sum(v * w for v, w in comps))
        den = float(sum(w for _, w in comps))
        score = float(num / max(1e-12, den))

        hist_conf = float(1.0 - math.exp(-float(self.obs_count) / float(self.conf_tau)))
        coverage = float(sum(comp_cov) / max(1, len(comp_cov)))
        reliability = float(hist_conf * coverage)
        return {
            "score": float(score),
            "reliability": float(reliability),
            "support": float(support_score),
            "zone": float(zone_score),
            "anchors": float(anchor_score),
            "neighbors": float(neighbor_score),
            "topology": float(topology_score),
        }

    def summary(self) -> dict:
        top_zone = None
        if self.zone_hist:
            top_zone = max(self.zone_hist.items(), key=lambda kv: float(kv[1]))[0]
        return {
            "enabled": bool(self.enabled),
            "support_oid": int(self.support_oid) if self.support_oid is not None else None,
            "support_conf": float(self.support_conf),
            "top_zone": top_zone,
            "obs_count": int(self.obs_count),
            "last_episode": int(self.last_episode),
        }
