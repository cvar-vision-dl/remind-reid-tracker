from __future__ import annotations

import itertools
import math

from memory.neighbor_distance_graph import compute_relation_observation


class PairAnchorDiscriminator:
    """
    Estadística histórica por pareja (obj_a, obj_b) y anchor.

    Se centra en responder:
      - cuánto separa históricamente un anchor a una pareja;
      - cuán consistente es esa separación;
      - qué resumen explicable podemos enseñar en debug.
    """

    def __init__(self, *, config: dict, memory_store) -> None:
        self.config = config or {}
        self.memory_store = memory_store

        self.min_edge_reliability = max(0.0, min(1.0, float(self.config.get("min_edge_reliability", 0.15))))
        self.min_anchor_informativeness = max(0.0, min(1.0, float(self.config.get("min_anchor_informativeness", 0.10))))
        self.anchor_span_ref = max(1e-6, float(self.config.get("anchor_span_ref", 0.20)))
        self.order_margin_ref = max(1e-6, float(self.config.get("order_margin_ref", self.anchor_span_ref)))
        self.support_penalty = max(0.0, min(1.0, float(self.config.get("support_penalty", 0.80))))
        self.anchor_pair_margin_weight = max(0.0, min(1.0, float(self.config.get("anchor_pair_margin_weight", 0.35))))
        self.anchor_pair_min_consistency = max(0.0, min(1.0, float(self.config.get("anchor_pair_min_consistency", 0.60))))
        self.anchor_pair_min_margin = max(0.0, float(self.config.get("anchor_pair_min_margin", 0.03)))
        self.anchor_pair_topk = max(1, int(self.config.get("anchor_pair_topk", 3)))
        self.debug_historical_anchor_topk = max(0, int(self.config.get("debug_historical_anchor_topk", 5)))
        self.distance_score_power = max(1.0, float(self.config.get("distance_score_power", 2.8)))
        self.distance_score_scale = max(1e-6, float(self.config.get("distance_score_scale", 1.10)))
        self.consistency_floor = max(0.0, min(1.0, float(self.config.get("consistency_floor", 0.45))))
        self.consistency_power = max(0.1, float(self.config.get("consistency_power", 0.75)))
        self.relation_observation_fn = None

    def distance_strength(self, value: float, *, ref: float) -> float:
        scaled_ref = max(1e-6, float(ref) * float(self.distance_score_scale))
        x = max(0.0, float(value)) / scaled_ref
        return float((1.0 - math.exp(-x)) ** float(self.distance_score_power))

    def softened_consistency(self, consistency: float) -> float:
        c = max(0.0, min(1.0, float(consistency)))
        curved = float(c ** float(self.consistency_power))
        return float(self.consistency_floor + ((1.0 - float(self.consistency_floor)) * curved))

    def observe_relation(
        self,
        geom_a: dict | None,
        geom_b: dict | None,
        *,
        scale_min: float,
        contact_margin_px: float = 2.0,
        near_thresh_n: float = 1.25,
        exact_gap_max_n: float = 1.75,
        geom_a_key=None,
        geom_b_key=None,
    ) -> dict | None:
        fn = getattr(self, "relation_observation_fn", None)
        if callable(fn):
            return fn(
                geom_a,
                geom_b,
                scale_min=float(scale_min),
                contact_margin_px=float(contact_margin_px),
                near_thresh_n=float(near_thresh_n),
                exact_gap_max_n=float(exact_gap_max_n),
                geom_a_key=geom_a_key,
                geom_b_key=geom_b_key,
            )
        return compute_relation_observation(
            geom_a,
            geom_b,
            scale_min=float(scale_min),
            contact_margin_px=float(contact_margin_px),
            near_thresh_n=float(near_thresh_n),
            exact_gap_max_n=float(exact_gap_max_n),
        )

    def anchor_pair_stat(self, *, anchor_id: int, oid_a: int, oid_b: int):
        anchor_obj = self.memory_store.get(int(anchor_id)) if self.memory_store is not None else None
        anchor_dg = getattr(anchor_obj, "neighbor_dist", None) if anchor_obj is not None else None
        if anchor_dg is None:
            return None
        return anchor_dg.get_pair_order_stat(int(oid_a), int(oid_b))

    def pair_order_stats(self, *, anchor_id: int, oid_a: int, oid_b: int) -> dict | None:
        stat = self.anchor_pair_stat(anchor_id=int(anchor_id), oid_a=int(oid_a), oid_b=int(oid_b))
        if stat is None or int(getattr(stat, "count", 0) or 0) <= 0:
            return None
        consistency = float(stat.consistency())
        mean_margin = float(stat.mean_margin())
        margin_std = float(stat.margin_std())
        reliability = float(stat.reliability())
        stable_margin = float(mean_margin / max(1e-6, mean_margin + margin_std))
        margin_norm = float(self.distance_strength(float(mean_margin), ref=float(self.order_margin_ref)))
        consistency_term = float(self.softened_consistency(float(consistency)))
        robustness = float(
            consistency_term
            * reliability
            * (
                ((1.0 - float(self.anchor_pair_margin_weight)) * stable_margin)
                + (float(self.anchor_pair_margin_weight) * margin_norm)
            )
        )
        return {
            "consistency": float(consistency),
            "mean_margin": float(mean_margin),
            "margin_std": float(margin_std),
            "reliability": float(reliability),
            "stable_margin": float(stable_margin),
            "margin_norm": float(margin_norm),
            "robustness": float(robustness),
        }

    def pair_order_probability(self, *, anchor_id: int, closer_oid: int, farther_oid: int) -> float | None:
        stat = self.anchor_pair_stat(anchor_id=int(anchor_id), oid_a=int(closer_oid), oid_b=int(farther_oid))
        if stat is None or int(getattr(stat, "count", 0) or 0) <= 0:
            return None
        key_first = int(getattr(stat, "first_id", -1))
        if int(closer_oid) == int(key_first):
            return float(stat.probability(first_before_second=True))
        if int(farther_oid) == int(key_first):
            return float(stat.probability(first_before_second=False))
        return None

    def anchor_informativeness(self, *, anchor_id: int, candidate_ids: list[int]) -> float:
        values = []
        support_like = []
        reliabilities = []
        for oid in candidate_ids or []:
            obj = self.memory_store.get(int(oid)) if self.memory_store is not None else None
            dg = getattr(obj, "neighbor_dist", None) if obj is not None else None
            edge = None if dg is None else dg.get_edge(int(anchor_id))
            if edge is None:
                continue
            reliability = float(edge.reliability())
            if reliability < float(self.min_edge_reliability):
                continue
            gap = edge.mean_gap()
            if gap is None or not math.isfinite(float(gap)):
                continue
            values.append(float(gap))
            support_like.append(float(edge.mean_support_like()))
            reliabilities.append(float(reliability))

        if len(values) < 2:
            if len(values) == 1 and reliabilities:
                support_pen = float(sum(support_like) / float(max(1, len(support_like))))
                rel = float(sum(reliabilities) / float(max(1, len(reliabilities))))
                return float(0.50 * rel * max(0.10, 1.0 - float(self.support_penalty) * support_pen))
            return 0.0

        span = float(max(values) - min(values))
        discr = float(self.distance_strength(float(span), ref=float(self.anchor_span_ref)))
        support_pen = float(sum(support_like) / float(max(1, len(support_like))))
        rel = float(sum(reliabilities) / float(max(1, len(reliabilities))))
        return float(discr * rel * max(0.10, 1.0 - float(self.support_penalty) * support_pen))

    def anchor_pair_usefulness(self, *, anchor_id: int, candidate_ids: list[int]) -> float:
        candidate_ids = sorted(set(int(x) for x in (candidate_ids or [])))
        pair_scores = []
        for oid_a, oid_b in itertools.combinations(candidate_ids, 2):
            obj_a = self.memory_store.get(int(oid_a)) if self.memory_store is not None else None
            obj_b = self.memory_store.get(int(oid_b)) if self.memory_store is not None else None
            dg_a = getattr(obj_a, "neighbor_dist", None) if obj_a is not None else None
            dg_b = getattr(obj_b, "neighbor_dist", None) if obj_b is not None else None
            edge_a = None if dg_a is None else dg_a.get_edge(int(anchor_id))
            edge_b = None if dg_b is None else dg_b.get_edge(int(anchor_id))
            gap_a = None if edge_a is None else edge_a.mean_gap()
            gap_b = None if edge_b is None else edge_b.mean_gap()
            if gap_a is None or gap_b is None:
                continue
            if not all(math.isfinite(float(x)) for x in (gap_a, gap_b)):
                continue
            stats = self.pair_order_stats(anchor_id=int(anchor_id), oid_a=int(oid_a), oid_b=int(oid_b))
            if not isinstance(stats, dict):
                continue
            if float(stats.get("consistency", 0.0)) < float(self.anchor_pair_min_consistency):
                continue
            pair_scores.append(float(stats.get("robustness", 0.0)))
        if not pair_scores:
            return 0.0
        pair_scores.sort(reverse=True)
        top_scores = pair_scores[: max(1, min(int(self.anchor_pair_topk), len(pair_scores)))]
        return float(sum(float(x) for x in top_scores) / float(max(1, len(top_scores))))

    def anchor_candidate_margin_means(self, *, anchor_id: int, candidate_ids: list[int]) -> dict[str, float | None]:
        gap_margins = []
        candidate_ids = sorted(set(int(x) for x in (candidate_ids or [])))
        for oid_a, oid_b in itertools.combinations(candidate_ids, 2):
            obj_a = self.memory_store.get(int(oid_a)) if self.memory_store is not None else None
            obj_b = self.memory_store.get(int(oid_b)) if self.memory_store is not None else None
            dg_a = getattr(obj_a, "neighbor_dist", None) if obj_a is not None else None
            dg_b = getattr(obj_b, "neighbor_dist", None) if obj_b is not None else None
            edge_a = None if dg_a is None else dg_a.get_edge(int(anchor_id))
            edge_b = None if dg_b is None else dg_b.get_edge(int(anchor_id))
            if edge_a is None or edge_b is None:
                continue
            gap_a = edge_a.mean_gap()
            gap_b = edge_b.mean_gap()
            if gap_a is not None and gap_b is not None and all(math.isfinite(float(x)) for x in (gap_a, gap_b)):
                gap_margins.append(float(abs(float(gap_a) - float(gap_b))))
        return {
            "gap_margin_mean": (float(sum(gap_margins) / float(len(gap_margins))) if gap_margins else None),
        }

    def anchor_candidate_history_profile(self, *, anchor_id: int, candidate_ids: list[int]) -> dict[str, list]:
        candidate_ids = [int(x) for x in (candidate_ids or []) if x is not None]
        anchor_obj = self.memory_store.get(int(anchor_id)) if self.memory_store is not None else None
        anchor_dg = getattr(anchor_obj, "neighbor_dist", None) if anchor_obj is not None else None
        if anchor_dg is None:
            return {"modes": [], "primary": [], "gap": []}

        modes = []
        primary = []
        gap_vals = []
        for oid in candidate_ids:
            edge = anchor_dg.get_edge(int(oid))
            if edge is None:
                continue
            gap = edge.mean_gap()
            if gap is not None and math.isfinite(float(gap)):
                modes.append("M")
            else:
                modes.append("")
            if gap is not None and math.isfinite(float(gap)):
                primary.append(float(gap))
            else:
                primary.append(float("nan"))
            gap_vals.append(float(gap) if gap is not None and math.isfinite(float(gap)) else float("nan"))
        return {
            "modes": list(modes),
            "primary": list(primary),
            "gap": list(gap_vals),
        }

    def anchor_pair_history_summary(self, *, anchor_id: int, candidate_ids: list[int]) -> dict[str, float | int | None]:
        candidate_ids = sorted(set(int(x) for x in (candidate_ids or [])))
        stats_list = []
        for oid_a, oid_b in itertools.combinations(candidate_ids, 2):
            stats = self.pair_order_stats(anchor_id=int(anchor_id), oid_a=int(oid_a), oid_b=int(oid_b))
            if isinstance(stats, dict):
                stats_list.append(dict(stats))
        if not stats_list:
            return {
                "pairs": 0,
                "consistency": None,
                "mean_margin": None,
                "reliability": None,
                "robustness": None,
            }
        return {
            "pairs": int(len(stats_list)),
            "consistency": float(sum(float(x.get("consistency", 0.0) or 0.0) for x in stats_list) / float(len(stats_list))),
            "mean_margin": float(sum(float(x.get("mean_margin", 0.0) or 0.0) for x in stats_list) / float(len(stats_list))),
            "reliability": float(sum(float(x.get("reliability", 0.0) or 0.0) for x in stats_list) / float(len(stats_list))),
            "robustness": float(sum(float(x.get("robustness", 0.0) or 0.0) for x in stats_list) / float(len(stats_list))),
        }

    def historical_anchor_score(self, *, anchor_id: int, candidate_ids: list[int]) -> dict[str, float | None]:
        base_use = float(self.anchor_informativeness(anchor_id=int(anchor_id), candidate_ids=list(candidate_ids)))
        pair_use = float(self.anchor_pair_usefulness(anchor_id=int(anchor_id), candidate_ids=list(candidate_ids)))
        margin_means = self.anchor_candidate_margin_means(anchor_id=int(anchor_id), candidate_ids=list(candidate_ids))
        gap_sep = float(margin_means.get("gap_margin_mean", 0.0) or 0.0)
        sep_term = float(self.distance_strength(float(gap_sep), ref=float(self.anchor_span_ref)))
        score = float((0.58 * base_use) + (0.17 * pair_use) + (0.25 * sep_term))
        return {
            "history_usefulness": float(max(base_use, min(1.0, score))),
            "base_usefulness": float(base_use),
            "pair_usefulness": float(pair_use),
            "sep_term": float(sep_term),
        }

    def describe_anchor_for_pair(
        self,
        *,
        anchor_id: int,
        candidate_ids: list[int],
        det_ids: list[int] | None = None,
        det_geom_by_id: dict[int, dict] | None = None,
        anchor_geom_by_oid: dict[int, dict] | None = None,
    ) -> dict:
        candidate_ids = [int(x) for x in (candidate_ids or []) if x is not None]
        det_ids = [int(x) for x in (det_ids or []) if x is not None]
        det_geom_by_id = dict(det_geom_by_id or {})
        anchor_geom_by_oid = dict(anchor_geom_by_oid or {})

        score_pack = self.historical_anchor_score(anchor_id=int(anchor_id), candidate_ids=list(candidate_ids))
        margin_means = self.anchor_candidate_margin_means(anchor_id=int(anchor_id), candidate_ids=list(candidate_ids))
        hist_profile = self.anchor_candidate_history_profile(anchor_id=int(anchor_id), candidate_ids=list(candidate_ids))
        pair_summary = self.anchor_pair_history_summary(anchor_id=int(anchor_id), candidate_ids=list(candidate_ids))

        obs_modes = []
        obs_primary = []
        obs_gap = []
        anchor_geom = anchor_geom_by_oid.get(int(anchor_id), None)
        if isinstance(anchor_geom, dict):
            for det_id in det_ids:
                det_geom = det_geom_by_id.get(int(det_id), None)
                if not isinstance(det_geom, dict):
                    continue
                obs = self.observe_relation(
                    det_geom,
                    anchor_geom,
                    scale_min=40.0,
                    geom_a_key=("det", int(det_id)),
                    geom_b_key=("anchor", int(anchor_id)),
                )
                if not isinstance(obs, dict):
                    continue
                if not bool(obs.get("gap_valid", False)):
                    continue
                obs_modes.append("M")
                obs_primary.append(float(obs.get("mask_gap_n", float("nan"))))
                obs_gap.append(float(obs.get("mask_gap_n", float("nan"))))

        if not obs_modes:
            obs_modes = list(hist_profile.get("modes", []) or [])
            obs_primary = list(hist_profile.get("primary", []) or [])
            obs_gap = list(hist_profile.get("gap", []) or [])

        return {
            "anchor_id": int(anchor_id),
            "history_usefulness": float(score_pack.get("history_usefulness", 0.0) or 0.0),
            "base_usefulness": float(score_pack.get("base_usefulness", 0.0) or 0.0),
            "pair_usefulness": float(score_pack.get("pair_usefulness", 0.0) or 0.0),
            "obs_modes": list(obs_modes),
            "obs_primary": list(obs_primary),
            "obs_gap": list(obs_gap),
            "gap_margin_mean": margin_means.get("gap_margin_mean", None),
            "pair_consistency": pair_summary.get("consistency", None),
            "pair_margin_mean": pair_summary.get("mean_margin", None),
            "pair_reliability": pair_summary.get("reliability", None),
            "pair_robustness": pair_summary.get("robustness", None),
            "visible_now": bool(int(anchor_id) in anchor_geom_by_oid),
        }

    def rank_historical_anchors(
        self,
        *,
        candidate_ids: list[int],
        det_ids: list[int] | None = None,
        det_geom_by_id: dict[int, dict] | None = None,
        anchor_geom_by_oid: dict[int, dict] | None = None,
        excluded_anchor_ids: set[int] | None = None,
        limit: int | None = None,
    ) -> list[dict]:
        if self.memory_store is None:
            return []

        excluded = {int(x) for x in (excluded_anchor_ids or set())}
        candidate_ids = sorted(set(int(x) for x in (candidate_ids or [])))
        rows = []
        for obj in self.memory_store.all_objects():
            anchor_id = int(getattr(obj, "object_id", -1))
            if anchor_id < 0 or anchor_id in excluded or anchor_id in candidate_ids:
                continue
            desc = self.describe_anchor_for_pair(
                anchor_id=int(anchor_id),
                candidate_ids=list(candidate_ids),
                det_ids=det_ids,
                det_geom_by_id=det_geom_by_id,
                anchor_geom_by_oid=anchor_geom_by_oid,
            )
            rows.append(
                {
                    "anchor_id": int(anchor_id),
                    "rank": None,
                    "usefulness": float(desc.get("history_usefulness", 0.0) or 0.0),
                    "raw_usefulness": float(desc.get("history_usefulness", 0.0) or 0.0),
                    "base_usefulness": float(desc.get("base_usefulness", 0.0) or 0.0),
                    "pair_usefulness": float(desc.get("pair_usefulness", 0.0) or 0.0),
                    "history_usefulness": float(desc.get("history_usefulness", 0.0) or 0.0),
                    "local_usefulness": None,
                    "obs_modes": list(desc.get("obs_modes", []) or []),
                    "obs_primary": list(desc.get("obs_primary", []) or []),
                    "obs_gap": list(desc.get("obs_gap", []) or []),
                    "gap_margin_mean": desc.get("gap_margin_mean", None),
                    "pair_consistency": desc.get("pair_consistency", None),
                    "pair_margin_mean": desc.get("pair_margin_mean", None),
                    "pair_reliability": desc.get("pair_reliability", None),
                    "pair_robustness": desc.get("pair_robustness", None),
                    "local_reason": "hist_only",
                    "selected": False,
                    "valid": bool(float(desc.get("history_usefulness", 0.0) or 0.0) >= float(self.min_anchor_informativeness)),
                    "source": "history_visible" if bool(desc.get("visible_now", False)) else "history",
                    "conf": None,
                    "why": "historical_visible" if bool(desc.get("visible_now", False)) else "historical_candidate",
                    "debug_rank_local": None,
                }
            )

        rows.sort(
            key=lambda item: (
                float(item.get("usefulness", 0.0) or 0.0),
                float(item.get("pair_usefulness", 0.0) or 0.0),
                float(item.get("base_usefulness", 0.0) or 0.0),
                -int(item.get("anchor_id", -1) or -1),
            ),
            reverse=True,
        )
        topk = self.debug_historical_anchor_topk if limit is None else int(limit)
        if topk > 0:
            rows = rows[: int(topk)]
        for rank, item in enumerate(rows, start=1):
            item["rank"] = int(rank)
        return list(rows)
