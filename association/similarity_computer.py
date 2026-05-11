# association/similarity_computer.py

from __future__ import annotations

import math
import numpy as np

from utils.math import cosine_sim
from utils.config import bg_partials_enabled


def use_stable_for_scores(config: dict | None) -> bool:
    assoc_cfg = ((config or {}).get("association", {}) or {})
    match_cfg = (assoc_cfg.get("matching", {}) or {})
    mode = str(match_cfg.get("proto_source_mode", "default") or "default").lower().strip()
    return mode != "work_only"


def max_two(a: float | None, b: float | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return float(b)
    if b is None:
        return float(a)
    return float(max(float(a), float(b)))


def prefer_stable(work: float | None, stable: float | None) -> float | None:
    return float(stable) if stable is not None else (None if work is None else float(work))


def merge_ws_scores(work: float | None, stable: float | None, *, include_combined: bool = False) -> dict:
    out = {
        "work": None if work is None else float(work),
        "stable": None if stable is None else float(stable),
        "max": max_two(work, stable),
    }
    if include_combined:
        out["combined"] = prefer_stable(work, stable)
    return out


def combine_two_optional(a: float | None, b: float | None, wa: float, wb: float) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return float(b)
    if b is None:
        return float(a)

    wa = float(max(0.0, wa))
    wb = float(max(0.0, wb))
    s = wa + wb
    if s <= 1e-12:
        return float(0.5 * float(a) + 0.5 * float(b))

    wa /= s
    wb /= s
    return float(wa * float(a) + wb * float(b))


def mean_topk(values: list[float], topk: int) -> float | None:
    if not values:
        return None
    values = [float(v) for v in values]
    values.sort(reverse=True)
    k = min(len(values), max(1, int(topk)))
    return float(np.mean(values[:k]))


def best_match_scores(observed_descs, proto_descs, clamp_nonneg: bool) -> list[float]:
    if not observed_descs or not proto_descs:
        return []

    best_sims = []
    for od in observed_descs:
        best = -1.0
        for p in proto_descs:
            best = max(best, cosine_sim(od, p))
        if clamp_nonneg:
            best = float(max(0.0, best))
        best_sims.append(float(best))

    return best_sims


def best_single_score(obs_desc, proto_descs, clamp_nonneg: bool) -> float | None:
    if obs_desc is None or not proto_descs:
        return None

    best = -1.0
    for p in proto_descs:
        best = max(best, cosine_sim(obs_desc, p))

    if best <= -1.0:
        return None

    if clamp_nonneg:
        best = float(max(0.0, best))

    return float(best)


def proto_quality_value(proto) -> float:
    if proto is None:
        return 0.0

    q_ema = float(getattr(proto, "quality_ema", 1.0) or 0.0)
    q_sum = float(getattr(proto, "quality_sum", q_ema) or 0.0)
    n_obs = max(1, int(getattr(proto, "n_obs", 1) or 1))
    q_mean = float(q_sum / float(n_obs))
    quality = float(max(0.0, min(1.0, max(q_mean, q_ema))))

    confidence = getattr(proto, "stability", None)
    if confidence is None:
        confidence = quality
    confidence = float(max(0.0, min(1.0, confidence)))

    # Prefer proto authority, not only visual "cleanliness": a weakly supported view
    # should not dominate like a consolidated proto even if its embedding fits well.
    return float(max(0.0, min(1.0, math.sqrt(max(0.0, quality * confidence)))))


def best_single_proto_score(obs_desc, protos, clamp_nonneg: bool) -> dict:
    if obs_desc is None or not protos:
        return {"score": None, "proto_q": None}

    best = -1.0
    best_proto = None
    for proto in protos:
        emb = getattr(proto, "embedding", None)
        if emb is None:
            continue
        s = cosine_sim(obs_desc, emb)
        if s > best:
            best = float(s)
            best_proto = proto

    if best <= -1.0 or best_proto is None:
        return {"score": None, "proto_q": None}

    if clamp_nonneg:
        best = float(max(0.0, best))

    return {"score": float(best), "proto_q": float(proto_quality_value(best_proto))}


class PartsSimilarityScorer:
    """Part score per channel with work/stable and max(work, stable) collapse."""

    def __init__(self, config: dict):
        parts_cfg = ((config.get("association", {}) or {}).get("similarity", {}) or {}).get("parts", {}) or {}
        self.topk = int(parts_cfg.get("topk", 3))
        self.use_stable = bool(use_stable_for_scores(config))

    def score(self, observed_parts: dict | None, tracked_object) -> dict:
        observed_parts = observed_parts or {}
        out = {}

        for channel_name in tracked_object.parts.channel_names():
            obs_pack = observed_parts.get(channel_name, None)
            obs_descs = obs_pack.get("part_descs", []) if isinstance(obs_pack, dict) else []
            out[str(channel_name)] = self.score_channel(obs_descs, tracked_object, str(channel_name))

        return out

    def score_channel(self, observed_descs, tracked_object, channel_name: str) -> dict:
        work = tracked_object.parts.get_channel_work_embeddings(channel_name)
        stable = tracked_object.parts.get_channel_stable_embeddings(channel_name) if self.use_stable else []

        s_work = self.mean_topk_best(observed_descs, work)
        s_stable = self.mean_topk_best(observed_descs, stable)

        return {"work": s_work, "stable": s_stable, "max": max_two(s_work, s_stable)}

    def mean_topk_best(self, observed_descs, proto_descs) -> float | None:
        sims = best_match_scores(observed_descs, proto_descs, clamp_nonneg=True)
        return mean_topk(sims, self.topk)


class BackgroundSimilarityScorer:
    """Background scoring with work/stable and max(work, stable) collapse per term."""

    def __init__(self, config: dict):
        bgp_cfg = ((config.get("association", {}) or {}).get("similarity", {}) or {}).get("background_partials", {}) or {}
        self.partials_topk = int(bgp_cfg.get("topk", 2))
        self.partials_enabled = bool(bg_partials_enabled(config))
        self.use_stable = bool(use_stable_for_scores(config))

    def score(self, observed_bg: dict | None, tracked_object) -> dict:
        bg_model = tracked_object.background
        if not bg_model.enabled:
            empty = self.empty_ws_scores()
            return {
                "inner": dict(empty),
                "outer": dict(empty),
                "combined": dict(empty),
                "partials_inner": dict(empty),
                "partials_outer": dict(empty),
                "partials": dict(empty),
            }

        observed_bg = observed_bg or {}

        inner_obs = observed_bg.get("inner", None)
        outer_obs = observed_bg.get("outer", None)

        inner_protos_obs = observed_bg.get("inner_protos", []) or []
        outer_protos_obs = observed_bg.get("outer_protos", []) or []

        inner_g_work = bg_model.inner_global_work.get_embeddings()
        inner_g_stable = bg_model.inner_global_stable.get_embeddings() if self.use_stable else []
        outer_g_work = bg_model.outer_global_work.get_embeddings()
        outer_g_stable = bg_model.outer_global_stable.get_embeddings() if self.use_stable else []

        s_inner = self.best_proto_score_ws(inner_obs, inner_g_work, inner_g_stable)
        s_outer = self.best_proto_score_ws(outer_obs, outer_g_work, outer_g_stable)

        s_comb_work = combine_two_optional(s_inner["work"], s_outer["work"], bg_model.w_inner, bg_model.w_outer)
        s_comb_stable = combine_two_optional(s_inner["stable"], s_outer["stable"], bg_model.w_inner, bg_model.w_outer)
        s_comb = merge_ws_scores(s_comb_work, s_comb_stable, include_combined=True)

        if self.partials_enabled:
            inner_p_work = bg_model.inner_partials_work.get_embeddings()
            inner_p_stable = bg_model.inner_partials_stable.get_embeddings() if self.use_stable else []
            outer_p_work = bg_model.outer_partials_work.get_embeddings()
            outer_p_stable = bg_model.outer_partials_stable.get_embeddings() if self.use_stable else []

            s_inner_p = self.mean_topk_best_set_ws(inner_protos_obs, inner_p_work, inner_p_stable)
            s_outer_p = self.mean_topk_best_set_ws(outer_protos_obs, outer_p_work, outer_p_stable)

            sp_comb_work = combine_two_optional(s_inner_p["work"], s_outer_p["work"], bg_model.w_inner, bg_model.w_outer)
            sp_comb_stable = combine_two_optional(s_inner_p["stable"], s_outer_p["stable"], bg_model.w_inner, bg_model.w_outer)
            s_partials = merge_ws_scores(sp_comb_work, sp_comb_stable, include_combined=False)
        else:
            s_inner_p = self.empty_ws_scores()
            s_outer_p = self.empty_ws_scores()
            s_partials = self.empty_ws_scores()

        return {
            "inner": s_inner,
            "outer": s_outer,
            "combined": s_comb,
            "partials_inner": s_inner_p,
            "partials_outer": s_outer_p,
            "partials": s_partials,
        }

    def best_proto_score_ws(self, obs_desc, proto_work, proto_stable) -> dict:
        s_work = best_single_score(obs_desc, proto_work, clamp_nonneg=True)
        s_stable = best_single_score(obs_desc, proto_stable, clamp_nonneg=True)
        return merge_ws_scores(s_work, s_stable, include_combined=True)

    def mean_topk_best_set_ws(self, observed_descs, proto_work, proto_stable) -> dict:
        s_work = self.mean_topk_best_set(observed_descs, proto_work)
        s_stable = self.mean_topk_best_set(observed_descs, proto_stable)
        return merge_ws_scores(s_work, s_stable, include_combined=False)

    def mean_topk_best_set(self, observed_descs, proto_descs) -> float | None:
        sims = best_match_scores(observed_descs, proto_descs, clamp_nonneg=True)
        return mean_topk(sims, self.partials_topk)

    def empty_ws_scores(self) -> dict:
        return {"work": None, "stable": None, "combined": None, "max": None}


class ObjectSimilarityScorer:
    """Object score per channel with stable priority and work fallback."""

    def __init__(self, config: dict | None = None):
        self.use_stable = bool(use_stable_for_scores(config))

    def score(self, observed_obj: dict | None, tracked_object) -> dict:
        observed_obj = observed_obj or {}
        out = {}

        for channel_name in tracked_object.appearance.channel_names():
            out[str(channel_name)] = self.score_channel(
                observed_obj=observed_obj,
                tracked_object=tracked_object,
                channel_name=str(channel_name),
            )

        return out

    def score_channel(self, observed_obj: dict, tracked_object, channel_name: str) -> dict:
        obs_pack = observed_obj.get(channel_name, None)
        obs_desc = obs_pack.get("desc", None) if isinstance(obs_pack, dict) else None

        ch = tracked_object.appearance.get_channel(channel_name)
        work_protos = list(getattr(ch, "work_protos", []) or []) if ch is not None else []
        stable_protos = list(getattr(ch, "stable_protos", []) or []) if (ch is not None and self.use_stable) else []

        work_pack = best_single_proto_score(obs_desc, work_protos, clamp_nonneg=True)
        stable_pack = best_single_proto_score(obs_desc, stable_protos, clamp_nonneg=True)
        return self.build_channel_score_pack(work_pack=work_pack, stable_pack=stable_pack)

    def build_channel_score_pack(self, work_pack: dict, stable_pack: dict) -> dict:
        s_work = work_pack.get("score", None)
        s_stable = stable_pack.get("score", None)
        q_work = work_pack.get("proto_q", None)
        q_stable = stable_pack.get("proto_q", None)

        best_source = self.best_source_name(s_work=s_work, s_stable=s_stable)
        q_combined = q_stable if s_stable is not None else q_work
        q_max = {"work": q_work, "stable": q_stable}.get(best_source, None)

        out = merge_ws_scores(s_work, s_stable, include_combined=True)
        out.update(
            {
                "work_proto_q": q_work,
                "stable_proto_q": q_stable,
                "combined_proto_q": q_combined,
                "max_proto_q": q_max,
            }
        )
        return out

    def best_source_name(self, s_work: float | None, s_stable: float | None) -> str | None:
        if s_work is None and s_stable is None:
            return None
        if s_work is None:
            return "stable"
        if s_stable is None:
            return "work"
        return "work" if float(s_work) > float(s_stable) else "stable"


class SimilarityComputer:
    """Evidence between one detection (det_feats) and a TrackedObject."""

    def __init__(self, config: dict):
        self.obj_scorer = ObjectSimilarityScorer(config)
        self.parts_scorer = PartsSimilarityScorer(config)
        self.bg_scorer = BackgroundSimilarityScorer(config)

    def object_score(self, det_feats: dict, tracked_object) -> dict:
        return self.obj_scorer.score(det_feats.get("obj", None), tracked_object)

    def background_score(self, det_feats: dict, tracked_object) -> dict:
        return self.bg_scorer.score(det_feats.get("bg", None), tracked_object)

    def parts_score(self, det_feats: dict, tracked_object) -> dict:
        return self.parts_scorer.score(det_feats.get("parts", None), tracked_object)
