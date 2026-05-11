from __future__ import annotations

from association.context.sets_context_builder import SetsContextBuilder
from association.policy.sets_rule_policy import SetsRulePolicy


class NeighborSetsInfluence:
    """
    Converts neighbor sets into a cheap additive context:
      - quality: global context reliability
      - support : positive support for candidates inside the plausible subcontext
      - contradiction: soft penalty for candidates clearly out of context

    Does not modify hypothesis search; only reuses its summarized outputs.
    """

    def __init__(self, config: dict, memory_store):
        self.config = config or {}
        self.memory_store = memory_store

        assoc = (self.config.get("association", {}) or {})
        match = (assoc.get("matching", {}) or {})
        ns = (match.get("neighbor_sets_influence", {}) or {})

        self.enabled = bool(ns.get("enabled", False))

        self.positive_cap = float(ns.get("positive_cap", 0.20))
        self.positive_cap = max(0.0, min(1.0, self.positive_cap))

        self.negative_cap = float(ns.get("negative_cap", 0.10))
        self.negative_cap = max(0.0, min(1.0, self.negative_cap))

        self.min_quality = float(ns.get("min_quality", 0.35))
        self.min_quality = max(0.0, min(1.0, self.min_quality))

        self.rescue_min_sim = float(ns.get("rescue_min_sim", 0.60))
        self.rescue_min_sim = max(0.0, min(1.0, self.rescue_min_sim))

        allow_status = ns.get("allow_status", ["WEAK", "AMBIGUOUS"])
        self.allow_status = {str(x).upper() for x in (allow_status or [])}

        qcfg = (ns.get("quality", {}) or {})
        self.min_best_score = float(qcfg.get("min_best_score", 0.45))
        self.min_best_score = max(0.0, min(1.0, self.min_best_score))
        self.min_coverage_eff = float(qcfg.get("min_coverage_eff", 0.35))
        self.min_coverage_eff = max(0.0, min(1.0, self.min_coverage_eff))
        self.min_size = int(qcfg.get("min_size", 2))
        self.min_size = max(1, self.min_size)
        self.size_tau = float(qcfg.get("size_tau", 3.0))
        self.size_tau = max(1e-6, self.size_tau)

        qw = (qcfg.get("weights", {}) or {})
        self.qw_best = float(qw.get("best_score", 0.25))
        self.qw_cov = float(qw.get("coverage_eff", 0.20))
        self.qw_maturity = float(qw.get("maturity", 0.15))
        self.qw_density = float(qw.get("density", 0.10))
        self.qw_size = float(qw.get("size", 0.15))
        self.qw_pruning = float(qw.get("pruning", 0.15))

        scfg = (ns.get("support", {}) or {})
        self.support_kernel_weight = float(scfg.get("kernel_weight", 0.75))
        self.support_kernel_weight = max(0.0, self.support_kernel_weight)
        self.support_hyp_weight = float(scfg.get("hyp_weight", 0.25))
        self.support_hyp_weight = max(0.0, self.support_hyp_weight)
        self.support_top_weight = float(scfg.get("top_weight", 0.65))
        self.support_top_weight = max(0.0, self.support_top_weight)
        self.support_sum_weight = float(scfg.get("sum_weight", 0.35))
        self.support_sum_weight = max(0.0, self.support_sum_weight)
        # Compress relative differences to avoid over-separating less consolidated identities.
        self.support_kernel_rel_gamma = float(scfg.get("kernel_rel_gamma", 0.75))
        self.support_kernel_rel_gamma = max(1e-6, self.support_kernel_rel_gamma)
        self.support_hyp_rel_gamma = float(scfg.get("hyp_rel_gamma", 0.75))
        self.support_hyp_rel_gamma = max(1e-6, self.support_hyp_rel_gamma)
        self.support_neutral_rel = float(scfg.get("neutral_rel", 0.20))
        self.support_neutral_rel = max(0.0, min(0.95, self.support_neutral_rel))
        self.support_band_rel = float(scfg.get("band_rel", 0.80))
        self.support_band_rel = max(0.0, min(1.0, self.support_band_rel))
        self.support_soft_band_rel = float(scfg.get("soft_band_rel", 0.65))
        self.support_soft_band_rel = max(0.0, min(1.0, self.support_soft_band_rel))
        self.support_soft_mix_kernel_weight = float(scfg.get("soft_band_mix_kernel_weight", 0.70))
        self.support_soft_mix_kernel_weight = max(0.0, self.support_soft_mix_kernel_weight)
        self.support_soft_mix_hyp_weight = float(scfg.get("soft_band_mix_hyp_weight", 0.30))
        self.support_soft_mix_hyp_weight = max(0.0, self.support_soft_mix_hyp_weight)
        self.support_min_kernel_abs = float(scfg.get("min_kernel_abs_for_support", 0.10))
        self.support_min_kernel_abs = max(0.0, self.support_min_kernel_abs)
        self.support_min_kernel_hits = int(scfg.get("min_kernel_hits_for_support", 2))
        self.support_min_kernel_hits = max(1, self.support_min_kernel_hits)
        self.support_min_kernel_hit_ratio = float(scfg.get("min_kernel_hit_ratio_for_support", 0.30))
        self.support_min_kernel_hit_ratio = max(0.0, min(1.0, self.support_min_kernel_hit_ratio))
        self.support_pruning_weight = float(scfg.get("pruning_weight", 0.55))
        self.support_pruning_weight = max(0.0, self.support_pruning_weight)
        self.support_rank_weight = float(scfg.get("rank_weight", 0.30))
        self.support_rank_weight = max(0.0, self.support_rank_weight)
        self.support_selectivity_weight = float(scfg.get("selectivity_weight", 0.15))
        self.support_selectivity_weight = max(0.0, self.support_selectivity_weight)
        self.support_local_weight = float(scfg.get("local_weight", 0.65))
        self.support_local_weight = max(0.0, self.support_local_weight)
        self.support_global_weight = float(scfg.get("global_weight", 0.35))
        self.support_global_weight = max(0.0, self.support_global_weight)
        self.support_local_hit_weight = float(scfg.get("local_hit_weight", 0.60))
        self.support_local_hit_weight = max(0.0, self.support_local_hit_weight)
        self.support_local_count_weight = float(scfg.get("local_count_weight", 0.20))
        self.support_local_count_weight = max(0.0, self.support_local_count_weight)
        self.support_local_raw_weight = float(scfg.get("local_raw_weight", 0.20))
        self.support_local_raw_weight = max(0.0, self.support_local_raw_weight)
        self.support_local_min_kernel_abs = float(scfg.get("local_min_kernel_abs_for_support", self.support_min_kernel_abs))
        self.support_local_min_kernel_abs = max(0.0, self.support_local_min_kernel_abs)
        self.support_local_min_kernel_hits = int(scfg.get("local_min_kernel_hits_for_support", self.support_min_kernel_hits))
        self.support_local_min_kernel_hits = max(1, self.support_local_min_kernel_hits)
        self.support_local_min_kernel_hit_ratio = float(
            scfg.get("local_min_kernel_hit_ratio_for_support", self.support_min_kernel_hit_ratio)
        )
        self.support_local_min_kernel_hit_ratio = max(0.0, min(1.0, self.support_local_min_kernel_hit_ratio))

        ccfg = (ns.get("contradiction", {}) or {})
        self.contradiction_min_pruning = float(ccfg.get("min_pruning", 0.35))
        self.contradiction_min_pruning = max(0.0, min(1.0, self.contradiction_min_pruning))
        self.contradiction_min_class_strength = float(ccfg.get("min_class_strength", 0.35))
        self.contradiction_min_class_strength = max(0.0, min(1.0, self.contradiction_min_class_strength))
        self.contradiction_max_rel = float(ccfg.get("max_rel", 0.10))
        self.contradiction_max_rel = max(0.0, min(1.0, self.contradiction_max_rel))

        set_scores_cfg = (((assoc.get("scores", {}) or {}).get("neighbor_sets", {}) or {}))
        self.min_edge_p = float(set_scores_cfg.get("min_edge_p", 0.05))
        self.min_edge_p = max(0.0, min(1.0, self.min_edge_p))
        self.context_k = int(((set_scores_cfg.get("context", {}) or {}).get("k", 6)))
        self.context_k = max(0, self.context_k)
        self.kernel_max = int(set_scores_cfg.get("kernel_max", 6))
        self.kernel_max = max(0, self.kernel_max)
        self.context_builder = SetsContextBuilder(memory_store=memory_store, influence=self)
        self.rule_policy = SetsRulePolicy(influence=self)

    def build_context(self, neighbor_sets_out) -> dict:
        return self.context_builder.build_context(neighbor_sets_out)

    def compress_rel(self, value: float, *, gamma: float) -> float:
        v = self.clamp01(value)
        if abs(float(gamma) - 1.0) <= 1e-9:
            return v
        return self.clamp01(v ** float(gamma))

    def extract_context_payload(self, neighbor_sets_out) -> dict | None:
        return self.context_builder.extract_context_payload(neighbor_sets_out)

    def build_global_quality(
        self,
        *,
        best_score: float,
        coverage_eff: float,
        maturity: float,
        density: float,
        k_best: int,
        n_hypotheses: int,
        class_ctx: dict[int, dict],
    ) -> tuple[dict, float, bool, str]:
        return self.context_builder.build_global_quality(
            best_score=best_score,
            coverage_eff=coverage_eff,
            maturity=maturity,
            density=density,
            k_best=k_best,
            n_hypotheses=n_hypotheses,
            class_ctx=class_ctx,
        )

    def build_quality_terms(
        self,
        *,
        best_score: float,
        coverage_eff: float,
        maturity: float,
        density: float,
        k_best: int,
        class_ctx: dict[int, dict],
    ) -> dict:
        return self.context_builder.build_quality_terms(
            best_score=best_score,
            coverage_eff=coverage_eff,
            maturity=maturity,
            density=density,
            k_best=k_best,
            class_ctx=class_ctx,
        )

    def quality_reason(
        self,
        *,
        n_hypotheses: int,
        k_best: int,
        best_term: float,
        coverage_term: float,
        quality_ok: bool,
    ) -> str:
        return self.context_builder.quality_reason(
            n_hypotheses=n_hypotheses,
            k_best=k_best,
            best_term=best_term,
            coverage_term=coverage_term,
            quality_ok=quality_ok,
        )

    def build_class_context(
        self,
        *,
        shortlist: set[int],
        prior_by_oid: dict[int, float],
        support_sum_by_oid: dict[int, float],
        anchors: list[int],
        hypotheses: list[dict],
        vocab_size: int | None,
    ) -> dict[int, dict]:
        return self.context_builder.build_class_context(
            shortlist=shortlist,
            prior_by_oid=prior_by_oid,
            support_sum_by_oid=support_sum_by_oid,
            anchors=anchors,
            hypotheses=hypotheses,
            vocab_size=vocab_size,
        )

    def objects_by_class(self) -> dict[int, list[int]]:
        return self.context_builder.objects_by_class()

    def build_class_pack(
        self,
        *,
        class_id: int,
        all_oids: list[int],
        shortlist: set[int],
        prior_by_oid: dict[int, float],
        support_sum_by_oid: dict[int, float],
        anchors: list[int],
        hypotheses: list[dict],
        vocab_size: int | None,
    ) -> dict | None:
        return self.context_builder.build_class_pack(
            class_id=class_id,
            all_oids=all_oids,
            shortlist=shortlist,
            prior_by_oid=prior_by_oid,
            support_sum_by_oid=support_sum_by_oid,
            anchors=anchors,
            hypotheses=hypotheses,
            vocab_size=vocab_size,
        )

    def kernel_support_by_oid(
        self,
        *,
        all_ids: list[int],
        kernel_ids: list[int],
        vocab_size: int | None,
    ) -> dict[int, float]:
        return self.context_builder.kernel_support_by_oid(
            all_ids=all_ids,
            kernel_ids=kernel_ids,
            vocab_size=vocab_size,
        )

    def class_relative_support_pack(
        self,
        *,
        all_ids: list[int],
        prior_by_oid: dict[int, float],
        support_sum_by_oid: dict[int, float],
        kernel_raw_by_oid: dict[int, float],
    ) -> dict:
        return self.context_builder.class_relative_support_pack(
            all_ids=all_ids,
            prior_by_oid=prior_by_oid,
            support_sum_by_oid=support_sum_by_oid,
            kernel_raw_by_oid=kernel_raw_by_oid,
        )

    def class_support_pack(
        self,
        *,
        all_ids: list[int],
        shortlist: set[int],
        kernel_raw_by_oid: dict[int, float],
        kernel_rel_by_oid: dict[int, float],
        hyp_rel_by_oid: dict[int, float],
    ) -> dict:
        return self.context_builder.class_support_pack(
            all_ids=all_ids,
            shortlist=shortlist,
            kernel_raw_by_oid=kernel_raw_by_oid,
            kernel_rel_by_oid=kernel_rel_by_oid,
            hyp_rel_by_oid=hyp_rel_by_oid,
        )

    def allow_for_report(self, report) -> bool:
        return self.rule_policy.allow_for_report(report)

    def bonus_for_candidate(self, det_class_id: int, object_id: int, ctx: dict) -> dict:
        return self.rule_policy.bonus_for_candidate(det_class_id, object_id, ctx)

    def explain_candidate(self, det_class_id: int, object_id: int, ctx: dict) -> dict:
        return self.rule_policy.explain_candidate(det_class_id, object_id, ctx)

    def allow_candidate_below_match_thr(self, det_class_id: int, object_id: int, score_sim: float, match_thr: float, ctx: dict) -> bool:
        return self.rule_policy.allow_candidate_below_match_thr(det_class_id, object_id, score_sim, match_thr, ctx)

    def class_pack(self, det_class_id: int, ctx: dict) -> dict | None:
        return self.context_builder.class_pack(det_class_id, ctx)

    def resolve_candidate_context(self, *, det_class_id: int, ctx: dict) -> tuple[dict | None, float | None]:
        return self.rule_policy.resolve_candidate_context(det_class_id=det_class_id, ctx=ctx)

    def candidate_context_values(self, *, object_id: int, pack: dict) -> dict:
        return self.rule_policy.candidate_context_values(object_id=object_id, pack=pack)

    def compute_candidate_support(self, *, candidate: dict, pack: dict) -> float:
        return self.rule_policy.compute_candidate_support(candidate=candidate, pack=pack)

    def compute_candidate_contradiction(self, *, candidate: dict, pack: dict) -> float:
        return self.rule_policy.compute_candidate_contradiction(candidate=candidate, pack=pack)

    def soft_gate_eps_for_class(self, det_class_id: int, ctx: dict) -> float:
        return self.rule_policy.soft_gate_eps_for_class(det_class_id, ctx)

    def anti_new_soft_thr(self, match_thr: float) -> float:
        return self.rule_policy.anti_new_soft_thr(match_thr)

    def empty_bonus(self) -> dict:
        return self.rule_policy.empty_bonus()

    def as_float_map(self, raw) -> dict[int, float]:
        return self.context_builder.as_float_map(raw)

    def clamp01(self, x: float) -> float:
        return self.context_builder.clamp01(x)

    def weighted_mean(self, items: list[tuple[float, float]]) -> float:
        return self.context_builder.weighted_mean(items)

    def class_kernel_ids(self, class_id: int, anchors: list[int], hypotheses: list[dict]) -> list[int]:
        return self.context_builder.class_kernel_ids(class_id, anchors, hypotheses)

    def object_support_to_kernel(self, object_id: int, kernel_obj_ids: list[int], vocab_size: int | None = None) -> float:
        return self.context_builder.object_support_to_kernel(
            object_id=object_id,
            kernel_obj_ids=kernel_obj_ids,
            vocab_size=vocab_size,
        )
