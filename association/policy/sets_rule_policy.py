from __future__ import annotations


class SetsRulePolicy:
    def __init__(self, *, influence) -> None:
        self.influence = influence

    def allow_for_report(self, report) -> bool:
        diag = getattr(report, "match_diag_sim", None)
        if not isinstance(diag, dict):
            return True
        st = str(diag.get("status", "")).upper()
        if not st:
            return True
        return bool(st in self.influence.allow_status)

    def bonus_for_candidate(self, det_class_id: int, object_id: int, ctx: dict) -> dict:
        pack, quality = self.resolve_candidate_context(det_class_id=det_class_id, ctx=ctx)
        if pack is None or quality is None:
            return self.empty_bonus()

        candidate = self.candidate_context_values(object_id=int(object_id), pack=pack)
        support_local = self.compute_candidate_local_support(candidate=candidate, pack=pack)
        support_global = self.compute_candidate_global_support(candidate=candidate, pack=pack)
        support = self.compute_candidate_support(
            candidate=candidate,
            pack=pack,
            support_local=support_local,
            support_global=support_global,
        )
        contradiction = self.compute_candidate_contradiction(candidate=candidate, pack=pack)

        pos = float(self.influence.positive_cap) * float(quality) * float(support)
        neg = float(self.influence.negative_cap) * float(quality) * float(contradiction)
        net = float(pos - neg)

        return {
            "support": float(support),
            "support_local": float(support_local),
            "support_global": float(support_global),
            "contradiction": float(contradiction),
            "quality": float(quality),
            "net_bonus": float(net),
            "signed_score": float(support - contradiction),
            "signed_score_local": float(support_local),
            "signed_score_global": float(support_global),
            "kernel_raw": float(candidate["kernel_raw"]),
            "kernel_hit_count": int(candidate["kernel_hit_count"]),
            "kernel_hit_ratio": float(candidate["kernel_hit_ratio"]),
            "compat_rel": float(candidate["compat_rel"]),
            "compat_band": int(bool(candidate["soft_inside"])),
            "kernel_rel": float(candidate["kernel_rel"]),
            "hyp_rel": float(candidate["hyp_rel"]),
        }

    def explain_candidate(self, det_class_id: int, object_id: int, ctx: dict) -> dict:
        oid = int(object_id)
        ctx_enabled = bool(self.influence.enabled and isinstance(ctx, dict) and ctx.get("enabled", False))
        global_ok = bool(ctx_enabled and bool(ctx.get("global_ok", False)))
        quality = float(ctx.get("quality", 0.0) or 0.0) if isinstance(ctx, dict) else 0.0
        quality_ok = bool(quality >= float(self.influence.min_quality))

        pack = self.influence.class_pack(int(det_class_id), ctx) if isinstance(ctx, dict) else None
        has_pack = isinstance(pack, dict)
        cand = self.candidate_context_values(object_id=oid, pack=pack) if has_pack else {
            "compat_rel": 0.0,
            "kernel_raw": 0.0,
            "kernel_rel": 0.0,
            "hyp_rel": 0.0,
            "abs_support_ok": False,
            "inside": False,
            "soft_inside": False,
        }
        support_local = self.compute_candidate_local_support(candidate=cand, pack=pack) if has_pack else 0.0
        support_global = self.compute_candidate_global_support(candidate=cand, pack=pack) if has_pack else 0.0
        support = self.compute_candidate_support(
            candidate=cand,
            pack=pack,
            support_local=support_local,
            support_global=support_global,
        ) if has_pack else 0.0
        contradiction = self.compute_candidate_contradiction(candidate=cand, pack=pack) if has_pack else 0.0
        pos = float(self.influence.positive_cap) * float(max(0.0, quality)) * float(support)
        neg = float(self.influence.negative_cap) * float(max(0.0, quality)) * float(contradiction)
        net = float(pos - neg)

        shortlist = set(int(x) for x in ((pack or {}).get("shortlist", set()) or set()))
        supported = set(int(x) for x in ((pack or {}).get("supported", set()) or set()))
        soft_supported = set(int(x) for x in ((pack or {}).get("soft_supported", set()) or set()))

        return {
            "enabled": bool(ctx_enabled),
            "global": {
                "global_ok": bool(global_ok),
                "quality": float(quality),
                "quality_ok": bool(quality_ok),
                "reason": str((ctx or {}).get("reason", "")) if isinstance(ctx, dict) else "",
                "best": float((ctx or {}).get("best", 0.0) or 0.0) if isinstance(ctx, dict) else 0.0,
                "coverage_eff": float((ctx or {}).get("coverage_eff", 0.0) or 0.0) if isinstance(ctx, dict) else 0.0,
                "k_best": int((ctx or {}).get("k_best", 0) or 0) if isinstance(ctx, dict) else 0,
            },
            "class": {
                "has_pack": bool(has_pack),
                "kept_count": int((pack or {}).get("kept_count", 0) or 0),
                "total_count": int((pack or {}).get("total_count", 0) or 0),
                "pruning_power": float((pack or {}).get("pruning_power", 0.0) or 0.0),
                "class_strength": float((pack or {}).get("class_strength", 0.0) or 0.0),
                "selectivity": float((pack or {}).get("selectivity", 0.0) or 0.0),
                "top_support_abs": float((pack or {}).get("top_support_abs", 0.0) or 0.0),
                "top_support_rel": float((pack or {}).get("top_support_rel", 0.0) or 0.0),
                "band_cutoff_rel": float((pack or {}).get("band_cutoff_rel", 0.0) or 0.0),
                "soft_band_cutoff_rel": float((pack or {}).get("soft_band_cutoff_rel", 0.0) or 0.0),
                "shortlist_hit": bool(oid in shortlist),
                "supported_hit": bool(oid in supported),
                "soft_supported_hit": bool(oid in soft_supported),
            },
            "candidate": {
                "compat_rel": float(cand.get("compat_rel", 0.0) or 0.0),
                "kernel_raw": float(cand.get("kernel_raw", 0.0) or 0.0),
                "kernel_hit_count": int(cand.get("kernel_hit_count", 0) or 0),
                "kernel_hit_ratio": float(cand.get("kernel_hit_ratio", 0.0) or 0.0),
                "kernel_rel": float(cand.get("kernel_rel", 0.0) or 0.0),
                "hyp_rel": float(cand.get("hyp_rel", 0.0) or 0.0),
                "abs_support_ok": bool(cand.get("abs_support_ok", False)),
                "coverage_ok": bool(cand.get("coverage_ok", False)),
                "inside": bool(cand.get("inside", False)),
                "soft_inside": bool(cand.get("soft_inside", False)),
                "support": float(support),
                "support_local": float(support_local),
                "support_global": float(support_global),
                "contradiction": float(contradiction),
                "quality": float(quality),
                "bonus_pos": float(pos),
                "bonus_neg": float(neg),
                "bonus_net": float(net),
            },
            "policy": {
                "report_allowed": True,
                "ctx_reason": "",
                "veto_reason": "",
                "gate_reason": "",
                "bonus_applied": 0.0,
                "bonus_used": False,
            },
        }

    def allow_candidate_below_match_thr(self, det_class_id: int, object_id: int, score_sim: float, match_thr: float, ctx: dict) -> bool:
        if float(score_sim) < float(self.influence.rescue_min_sim):
            return False

        pack = self.bonus_for_candidate(det_class_id, object_id, ctx)
        if float(pack.get("net_bonus", 0.0)) <= 0.0:
            return False

        return bool(float(score_sim) + float(pack.get("net_bonus", 0.0)) >= float(match_thr))

    def resolve_candidate_context(self, *, det_class_id: int, ctx: dict) -> tuple[dict | None, float | None]:
        if not self.influence.enabled or not ctx or not ctx.get("enabled", False) or not ctx.get("global_ok", False):
            return None, None

        quality = float(ctx.get("quality", 0.0))
        if quality < float(self.influence.min_quality):
            return None, None

        pack = self.influence.class_pack(int(det_class_id), ctx)
        if pack is None:
            return None, None
        return pack, float(quality)

    def candidate_context_values(self, *, object_id: int, pack: dict) -> dict:
        oid = int(object_id)
        kernel_raw = float((pack.get("kernel_raw_by_oid", {}) or {}).get(oid, 0.0))
        kernel_hit_count = int((pack.get("kernel_hit_count_by_oid", {}) or {}).get(oid, 0) or 0)
        kernel_hit_ratio = float((pack.get("kernel_hit_ratio_by_oid", {}) or {}).get(oid, 0.0) or 0.0)
        kernel_rel = float((pack.get("kernel_rel_by_oid", {}) or {}).get(oid, 0.0))
        hyp_rel = float((pack.get("hyp_rel_by_oid", {}) or {}).get(oid, 0.0))
        top_support_rel = float(pack.get("top_support_rel", 0.0))
        band_cutoff = float(pack.get("band_cutoff_rel", 0.0))
        soft_band_cutoff = float(pack.get("soft_band_cutoff_rel", band_cutoff))
        supported = pack.get("supported", set()) or set()
        soft_supported = pack.get("soft_supported", set()) or set()
        abs_support_ok = bool(float(kernel_raw) >= float(self.influence.support_min_kernel_abs))
        coverage_ok = bool((pack.get("coverage_ok_by_oid", {}) or {}).get(oid, False))
        support_ready = bool(abs_support_ok and coverage_ok)

        inside = bool(
            support_ready
            and top_support_rel > 0.0
            and kernel_rel > 0.0
            and kernel_rel >= max(0.0, band_cutoff - 1e-9)
        )
        if not inside and support_ready and oid in supported:
            inside = True

        soft_inside = bool(
            support_ready
            and top_support_rel > 0.0
            and kernel_rel > 0.0
            and kernel_rel >= max(0.0, soft_band_cutoff - 1e-9)
        )
        if not soft_inside and support_ready and oid in soft_supported:
            soft_inside = True

        return {
            "compat_rel": float(kernel_rel),
            "kernel_raw": float(kernel_raw),
            "kernel_hit_count": int(kernel_hit_count),
            "kernel_hit_ratio": float(kernel_hit_ratio),
            "kernel_rel": float(kernel_rel),
            "hyp_rel": float(hyp_rel),
            "abs_support_ok": bool(abs_support_ok),
            "coverage_ok": bool(coverage_ok),
            "inside": bool(inside),
            "soft_inside": bool(soft_inside),
        }

    def compute_candidate_global_support(self, *, candidate: dict, pack: dict) -> float:
        if not bool(candidate.get("soft_inside", False)) or int(pack.get("kept_count", 0)) <= 0:
            return 0.0

        pruning = self.influence.clamp01(pack.get("pruning_power", 0.0))
        compat_gain = float(
            (float(candidate["kernel_rel"]) - float(self.influence.support_neutral_rel))
            / max(1e-12, 1.0 - float(self.influence.support_neutral_rel))
        )
        compat_gain = self.influence.clamp01(compat_gain)
        evidence_gain = float(
            (float(candidate["hyp_rel"]) - float(self.influence.support_neutral_rel))
            / max(1e-12, 1.0 - float(self.influence.support_neutral_rel))
        )
        evidence_gain = self.influence.clamp01(evidence_gain)
        selectivity = float(pack.get("selectivity", 0.0))

        support = (
            float(self.influence.support_pruning_weight) * float(pruning)
            + float(self.influence.support_rank_weight) * float(compat_gain)
            + float(self.influence.support_selectivity_weight) * float(evidence_gain) * (0.5 + 0.5 * float(selectivity))
        )
        return self.influence.clamp01(support)

    def compute_candidate_local_support(self, *, candidate: dict, pack: dict) -> float:
        if not bool(candidate.get("coverage_ok", False)):
            return 0.0

        kernel_raw = float(candidate.get("kernel_raw", 0.0) or 0.0)
        kernel_hit_count = int(candidate.get("kernel_hit_count", 0) or 0)
        kernel_hit_ratio = float(candidate.get("kernel_hit_ratio", 0.0) or 0.0)
        if kernel_raw < float(self.influence.support_local_min_kernel_abs):
            return 0.0
        if kernel_hit_count < int(self.influence.support_local_min_kernel_hits):
            return 0.0
        if kernel_hit_ratio < float(self.influence.support_local_min_kernel_hit_ratio):
            return 0.0

        hit_term = self.influence.clamp01(
            (float(kernel_hit_ratio) - float(self.influence.support_local_min_kernel_hit_ratio))
            / max(1e-12, 1.0 - float(self.influence.support_local_min_kernel_hit_ratio))
        )
        count_term = self.influence.clamp01(
            float(kernel_hit_count) / float(max(1, int(self.influence.support_local_min_kernel_hits)))
        )
        top_support_abs = float(pack.get("top_support_abs", 0.0) or 0.0)
        raw_den = max(
            1e-12,
            float(top_support_abs),
            float(self.influence.support_local_min_kernel_abs),
        )
        raw_term = self.influence.clamp01(float(kernel_raw) / float(raw_den))

        support = (
            float(self.influence.support_local_hit_weight) * float(hit_term)
            + float(self.influence.support_local_count_weight) * float(count_term)
            + float(self.influence.support_local_raw_weight) * float(raw_term)
        )
        return self.influence.clamp01(support)

    def compute_candidate_support(
        self,
        *,
        candidate: dict,
        pack: dict,
        support_local: float | None = None,
        support_global: float | None = None,
    ) -> float:
        if support_local is None:
            support_local = self.compute_candidate_local_support(candidate=candidate, pack=pack)
        if support_global is None:
            support_global = self.compute_candidate_global_support(candidate=candidate, pack=pack)

        lw = float(self.influence.support_local_weight)
        gw = float(self.influence.support_global_weight)
        z = float(max(1e-12, lw + gw))
        support = ((lw * float(support_local)) + (gw * float(support_global))) / z
        return self.influence.clamp01(support)

    def compute_candidate_contradiction(self, *, candidate: dict, pack: dict) -> float:
        if bool(candidate.get("soft_inside", False)):
            return 0.0
        if int(pack.get("kept_count", 0)) <= 0:
            return 0.0

        pruning = self.influence.clamp01(pack.get("pruning_power", 0.0))
        if pruning < float(self.influence.contradiction_min_pruning):
            return 0.0
        if float(pack.get("class_strength", 0.0)) < float(self.influence.contradiction_min_class_strength):
            return 0.0
        if float(candidate["kernel_rel"]) > float(self.influence.contradiction_max_rel):
            return 0.0
        return self.influence.clamp01(pruning)

    def soft_gate_eps_for_class(self, det_class_id: int, ctx: dict) -> float:
        return 0.0

    def anti_new_soft_thr(self, match_thr: float) -> float:
        return float(match_thr)

    def empty_bonus(self) -> dict:
        return {
            "support": 0.0,
            "support_local": 0.0,
            "support_global": 0.0,
            "contradiction": 0.0,
            "quality": 0.0,
            "net_bonus": 0.0,
            "signed_score": 0.0,
            "signed_score_local": 0.0,
            "signed_score_global": 0.0,
            "kernel_raw": 0.0,
            "compat_rel": 0.0,
            "compat_band": 0,
            "kernel_rel": 0.0,
            "hyp_rel": 0.0,
        }
