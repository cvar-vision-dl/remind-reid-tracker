from __future__ import annotations

import math

from association.policy.known_plausible_keep_policy import KnownPlausibleKeepPolicy


class CandidateScorePolicy:
    """Construye tablas det->obj aplicando la policy actual de gating y scoring."""

    def __init__(
        self,
        *,
        gate_by_match_thr: bool,
        gate_by_min_match: bool,
        debug_assoc_enabled: bool,
        dummy_score: float,
        use_confidence_dummy: bool,
        conf_alpha: float,
        dummy_score_cap: float,
        ctx_veto_enabled: bool,
        ctx_veto_supported_max: int,
        ctx_veto_min_quality: float,
        ctx_veto_min_pruning: float,
        ctx_veto_min_class_strength: float,
        ctx_veto_max_compat_rel: float,
        ctx_veto_max_score_sets: float,
        ctx_veto_local_enabled: bool,
        ctx_veto_local_min_quality: float,
        ctx_veto_local_min_episodes: int,
        ctx_veto_local_min_kernel_size: int,
        ctx_veto_local_min_expected_neighbors: int,
        ctx_veto_local_max_hit_ratio: float,
        ctx_veto_local_expected_mass_target: float,
        ctx_veto_local_expected_topk_scale: float,
        ctx_veto_local_require_supported_alternative: bool,
    ):
        self.gate_by_match_thr = bool(gate_by_match_thr)
        self.gate_by_min_match = bool(gate_by_min_match)
        self.debug_assoc_enabled = bool(debug_assoc_enabled)

        self.dummy_score = float(dummy_score)
        self.use_confidence_dummy = bool(use_confidence_dummy)
        self.conf_alpha = float(conf_alpha)
        self.dummy_score_cap = float(dummy_score_cap)

        self.ctx_veto_enabled = bool(ctx_veto_enabled)
        self.ctx_veto_supported_max = int(ctx_veto_supported_max)
        self.ctx_veto_min_quality = float(ctx_veto_min_quality)
        self.ctx_veto_min_pruning = float(ctx_veto_min_pruning)
        self.ctx_veto_min_class_strength = float(ctx_veto_min_class_strength)
        self.ctx_veto_max_compat_rel = float(ctx_veto_max_compat_rel)
        self.ctx_veto_max_score_sets = float(ctx_veto_max_score_sets)
        self.ctx_veto_local_enabled = bool(ctx_veto_local_enabled)
        self.ctx_veto_local_min_quality = float(ctx_veto_local_min_quality)
        self.ctx_veto_local_min_episodes = int(ctx_veto_local_min_episodes)
        self.ctx_veto_local_min_kernel_size = int(ctx_veto_local_min_kernel_size)
        self.ctx_veto_local_min_expected_neighbors = int(ctx_veto_local_min_expected_neighbors)
        self.ctx_veto_local_max_hit_ratio = float(ctx_veto_local_max_hit_ratio)
        self.ctx_veto_local_expected_mass_target = float(ctx_veto_local_expected_mass_target)
        self.ctx_veto_local_expected_topk_scale = float(ctx_veto_local_expected_topk_scale)
        self.ctx_veto_local_require_supported_alternative = bool(ctx_veto_local_require_supported_alternative)
        self.known_plausible_keep_policy = KnownPlausibleKeepPolicy(
            context_veto_reason_fn=self.candidate_context_veto_reason,
        )

    def report_status(self, report) -> str:
        diag = getattr(report, "match_diag_sim", None)
        if isinstance(diag, dict):
            return str(diag.get("status", "")).upper()
        return ""

    def default_sets_trace(self, report) -> dict:
        return {
            "enabled": False,
            "global": {
                "global_ok": False,
                "quality": 0.0,
                "quality_ok": False,
                "reason": "",
                "best": 0.0,
                "coverage_eff": 0.0,
                "k_best": 0,
            },
            "class": {
                "has_pack": False,
                "kept_count": 0,
                "total_count": 0,
                "pruning_power": 0.0,
                "class_strength": 0.0,
                "selectivity": 0.0,
                "top_support_abs": 0.0,
                "top_support_rel": 0.0,
                "band_cutoff_rel": 0.0,
                "soft_band_cutoff_rel": 0.0,
                "shortlist_hit": False,
                "supported_hit": False,
                "soft_supported_hit": False,
            },
                "candidate": {
                    "compat_rel": 0.0,
                    "kernel_rel": 0.0,
                    "kernel_hit_count": 0,
                    "kernel_hit_ratio": 0.0,
                    "hyp_rel": 0.0,
                    "inside": False,
                    "soft_inside": False,
                    "support": 0.0,
                "contradiction": 0.0,
                "quality": 0.0,
                "bonus_pos": 0.0,
                "bonus_neg": 0.0,
                "bonus_net": 0.0,
            },
            "policy": {
                "report_allowed": True,
                "report_status": str(self.report_status(report)),
                "ctx_reason": "",
                "known_plausible_reason": "",
                "veto_reason": "",
                "gate_reason": "",
                "bonus_applied": 0.0,
                "bonus_used": False,
            },
        }

    def format_sets_trace_summary(self, trace: dict) -> tuple[str, str, str]:
        if not isinstance(trace, dict) or not bool(trace.get("enabled", False)):
            return ("off", "off", "off")

        g = trace.get("global", {}) or {}
        c = trace.get("class", {}) or {}
        p = trace.get("policy", {}) or {}

        ctx = (
            f"go={int(bool(g.get('global_ok', False)))} "
            f"q={float(g.get('quality', 0.0)):.2f} "
            f"k={int(g.get('k_best', 0))} "
            f"rep={int(bool(p.get('report_allowed', True)))}"
        )
        cls = (
            f"sh={int(bool(c.get('shortlist_hit', False)))} "
            f"sup={int(bool(c.get('supported_hit', False)))} "
            f"ss={int(bool(c.get('soft_supported_hit', False)))} "
            f"keep={int(c.get('kept_count', 0))}/{int(c.get('total_count', 0))} "
            f"pr={float(c.get('pruning_power', 0.0)):.2f} "
            f"sel={float(c.get('selectivity', 0.0)):.2f} "
            f"ktop={float(c.get('top_support_abs', 0.0)):.2f}"
        )
        pol = (
            f"ctx={str(p.get('ctx_reason', '-')) or '-'} "
            f"veto={str(p.get('veto_reason', '-')) or '-'} "
            f"gate={str(p.get('gate_reason', '-')) or '-'} "
            f"b={float(p.get('bonus_applied', 0.0)):.2f}"
        )
        return ctx, cls, pol

    def attach_sets_trace_fields(self, candidate: dict, trace: dict) -> None:
        if not isinstance(candidate, dict):
            return
        ctx, cls, pol = self.format_sets_trace_summary(trace)
        candidate["sets_trace"] = trace if isinstance(trace, dict) else {"enabled": False}
        candidate["sets_ctx_summary"] = str(ctx)
        candidate["sets_class_summary"] = str(cls)
        candidate["sets_policy_summary"] = str(pol)

    def resolve_report_confidence(self, report) -> float | None:
        diag = getattr(report, "match_diag_sim", None)
        if isinstance(diag, dict):
            c = diag.get("confidence", None)
            if c is not None:
                return float(c)
        return None

    def resolve_dummy_score(self, report) -> float:
        s = float(self.dummy_score)
        if not self.use_confidence_dummy:
            return s

        conf = self.resolve_report_confidence(report)
        if conf is None:
            return s

        conf = max(0.0, min(1.0, float(conf)))
        s = float(s + self.conf_alpha * (1.0 - conf))
        s = max(0.0, min(float(self.dummy_score_cap), s))
        return float(s)

    def build_frame_local_plausible_kernel_ids(
        self,
        *,
        det_id: int,
        reports: dict,
        snapshot_ids: set[int],
        min_score: float,
        topk_per_detection: int = 3,
        source_topk_by_det: dict[int, list[int]] | None = None,
    ) -> list[int]:
        kernel_ids: list[int] = []
        seen: set[int] = set()
        reports = reports or {}
        if source_topk_by_det is not None:
            for other_det_id, _ in reports.items():
                if int(other_det_id) == int(det_id):
                    continue
                for object_id in (source_topk_by_det.get(int(other_det_id), []) or []):
                    object_id = int(object_id)
                    if int(object_id) in seen:
                        continue
                    seen.add(int(object_id))
                    kernel_ids.append(int(object_id))
            return kernel_ids

        min_score = float(max(0.0, min(1.0, float(min_score))))
        topk = max(1, int(topk_per_detection))
        for other_det_id, other_rep in reports.items():
            if int(other_det_id) == int(det_id):
                continue
            candidates = getattr(other_rep, 'candidates', None)
            if not isinstance(candidates, list) or not candidates:
                continue

            kept = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                object_id = candidate.get('object_id', None)
                if object_id is None:
                    continue
                object_id = int(object_id)
                if object_id not in snapshot_ids:
                    continue
                score_sim = float(candidate.get('score_sim', 0.0) or 0.0)
                if score_sim < float(min_score):
                    continue
                kept.append((float(score_sim), int(object_id)))

            if not kept:
                continue

            kept.sort(key=lambda item: (float(item[0]), int(item[1])), reverse=True)
            for _, object_id in kept[:topk]:
                if int(object_id) in seen:
                    continue
                seen.add(int(object_id))
                kernel_ids.append(int(object_id))

        return kernel_ids

    def build_frame_local_plausible_source_topk_by_det(
        self,
        *,
        reports: dict,
        snapshot_ids: set[int],
        min_score: float,
        topk_per_detection: int = 3,
    ) -> dict[int, list[int]]:
        reports = reports or {}
        min_score = float(max(0.0, min(1.0, float(min_score))))
        topk = max(1, int(topk_per_detection))
        out: dict[int, list[int]] = {}
        for det_id, rep in reports.items():
            candidates = getattr(rep, "candidates", None)
            if not isinstance(candidates, list) or not candidates:
                out[int(det_id)] = []
                continue
            kept: list[tuple[float, int]] = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                object_id = candidate.get("object_id", None)
                if object_id is None:
                    continue
                object_id = int(object_id)
                if object_id not in snapshot_ids:
                    continue
                score_sim = float(candidate.get("score_sim", 0.0) or 0.0)
                if score_sim < float(min_score):
                    continue
                kept.append((float(score_sim), int(object_id)))
            if not kept:
                out[int(det_id)] = []
                continue
            kept.sort(key=lambda item: (float(item[0]), int(item[1])), reverse=True)
            out[int(det_id)] = [int(object_id) for _, object_id in kept[:topk]]
        return out

    def build_score_tables(
        self,
        det_ids: list[int],
        detections_by_id: dict[int, object],
        reports: dict,
        snapshot_ids: set[int],
        used_obj_ids: set[int],
        *,
        use_neighbor_sets: bool,
        ns_ctx: dict | None,
        neighbor_sets_influence,
        match_thr: float,
        min_match_score: float,
        min_score: float | None = None,
        gate_by_match_thr: bool | None = None,
    ) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]], dict[int, dict[int, float]], set[int]]:
        table_sim: dict[int, dict[int, float]] = {}
        table_assign: dict[int, dict[int, float]] = {}
        table_final: dict[int, dict[int, float]] = {}
        objs: set[int] = set()

        base_thr_min = float(min_match_score) if min_score is None else float(min_score)
        gate_thr = self.gate_by_match_thr if gate_by_match_thr is None else bool(gate_by_match_thr)
        thr_match = float(match_thr)
        kernel_min_score = max(float(thr_match), float(base_thr_min))
        frame_local_source_topk_by_det = self.build_frame_local_plausible_source_topk_by_det(
            reports=reports,
            snapshot_ids=snapshot_ids,
            min_score=kernel_min_score,
        )

        for did in det_ids:
            rep = reports.get(int(did), None)
            cands = getattr(rep, "candidates", None) if rep is not None else None
            frame_local_kernel_ids = self.build_frame_local_plausible_kernel_ids(
                det_id=int(did),
                reports=reports,
                snapshot_ids=snapshot_ids,
                min_score=kernel_min_score,
                source_topk_by_det=frame_local_source_topk_by_det,
            )
            if not isinstance(cands, list) or not cands:
                continue
            for c in cands:
                if isinstance(c, dict):
                    c["decision_keep"] = 0

            det = detections_by_id.get(int(did), None)
            det_cid = int(getattr(det, "class_id", -1)) if det is not None else -1

            allow_influence = False
            if use_neighbor_sets and ns_ctx and neighbor_sets_influence is not None and rep is not None:
                allow_influence = bool(neighbor_sets_influence.allow_for_report(rep))
                if not allow_influence and used_obj_ids:
                    report_candidate_ids = {
                        int(c.get("object_id"))
                        for c in cands
                        if isinstance(c, dict) and c.get("object_id", None) is not None
                    }
                    if report_candidate_ids & set(int(x) for x in used_obj_ids):
                        allow_influence = True

            thr_min = float(base_thr_min)

            row_sim: dict[int, float] = {}
            row_assign_base: dict[int, float] = {}
            row_final_base: dict[int, float] = {}
            row_candidate_by_oid: dict[int, dict] = {}
            row_sets_trace_by_oid: dict[int, dict] = {}

            for c in cands:
                if not isinstance(c, dict):
                    continue

                oid = c.get("object_id", None)
                if oid is None:
                    continue
                oid = int(oid)

                c["ctx_keep"] = 1
                ctx_reason = "NO_CLASS_SHORTLIST"

                c["frame_local_ctx_kernel_ids"] = [int(x) for x in (frame_local_kernel_ids or [])]
                c["frame_local_ctx_kernel_size"] = int(len(frame_local_kernel_ids or []))

                sets_trace = self.default_sets_trace(rep)
                if self.debug_assoc_enabled and neighbor_sets_influence is not None and isinstance(ns_ctx, dict):
                    sets_trace = neighbor_sets_influence.explain_candidate(
                        det_class_id=int(det_cid),
                        object_id=int(oid),
                        ctx=ns_ctx,
                    )
                policy = (sets_trace.get("policy", {}) or {})
                policy["report_allowed"] = bool(allow_influence)
                policy["report_status"] = str(self.report_status(rep))
                policy["ctx_reason"] = str(ctx_reason)
                self.attach_sets_trace_fields(c, sets_trace)

                if oid not in snapshot_ids:
                    continue

                s_sim = float(c.get("score_sim", 0.0))
                bonus_sets_raw = 0.0
                bonus_sets = 0.0
                score_sets = 0.0
                penalty_sets = 0.0
                support_sets = 0.0
                support_local_sets = 0.0
                support_global_sets = 0.0
                quality_sets = 0.0
                score_ctx_local = 0.0
                score_ctx_global = 0.0
                compat_rel = 0.0
                compat_band = 0
                kernel_raw = 0.0
                kernel_rel = 0.0
                kernel_hit_count = 0
                kernel_hit_ratio = 0.0
                hyp_rel = 0.0
                raw_bonus_pack = None
                if neighbor_sets_influence is not None and isinstance(ns_ctx, dict):
                    raw_bonus_pack = neighbor_sets_influence.bonus_for_candidate(det_cid, oid, ns_ctx)
                    if isinstance(raw_bonus_pack, dict):
                        bonus_sets_raw = float(raw_bonus_pack.get("net_bonus", 0.0))
                        score_sets = float(raw_bonus_pack.get("signed_score", 0.0))
                        penalty_sets = float(raw_bonus_pack.get("contradiction", 0.0))
                        support_sets = float(raw_bonus_pack.get("support", 0.0))
                        support_local_sets = float(raw_bonus_pack.get("support_local", 0.0))
                        support_global_sets = float(raw_bonus_pack.get("support_global", 0.0))
                        quality_sets = float(raw_bonus_pack.get("quality", 0.0))
                        score_ctx_local = float(quality_sets * support_local_sets)
                        score_ctx_global = float(quality_sets * support_global_sets)
                        compat_rel = float(raw_bonus_pack.get("compat_rel", 0.0))
                        compat_band = int(raw_bonus_pack.get("compat_band", 0))
                        kernel_raw = float(raw_bonus_pack.get("kernel_raw", 0.0))
                        kernel_hit_count = int(raw_bonus_pack.get("kernel_hit_count", 0) or 0)
                        kernel_hit_ratio = float(raw_bonus_pack.get("kernel_hit_ratio", 0.0) or 0.0)
                        kernel_rel = float(raw_bonus_pack.get("kernel_rel", 0.0))
                        hyp_rel = float(raw_bonus_pack.get("hyp_rel", 0.0))
                    else:
                        bonus_sets_raw = float(raw_bonus_pack or 0.0)
                        score_sets = float(bonus_sets_raw)
                if allow_influence:
                    bonus_sets = float(bonus_sets_raw)

                c["score_sets"] = float(score_sets)
                c["bonus_sets_raw"] = float(bonus_sets_raw)
                c["bonus_sets"] = float(bonus_sets)
                c["penalty_sets"] = float(penalty_sets)
                c["support_sets"] = float(support_sets)
                c["support_local_sets"] = float(support_local_sets)
                c["support_global_sets"] = float(support_global_sets)
                c["quality_sets"] = float(quality_sets)
                c["score_ctx_local"] = float(score_ctx_local)
                c["score_ctx_global"] = float(score_ctx_global)
                c["compat_rel"] = float(compat_rel)
                c["compat_band"] = int(compat_band)
                c["kernel_raw"] = float(kernel_raw)
                c["kernel_hit_count"] = int(kernel_hit_count)
                c["kernel_hit_ratio"] = float(kernel_hit_ratio)
                c["kernel_rel"] = float(kernel_rel)
                c["hyp_rel"] = float(hyp_rel)

                if isinstance(sets_trace, dict):
                    cand_trace = (sets_trace.get("candidate", {}) or {})
                    cand_trace["kernel_raw"] = float(kernel_raw)
                    cand_trace["kernel_hit_count"] = int(kernel_hit_count)
                    cand_trace["kernel_hit_ratio"] = float(kernel_hit_ratio)
                    cand_trace["support"] = float(support_sets)
                    cand_trace["support_local"] = float(support_local_sets)
                    cand_trace["support_global"] = float(support_global_sets)
                    cand_trace["contradiction"] = float(penalty_sets)
                    cand_trace["bonus_net"] = float(bonus_sets_raw)
                    policy = (sets_trace.get("policy", {}) or {})
                    policy["bonus_applied"] = float(bonus_sets)
                    policy["bonus_used"] = bool(abs(float(bonus_sets)) > 1e-12)
                    self.attach_sets_trace_fields(c, sets_trace)

                known_plausible_keep = self.known_plausible_keep_policy.evaluate(
                    det_class_id=det_cid,
                    object_id=oid,
                    candidate=c,
                    ns_ctx=ns_ctx,
                    neighbor_sets_influence=neighbor_sets_influence,
                )
                c["known_plausible_keep"] = int(known_plausible_keep.get("keep", 0) or 0)
                c["known_plausible_reason"] = str(known_plausible_keep.get("reason", "") or "")
                if isinstance(sets_trace, dict):
                    policy = (sets_trace.get("policy", {}) or {})
                    policy["known_plausible_reason"] = str(known_plausible_keep.get("reason", "") or "")
                    policy["veto_reason"] = str(known_plausible_keep.get("veto_reason", "") or "")
                    self.attach_sets_trace_fields(c, sets_trace)
                if int(c["known_plausible_keep"]) != 1:
                    continue

                if self.gate_by_min_match and s_sim < float(thr_min):
                    if isinstance(sets_trace, dict):
                        policy = (sets_trace.get("policy", {}) or {})
                        policy["gate_reason"] = "BLOCK_MIN_MATCH"
                        self.attach_sets_trace_fields(c, sets_trace)
                    continue
                if gate_thr and s_sim < thr_match:
                    sets_ok = False
                    if allow_influence and neighbor_sets_influence is not None:
                        sets_ok = bool(
                            s_sim >= float(neighbor_sets_influence.rescue_min_sim)
                            and bonus_sets > 0.0
                            and (s_sim + bonus_sets) >= float(thr_match)
                        )
                    if isinstance(sets_trace, dict):
                        policy = (sets_trace.get("policy", {}) or {})
                        if sets_ok:
                            policy["gate_reason"] = "SETS_RESCUE"
                        else:
                            policy["gate_reason"] = "BLOCK_MATCH_THR"
                        self.attach_sets_trace_fields(c, sets_trace)
                    if not sets_ok:
                        continue
                elif isinstance(sets_trace, dict):
                    policy = (sets_trace.get("policy", {}) or {})
                    policy["gate_reason"] = "PASS_MATCH_THR"
                    self.attach_sets_trace_fields(c, sets_trace)

                if oid in used_obj_ids:
                    if isinstance(sets_trace, dict):
                        policy = (sets_trace.get("policy", {}) or {})
                        policy["gate_reason"] = "BLOCK_USED_OBJECT"
                        self.attach_sets_trace_fields(c, sets_trace)
                    continue

                s_base = float(s_sim + bonus_sets)
                s_assign_base = float(s_sim + min(0.0, float(bonus_sets)))
                row_sim[int(oid)] = float(s_sim)
                row_assign_base[int(oid)] = float(s_assign_base)
                row_final_base[int(oid)] = float(s_base)
                row_candidate_by_oid[int(oid)] = c
                row_sets_trace_by_oid[int(oid)] = sets_trace if isinstance(sets_trace, dict) else {"enabled": False}
                c["decision_keep"] = 1
                objs.add(int(oid))

            if not row_sim:
                continue

            row_assign = dict(row_assign_base)
            row_final = dict(row_final_base)

            for oid, c in row_candidate_by_oid.items():
                c["score_assign"] = float(row_assign.get(int(oid), row_assign_base.get(int(oid), c.get("score_sim", 0.0))))
                c["score_final"] = float(row_final.get(int(oid), row_final_base.get(int(oid), c.get("score_final", 0.0))))
                c["score_known"] = float(
                    max(
                        float(c["score_final"]),
                        float(c.get("score_sim", 0.0) or 0.0)
                        + float(c.get("bonus_sets_raw", 0.0) or 0.0),
                    )
                )
                sets_trace = row_sets_trace_by_oid.get(int(oid), None)
                if isinstance(sets_trace, dict):
                    self.attach_sets_trace_fields(c, sets_trace)

            table_sim[int(did)] = row_sim
            table_assign[int(did)] = row_assign
            table_final[int(did)] = row_final

        return table_sim, table_assign, table_final, objs

    def candidate_context_veto_reason(
        self,
        *,
        det_class_id: int,
        object_id: int,
        candidate: dict,
        ns_ctx: dict | None,
        neighbor_sets_influence,
    ) -> str:
        if not self.ctx_veto_enabled:
            return ""
        if neighbor_sets_influence is None or not isinstance(ns_ctx, dict):
            return ""
        if not bool(ns_ctx.get("enabled", False)):
            return ""

        quality = float(ns_ctx.get("quality", 0.0) or 0.0)
        pack = neighbor_sets_influence.class_pack(int(det_class_id), ns_ctx)
        if not isinstance(pack, dict):
            return ""

        oid = int(object_id)
        supported = set(int(x) for x in (pack.get("soft_supported", set()) or set()))
        if oid in supported:
            return ""

        shortlist = set(int(x) for x in (pack.get("shortlist", set()) or set()))
        if oid in shortlist:
            return ""

        compat_band = int(bool(candidate.get("compat_band", 0)))
        compat_rel = float(candidate.get("compat_rel", 0.0) or 0.0)
        score_sets = float(candidate.get("score_sets", 0.0) or 0.0)

        if compat_band != 0:
            return ""

        local_reason = self.candidate_local_context_veto_reason(
            object_id=oid,
            candidate=candidate,
            pack=pack,
            quality=quality,
            neighbor_sets_influence=neighbor_sets_influence,
        )
        if local_reason:
            return str(local_reason)

        if compat_rel > float(self.ctx_veto_max_compat_rel):
            return ""
        if score_sets > float(self.ctx_veto_max_score_sets):
            return ""

        if not bool(ns_ctx.get("global_ok", False)):
            return ""
        if quality < float(self.ctx_veto_min_quality):
            return ""

        kept_count = int(pack.get("kept_count", 0) or 0)
        if kept_count <= 0 or kept_count > int(self.ctx_veto_supported_max):
            return ""

        pruning = float(pack.get("pruning_power", 0.0) or 0.0)
        class_strength = float(pack.get("class_strength", 0.0) or 0.0)
        if pruning < float(self.ctx_veto_min_pruning):
            return ""
        if class_strength < float(self.ctx_veto_min_class_strength):
            return ""
        return "OUTSIDE_CTX"

    def candidate_local_context_veto_reason(
        self,
        *,
        object_id: int,
        candidate: dict,
        pack: dict,
        quality: float,
        neighbor_sets_influence,
    ) -> str:
        if not self.ctx_veto_local_enabled:
            return ""
        if not isinstance(pack, dict) or neighbor_sets_influence is None:
            return ""
        if float(quality) < float(self.ctx_veto_local_min_quality):
            return ""

        oid = int(object_id)
        if bool(candidate.get("compat_band", 0)):
            return ""

        supported = set(int(x) for x in (pack.get("soft_supported", set()) or set()))
        supported.discard(int(oid))
        candidate["local_ctx_has_supported_alternative"] = int(bool(supported))

        memory_store = getattr(neighbor_sets_influence, "memory_store", None)
        if memory_store is None:
            return ""
        obj = memory_store.get(int(oid))
        if obj is None:
            return ""
        graph = getattr(obj, "neighbors", None)
        if graph is None or not getattr(graph, "enabled", False):
            return ""

        episode_count = int(max(0, int(getattr(graph, "episode_count", 0) or 0)))
        candidate["local_ctx_episode_count"] = int(episode_count)
        if episode_count < int(self.ctx_veto_local_min_episodes):
            return ""

        frame_local_kernel_ids = [
            int(x)
            for x in (candidate.get("frame_local_ctx_kernel_ids", []) or [])
            if x is not None
        ]
        use_frame_local_kernel = bool(frame_local_kernel_ids or ("frame_local_ctx_kernel_ids" in candidate))
        kernel_ids = list(frame_local_kernel_ids) if use_frame_local_kernel else [
            int(x) for x in (pack.get("kernel_ids", []) or []) if x is not None
        ]
        candidate["local_ctx_kernel_source"] = "frame_visible" if use_frame_local_kernel else "sets_kernel"
        candidate["local_ctx_kernel_size"] = int(len(kernel_ids))
        candidate["local_ctx_frame_kernel_size"] = int(len(frame_local_kernel_ids))
        if len(kernel_ids) < int(self.ctx_veto_local_min_kernel_size):
            return ""

        vocab_size = len(memory_store.all_objects()) if hasattr(memory_store, "all_objects") else None
        expected_ids = self.expected_context_neighbors(
            graph=graph,
            context_k=int(getattr(neighbor_sets_influence, "context_k", 6)),
            min_p=float(getattr(neighbor_sets_influence, "min_edge_p", 0.05)),
            vocab_size=vocab_size,
        )
        candidate["local_ctx_expected_count"] = int(len(expected_ids))
        if len(expected_ids) < int(self.ctx_veto_local_min_expected_neighbors):
            return ""

        kernel_set = set(int(x) for x in kernel_ids)
        hit_count = sum(1 for nid in expected_ids if int(nid) in kernel_set)
        hit_ratio = float(hit_count) / float(max(1, len(expected_ids)))
        candidate["local_ctx_hit_count"] = int(hit_count)
        candidate["local_ctx_hit_ratio"] = float(hit_ratio)
        maturity = 1.0 - math.exp(-0.25 * float(max(0, episode_count)))
        candidate["local_ctx_maturity"] = float(maturity)
        if hit_ratio >= float(self.ctx_veto_local_max_hit_ratio):
            return ""
        if maturity < 0.5:
            return ""
        return "LOCAL_CTX_CONTRADICTION"

    def expected_context_neighbors(
        self,
        *,
        graph,
        context_k: int,
        min_p: float,
        vocab_size: int | None,
    ) -> list[int]:
        out: list[int] = []
        topk = max(1, int(context_k))
        max_count = max(topk, int(math.ceil(float(topk) * float(self.ctx_veto_local_expected_topk_scale))))
        thr = max(0.0, min(1.0, float(min_p)))
        cum_p = 0.0
        target_mass = max(0.0, min(1.0, float(self.ctx_veto_local_expected_mass_target)))
        for pack in (graph.neighbors() or []):
            nid = int(pack.get("dst_id", -1))
            if nid < 0:
                continue
            if int(pack.get("cooc_count", 0) or 0) <= 0:
                continue
            try:
                p = float(graph.p_conditional(int(nid), vocab_size=vocab_size))
            except Exception:
                p = 0.0
            p = max(0.0, min(1.0, p))
            if p < thr:
                continue
            out.append(int(nid))
            cum_p += float(p)
            if len(out) >= int(max_count):
                break
            if len(out) >= int(topk) and cum_p >= float(target_mass):
                break
        return out

    def candidate_vetoed_by_context(
        self,
        *,
        det_class_id: int,
        object_id: int,
        candidate: dict,
        ns_ctx: dict | None,
        neighbor_sets_influence,
    ) -> bool:
        return bool(
            self.candidate_context_veto_reason(
                det_class_id=det_class_id,
                object_id=object_id,
                candidate=candidate,
                ns_ctx=ns_ctx,
                neighbor_sets_influence=neighbor_sets_influence,
            )
        )
