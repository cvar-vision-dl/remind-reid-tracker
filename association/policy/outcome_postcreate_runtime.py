from __future__ import annotations


class OutcomePostcreateRuntime:
    """Runtime helper for temporal postcreate decisions."""

    def __init__(self, *, policy):
        self.policy = policy

    @staticmethod
    def empty_temporal_decisions() -> dict:
        return {"ambiguous_entries": [], "provisional_entries": [], "debug_entries": []}

    @staticmethod
    def new_postcreate_debug_entry(*, det_id: int, class_id: int) -> dict:
        return {
            "det_id": int(det_id),
            "class_id": int(class_id),
            "temporal_status": "",
            "decision_kind": "SKIP",
            "decision_reason": "",
            "skip_reason": "",
            "focus_source": "none",
            "has_known_context": 0,
            "visual_fallback_ok": 0,
            "known_blocked_ok": 0,
            "status_not_allowed": 0,
            "context_mode": "none",
            "support_mode": "none",
            "relation": "none",
            "support_known_ids": [],
            "support_known_scores": {},
            "blocked_known_ids": [],
            "blocked_known_scores": {},
            "related_known_ids": [],
            "related_known_scores": {},
            "best_object_id": None,
            "best_score": None,
            "top_supported_object_id": None,
            "top_supported_score": None,
            "provisional_parent_status_ok": 0,
            "provisional_parent_ok": 0,
            "candidate_rows": [],
        }

    def append_postcreate_skip_debug(
        self,
        *,
        debug_out: list[dict],
        debug_entry: dict,
        reason: str,
        candidates: list[dict] | None = None,
        best_score: float | None = None,
        score_map: dict[int, float] | None = None,
        assigned_object_ids: set[int] | None = None,
        support_known_ids: list[int] | None = None,
        temporal_scores_by_id: dict[int, float] | None = None,
    ) -> None:
        debug_entry["skip_reason"] = str(reason or "")
        if candidates is not None and best_score is not None and assigned_object_ids is not None:
            debug_entry["candidate_rows"] = self.build_postcreate_candidate_debug_rows(
                candidates=candidates,
                best_score=float(best_score),
                score_map=score_map,
                assigned_object_ids=set(int(x) for x in (assigned_object_ids or set())),
                support_known_ids=list(support_known_ids or []),
                temporal_scores_by_id=temporal_scores_by_id,
            )
        debug_out.append(debug_entry)

    @staticmethod
    def postcreate_debug_candidates_pack(
        *,
        known_candidates: list[dict],
        focus_candidates: list[dict],
        known_score_map: dict[int, float],
        focus_score_map: dict[int, float],
        known_temporal_scores: dict[int, float],
        focus_temporal_scores: dict[int, float],
    ) -> tuple[list[dict], dict[int, float], dict[int, float]]:
        if known_candidates:
            return list(known_candidates), dict(known_score_map), dict(known_temporal_scores)
        return list(focus_candidates), dict(focus_score_map), dict(focus_temporal_scores)

    def build_postcreate_temporal_decisions(
        self,
        *,
        reports_by_det_id: dict,
        to_create: list[dict],
        assigned_by_det_id: dict[int, int] | None = None,
        excluded_det_ids: set[int] | None = None,
    ) -> dict:
        policy = self.policy
        if not policy.prov_new_enabled:
            return self.empty_temporal_decisions()
        if not to_create:
            return self.empty_temporal_decisions()

        excluded = set(int(x) for x in (excluded_det_ids or set()))
        assigned_object_ids = {int(x) for x in ((assigned_by_det_id or {}).values()) if x is not None}
        ambiguous_out: list[dict] = []
        provisional_out: list[dict] = []
        debug_out: list[dict] = []
        create_det_ids = sorted(
            int(x.get("det_id", -1))
            for x in (to_create or [])
            if isinstance(x, dict) and int(x.get("det_id", -1)) >= 0
        )

        for did in create_det_ids:
            if int(did) in excluded:
                continue
            rep = (reports_by_det_id or {}).get(int(did), None)
            if rep is None:
                continue
            debug_entry = self.new_postcreate_debug_entry(
                det_id=int(did),
                class_id=int(getattr(rep, "class_id", -1)),
            )

            cands = [c for c in policy.iter_candidates(rep, scope="ambiguity") if isinstance(c, dict)]
            raw_cands = [c for c in policy.iter_candidates(rep, scope="raw") if isinstance(c, dict)]
            if not cands and not raw_cands:
                self.append_postcreate_skip_debug(
                    debug_out=debug_out,
                    debug_entry=debug_entry,
                    reason="no_candidates",
                )
                continue

            supported = policy.ambiguous_supported_candidates(cands)
            score_map = policy.compute_comparable_score_map(cands)
            cand_temporal_scores = {
                id(candidate): float(policy.temporal_candidate_score(candidate, score_map=score_map))
                for candidate in cands
            }
            supported.sort(
                key=lambda c: float(cand_temporal_scores.get(id(c), 0.0)),
                reverse=True,
            )
            has_known_context = bool(supported)
            context_missing = bool(not has_known_context)
            # For provisionals, one single "supported" candidate should not
            # collapse temporal diagnostics to STRONG by itself if the rest
            # if the rest of plausible known candidates still compete visually. Reserve
            # the diagnostics over the supported subset for truly
            # multi-support cases; in singleton cases evaluate against all plausible candidates.
            diag_candidates = list(supported) if len(supported) >= 2 else list(cands)
            diag_temporal = policy.compute_temporal_support_diag_from_candidates(
                diag_candidates,
                mode="provisional",
                score_map=score_map,
            )
            st = str((diag_temporal or {}).get("status", "")).upper()
            debug_entry["temporal_status"] = str(st)
            debug_entry["has_known_context"] = int(bool(has_known_context))
            top_supported = supported[0] if supported else None
            top_supported_oid = None if top_supported is None else top_supported.get("object_id", None)
            top_supported_blocked = bool(top_supported_oid is not None and int(top_supported_oid) in assigned_object_ids)
            raw_score_map = policy.compute_comparable_score_map(raw_cands)
            raw_temporal_scores = {
                id(candidate): float(policy.temporal_candidate_score(candidate, score_map=raw_score_map))
                for candidate in raw_cands
            }
            top_raw = None
            if raw_cands:
                top_raw = max(
                    raw_cands,
                    key=lambda c: float(raw_temporal_scores.get(id(c), 0.0)),
                )
            top_raw_oid = None if top_raw is None else top_raw.get("object_id", None)
            known_blocked_ok = bool(
                top_supported_blocked
                and has_known_context
                and float(cand_temporal_scores.get(id(top_supported), 0.0)) >= float(policy.prov_new_min_top_score)
            )
            debug_entry["known_blocked_ok"] = int(bool(known_blocked_ok))
            debug_entry["top_supported_object_id"] = None if top_supported_oid is None else int(top_supported_oid)
            if top_supported is not None:
                debug_entry["top_supported_score"] = float(cand_temporal_scores.get(id(top_supported), 0.0))
            visual_fallback_ok = policy.provisional_visual_fallback_ok(
                report=rep,
                raw_candidates=raw_cands,
                context_missing=context_missing,
            )
            debug_entry["visual_fallback_ok"] = int(bool(visual_fallback_ok))
            status_not_allowed = bool(
                policy.prov_new_allow_status
                and st not in policy.prov_new_allow_status
                and not known_blocked_ok
                and not visual_fallback_ok
            )
            debug_entry["status_not_allowed"] = int(bool(status_not_allowed))

            if policy.prov_new_require_context and not has_known_context and not visual_fallback_ok:
                self.append_postcreate_skip_debug(
                    debug_out=debug_out,
                    debug_entry=debug_entry,
                    reason="context_required",
                )
                continue

            focus_score_map = raw_score_map
            focus_temporal_scores = raw_temporal_scores
            focus_needs_sort = True
            focus = list(supported) if supported else (list(cands) if cands else (list(raw_cands) if visual_fallback_ok else []))
            if supported:
                debug_entry["focus_source"] = "supported"
                focus_score_map = score_map
                focus_temporal_scores = cand_temporal_scores
                focus_needs_sort = False
            elif cands:
                debug_entry["focus_source"] = "known_candidates"
                focus_score_map = score_map
                focus_temporal_scores = cand_temporal_scores
            elif raw_cands and visual_fallback_ok:
                debug_entry["focus_source"] = "raw_visual_fallback"
                focus_score_map = raw_score_map
                focus_temporal_scores = raw_temporal_scores
            if focus_needs_sort:
                focus.sort(
                    key=lambda c: float(focus_temporal_scores.get(id(c), 0.0)),
                    reverse=True,
                )
            if not focus:
                self.append_postcreate_skip_debug(
                    debug_out=debug_out,
                    debug_entry=debug_entry,
                    reason="no_focus_candidates",
                )
                continue

            best = focus[0]
            best_score = float(focus_temporal_scores.get(id(best), 0.0))
            best_oid = best.get("object_id", None)
            debug_entry["best_object_id"] = None if best_oid is None else int(best_oid)
            debug_entry["best_score"] = float(best_score)
            debug_candidates, debug_score_map, debug_temporal_scores = self.postcreate_debug_candidates_pack(
                known_candidates=cands,
                focus_candidates=focus,
                known_score_map=score_map,
                focus_score_map=focus_score_map,
                known_temporal_scores=cand_temporal_scores,
                focus_temporal_scores=focus_temporal_scores,
            )
            if best_score < float(policy.prov_new_min_top_score):
                self.append_postcreate_skip_debug(
                    debug_out=debug_out,
                    debug_entry=debug_entry,
                    reason="best_score_below_min",
                    candidates=debug_candidates,
                    best_score=best_score,
                    score_map=debug_score_map,
                    assigned_object_ids=assigned_object_ids,
                    support_known_ids=[],
                    temporal_scores_by_id=debug_temporal_scores,
                )
                continue

            support_known_ids: list[int] = []
            support_known_scores: dict[int, float] = {}
            blocked_known_ids: list[int] = []
            blocked_known_scores: dict[int, float] = {}
            for c in supported:
                score = float(cand_temporal_scores.get(id(c), 0.0))
                if score < float(policy.prov_new_min_candidate_score):
                    continue
                if not policy.prov_new_gap_allows(best_score, score):
                    continue
                oid = c.get("object_id", None)
                if oid is None:
                    continue
                oid = int(oid)
                support_known_ids.append(int(oid))
                support_known_scores[int(oid)] = float(score)
                if int(oid) in assigned_object_ids:
                    blocked_known_ids.append(int(oid))
                    blocked_known_scores[int(oid)] = float(score)
                if len(support_known_ids) >= int(policy.prov_new_support_topk):
                    break

            if policy.prov_new_require_context and not support_known_ids and not visual_fallback_ok and not known_blocked_ok:
                self.append_postcreate_skip_debug(
                    debug_out=debug_out,
                    debug_entry=debug_entry,
                    reason="no_support_context",
                    candidates=debug_candidates,
                    best_score=best_score,
                    score_map=debug_score_map,
                    assigned_object_ids=assigned_object_ids,
                    support_known_ids=[],
                    temporal_scores_by_id=debug_temporal_scores,
                )
                continue

            context_mode = policy.provisional_context_mode(
                support_known_ids=support_known_ids,
                blocked_known_ids=blocked_known_ids,
                visual_fallback_ok=visual_fallback_ok,
            )
            if not support_known_ids and known_blocked_ok and top_supported_oid is not None:
                oid = int(top_supported_oid)
                score = float(cand_temporal_scores.get(id(top_supported), 0.0))
                support_known_ids = [int(oid)]
                support_known_scores = {int(oid): float(score)}
                blocked_known_ids = [int(oid)]
                blocked_known_scores = {int(oid): float(score)}
                context_mode = "blocked_known"
            if not support_known_ids and visual_fallback_ok and raw_cands:
                (
                    support_known_ids,
                    support_known_scores,
                    blocked_known_ids,
                    blocked_known_scores,
                    context_mode,
                ) = policy.provisional_visual_parent_hint(
                    raw_candidates=raw_cands,
                    assigned_object_ids=assigned_object_ids,
                )
            (
                support_known_ids,
                support_known_scores,
                blocked_known_ids,
                blocked_known_scores,
                context_mode,
            ) = policy.promote_blocked_known_fallback_to_ambiguous_support(
                candidates=cands,
                score_map=score_map,
                assigned_object_ids=assigned_object_ids,
                support_known_ids=support_known_ids,
                support_known_scores=support_known_scores,
                blocked_known_ids=blocked_known_ids,
                blocked_known_scores=blocked_known_scores,
                context_mode=context_mode,
            )
            support = policy.make_provisional_support_profile(
                support_known_ids=support_known_ids,
                support_known_scores=support_known_scores,
                blocked_known_ids=blocked_known_ids,
                blocked_known_scores=blocked_known_scores,
                context_mode=context_mode,
            )
            debug_entry["context_mode"] = str(support.context_mode)
            debug_entry["support_mode"] = str(support.support_mode)
            debug_entry["relation"] = str(support.relation)
            debug_entry["support_known_ids"] = [int(x) for x in (support.support_known_ids or [])]
            debug_entry["support_known_scores"] = {
                int(k): float(v)
                for k, v in ((support.support_known_scores or {}).items())
                if k is not None and v is not None
            }
            debug_entry["blocked_known_ids"] = [int(x) for x in (support.blocked_known_ids or [])]
            debug_entry["blocked_known_scores"] = {
                int(k): float(v)
                for k, v in ((support.blocked_known_scores or {}).items())
                if k is not None and v is not None
            }
            debug_entry["related_known_ids"] = [int(x) for x in (support.related_known_ids or [])]
            debug_entry["related_known_scores"] = {
                int(k): float(v)
                for k, v in ((support.related_known_scores or {}).items())
                if k is not None and v is not None
            }
            debug_entry["candidate_rows"] = self.build_postcreate_candidate_debug_rows(
                candidates=debug_candidates,
                best_score=best_score,
                score_map=debug_score_map,
                assigned_object_ids=assigned_object_ids,
                support_known_ids=support.support_known_ids,
                temporal_scores_by_id=debug_temporal_scores,
            )
            provisional_parent_status_ok = bool(status_not_allowed or visual_fallback_ok)
            debug_entry["provisional_parent_status_ok"] = int(bool(provisional_parent_status_ok))
            provisional_parent_ok = bool(
                provisional_parent_status_ok
                and str(st).upper() == "STRONG"
                and str(support.support_mode or "") == "contextual"
                and policy.provisional_parent_alignment_ok(
                    support=support,
                    best_oid=(None if best_oid is None else int(best_oid)),
                    top_supported_oid=(None if top_supported_oid is None else int(top_supported_oid)),
                    top_raw_oid=(None if top_raw_oid is None else int(top_raw_oid)),
                )
            )
            debug_entry["provisional_parent_ok"] = int(bool(provisional_parent_ok))
            if len(support.support_known_ids) >= 2:
                decision = policy.make_ambiguous_decision_from_support(
                    class_id=int(getattr(rep, "class_id", -1)),
                    best_score=best_score,
                    support=support,
                )
                debug_entry["decision_kind"] = "AMBIG"
                debug_entry["decision_reason"] = str(decision.reason)
                ambiguous_out.append(decision.as_payload(det_id=int(did)))
                debug_out.append(debug_entry)
                continue
            if provisional_parent_ok:
                decision = policy.make_provisional_parent_decision(
                    class_id=int(getattr(rep, "class_id", -1)),
                    best_score=best_score,
                    support=support,
                )
                debug_entry["decision_kind"] = "PROV_PARENT"
                debug_entry["decision_reason"] = str(decision.reason)
                provisional_out.append(decision.as_payload(det_id=int(did)))
                debug_out.append(debug_entry)
                continue
            if status_not_allowed:
                self.append_postcreate_skip_debug(
                    debug_out=debug_out,
                    debug_entry=debug_entry,
                    reason="status_not_allowed",
                )
                continue
            decision = policy.make_provisional_decision(
                class_id=int(getattr(rep, "class_id", -1)),
                best_score=best_score,
                support=support,
                visual_fallback_ok=visual_fallback_ok,
            )
            debug_entry["decision_kind"] = "PROV"
            debug_entry["decision_reason"] = str(decision.reason)
            provisional_out.append(decision.as_payload(det_id=int(did)))
            debug_out.append(debug_entry)

        return {
            "ambiguous_entries": list(ambiguous_out),
            "provisional_entries": list(provisional_out),
            "debug_entries": list(debug_out),
        }

    def build_postcreate_candidate_debug_rows(
        self,
        *,
        candidates: list[dict],
        best_score: float,
        score_map: dict[int, float] | None,
        assigned_object_ids: set[int],
        support_known_ids: list[int],
        temporal_scores_by_id: dict[int, float] | None = None,
    ) -> list[dict]:
        policy = self.policy
        out: list[dict] = []
        support_ids = {int(x) for x in (support_known_ids or [])}
        cached_scores = temporal_scores_by_id if isinstance(temporal_scores_by_id, dict) else {}
        for candidate in (candidates or []):
            if not isinstance(candidate, dict):
                continue

            oid = candidate.get("object_id", None)
            oid_int = None if oid is None else int(oid)
            temp_score = cached_scores.get(id(candidate), None)
            if temp_score is None:
                temp_score = float(policy.temporal_candidate_score(candidate, score_map=score_map))
            else:
                temp_score = float(temp_score)
            min_ok = bool(temp_score >= float(policy.prov_new_min_candidate_score))
            gap_ok = bool(policy.prov_new_gap_allows(float(best_score), float(temp_score)))
            support_ok = bool(policy.candidate_has_ambiguous_support(candidate))
            blocked = bool(oid_int is not None and int(oid_int) in assigned_object_ids)

            why_bits: list[str] = []
            if oid_int is None:
                why_bits.append("no_oid")
            if not support_ok:
                why_bits.append("no_ctx")
            if not min_ok:
                why_bits.append("low")
            if not gap_ok:
                why_bits.append("gap")
            if int(candidate.get("known_plausible_keep", 0) or 0) != 1:
                why_bits.append("kp0")
            if int(candidate.get("decision_keep", 0) or 0) != 1:
                why_bits.append("dk0")
            if blocked:
                why_bits.append("blocked")
            if oid_int is not None and int(oid_int) in support_ids:
                why_bits.insert(0, "keep")
            if not why_bits:
                why_bits.append("ok")

            out.append(
                {
                    "object_id": oid_int,
                    "temp_score": float(temp_score),
                    "score_sim": float(candidate.get("score_sim", 0.0) or 0.0),
                    "score_final": float(candidate.get("score_final", temp_score) or 0.0),
                    "support_ctx": int(bool(support_ok)),
                    "blocked": int(bool(blocked)),
                    "support_final": int(bool(oid_int is not None and int(oid_int) in support_ids)),
                    "known_plausible_keep": int(candidate.get("known_plausible_keep", 0) or 0),
                    "decision_keep": int(candidate.get("decision_keep", 0) or 0),
                    "min_ok": int(bool(min_ok)),
                    "gap_ok": int(bool(gap_ok)),
                    "why": ",".join(why_bits),
                }
            )

        out.sort(
            key=lambda row: (
                int(row.get("support_final", 0) or 0),
                float(row.get("temp_score", 0.0) or 0.0),
                -1 if row.get("object_id", None) is None else int(row.get("object_id")),
            ),
            reverse=True,
        )
        return out
