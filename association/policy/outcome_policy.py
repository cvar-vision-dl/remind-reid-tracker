from __future__ import annotations

from dataclasses import dataclass, field

from association.policy.confidence_metrics import ConfidenceMetrics
from association.policy.outcome_candidate_runtime import OutcomeCandidateRuntime
from association.policy.outcome_postcreate_runtime import OutcomePostcreateRuntime
from association.policy.temporal_support_diagnostics import TemporalSupportDiagnostics
from association.reports import SimilarityReport
from utils.config import cfg_bool, cfg_float, cfg_get, cfg_int, cfg_str


@dataclass(frozen=True)
class TemporalSupportProfile:
    support_known_ids: list[int] = field(default_factory=list)
    support_known_scores: dict[int, float] = field(default_factory=dict)
    blocked_known_ids: list[int] = field(default_factory=list)
    blocked_known_scores: dict[int, float] = field(default_factory=dict)
    related_known_ids: list[int] = field(default_factory=list)
    related_known_scores: dict[int, float] = field(default_factory=dict)
    support_mode: str = "none"
    relation: str = "none"
    context_mode: str = "none"


@dataclass(frozen=True)
class TemporalDecision:
    kind: str
    class_id: int
    best_score: float
    reason: str
    score_gap: float = 0.0
    candidate_ids: list[int] = field(default_factory=list)
    candidate_scores: dict[int, float] = field(default_factory=dict)
    support: TemporalSupportProfile | None = None

    def as_payload(self, *, det_id: int) -> dict:
        payload = {
            "det_id": int(det_id),
            "class_id": int(self.class_id),
            "best_score": float(self.best_score),
            "reason": str(self.reason),
        }
        if self.kind == "ambiguous":
            payload["candidate_ids"] = [int(x) for x in (self.candidate_ids or [])]
            payload["candidate_scores"] = {
                int(k): float(v) for k, v in ((self.candidate_scores or {}).items())
            }
            payload["score_gap"] = float(self.score_gap)
            return payload

        support = self.support or TemporalSupportProfile()
        payload["support_known_ids"] = [int(x) for x in (support.support_known_ids or [])]
        payload["support_known_scores"] = {
            int(k): float(v) for k, v in ((support.support_known_scores or {}).items())
        }
        payload["blocked_known_ids"] = [int(x) for x in (support.blocked_known_ids or [])]
        payload["blocked_known_scores"] = {
            int(k): float(v) for k, v in ((support.blocked_known_scores or {}).items())
        }
        payload["related_known_ids"] = [int(x) for x in (support.related_known_ids or [])]
        payload["related_known_scores"] = {
            int(k): float(v) for k, v in ((support.related_known_scores or {}).items())
        }
        payload["context_mode"] = str(support.context_mode)
        return payload


class AssociationOutcomePolicy:
    """Apply ambiguity diagnostics and reinterpret resolver outcomes."""

    def __init__(self, *, config: dict, combiner, neighbor_sets_influence):
        self.config = config or {}
        self.combiner = combiner
        self.neighbor_sets_influence = neighbor_sets_influence

        self.match_thr = cfg_float(self.config, "association.matching.match_thr", 0.0)
        self.clear_margin = cfg_float(self.config, "association.matching.clear_margin", 0.07)

        self.amb_enabled = cfg_bool(self.config, "association.ambiguity.enabled", True)
        self.amb_close_delta = cfg_float(self.config, "association.ambiguity.close_delta", 0.03)
        self.amb_strong_gap = cfg_float(self.config, "association.ambiguity.strong_gap", self.clear_margin)
        self.amb_strong_min = cfg_float(self.config, "association.ambiguity.strong_min_score", self.match_thr)
        self.amb_amb_min = cfg_float(self.config, "association.ambiguity.ambiguous_min_score", float(self.match_thr))

        self.conf_enabled = cfg_bool(self.config, "association.confidence.enabled", True)
        self.conf_T = cfg_float(self.config, "association.confidence.temperature", 0.05)
        self.conf_cov_target = cfg_float(self.config, "association.confidence.coverage_target", 0.60)
        self.conf_eps = cfg_float(self.config, "association.confidence.eps", 1e-6)
        self.conf_min_temperature = cfg_float(self.config, "association.confidence.min_temperature", 0.05)
        self.conf_min_gap_k = cfg_float(self.config, "association.confidence.min_gap_k", 0.05)
        self.conf_margin_mode = cfg_str(self.config, "association.confidence.margin_mode", "sigmoid_abs")
        self.conf_gap_center = cfg_float(self.config, "association.confidence.gap_center", 0.10)
        self.conf_gap_k = cfg_float(self.config, "association.confidence.gap_k", 0.05)
        self.confidence_metrics = ConfidenceMetrics(
            combiner=self.combiner,
            conf_enabled=self.conf_enabled,
            conf_T=self.conf_T,
            conf_cov_target=self.conf_cov_target,
            conf_eps=self.conf_eps,
            conf_min_temperature=self.conf_min_temperature,
            conf_min_gap_k=self.conf_min_gap_k,
            conf_margin_mode=self.conf_margin_mode,
            conf_gap_center=self.conf_gap_center,
            conf_gap_k=self.conf_gap_k,
        )

        self.min_match_score = cfg_float(self.config, "update.min_match_score", 0.0)

        self.amb_track_enabled = cfg_bool(self.config, "association.ambiguous_tracks.enabled", False)
        self.amb_track_max_candidates = cfg_int(self.config, "association.ambiguous_tracks.max_candidates", 3, min_value=2)
        self.amb_track_min_top_score = cfg_float(self.config, "association.ambiguous_tracks.min_top_score", 0.45)
        self.amb_track_min_candidate_score = cfg_float(self.config, "association.ambiguous_tracks.min_candidate_score", 0.40)
        self.amb_track_gap_max = cfg_float(self.config, "association.ambiguous_tracks.gap_max", 0.05)
        self.amb_track_gap_rel_max = cfg_float(self.config, "association.ambiguous_tracks.gap_rel_max", 0.10)
        self.amb_track_min_compat_rel = cfg_float(self.config, "association.ambiguous_tracks.min_compat_rel", 0.65)
        self.amb_track_min_sets_score = cfg_float(self.config, "association.ambiguous_tracks.min_sets_score", 0.20)
        self.amb_track_require_context = cfg_bool(self.config, "association.ambiguous_tracks.require_context", True)
        self.amb_track_allow_override_match = cfg_bool(self.config, "association.ambiguous_tracks.allow_override_match", True)
        self.amb_track_gap_eps = cfg_float(self.config, "association.ambiguous_tracks.gap_eps", 1e-6)
        self.amb_track_supported_only = cfg_bool(self.config, "association.ambiguous_tracks.supported_only", True)
        self.amb_track_max_supported_candidates = max(
            2,
            cfg_int(self.config, "association.ambiguous_tracks.max_supported_candidates", self.amb_track_max_candidates),
        )
        self.amb_track_score_mode = cfg_str(self.config, "association.ambiguous_tracks.score_mode", "known_set").strip().lower()
        allow_status = cfg_get(self.config, "association.ambiguous_tracks.allow_status", ["WEAK", "AMBIGUOUS"])
        self.amb_track_allow_status = {str(x).upper() for x in (allow_status or [])}
        self.amb_visual_enabled = cfg_bool(self.config, "association.ambiguous_tracks.visual_fallback.enabled", False)
        self.amb_visual_require_context_missing = cfg_bool(
            self.config,
            "association.ambiguous_tracks.visual_fallback.require_context_missing",
            True,
        )
        self.amb_visual_min_top_score = cfg_float(
            self.config,
            "association.ambiguous_tracks.visual_fallback.min_top_score",
            self.amb_amb_min,
        )
        self.amb_visual_gap_max = cfg_float(
            self.config,
            "association.ambiguous_tracks.visual_fallback.gap_max",
            self.amb_close_delta,
        )
        self.amb_visual_gap_eps = cfg_float(
            self.config,
            "association.ambiguous_tracks.visual_fallback.gap_eps",
            self.amb_track_gap_eps,
        )
        self.amb_visual_max_candidates = max(
            2,
            cfg_int(
                self.config,
                "association.ambiguous_tracks.visual_fallback.max_candidates",
                self.amb_track_max_candidates,
            ),
        )

        self.prov_new_enabled = cfg_bool(self.config, "association.provisional_new.enabled", False)
        self.prov_new_min_top_score = cfg_float(self.config, "association.provisional_new.min_top_score", self.min_match_score)
        self.prov_new_min_top_score = max(0.0, min(1.0, self.prov_new_min_top_score))
        self.prov_new_min_candidate_score = cfg_float(
            self.config,
            "association.provisional_new.min_candidate_score",
            self.min_match_score,
        )
        self.prov_new_min_candidate_score = max(0.0, min(1.0, self.prov_new_min_candidate_score))
        self.prov_new_gap_max = cfg_float(
            self.config,
            "association.provisional_new.support_gap_max",
            cfg_float(self.config, "association.ambiguous_tracks.gap_max", 0.05),
        )
        self.prov_new_gap_rel_max = cfg_float(
            self.config,
            "association.provisional_new.support_gap_rel_max",
            cfg_float(self.config, "association.ambiguous_tracks.gap_rel_max", 0.10),
        )
        self.prov_new_gap_eps = cfg_float(
            self.config,
            "association.provisional_new.support_gap_eps",
            cfg_float(self.config, "association.ambiguous_tracks.gap_eps", 1e-6),
        )
        self.prov_new_support_topk = cfg_int(self.config, "association.provisional_new.support_topk", 3, min_value=1)
        self.prov_new_require_context = cfg_bool(self.config, "association.provisional_new.require_context", True)
        allow_status = cfg_get(self.config, "association.provisional_new.allow_status", ["WEAK", "AMBIGUOUS"])
        self.prov_new_allow_status = {str(x).upper() for x in (allow_status or [])}
        self.prov_new_visual_enabled = cfg_bool(self.config, "association.provisional_new.visual_fallback.enabled", False)
        self.prov_new_visual_eps = cfg_float(self.config, "association.provisional_new.visual_fallback.eps", 0.02)
        self.prov_new_visual_eps = max(0.0, min(1.0, self.prov_new_visual_eps))
        self.prov_new_visual_min_score = cfg_float(
            self.config,
            "association.provisional_new.visual_fallback.min_score",
            self.min_match_score,
        )
        self.prov_new_visual_min_score = max(0.0, min(1.0, self.prov_new_visual_min_score))
        self.prov_new_visual_min_gap = cfg_float(
            self.config,
            "association.provisional_new.visual_fallback.min_gap",
            self.clear_margin,
        )
        self.prov_new_visual_min_gap = max(0.0, min(1.0, self.prov_new_visual_min_gap))
        self.prov_new_visual_max_close = cfg_int(
            self.config,
            "association.provisional_new.visual_fallback.max_close",
            1,
            min_value=1,
        )
        self.prov_new_visual_single_candidate_only = cfg_bool(
            self.config,
            "association.provisional_new.visual_fallback.single_candidate_only",
            False,
        )
        self.prov_new_visual_require_context_missing = cfg_bool(
            self.config,
            "association.provisional_new.visual_fallback.require_context_missing",
            True,
        )
        self.temporal_support_diagnostics = TemporalSupportDiagnostics(
            amb_track_min_top_score=self.amb_track_min_top_score,
            prov_new_min_top_score=self.prov_new_min_top_score,
            amb_track_min_candidate_score=self.amb_track_min_candidate_score,
            prov_new_min_candidate_score=self.prov_new_min_candidate_score,
            amb_track_gap_max=self.amb_track_gap_max,
            amb_track_gap_rel_max=self.amb_track_gap_rel_max,
            amb_track_gap_eps=self.amb_track_gap_eps,
            prov_new_gap_max=self.prov_new_gap_max,
            prov_new_gap_rel_max=self.prov_new_gap_rel_max,
            prov_new_gap_eps=self.prov_new_gap_eps,
            temporal_candidate_score_fn=self.temporal_candidate_score,
        )
        self._candidate_runtime = OutcomeCandidateRuntime(combiner=self.combiner)
        self._postcreate_runtime = OutcomePostcreateRuntime(policy=self)
        self.reset_runtime_caches()

    def reset_runtime_caches(self) -> None:
        self._candidate_runtime.reset_runtime_caches()

    def annotate_similarity_ambiguity(self, reports_by_det_id: dict) -> dict[str, int]:
        strong = 0
        ambiguous = 0
        weak = 0
        for rep in (reports_by_det_id or {}).values():
            if not self.amb_enabled:
                rep.match_diag_sim = None
                continue
            rep.match_diag_sim = self.compute_match_diag(rep, key="score_sim")
            status = (rep.match_diag_sim or {}).get("status", None)
            if status == "STRONG":
                strong += 1
            elif status == "AMBIGUOUS":
                ambiguous += 1
            elif status == "WEAK":
                weak += 1
        return {"strong": int(strong), "ambiguous": int(ambiguous), "weak": int(weak)}

    def annotate_final_ambiguity(self, reports_by_det_id: dict) -> None:
        for rep in (reports_by_det_id or {}).values():
            if not self.amb_enabled:
                rep.match_diag_final = None
                continue
            rep.match_diag_final = self.compute_match_diag(rep, key="score_final", eligible_only=True)

    def annotate_reports_final_decisions(
        self,
        reports_by_det_id: dict,
        assigned_by_det_id: dict[int, int],
        created_det_ids: set[int],
        ambiguous_by_det_id: dict[int, dict],
        provisional_by_det_id: dict[int, dict],
        score_final_by_det_id: dict[int, float],
    ) -> None:
        for did, rep in (reports_by_det_id or {}).items():
            did = int(did)

            amb = (ambiguous_by_det_id or {}).get(int(did), None)
            if isinstance(amb, dict):
                rep.final_object_id = None
                rep.final_decision = "AMBIGUOUS_TRACK"
                rep.final_score = float(amb.get("best_score", 0.0) or 0.0)
                rep.final_reason = str(amb.get("reason", "KNOWN_BUT_AMBIGUOUS"))
                rep.ambiguous_candidate_ids = [int(x) for x in (amb.get("candidate_ids", []) or [])]
                rep.ambiguous_candidate_scores = {
                    int(k): float(v)
                    for k, v in ((amb.get("candidate_scores", {}) or {}).items())
                    if k is not None and v is not None
                }
                continue

            provisional = (provisional_by_det_id or {}).get(int(did), None)
            if isinstance(provisional, dict):
                provisional_reason = str(provisional.get("reason", "UNCERTAIN_NEW") or "UNCERTAIN_NEW")
                provisional_decision = "PROVISIONAL_PARENT" if provisional_reason.startswith("UNCERTAIN_PARENT") else "PROVISIONAL_NEW"
                rep.final_object_id = None
                rep.final_decision = str(provisional_decision)
                rep.final_score = float(provisional.get("best_score", 0.0) or 0.0)
                rep.final_reason = str(provisional_reason)
                rep.provisional_support_ids = [int(x) for x in (provisional.get("support_known_ids", []) or [])]
                rep.provisional_support_scores = {
                    int(k): float(v)
                    for k, v in ((provisional.get("support_known_scores", {}) or {}).items())
                    if k is not None and v is not None
                }
                rep.provisional_blocked_known_ids = [int(x) for x in (provisional.get("blocked_known_ids", []) or [])]
                rep.provisional_blocked_known_scores = {
                    int(k): float(v)
                    for k, v in ((provisional.get("blocked_known_scores", {}) or {}).items())
                    if k is not None and v is not None
                }
                rep.provisional_related_known_ids = list(rep.provisional_support_ids)
                rep.provisional_related_known_scores = dict(rep.provisional_support_scores)
                continue

            if did in created_det_ids:
                rep.final_object_id = None
                rep.final_decision = "NEW"
                rep.final_score = 0.0
                rep.final_reason = "CREATED_NEW"
                continue

            final_oid = assigned_by_det_id.get(int(did), None)
            if final_oid is None:
                rep.final_object_id = None
                rep.final_decision = "UNASSIGNED"
                rep.final_score = 0.0
                rep.final_reason = "NO_ASSIGNMENT"
                continue

            rep.final_object_id = int(final_oid)
            rep.final_decision = "MATCH"
            rep.final_reason = "ASSIGNED"
            rep.final_score = float(score_final_by_det_id.get(int(did), 0.0))

    def build_ambiguous_track_candidates(
        self,
        reports_by_det_id: dict,
        decided_matches: list[tuple[int, int, float]],
        to_create: list[dict],
        assigned_by_det_id: dict[int, int],
    ) -> list[dict]:
        if not self.amb_track_enabled:
            return []

        out: list[dict] = []
        matched_by_det = {
            int(det_id): {"object_id": int(obj_id), "score_final": float(score_final)}
            for det_id, obj_id, score_final in (decided_matches or [])
        }
        create_det_ids = set(int(x.get("det_id", -1)) for x in (to_create or []) if isinstance(x, dict))
        target_det_ids = sorted(set(create_det_ids) | set(matched_by_det.keys()))

        for did in target_det_ids:
            rep = (reports_by_det_id or {}).get(int(did), None)
            if rep is None:
                continue
            class_id = int(getattr(rep, "class_id", -1))

            matched = matched_by_det.get(int(did), None)
            status_allowed = True
            if matched is not None:
                if not self.amb_track_allow_override_match:
                    continue
                diag_temporal = self.compute_temporal_support_diag(rep, mode="ambiguous", scope="ambiguity")
                st = str((diag_temporal or {}).get("status", "")).upper()
                status_allowed = not bool(self.amb_track_allow_status and st not in self.amb_track_allow_status)

            all_cands = []
            for c in self.iter_candidates(rep, scope="ambiguity"):
                if not isinstance(c, dict):
                    continue
                oid = c.get("object_id", None)
                if oid is None:
                    continue
                all_cands.append(c)

            score_map_all = self.compute_comparable_score_map(all_cands)
            all_cands.sort(
                key=lambda c: self.temporal_candidate_score(c, score_map=score_map_all),
                reverse=True,
            )

            cands = []
            for c in all_cands:
                oid = c.get("object_id", None)
                if oid is None:
                    continue
                if int(oid) in (assigned_by_det_id or {}).values() and (matched is None or int(oid) != int(matched["object_id"])):
                    continue
                cands.append(c)

            allow_blocked_ambiguity = bool(
                matched is None
                and len(cands) < 2
                and len(self.ambiguous_supported_candidates(all_cands)) >= 2
            )
            if allow_blocked_ambiguity:
                cands = list(all_cands)

            if len(cands) < 2:
                continue

            score_map = self.compute_comparable_score_map(cands)
            cands.sort(
                key=lambda c: self.temporal_candidate_score(c, score_map=score_map),
                reverse=True,
            )
            supported = self.ambiguous_supported_candidates(cands)
            supported.sort(
                key=lambda c: self.temporal_candidate_score(c, score_map=score_map),
                reverse=True,
            )
            visual_plausible = self.ambiguous_visual_plausible_candidates(
                candidates=cands,
                supported=supported,
            )
            focus = self.ambiguous_focus_candidates(
                candidates=cands,
                supported=supported,
                visual_plausible=visual_plausible,
            )
            matched_focus_override = self.matched_context_override_focus_candidates(
                matched=matched,
                candidates=cands,
                supported=supported,
                score_map=score_map,
            )
            if matched_focus_override:
                focus = list(matched_focus_override)
            matched_known_conflict = self.matched_supported_rival_conflict_candidates(
                matched=matched,
                candidates=cands,
                supported=supported,
            )
            if matched_known_conflict:
                candidate_ids = [int(c["object_id"]) for c in matched_known_conflict if c.get("object_id", None) is not None]
                candidate_scores = {
                    int(c["object_id"]): float(self.temporal_candidate_score(c, score_map=score_map))
                    for c in matched_known_conflict
                    if c.get("object_id", None) is not None
                }
                ranked_conflict = sorted(
                    matched_known_conflict,
                    key=lambda c: self.temporal_candidate_score(c, score_map=score_map),
                    reverse=True,
                )
                s1_conflict = float(self.temporal_candidate_score(ranked_conflict[0], score_map=score_map))
                s2_conflict = (
                    float(self.temporal_candidate_score(ranked_conflict[1], score_map=score_map))
                    if len(ranked_conflict) > 1
                    else 0.0
                )
                decision = self.make_ambiguous_decision(
                    class_id=class_id,
                    best_score=s1_conflict,
                    score_gap=max(0.0, s1_conflict - s2_conflict),
                    candidate_ids=candidate_ids,
                    candidate_scores=candidate_scores,
                    matched=True,
                    supported_count=len(supported),
                )
                out.append(decision.as_payload(det_id=int(did)))
                continue
            if matched is not None and not status_allowed:
                continue
            if not focus:
                continue

            focus_score_map = self.compute_comparable_score_map(focus)
            diag_temporal = self.compute_temporal_support_diag_from_candidates(
                focus,
                mode="ambiguous",
                score_map=focus_score_map,
            )
            s1 = float((diag_temporal or {}).get("s1", 0.0) or 0.0)
            s2 = float((diag_temporal or {}).get("s2", 0.0) or 0.0)
            plausible = list((diag_temporal or {}).get("plausible_candidates", []) or [])
            if s1 < float(self.amb_track_min_top_score):
                continue
            if len(plausible) < 2:
                continue
            if len(plausible) < 2 or len(plausible) > int(self.amb_track_max_candidates):
                continue

            known_support = bool(len(supported) >= 1)
            visual_support = bool(len(visual_plausible) >= 2)
            if bool(self.amb_track_require_context) and not bool(known_support or visual_support):
                continue

            candidate_ids = [int(c["object_id"]) for c in plausible]
            candidate_scores = {
                int(c["object_id"]): float(self.temporal_candidate_score(c, score_map=focus_score_map))
                for c in plausible
            }
            decision = self.make_ambiguous_decision(
                class_id=class_id,
                best_score=s1,
                score_gap=max(0.0, s1 - s2),
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
                matched=matched is not None,
                supported_count=len(supported),
            )
            out.append(decision.as_payload(det_id=int(did)))

        return out

    def build_provisional_new_candidates(
        self,
        *,
        reports_by_det_id: dict,
        to_create: list[dict],
        assigned_by_det_id: dict[int, int] | None = None,
        excluded_det_ids: set[int] | None = None,
    ) -> list[dict]:
        decisions = self.build_postcreate_temporal_decisions(
            reports_by_det_id=reports_by_det_id,
            to_create=to_create,
            assigned_by_det_id=assigned_by_det_id,
            excluded_det_ids=excluded_det_ids,
        )
        return [dict(item) for item in ((decisions or {}).get("provisional_entries", []) or [])]

    def build_postcreate_temporal_decisions(
        self,
        *,
        reports_by_det_id: dict,
        to_create: list[dict],
        assigned_by_det_id: dict[int, int] | None = None,
        excluded_det_ids: set[int] | None = None,
    ) -> dict:
        return self._postcreate_runtime.build_postcreate_temporal_decisions(
            reports_by_det_id=reports_by_det_id,
            to_create=to_create,
            assigned_by_det_id=assigned_by_det_id,
            excluded_det_ids=excluded_det_ids,
        )

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
        return self._postcreate_runtime.build_postcreate_candidate_debug_rows(
            candidates=candidates,
            best_score=best_score,
            score_map=score_map,
            assigned_object_ids=assigned_object_ids,
            support_known_ids=support_known_ids,
            temporal_scores_by_id=temporal_scores_by_id,
        )

    def ambiguous_supported_candidates(self, candidates: list[dict]) -> list[dict]:
        return [c for c in (candidates or []) if self.candidate_has_ambiguous_support(c)]

    def make_ambiguous_decision(
        self,
        *,
        class_id: int,
        best_score: float,
        score_gap: float,
        candidate_ids: list[int],
        candidate_scores: dict[int, float],
        matched: bool,
        supported_count: int,
    ) -> TemporalDecision:
        support_mode = "contextual" if int(supported_count) >= 2 else "visual_fallback"
        if matched:
            reason = "KNOWN_BUT_AMBIGUOUS_MATCH" if support_mode == "contextual" else "KNOWN_BUT_AMBIGUOUS_VISUAL_MATCH"
        else:
            reason = "KNOWN_BUT_AMBIGUOUS" if support_mode == "contextual" else "KNOWN_BUT_AMBIGUOUS_VISUAL"
        return TemporalDecision(
            kind="ambiguous",
            class_id=int(class_id),
            best_score=float(best_score),
            score_gap=float(score_gap),
            candidate_ids=list(candidate_ids),
            candidate_scores=dict(candidate_scores),
            reason=str(reason),
        )

    def make_ambiguous_decision_from_support(
        self,
        *,
        class_id: int,
        best_score: float,
        support: TemporalSupportProfile,
    ) -> TemporalDecision:
        scored_ids = []
        for oid in (support.support_known_ids or []):
            oid = int(oid)
            score = float((support.support_known_scores or {}).get(int(oid), 0.0) or 0.0)
            scored_ids.append((float(score), int(oid)))
        scored_ids.sort(key=lambda item: (float(item[0]), int(item[1])), reverse=True)
        candidate_ids = [int(oid) for _, oid in scored_ids[: int(self.amb_track_max_candidates)]]
        candidate_scores = {
            int(oid): float((support.support_known_scores or {}).get(int(oid), 0.0) or 0.0)
            for oid in candidate_ids
        }
        top_scores = [float(candidate_scores.get(int(oid), 0.0) or 0.0) for oid in candidate_ids[:2]]
        score_gap = 0.0
        if len(top_scores) >= 2:
            score_gap = float(max(0.0, float(top_scores[0]) - float(top_scores[1])))

        blocked_ids = {int(x) for x in (support.blocked_known_ids or [])}
        if blocked_ids and blocked_ids.issuperset(set(candidate_ids)):
            reason = "KNOWN_BUT_AMBIGUOUS_BLOCKED_FALLBACK"
        else:
            reason = "KNOWN_BUT_AMBIGUOUS_CONTEXTUAL_FALLBACK"
        return TemporalDecision(
            kind="ambiguous",
            class_id=int(class_id),
            best_score=float(best_score),
            score_gap=float(score_gap),
            candidate_ids=list(candidate_ids),
            candidate_scores=dict(candidate_scores),
            reason=str(reason),
        )

    def make_provisional_support_profile(
        self,
        *,
        support_known_ids: list[int],
        support_known_scores: dict[int, float],
        blocked_known_ids: list[int],
        blocked_known_scores: dict[int, float],
        context_mode: str,
    ) -> TemporalSupportProfile:
        support_ids = [int(x) for x in (support_known_ids or [])]
        blocked_ids = [int(x) for x in (blocked_known_ids or [])]
        if support_ids:
            if blocked_ids and len(support_ids) == len(blocked_ids):
                support_mode = "blocked"
            else:
                support_mode = "contextual"
            relation = "known_new"
        elif "visual_fallback" in str(context_mode):
            support_mode = "visual_fallback"
            relation = "known_new"
        else:
            support_mode = "none"
            relation = "new_like"
        return TemporalSupportProfile(
            support_known_ids=list(support_ids),
            support_known_scores={int(k): float(v) for k, v in ((support_known_scores or {}).items())},
            blocked_known_ids=list(blocked_ids),
            blocked_known_scores={int(k): float(v) for k, v in ((blocked_known_scores or {}).items())},
            related_known_ids=list(support_ids),
            related_known_scores={int(k): float(v) for k, v in ((support_known_scores or {}).items())},
            support_mode=str(support_mode),
            relation=str(relation),
            context_mode=str(context_mode),
        )

    def make_provisional_decision(
        self,
        *,
        class_id: int,
        best_score: float,
        support: TemporalSupportProfile,
        visual_fallback_ok: bool,
    ) -> TemporalDecision:
        if support.blocked_known_ids:
            reason = "UNCERTAIN_KNOWN_BLOCKED"
        elif support.support_known_ids:
            reason = "UNCERTAIN_NEW_WITH_KNOWN_CONTEXT"
        else:
            reason = "UNCERTAIN_NEW_VISUAL_FALLBACK" if visual_fallback_ok else "UNCERTAIN_NEW"
        return TemporalDecision(
            kind="provisional",
            class_id=int(class_id),
            best_score=float(best_score),
            reason=str(reason),
            support=support,
        )

    def make_provisional_parent_decision(
        self,
        *,
        class_id: int,
        best_score: float,
        support: TemporalSupportProfile,
    ) -> TemporalDecision:
        return TemporalDecision(
            kind="provisional",
            class_id=int(class_id),
            best_score=float(best_score),
            reason="UNCERTAIN_PARENT",
            support=support,
        )

    def promote_blocked_known_fallback_to_ambiguous_support(
        self,
        *,
        candidates: list[dict],
        score_map: dict[int, float] | None,
        assigned_object_ids: set[int],
        support_known_ids: list[int],
        support_known_scores: dict[int, float],
        blocked_known_ids: list[int],
        blocked_known_scores: dict[int, float],
        context_mode: str,
    ) -> tuple[list[int], dict[int, float], list[int], dict[int, float], str]:
        support_ids = [int(x) for x in (support_known_ids or [])]
        if len(support_ids) >= 2 or not blocked_known_ids:
            return (
                support_ids,
                {int(k): float(v) for k, v in ((support_known_scores or {}).items())},
                [int(x) for x in (blocked_known_ids or [])],
                {int(k): float(v) for k, v in ((blocked_known_scores or {}).items())},
                str(context_mode),
            )

        support_scores = {int(k): float(v) for k, v in ((support_known_scores or {}).items())}
        blocked_ids = [int(x) for x in (blocked_known_ids or [])]
        blocked_scores = {int(k): float(v) for k, v in ((blocked_known_scores or {}).items())}
        min_secondary_score = float(max(self.prov_new_min_candidate_score, self.min_match_score))

        extras: list[tuple[float, int]] = []
        for c in (candidates or []):
            if not isinstance(c, dict):
                continue
            oid = c.get("object_id", None)
            if oid is None:
                continue
            oid = int(oid)
            if oid in support_ids:
                continue
            if int(c.get("known_plausible_keep", 0) or 0) != 1:
                continue

            score = float(self.temporal_candidate_score(c, score_map=score_map))
            if score < float(min_secondary_score):
                continue
            if not self.candidate_has_ambiguous_support(c):
                continue
            # If the best known candidate is blocked, prefer keeping known
            # ambiguity instead of degrading to "new-like". We do not require
            # `decision_keep`, because that gate already depends on row occupancy
            # in the frame and can hide the second plausible known candidate.
            extras.append((float(score), int(oid)))

        extras.sort(key=lambda item: (float(item[0]), int(item[1])), reverse=True)
        for score, oid in extras:
            support_ids.append(int(oid))
            support_scores[int(oid)] = float(score)
            if int(oid) in assigned_object_ids and int(oid) not in blocked_ids:
                blocked_ids.append(int(oid))
                blocked_scores[int(oid)] = float(score)
            if len(support_ids) >= int(self.prov_new_support_topk):
                break

        promoted_mode = str(context_mode)
        if len(support_ids) >= 2 and blocked_ids:
            promoted_mode = "blocked_known_ambiguous_fallback"

        return support_ids, support_scores, blocked_ids, blocked_scores, promoted_mode

    def ambiguous_visual_plausible_candidates(
        self,
        *,
        candidates: list[dict],
        supported: list[dict],
    ) -> list[dict]:
        visual_fallback_ok = bool(
            self.amb_visual_enabled
            and (not self.amb_visual_require_context_missing or not supported)
        )
        if not visual_fallback_ok:
            return []
        return self.visual_ambiguous_plausible_candidates(candidates)

    def ambiguous_focus_candidates(
        self,
        *,
        candidates: list[dict],
        supported: list[dict],
        visual_plausible: list[dict],
    ) -> list[dict]:
        if self.amb_track_supported_only:
            if len(supported) < 2 and len(visual_plausible) < 2:
                return []
            if len(supported) >= 2 and len(supported) > int(self.amb_track_max_supported_candidates):
                return []
            return list(supported) if len(supported) >= 2 else list(visual_plausible)

        if len(supported) >= 2:
            return list(supported)
        if len(visual_plausible) >= 2:
            return list(visual_plausible)
        return list(candidates)

    def matched_context_override_focus_candidates(
        self,
        *,
        matched: dict | None,
        candidates: list[dict],
        supported: list[dict],
        score_map: dict[int, float] | None,
    ) -> list[dict]:
        if not isinstance(matched, dict):
            return []

        matched_oid = matched.get("object_id", None)
        if matched_oid is None:
            return []
        matched_oid = int(matched_oid)

        matched_candidate = None
        for candidate in (candidates or []):
            if not isinstance(candidate, dict):
                continue
            oid = candidate.get("object_id", None)
            if oid is not None and int(oid) == int(matched_oid):
                matched_candidate = candidate
                break
        if matched_candidate is None:
            return []

        if self.candidate_has_ambiguous_support(matched_candidate):
            return []

        supported_rivals = []
        for candidate in (supported or []):
            if not isinstance(candidate, dict):
                continue
            oid = candidate.get("object_id", None)
            if oid is None or int(oid) == int(matched_oid):
                continue
            supported_rivals.append(candidate)
        if not supported_rivals:
            return []

        # If the selected winner has no contextual support but supported known
        # rivals are close, we want ambiguity reasoning to compare them together
        # instead of discarding the visual winner from the focus set.
        ranked_supported = sorted(
            supported_rivals,
            key=lambda c: self.temporal_candidate_score(c, score_map=score_map),
            reverse=True,
        )
        out = [matched_candidate]
        seen = {int(matched_oid)}
        for candidate in ranked_supported:
            if len(out) >= int(self.amb_track_max_candidates):
                break
            oid = candidate.get("object_id", None)
            if oid is None:
                continue
            oid = int(oid)
            if oid in seen:
                continue
            out.append(candidate)
            seen.add(int(oid))

        return out

    def matched_supported_rival_conflict_candidates(
        self,
        *,
        matched: dict | None,
        candidates: list[dict],
        supported: list[dict],
    ) -> list[dict]:
        if not isinstance(matched, dict):
            return []

        matched_oid = matched.get("object_id", None)
        if matched_oid is None:
            return []
        matched_oid = int(matched_oid)

        matched_candidate = None
        for candidate in (candidates or []):
            if not isinstance(candidate, dict):
                continue
            oid = candidate.get("object_id", None)
            if oid is not None and int(oid) == int(matched_oid):
                matched_candidate = candidate
                break
        if matched_candidate is None:
            return []
        if self.candidate_has_ambiguous_support(matched_candidate):
            return []

        matched_assign = float(
            matched_candidate.get(
                "score_assign",
                matched_candidate.get("score_sim", 0.0),
            )
            or 0.0
        )

        rivals: list[dict] = []
        for candidate in (supported or []):
            if not isinstance(candidate, dict):
                continue
            oid = candidate.get("object_id", None)
            if oid is None or int(oid) == int(matched_oid):
                continue
            rival_assign = float(candidate.get("score_assign", candidate.get("score_sim", 0.0)) or 0.0)
            if not self.candidate_gap_allows(
                top_score=max(float(matched_assign), float(rival_assign)),
                cand_score=min(float(matched_assign), float(rival_assign)),
                gap_max=max(float(self.amb_track_gap_max), float(self.clear_margin)),
                gap_rel_max=float(self.amb_track_gap_rel_max),
                gap_eps=float(self.amb_track_gap_eps),
            ):
                continue
            rivals.append(candidate)

        if not rivals:
            return []

        ranked_rivals = sorted(
            rivals,
            key=lambda c: float(c.get("score_assign", c.get("score_sim", 0.0)) or 0.0),
            reverse=True,
        )
        out = [matched_candidate]
        seen = {int(matched_oid)}
        for candidate in ranked_rivals:
            if len(out) >= int(self.amb_track_max_candidates):
                break
            oid = candidate.get("object_id", None)
            if oid is None:
                continue
            oid = int(oid)
            if oid in seen:
                continue
            out.append(candidate)
            seen.add(int(oid))
        return out if len(out) >= 2 else []

    def provisional_visual_fallback_ok(
        self,
        *,
        report: SimilarityReport,
        raw_candidates: list[dict],
        context_missing: bool,
    ) -> bool:
        if not self.prov_new_visual_enabled:
            return False
        if self.prov_new_visual_require_context_missing and not context_missing:
            return False

        diag_sim = self.compute_match_diag(report, key="score_sim", scope="ambiguity")
        if not diag_sim or float((diag_sim or {}).get("s1", 0.0) or 0.0) <= 0.0:
            diag_sim = self.compute_match_diag(report, key="score_sim")
        s1 = float((diag_sim or {}).get("s1", 0.0) or 0.0)
        gap = float((diag_sim or {}).get("gap", 0.0) or 0.0)
        n_close = int((diag_sim or {}).get("n_close", 0) or 0)
        near_to_match = bool(
            s1 >= float(self.prov_new_visual_min_score)
            and float(self.match_thr) > 0.0
            and (float(self.match_thr) - float(s1)) >= 0.0
            and (float(self.match_thr) - float(s1)) <= float(self.prov_new_visual_eps)
        )
        unique_top = bool(
            gap >= float(self.prov_new_visual_min_gap)
            and n_close <= int(self.prov_new_visual_max_close)
        )
        if self.prov_new_visual_single_candidate_only:
            unique_top = bool(unique_top and len(raw_candidates) == 1)
        return bool(near_to_match and unique_top)

    def provisional_context_mode(
        self,
        *,
        support_known_ids: list[int],
        blocked_known_ids: list[int],
        visual_fallback_ok: bool,
    ) -> str:
        if support_known_ids:
            if blocked_known_ids and len(support_known_ids) == len(blocked_known_ids):
                return "blocked_known"
            if len(support_known_ids) >= 2:
                return "strong_known"
            return "weak_known"
        if visual_fallback_ok:
            return "visual_fallback"
        return "none"

    def provisional_parent_alignment_ok(
        self,
        *,
        support: TemporalSupportProfile,
        best_oid: int | None,
        top_supported_oid: int | None,
        top_raw_oid: int | None,
    ) -> bool:
        support_ids = [int(x) for x in (support.support_known_ids or [])]
        if len(support_ids) != 1 or (support.blocked_known_ids or []):
            return False

        support_oid = int(support_ids[0])
        context_mode = str(support.context_mode or "")
        if "visual_fallback" in context_mode:
            return bool(best_oid is not None and int(support_oid) == int(best_oid))

        if top_supported_oid is None or top_raw_oid is None:
            return False
        return bool(
            int(support_oid) == int(top_supported_oid)
            and int(support_oid) == int(top_raw_oid)
        )

    def provisional_visual_parent_hint(
        self,
        *,
        raw_candidates: list[dict],
        assigned_object_ids: set[int],
    ) -> tuple[list[int], dict[int, float], list[int], dict[int, float], str]:
        raw_sorted = sorted(
            (c for c in (raw_candidates or []) if isinstance(c, dict) and c.get("object_id", None) is not None),
            key=lambda c: float(c.get("score_sim", 0.0) or 0.0),
            reverse=True,
        )
        if not raw_sorted:
            return [], {}, [], {}, "visual_fallback"

        top_visual = raw_sorted[0]
        oid = int(top_visual.get("object_id"))
        score = float(top_visual.get("score_sim", 0.0) or 0.0)
        support_known_ids = [int(oid)]
        support_known_scores = {int(oid): float(score)}
        if int(oid) in assigned_object_ids:
            blocked_known_ids = [int(oid)]
            blocked_known_scores = {int(oid): float(score)}
            context_mode = "visual_fallback_blocked_hint"
        else:
            blocked_known_ids = []
            blocked_known_scores = {}
            context_mode = "visual_fallback_known_hint"
        return (
            support_known_ids,
            support_known_scores,
            blocked_known_ids,
            blocked_known_scores,
            context_mode,
        )

    def amb_track_score(self, candidate: dict) -> float:
        return float(self.temporal_candidate_score(candidate))

    def candidate_has_ambiguous_support(self, candidate: dict) -> bool:
        if not isinstance(candidate, dict):
            return False

        compat_band = int(bool(candidate.get("compat_band", 0)))
        compat_rel = float(candidate.get("compat_rel", 0.0) or 0.0)
        kernel_raw = float(candidate.get("kernel_raw", 0.0) or 0.0)
        score_sets = float(candidate.get("score_sets", 0.0) or 0.0)
        support_sets = float(candidate.get("support_sets", 0.0) or 0.0)
        min_kernel_abs = float(getattr(self.neighbor_sets_influence, "support_min_kernel_abs", 0.0) or 0.0)
        # `compat_band`/`compat_rel` capture soft membership in the plausible
        # neighborhood, but by themselves can appear without enough coverage
        # for real contextual support. For final ambiguity we only treat as
        # "support" only what already carries effective positive set support.
        has_positive_sets_support = bool(support_sets > 1e-12 or score_sets > 1e-12)
        compat_rel_ok = bool(
            has_positive_sets_support
            and
            compat_rel >= float(self.amb_track_min_compat_rel)
            and kernel_raw >= float(min_kernel_abs)
        )
        compat_band_ok = bool(has_positive_sets_support and compat_band)

        return bool(
            compat_band_ok
            or compat_rel_ok
            or score_sets >= float(self.amb_track_min_sets_score)
        )

    def amb_track_gap_allows(self, top_score: float, cand_score: float) -> bool:
        return self.temporal_support_diagnostics.amb_track_gap_allows(
            top_score=top_score,
            cand_score=cand_score,
        )

    def prov_new_gap_allows(self, top_score: float, cand_score: float) -> bool:
        return self.temporal_support_diagnostics.prov_new_gap_allows(
            top_score=top_score,
            cand_score=cand_score,
        )

    def candidate_gap_allows(
        self,
        *,
        top_score: float,
        cand_score: float,
        gap_max: float,
        gap_rel_max: float,
        gap_eps: float,
    ) -> bool:
        return self.temporal_support_diagnostics.candidate_gap_allows(
            top_score=top_score,
            cand_score=cand_score,
            gap_max=gap_max,
            gap_rel_max=gap_rel_max,
            gap_eps=gap_eps,
        )

    def visual_ambiguous_plausible_candidates(self, candidates: list[dict]) -> list[dict]:
        scored: list[tuple[float, dict]] = []
        for c in (candidates or []):
            if not isinstance(c, dict):
                continue
            scored.append((float(c.get("score_sim", 0.0) or 0.0), c))
        if len(scored) < 2:
            return []

        scored.sort(key=lambda item: float(item[0]), reverse=True)
        top_score = float(scored[0][0])
        if top_score < float(self.amb_visual_min_top_score):
            return []

        plausible: list[dict] = []
        for score, candidate in scored:
            if len(plausible) >= int(self.amb_visual_max_candidates):
                break
            if self.candidate_gap_allows(
                top_score=top_score,
                cand_score=float(score),
                gap_max=self.amb_visual_gap_max,
                gap_rel_max=self.amb_track_gap_rel_max,
                gap_eps=self.amb_visual_gap_eps,
            ):
                plausible.append(candidate)
        return plausible if len(plausible) >= 2 else []

    def compute_temporal_support_diag(self, rep: SimilarityReport, *, mode: str, scope: str = "ambiguity") -> dict:
        cands = self.iter_candidates(rep, scope=scope)
        score_map = self.compute_comparable_score_map(cands)
        return self.compute_temporal_support_diag_from_candidates(cands, mode=mode, score_map=score_map)

    def compute_temporal_support_diag_from_candidates(
        self,
        candidates: list[dict],
        *,
        mode: str,
        score_map: dict[int, float] | None = None,
    ) -> dict:
        return self.temporal_support_diagnostics.compute_temporal_support_diag_from_candidates(
            candidates=candidates,
            mode=mode,
            score_map=score_map,
        )

    def iter_candidates(self, rep: SimilarityReport, *, scope: str = "raw") -> list[dict]:
        return self._candidate_runtime.iter_candidates(rep, scope=scope)

    def compute_comparable_score_map(self, candidates: list[dict]) -> dict[int, float]:
        return self._candidate_runtime.compute_comparable_score_map(candidates)

    def temporal_candidate_score(self, candidate: dict, *, score_map: dict[int, float] | None = None) -> float:
        if not isinstance(candidate, dict):
            return 0.0

        score_sim = None
        if isinstance(score_map, dict):
            score_sim = score_map.get(id(candidate), None)
        if score_sim is None:
            score_sim = float(candidate.get("score_sim", 0.0) or 0.0)

        score_final = float(candidate.get("score_final", score_sim) or 0.0)
        bonus_sets = float(candidate.get("bonus_sets_raw", candidate.get("bonus_sets", 0.0)) or 0.0)
        score_known = float(score_sim + bonus_sets)

        if self.amb_track_score_mode == "final":
            return float(score_final)
        return float(max(score_final, score_known))

    def compute_match_diag(self, rep: SimilarityReport, key: str = "score_sim", *, scope: str = "raw", eligible_only: bool | None = None) -> dict | None:
        if eligible_only is not None:
            scope = "eligible" if bool(eligible_only) else "raw"
        cands = self.iter_candidates(rep, scope=scope)
        if not isinstance(cands, list) or not cands:
            return {
                "s1": 0.0,
                "s2": 0.0,
                "gap": 0.0,
                "n_close": 0,
                "status": "WEAK",
                "reason": (
                    "NO_ELIGIBLE_CANDIDATES" if str(scope) == "eligible"
                    else ("NO_KNOWN_CANDIDATES" if str(scope) == "ambiguity" else "NO_CANDIDATES")
                ),
                "close_delta": float(self.amb_close_delta),
                "strong_gap": float(self.amb_strong_gap),
                "confidence": 0.0,
            }

        scores = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            scores.append(float(c.get(key, 0.0)))

        if not scores:
            return {
                "s1": 0.0,
                "s2": 0.0,
                "gap": 0.0,
                "n_close": 0,
                "status": "WEAK",
                "reason": "NO_SCORES",
                "close_delta": float(self.amb_close_delta),
                "strong_gap": float(self.amb_strong_gap),
                "confidence": 0.0,
            }

        scores_sorted = sorted(scores, reverse=True)
        s1 = float(scores_sorted[0])
        s2 = float(scores_sorted[1]) if len(scores_sorted) > 1 else 0.0
        gap = float(s1 - s2)

        delta = float(max(0.0, self.amb_close_delta))
        thr_close = float(s1 - delta)
        n_close = int(sum(1 for s in scores_sorted if float(s) >= thr_close))

        status = "WEAK"
        reason = "LOW_SIM"

        if s1 >= float(self.amb_strong_min) and gap >= float(self.amb_strong_gap) and n_close <= 1:
            status = "STRONG"
            reason = "CLEAR_WIN"
        elif s1 >= float(self.amb_amb_min):
            status = "AMBIGUOUS"
            reason = "HIGH_AMBIGUITY" if (gap < float(self.amb_strong_gap) or n_close > 1) else "MID"

        conf_pack = self.compute_confidence(rep, s1=s1, s2=s2, gap=gap, scores_sorted=scores_sorted)

        out = {
            "s1": float(s1),
            "s2": float(s2),
            "gap": float(gap),
            "n_close": int(n_close),
            "status": str(status),
            "reason": str(reason),
            "close_delta": float(self.amb_close_delta),
            "strong_gap": float(self.amb_strong_gap),
            "strong_min_score": float(self.amb_strong_min),
            "ambiguous_min_score": float(self.amb_amb_min),
        }
        out.update(conf_pack)
        return out

    def compute_confidence(
        self,
        rep: SimilarityReport,
        s1: float,
        s2: float,
        gap: float,
        scores_sorted: list[float],
    ) -> dict:
        return self.confidence_metrics.compute_confidence(
            rep=rep,
            s1=s1,
            s2=s2,
            gap=gap,
            scores_sorted=scores_sorted,
        )

    def compute_margin_factor(self, s1: float, gap: float, eps: float) -> tuple[float, float]:
        return self.confidence_metrics.compute_margin_factor(s1=s1, gap=gap, eps=eps)

    def sigmoid(self, x: float) -> float:
        return self.confidence_metrics.sigmoid(x)

    def compute_coverage_factor(self, rep: SimilarityReport) -> float:
        return self.confidence_metrics.compute_coverage_factor(rep)

    def softmax_top2_p1(self, s1: float, s2: float, T: float) -> float:
        return self.confidence_metrics.softmax_top2_p1(s1=s1, s2=s2, T=T)
