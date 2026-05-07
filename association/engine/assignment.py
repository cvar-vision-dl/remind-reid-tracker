# association/assignment.py

from __future__ import annotations

from association.engine.candidate_shaping import CandidateScoreShaper
from association.engine.assignment_path import AssignmentPathSupport
from association.models import AssignmentContext
from association.policy.candidate_score_policy import CandidateScorePolicy
from association.resolver.hungarian_resolver import HungarianResolver
from association.resolver.lock_resolver import LockResolver
from utils.config import cfg_bool, cfg_float, cfg_int


class HungarianAssigner:
    """
    Asignación det->obj usando Hungarian con dummies, locks y neighbor-sets influence.

    Convención de scores:
      - score_sim   : similitud pura (gating + locks)
      - score_assign: score estable para Hungarian; no incluye sets_rescue
      - bonus_sets  : ajuste contextual firmado por neighbor-sets (si aplica)
      - score_final : score_sim + bonus_sets
    """

    def __init__(self, config: dict, memory_store, trace_collector=None):
        self.config = config or {}
        self.memory_store = memory_store
        self.assignment_path_support = AssignmentPathSupport()
        self.trace_collector = trace_collector
        self._trace_frame_id: int | None = None

        self.enable_dummies = cfg_bool(self.config, "association.matching.hungarian.enable_dummies", True)
        self.dummy_score = cfg_float(self.config, "association.matching.hungarian.dummy_score", 0.0, min_value=0.0, max_value=1.0)

        self.use_confidence_dummy = cfg_bool(self.config, "association.matching.hungarian.use_confidence_dummy", True)
        self.conf_alpha = cfg_float(self.config, "association.matching.hungarian.conf_alpha", 0.15, min_value=0.0)
        self.dummy_score_cap = cfg_float(self.config, "association.matching.hungarian.dummy_score_cap", 0.8, min_value=0.0, max_value=1.0)

        self.gate_by_match_thr = cfg_bool(self.config, "association.matching.hungarian.gate_by_match_thr", True)
        self.gate_by_min_match = cfg_bool(self.config, "association.matching.hungarian.gate_by_min_match_score", True)

        self.locks_enabled = cfg_bool(self.config, "association.matching.hungarian.locks.enabled", True)
        self.locks_object_enabled = cfg_bool(self.config, "association.matching.hungarian.locks.object_enabled", True)
        self.locks_det_enabled = cfg_bool(self.config, "association.matching.hungarian.locks.det_enabled", True)

        self.locks_thr = cfg_float(self.config, "association.matching.hungarian.locks.thr", 0.90, min_value=0.0, max_value=1.0)
        self.locks_gap_abs_min = cfg_float(self.config, "association.matching.hungarian.locks.gap_abs_min", 0.03, min_value=0.0)
        self.locks_gap_rel_thr = cfg_float(self.config, "association.matching.hungarian.locks.gap_rel_thr", 0.06, min_value=0.0)

        self.ctx_veto_enabled = cfg_bool(self.config, "association.matching.neighbor_sets_context_veto.enabled", True)
        self.ctx_veto_supported_max = cfg_int(self.config, "association.matching.neighbor_sets_context_veto.supported_max", 3, min_value=1)
        self.ctx_veto_min_quality = cfg_float(self.config, "association.matching.neighbor_sets_context_veto.min_quality", 0.60, min_value=0.0, max_value=1.0)
        self.ctx_veto_min_pruning = cfg_float(self.config, "association.matching.neighbor_sets_context_veto.min_pruning", 0.35, min_value=0.0, max_value=1.0)
        self.ctx_veto_min_class_strength = cfg_float(self.config, "association.matching.neighbor_sets_context_veto.min_class_strength", 0.50, min_value=0.0, max_value=1.0)
        self.ctx_veto_max_compat_rel = cfg_float(self.config, "association.matching.neighbor_sets_context_veto.max_compat_rel", 0.10, min_value=0.0, max_value=1.0)
        self.ctx_veto_max_score_sets = cfg_float(self.config, "association.matching.neighbor_sets_context_veto.max_score_sets", 0.05, min_value=0.0, max_value=1.0)
        self.ctx_veto_local_enabled = cfg_bool(self.config, "association.matching.neighbor_sets_context_veto.local.enabled", True)
        self.ctx_veto_local_min_quality = cfg_float(self.config, "association.matching.neighbor_sets_context_veto.local.min_quality", 0.45, min_value=0.0, max_value=1.0)
        self.ctx_veto_local_min_episodes = cfg_int(self.config, "association.matching.neighbor_sets_context_veto.local.min_episodes", 4, min_value=1)
        self.ctx_veto_local_min_kernel_size = cfg_int(self.config, "association.matching.neighbor_sets_context_veto.local.min_kernel_size", 3, min_value=1)
        self.ctx_veto_local_min_expected_neighbors = cfg_int(self.config, "association.matching.neighbor_sets_context_veto.local.min_expected_neighbors", 3, min_value=1)
        self.ctx_veto_local_max_hit_ratio = cfg_float(self.config, "association.matching.neighbor_sets_context_veto.local.max_hit_ratio", 0.10, min_value=0.0, max_value=1.0)
        self.ctx_veto_local_expected_mass_target = cfg_float(self.config, "association.matching.neighbor_sets_context_veto.local.expected_mass_target", 0.75, min_value=0.0, max_value=1.0)
        self.ctx_veto_local_expected_topk_scale = cfg_float(self.config, "association.matching.neighbor_sets_context_veto.local.expected_topk_scale", 2.0, min_value=1.0)
        self.ctx_veto_local_require_supported_alternative = cfg_bool(
            self.config,
            "association.matching.neighbor_sets_context_veto.local.require_supported_alternative",
            True,
        )

        self.debug_assoc_enabled = cfg_bool(self.config, "debug.association.enabled", False)
        self.lock_resolver = LockResolver(
            locks_enabled=self.locks_enabled,
            locks_object_enabled=self.locks_object_enabled,
            locks_det_enabled=self.locks_det_enabled,
            locks_thr=self.locks_thr,
            locks_gap_abs_min=self.locks_gap_abs_min,
            locks_gap_rel_thr=self.locks_gap_rel_thr,
        )
        self.hungarian_resolver = HungarianResolver(
            enable_dummies=self.enable_dummies,
        )
        self.candidate_score_policy = CandidateScorePolicy(
            gate_by_match_thr=self.gate_by_match_thr,
            gate_by_min_match=self.gate_by_min_match,
            debug_assoc_enabled=self.debug_assoc_enabled,
            dummy_score=self.dummy_score,
            use_confidence_dummy=self.use_confidence_dummy,
            conf_alpha=self.conf_alpha,
            dummy_score_cap=self.dummy_score_cap,
            ctx_veto_enabled=self.ctx_veto_enabled,
            ctx_veto_supported_max=self.ctx_veto_supported_max,
            ctx_veto_min_quality=self.ctx_veto_min_quality,
            ctx_veto_min_pruning=self.ctx_veto_min_pruning,
            ctx_veto_min_class_strength=self.ctx_veto_min_class_strength,
            ctx_veto_max_compat_rel=self.ctx_veto_max_compat_rel,
            ctx_veto_max_score_sets=self.ctx_veto_max_score_sets,
            ctx_veto_local_enabled=self.ctx_veto_local_enabled,
            ctx_veto_local_min_quality=self.ctx_veto_local_min_quality,
            ctx_veto_local_min_episodes=self.ctx_veto_local_min_episodes,
            ctx_veto_local_min_kernel_size=self.ctx_veto_local_min_kernel_size,
            ctx_veto_local_min_expected_neighbors=self.ctx_veto_local_min_expected_neighbors,
            ctx_veto_local_max_hit_ratio=self.ctx_veto_local_max_hit_ratio,
            ctx_veto_local_expected_mass_target=self.ctx_veto_local_expected_mass_target,
            ctx_veto_local_expected_topk_scale=self.ctx_veto_local_expected_topk_scale,
            ctx_veto_local_require_supported_alternative=self.ctx_veto_local_require_supported_alternative,
        )
        self.candidate_score_shaper = CandidateScoreShaper(self.candidate_score_policy)

    def set_trace_frame(self, frame_id: int | None) -> None:
        self._trace_frame_id = None if frame_id is None else int(frame_id)

    def clear_trace_frame(self) -> None:
        self._trace_frame_id = None

    def trace_scope_key(self, *, class_id: int) -> dict:
        return {
            "frame_id": None if self._trace_frame_id is None else int(self._trace_frame_id),
            "class_id": int(class_id),
        }

    def trace_allow_for_report(
        self,
        *,
        class_id: int,
        det_ids: list[int],
        reports: dict,
        snapshot_ids: set[int],
        neighbor_sets_influence,
        ns_ctx: dict | None,
        used_obj_ids: set[int] | None = None,
        stage: str = "pre_locks",
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        used_obj_ids = {int(x) for x in (used_obj_ids or set())}
        stage = str(stage or "pre_locks")
        class_object_ids = sorted(
            int(getattr(obj, "object_id", -1))
            for obj in (self.memory_store.get_by_class(int(class_id)) or [])
            if getattr(obj, "object_id", None) is not None
        )
        scope_key = self.trace_scope_key(class_id=int(class_id))
        participants = {
            "det_ids": [int(did) for did in (det_ids or [])],
            "object_ids": list(class_object_ids),
        }
        self.trace_collector.enter_node(
            "shape.allow_for_report",
            scope_key,
            participants=participants,
        )

        enabled = bool(neighbor_sets_influence is not None and isinstance(ns_ctx, dict) and ns_ctx.get("enabled", False))
        det_count = int(len(det_ids or []))
        if not enabled:
            for det_id in (det_ids or []):
                rep = reports.get(int(det_id), None)
                status = self.report_status(rep) if rep is not None else ""
                self.trace_collector.add_detection_row(
                    "shape.allow_for_report",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "report_status": str(status),
                        "allowed": False,
                        "reason": "CONTEXT_DISABLED",
                    },
                )
            self.trace_collector.set_values(
                "shape.allow_for_report",
                scope_key,
                {
                    "class_id": int(class_id),
                    "stage": str(stage),
                    "class_context_available": False,
                    "detection_count": int(det_count),
                    "allowed_count": 0,
                    "blocked_count": int(det_count),
                    "used_object_ids_count": int(len(used_obj_ids)),
                },
            )
            self.trace_collector.set_decision(
                "shape.allow_for_report",
                scope_key,
                {
                    "status": "N/A",
                    "branch": "context_disabled",
                },
            )
            self.trace_collector.leave_node("shape.allow_for_report", scope_key)
            return

        allowed_count = 0
        for det_id in (det_ids or []):
            rep = reports.get(int(det_id), None)
            status = self.report_status(rep) if rep is not None else ""
            allowed_by_report_status = bool(neighbor_sets_influence.allow_for_report(rep)) if rep is not None else False
            allowed = bool(allowed_by_report_status)
            allowed_by_used_overlap = False
            report_candidate_ids: set[int] = set()
            if not allowed and rep is not None and used_obj_ids:
                report_candidate_ids = {
                    int(c.get("object_id"))
                    for c in (getattr(rep, "candidates", None) or [])
                    if isinstance(c, dict) and c.get("object_id", None) is not None
                }
                if report_candidate_ids & used_obj_ids:
                    allowed = True
                    allowed_by_used_overlap = True
            if allowed:
                allowed_count += 1
            self.trace_collector.add_detection_row(
                "shape.allow_for_report",
                scope_key,
                {
                    "det_id": int(det_id),
                    "stage": str(stage),
                    "report_status": str(status),
                    "allowed": bool(allowed),
                    "allowed_by_used_object_overlap": bool(allowed_by_used_overlap),
                    "used_object_ids_count": int(len(used_obj_ids)),
                    "reason": (
                        "REPORT_ALLOWED_USED_OBJECT_OVERLAP"
                        if allowed_by_used_overlap
                        else ("REPORT_ALLOWED" if allowed else "REPORT_NOT_ALLOWED")
                    ),
                    "checks": [
                        {
                            "id": "allow_for_report.status",
                            "label": "report_status_allowed",
                            "lhs": str(status),
                            "op": "allow_for_report",
                            "rhs": True,
                            "passed": bool(allowed_by_report_status),
                            "reason": (
                                "REPORT_ALLOWED"
                                if allowed_by_report_status
                                else "REPORT_NOT_ALLOWED"
                            ),
                            "effect": (
                                "use_context"
                                if allowed_by_report_status
                                else "evaluate_used_object_overlap"
                            ),
                        },
                        {
                            "id": "allow_for_report.used_object_overlap",
                            "label": "candidate_overlaps_used_object",
                            "lhs": int(len(report_candidate_ids & used_obj_ids)),
                            "op": ">",
                            "rhs": 0,
                            "passed": bool(allowed_by_used_overlap),
                            "reason": (
                                "REPORT_ALLOWED_USED_OBJECT_OVERLAP"
                                if allowed_by_used_overlap
                                else "NO_USED_OBJECT_OVERLAP"
                            ),
                            "effect": (
                                "use_context_via_overlap_override"
                                if allowed_by_used_overlap
                                else "no_overlap_override"
                            ),
                        }
                    ],
                },
            )

        self.trace_collector.set_values(
            "shape.allow_for_report",
            scope_key,
            {
                "class_id": int(class_id),
                "stage": str(stage),
                "class_context_available": True,
                "detection_count": int(det_count),
                "allowed_count": int(allowed_count),
                "blocked_count": int(max(0, det_count - allowed_count)),
                "used_object_ids_count": int(len(used_obj_ids)),
            },
        )
        self.trace_collector.set_decision(
            "shape.allow_for_report",
            scope_key,
            {
                "status": "PASS",
                "branch": f"evaluated_{stage}",
            },
        )
        self.trace_collector.leave_node("shape.allow_for_report", scope_key)

    def trace_valid_detections(
        self,
        *,
        class_id: int,
        det_ids: list[int],
        valid_det_ids: list[int],
        create_entries: list[tuple[int, int]],
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        class_object_ids = sorted(
            int(getattr(obj, "object_id", -1))
            for obj in (self.memory_store.get_by_class(int(class_id)) or [])
            if getattr(obj, "object_id", None) is not None
        )
        valid_set = {int(det_id) for det_id in (valid_det_ids or [])}
        missing_feature_set = {int(det_id) for det_id, create_class_id in (create_entries or []) if int(create_class_id) == int(class_id)}
        scope_key = self.trace_scope_key(class_id=int(class_id))
        participants = {
            "det_ids": [int(did) for did in (det_ids or [])],
            "object_ids": list(class_object_ids),
        }
        self.trace_collector.enter_node(
            "prepare.valid_detections",
            scope_key,
            participants=participants,
        )

        for det_id in (det_ids or []):
            valid = int(det_id) in valid_set
            reason = "VALID_FOR_MATCHING" if valid else ("NO_FEATURES" if int(det_id) in missing_feature_set else "FILTERED_OUT")
            self.trace_collector.add_detection_row(
                "prepare.valid_detections",
                scope_key,
                {
                    "det_id": int(det_id),
                    "valid": bool(valid),
                    "reason": str(reason),
                    "checks": [
                        {
                            "id": "valid_detections.has_features",
                            "label": "detection_has_features",
                            "lhs": bool(valid),
                            "op": "==",
                            "rhs": True,
                            "passed": bool(valid),
                            "reason": str(reason),
                            "effect": "keep_for_assignment" if valid else "route_to_create_or_skip",
                        }
                    ],
                },
            )

        self.trace_collector.set_values(
            "prepare.valid_detections",
            scope_key,
            {
                "class_id": int(class_id),
                "detection_count": int(len(det_ids or [])),
                "valid_det_ids": sorted(int(x) for x in valid_set),
                "valid_count": int(len(valid_set)),
                "missing_feature_det_ids": sorted(int(x) for x in missing_feature_set),
                "missing_feature_count": int(len(missing_feature_set)),
            },
        )
        self.trace_collector.set_decision(
            "prepare.valid_detections",
            scope_key,
            {
                "status": "PASS" if valid_set else "N/A",
                "branch": "has_valid_detections" if valid_set else "no_valid_detections",
            },
        )
        self.trace_collector.leave_node("prepare.valid_detections", scope_key)

    def trace_skip_node_for_class(
        self,
        *,
        node_id: str,
        class_id: int,
        reason: str,
        det_ids: list[int] | None = None,
        object_ids: list[int] | set[int] | None = None,
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return
        class_object_ids = sorted(
            int(getattr(obj, "object_id", -1))
            for obj in (self.memory_store.get_by_class(int(class_id)) or [])
            if getattr(obj, "object_id", None) is not None
        )
        participant_object_ids = (
            [int(oid) for oid in (object_ids or [])]
            if object_ids is not None
            else list(class_object_ids)
        )
        self.trace_collector.skip_node(
            node_id,
            self.trace_scope_key(class_id=int(class_id)),
            str(reason),
            participants={
                "det_ids": [int(did) for did in (det_ids or [])],
                "object_ids": sorted({int(oid) for oid in participant_object_ids}),
            },
        )

    def trace_context_veto(
        self,
        *,
        class_id: int,
        det_ids: list[int],
        reports: dict,
        snapshot_ids: set[int],
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        class_object_ids = sorted(
            int(getattr(obj, "object_id", -1))
            for obj in (self.memory_store.get_by_class(int(class_id)) or [])
            if getattr(obj, "object_id", None) is not None
        )
        scope_key = self.trace_scope_key(class_id=int(class_id))
        participants = {
            "det_ids": [int(did) for did in (det_ids or [])],
            "object_ids": list(class_object_ids),
        }
        self.trace_collector.enter_node(
            "shape.context_veto",
            scope_key,
            participants=participants,
        )

        candidate_count = 0
        kept_count = 0
        vetoed_count = 0

        for det_id in (det_ids or []):
            rep = reports.get(int(det_id), None)
            if rep is None:
                continue
            for candidate in (getattr(rep, "candidates", None) or []):
                if not isinstance(candidate, dict):
                    continue
                object_id = candidate.get("object_id", None)
                if object_id is None:
                    continue
                object_id = int(object_id)
                if object_id not in snapshot_ids:
                    continue
                candidate_count += 1
                veto_reason = str(candidate.get("known_plausible_reason", "") or "")
                policy = ((candidate.get("sets_trace", {}) or {}).get("policy", {}) or {})
                raw_veto_reason = str(policy.get("veto_reason", "") or veto_reason)
                gate_reason = str(policy.get("gate_reason", "") or "")
                kept = int(candidate.get("decision_keep", 0) or 0) == 1
                if kept:
                    kept_count += 1
                elif raw_veto_reason:
                    vetoed_count += 1

                self.trace_collector.add_candidate_row(
                    "shape.context_veto",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "object_id": int(object_id),
                        "known_plausible_keep": int(candidate.get("known_plausible_keep", 0) or 0),
                        "known_plausible_reason": str(candidate.get("known_plausible_reason", "") or ""),
                        "ctx_keep": int(candidate.get("ctx_keep", 0) or 0),
                        "decision_keep": int(candidate.get("decision_keep", 0) or 0),
                        "veto_reason": str(raw_veto_reason),
                        "gate_reason": str(gate_reason),
                        "score_sim": float(candidate.get("score_sim", 0.0) or 0.0),
                        "score_sets": float(candidate.get("score_sets", 0.0) or 0.0),
                        "bonus_sets": float(candidate.get("bonus_sets", 0.0) or 0.0),
                        "support_sets": float(candidate.get("support_sets", 0.0) or 0.0),
                        "support_local_sets": float(candidate.get("support_local_sets", 0.0) or 0.0),
                        "support_global_sets": float(candidate.get("support_global_sets", 0.0) or 0.0),
                        "quality_sets": float(candidate.get("quality_sets", 0.0) or 0.0),
                        "compat_rel": float(candidate.get("compat_rel", 0.0) or 0.0),
                        "compat_band": int(candidate.get("compat_band", 0) or 0),
                        "kernel_raw": float(candidate.get("kernel_raw", 0.0) or 0.0),
                        "kernel_hit_count": int(candidate.get("kernel_hit_count", 0) or 0),
                        "kernel_hit_ratio": float(candidate.get("kernel_hit_ratio", 0.0) or 0.0),
                        "kernel_rel": float(candidate.get("kernel_rel", 0.0) or 0.0),
                        "hyp_rel": float(candidate.get("hyp_rel", 0.0) or 0.0),
                        "shortlist_hit": bool(candidate.get("shortlist_hit", False)),
                        "supported_hit": bool(candidate.get("supported_hit", False)),
                        "soft_supported_hit": bool(candidate.get("soft_supported_hit", False)),
                        "local_ctx_has_supported_alternative": int(candidate.get("local_ctx_has_supported_alternative", 0) or 0),
                        "local_ctx_episode_count": int(candidate.get("local_ctx_episode_count", 0) or 0),
                        "local_ctx_kernel_source": str(candidate.get("local_ctx_kernel_source", "") or ""),
                        "local_ctx_kernel_size": int(candidate.get("local_ctx_kernel_size", 0) or 0),
                        "local_ctx_frame_kernel_size": int(candidate.get("local_ctx_frame_kernel_size", 0) or 0),
                        "local_ctx_expected_count": int(candidate.get("local_ctx_expected_count", 0) or 0),
                        "local_ctx_hit_count": int(candidate.get("local_ctx_hit_count", 0) or 0),
                        "local_ctx_hit_ratio": float(candidate.get("local_ctx_hit_ratio", 0.0) or 0.0),
                        "local_ctx_maturity": float(candidate.get("local_ctx_maturity", 0.0) or 0.0),
                        "score_assign": float(candidate.get("score_assign", 0.0) or 0.0),
                        "score_final": float(candidate.get("score_final", 0.0) or 0.0),
                        "checks": [
                            {
                                "id": "context_veto.kept",
                                "label": "candidate_kept_after_context_veto",
                                "lhs": int(1 if kept else 0),
                                "op": "==",
                                "rhs": 1,
                                "passed": bool(kept),
                                "reason": str("KEPT" if kept else (gate_reason or raw_veto_reason or "VETOED")),
                                "effect": "keep_candidate" if kept else "drop_candidate",
                            }
                        ],
                    },
                )

        self.trace_collector.set_values(
            "shape.context_veto",
            scope_key,
            {
                "class_id": int(class_id),
                "candidate_count": int(candidate_count),
                "kept_count": int(kept_count),
                "vetoed_count": int(vetoed_count),
            },
        )
        self.trace_collector.set_decision(
            "shape.context_veto",
            scope_key,
            {
                "status": "PASS",
                "branch": "filtered",
            },
        )
        self.trace_collector.leave_node("shape.context_veto", scope_key)

    def trace_final_score_tables(
        self,
        *,
        class_id: int,
        det_ids: list[int],
        reports: dict,
        table_sim: dict[int, dict[int, float]],
        table_assign: dict[int, dict[int, float]],
        table_final: dict[int, dict[int, float]],
        candidate_obj_ids: set[int] | list[int],
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        object_ids = sorted(int(x) for x in (candidate_obj_ids or []))
        scope_key = self.trace_scope_key(class_id=int(class_id))
        participants = {
            "det_ids": [int(did) for did in (det_ids or [])],
            "object_ids": list(object_ids),
        }
        self.trace_collector.enter_node(
            "shape.final_score_tables",
            scope_key,
            participants=participants,
        )

        row_count = 0
        for det_id in (det_ids or []):
            rep = reports.get(int(det_id), None)
            candidate_by_oid = {
                int(candidate.get("object_id")): candidate
                for candidate in (getattr(rep, "candidates", None) or [])
                if isinstance(candidate, dict) and candidate.get("object_id", None) is not None
            }
            row_final = {
                int(object_id): float(score)
                for object_id, score in ((table_final.get(int(det_id), {}) or {}).items())
            }
            row_assign = {
                int(object_id): float(score)
                for object_id, score in ((table_assign.get(int(det_id), {}) or {}).items())
            }
            row_sim = {
                int(object_id): float(score)
                for object_id, score in ((table_sim.get(int(det_id), {}) or {}).items())
            }
            ranked = sorted(
                row_final.items(),
                key=lambda item: (float(item[1]), int(item[0])),
                reverse=True,
            )
            row_count += int(len(ranked))
            best_oid = int(ranked[0][0]) if ranked else None
            best_score = float(ranked[0][1]) if ranked else None
            self.trace_collector.add_detection_row(
                "shape.final_score_tables",
                scope_key,
                {
                    "det_id": int(det_id),
                    "candidate_count": int(len(ranked)),
                    "best_object_id": best_oid,
                    "best_score_final": best_score,
                    "checks": [
                        {
                            "id": "final_scores.has_rows",
                            "label": "final_score_rows_available",
                            "lhs": int(len(ranked)),
                            "op": ">",
                            "rhs": 0,
                            "passed": bool(ranked),
                            "reason": "FINAL_ROWS_READY" if ranked else "NO_FINAL_ROWS",
                            "effect": "continue_to_resolution" if ranked else "no_resolution_candidates",
                        }
                    ],
                },
            )
            for rank, (object_id, score_final) in enumerate(ranked, start=1):
                candidate = candidate_by_oid.get(int(object_id), {}) if isinstance(candidate_by_oid, dict) else {}
                self.trace_collector.add_candidate_row(
                    "shape.final_score_tables",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "object_id": int(object_id),
                        "rank": int(rank),
                        "score_sim": float(row_sim.get(int(object_id), 0.0) or 0.0),
                        "score_assign": float(row_assign.get(int(object_id), 0.0) or 0.0),
                        "score_final": float(score_final),
                        "score_sets": float(candidate.get("score_sets", 0.0) or 0.0),
                        "bonus_sets": float(candidate.get("bonus_sets", 0.0) or 0.0),
                        "score_ctx_local": float(candidate.get("score_ctx_local", 0.0) or 0.0),
                        "score_ctx_global": float(candidate.get("score_ctx_global", 0.0) or 0.0),
                        "gate_reason": str(((candidate.get("sets_trace", {}) or {}).get("policy", {}) or {}).get("gate_reason", "") or ""),
                        "known_plausible_keep": int(candidate.get("known_plausible_keep", 0) or 0),
                    },
                )

        self.trace_collector.set_values(
            "shape.final_score_tables",
            scope_key,
            {
                "class_id": int(class_id),
                "detection_count": int(len(det_ids or [])),
                "candidate_object_count": int(len(object_ids)),
                "ranked_row_count": int(row_count),
            },
        )
        self.trace_collector.set_decision(
            "shape.final_score_tables",
            scope_key,
            {
                "status": "PASS" if row_count > 0 else "N/A",
                "branch": "scores_built" if row_count > 0 else "no_scores",
            },
        )
        self.trace_collector.leave_node("shape.final_score_tables", scope_key)

    def trace_locks(
        self,
        *,
        class_id: int,
        valid_det_ids: list[int],
        cand_obj_ids_all,
        table_sim: dict,
        table_final: dict,
        locked_matches: list[tuple[int, int, float]],
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        object_ids = sorted(int(x) for x in (cand_obj_ids_all or []))
        scope_key = self.trace_scope_key(class_id=int(class_id))
        participants = {
            "det_ids": [int(did) for did in (valid_det_ids or [])],
            "object_ids": list(object_ids),
        }
        self.trace_collector.enter_node(
            "resolve.locks",
            scope_key,
            participants=participants,
        )

        # Reproduce lock source (object pass vs detection pass) only for trace fidelity.
        replay_used_det_ids: set[int] = set()
        replay_used_obj_ids: set[int] = set()
        replay_lock_source_by_pair: dict[tuple[int, int], str] = {}
        replay_object_locks = self.lock_resolver.compute_object_locks(
            det_ids=[int(did) for did in (valid_det_ids or [])],
            table_sim=table_sim,
            table_final=table_final,
            obj_ids={int(x) for x in (cand_obj_ids_all or [])},
            used_det_ids=replay_used_det_ids,
            used_obj_ids=replay_used_obj_ids,
        )
        for det_id, obj_id, _ in replay_object_locks:
            replay_lock_source_by_pair[(int(det_id), int(obj_id))] = "object"
        replay_det_locks = self.lock_resolver.compute_det_locks(
            det_ids=[int(did) for did in (valid_det_ids or [])],
            table_sim=table_sim,
            table_final=table_final,
            used_det_ids=replay_used_det_ids,
            used_obj_ids=replay_used_obj_ids,
        )
        for det_id, obj_id, _ in replay_det_locks:
            replay_lock_source_by_pair.setdefault((int(det_id), int(obj_id)), "detection")

        locked_by_det_id = {int(det_id): (int(obj_id), float(score_final)) for det_id, obj_id, score_final in (locked_matches or [])}
        locked_obj_ids = {int(obj_id) for _, obj_id, _ in (locked_matches or [])}

        for det_id, obj_id, score_final in (locked_matches or []):
            lock_source = str(replay_lock_source_by_pair.get((int(det_id), int(obj_id)), "unknown"))
            det_row = {int(oid): float(score) for oid, score in ((table_sim.get(int(det_id), {}) or {}).items())}
            det_scored = sorted(
                [(float(score), int(oid)) for oid, score in det_row.items()],
                key=lambda item: (float(item[0]), int(item[1])),
                reverse=True,
            )
            det_s1 = float(det_scored[0][0]) if det_scored else 0.0
            det_best_oid = int(det_scored[0][1]) if det_scored else -1
            det_s2 = float(det_scored[1][0]) if len(det_scored) > 1 else 0.0
            det_lock_pass = bool(det_best_oid == int(obj_id) and self.lock_passes(det_s1, det_s2))

            obj_scored = []
            for candidate_det_id in (valid_det_ids or []):
                score = (table_sim.get(int(candidate_det_id), {}) or {}).get(int(obj_id), None)
                if score is None:
                    continue
                obj_scored.append((float(score), int(candidate_det_id)))
            obj_scored.sort(key=lambda item: (float(item[0]), int(item[1])), reverse=True)
            obj_s1 = float(obj_scored[0][0]) if obj_scored else 0.0
            obj_best_det_id = int(obj_scored[0][1]) if obj_scored else -1
            obj_s2 = float(obj_scored[1][0]) if len(obj_scored) > 1 else 0.0
            obj_lock_pass = bool(obj_best_det_id == int(det_id) and self.lock_passes(obj_s1, obj_s2))

            lock_modes = []
            if obj_lock_pass:
                lock_modes.append("object")
            if det_lock_pass:
                lock_modes.append("detection")
            if not lock_modes:
                lock_modes.append("unknown")

            self.trace_collector.add_detection_row(
                "resolve.locks",
                scope_key,
                {
                    "det_id": int(det_id),
                    "locked": True,
                    "locked_object_id": int(obj_id),
                    "score_final": float(score_final),
                    "lock_source": str(lock_source),
                    "lock_modes": list(lock_modes),
                    "checks": [
                        {
                            "id": "locks.object",
                            "label": "object_lock_pass",
                            "lhs": float(obj_s1 - obj_s2),
                            "op": "lock_passes",
                            "rhs": {
                                "top_score": float(obj_s1),
                                "second_score": float(obj_s2),
                            },
                            "passed": bool(obj_lock_pass),
                            "reason": "OBJECT_LOCK_PASS" if obj_lock_pass else "OBJECT_LOCK_FAIL",
                            "effect": "lock_match" if obj_lock_pass else "no_object_lock",
                        },
                        {
                            "id": "locks.detection",
                            "label": "detection_lock_pass",
                            "lhs": float(det_s1 - det_s2),
                            "op": "lock_passes",
                            "rhs": {
                                "top_score": float(det_s1),
                                "second_score": float(det_s2),
                            },
                            "passed": bool(det_lock_pass),
                            "reason": "DETECTION_LOCK_PASS" if det_lock_pass else "DETECTION_LOCK_FAIL",
                            "effect": "lock_match" if det_lock_pass else "no_detection_lock",
                        },
                    ],
                },
            )

        self.trace_collector.set_values(
            "resolve.locks",
            scope_key,
            {
                "class_id": int(class_id),
                "candidate_det_count": int(len(valid_det_ids or [])),
                "candidate_object_count": int(len(object_ids)),
                "locked_count": int(len(locked_matches or [])),
                "locked_det_ids": sorted(int(det_id) for det_id in locked_by_det_id.keys()),
                "locked_object_ids": sorted(int(obj_id) for obj_id in locked_obj_ids),
            },
        )
        self.trace_collector.set_decision(
            "resolve.locks",
            scope_key,
            {
                "status": "PASS" if locked_matches else "N/A",
                "branch": "locks_applied" if locked_matches else "no_locks",
            },
        )
        self.trace_collector.leave_node("resolve.locks", scope_key)

    def trace_hungarian_result(
        self,
        *,
        class_id: int,
        remaining_det_ids: list[int],
        cand_obj_list: list[int],
        table_sim_rem: dict,
        table_assign_rem: dict,
        table_final_rem: dict,
        reports: dict,
        row_ind,
        col_ind,
        match_thr: float,
        min_match_score: float,
        matches: list[tuple[int, int, float]],
        to_create: list[tuple[int, int]],
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        scope_key = self.trace_scope_key(class_id=int(class_id))
        participants = {
            "det_ids": [int(did) for did in (remaining_det_ids or [])],
            "object_ids": [int(oid) for oid in (cand_obj_list or [])],
        }
        self.trace_collector.enter_node(
            "resolve.hungarian",
            scope_key,
            participants=participants,
        )
        assignment_by_det_id = {
            int(remaining_det_ids[int(i)]): int(j)
            for i, j in zip(row_ind, col_ind)
        }
        match_by_det_id = {
            int(det_id): {"object_id": int(object_id), "score_final": float(score_final)}
            for det_id, object_id, score_final in (matches or [])
        }
        create_det_ids = {int(det_id) for det_id, _ in (to_create or [])}
        dummy_count = max(0, int(len(remaining_det_ids or []))) if self.enable_dummies else 0
        object_count = int(len(cand_obj_list or []))
        total_columns = int(object_count + dummy_count)

        for det_id in (remaining_det_ids or []):
            det_id = int(det_id)
            assigned_col = assignment_by_det_id.get(int(det_id), None)
            assigned_object_id = None
            assigned_kind = "UNASSIGNED"
            if assigned_col is not None:
                if int(assigned_col) < object_count:
                    assigned_kind = "OBJECT"
                    assigned_object_id = int(cand_obj_list[int(assigned_col)])
                else:
                    assigned_kind = "DUMMY"

            row_assign = {
                int(object_id): float(score)
                for object_id, score in ((table_assign_rem.get(int(det_id), {}) or {}).items())
            }
            row_sim = {
                int(object_id): float(score)
                for object_id, score in ((table_sim_rem.get(int(det_id), {}) or {}).items())
            }
            row_final = {
                int(object_id): float(score)
                for object_id, score in ((table_final_rem.get(int(det_id), {}) or {}).items())
            }
            dummy_score = float(self.resolve_dummy_score(reports.get(int(det_id), None)))
            selected_score_assign = None if assigned_object_id is None else float(row_assign.get(int(assigned_object_id), 0.0))
            selected_score_sim = None if assigned_object_id is None else float(row_sim.get(int(assigned_object_id), 0.0))
            selected_score_final = None if assigned_object_id is None else float(row_final.get(int(assigned_object_id), 0.0))
            passes_match_thr = bool(
                assigned_object_id is not None
                and selected_score_final is not None
                and float(selected_score_final) >= float(match_thr)
            )
            passes_min_match_score = bool(
                assigned_object_id is not None
                and selected_score_sim is not None
                and float(selected_score_sim) >= float(min_match_score)
            )
            final_action = "CREATE"
            action_reason = "ASSIGNED_TO_DUMMY"
            if int(det_id) in match_by_det_id:
                final_action = "MATCH"
                action_reason = "OBJECT_ACCEPTED"
            elif assigned_object_id is not None:
                action_reason = "OBJECT_REJECTED_BY_THRESHOLDS"

            self.trace_collector.add_detection_row(
                "resolve.hungarian",
                scope_key,
                {
                    "det_id": int(det_id),
                    "assigned_kind": str(assigned_kind),
                    "assigned_column": None if assigned_col is None else int(assigned_col),
                    "assigned_object_id": assigned_object_id,
                    "selected_score_assign": selected_score_assign,
                    "selected_score_sim": selected_score_sim,
                    "selected_score_final": selected_score_final,
                    "dummy_score": float(dummy_score),
                    "passes_match_thr": bool(passes_match_thr),
                    "passes_min_match_score": bool(passes_min_match_score),
                    "final_action": str(final_action),
                    "reason": str(action_reason),
                    "checks": [
                        {
                            "id": "hungarian.assignment_kind",
                            "label": "assigned_to_real_object",
                            "lhs": str(assigned_kind),
                            "op": "==",
                            "rhs": "OBJECT",
                            "passed": bool(assigned_object_id is not None),
                            "reason": str(action_reason),
                            "effect": "evaluate_thresholds" if assigned_object_id is not None else "route_to_create",
                        },
                        {
                            "id": "hungarian.match_thr",
                            "label": "selected_score_final_reaches_match_thr",
                            "lhs": None if selected_score_final is None else float(selected_score_final),
                            "op": ">=",
                            "rhs": float(match_thr),
                            "passed": bool(passes_match_thr),
                            "reason": str(action_reason),
                            "effect": "keep_object_assignment" if passes_match_thr else "reject_object_assignment",
                        },
                        {
                            "id": "hungarian.min_match_score",
                            "label": "selected_score_sim_reaches_min_match_score",
                            "lhs": None if selected_score_sim is None else float(selected_score_sim),
                            "op": ">=",
                            "rhs": float(min_match_score),
                            "passed": bool(passes_min_match_score),
                            "reason": str(action_reason),
                            "effect": "allow_match" if passes_min_match_score else "fallback_to_create",
                        },
                    ],
                },
            )

            ranked = sorted(
                row_assign.items(),
                key=lambda item: (float(item[1]), int(item[0])),
                reverse=True,
            )
            for rank, (object_id, score_assign) in enumerate(ranked, start=1):
                self.trace_collector.add_candidate_row(
                    "resolve.hungarian",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "object_id": int(object_id),
                        "rank": int(rank),
                        "selected": bool(assigned_object_id is not None and int(object_id) == int(assigned_object_id)),
                        "score_assign": float(score_assign),
                        "score_sim": float(row_sim.get(int(object_id), 0.0) or 0.0),
                        "score_final": float(row_final.get(int(object_id), 0.0) or 0.0),
                    },
                )
        for det_id, object_id, score_final in (matches or []):
            self.trace_collector.add_global_row(
                "resolve.hungarian",
                scope_key,
                {
                    "kind": "match",
                    "det_id": int(det_id),
                    "object_id": int(object_id),
                    "score_final": float(score_final),
                },
            )
        for det_id, class_id_create in (to_create or []):
            self.trace_collector.add_global_row(
                "resolve.hungarian",
                scope_key,
                {
                    "kind": "create",
                    "det_id": int(det_id),
                    "class_id": int(class_id_create),
                },
            )
        self.trace_collector.set_values(
            "resolve.hungarian",
            scope_key,
            {
                "class_id": int(class_id),
                "participant_det_ids": [int(did) for did in (remaining_det_ids or [])],
                "participant_object_ids": [int(oid) for oid in (cand_obj_list or [])],
                "object_column_count": int(object_count),
                "dummy_column_count": int(dummy_count),
                "total_column_count": int(total_columns),
                "n_matches": int(len(matches or [])),
                "n_creates": int(len(to_create or [])),
                "create_det_ids": sorted(int(det_id) for det_id in create_det_ids),
            },
        )
        self.trace_collector.set_decision(
            "resolve.hungarian",
            scope_key,
            {
                "status": "PASS",
                "branch": "resolved",
            },
        )
        self.trace_collector.leave_node("resolve.hungarian", scope_key)

    def assign(
        self,
        detections: list,
        det_features_by_id: dict,
        reports: dict,
        snapshot_ids: set[int],
        association_output,
        *,
        use_neighbor_sets: bool,
        match_thr: float,
        min_match_score: float,
        neighbor_sets_influence=None,
        ns_ctx_override: dict | None = None,
        timer=None,
        timer_prefix: str = "",
    ):
        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as e:
            raise RuntimeError("Hungarian requiere scipy. Instala: pip install scipy") from e

        step = (lambda suffix: f"{timer_prefix}{suffix}" if timer_prefix else str(suffix))
        if timer is not None:
            partition = timer.run(step("partition"), self.partition_assignment_detections, detections)
            prepared = timer.run(
                step("prepare"),
                self.prepare_assignment_inputs,
                by_class=partition.det_ids_by_class,
                snapshot_ids=snapshot_ids,
                association_output=association_output,
                use_neighbor_sets=use_neighbor_sets,
                neighbor_sets_influence=neighbor_sets_influence,
                ns_ctx_override=ns_ctx_override,
                match_thr=float(match_thr),
                timer=timer,
                timer_prefix=step("prepare/"),
            )
        else:
            partition = self.partition_assignment_detections(detections)
            prepared = self.prepare_assignment_inputs(
                by_class=partition.det_ids_by_class,
                snapshot_ids=snapshot_ids,
                association_output=association_output,
                use_neighbor_sets=use_neighbor_sets,
                neighbor_sets_influence=neighbor_sets_influence,
                ns_ctx_override=ns_ctx_override,
                match_thr=float(match_thr),
            )
        if timer is not None:
            return timer.run(
                step("assign_classes"),
                self.assign_partitioned_classes,
                linear_sum_assignment=linear_sum_assignment,
                partition=partition,
                prepared=prepared,
                det_features_by_id=det_features_by_id,
                reports=reports,
                use_neighbor_sets=use_neighbor_sets,
                match_thr=float(match_thr),
                min_match_score=float(min_match_score),
                neighbor_sets_influence=neighbor_sets_influence,
                timer=timer,
                timer_prefix=step("assign_classes/"),
            )
        return self.assign_partitioned_classes(
            linear_sum_assignment=linear_sum_assignment,
            partition=partition,
            prepared=prepared,
            det_features_by_id=det_features_by_id,
            reports=reports,
            use_neighbor_sets=use_neighbor_sets,
            match_thr=float(match_thr),
            min_match_score=float(min_match_score),
            neighbor_sets_influence=neighbor_sets_influence,
        )

    def partition_assignment_detections(self, detections: list):
        return self.assignment_path_support.partition_detections(detections)

    def prepare_assignment_inputs(
        self,
        *,
        by_class: dict[int, list[int]],
        snapshot_ids: set[int],
        association_output,
        use_neighbor_sets: bool,
        neighbor_sets_influence,
        ns_ctx_override: dict | None,
        match_thr: float,
        timer=None,
        timer_prefix: str = "",
    ):
        return self.assignment_path_support.prepare_assignment_inputs(
            assigner=self,
            by_class=by_class,
            snapshot_ids=snapshot_ids,
            association_output=association_output,
            use_neighbor_sets=use_neighbor_sets,
            neighbor_sets_influence=neighbor_sets_influence,
            ns_ctx_override=ns_ctx_override,
            match_thr=match_thr,
            timer=timer,
            timer_prefix=timer_prefix,
        )

    def assign_partitioned_classes(
        self,
        *,
        linear_sum_assignment,
        partition,
        prepared,
        det_features_by_id: dict,
        reports: dict,
        use_neighbor_sets: bool,
        match_thr: float,
        min_match_score: float,
        neighbor_sets_influence,
        timer=None,
        timer_prefix: str = "",
    ):
        decided_matches: list[tuple[int, int, float]] = []
        to_create: list[tuple[int, int]] = []

        for class_id, det_ids in partition.det_ids_by_class.items():
            class_timer_prefix = f"{timer_prefix}" if timer_prefix else ""
            if timer is not None:
                class_matches, class_creates = self.assign_class(
                    linear_sum_assignment=linear_sum_assignment,
                    class_id=int(class_id),
                    det_ids=[int(did) for did in (det_ids or [])],
                    det_features_by_id=det_features_by_id,
                    detections_by_id=partition.detections_by_id,
                    reports=reports,
                    snapshot_ids=prepared.snapshot_ids,
                    use_neighbor_sets=use_neighbor_sets,
                    match_thr=float(match_thr),
                    min_match_score=float(min_match_score),
                    neighbor_sets_influence=neighbor_sets_influence,
                    ns_ctx=prepared.context.ns_ctx,
                    timer=timer,
                    timer_prefix=class_timer_prefix,
                )
            else:
                class_matches, class_creates = self.assign_class(
                    linear_sum_assignment=linear_sum_assignment,
                    class_id=int(class_id),
                    det_ids=[int(did) for did in (det_ids or [])],
                    det_features_by_id=det_features_by_id,
                    detections_by_id=partition.detections_by_id,
                    reports=reports,
                    snapshot_ids=prepared.snapshot_ids,
                    use_neighbor_sets=use_neighbor_sets,
                    match_thr=float(match_thr),
                    min_match_score=float(min_match_score),
                    neighbor_sets_influence=neighbor_sets_influence,
                    ns_ctx=prepared.context.ns_ctx,
                )
            decided_matches.extend(class_matches)
            to_create.extend(class_creates)

        return decided_matches, to_create

    def resolve_assignment_context(
        self,
        *,
        association_output,
        use_neighbor_sets: bool,
        neighbor_sets_influence,
        ns_ctx_override: dict | None,
    ) -> AssignmentContext:
        ns_ctx = ns_ctx_override if isinstance(ns_ctx_override, dict) else {}
        if not ns_ctx and use_neighbor_sets and association_output is not None and neighbor_sets_influence is not None:
            ns_ctx = neighbor_sets_influence.build_context(getattr(association_output, "neighbor_sets_out", None))

        use_sets_bonus = bool(
            use_neighbor_sets
            and neighbor_sets_influence is not None
            and ns_ctx
            and ns_ctx.get("enabled", False)
        )
        return AssignmentContext(
            ns_ctx=ns_ctx,
            use_sets_bonus=bool(use_sets_bonus),
        )

    def assign_class(
        self,
        *,
        linear_sum_assignment,
        class_id: int,
        det_ids: list[int],
        det_features_by_id: dict,
        detections_by_id: dict[int, object],
        reports: dict,
        snapshot_ids: set[int],
        use_neighbor_sets: bool,
        match_thr: float,
        min_match_score: float,
        neighbor_sets_influence,
        ns_ctx: dict,
        timer=None,
        timer_prefix: str = "",
    ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int]]]:
        valid_det_ids, create_from_missing_features = self.split_class_detection_inputs(
            class_id=int(class_id),
            det_ids=det_ids,
            det_features_by_id=det_features_by_id,
        )
        self.trace_valid_detections(
            class_id=int(class_id),
            det_ids=[int(did) for did in (det_ids or [])],
            valid_det_ids=valid_det_ids,
            create_entries=create_from_missing_features,
        )

        if not valid_det_ids:
            self.trace_skip_node_for_class(
                node_id="shape.allow_for_report",
                class_id=int(class_id),
                reason="NO_VALID_DETECTIONS",
                det_ids=det_ids,
            )
            self.trace_skip_node_for_class(
                node_id="shape.context_veto",
                class_id=int(class_id),
                reason="NO_VALID_DETECTIONS",
                det_ids=det_ids,
            )
            self.trace_skip_node_for_class(
                node_id="shape.final_score_tables",
                class_id=int(class_id),
                reason="NO_VALID_DETECTIONS",
                det_ids=det_ids,
            )
            self.trace_skip_node_for_class(
                node_id="resolve.locks",
                class_id=int(class_id),
                reason="NO_VALID_DETECTIONS",
                det_ids=det_ids,
            )
            self.trace_skip_node_for_class(
                node_id="resolve.hungarian",
                class_id=int(class_id),
                reason="NO_VALID_DETECTIONS",
                det_ids=det_ids,
            )
            return [], create_from_missing_features

        step = (lambda suffix: f"{timer_prefix}{suffix}" if timer_prefix else str(suffix))
        used_det_ids: set[int] = set()
        used_obj_ids: set[int] = set()

        self.trace_allow_for_report(
            class_id=int(class_id),
            det_ids=valid_det_ids,
            reports=reports,
            snapshot_ids=snapshot_ids,
            neighbor_sets_influence=neighbor_sets_influence,
            ns_ctx=ns_ctx,
            used_obj_ids=used_obj_ids,
            stage="pre_locks",
        )

        if timer is not None:
            table_sim, table_assign, table_final, candidate_obj_ids = timer.run(
                step("score_tables"),
                self.build_class_score_tables,
                det_ids=valid_det_ids,
                detections_by_id=detections_by_id,
                reports=reports,
                snapshot_ids=snapshot_ids,
                used_obj_ids=used_obj_ids,
                use_neighbor_sets=use_neighbor_sets,
                ns_ctx=ns_ctx,
                neighbor_sets_influence=neighbor_sets_influence,
                match_thr=float(match_thr),
                min_match_score=float(min_match_score),
            )
        else:
            table_sim, table_assign, table_final, candidate_obj_ids = self.build_class_score_tables(
                det_ids=valid_det_ids,
                detections_by_id=detections_by_id,
                reports=reports,
                snapshot_ids=snapshot_ids,
                used_obj_ids=used_obj_ids,
                use_neighbor_sets=use_neighbor_sets,
                ns_ctx=ns_ctx,
                neighbor_sets_influence=neighbor_sets_influence,
                match_thr=float(match_thr),
                min_match_score=float(min_match_score),
            )
        self.trace_context_veto(
            class_id=int(class_id),
            det_ids=valid_det_ids,
            reports=reports,
            snapshot_ids=snapshot_ids,
        )
        self.trace_final_score_tables(
            class_id=int(class_id),
            det_ids=valid_det_ids,
            reports=reports,
            table_sim=table_sim,
            table_assign=table_assign,
            table_final=table_final,
            candidate_obj_ids=candidate_obj_ids,
        )
        if not table_sim:
            self.trace_skip_node_for_class(
                node_id="resolve.locks",
                class_id=int(class_id),
                reason="NO_SCORE_ROWS",
                det_ids=valid_det_ids,
                object_ids=candidate_obj_ids,
            )
            self.trace_skip_node_for_class(
                node_id="resolve.hungarian",
                class_id=int(class_id),
                reason="NO_SCORE_ROWS",
                det_ids=valid_det_ids,
                object_ids=candidate_obj_ids,
            )
            return [], create_from_missing_features + [(int(did), int(class_id)) for did in valid_det_ids]

        if timer is not None:
            locked_matches = timer.run(
                step("locks"),
                self.locked_matches_for_class,
                class_id=int(class_id),
                valid_det_ids=valid_det_ids,
                table_sim=table_sim,
                table_final=table_final,
                cand_obj_ids_all=candidate_obj_ids,
                used_det_ids=used_det_ids,
                used_obj_ids=used_obj_ids,
            )
        else:
            locked_matches = self.locked_matches_for_class(
                class_id=int(class_id),
                valid_det_ids=valid_det_ids,
                table_sim=table_sim,
                table_final=table_final,
                cand_obj_ids_all=candidate_obj_ids,
                used_det_ids=used_det_ids,
                used_obj_ids=used_obj_ids,
            )
        self.trace_locks(
            class_id=int(class_id),
            valid_det_ids=valid_det_ids,
            cand_obj_ids_all=candidate_obj_ids,
            table_sim=table_sim,
            table_final=table_final,
            locked_matches=locked_matches,
        )

        remaining_det_ids = [int(did) for did in valid_det_ids if int(did) not in used_det_ids]
        if not remaining_det_ids:
            self.trace_skip_node_for_class(
                node_id="resolve.hungarian",
                class_id=int(class_id),
                reason="ALL_RESOLVED_BY_LOCKS",
                det_ids=valid_det_ids,
                object_ids=candidate_obj_ids,
            )
            return locked_matches, create_from_missing_features

        if timer is not None:
            hungarian_matches, hungarian_creates = timer.run(
                step("solve"),
                self.hungarian_assign_class,
                linear_sum_assignment=linear_sum_assignment,
                class_id=int(class_id),
                remaining_det_ids=remaining_det_ids,
                detections_by_id=detections_by_id,
                reports=reports,
                snapshot_ids=snapshot_ids,
                used_obj_ids=used_obj_ids,
                use_neighbor_sets=use_neighbor_sets,
                neighbor_sets_influence=neighbor_sets_influence,
                ns_ctx=ns_ctx,
                match_thr=float(match_thr),
                min_match_score=float(min_match_score),
                timer=timer,
                timer_prefix=step("solve/"),
            )
        else:
            hungarian_matches, hungarian_creates = self.hungarian_assign_class(
                linear_sum_assignment=linear_sum_assignment,
                class_id=int(class_id),
                remaining_det_ids=remaining_det_ids,
                detections_by_id=detections_by_id,
                reports=reports,
                snapshot_ids=snapshot_ids,
                used_obj_ids=used_obj_ids,
                use_neighbor_sets=use_neighbor_sets,
                neighbor_sets_influence=neighbor_sets_influence,
                ns_ctx=ns_ctx,
                match_thr=float(match_thr),
                min_match_score=float(min_match_score),
            )
        return locked_matches + hungarian_matches, create_from_missing_features + hungarian_creates

    def split_class_detection_inputs(
        self,
        *,
        class_id: int,
        det_ids: list[int],
        det_features_by_id: dict,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        valid_det_ids: list[int] = []
        create_entries: list[tuple[int, int]] = []
        for det_id in (det_ids or []):
            if det_features_by_id.get(int(det_id), None) is not None:
                valid_det_ids.append(int(det_id))
            else:
                create_entries.append((int(det_id), int(class_id)))
        return valid_det_ids, create_entries

    def build_class_score_tables(
        self,
        *,
        det_ids: list[int],
        detections_by_id: dict[int, object],
        reports: dict,
        snapshot_ids: set[int],
        used_obj_ids: set[int],
        use_neighbor_sets: bool,
        ns_ctx: dict,
        neighbor_sets_influence,
        match_thr: float,
        min_match_score: float,
    ) -> tuple[dict[int, dict[int, float]], dict[int, dict[int, float]], dict[int, dict[int, float]], set[int]]:
        return self.build_score_tables(
            det_ids=det_ids,
            detections_by_id=detections_by_id,
            reports=reports,
            snapshot_ids=snapshot_ids,
            used_obj_ids=used_obj_ids,
            use_neighbor_sets=use_neighbor_sets,
            ns_ctx=ns_ctx,
            neighbor_sets_influence=neighbor_sets_influence,
            match_thr=float(match_thr),
            min_match_score=float(min_match_score),
        )

    def locked_matches_for_class(
        self,
        *,
        class_id: int,
        valid_det_ids: list[int],
        table_sim: dict,
        table_final: dict,
        cand_obj_ids_all,
        used_det_ids: set[int],
        used_obj_ids: set[int],
    ) -> list[tuple[int, int, float]]:
        locked = []
        locked += self.lock_resolver.compute_object_locks(
            det_ids=valid_det_ids,
            table_sim=table_sim,
            table_final=table_final,
            obj_ids=cand_obj_ids_all,
            used_det_ids=used_det_ids,
            used_obj_ids=used_obj_ids,
        )
        locked += self.lock_resolver.compute_det_locks(
            det_ids=valid_det_ids,
            table_sim=table_sim,
            table_final=table_final,
            used_det_ids=used_det_ids,
            used_obj_ids=used_obj_ids,
        )
        return [(int(did), int(oid), float(score)) for did, oid, score in locked]

    def hungarian_assign_class(
        self,
        *,
        linear_sum_assignment,
        class_id: int,
        remaining_det_ids: list[int],
        detections_by_id: dict[int, object],
        reports: dict,
        snapshot_ids: set[int],
        used_obj_ids: set[int],
        use_neighbor_sets: bool,
        neighbor_sets_influence,
        ns_ctx: dict,
        match_thr: float,
        min_match_score: float,
        timer=None,
        timer_prefix: str = "",
    ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int]]]:
        step = (lambda suffix: f"{timer_prefix}{suffix}" if timer_prefix else str(suffix))
        self.trace_allow_for_report(
            class_id=int(class_id),
            det_ids=remaining_det_ids,
            reports=reports,
            snapshot_ids=snapshot_ids,
            neighbor_sets_influence=neighbor_sets_influence,
            ns_ctx=ns_ctx,
            used_obj_ids=used_obj_ids,
            stage="post_locks",
        )
        score_table_kwargs = dict(
            det_ids=remaining_det_ids,
            detections_by_id=detections_by_id,
            reports=reports,
            snapshot_ids=snapshot_ids,
            used_obj_ids=used_obj_ids,
            use_neighbor_sets=use_neighbor_sets,
            ns_ctx=ns_ctx,
            neighbor_sets_influence=neighbor_sets_influence,
            match_thr=float(match_thr),
            min_match_score=float(min_match_score),
        )
        if timer is not None:
            table_sim_rem, table_assign_rem, table_final_rem, cand_obj_ids = timer.run(
                step("score_tables"),
                self.build_score_tables,
                **score_table_kwargs,
            )
        else:
            table_sim_rem, table_assign_rem, table_final_rem, cand_obj_ids = self.build_score_tables(
                **score_table_kwargs,
            )
        cand_obj_ids = set(int(x) for x in cand_obj_ids)
        if not cand_obj_ids:
            return [], [(int(did), int(class_id)) for did in remaining_det_ids]

        cand_obj_list = sorted(cand_obj_ids)
        cost_kwargs = dict(
            remaining_det_ids=remaining_det_ids,
            cand_obj_list=cand_obj_list,
            table_assign_rem=table_assign_rem,
            reports=reports,
            report_status_fn=self.report_status,
            resolve_dummy_score_fn=self.resolve_dummy_score,
        )
        if timer is not None:
            cost = timer.run(
                step("cost_matrix"),
                self.hungarian_resolver.build_cost_matrix,
                **cost_kwargs,
            )
        else:
            cost = self.hungarian_resolver.build_cost_matrix(**cost_kwargs)
        row_ind, col_ind = linear_sum_assignment(cost)
        resolve_kwargs = dict(
            class_id=int(class_id),
            remaining_det_ids=remaining_det_ids,
            cand_obj_list=cand_obj_list,
            row_ind=row_ind,
            col_ind=col_ind,
            table_sim_rem=table_sim_rem,
            table_final_rem=table_final_rem,
            reports=reports,
            match_thr=float(match_thr),
            min_match_score=float(min_match_score),
            report_status_fn=self.report_status,
        )
        if timer is not None:
            matches, to_create = timer.run(
                step("resolve"),
                self.hungarian_resolver.resolve_assignment,
                **resolve_kwargs,
            )
        else:
            matches, to_create = self.hungarian_resolver.resolve_assignment(**resolve_kwargs)
        self.trace_hungarian_result(
            class_id=int(class_id),
            remaining_det_ids=remaining_det_ids,
            cand_obj_list=cand_obj_list,
            table_sim_rem=table_sim_rem,
            table_assign_rem=table_assign_rem,
            table_final_rem=table_final_rem,
            reports=reports,
            row_ind=row_ind,
            col_ind=col_ind,
            match_thr=float(match_thr),
            min_match_score=float(min_match_score),
            matches=matches,
            to_create=to_create,
        )
        return matches, to_create

    def build_hungarian_cost_matrix(
        self,
        *,
        remaining_det_ids: list[int],
        cand_obj_list: list[int],
        table_assign_rem: dict,
        reports: dict,
    ) -> list[list[float]]:
        return self.hungarian_resolver.build_cost_matrix(
            remaining_det_ids=remaining_det_ids,
            cand_obj_list=cand_obj_list,
            table_assign_rem=table_assign_rem,
            reports=reports,
            report_status_fn=self.report_status,
            resolve_dummy_score_fn=self.resolve_dummy_score,
        )

    def resolve_hungarian_assignment(
        self,
        *,
        class_id: int,
        remaining_det_ids: list[int],
        cand_obj_list: list[int],
        row_ind,
        col_ind,
        table_sim_rem: dict,
        table_final_rem: dict,
        reports: dict,
        match_thr: float,
        min_match_score: float,
    ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int]]]:
        return self.hungarian_resolver.resolve_assignment(
            class_id=class_id,
            remaining_det_ids=remaining_det_ids,
            cand_obj_list=cand_obj_list,
            row_ind=row_ind,
            col_ind=col_ind,
            table_sim_rem=table_sim_rem,
            table_final_rem=table_final_rem,
            reports=reports,
            match_thr=match_thr,
            min_match_score=min_match_score,
            report_status_fn=self.report_status,
        )

    def report_status(self, report) -> str:
        return self.candidate_score_shaper.report_status(report)

    def default_sets_trace(self, report) -> dict:
        return self.candidate_score_shaper.default_sets_trace(report)

    def format_sets_trace_summary(self, trace: dict) -> tuple[str, str, str]:
        return self.candidate_score_shaper.format_sets_trace_summary(trace)

    def attach_sets_trace_fields(self, candidate: dict, trace: dict) -> None:
        self.candidate_score_shaper.attach_sets_trace_fields(candidate, trace)

    def resolve_report_confidence(self, report) -> float | None:
        return self.candidate_score_shaper.resolve_report_confidence(report)

    def resolve_dummy_score(self, report) -> float:
        return self.candidate_score_shaper.resolve_dummy_score(report)

    def lock_passes(self, s1: float, s2: float) -> bool:
        return self.lock_resolver.lock_passes(s1, s2)

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
        return self.candidate_score_shaper.build_score_tables(
            det_ids=det_ids,
            detections_by_id=detections_by_id,
            reports=reports,
            snapshot_ids=snapshot_ids,
            used_obj_ids=used_obj_ids,
            use_neighbor_sets=use_neighbor_sets,
            ns_ctx=ns_ctx,
            neighbor_sets_influence=neighbor_sets_influence,
            match_thr=match_thr,
            min_match_score=min_match_score,
            min_score=min_score,
            gate_by_match_thr=gate_by_match_thr,
        )

    def candidate_context_veto_reason(
        self,
        *,
        det_class_id: int,
        object_id: int,
        candidate: dict,
        ns_ctx: dict | None,
        neighbor_sets_influence,
    ) -> str:
        return self.candidate_score_shaper.context_veto_reason(
            det_class_id=det_class_id,
            object_id=object_id,
            candidate=candidate,
            ns_ctx=ns_ctx,
            neighbor_sets_influence=neighbor_sets_influence,
        )

    def candidate_vetoed_by_context(
        self,
        *,
        det_class_id: int,
        object_id: int,
        candidate: dict,
        ns_ctx: dict | None,
        neighbor_sets_influence,
    ) -> bool:
        return self.candidate_score_shaper.candidate_vetoed_by_context(
            det_class_id=det_class_id,
            object_id=object_id,
            candidate=candidate,
            ns_ctx=ns_ctx,
            neighbor_sets_influence=neighbor_sets_influence,
        )

    def compute_object_locks(
        self,
        det_ids: list[int],
        table_sim: dict[int, dict[int, float]],
        table_final: dict[int, dict[int, float]],
        obj_ids: set[int],
        used_det_ids: set[int],
        used_obj_ids: set[int],
    ) -> list[tuple[int, int, float]]:
        return self.lock_resolver.compute_object_locks(
            det_ids=det_ids,
            table_sim=table_sim,
            table_final=table_final,
            obj_ids=obj_ids,
            used_det_ids=used_det_ids,
            used_obj_ids=used_obj_ids,
        )

    def compute_det_locks(
        self,
        det_ids: list[int],
        table_sim: dict[int, dict[int, float]],
        table_final: dict[int, dict[int, float]],
        used_det_ids: set[int],
        used_obj_ids: set[int],
    ) -> list[tuple[int, int, float]]:
        return self.lock_resolver.compute_det_locks(
            det_ids=det_ids,
            table_sim=table_sim,
            table_final=table_final,
            used_det_ids=used_det_ids,
            used_obj_ids=used_obj_ids,
        )
