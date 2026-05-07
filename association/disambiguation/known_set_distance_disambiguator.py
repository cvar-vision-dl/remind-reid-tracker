from __future__ import annotations

from .anchor_selection import AnchorSelectionMixin
from .assignment_scoring import AssignmentScoringMixin
from .component_building import ComponentBuildingMixin
from .pair_anchor_discriminator import PairAnchorDiscriminator
from memory.neighbor_distance_graph import compute_relation_observation, prepare_relation_mask_runtime


class KnownSetDistanceDisambiguator(
    ComponentBuildingMixin,
    AnchorSelectionMixin,
    AssignmentScoringMixin,
):
    """
    Resuelve componentes ambiguos cerrados de IDs conocidos usando memoria relacional.

    El módulo no decide NEW vs KNOWN. Solo desambiguación bajo la premisa de que
    el conjunto correcto de IDs conocidos ya está acotado.
    """

    def __init__(self, *, config: dict, memory_store) -> None:
        self.config = config or {}
        self.memory_store = memory_store
        self.last_debug_rows = []
        self.last_pair_anchor_rows = []
        self.last_resolved_sources = {}

        assoc = (self.config.get("association", {}) or {})
        amb_cfg = (assoc.get("ambiguous_tracks", {}) or {})
        cfg = (amb_cfg.get("known_set_distance_disambiguation", {}) or {})
        self.pair_anchor_discriminator = PairAnchorDiscriminator(config=cfg, memory_store=memory_store)
        self.pair_anchor_discriminator.relation_observation_fn = self.relation_observation_cached
        self._relation_obs_cache = {}
        self._mask_runtime_by_geom_key = {}

        self.enabled = bool(cfg.get("enabled", True))
        self.max_group_size = max(1, int(cfg.get("max_group_size", 4)))
        self.max_candidate_union = max(2, int(cfg.get("max_candidate_union", 6)))
        self.max_anchors = max(1, int(cfg.get("max_anchors", 4)))
        self.discriminative_anchor_topk = max(1, int(cfg.get("discriminative_anchor_topk", 2)))
        self.discriminative_anchor_min_sep = max(0.0, float(cfg.get("discriminative_anchor_min_sep", 0.02)))
        self.selected_anchor_score_ratio_min = max(0.0, min(1.0, float(cfg.get("selected_anchor_score_ratio_min", 0.60))))

        self.anchor_weight = max(0.0, float(cfg.get("anchor_weight", 0.55)))
        self.visual_weight = max(0.0, float(cfg.get("visual_weight", 0.10)))
        self.anchor_history_score_weight = max(0.0, float(cfg.get("anchor_history_score_weight", 0.60)))
        self.anchor_frame_score_weight = max(0.0, float(cfg.get("anchor_frame_score_weight", 0.40)))

        self.min_edge_reliability = max(0.0, min(1.0, float(cfg.get("min_edge_reliability", 0.15))))
        self.min_anchor_informativeness = max(0.0, min(1.0, float(cfg.get("min_anchor_informativeness", 0.10))))
        self.min_total_evidence = max(0.0, float(cfg.get("min_total_evidence", 0.20)))
        self.min_assignment_score = max(0.0, float(cfg.get("min_assignment_score", 0.20)))
        self.min_gap = max(0.0, float(cfg.get("min_gap", 0.08)))
        self.min_core_assignment_score = max(0.0, float(cfg.get("min_core_assignment_score", 0.20)))
        self.min_core_gap = max(0.0, float(cfg.get("min_core_gap", 0.08)))
        self.min_singleton_core_assignment_score = max(
            0.0,
            float(cfg.get("min_singleton_core_assignment_score", cfg.get("min_core_assignment_score", 0.20))),
        )
        self.min_singleton_core_gap = max(
            0.0,
            float(cfg.get("min_singleton_core_gap", cfg.get("min_core_gap", 0.08))),
        )
        self.min_full_sibling_core_assignment_score = max(
            0.0,
            float(cfg.get("min_full_sibling_core_assignment_score", cfg.get("min_core_assignment_score", 0.20))),
        )
        self.min_full_sibling_core_gap = max(
            0.0,
            float(cfg.get("min_full_sibling_core_gap", cfg.get("min_gap", 0.08))),
        )
        self.min_partial_same_class_core_assignment_score = max(
            0.0,
            float(cfg.get("min_partial_same_class_core_assignment_score", cfg.get("min_singleton_core_assignment_score", 0.20))),
        )
        self.min_partial_same_class_core_gap = max(
            0.0,
            float(cfg.get("min_partial_same_class_core_gap", cfg.get("min_singleton_core_gap", 0.10))),
        )

        self.gap_sigma = max(1e-6, float(cfg.get("gap_sigma", 0.20)))
        self.center_sigma = max(1e-6, float(cfg.get("center_sigma", 0.35)))
        self.rank_sigma = max(1e-6, float(cfg.get("rank_sigma", 1.0)))
        self.anchor_span_ref = max(1e-6, float(cfg.get("anchor_span_ref", 0.20)))
        self.order_margin_ref = max(1e-6, float(cfg.get("order_margin_ref", self.anchor_span_ref)))

        self.support_penalty = max(0.0, min(1.0, float(cfg.get("support_penalty", 0.80))))
        self.rank_weight = max(0.0, min(1.0, float(cfg.get("rank_weight", 0.20))))
        self.max_component_pair_distance = max(0.0, float(cfg.get("max_component_pair_distance", 1.75)))
        self.min_component_pair_score = max(0.0, float(cfg.get("min_component_pair_score", 0.08)))
        self.obs_gap_quality_weight = max(0.0, min(1.0, float(cfg.get("obs_gap_quality_weight", 0.35))))
        self.obs_truncation_penalty = max(0.0, min(1.0, float(cfg.get("obs_truncation_penalty", 0.50))))

        self.soft_anchors_enabled = bool(cfg.get("soft_anchors_enabled", True))
        self.soft_anchor_min_score = max(0.0, float(cfg.get("soft_anchor_min_score", 0.75)))
        self.soft_anchor_max = max(0, int(cfg.get("soft_anchor_max", 6)))
        self.soft_anchor_conf_weight = max(0.0, float(cfg.get("soft_anchor_conf_weight", 0.35)))
        self.anchor_pair_order_weight = max(0.0, min(1.0, float(cfg.get("anchor_pair_order_weight", 0.65))))
        self.anchor_pair_margin_weight = max(0.0, min(1.0, float(cfg.get("anchor_pair_margin_weight", 0.35))))
        self.anchor_pair_topk = max(1, int(cfg.get("anchor_pair_topk", min(3, self.discriminative_anchor_topk))))
        self.debug_anchor_topk = max(3, int(cfg.get("debug_anchor_topk", 5)))
        self.debug_historical_anchor_topk = max(0, int(cfg.get("debug_historical_anchor_topk", 5)))

    def reset_relation_runtime_cache(self) -> None:
        self._relation_obs_cache = {}
        self._mask_runtime_by_geom_key = {}

    def register_relation_geometries(self, *, prefix: str, geom_by_id: dict[int, dict] | None) -> None:
        tag = str(prefix or "")
        for raw_id, geom in ((geom_by_id or {}).items()):
            if not isinstance(geom, dict):
                continue
            self._mask_runtime_by_geom_key[(tag, int(raw_id))] = prepare_relation_mask_runtime(
                geom.get("mask", None),
                compute_bbox=False,
            )

    def relation_observation_cached(
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
        key_a = tuple(geom_a_key) if isinstance(geom_a_key, tuple) else geom_a_key
        key_b = tuple(geom_b_key) if isinstance(geom_b_key, tuple) else geom_b_key
        cache_key = None
        if key_a is not None and key_b is not None:
            cache_key = (
                key_a,
                key_b,
                float(scale_min),
                float(contact_margin_px),
                float(near_thresh_n),
                float(exact_gap_max_n),
            )
            if cache_key in self._relation_obs_cache:
                return self._relation_obs_cache[cache_key]

        obs = compute_relation_observation(
            geom_a,
            geom_b,
            scale_min=float(scale_min),
            contact_margin_px=float(contact_margin_px),
            near_thresh_n=float(near_thresh_n),
            exact_gap_max_n=float(exact_gap_max_n),
            mask_runtime_a=self._mask_runtime_by_geom_key.get(key_a, None) if key_a is not None else None,
            mask_runtime_b=self._mask_runtime_by_geom_key.get(key_b, None) if key_b is not None else None,
        )
        if cache_key is not None:
            self._relation_obs_cache[cache_key] = obs
        return obs

    def resolve(
        self,
        *,
        ambiguous_entries: list[dict],
        reports_by_det_id: dict,
        detections: list,
        decided_matches: list[tuple[int, int, float]],
    ) -> tuple[list[tuple[int, int, float]], list[dict]]:
        if not self.enabled or not ambiguous_entries:
            self.last_debug_rows = []
            self.last_pair_anchor_rows = []
            self.last_resolved_sources = {}
            return [], list(ambiguous_entries or [])

        det_geom_by_id = self.build_det_geom_by_id(detections)
        if not det_geom_by_id:
            self.last_debug_rows = []
            self.last_pair_anchor_rows = []
            self.last_resolved_sources = {}
            return [], list(ambiguous_entries or [])

        self.reset_relation_runtime_cache()
        self.register_relation_geometries(prefix="det", geom_by_id=det_geom_by_id)

        ambiguous_det_ids = {
            int(item.get("det_id", -1))
            for item in (ambiguous_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        stable_decided_matches = [
            (int(det_id), int(oid), float(score))
            for det_id, oid, score in (decided_matches or [])
            if int(det_id) not in ambiguous_det_ids
        ]

        matched_geom_by_oid = self.build_matched_geom_by_oid(
            decided_matches=stable_decided_matches,
            det_geom_by_id=det_geom_by_id,
        )
        soft_anchor_pool = self.build_soft_anchor_pool(
            reports_by_det_id=reports_by_det_id,
            det_geom_by_id=det_geom_by_id,
            decided_matches=stable_decided_matches,
            excluded_det_ids=ambiguous_det_ids,
        )
        anchor_geom_by_oid = dict(matched_geom_by_oid)
        for oid, pack in (soft_anchor_pool or {}).items():
            if int(oid) not in anchor_geom_by_oid:
                geom = pack.get("geom", None)
                if isinstance(geom, dict):
                    anchor_geom_by_oid[int(oid)] = dict(geom)
        self.register_relation_geometries(prefix="anchor", geom_by_id=anchor_geom_by_oid)
        components = self.build_components(
            ambiguous_entries=ambiguous_entries,
            det_geom_by_id=det_geom_by_id,
        )
        if not components:
            self.last_debug_rows = []
            self.last_pair_anchor_rows = []
            self.last_resolved_sources = {}
            self.reset_relation_runtime_cache()
            return [], list(ambiguous_entries or [])

        remaining_by_det_id = {
            int(item.get("det_id", -1)): dict(item)
            for item in (ambiguous_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        resolved_matches: list[tuple[int, int, float]] = []
        debug_rows = []
        resolved_sources = {}

        for component_idx, component in enumerate(components):
            policy_mode = self.component_policy_mode(
                det_ids=list(component.get("det_ids", []) or []),
                candidates_by_det={
                    int(k): [int(x) for x in (v or [])]
                    for k, v in ((component.get("candidates_by_det", {}) or {}).items())
                },
            )
            dbg_row = {
                "det_ids": list(component.get("det_ids", []) or []),
                "candidate_union": list(component.get("candidate_union", []) or []),
                "status": "pending",
                "reason": "",
                "anchor_ids": [],
                "anchor_candidates": [],
                "best_score": None,
                "second_score": None,
                "gap": None,
                "core_score": None,
                "core_second_score": None,
                "core_gap": None,
                "policy_mode": str(policy_mode),
                "evidence": None,
                "anchor_term": None,
                "history_term": None,
                "frame_term": None,
                "anchor_quality_term": None,
                "order_term": None,
                "peer_term": None,
                "sibling_term": None,
                "visual_term": None,
                "known_assignments": 0,
                "top_solutions": [],
                "stable_det_assignments": {},
                "frontier_size": 0,
            }
            resolvability_reason = self.component_resolvability_reason(component)
            if resolvability_reason is not None:
                dbg_row["status"] = "skip_not_resolvable"
                dbg_row["reason"] = str(resolvability_reason)
                debug_rows.append(dbg_row)
                continue

            if str(policy_mode) != "full_sibling":
                dbg_row["status"] = "skip_not_full_sibling"
                dbg_row["reason"] = str(policy_mode)
                debug_rows.append(dbg_row)
                continue

            ranked_anchor_candidates = self.rank_anchor_candidates(
                component=component,
                decided_matches=stable_decided_matches,
                det_geom_by_id=det_geom_by_id,
                anchor_geom_by_oid=anchor_geom_by_oid,
                soft_anchor_pool=soft_anchor_pool,
            )
            dbg_row["anchor_candidates"] = list(ranked_anchor_candidates[: int(self.debug_anchor_topk)])
            anchor_ids = [int(item["anchor_id"]) for item in ranked_anchor_candidates if bool(item.get("selected", False))]
            dbg_row["anchor_ids"] = list(anchor_ids)

            solution = self.solve_component(
                component=component,
                reports_by_det_id=reports_by_det_id,
                det_geom_by_id=det_geom_by_id,
                anchor_ids=anchor_ids,
                anchor_geom_by_oid=anchor_geom_by_oid,
            )
            if solution is None:
                dbg_row["status"] = "skip_no_valid_assignment"
                debug_rows.append(dbg_row)
                continue

            dbg_row["best_score"] = float(solution["best_score"])
            dbg_row["second_score"] = float(solution["second_score"])
            dbg_row["gap"] = float(solution["gap"])
            dbg_row["core_score"] = float(solution.get("core_score", 0.0))
            dbg_row["core_second_score"] = float(solution.get("core_second_score", -1.0))
            dbg_row["core_gap"] = float(solution.get("core_gap", 0.0))
            dbg_row["policy_mode"] = str(solution.get("policy_mode", "") or "")
            dbg_row["evidence"] = float(solution["evidence"])
            dbg_row["anchor_term"] = float(solution.get("anchor_term", 0.0))
            dbg_row["history_term"] = float(solution.get("history_term", 0.0))
            dbg_row["frame_term"] = float(solution.get("frame_term", 0.0))
            dbg_row["anchor_quality_term"] = float(solution.get("anchor_quality_term", 0.0))
            dbg_row["order_term"] = float(solution.get("order_term", 0.0))
            dbg_row["peer_term"] = float(solution.get("peer_term", 0.0))
            dbg_row["sibling_term"] = float(solution.get("sibling_term", 0.0))
            dbg_row["visual_term"] = float(solution.get("visual_term", 0.0))
            dbg_row["known_assignments"] = int(solution.get("known_assignments", 0) or 0)
            dbg_row["top_solutions"] = list(solution.get("top_solutions", []) or [])
            dbg_row["stable_det_assignments"] = dict(solution.get("stable_det_assignments", {}) or {})
            dbg_row["frontier_size"] = int(solution.get("frontier_size", 0) or 0)

            if float(solution["evidence"]) < float(self.min_total_evidence):
                dbg_row["status"] = "keep_low_evidence"
                debug_rows.append(dbg_row)
                continue
            if float(solution["best_score"]) < float(self.min_assignment_score):
                dbg_row["status"] = "keep_low_score"
                debug_rows.append(dbg_row)
                continue
            min_core_score, min_core_gap = self.policy_core_thresholds(
                policy_mode=str(solution.get("policy_mode", "") or ""),
                det_count=len(component.get("det_ids", []) or []),
            )
            if str(solution.get("core_rank_status", "")) == "visual_rank_flip":
                dbg_row["status"] = "keep_visual_rank_flip"
                debug_rows.append(dbg_row)
                continue
            if float(solution.get("core_score", 0.0) or 0.0) < float(min_core_score):
                dbg_row["status"] = "keep_low_core_score"
                debug_rows.append(dbg_row)
                continue
            if float(solution.get("core_gap", 0.0) or 0.0) < float(min_core_gap):
                dbg_row["status"] = "keep_low_core_gap"
                debug_rows.append(dbg_row)
                continue
            if float(solution["gap"]) < float(self.min_gap):
                partial = self.accepted_det_assignments(
                    solution=solution,
                    det_count=len(component.get("det_ids", []) or []),
                    policy_mode=str(solution.get("policy_mode", "") or ""),
                )
                if partial:
                    for det_id, oid in partial.items():
                        remaining_by_det_id.pop(int(det_id), None)
                        if self.is_real_object_id(int(oid)):
                            rep = reports_by_det_id.get(int(det_id), None)
                            score_final = self.resolve_candidate_score(rep, int(oid))
                            resolved_matches.append((int(det_id), int(oid), float(score_final)))
                            resolved_sources[int(det_id)] = "distance_partial"
                    dbg_row["status"] = "partial_resolved"
                else:
                    dbg_row["status"] = "keep_low_gap"
                debug_rows.append(dbg_row)
                continue

            accepted = self.accepted_det_assignments(
                solution=solution,
                det_count=len(component.get("det_ids", []) or []),
                policy_mode=str(solution.get("policy_mode", "") or ""),
            )
            if not accepted:
                dbg_row["status"] = "keep_not_stable_local"
                debug_rows.append(dbg_row)
                continue

            for det_id, oid in accepted.items():
                remaining_by_det_id.pop(int(det_id), None)
                if self.is_real_object_id(int(oid)):
                    rep = reports_by_det_id.get(int(det_id), None)
                    score_final = self.resolve_candidate_score(rep, int(oid))
                    resolved_matches.append((int(det_id), int(oid), float(score_final)))
                    resolved_sources[int(det_id)] = "distance_full"
            if len(accepted) >= len(solution.get("assignment", {}) or {}):
                dbg_row["status"] = "resolved"
            else:
                dbg_row["status"] = "partial_resolved"
            debug_rows.append(dbg_row)

        self.last_debug_rows = list(debug_rows)
        self.last_pair_anchor_rows = self.build_frame_pair_anchor_rows(
            component_rows=debug_rows,
            det_geom_by_id=det_geom_by_id,
            anchor_geom_by_oid=anchor_geom_by_oid,
        )
        self.last_resolved_sources = dict(resolved_sources)
        remaining_entries = [item for _, item in sorted(remaining_by_det_id.items())]
        self.reset_relation_runtime_cache()
        return resolved_matches, remaining_entries

    def debug_pack(self) -> dict:
        return {
            "components": list(self.last_debug_rows or []),
            "pair_anchors": list(self.last_pair_anchor_rows or []),
        }

    def resolved_source_by_det_id(self) -> dict[int, str]:
        return dict(self.last_resolved_sources or {})
