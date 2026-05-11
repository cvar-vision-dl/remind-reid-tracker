from __future__ import annotations

from association.disambiguation.known_set_distance_disambiguator import KnownSetDistanceDisambiguator
from association.engine.post_assignment import (
    NormalizedAssignmentState,
    PostAssignmentSupport,
    TemporalResolutionState,
)
from utils.config import cfg_bool, cfg_float, cfg_get, cfg_int


class AssignmentResultApplier:
    def __init__(self, *, outcome_policy, config, memory_store=None):
        self.outcome_policy = outcome_policy
        self.memory_store = memory_store
        config = config or {}
        identity_stability_cfg_key = "association.matching.hungarian.identity_stability"
        legacy_identity_stability_cfg_key = "association.matching.identity_stability"

        self.identity_stability_enabled = cfg_bool(
            config,
            f"{identity_stability_cfg_key}.enabled",
            cfg_bool(config, f"{legacy_identity_stability_cfg_key}.enabled", True),
        )
        self.identity_stability_alt_margin = cfg_float(
            config,
            f"{identity_stability_cfg_key}.alt_margin",
            cfg_float(config, f"{legacy_identity_stability_cfg_key}.alt_margin", 0.05),
        )
        self.identity_stability_keep_margin = cfg_float(
            config,
            f"{identity_stability_cfg_key}.keep_margin",
            cfg_float(config, f"{legacy_identity_stability_cfg_key}.keep_margin", 0.03),
        )
        self.identity_stability_component_max_size = cfg_int(
            config,
            f"{identity_stability_cfg_key}.component_max_size",
            cfg_int(config, f"{legacy_identity_stability_cfg_key}.component_max_size", 4),
            min_value=1,
        )
        self.identity_stability_assignment_gap_max = cfg_float(
            config,
            f"{identity_stability_cfg_key}.assignment_gap_max",
            cfg_float(
                config,
                f"{legacy_identity_stability_cfg_key}.assignment_gap_max",
                self.identity_stability_alt_margin,
            ),
        )
        self.identity_stability_fragile_gate_reasons = {
            str(x).upper()
            for x in (
                cfg_get(
                    config,
                    f"{identity_stability_cfg_key}.fragile_gate_reasons",
                    cfg_get(config, f"{legacy_identity_stability_cfg_key}.fragile_gate_reasons", ["SETS_RESCUE"]),
                )
                or []
            )
        }
        self.committed_new_enabled = cfg_bool(config, "association.matching.hungarian.committed_new_competition.enabled", True)
        self.committed_new_parent_max_age = cfg_int(
            config,
            "association.matching.hungarian.committed_new_competition.parent_max_age",
            2,
        )
        self.committed_new_min_score = cfg_float(
            config,
            "association.matching.hungarian.committed_new_competition.min_score",
            cfg_float(config, "association.matching.match_thr", 0.0),
        )
        self.committed_new_pair_gap_max = cfg_float(
            config,
            "association.matching.hungarian.committed_new_competition.pair_gap_max",
            0.05,
        )
        self.committed_new_require_young_parent = cfg_bool(
            config,
            "association.matching.hungarian.committed_new_competition.require_young_parent",
            True,
        )
        self.committed_new_require_single_create = cfg_bool(
            config,
            "association.matching.hungarian.committed_new_competition.require_single_create_per_parent",
            True,
        )
        self.committed_new_allow_confirmed_parent = cfg_bool(
            config,
            "association.matching.hungarian.committed_new_competition.allow_confirmed_parent",
            True,
        )
        self.committed_new_confirmed_min_score = cfg_float(
            config,
            "association.matching.hungarian.committed_new_competition.confirmed_min_score",
            max(self.committed_new_min_score, 0.85),
        )
        self.committed_new_confirmed_pair_gap_max = cfg_float(
            config,
            "association.matching.hungarian.committed_new_competition.confirmed_pair_gap_max",
            min(self.committed_new_pair_gap_max, 0.03),
        )
        self.known_set_disambiguator = KnownSetDistanceDisambiguator(
            config=config,
            memory_store=memory_store,
        )
        self.temporal_reconcile_max_passes = cfg_int(
            config,
            "association.ambiguous_tracks.known_set_distance_disambiguation.max_passes",
            3,
            min_value=1,
        )
        self.post_assignment_support = PostAssignmentSupport()

    def publish_temporal_debug(
        self,
        *,
        out,
        temporal_state: TemporalResolutionState,
    ) -> None:
        debug_bucket = self.ensure_debug_bucket(out)
        debug_bucket["known_set_distance_disambiguation"] = dict(temporal_state.known_set_distance_disambiguation)
        debug_bucket["postcreate_temporal"] = [
            dict(item)
            for item in (temporal_state.postcreate_debug_entries or [])
            if isinstance(item, dict)
        ]

    def publish_final_pack_to_output(
        self,
        *,
        out,
        final_pack: dict,
        resolved_source_by_det_id: dict[int, str],
    ) -> None:
        out.decided_matches.extend(
            {
                "det_id": int(det_id),
                "object_id": int(obj_id),
                "score_final": float(score_final),
                "source": str(resolved_source_by_det_id.get(int(det_id), "association")),
            }
            for det_id, obj_id, score_final in final_pack["matches"]
        )
        out.to_create.extend(dict(item) for item in final_pack["create_entries"])
        out.to_ambiguous.extend(dict(item) for item in final_pack["ambiguous_entries"])
        out.to_provisional_new.extend(dict(item) for item in final_pack["provisional_entries"])
        out.assigned_by_det_id = dict(final_pack["assigned_by_det_id"])

    def apply(
        self,
        *,
        out,
        detections: list,
        decided_matches: list[tuple[int, int, float]],
        to_create: list[tuple[int, int]],
        timer,
    ) -> None:
        normalized_state = self.build_normalized_assignment_state(decided_matches)
        decided_matches, to_create = timer.run(
            "post_assignment/identity_stability",
            self.apply_identity_stability_policy,
            reports_by_det_id=out.reports_by_det_id,
            decided_matches=normalized_state.decided_matches,
            to_create=to_create,
        )
        guarded_state = self.build_normalized_assignment_state(decided_matches)
        create_entries = self.prepare_create_entries_after_guards(
            to_create=to_create,
            assigned_by_det_id=guarded_state.assigned_by_det_id,
        )
        identity_ambiguous_entries = self.build_identity_stability_ambiguous_entries(
            reports_by_det_id=out.reports_by_det_id,
            create_entries=create_entries,
        )
        if identity_ambiguous_entries:
            create_entries = self.drop_create_entries_for_det_ids(
                create_entries,
                self.det_ids_from_entries(identity_ambiguous_entries),
            )
        ambiguous_pack = timer.run(
            "post_assignment/ambiguous_candidates",
            self.build_post_assignment_ambiguous_entries,
            reports_by_det_id=out.reports_by_det_id,
            decided_matches=guarded_state.decided_matches,
            create_entries=create_entries,
            assigned_by_det_id=guarded_state.assigned_by_det_id,
            identity_ambiguous_entries=identity_ambiguous_entries,
        )
        ambiguous_entries = list((ambiguous_pack or {}).get("entries", []) or [])
        temporal_state = timer.run(
            "post_assignment/temporal_reconcile",
            self.reconcile_temporal_resolution,
            reports_by_det_id=out.reports_by_det_id,
            detections=detections,
            decided_matches=guarded_state.decided_matches,
            create_entries=create_entries,
            ambiguous_entries=ambiguous_entries,
            assigned_by_det_id=guarded_state.assigned_by_det_id,
        )
        self.publish_temporal_debug(
            out=out,
            temporal_state=temporal_state,
        )
        blocked_create_det_ids = self.det_ids_from_entries(temporal_state.provisional_entries)
        blocked_create_det_ids.update(self.det_ids_from_entries(temporal_state.ambiguous_entries))
        create_entries = self.drop_create_entries_for_det_ids(create_entries, blocked_create_det_ids)

        final_pack = timer.run(
            "post_assignment/final_pack",
            self.build_final_decision_pack,
            decided_matches=temporal_state.decided_matches,
            create_entries=create_entries,
            ambiguous_entries=temporal_state.ambiguous_entries,
            provisional_entries=temporal_state.provisional_entries,
        )
        self.publish_final_pack_to_output(
            out=out,
            final_pack=final_pack,
            resolved_source_by_det_id=temporal_state.resolved_source_by_det_id,
        )

        out.geom_by_object_id = timer.run(
            "post_assignment/geom_pack",
            self.build_geom_by_object_id,
            decided_matches=out.decided_matches,
            detections=detections,
        )

        timer.run(
            "post_assignment/finalize",
            self.outcome_policy.annotate_reports_final_decisions,
            reports_by_det_id=out.reports_by_det_id,
            assigned_by_det_id=out.assigned_by_det_id,
            created_det_ids={int(x["det_id"]) for x in out.to_create},
            ambiguous_by_det_id={int(x["det_id"]): dict(x) for x in (out.to_ambiguous or [])},
            provisional_by_det_id={int(x["det_id"]): dict(x) for x in (out.to_provisional_new or [])},
            score_final_by_det_id=final_pack["score_final_by_det_id"],
        )

    def build_normalized_assignment_state(
        self,
        decided_matches: list[tuple[int, int, float]] | None,
    ) -> NormalizedAssignmentState:
        pack = self.post_assignment_support.build_assignment_state(decided_matches)
        return NormalizedAssignmentState(
            decided_matches=list(pack["decided_matches"]),
            assigned_by_det_id=dict(pack["assigned_by_det_id"]),
        )

    def prepare_create_entries_after_guards(
        self,
        *,
        to_create: list[tuple[int, int]] | list[dict] | None,
        assigned_by_det_id: dict[int, int],
    ) -> list[dict]:
        create_entries = self.normalize_create_entries(to_create)
        return self.drop_create_entries_for_det_ids(create_entries, assigned_by_det_id.keys())

    def build_post_assignment_ambiguous_entries(
        self,
        *,
        reports_by_det_id: dict,
        decided_matches: list[tuple[int, int, float]],
        create_entries: list[dict],
        assigned_by_det_id: dict[int, int],
        identity_ambiguous_entries: list[dict] | None = None,
    ) -> dict:
        committed_new_entries = self.build_committed_new_competition_entries(
            reports_by_det_id=reports_by_det_id,
            decided_matches=decided_matches,
            to_create=create_entries,
            assigned_by_det_id=assigned_by_det_id,
        )
        ambiguous_entries = self.outcome_policy.build_ambiguous_track_candidates(
            reports_by_det_id=reports_by_det_id,
            decided_matches=decided_matches,
            to_create=create_entries,
            assigned_by_det_id=assigned_by_det_id,
        )
        merged_by_det_id: dict[int, dict] = {}
        selected_source_by_det_id: dict[int, str] = {}
        source_priority = {
            "ambiguous_track_policy": 10,
            "identity_stability": 20,
            "committed_new_competition": 30,
        }

        def register(item: dict, *, source: str) -> None:
            det_id = int(item.get("det_id", -1))
            if det_id < 0:
                return
            existing = merged_by_det_id.get(int(det_id), None)
            current_priority = int(source_priority.get(str(source), 0))
            existing_priority = int(source_priority.get(str(selected_source_by_det_id.get(int(det_id), "")), 0))
            if existing is None or current_priority >= existing_priority:
                merged_item = dict(item)
                merged_by_det_id[int(det_id)] = merged_item
                selected_source_by_det_id[int(det_id)] = str(source)

        for source, entries in (
            ("ambiguous_track_policy", ambiguous_entries),
            ("identity_stability", identity_ambiguous_entries),
            ("committed_new_competition", committed_new_entries),
        ):
            for item in (entries or []):
                if isinstance(item, dict):
                    register(dict(item), source=str(source))

        merged_entries = []
        for det_id, item in sorted(merged_by_det_id.items(), key=lambda kv: int(kv[0])):
            merged_entries.append(dict(item))

        return {
            "entries": list(merged_entries),
        }

    def ensure_debug_bucket(self, out) -> dict:
        dbg = getattr(out, "debug", None)
        if not isinstance(dbg, dict):
            dbg = {}
            out.debug = dbg
        return dbg

    def reconcile_temporal_resolution(
        self,
        *,
        reports_by_det_id: dict,
        detections: list,
        decided_matches: list[tuple[int, int, float]],
        create_entries: list[dict],
        ambiguous_entries: list[dict],
        assigned_by_det_id: dict[int, int],
    ) -> TemporalResolutionState:
        pack = self.reconcile_known_ambiguity_and_postcreate(
            reports_by_det_id=reports_by_det_id,
            detections=detections,
            decided_matches=decided_matches,
            create_entries=create_entries,
            ambiguous_entries=ambiguous_entries,
            assigned_by_det_id=assigned_by_det_id,
        )
        return TemporalResolutionState(
            decided_matches=list(pack["decided_matches"]),
            create_entries=list(pack["create_entries"]),
            ambiguous_entries=list(pack["ambiguous_entries"]),
            provisional_entries=list(pack["provisional_entries"]),
            resolved_source_by_det_id=dict(pack["resolved_source_by_det_id"]),
            known_set_distance_disambiguation=dict(pack["known_set_distance_disambiguation"]),
            postcreate_debug_entries=list(pack["postcreate_debug_entries"]),
        )

    def reconcile_known_ambiguity_and_postcreate(
        self,
        *,
        reports_by_det_id: dict,
        detections: list,
        decided_matches: list[tuple[int, int, float]],
        create_entries: list[dict],
        ambiguous_entries: list[dict],
        assigned_by_det_id: dict[int, int],
    ) -> dict:
        current_matches = self.normalize_matches(decided_matches)
        create_pool = [dict(item) for item in (create_entries or []) if isinstance(item, dict)]
        current_ambiguous_entries = [dict(item) for item in (ambiguous_entries or []) if isinstance(item, dict)]
        current_assigned_by_det_id = {int(k): int(v) for k, v in ((assigned_by_det_id or {}).items())}

        resolved_source_by_det_id: dict[int, str] = {}
        provisional_entries: list[dict] = []
        postcreate_debug_entries: list[dict] = []
        disambiguation_passes: list[dict] = []
        all_components: list[dict] = []
        all_pair_anchors: list[dict] = []
        max_passes = int(self.temporal_reconcile_max_passes)

        if not current_ambiguous_entries and not create_pool:
            return {
                "decided_matches": current_matches,
                "create_entries": create_pool,
                "ambiguous_entries": current_ambiguous_entries,
                "provisional_entries": provisional_entries,
                "resolved_source_by_det_id": resolved_source_by_det_id,
                "known_set_distance_disambiguation": {
                    "components": [],
                    "pair_anchors": [],
                    "passes": [],
                    "resolved_source_by_det_id": {},
                },
                "postcreate_debug_entries": postcreate_debug_entries,
            }

        for pass_idx in range(max_passes):
            pass_input_entries = current_ambiguous_entries
            resolved_matches, remaining_ambiguous_entries = self.known_set_disambiguator.resolve(
                ambiguous_entries=pass_input_entries,
                reports_by_det_id=reports_by_det_id,
                detections=detections,
                decided_matches=current_matches,
            )
            pass_debug_pack = self.known_set_disambiguator.debug_pack()
            pass_sources = self.known_set_disambiguator.resolved_source_by_det_id()

            if resolved_matches:
                resolved_det_ids = {int(det_id) for det_id, _, _ in (resolved_matches or [])}
                current_matches = self.merge_matches(
                    base_matches=current_matches,
                    override_matches=resolved_matches,
                )
                current_assigned_by_det_id.update(
                    {
                        int(det_id): int(obj_id)
                        for det_id, obj_id, _ in (resolved_matches or [])
                    }
                )
                create_pool = self.drop_create_entries_for_det_ids(create_pool, resolved_det_ids)
                for det_id, source in (pass_sources or {}).items():
                    src = str(source or "")
                    if pass_idx > 0 and src.startswith("distance_"):
                        src = f"{src}_postcreate"
                    resolved_source_by_det_id[int(det_id)] = str(src)

            remaining_ambiguous_det_ids = {
                int(item["det_id"])
                for item in (remaining_ambiguous_entries or [])
                if int(item.get("det_id", -1)) >= 0
            }
            postcreate_temporal = self.outcome_policy.build_postcreate_temporal_decisions(
                reports_by_det_id=reports_by_det_id,
                to_create=create_pool,
                assigned_by_det_id=current_assigned_by_det_id,
                excluded_det_ids=remaining_ambiguous_det_ids,
            )
            provisional_entries = list((postcreate_temporal or {}).get("provisional_entries", []) or [])
            promoted_ambiguous_entries = list((postcreate_temporal or {}).get("ambiguous_entries", []) or [])
            postcreate_debug_entries = [
                item
                for item in ((postcreate_temporal or {}).get("debug_entries", []) or [])
                if isinstance(item, dict)
            ]
            next_ambiguous_entries = self.merge_ambiguous_entries(
                base_entries=remaining_ambiguous_entries,
                extra_entries=promoted_ambiguous_entries,
            )

            input_det_ids = sorted(
                int(item.get("det_id", -1))
                for item in (pass_input_entries or [])
                if int(item.get("det_id", -1)) >= 0
            )
            resolved_det_ids = sorted(int(det_id) for det_id, _, _ in (resolved_matches or []))
            remaining_det_ids = sorted(
                int(item.get("det_id", -1))
                for item in (next_ambiguous_entries or [])
                if int(item.get("det_id", -1)) >= 0
            )
            pass_components = list((pass_debug_pack or {}).get("components", []) or [])
            pass_pair_anchors = list((pass_debug_pack or {}).get("pair_anchors", []) or [])

            disambiguation_passes.append(
                {
                    "pass_index": int(pass_idx + 1),
                    "input_det_ids": list(input_det_ids),
                    "resolved_det_ids": list(resolved_det_ids),
                    "remaining_det_ids": list(remaining_det_ids),
                    "components": pass_components,
                    "pair_anchors": pass_pair_anchors,
                }
            )

            pass_index = int(pass_idx + 1)
            for component in pass_components:
                if not isinstance(component, dict):
                    continue
                comp_row = dict(component)
                comp_row["pass_index"] = int(pass_index)
                comp_row["pass_input_det_ids"] = list(input_det_ids)
                comp_row["pass_resolved_det_ids"] = list(resolved_det_ids)
                comp_row["pass_remaining_det_ids"] = list(remaining_det_ids)
                all_components.append(comp_row)

            for anchor_pack in pass_pair_anchors:
                if not isinstance(anchor_pack, dict):
                    continue
                pair_row = dict(anchor_pack)
                pair_row["pass_index"] = int(pass_index)
                pair_row["pass_input_det_ids"] = list(input_det_ids)
                pair_row["pass_resolved_det_ids"] = list(resolved_det_ids)
                pair_row["pass_remaining_det_ids"] = list(remaining_det_ids)
                all_pair_anchors.append(pair_row)

            same_input = self.same_ambiguous_det_ids(pass_input_entries, next_ambiguous_entries)
            current_ambiguous_entries = next_ambiguous_entries
            if not resolved_matches and same_input:
                break

        if disambiguation_passes:
            known_set_debug = {
                "components": list(all_components),
                "pair_anchors": list(all_pair_anchors),
                "passes": list(disambiguation_passes),
                "resolved_source_by_det_id": dict(resolved_source_by_det_id),
            }
        else:
            known_set_debug = {
                "components": [],
                "pair_anchors": [],
                "passes": [],
                "resolved_source_by_det_id": {},
            }

        return {
            "decided_matches": current_matches,
            "create_entries": create_pool,
            "ambiguous_entries": current_ambiguous_entries,
            "provisional_entries": provisional_entries,
            "resolved_source_by_det_id": resolved_source_by_det_id,
            "known_set_distance_disambiguation": known_set_debug,
            "postcreate_debug_entries": postcreate_debug_entries,
        }

    def merge_ambiguous_entries(
        self,
        *,
        base_entries: list[dict] | None,
        extra_entries: list[dict] | None,
    ) -> list[dict]:
        return self.post_assignment_support.merge_ambiguous_entries(
            base_entries=base_entries,
            extra_entries=extra_entries,
        )

    def same_ambiguous_det_ids(
        self,
        prev_entries: list[dict] | None,
        next_entries: list[dict] | None,
    ) -> bool:
        return self.post_assignment_support.same_ambiguous_det_ids(prev_entries, next_entries)

    def normalize_matches(
        self,
        matches: list[tuple[int, int, float]] | None,
    ) -> list[tuple[int, int, float]]:
        return self.post_assignment_support.normalize_matches(matches)

    def merge_matches(
        self,
        *,
        base_matches: list[tuple[int, int, float]] | None,
        override_matches: list[tuple[int, int, float]] | None,
    ) -> list[tuple[int, int, float]]:
        return self.post_assignment_support.merge_matches(
            base_matches=base_matches,
            override_matches=override_matches,
        )

    def normalize_create_entries(
        self,
        to_create: list[tuple[int, int]] | list[dict] | None,
    ) -> list[dict]:
        return self.post_assignment_support.normalize_create_entries(to_create)

    def drop_create_entries_for_det_ids(
        self,
        create_entries: list[dict] | None,
        det_ids,
    ) -> list[dict]:
        return self.post_assignment_support.drop_create_entries_for_det_ids(create_entries, det_ids)

    def det_ids_from_entries(self, entries: list[dict] | None) -> set[int]:
        return {
            int(item.get("det_id", -1))
            for item in (entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }

    def build_final_decision_pack(
        self,
        *,
        decided_matches: list[tuple[int, int, float]],
        create_entries: list[dict],
        ambiguous_entries: list[dict],
        provisional_entries: list[dict],
    ) -> dict:
        return self.post_assignment_support.build_final_decision_pack(
            decided_matches=decided_matches,
            create_entries=create_entries,
            ambiguous_entries=ambiguous_entries,
            provisional_entries=provisional_entries,
        )

    def return_identity_stability_unchanged(
        self,
        *,
        reports_by_det_id: dict,
        initial_matches: list[tuple[int, int, float]],
        to_create: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int, float]], list[dict]]:
        final_matches = list(initial_matches or [])
        final_creates = self.normalize_create_entries(to_create)
        return final_matches, final_creates

    def build_identity_candidate_edges(
        self,
        *,
        det_ids: list[int],
        reports_by_det_id: dict,
        class_current_obj_ids: set[int],
    ) -> tuple[dict[int, list[dict]], dict[int, set[int]]]:
        edges_by_det_id: dict[int, list[dict]] = {}
        det_ids_by_obj_id: dict[int, set[int]] = {}
        for det_id in (det_ids or []):
            rep = reports_by_det_id.get(int(det_id), None)
            if rep is None:
                continue
            kept: list[dict] = []
            for candidate in (getattr(rep, "candidates", None) or []):
                if not isinstance(candidate, dict):
                    continue
                oid = candidate.get("object_id", None)
                if oid is None:
                    continue
                oid = int(oid)
                if oid not in class_current_obj_ids:
                    continue
                if int(candidate.get("decision_keep", 0) or 0) != 1:
                    continue
                kept.append(candidate)
                det_ids_by_obj_id.setdefault(int(oid), set()).add(int(det_id))
            if kept:
                edges_by_det_id[int(det_id)] = list(kept)
        return edges_by_det_id, det_ids_by_obj_id

    @staticmethod
    def append_assignment_ambiguity_row(
        *,
        rows_by_class: dict[int, list[dict]],
        class_id: int,
        component_det_ids: set[int],
        component_obj_ids: set[int],
        assignment_ambiguity: dict,
    ) -> None:
        rows_by_class.setdefault(int(class_id), []).append(
            {
                "component_det_ids": sorted(int(x) for x in (component_det_ids or set())),
                "component_object_ids": sorted(int(x) for x in (component_obj_ids or set())),
                "is_ambiguous": bool((assignment_ambiguity or {}).get("is_ambiguous", False)),
                "reason": str((assignment_ambiguity or {}).get("reason", "") or ""),
                "gap": float((assignment_ambiguity or {}).get("gap", 0.0) or 0.0),
                "best_score": float((assignment_ambiguity or {}).get("best_score", 0.0) or 0.0),
                "second_score": float((assignment_ambiguity or {}).get("second_score", 0.0) or 0.0),
                "ambiguous_det_ids": [
                    int(x) for x in ((assignment_ambiguity or {}).get("ambiguous_det_ids", []) or [])
                ],
                "current_assignment": dict((assignment_ambiguity or {}).get("current_assignment", {}) or {}),
                "best_assignment": dict((assignment_ambiguity or {}).get("best_assignment", {}) or {}),
                "second_assignment": dict((assignment_ambiguity or {}).get("second_assignment", {}) or {}),
            }
        )

    def apply_component_assignment_ambiguity(
        self,
        *,
        class_id: int,
        component_det_ids: set[int],
        component_obj_ids: set[int],
        assignment_ambiguity: dict,
        match_by_det_id: dict[int, tuple[int, float]],
        final_match_by_det_id: dict[int, tuple[int, float]],
        create_by_det_id: dict[int, dict],
    ) -> bool:
        if not bool((assignment_ambiguity or {}).get("is_ambiguous", False)):
            return False

        best_assignment = dict((assignment_ambiguity or {}).get("best_assignment", {}) or {})
        ambiguous_det_ids = {
            int(x) for x in ((assignment_ambiguity or {}).get("ambiguous_det_ids", []) or [])
        }
        if not ambiguous_det_ids:
            return False

        for det_id, pack in best_assignment.items():
            if int(det_id) in ambiguous_det_ids:
                continue
            final_match_by_det_id[int(det_id)] = (int(pack["object_id"]), float(pack["score_final"]))
            create_by_det_id.pop(int(det_id), None)

        second_assignment = dict((assignment_ambiguity or {}).get("second_assignment", {}) or {})
        for det_id in component_det_ids:
            if int(det_id) not in ambiguous_det_ids:
                continue
            current = match_by_det_id.get(int(det_id), None)
            current_obj_id = int(current[0]) if current is not None else -1
            best_obj_id = int(best_assignment.get(int(det_id), {}).get("object_id", -1))
            second_obj_id = int(second_assignment.get(int(det_id), {}).get("object_id", -1))
            support_known_ids = sorted(
                {
                    int(oid)
                    for oid in (current_obj_id, best_obj_id, second_obj_id)
                    if int(oid) in component_obj_ids and int(oid) >= 0
                }
            )
            if len(support_known_ids) < 2:
                support_known_ids = sorted(int(x) for x in component_obj_ids)
            final_match_by_det_id.pop(int(det_id), None)
            entry = dict(create_by_det_id.get(int(det_id), {}))
            entry["det_id"] = int(det_id)
            entry["class_id"] = int(class_id)
            entry["origin_mode"] = "IDENTITY_STABILITY_KNOWN_COMPONENT"
            entry["origin_reason"] = "IDENTITY_COMPONENT_ASSIGNMENT_AMBIGUITY"
            entry["support_known_ids"] = list(support_known_ids)
            entry["related_known_ids"] = list(support_known_ids)
            entry["assignment_gap"] = float((assignment_ambiguity or {}).get("gap", 0.0) or 0.0)
            entry["assignment_best_score"] = float(
                (assignment_ambiguity or {}).get("best_score", 0.0) or 0.0
            )
            entry["assignment_second_score"] = float(
                (assignment_ambiguity or {}).get("second_score", 0.0) or 0.0
            )
            create_by_det_id[int(det_id)] = dict(entry)

        return True

    def apply_identity_stability_policy(
        self,
        *,
        reports_by_det_id: dict,
        decided_matches: list[tuple[int, int, float]],
        to_create: list[tuple[int, int]],
    ) -> tuple[list[tuple[int, int, float]], list[dict]]:
        if not self.identity_stability_enabled:
            return self.return_identity_stability_unchanged(
                reports_by_det_id=reports_by_det_id,
                initial_matches=list(decided_matches or []),
                to_create=to_create,
            )

        current_matches = [
            tuple((int(det_id), int(obj_id), float(score_final)))
            for det_id, obj_id, score_final in (decided_matches or [])
        ]
        if len(current_matches) < 2:
            return self.return_identity_stability_unchanged(
                reports_by_det_id=reports_by_det_id,
                initial_matches=current_matches,
                to_create=to_create,
            )

        match_by_det_id = {int(det_id): (int(obj_id), float(score_final)) for det_id, obj_id, score_final in current_matches}
        final_match_by_det_id = dict(match_by_det_id)
        create_by_det_id = {
            int(item.get("det_id", -1)): dict(item)
            for item in self.normalize_create_entries(to_create)
            if int(item.get("det_id", -1)) >= 0
        }

        matched_det_ids_by_class: dict[int, list[int]] = {}
        assignment_ambiguity_rows_by_class: dict[int, list[dict]] = {}
        for det_id, _, _ in current_matches:
            rep = reports_by_det_id.get(int(det_id), None)
            if rep is None:
                continue
            class_id = int(getattr(rep, "class_id", -1))
            matched_det_ids_by_class.setdefault(int(class_id), []).append(int(det_id))
            assignment_ambiguity_rows_by_class.setdefault(int(class_id), [])

        for class_id, det_ids in matched_det_ids_by_class.items():
            if len(det_ids) < 2:
                continue

            class_current_obj_ids = {
                int(match_by_det_id[int(det_id)][0])
                for det_id in det_ids
                if int(det_id) in match_by_det_id
            }
            if len(class_current_obj_ids) < 2:
                continue

            edges_by_det_id, det_ids_by_obj_id = self.build_identity_candidate_edges(
                det_ids=det_ids,
                reports_by_det_id=reports_by_det_id,
                class_current_obj_ids=class_current_obj_ids,
            )

            components = self.build_identity_components(
                det_ids=[int(x) for x in det_ids if int(x) in edges_by_det_id],
                edges_by_det_id=edges_by_det_id,
                det_ids_by_obj_id=det_ids_by_obj_id,
            )

            for component_det_ids, component_obj_ids in components:
                if len(component_det_ids) < 2 or len(component_obj_ids) < 2:
                    continue
                # Before locking a same-class component, compare the best two
                # complete assignments. If they are nearly tied, only the
                # detections that actually swap between those assignments should
                # stay temporal/ambiguous.
                assignment_ambiguity = self.analyze_identity_component_assignments(
                    component_det_ids=component_det_ids,
                    component_obj_ids=component_obj_ids,
                    edges_by_det_id=edges_by_det_id,
                    match_by_det_id=match_by_det_id,
                )
                self.append_assignment_ambiguity_row(
                    rows_by_class=assignment_ambiguity_rows_by_class,
                    class_id=int(class_id),
                    component_det_ids=component_det_ids,
                    component_obj_ids=component_obj_ids,
                    assignment_ambiguity=assignment_ambiguity,
                )
                if self.apply_component_assignment_ambiguity(
                    class_id=int(class_id),
                    component_det_ids=component_det_ids,
                    component_obj_ids=component_obj_ids,
                    assignment_ambiguity=assignment_ambiguity,
                    match_by_det_id=match_by_det_id,
                    final_match_by_det_id=final_match_by_det_id,
                    create_by_det_id=create_by_det_id,
                ):
                    continue
                if not self.component_has_identity_instability(
                    component_det_ids=component_det_ids,
                    component_obj_ids=component_obj_ids,
                    edges_by_det_id=edges_by_det_id,
                    match_by_det_id=match_by_det_id,
                ):
                    continue

                replacement_matches = self.resolve_identity_component_greedily(
                    component_det_ids=component_det_ids,
                    component_obj_ids=component_obj_ids,
                    edges_by_det_id=edges_by_det_id,
                )
                kept_det_ids = {int(det_id) for det_id, _, _ in replacement_matches}

                for det_id in component_det_ids:
                    final_match_by_det_id.pop(int(det_id), None)
                    if int(det_id) not in kept_det_ids:
                        entry = dict(create_by_det_id.get(int(det_id), {}))
                        entry["det_id"] = int(det_id)
                        entry["class_id"] = int(class_id)
                        entry["origin_mode"] = "IDENTITY_STABILITY_KNOWN_COMPONENT"
                        entry["support_known_ids"] = sorted(int(x) for x in component_obj_ids)
                        entry["related_known_ids"] = sorted(int(x) for x in component_obj_ids)
                        create_by_det_id[int(det_id)] = dict(entry)

                for det_id, obj_id, score_final in replacement_matches:
                    final_match_by_det_id[int(det_id)] = (int(obj_id), float(score_final))
                    create_by_det_id.pop(int(det_id), None)

        final_matches = []
        for det_id, _, _ in current_matches:
            pack = final_match_by_det_id.get(int(det_id), None)
            if pack is None:
                continue
            final_matches.append((int(det_id), int(pack[0]), float(pack[1])))

        extra_match_det_ids = set(int(det_id) for det_id, _, _ in final_matches)
        for det_id, pack in final_match_by_det_id.items():
            if int(det_id) in extra_match_det_ids:
                continue
            final_matches.append((int(det_id), int(pack[0]), float(pack[1])))

        final_creates = [
            dict(item)
            for _, item in sorted(create_by_det_id.items(), key=lambda kv: int(kv[0]))
            if isinstance(item, dict)
        ]
        return final_matches, final_creates

    def build_identity_stability_ambiguous_entries(
        self,
        *,
        reports_by_det_id: dict,
        create_entries: list[dict],
    ) -> list[dict]:
        if self.outcome_policy is None:
            return []

        out: list[dict] = []
        min_support_score = float(getattr(self.outcome_policy, "min_match_score", 0.0) or 0.0)
        max_candidates = int(getattr(self.outcome_policy, "amb_track_max_candidates", 3) or 3)

        for item in (create_entries or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("origin_mode", "") or "").upper() != "IDENTITY_STABILITY_KNOWN_COMPONENT":
                continue

            det_id = int(item.get("det_id", -1))
            rep = (reports_by_det_id or {}).get(int(det_id), None)
            if rep is None:
                continue

            support_known_ids = {
                int(x)
                for x in (item.get("support_known_ids", []) or [])
                if x is not None
            }
            if len(support_known_ids) < 2:
                continue

            candidates = [
                c
                for c in self.outcome_policy.iter_candidates(rep, scope="ambiguity")
                if isinstance(c, dict)
                and c.get("object_id", None) is not None
                and int(c.get("object_id")) in support_known_ids
            ]
            if len(candidates) < 2:
                continue

            score_map = self.outcome_policy.compute_comparable_score_map(candidates)
            scored = []
            for candidate in candidates:
                oid = int(candidate.get("object_id"))
                score = float(self.outcome_policy.temporal_candidate_score(candidate, score_map=score_map))
                if score < float(min_support_score):
                    continue
                scored.append((float(score), int(oid)))

            if len(scored) < 2:
                continue

            scored.sort(key=lambda item: (float(item[0]), int(item[1])), reverse=True)
            scored = scored[: max(2, int(max_candidates))]
            support_known_ids = [int(oid) for _, oid in scored]
            support_known_scores = {int(oid): float(score) for score, oid in scored}
            support = self.outcome_policy.make_provisional_support_profile(
                support_known_ids=support_known_ids,
                support_known_scores=support_known_scores,
                blocked_known_ids=[],
                blocked_known_scores={},
                context_mode="identity_stability_component",
            )
            decision = self.outcome_policy.make_ambiguous_decision_from_support(
                class_id=int(item.get("class_id", -1)),
                best_score=float(scored[0][0]),
                support=support,
            )
            payload = decision.as_payload(det_id=int(det_id))
            payload["reason"] = "KNOWN_BUT_AMBIGUOUS_IDENTITY_STABILITY"
            out.append(dict(payload))

        return out

    def build_committed_new_competition_entries(
        self,
        *,
        reports_by_det_id: dict,
        decided_matches: list[tuple[int, int, float]],
        to_create: list[dict],
        assigned_by_det_id: dict[int, int],
    ) -> list[dict]:
        if not self.committed_new_enabled or self.memory_store is None:
            return []

        create_entries = [dict(x) for x in (to_create or []) if isinstance(x, dict)]
        if not create_entries or not decided_matches:
            return []

        matched_det_by_obj_id = {int(obj_id): int(det_id) for det_id, obj_id, _ in (decided_matches or [])}
        score_final_by_det_id = {int(det_id): float(score_final) for det_id, _, score_final in (decided_matches or [])}
        create_candidates_by_parent: dict[int, list[dict]] = {}
        competition_rows_by_class: dict[int, list[dict]] = {}

        for item in create_entries:
            det_id = int(item.get("det_id", -1))
            rep = (reports_by_det_id or {}).get(int(det_id), None)
            if rep is None:
                continue
            top_parent = self.find_top_assigned_parent_candidate(
                report=rep,
                assigned_by_det_id=assigned_by_det_id,
            )
            if top_parent is None:
                continue

            parent_oid = int(top_parent["object_id"])
            parent_det_id = matched_det_by_obj_id.get(int(parent_oid), None)
            if parent_det_id is None:
                continue
            parent_obj = self.memory_store.get(int(parent_oid))
            if parent_obj is None:
                continue
            if not self.parent_is_eligible_for_committed_new(parent_obj):
                continue

            competition_mode = self.parent_new_competition_mode(
                top_parent_candidate=top_parent.get("candidate", None),
            )
            parent_rep = (reports_by_det_id or {}).get(int(parent_det_id), None)
            parent_match_score = float(score_final_by_det_id.get(int(parent_det_id), 0.0) or 0.0)
            if parent_rep is not None:
                parent_cand = self.find_candidate_for_object(getattr(parent_rep, "candidates", None) or [], int(parent_oid))
                if parent_cand is not None:
                    parent_match_score = float(self.candidate_known_score(parent_cand))

            create_score = float(top_parent["score"])
            min_score, pair_gap_max = self.parent_new_competition_thresholds(
                parent_obj=parent_obj,
                mode=str(competition_mode),
            )
            if create_score < float(min_score):
                continue
            if parent_match_score < float(min_score):
                continue
            if abs(float(parent_match_score) - float(create_score)) > float(pair_gap_max):
                continue
            if self.report_has_competing_supported_known(
                report=rep,
                primary_object_id=int(parent_oid),
                min_score=float(min_score),
            ):
                continue
            if self.report_has_competing_supported_known(
                report=parent_rep,
                primary_object_id=int(parent_oid),
                min_score=float(min_score),
            ):
                continue

            create_candidates_by_parent.setdefault(int(parent_oid), []).append(
                {
                    "det_id": int(det_id),
                    "class_id": int(item.get("class_id", -1)),
                    "parent_oid": int(parent_oid),
                    "parent_det_id": int(parent_det_id),
                    "create_score": float(create_score),
                    "parent_score": float(parent_match_score),
                    "mode": str(competition_mode),
                }
            )

        competitions: list[dict] = []
        used_parent_det_ids: set[int] = set()
        used_create_det_ids: set[int] = set()
        for parent_oid, packs in sorted(create_candidates_by_parent.items(), key=lambda kv: int(kv[0])):
            if self.committed_new_require_single_create and len(packs) != 1:
                continue
            packs = sorted(packs, key=lambda item: (float(item["create_score"]), -int(item["det_id"])), reverse=True)
            pack = packs[0]
            if int(pack["parent_det_id"]) in used_parent_det_ids or int(pack["det_id"]) in used_create_det_ids:
                continue
            used_parent_det_ids.add(int(pack["parent_det_id"]))
            used_create_det_ids.add(int(pack["det_id"]))
            competitions.append(dict(pack))
            class_id = int(pack.get("class_id", -1))
            if class_id >= 0:
                min_score, pair_gap_max = self.committed_new_thresholds_for_parent(self.memory_store.get(int(parent_oid)))
                competition_rows_by_class.setdefault(int(class_id), []).append(
                    {
                        "class_id": int(class_id),
                        "selected": True,
                        "parent_oid": int(parent_oid),
                        "parent_det_id": int(pack["parent_det_id"]),
                        "create_det_id": int(pack["det_id"]),
                        "create_score": float(pack["create_score"]),
                        "parent_score": float(pack["parent_score"]),
                        "min_score": float(min_score),
                        "pair_gap_max": float(pair_gap_max),
                        "mode": str(pack.get("mode", "matched_create") or "matched_create"),
                        "reason": "KNOWN_AND_COMMITTED_NEW_COMPETITION",
                    }
                )

        if not competitions:
            return []

        remaining_create_det_ids = {
            int(item.get("det_id", -1))
            for item in create_entries
            if int(item.get("det_id", -1)) >= 0 and int(item.get("det_id", -1)) not in used_create_det_ids
        }
        next_virtual_object_id = int(getattr(self.memory_store, "next_object_id", 0)) + int(len(remaining_create_det_ids))

        out: list[dict] = []
        for comp in competitions:
            virtual_oid = int(next_virtual_object_id)
            next_virtual_object_id += 1
            candidate_ids = [int(comp["parent_oid"]), int(virtual_oid)]
            parent_oid = int(comp["parent_oid"])
            create_score = float(comp["create_score"])
            parent_score = float(comp["parent_score"])
            ambiguous_scores = {
                int(parent_oid): float(max(parent_score, create_score)),
                int(virtual_oid): float(max(parent_score, create_score)),
            }
            base = {
                "class_id": int(comp["class_id"]),
                "candidate_ids": list(candidate_ids),
                "candidate_scores": dict(ambiguous_scores),
                "best_score": float(max(parent_score, create_score)),
                "score_gap": 0.0,
                "reason": "KNOWN_AND_COMMITTED_NEW_COMPETITION",
                "committed_new_object_id": int(virtual_oid),
                "committed_new_parent_ids": [int(parent_oid)],
                "committed_new_parent_scores": {int(parent_oid): float(create_score)},
                "committed_new_seed_det_id": int(comp["det_id"]),
                "committed_new_mode": str(comp.get("mode", "matched_create") or "matched_create"),
            }
            out.append(
                {
                    "det_id": int(comp["parent_det_id"]),
                    **base,
                }
            )
            out.append(
                {
                    "det_id": int(comp["det_id"]),
                    **base,
                }
            )

        return out

    def find_top_assigned_parent_candidate(
        self,
        *,
        report,
        assigned_by_det_id: dict[int, int],
    ) -> dict | None:
        assigned_object_ids = {int(x) for x in ((assigned_by_det_id or {}).values()) if x is not None}
        best = None
        best_score = float("-inf")
        for candidate in (getattr(report, "candidates", None) or []):
            if not isinstance(candidate, dict):
                continue
            oid = candidate.get("object_id", None)
            if oid is None:
                continue
            oid = int(oid)
            if oid not in assigned_object_ids:
                continue
            score = float(self.candidate_known_score(candidate))
            if score > best_score:
                best_score = float(score)
                best = {"object_id": int(oid), "score": float(score), "candidate": candidate}
        return best

    @staticmethod
    def candidate_known_score(candidate: dict) -> float:
        score = candidate.get("score_known", None)
        if score is None:
            score = candidate.get("score_final", candidate.get("score_sim", 0.0))
        return float(score or 0.0)

    @staticmethod
    def parent_new_competition_mode(*, top_parent_candidate: dict | None) -> str:
        if not isinstance(top_parent_candidate, dict):
            return "matched_create"
        known_plausible_keep = int(top_parent_candidate.get("known_plausible_keep", 0) or 0)
        decision_keep = int(top_parent_candidate.get("decision_keep", 0) or 0)
        if known_plausible_keep == 1 and decision_keep != 1:
            return "blocked_known_create"
        return "matched_create"

    def parent_new_competition_thresholds(
        self,
        *,
        parent_obj,
        mode: str,
    ) -> tuple[float, float]:
        min_score, pair_gap_max = self.committed_new_thresholds_for_parent(parent_obj)
        if str(mode or "") != "blocked_known_create":
            return float(min_score), float(pair_gap_max)

        # A blocked-known create is already telling us "the parent is good
        # enough, but the 1-to-1 assignment consumed it first". We keep the same
        # competition semantics, but avoid over-penalizing confirmed parents with
        # the stricter confirmed-only gap in this specific mode.
        return (
            float(min(float(min_score), float(self.committed_new_min_score))),
            float(max(float(pair_gap_max), float(self.committed_new_pair_gap_max))),
        )

    def report_has_competing_supported_known(
        self,
        *,
        report,
        primary_object_id: int,
        min_score: float,
    ) -> bool:
        if report is None or self.outcome_policy is None:
            return False

        candidates = [
            c
            for c in self.outcome_policy.iter_candidates(report, scope="ambiguity")
            if isinstance(c, dict) and c.get("object_id", None) is not None
        ]
        if len(candidates) < 2:
            return False

        supported = self.outcome_policy.ambiguous_supported_candidates(candidates)
        if len(supported) < 2:
            return False

        score_map = self.outcome_policy.compute_comparable_score_map(supported)
        primary_present = any(int(candidate.get("object_id")) == int(primary_object_id) for candidate in supported)
        if not primary_present:
            return False

        for candidate in supported:
            oid = int(candidate.get("object_id"))
            score = max(
                float(self.candidate_known_score(candidate)),
                float(self.outcome_policy.temporal_candidate_score(candidate, score_map=score_map)),
            )
            if int(oid) == int(primary_object_id):
                continue
            if float(score) >= float(min_score):
                return True
        return False

    def parent_is_eligible_for_committed_new(self, parent_obj) -> bool:
        if parent_obj is None:
            return False
        state = str(getattr(parent_obj, "state", "") or "").upper()
        age = int(getattr(parent_obj, "age", 0) or 0)
        if state in ("NEW", "TENTATIVE"):
            if not self.committed_new_require_young_parent:
                return True
            return bool(age <= int(self.committed_new_parent_max_age))
        if state == "CONFIRMED":
            return bool(self.committed_new_allow_confirmed_parent)
        return False

    def committed_new_thresholds_for_parent(self, parent_obj) -> tuple[float, float]:
        state = str(getattr(parent_obj, "state", "") or "").upper() if parent_obj is not None else ""
        if state == "CONFIRMED":
            return (
                float(self.committed_new_confirmed_min_score),
                float(self.committed_new_confirmed_pair_gap_max),
            )
        return (
            float(self.committed_new_min_score),
            float(self.committed_new_pair_gap_max),
        )

    def build_identity_components(
        self,
        *,
        det_ids: list[int],
        edges_by_det_id: dict[int, list[dict]],
        det_ids_by_obj_id: dict[int, set[int]],
    ) -> list[tuple[set[int], set[int]]]:
        components: list[tuple[set[int], set[int]]] = []
        pending = set(int(x) for x in (det_ids or []))

        while pending:
            seed_det_id = int(next(iter(pending)))
            stack = [("det", int(seed_det_id))]
            component_det_ids: set[int] = set()
            component_obj_ids: set[int] = set()

            while stack:
                node_kind, node_id = stack.pop()
                if node_kind == "det":
                    if int(node_id) in component_det_ids:
                        continue
                    component_det_ids.add(int(node_id))
                    pending.discard(int(node_id))
                    for candidate in (edges_by_det_id.get(int(node_id), []) or []):
                        oid = candidate.get("object_id", None)
                        if oid is None:
                            continue
                        stack.append(("obj", int(oid)))
                else:
                    if int(node_id) in component_obj_ids:
                        continue
                    component_obj_ids.add(int(node_id))
                    for det_id in (det_ids_by_obj_id.get(int(node_id), set()) or set()):
                        stack.append(("det", int(det_id)))

            components.append((component_det_ids, component_obj_ids))

        return components

    def analyze_identity_component_assignments(
        self,
        *,
        component_det_ids: set[int],
        component_obj_ids: set[int],
        edges_by_det_id: dict[int, list[dict]],
        match_by_det_id: dict[int, tuple[int, float]],
    ) -> dict:
        det_ids = sorted(int(x) for x in (component_det_ids or set()))
        obj_ids = sorted(int(x) for x in (component_obj_ids or set()))
        max_size = int(self.identity_stability_component_max_size)
        if len(det_ids) < 2 or len(obj_ids) < 2:
            return {"is_ambiguous": False, "reason": "COMPONENT_TOO_SMALL"}
        if len(det_ids) > max_size or len(obj_ids) > max_size or len(det_ids) != len(obj_ids):
            return {"is_ambiguous": False, "reason": "COMPONENT_SIZE_UNSUPPORTED"}

        edges_by_det_obj: dict[int, dict[int, dict]] = {}
        for det_id in det_ids:
            by_obj_id: dict[int, dict] = {}
            for candidate in (edges_by_det_id.get(int(det_id), []) or []):
                if not isinstance(candidate, dict):
                    continue
                oid = candidate.get("object_id", None)
                if oid is None:
                    continue
                oid = int(oid)
                if oid not in component_obj_ids:
                    continue
                prev = by_obj_id.get(int(oid), None)
                if prev is None or float(self.candidate_known_score(candidate)) > float(self.candidate_known_score(prev)):
                    by_obj_id[int(oid)] = candidate
            if not by_obj_id:
                return {"is_ambiguous": False, "reason": "MISSING_COMPONENT_EDGES"}
            edges_by_det_obj[int(det_id)] = dict(by_obj_id)

        det_order = sorted(det_ids, key=lambda det_id: (len(edges_by_det_obj.get(int(det_id), {})), int(det_id)))
        assignments: list[dict] = []

        def backtrack(idx: int, used_obj_ids: set[int], chosen: list[tuple[int, int, dict]], total_score: float) -> None:
            if idx >= len(det_order):
                assignment = {
                    int(det_id): {
                        "object_id": int(obj_id),
                        "candidate": candidate,
                        "score_known": float(self.candidate_known_score(candidate)),
                        "score_final": float(self.candidate_score_final(candidate)),
                    }
                    for det_id, obj_id, candidate in (chosen or [])
                }
                assignments.append({"score": float(total_score), "assignment": assignment})
                return

            det_id = int(det_order[idx])
            options = sorted(
                edges_by_det_obj.get(int(det_id), {}).items(),
                key=lambda item: float(self.candidate_known_score(item[1])),
                reverse=True,
            )
            for obj_id, candidate in options:
                if int(obj_id) in used_obj_ids:
                    continue
                chosen.append((int(det_id), int(obj_id), candidate))
                backtrack(
                    idx + 1,
                    used_obj_ids | {int(obj_id)},
                    chosen,
                    float(total_score) + float(self.candidate_known_score(candidate)),
                )
                chosen.pop()

        backtrack(0, set(), [], 0.0)
        if len(assignments) < 2:
            return {"is_ambiguous": False, "reason": "NO_SECOND_ASSIGNMENT"}

        assignments.sort(key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)
        best = dict(assignments[0])
        second = dict(assignments[1])
        gap = float(max(0.0, float(best.get("score", 0.0) or 0.0) - float(second.get("score", 0.0) or 0.0)))
        if gap > float(self.identity_stability_assignment_gap_max):
            return {
                "is_ambiguous": False,
                "reason": "GAP_TOO_LARGE",
                "gap": float(gap),
                "best_score": float(best.get("score", 0.0) or 0.0),
                "second_score": float(second.get("score", 0.0) or 0.0),
            }

        best_assignment = dict(best.get("assignment", {}) or {})
        second_assignment = dict(second.get("assignment", {}) or {})
        ambiguous_det_ids = sorted(
            int(det_id)
            for det_id in det_ids
            if int(best_assignment.get(int(det_id), {}).get("object_id", -1))
            != int(second_assignment.get(int(det_id), {}).get("object_id", -1))
        )
        if not ambiguous_det_ids:
            return {
                "is_ambiguous": False,
                "reason": "SAME_ASSIGNMENT",
                "gap": float(gap),
                "best_score": float(best.get("score", 0.0) or 0.0),
                "second_score": float(second.get("score", 0.0) or 0.0),
                "best_assignment": dict(best_assignment),
                "second_assignment": dict(second_assignment),
            }

        current_assignment = {
            int(det_id): int(pack[0])
            for det_id, pack in (match_by_det_id or {}).items()
            if int(det_id) in component_det_ids
        }
        return {
            "is_ambiguous": True,
            "reason": "AMBIGUOUS_ASSIGNMENTS",
            "gap": float(gap),
            "best_score": float(best.get("score", 0.0) or 0.0),
            "second_score": float(second.get("score", 0.0) or 0.0),
            "ambiguous_det_ids": list(ambiguous_det_ids),
            "best_assignment": dict(best_assignment),
            "second_assignment": dict(second_assignment),
            "current_assignment": dict(current_assignment),
        }

    def component_has_identity_instability(
        self,
        *,
        component_det_ids: set[int],
        component_obj_ids: set[int],
        edges_by_det_id: dict[int, list[dict]],
        match_by_det_id: dict[int, tuple[int, float]],
    ) -> bool:
        margin = float(self.identity_stability_alt_margin)
        for det_id in component_det_ids:
            current = match_by_det_id.get(int(det_id), None)
            if current is None:
                continue
            current_obj_id = int(current[0])
            current_candidate = self.find_candidate_for_object(edges_by_det_id.get(int(det_id), []), int(current_obj_id))
            if current_candidate is None or not self.is_fragile_identity_edge(current_candidate):
                continue
            current_score = float(self.candidate_known_score(current_candidate))

            best_alt_score = None
            for candidate in (edges_by_det_id.get(int(det_id), []) or []):
                oid = candidate.get("object_id", None)
                if oid is None or int(oid) == int(current_obj_id) or int(oid) not in component_obj_ids:
                    continue
                score = self.candidate_known_score(candidate)
                if best_alt_score is None or float(score) > float(best_alt_score):
                    best_alt_score = float(score)

            strongest_other_claim = None
            for other_det_id in component_det_ids:
                if int(other_det_id) == int(det_id):
                    continue
                other_candidate = self.find_candidate_for_object(edges_by_det_id.get(int(other_det_id), []), int(current_obj_id))
                if other_candidate is None:
                    continue
                score = self.candidate_known_score(other_candidate)
                if strongest_other_claim is None or float(score) > float(strongest_other_claim):
                    strongest_other_claim = float(score)

            if best_alt_score is not None and float(best_alt_score) >= float(current_score) + float(margin):
                return True
            if strongest_other_claim is not None and float(strongest_other_claim) >= float(current_score) + float(margin):
                return True

        return False

    def resolve_identity_component_greedily(
        self,
        *,
        component_det_ids: set[int],
        component_obj_ids: set[int],
        edges_by_det_id: dict[int, list[dict]],
    ) -> list[tuple[int, int, float]]:
        edges = []
        for det_id in component_det_ids:
            for candidate in (edges_by_det_id.get(int(det_id), []) or []):
                oid = candidate.get("object_id", None)
                if oid is None or int(oid) not in component_obj_ids:
                    continue
                edges.append((int(det_id), int(oid), candidate))

        edges.sort(
            key=lambda item: (
                float(self.candidate_known_score(item[2])),
                float(item[2].get("score_sim", 0.0) or 0.0),
            ),
            reverse=True,
        )

        greedy = []
        used_det_ids: set[int] = set()
        used_obj_ids: set[int] = set()
        for det_id, obj_id, candidate in edges:
            if int(det_id) in used_det_ids or int(obj_id) in used_obj_ids:
                continue
            greedy.append((int(det_id), int(obj_id), candidate))
            used_det_ids.add(int(det_id))
            used_obj_ids.add(int(obj_id))

        keep_margin = float(self.identity_stability_keep_margin)
        filtered_matches = []
        for det_id, obj_id, candidate in greedy:
            if self.is_hard_identity_edge(candidate):
                filtered_matches.append((int(det_id), int(obj_id), float(self.candidate_score_final(candidate))))
                continue

            best_competing_score = None
            for other_det_id, other_obj_id, other_candidate in edges:
                if int(other_det_id) == int(det_id) and int(other_obj_id) == int(obj_id):
                    continue
                if int(other_det_id) != int(det_id) and int(other_obj_id) != int(obj_id):
                    continue
                score = self.candidate_known_score(other_candidate)
                if best_competing_score is None or float(score) > float(best_competing_score):
                    best_competing_score = float(score)

            if best_competing_score is None or float(self.candidate_known_score(candidate)) >= float(best_competing_score) + float(keep_margin):
                filtered_matches.append((int(det_id), int(obj_id), float(self.candidate_score_final(candidate))))

        return filtered_matches

    def find_candidate_for_object(self, candidates: list[dict], object_id: int) -> dict | None:
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            oid = candidate.get("object_id", None)
            if oid is None:
                continue
            if int(oid) == int(object_id):
                return candidate
        return None

    def candidate_score_final(self, candidate: dict) -> float:
        if not isinstance(candidate, dict):
            return 0.0
        return float(candidate.get("score_final", candidate.get("score_sim", 0.0)) or 0.0)

    def candidate_gate_reason(self, candidate: dict) -> str:
        if not isinstance(candidate, dict):
            return ""
        trace = candidate.get("sets_trace", {}) or {}
        policy = trace.get("policy", {}) or {}
        return str(policy.get("gate_reason", "") or "").upper()

    def is_fragile_identity_edge(self, candidate: dict) -> bool:
        return str(self.candidate_gate_reason(candidate)) in self.identity_stability_fragile_gate_reasons

    def is_hard_identity_edge(self, candidate: dict) -> bool:
        return str(self.candidate_gate_reason(candidate)) == "PASS_MATCH_THR"

    def build_geom_by_object_id(self, decided_matches: list, detections: list) -> dict:
        det_by_id = {}
        for det in detections or []:
            det_id = getattr(det, "detection_id", None)
            if det_id is None:
                continue
            det_by_id[int(det_id)] = det

        out = {}
        for m in decided_matches or []:
            det_id = int(m.get("det_id", -1))
            obj_id = int(m.get("object_id", -1))

            det = det_by_id.get(int(det_id), None)
            if det is None:
                raise RuntimeError(f"Detection {det_id} missing for build_geom_by_object_id.")

            geom = getattr(det, "geom", None)
            if not isinstance(geom, dict):
                raise RuntimeError(f"Detection {det_id} no tiene geom (dict) en det.geom.")

            center = geom.get("center", None)
            area = geom.get("area", None)
            if center is None or area is None or not isinstance(center, (tuple, list)) or len(center) != 2:
                raise RuntimeError(f"Detection {det_id} has invalid geometry: {geom}")

            bbox = getattr(det, "bbox", None)
            mask = getattr(det, "mask", None)
            out[int(obj_id)] = {
                "center": (float(center[0]), float(center[1])),
                "area": float(area),
                "bbox": tuple(float(x) for x in bbox[:4]) if bbox is not None and len(bbox) >= 4 else None,
                "mask": mask,
            }

        return out
