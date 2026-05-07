from __future__ import annotations


class AssignmentResultTraceRuntime:
    """Runtime de traza para AssignmentResultApplier.

    Separa el estado y la lógica de tracing del flujo principal de
    post-asignación para reducir acoplamiento con el pipeline.
    """

    def __init__(self, applier):
        self._applier = applier
        self._trace_frame_id: int | None = None

    def __getattr__(self, name):
        return getattr(self._applier, name)

    def set_trace_frame(self, frame_id: int) -> None:
        self._trace_frame_id = int(frame_id)

    def clear_trace_frame(self) -> None:
        self._trace_frame_id = None

    def trace_scope_key(self, *, class_id: int) -> dict:
        return {
            "frame_id": None if self._trace_frame_id is None else int(self._trace_frame_id),
            "class_id": int(class_id),
        }

    def trace_participants_by_class(self, *, reports_by_det_id: dict) -> dict[int, dict[str, list[int]]]:
        participants_by_class: dict[int, dict[str, set[int]]] = {}
        for det_id, rep in ((reports_by_det_id or {}).items()):
            if rep is None:
                continue
            class_id = int(getattr(rep, "class_id", -1))
            if class_id < 0:
                continue
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            pack["det_ids"].add(int(det_id))
        if self.memory_store is not None:
            for class_id, pack in participants_by_class.items():
                for obj in (self.memory_store.get_by_class(int(class_id)) or []):
                    object_id = getattr(obj, "object_id", None)
                    if object_id is not None:
                        pack["object_ids"].add(int(object_id))
        return {
            int(class_id): {
                "det_ids": sorted(int(x) for x in (pack.get("det_ids", set()) or set())),
                "object_ids": sorted(int(x) for x in (pack.get("object_ids", set()) or set())),
            }
            for class_id, pack in participants_by_class.items()
        }

    def trace_skip_node_for_class(
        self,
        *,
        node_id: str,
        class_id: int,
        reason: str,
        participants: dict | None,
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return
        self.trace_collector.skip_node(
            str(node_id),
            self.trace_scope_key(class_id=int(class_id)),
            str(reason),
            participants=participants or {},
        )

    def trace_identity_stability(
        self,
        *,
        reports_by_det_id: dict,
        initial_matches: list[tuple[int, int, float]],
        final_matches: list[tuple[int, int, float]],
        final_creates: list[dict],
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        trace_participants = self.trace_participants_by_class(reports_by_det_id=reports_by_det_id)
        initial_by_det = {int(det_id): (int(obj_id), float(score_final)) for det_id, obj_id, score_final in (initial_matches or [])}
        final_by_det = {int(det_id): (int(obj_id), float(score_final)) for det_id, obj_id, score_final in (final_matches or [])}
        create_by_det = {
            int(item.get("det_id", -1)): dict(item)
            for item in (final_creates or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }

        det_ids_by_class: dict[int, list[int]] = {}
        for det_id in initial_by_det.keys():
            rep = (reports_by_det_id or {}).get(int(det_id), None)
            if rep is None:
                continue
            class_id = int(getattr(rep, "class_id", -1))
            if class_id < 0:
                continue
            det_ids_by_class.setdefault(int(class_id), []).append(int(det_id))

        for class_id, participants in sorted(trace_participants.items()):
            class_det_ids = list(det_ids_by_class.get(int(class_id), []) or [])
            if not class_det_ids:
                self.trace_skip_node_for_class(
                    node_id="post.identity_stability",
                    class_id=int(class_id),
                    reason="NO_INITIAL_MATCHES",
                    participants=participants,
                )
                continue
            scope_key = self.trace_scope_key(class_id=int(class_id))
            node_participants = {
                "det_ids": sorted(int(det_id) for det_id in class_det_ids),
                "object_ids": sorted(
                    {
                        int(obj_id)
                        for det_id in class_det_ids
                        for obj_id in [
                            initial_by_det.get(int(det_id), (-1, 0.0))[0],
                            final_by_det.get(int(det_id), (-1, 0.0))[0],
                        ]
                        if int(obj_id) >= 0
                    }
                ),
            }
            self.trace_collector.enter_node(
                "post.identity_stability",
                scope_key,
                participants=node_participants,
            )

            kept_count = 0
            remapped_count = 0
            diverted_count = 0
            for det_id in sorted(int(det_id) for det_id in class_det_ids):
                initial_obj_id, initial_score_final = initial_by_det.get(int(det_id), (-1, 0.0))
                final_match = final_by_det.get(int(det_id), None)
                create_entry = create_by_det.get(int(det_id), None)

                if final_match is not None:
                    final_obj_id, final_score_final = final_match
                    if int(final_obj_id) == int(initial_obj_id):
                        state = "kept"
                        reason = "IDENTITY_STABILITY_KEEP"
                        kept_count += 1
                    else:
                        state = "remapped"
                        reason = "IDENTITY_STABILITY_REMAP"
                        remapped_count += 1
                else:
                    final_obj_id = None
                    final_score_final = None
                    state = "diverted_to_create"
                    reason = str((create_entry or {}).get("origin_reason", "") or "IDENTITY_STABILITY_DIVERTED")
                    diverted_count += 1

                self.trace_collector.add_detection_row(
                    "post.identity_stability",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "initial_object_id": int(initial_obj_id),
                        "final_object_id": None if final_obj_id is None else int(final_obj_id),
                        "initial_score_final": float(initial_score_final),
                        "final_score_final": None if final_score_final is None else float(final_score_final),
                        "state": str(state),
                        "origin_mode": None if create_entry is None else str(create_entry.get("origin_mode", "") or ""),
                        "checks": [
                            {
                                "id": "identity_stability.kept_match",
                                "label": "match_kept_after_identity_stability",
                                "lhs": int(1 if state == "kept" else 0),
                                "op": "==",
                                "rhs": 1,
                                "passed": bool(state == "kept"),
                                "reason": str(reason),
                                "effect": "keep_match" if state == "kept" else ("remap_match" if state == "remapped" else "divert_to_create"),
                            }
                        ],
                    },
                )

            self.trace_collector.set_values(
                "post.identity_stability",
                scope_key,
                {
                    "class_id": int(class_id),
                    "initial_match_count": int(len(class_det_ids)),
                    "final_match_count": int(sum(1 for det_id in class_det_ids if int(det_id) in final_by_det)),
                    "create_count": int(sum(1 for det_id in class_det_ids if int(det_id) in create_by_det)),
                    "kept_count": int(kept_count),
                    "remapped_count": int(remapped_count),
                    "diverted_count": int(diverted_count),
                },
            )
            self.trace_collector.set_decision(
                "post.identity_stability",
                scope_key,
                {
                    "status": "PASS",
                    "branch": "evaluated",
                },
            )
            self.trace_collector.leave_node("post.identity_stability", scope_key)

    def trace_assignment_ambiguity(
        self,
        *,
        reports_by_det_id: dict,
        match_by_det_id: dict[int, tuple[int, float]],
        component_rows_by_class: dict[int, list[dict]] | None,
        enabled: bool,
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        trace_participants = self.trace_participants_by_class(reports_by_det_id=reports_by_det_id)
        matched_det_ids_by_class: dict[int, list[int]] = {}
        matched_obj_ids_by_class: dict[int, set[int]] = {}
        for det_id, pack in ((match_by_det_id or {}).items()):
            rep = (reports_by_det_id or {}).get(int(det_id), None)
            if rep is None:
                continue
            class_id = int(getattr(rep, "class_id", -1))
            if class_id < 0:
                continue
            matched_det_ids_by_class.setdefault(int(class_id), []).append(int(det_id))
            matched_obj_ids_by_class.setdefault(int(class_id), set()).add(int(pack[0]))

        for class_id, participants in sorted(trace_participants.items()):
            class_det_ids = list(matched_det_ids_by_class.get(int(class_id), []) or [])
            if not class_det_ids:
                self.trace_skip_node_for_class(
                    node_id="post.assignment_ambiguity",
                    class_id=int(class_id),
                    reason="NO_MATCHES_TO_COMPARE",
                    participants=participants,
                )
                continue
            scope_key = self.trace_scope_key(class_id=int(class_id))
            node_participants = {
                "det_ids": sorted(int(det_id) for det_id in class_det_ids),
                "object_ids": sorted(int(obj_id) for obj_id in (matched_obj_ids_by_class.get(int(class_id), set()) or set())),
            }
            self.trace_collector.enter_node(
                "post.assignment_ambiguity",
                scope_key,
                participants=node_participants,
            )

            component_rows = list((component_rows_by_class or {}).get(int(class_id), []) or [])
            ambiguous_component_count = 0
            for row in component_rows:
                is_ambiguous = bool(row.get("is_ambiguous", False))
                if is_ambiguous:
                    ambiguous_component_count += 1
                gap = float(row.get("gap", 0.0) or 0.0)
                ambiguous_det_ids = [int(x) for x in (row.get("ambiguous_det_ids", []) or [])]
                reason = str(row.get("reason", "") or ("AMBIGUOUS_ASSIGNMENTS" if is_ambiguous else "NOT_AMBIGUOUS"))
                payload = dict(row)
                payload["checks"] = [
                    {
                        "id": "assignment_ambiguity.gap",
                        "label": "assignment_gap_within_threshold",
                        "lhs": float(gap),
                        "op": "<=",
                        "rhs": float(self.identity_stability_assignment_gap_max),
                        "passed": bool(gap <= float(self.identity_stability_assignment_gap_max)),
                        "reason": str(reason),
                        "effect": "keep_component_ambiguous" if is_ambiguous else "component_not_ambiguous",
                    },
                    {
                        "id": "assignment_ambiguity.swap",
                        "label": "assignments_disagree_for_some_detections",
                        "lhs": int(len(ambiguous_det_ids)),
                        "op": ">",
                        "rhs": 0,
                        "passed": bool(len(ambiguous_det_ids) > 0),
                        "reason": str(reason),
                        "effect": "ambiguous_detections_present" if ambiguous_det_ids else "same_assignment",
                    },
                ]
                self.trace_collector.add_global_row(
                    "post.assignment_ambiguity",
                    scope_key,
                    payload,
                )

            self.trace_collector.set_values(
                "post.assignment_ambiguity",
                scope_key,
                {
                    "class_id": int(class_id),
                    "matched_det_count": int(len(class_det_ids)),
                    "matched_object_count": int(len(matched_obj_ids_by_class.get(int(class_id), set()) or set())),
                    "component_count": int(len(component_rows)),
                    "ambiguous_component_count": int(ambiguous_component_count),
                },
            )
            if not enabled:
                decision = {
                    "status": "N/A",
                    "branch": "policy_disabled",
                }
            elif not component_rows:
                decision = {
                    "status": "N/A",
                    "branch": "no_components_to_compare",
                }
            else:
                decision = {
                    "status": "PASS",
                    "branch": "ambiguous_components_found" if ambiguous_component_count else "no_ambiguous_components",
                }
            self.trace_collector.set_decision(
                "post.assignment_ambiguity",
                scope_key,
                decision,
            )
            self.trace_collector.leave_node("post.assignment_ambiguity", scope_key)

    def trace_create_competition(
        self,
        *,
        reports_by_det_id: dict,
        decided_matches: list[tuple[int, int, float]],
        create_entries: list[dict],
        competition_rows_by_class: dict[int, list[dict]] | None,
        enabled: bool,
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        trace_participants = self.trace_participants_by_class(reports_by_det_id=reports_by_det_id)
        class_ids: set[int] = set()
        participants_by_class: dict[int, dict[str, set[int]]] = {}

        for det_id, obj_id, _ in (decided_matches or []):
            rep = (reports_by_det_id or {}).get(int(det_id), None)
            if rep is None:
                continue
            class_id = int(getattr(rep, "class_id", -1))
            if class_id < 0:
                continue
            class_ids.add(int(class_id))
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            pack["det_ids"].add(int(det_id))
            pack["object_ids"].add(int(obj_id))

        for item in (create_entries or []):
            if not isinstance(item, dict):
                continue
            class_id = int(item.get("class_id", -1))
            det_id = int(item.get("det_id", -1))
            if class_id < 0 or det_id < 0:
                continue
            class_ids.add(int(class_id))
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            pack["det_ids"].add(int(det_id))

        for class_id, base_participants in sorted(trace_participants.items()):
            if int(class_id) not in class_ids:
                self.trace_skip_node_for_class(
                    node_id="post.create_competition",
                    class_id=int(class_id),
                    reason="NO_CREATES_OR_MATCHES",
                    participants=base_participants,
                )
                continue
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            scope_key = self.trace_scope_key(class_id=int(class_id))
            node_participants = {
                "det_ids": sorted(int(x) for x in pack["det_ids"]),
                "object_ids": sorted(int(x) for x in pack["object_ids"]),
            }
            self.trace_collector.enter_node(
                "post.create_competition",
                scope_key,
                participants=node_participants,
            )

            class_rows = list((competition_rows_by_class or {}).get(int(class_id), []) or [])
            for row in class_rows:
                parent_score = float(row.get("parent_score", 0.0) or 0.0)
                create_score = float(row.get("create_score", 0.0) or 0.0)
                score_gap = float(abs(parent_score - create_score))
                payload = dict(row)
                payload["checks"] = [
                    {
                        "id": "create_competition.score_gap",
                        "label": "parent_create_gap_within_threshold",
                        "lhs": float(score_gap),
                        "op": "<=",
                        "rhs": float(row.get("pair_gap_max", 0.0) or 0.0),
                        "passed": bool(score_gap <= float(row.get("pair_gap_max", 0.0) or 0.0)),
                        "reason": str(row.get("reason", "") or "COMPETITION_EVALUATED"),
                        "effect": "keep_competition" if bool(row.get("selected", False)) else "no_competition",
                    }
                ]
                self.trace_collector.add_global_row(
                    "post.create_competition",
                    scope_key,
                    payload,
                )

            self.trace_collector.set_values(
                "post.create_competition",
                scope_key,
                {
                    "class_id": int(class_id),
                    "create_entry_count": int(
                        sum(
                            1
                            for item in (create_entries or [])
                            if isinstance(item, dict) and int(item.get("class_id", -1)) == int(class_id)
                        )
                    ),
                    "competition_count": int(len(class_rows)),
                    "selected_competition_count": int(sum(1 for row in class_rows if bool(row.get("selected", False)))),
                },
            )
            if not enabled:
                decision = {
                    "status": "N/A",
                    "branch": "policy_disabled",
                }
            elif not class_rows:
                decision = {
                    "status": "N/A",
                    "branch": "no_competitions",
                }
            else:
                decision = {
                    "status": "PASS",
                    "branch": "competitions_found",
                }
            self.trace_collector.set_decision(
                "post.create_competition",
                scope_key,
                decision,
            )
            self.trace_collector.leave_node("post.create_competition", scope_key)

    def trace_ambiguous_track_candidates(
        self,
        *,
        reports_by_det_id: dict,
        trace_rows: list[dict],
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        trace_participants = self.trace_participants_by_class(reports_by_det_id=reports_by_det_id)
        rows_by_class: dict[int, list[dict]] = {}
        participants_by_class: dict[int, dict[str, set[int]]] = {}

        for item in (trace_rows or []):
            if not isinstance(item, dict):
                continue
            class_id = int(item.get("class_id", -1))
            det_id = int(item.get("det_id", -1))
            if class_id < 0 or det_id < 0:
                continue
            rows_by_class.setdefault(int(class_id), []).append(dict(item))
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            pack["det_ids"].add(int(det_id))
            for object_id in (item.get("candidate_ids", []) or []):
                if object_id is not None:
                    pack["object_ids"].add(int(object_id))
            for object_id in (item.get("committed_new_parent_ids", []) or []):
                if object_id is not None:
                    pack["object_ids"].add(int(object_id))
            committed_new_object_id = item.get("committed_new_object_id", None)
            if committed_new_object_id is not None:
                pack["object_ids"].add(int(committed_new_object_id))

        for class_id, base_participants in sorted(trace_participants.items()):
            class_rows = list(rows_by_class.get(int(class_id), []) or [])
            if not class_rows:
                self.trace_skip_node_for_class(
                    node_id="post.ambiguous_track_candidates",
                    class_id=int(class_id),
                    reason="NO_AMBIGUOUS_CANDIDATES",
                    participants=base_participants,
                )
                continue

            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            scope_key = self.trace_scope_key(class_id=int(class_id))
            node_participants = {
                "det_ids": sorted(int(x) for x in pack["det_ids"]),
                "object_ids": sorted(int(x) for x in pack["object_ids"]),
            }
            self.trace_collector.enter_node(
                "post.ambiguous_track_candidates",
                scope_key,
                participants=node_participants,
            )

            policy_count = 0
            identity_count = 0
            committed_new_count = 0
            for row in sorted(class_rows, key=lambda item: int(item.get("det_id", -1))):
                if bool(row.get("from_policy", False)):
                    policy_count += 1
                if bool(row.get("from_identity_stability", False)):
                    identity_count += 1
                if bool(row.get("from_committed_new", False)):
                    committed_new_count += 1
                candidate_ids = [int(x) for x in (row.get("candidate_ids", []) or []) if x is not None]
                source_count = int(
                    sum(
                        1
                        for flag in (
                            row.get("from_policy", False),
                            row.get("from_identity_stability", False),
                            row.get("from_committed_new", False),
                        )
                        if bool(flag)
                    )
                )
                self.trace_collector.add_detection_row(
                    "post.ambiguous_track_candidates",
                    scope_key,
                    {
                        "det_id": int(row.get("det_id", -1)),
                        "class_id": int(row.get("class_id", -1)),
                        "from_policy": bool(row.get("from_policy", False)),
                        "from_identity_stability": bool(row.get("from_identity_stability", False)),
                        "from_committed_new": bool(row.get("from_committed_new", False)),
                        "selected_source": str(row.get("selected_source", "") or ""),
                        "candidate_ids": list(candidate_ids),
                        "candidate_scores": {
                            int(k): float(v)
                            for k, v in ((row.get("candidate_scores", {}) or {}).items())
                            if k is not None and v is not None
                        },
                        "candidate_count": int(len(candidate_ids)),
                        "best_score": float(row.get("best_score", 0.0) or 0.0),
                        "score_gap": float(row.get("score_gap", 0.0) or 0.0),
                        "reason": str(row.get("reason", "") or ""),
                        "committed_new_object_id": row.get("committed_new_object_id", None),
                        "committed_new_parent_ids": [
                            int(x) for x in (row.get("committed_new_parent_ids", []) or []) if x is not None
                        ],
                        "checks": [
                            {
                                "id": "ambiguous_track.source_present",
                                "label": "candidate_has_ambiguous_source",
                                "lhs": int(source_count),
                                "op": ">=",
                                "rhs": 1,
                                "passed": bool(source_count >= 1),
                                "reason": str(row.get("reason", "") or ""),
                                "effect": "keep_candidate" if source_count >= 1 else "drop_candidate",
                            },
                            {
                                "id": "ambiguous_track.min_candidate_count",
                                "label": "candidate_has_multiple_alternatives",
                                "lhs": int(len(candidate_ids)),
                                "op": ">=",
                                "rhs": 2,
                                "passed": bool(len(candidate_ids) >= 2),
                                "reason": str(row.get("reason", "") or ""),
                                "effect": "pass_to_temporal_resolution" if len(candidate_ids) >= 2 else "drop_candidate",
                            },
                        ],
                    },
                )

            self.trace_collector.set_values(
                "post.ambiguous_track_candidates",
                scope_key,
                {
                    "class_id": int(class_id),
                    "candidate_count": int(len(class_rows)),
                    "policy_count": int(policy_count),
                    "identity_stability_count": int(identity_count),
                    "committed_new_count": int(committed_new_count),
                },
            )
            self.trace_collector.set_decision(
                "post.ambiguous_track_candidates",
                scope_key,
                {
                    "status": "PASS",
                    "branch": "candidates_built",
                },
            )
            self.trace_collector.leave_node("post.ambiguous_track_candidates", scope_key)

    def trace_known_set_distance_disambiguation(
        self,
        *,
        reports_by_det_id: dict,
        known_set_debug: dict,
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        trace_participants = self.trace_participants_by_class(reports_by_det_id=reports_by_det_id)
        components = [dict(item) for item in ((known_set_debug or {}).get("components", []) or []) if isinstance(item, dict)]
        pair_anchors = [dict(item) for item in ((known_set_debug or {}).get("pair_anchors", []) or []) if isinstance(item, dict)]
        passes = [dict(item) for item in ((known_set_debug or {}).get("passes", []) or []) if isinstance(item, dict)]
        components_by_class: dict[int, list[dict]] = {}
        pair_anchors_by_class: dict[int, list[dict]] = {}
        passes_by_class: dict[int, list[dict]] = {}
        participants_by_class: dict[int, dict[str, set[int]]] = {}

        for component in components:
            det_ids = [int(x) for x in (component.get("det_ids", []) or [])]
            if not det_ids:
                continue
            rep = (reports_by_det_id or {}).get(int(det_ids[0]), None)
            if rep is None:
                continue
            class_id = int(getattr(rep, "class_id", -1))
            if class_id < 0:
                continue
            components_by_class.setdefault(int(class_id), []).append(dict(component))
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            for det_id in det_ids:
                pack["det_ids"].add(int(det_id))
            for object_id in (component.get("candidate_union", []) or []):
                if object_id is None:
                    continue
                pack["object_ids"].add(int(object_id))

        for anchor_pack in pair_anchors:
            det_ids = [int(x) for x in (anchor_pack.get("det_pair", []) or []) if x is not None]
            if not det_ids:
                continue
            rep = (reports_by_det_id or {}).get(int(det_ids[0]), None)
            if rep is None:
                continue
            class_id = int(getattr(rep, "class_id", -1))
            if class_id < 0:
                continue
            pair_anchors_by_class.setdefault(int(class_id), []).append(dict(anchor_pack))
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            for det_id in det_ids:
                pack["det_ids"].add(int(det_id))
            for object_id in (anchor_pack.get("anchor_pair", []) or []):
                if object_id is not None:
                    pack["object_ids"].add(int(object_id))

        det_class_by_id = {
            int(det_id): int(getattr(rep, "class_id", -1))
            for det_id, rep in ((reports_by_det_id or {}).items())
            if rep is not None and int(getattr(rep, "class_id", -1)) >= 0
        }
        for pass_info in passes:
            input_det_ids = [
                int(x) for x in (pass_info.get("input_det_ids", []) or [])
                if x is not None and int(x) in det_class_by_id
            ]
            resolved_det_ids = [
                int(x) for x in (pass_info.get("resolved_det_ids", []) or [])
                if x is not None and int(x) in det_class_by_id
            ]
            remaining_det_ids = [
                int(x) for x in (pass_info.get("remaining_det_ids", []) or [])
                if x is not None and int(x) in det_class_by_id
            ]
            class_ids = {
                int(det_class_by_id[int(det_id)])
                for det_id in (input_det_ids + resolved_det_ids + remaining_det_ids)
                if int(det_id) in det_class_by_id
            }
            if not class_ids:
                continue
            for class_id in sorted(class_ids):
                class_input_det_ids = [
                    int(det_id) for det_id in input_det_ids
                    if int(det_class_by_id.get(int(det_id), -1)) == int(class_id)
                ]
                class_resolved_det_ids = [
                    int(det_id) for det_id in resolved_det_ids
                    if int(det_class_by_id.get(int(det_id), -1)) == int(class_id)
                ]
                class_remaining_det_ids = [
                    int(det_id) for det_id in remaining_det_ids
                    if int(det_class_by_id.get(int(det_id), -1)) == int(class_id)
                ]
                if not class_input_det_ids and not class_resolved_det_ids and not class_remaining_det_ids:
                    continue
                passes_by_class.setdefault(int(class_id), []).append(
                    {
                        "pass_index": int(pass_info.get("pass_index", 0) or 0),
                        "input_det_ids": list(class_input_det_ids),
                        "resolved_det_ids": list(class_resolved_det_ids),
                        "remaining_det_ids": list(class_remaining_det_ids),
                        "component_count": int(
                            sum(
                                1
                                for component in components_by_class.get(int(class_id), [])
                                if int(component.get("pass_index", 0) or 0) == int(pass_info.get("pass_index", 0) or 0)
                            )
                        ),
                        "pair_anchor_count": int(
                            sum(
                                1
                                for anchor_pack in pair_anchors_by_class.get(int(class_id), [])
                                if int(anchor_pack.get("pass_index", 0) or 0) == int(pass_info.get("pass_index", 0) or 0)
                            )
                        ),
                    }
                )

        for class_id, base_participants in sorted(trace_participants.items()):
            class_components = list(components_by_class.get(int(class_id), []) or [])
            class_pair_anchors = list(pair_anchors_by_class.get(int(class_id), []) or [])
            class_passes = list(passes_by_class.get(int(class_id), []) or [])
            if not class_components:
                self.trace_skip_node_for_class(
                    node_id="post.known_set_distance_disambiguation",
                    class_id=int(class_id),
                    reason="NO_DISAMBIGUATION_COMPONENTS",
                    participants=base_participants,
                )
                continue
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            scope_key = self.trace_scope_key(class_id=int(class_id))
            node_participants = {
                "det_ids": sorted(int(x) for x in pack["det_ids"]),
                "object_ids": sorted(int(x) for x in pack["object_ids"]),
            }
            self.trace_collector.enter_node(
                "post.known_set_distance_disambiguation",
                scope_key,
                participants=node_participants,
            )

            resolved_count = 0
            partial_count = 0
            for component in class_components:
                status = str(component.get("status", "") or "")
                reason = str(component.get("reason", "") or status or "NO_REASON")
                det_ids = [int(x) for x in (component.get("det_ids", []) or [])]
                policy_mode = str(component.get("policy_mode", "") or "")
                core_score = component.get("core_score", None)
                core_gap = component.get("core_gap", None)
                min_core_score, min_core_gap = self.known_set_disambiguator.policy_core_thresholds(
                    policy_mode=policy_mode,
                    det_count=len(det_ids),
                )
                if status in {"resolved", "partial_resolved"}:
                    resolved_count += 1
                if status == "partial_resolved":
                    partial_count += 1

                checks = []
                evidence = component.get("evidence", None)
                if evidence is not None:
                    checks.append(
                        {
                            "id": "known_set_distance.evidence",
                            "label": "evidence_minimum",
                            "lhs": float(evidence),
                            "op": ">=",
                            "rhs": float(self.known_set_disambiguator.min_total_evidence),
                            "passed": bool(float(evidence) >= float(self.known_set_disambiguator.min_total_evidence)),
                            "reason": str(reason),
                            "effect": "continue"
                            if float(evidence) >= float(self.known_set_disambiguator.min_total_evidence)
                            else "keep_ambiguous",
                        }
                    )
                best_score = component.get("best_score", None)
                if best_score is not None:
                    checks.append(
                        {
                            "id": "known_set_distance.best_score",
                            "label": "assignment_score_minimum",
                            "lhs": float(best_score),
                            "op": ">=",
                            "rhs": float(self.known_set_disambiguator.min_assignment_score),
                            "passed": bool(
                                float(best_score) >= float(self.known_set_disambiguator.min_assignment_score)
                            ),
                            "reason": str(reason),
                            "effect": "continue"
                            if float(best_score) >= float(self.known_set_disambiguator.min_assignment_score)
                            else "keep_ambiguous",
                        }
                    )
                if core_score is not None:
                    checks.append(
                        {
                            "id": "known_set_distance.core_score",
                            "label": "core_assignment_score_minimum",
                            "lhs": float(core_score),
                            "op": ">=",
                            "rhs": float(min_core_score),
                            "passed": bool(float(core_score) >= float(min_core_score)),
                            "reason": str(reason),
                            "effect": "continue" if float(core_score) >= float(min_core_score) else "keep_ambiguous",
                        }
                    )
                if core_gap is not None:
                    checks.append(
                        {
                            "id": "known_set_distance.core_gap",
                            "label": "core_gap_minimum",
                            "lhs": float(core_gap),
                            "op": ">=",
                            "rhs": float(min_core_gap),
                            "passed": bool(float(core_gap) >= float(min_core_gap)),
                            "reason": str(reason),
                            "effect": "continue" if float(core_gap) >= float(min_core_gap) else "keep_ambiguous",
                        }
                    )
                gap = component.get("gap", None)
                if gap is not None:
                    checks.append(
                        {
                            "id": "known_set_distance.gap",
                            "label": "assignment_gap_minimum",
                            "lhs": float(gap),
                            "op": ">=",
                            "rhs": float(self.known_set_disambiguator.min_gap),
                            "passed": bool(float(gap) >= float(self.known_set_disambiguator.min_gap)),
                            "reason": str(reason),
                            "effect": "resolve_component"
                            if float(gap) >= float(self.known_set_disambiguator.min_gap)
                            else "partial_or_keep",
                        }
                    )

                payload = dict(component)
                payload["row_type"] = "component"
                payload["reason"] = str(reason)
                payload["checks"] = checks
                self.trace_collector.add_global_row(
                    "post.known_set_distance_disambiguation",
                    scope_key,
                    payload,
                )

            for anchor_pack in class_pair_anchors:
                payload = dict(anchor_pack)
                payload["row_type"] = "pair_anchor"
                self.trace_collector.add_global_row(
                    "post.known_set_distance_disambiguation",
                    scope_key,
                    payload,
                )

            for pass_info in class_passes:
                payload = dict(pass_info)
                payload["row_type"] = "pass_summary"
                self.trace_collector.add_global_row(
                    "post.known_set_distance_disambiguation",
                    scope_key,
                    payload,
                )

            self.trace_collector.set_values(
                "post.known_set_distance_disambiguation",
                scope_key,
                {
                    "class_id": int(class_id),
                    "component_count": int(len(class_components)),
                    "pair_anchor_count": int(len(class_pair_anchors)),
                    "pass_count": int(len(passes)),
                    "resolved_component_count": int(resolved_count),
                    "partial_component_count": int(partial_count),
                },
            )
            self.trace_collector.set_decision(
                "post.known_set_distance_disambiguation",
                scope_key,
                {
                    "status": "PASS",
                    "branch": "components_evaluated",
                },
            )
            self.trace_collector.leave_node("post.known_set_distance_disambiguation", scope_key)

    def trace_provisional_reconciliation(
        self,
        *,
        reports_by_det_id: dict,
        provisional_entries: list[dict],
        postcreate_debug_entries: list[dict],
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        trace_participants = self.trace_participants_by_class(reports_by_det_id=reports_by_det_id)
        provisional_by_det_id = {
            int(item.get("det_id", -1)): dict(item)
            for item in (provisional_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        debug_rows_by_class: dict[int, list[dict]] = {}
        participants_by_class: dict[int, dict[str, set[int]]] = {}

        for item in (postcreate_debug_entries or []):
            if not isinstance(item, dict):
                continue
            det_id = int(item.get("det_id", -1))
            class_id = int(item.get("class_id", -1))
            if det_id < 0 or class_id < 0:
                continue
            debug_rows_by_class.setdefault(int(class_id), []).append(dict(item))
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            pack["det_ids"].add(int(det_id))
            for object_id in (item.get("support_known_ids", []) or []):
                if object_id is not None:
                    pack["object_ids"].add(int(object_id))
            for object_id in (item.get("blocked_known_ids", []) or []):
                if object_id is not None:
                    pack["object_ids"].add(int(object_id))
            for object_id in (item.get("related_known_ids", []) or []):
                if object_id is not None:
                    pack["object_ids"].add(int(object_id))

        for class_id, base_participants in sorted(trace_participants.items()):
            debug_rows = list(debug_rows_by_class.get(int(class_id), []) or [])
            if not debug_rows:
                self.trace_skip_node_for_class(
                    node_id="post.provisional_reconciliation",
                    class_id=int(class_id),
                    reason="NO_POSTCREATE_DEBUG",
                    participants=base_participants,
                )
                continue
            pack = participants_by_class.setdefault(int(class_id), {"det_ids": set(), "object_ids": set()})
            scope_key = self.trace_scope_key(class_id=int(class_id))
            node_participants = {
                "det_ids": sorted(int(x) for x in pack["det_ids"]),
                "object_ids": sorted(int(x) for x in pack["object_ids"]),
            }
            self.trace_collector.enter_node(
                "post.provisional_reconciliation",
                scope_key,
                participants=node_participants,
            )

            provisional_count = 0
            promoted_ambiguous_count = 0
            for row in debug_rows:
                det_id = int(row.get("det_id", -1))
                decision_kind = str(row.get("decision_kind", "") or "SKIP")
                skip_reason = str(row.get("skip_reason", "") or "")
                decision_reason = str(row.get("decision_reason", "") or "")
                reason = str(decision_reason or skip_reason or decision_kind)
                provisional = provisional_by_det_id.get(int(det_id), None)
                final_kind = None if provisional is None else str(provisional.get("kind", "") or provisional.get("decision_kind", "") or "PROVISIONAL")
                if decision_kind in {"PROV", "PROV_PARENT"}:
                    provisional_count += 1
                if decision_kind == "AMBIG":
                    promoted_ambiguous_count += 1

                self.trace_collector.add_detection_row(
                    "post.provisional_reconciliation",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "temporal_status": str(row.get("temporal_status", "") or ""),
                        "decision_kind": str(decision_kind),
                        "final_kind": final_kind,
                        "reason": str(reason),
                        "focus_source": str(row.get("focus_source", "") or "none"),
                        "context_mode": str(row.get("context_mode", "") or "none"),
                        "support_mode": str(row.get("support_mode", "") or "none"),
                        "relation": str(row.get("relation", "") or "none"),
                        "has_known_context": int(bool(row.get("has_known_context", 0))),
                        "visual_fallback_ok": int(bool(row.get("visual_fallback_ok", 0))),
                        "known_blocked_ok": int(bool(row.get("known_blocked_ok", 0))),
                        "status_not_allowed": int(bool(row.get("status_not_allowed", 0))),
                        "provisional_parent_status_ok": int(bool(row.get("provisional_parent_status_ok", 0))),
                        "provisional_parent_ok": int(bool(row.get("provisional_parent_ok", 0))),
                        "support_known_ids": [int(x) for x in (row.get("support_known_ids", []) or []) if x is not None],
                        "support_known_scores": {
                            int(k): float(v)
                            for k, v in ((row.get("support_known_scores", {}) or {}).items())
                            if k is not None and v is not None
                        },
                        "blocked_known_ids": [int(x) for x in (row.get("blocked_known_ids", []) or []) if x is not None],
                        "blocked_known_scores": {
                            int(k): float(v)
                            for k, v in ((row.get("blocked_known_scores", {}) or {}).items())
                            if k is not None and v is not None
                        },
                        "related_known_ids": [int(x) for x in (row.get("related_known_ids", []) or []) if x is not None],
                        "related_known_scores": {
                            int(k): float(v)
                            for k, v in ((row.get("related_known_scores", {}) or {}).items())
                            if k is not None and v is not None
                        },
                        "best_object_id": row.get("best_object_id", None),
                        "best_score": row.get("best_score", None),
                        "top_supported_object_id": row.get("top_supported_object_id", None),
                        "top_supported_score": row.get("top_supported_score", None),
                        "candidate_rows": list(row.get("candidate_rows", []) or []),
                        "checks": [
                            {
                                "id": "provisional_reconciliation.context",
                                "label": "known_context_or_visual_fallback",
                                "lhs": int(bool(row.get("has_known_context", 0) or row.get("visual_fallback_ok", 0))),
                                "op": "==",
                                "rhs": 1,
                                "passed": bool(row.get("has_known_context", 0) or row.get("visual_fallback_ok", 0)),
                                "reason": str(reason),
                                "effect": "continue" if bool(row.get("has_known_context", 0) or row.get("visual_fallback_ok", 0)) else "skip_temporal_resolution",
                            },
                            {
                                "id": "provisional_reconciliation.status",
                                "label": "status_allowed_or_exception",
                                "lhs": int(not bool(row.get("status_not_allowed", 0))),
                                "op": "==",
                                "rhs": 1,
                                "passed": bool(not row.get("status_not_allowed", 0)),
                                "reason": str(reason),
                                "effect": "allow_provisional" if not row.get("status_not_allowed", 0) else "force_skip_or_parent_exception",
                            },
                        ],
                    },
                )

            self.trace_collector.set_values(
                "post.provisional_reconciliation",
                scope_key,
                {
                    "class_id": int(class_id),
                    "debug_entry_count": int(len(debug_rows)),
                    "provisional_count": int(provisional_count),
                    "promoted_ambiguous_count": int(promoted_ambiguous_count),
                },
            )
            self.trace_collector.set_decision(
                "post.provisional_reconciliation",
                scope_key,
                {
                    "status": "PASS" if debug_rows else "N/A",
                    "branch": "evaluated" if debug_rows else "no_postcreate_debug",
                },
            )
            self.trace_collector.leave_node("post.provisional_reconciliation", scope_key)

    def trace_final_decision_pack(
        self,
        *,
        reports_by_det_id: dict,
        decided_matches: list[tuple[int, int, float]],
        create_entries: list[dict],
        ambiguous_entries: list[dict],
        provisional_entries: list[dict],
        final_pack: dict,
    ) -> None:
        if self.trace_collector is None or self._trace_frame_id is None:
            return

        trace_participants = self.trace_participants_by_class(reports_by_det_id=reports_by_det_id)
        input_match_by_det_id = {
            int(det_id): {"object_id": int(obj_id), "score_final": float(score_final)}
            for det_id, obj_id, score_final in (decided_matches or [])
        }
        input_create_by_det_id = {
            int(item.get("det_id", -1)): dict(item)
            for item in (create_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        input_ambiguous_by_det_id = {
            int(item.get("det_id", -1)): dict(item)
            for item in (ambiguous_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        input_provisional_by_det_id = {
            int(item.get("det_id", -1)): dict(item)
            for item in (provisional_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }

        final_match_by_det_id = {
            int(det_id): {"object_id": int(obj_id), "score_final": float(score_final)}
            for det_id, obj_id, score_final in ((final_pack or {}).get("matches", []) or [])
        }
        final_create_by_det_id = {
            int(item.get("det_id", -1)): dict(item)
            for item in ((final_pack or {}).get("create_entries", []) or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        final_ambiguous_by_det_id = {
            int(item.get("det_id", -1)): dict(item)
            for item in ((final_pack or {}).get("ambiguous_entries", []) or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        final_provisional_by_det_id = {
            int(item.get("det_id", -1)): dict(item)
            for item in ((final_pack or {}).get("provisional_entries", []) or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }

        for class_id, participants in sorted(trace_participants.items()):
            det_ids = list(participants.get("det_ids", []) or [])
            if not det_ids:
                self.trace_skip_node_for_class(
                    node_id="post.final_decision_pack",
                    class_id=int(class_id),
                    reason="NO_DETECTIONS_IN_CLASS",
                    participants=participants,
                )
                continue

            scope_key = self.trace_scope_key(class_id=int(class_id))
            self.trace_collector.enter_node(
                "post.final_decision_pack",
                scope_key,
                participants=participants,
            )

            input_match_count = 0
            input_create_count = 0
            input_ambiguous_count = 0
            input_provisional_count = 0
            final_match_count = 0
            final_create_count = 0
            final_ambiguous_count = 0
            final_provisional_count = 0

            for det_id in sorted(int(x) for x in det_ids):
                input_match = input_match_by_det_id.get(int(det_id), None)
                input_create = input_create_by_det_id.get(int(det_id), None)
                input_ambiguous = input_ambiguous_by_det_id.get(int(det_id), None)
                input_provisional = input_provisional_by_det_id.get(int(det_id), None)
                final_match = final_match_by_det_id.get(int(det_id), None)
                final_create = final_create_by_det_id.get(int(det_id), None)
                final_ambiguous = final_ambiguous_by_det_id.get(int(det_id), None)
                final_provisional = final_provisional_by_det_id.get(int(det_id), None)

                if input_match is not None:
                    input_match_count += 1
                if input_create is not None:
                    input_create_count += 1
                if input_ambiguous is not None:
                    input_ambiguous_count += 1
                if input_provisional is not None:
                    input_provisional_count += 1
                if final_match is not None:
                    final_match_count += 1
                if final_create is not None:
                    final_create_count += 1
                if final_ambiguous is not None:
                    final_ambiguous_count += 1
                if final_provisional is not None:
                    final_provisional_count += 1

                final_bucket = "UNUSED"
                if final_match is not None:
                    final_bucket = "MATCH"
                elif final_create is not None:
                    final_bucket = "CREATE"
                elif final_ambiguous is not None:
                    final_bucket = "AMBIGUOUS"
                elif final_provisional is not None:
                    final_bucket = "PROVISIONAL"

                blocked_match = bool(input_match is not None and final_match is None)
                blocked_create = bool(input_create is not None and final_create is None)
                reason = ""
                if final_ambiguous is not None:
                    reason = str(final_ambiguous.get("reason", "") or "AMBIGUOUS_PRIORITY")
                elif final_provisional is not None:
                    reason = str(final_provisional.get("reason", "") or "PROVISIONAL_PRIORITY")
                elif final_create is not None:
                    reason = str(final_create.get("origin_reason", "") or "CREATE_SURVIVES")
                elif final_match is not None:
                    reason = "MATCH_SURVIVES"
                elif blocked_match:
                    reason = "MATCH_BLOCKED_BY_HIGHER_PRIORITY"
                elif blocked_create:
                    reason = "CREATE_BLOCKED_BY_HIGHER_PRIORITY"
                elif input_ambiguous is not None:
                    reason = str(input_ambiguous.get("reason", "") or "AMBIGUOUS_INPUT")
                elif input_provisional is not None:
                    reason = str(input_provisional.get("reason", "") or "PROVISIONAL_INPUT")
                else:
                    reason = "NO_FINAL_BUCKET"

                self.trace_collector.add_detection_row(
                    "post.final_decision_pack",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "input_match": bool(input_match is not None),
                        "input_create": bool(input_create is not None),
                        "input_ambiguous": bool(input_ambiguous is not None),
                        "input_provisional": bool(input_provisional is not None),
                        "blocked_match": bool(blocked_match),
                        "blocked_create": bool(blocked_create),
                        "final_bucket": str(final_bucket),
                        "final_object_id": None if final_match is None else int(final_match["object_id"]),
                        "final_score": None if final_match is None else float(final_match["score_final"]),
                        "reason": str(reason),
                        "checks": [
                            {
                                "id": "final_pack.bucket",
                                "label": "detection_has_final_bucket",
                                "lhs": str(final_bucket),
                                "op": "!=",
                                "rhs": "UNUSED",
                                "passed": bool(final_bucket != "UNUSED"),
                                "reason": str(reason),
                                "effect": "materialize_final_bucket" if final_bucket != "UNUSED" else "drop_from_final_pack",
                            },
                            {
                                "id": "final_pack.match_priority",
                                "label": "match_survives_after_ambiguous_or_provisional",
                                "lhs": int(0 if blocked_match else 1),
                                "op": "==",
                                "rhs": 1,
                                "passed": bool(not blocked_match),
                                "reason": str(reason),
                                "effect": "keep_match" if not blocked_match else "remove_match",
                            },
                        ],
                    },
                )

            self.trace_collector.set_values(
                "post.final_decision_pack",
                scope_key,
                {
                    "class_id": int(class_id),
                    "input_match_count": int(input_match_count),
                    "input_create_count": int(input_create_count),
                    "input_ambiguous_count": int(input_ambiguous_count),
                    "input_provisional_count": int(input_provisional_count),
                    "final_match_count": int(final_match_count),
                    "final_create_count": int(final_create_count),
                    "final_ambiguous_count": int(final_ambiguous_count),
                    "final_provisional_count": int(final_provisional_count),
                },
            )
            self.trace_collector.set_decision(
                "post.final_decision_pack",
                scope_key,
                {
                    "status": "PASS",
                    "branch": "packed",
                },
            )
            self.trace_collector.leave_node("post.final_decision_pack", scope_key)

