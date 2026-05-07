from __future__ import annotations

from association.reports import FrameAssociationOutput


class DataAssociationTraceRuntime:
    """Runtime de traza de asociación desacoplado del engine principal.

    Mantiene estado por frame/clase de la traza y recibe explícitamente
    sus dependencias de lectura/escritura para evitar acoplamiento implícito
    con el engine completo.
    """

    def __init__(
        self,
        *,
        trace_collector,
        class_id_to_name,
        memory_store,
        assigner,
        assignment_result_applier,
        pick_best,
        pick_second_best,
        confirm_thr_strong: float,
        confirm_clear_margin: float,
        sets_provider,
        scores,
        neighbor_sets_influence,
    ):
        self.trace_collector = trace_collector
        self.class_id_to_name = class_id_to_name
        self.memory_store = memory_store
        self.assigner = assigner
        self.assignment_result_applier = assignment_result_applier
        self.pick_best = pick_best
        self.pick_second_best = pick_second_best
        self.confirm_thr_strong = float(confirm_thr_strong)
        self.confirm_clear_margin = float(confirm_clear_margin)
        self.sets_provider = sets_provider
        self.scores = scores
        self.neighbor_sets_influence = neighbor_sets_influence
        self._trace_frame_id: int | None = None
        self._trace_timestamp: float | None = None
        self._trace_active_class_ids: set[int] = set()

    def reset_trace_state(self) -> None:
        self.assigner.clear_trace_frame()
        self.assignment_result_applier.clear_trace_frame()
        self._trace_frame_id = None
        self._trace_timestamp = None
        self._trace_active_class_ids = set()

    def resolve_trace_class_name(self, class_id: int, detections: list | None = None) -> str | None:
        class_id = int(class_id)
        if self.class_id_to_name is not None:
            class_name = self.class_id_to_name.get(int(class_id), None)
            if class_name:
                return str(class_name)

        for det in (detections or []):
            if int(getattr(det, "class_id", -1)) != int(class_id):
                continue
            class_name = getattr(det, "class_name", None)
            if class_name:
                return str(class_name)

        class_objects = list(self.memory_store.get_by_class(int(class_id)) or [])
        if class_objects:
            class_name = getattr(class_objects[0], "class_name", None)
            if class_name:
                return str(class_name)
        return None

    def start_trace_frame(self, *, frame_context, detections: list) -> None:
        if frame_context is None:
            self.reset_trace_state()
            return
        frame_id = getattr(frame_context, "frame_id", None)
        timestamp = getattr(frame_context, "timestamp", None)
        if frame_id is None or timestamp is None:
            self.reset_trace_state()
            return
        det_ids = [
            int(det_id)
            for det in (detections or [])
            for det_id in [getattr(det, "detection_id", None)]
            if det_id is not None
        ]
        self.trace_collector.start_frame(
            frame_id=int(frame_id),
            timestamp=float(timestamp),
            det_ids=det_ids,
        )
        self.assigner.set_trace_frame(int(frame_id))
        self.assignment_result_applier.set_trace_frame(int(frame_id))
        self._trace_frame_id = int(frame_id)
        self._trace_timestamp = float(timestamp)
        self._trace_active_class_ids = set()

    def finish_trace_frame(self, *, frame_context) -> None:
        if frame_context is None:
            self.reset_trace_state()
            return
        frame_id = getattr(frame_context, "frame_id", None)
        if frame_id is None:
            self.reset_trace_state()
            return
        self.trace_collector.finish_frame(int(frame_id))
        self.reset_trace_state()

    def trace_scope_key(self, *, class_id: int) -> dict:
        return {
            "frame_id": None if self._trace_frame_id is None else int(self._trace_frame_id),
            "class_id": int(class_id),
        }

    def build_trace_class_specs(
        self,
        *,
        detections: list | None = None,
        reports_by_det_id: dict | None = None,
    ) -> dict[int, dict]:
        specs: dict[int, dict] = {}

        for det in (detections or []):
            det_id = getattr(det, "detection_id", None)
            class_id = getattr(det, "class_id", None)
            if det_id is None or class_id is None:
                continue
            class_id = int(class_id)
            spec = specs.setdefault(
                int(class_id),
                {
                    "class_id": int(class_id),
                    "class_name": None,
                    "det_ids": set(),
                    "snapshot_object_ids": set(),
                },
            )
            spec["det_ids"].add(int(det_id))
            if not spec["class_name"]:
                spec["class_name"] = self.resolve_trace_class_name(int(class_id), detections=detections)

        for det_id, rep in ((reports_by_det_id or {}).items()):
            if rep is None:
                continue
            class_id = int(getattr(rep, "class_id", -1))
            if class_id < 0:
                continue
            spec = specs.setdefault(
                int(class_id),
                {
                    "class_id": int(class_id),
                    "class_name": None,
                    "det_ids": set(),
                    "snapshot_object_ids": set(),
                },
            )
            spec["det_ids"].add(int(det_id))

        for class_id, spec in specs.items():
            snapshot_ids = {
                int(getattr(obj, "object_id", -1))
                for obj in (self.memory_store.get_by_class(int(class_id)) or [])
                if getattr(obj, "object_id", None) is not None
            }
            spec["snapshot_object_ids"] = set(snapshot_ids)
            if not spec["class_name"]:
                spec["class_name"] = self.resolve_trace_class_name(int(class_id), detections=detections)
            spec["det_ids"] = sorted(int(x) for x in spec["det_ids"])
            spec["snapshot_object_ids"] = sorted(int(x) for x in spec["snapshot_object_ids"])

        return {int(cid): dict(spec) for cid, spec in specs.items()}

    def start_trace_classes(self, *, detections: list | None = None, reports_by_det_id: dict | None = None) -> None:
        if self._trace_frame_id is None:
            return
        specs = self.build_trace_class_specs(detections=detections, reports_by_det_id=reports_by_det_id)
        for class_id, spec in sorted(specs.items()):
            self.trace_collector.start_class(
                frame_id=int(self._trace_frame_id),
                class_id=int(class_id),
                class_name=spec.get("class_name"),
                det_ids=list(spec.get("det_ids", []) or []),
                snapshot_object_ids=list(spec.get("snapshot_object_ids", []) or []),
            )
            self._trace_active_class_ids.add(int(class_id))

    def finish_trace_classes(self, *, reports_by_det_id: dict | None = None) -> None:
        del reports_by_det_id
        if self._trace_frame_id is None:
            return
        for class_id in sorted(int(x) for x in (self._trace_active_class_ids or set())):
            self.trace_collector.finish_class(int(self._trace_frame_id), int(class_id))

    def trace_reliable_visual_anchors(self, *, out: FrameAssociationOutput) -> None:
        specs = self.build_trace_class_specs(reports_by_det_id=out.reports_by_det_id)
        global_anchor_pairs = sorted(
            [
                {
                    "object_id": int(object_id),
                    "det_id": int(det_id),
                }
                for object_id, det_id in ((out.reliable_anchor_det_by_object_id or {}).items())
            ],
            key=lambda row: (int(row["det_id"]), int(row["object_id"])),
        )
        global_anchor_ids = sorted(int(row["object_id"]) for row in global_anchor_pairs)
        for class_id, spec in sorted(specs.items()):
            scope_key = self.trace_scope_key(class_id=int(class_id))
            participants = {
                "det_ids": list(spec.get("det_ids", []) or []),
                "object_ids": list(spec.get("snapshot_object_ids", []) or []),
            }
            anchor_ids = sorted(
                int(object_id)
                for object_id, det_id in ((out.reliable_anchor_det_by_object_id or {}).items())
                if int(getattr((out.reports_by_det_id or {}).get(int(det_id), None), "class_id", -1)) == int(class_id)
            )
            anchor_pairs = sorted(
                [
                    {
                        "object_id": int(object_id),
                        "det_id": int(det_id),
                    }
                    for object_id, det_id in ((out.reliable_anchor_det_by_object_id or {}).items())
                    if int(getattr((out.reports_by_det_id or {}).get(int(det_id), None), "class_id", -1)) == int(class_id)
                ],
                key=lambda row: (int(row["det_id"]), int(row["object_id"])),
            )
            self.trace_collector.enter_node(
                "prepare.reliable_visual_anchors",
                scope_key,
                participants=participants,
            )
            for det_id in list(spec.get("det_ids", []) or []):
                rep = (out.reports_by_det_id or {}).get(int(det_id), None)
                candidates = list(getattr(rep, "candidates", None) or []) if rep is not None else []
                best = self.pick_best(candidates, key="score_sim") if candidates else None
                second = self.pick_second_best(candidates, best, key="score_sim") if best is not None else None

                best_object_id = None if best is None else int(best.get("object_id", -1))
                second_object_id = None if second is None else int(second.get("object_id", -1))
                best_score = 0.0 if best is None else float(best.get("score_sim", 0.0) or 0.0)
                second_score = 0.0 if second is None else float(second.get("score_sim", 0.0) or 0.0)
                gap = float(best_score - second_score) if best is not None else 0.0
                passes_score = bool(best is not None and best_score >= float(self.confirm_thr_strong))
                passes_margin = bool(best is not None and gap >= float(self.confirm_clear_margin))
                selected_anchor_det_id = None
                selected_as_anchor = False
                if best_object_id is not None:
                    owner_det_id = (out.reliable_anchor_det_by_object_id or {}).get(int(best_object_id), None)
                    selected_anchor_det_id = None if owner_det_id is None else int(owner_det_id)
                    selected_as_anchor = bool(selected_anchor_det_id == int(det_id))

                if not candidates:
                    reason = "NO_VISUAL_CANDIDATES"
                elif not passes_score:
                    reason = "BELOW_STRONG_THRESHOLD"
                elif not passes_margin:
                    reason = "BELOW_CLEAR_MARGIN"
                elif selected_as_anchor:
                    reason = "RELIABLE_VISUAL_ANCHOR"
                else:
                    reason = "SUPERSEDED_BY_BETTER_DETECTION"

                self.trace_collector.add_detection_row(
                    "prepare.reliable_visual_anchors",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "candidate_count": int(len(candidates)),
                        "best_object_id": best_object_id,
                        "best_score_sim": float(best_score),
                        "second_object_id": second_object_id,
                        "second_score_sim": float(second_score),
                        "gap": float(gap),
                        "score_threshold": float(self.confirm_thr_strong),
                        "margin_threshold": float(self.confirm_clear_margin),
                        "passes_score_threshold": bool(passes_score),
                        "passes_margin_threshold": bool(passes_margin),
                        "selected_as_anchor": bool(selected_as_anchor),
                        "selected_anchor_det_id": selected_anchor_det_id,
                        "reason": str(reason),
                        "checks": [
                            {
                                "id": "reliable_visual_anchor.score_threshold",
                                "label": "best_score_reaches_strong_threshold",
                                "lhs": float(best_score),
                                "op": ">=",
                                "rhs": float(self.confirm_thr_strong),
                                "passed": bool(passes_score),
                                "logic_group": "anchor_selection_gate",
                                "logic_group_label": "compuerta anchor fiable",
                                "logic_op": "AND",
                                "logic_order": 1,
                                "reason": str(reason),
                                "effect": "eligible_by_score" if passes_score else "reject_anchor",
                            },
                            {
                                "id": "reliable_visual_anchor.margin_threshold",
                                "label": "best_second_gap_reaches_clear_margin",
                                "lhs": float(gap),
                                "op": ">=",
                                "rhs": float(self.confirm_clear_margin),
                                "passed": bool(passes_margin),
                                "logic_group": "anchor_selection_gate",
                                "logic_group_label": "compuerta anchor fiable",
                                "logic_op": "AND",
                                "logic_order": 2,
                                "reason": str(reason),
                                "effect": "eligible_by_gap" if passes_margin else "reject_anchor",
                            },
                            {
                                "id": "reliable_visual_anchor.selected",
                                "label": "best_candidate_survives_as_anchor_for_its_object",
                                "lhs": int(1 if selected_as_anchor else 0),
                                "op": "==",
                                "rhs": 1,
                                "passed": bool(selected_as_anchor),
                                "logic_group": "anchor_selection_gate",
                                "logic_group_label": "compuerta anchor fiable",
                                "logic_op": "AND",
                                "logic_order": 3,
                                "reason": str(reason),
                                "effect": "select_anchor" if selected_as_anchor else "do_not_select_anchor",
                            },
                        ],
                    },
                )
            self.trace_collector.add_check(
                "prepare.reliable_visual_anchors",
                scope_key,
                {
                    "id": "reliable_visual_anchors.exists",
                    "label": "has_reliable_visual_anchors",
                    "lhs": int(len(anchor_ids)),
                    "op": ">",
                    "rhs": 0,
                    "passed": bool(anchor_ids),
                    "reason": "RELIABLE_ANCHORS_PRESENT" if anchor_ids else "NO_RELIABLE_ANCHORS",
                    "effect": "anchors_available" if anchor_ids else "anchors_missing",
                },
            )
            self.trace_collector.set_values(
                "prepare.reliable_visual_anchors",
                scope_key,
                {
                    "class_id": int(class_id),
                    "class_name": spec.get("class_name"),
                    "global_anchor_object_ids": list(global_anchor_ids),
                    "global_anchor_pairs": list(global_anchor_pairs),
                    "anchor_object_ids": list(anchor_ids),
                    "anchor_pairs": list(anchor_pairs),
                    "strong_score_threshold": float(self.confirm_thr_strong),
                    "clear_margin_threshold": float(self.confirm_clear_margin),
                    "anchor_count_global": int(len(global_anchor_ids)),
                    "anchor_count": int(len(anchor_ids)),
                },
            )
            self.trace_collector.set_decision(
                "prepare.reliable_visual_anchors",
                scope_key,
                {
                    "status": "PASS" if anchor_ids else "N/A",
                    "branch": "has_reliable_anchors" if anchor_ids else "no_reliable_anchors",
                },
            )
            self.trace_collector.leave_node("prepare.reliable_visual_anchors", scope_key)

    def trace_neighbor_sets_hypotheses(self, *, out: FrameAssociationOutput) -> None:
        specs = self.build_trace_class_specs(reports_by_det_id=out.reports_by_det_id)
        neighbor_sets_out = getattr(out, "neighbor_sets_out", None)
        ns_ctx = self.sets_provider.build_context(neighbor_sets_out)
        core = neighbor_sets_out.get("core", None) if isinstance(neighbor_sets_out, dict) else None
        debug = neighbor_sets_out.get("debug", None) if isinstance(neighbor_sets_out, dict) else None
        meta = debug.get("meta", None) if isinstance(debug, dict) else None
        hypotheses = debug.get("set_hypotheses", None) if isinstance(debug, dict) else None

        if not isinstance(core, dict):
            core = {}
        if not isinstance(debug, dict):
            debug = {}
        if not isinstance(meta, dict):
            meta = {}
        if not isinstance(hypotheses, list):
            hypotheses = []

        computed = bool(isinstance(neighbor_sets_out, dict))
        global_anchors = [
            int(x)
            for x in (core.get("anchors", []) or meta.get("anchors", []) or [])
            if x is not None
        ]
        global_shortlist = sorted(int(x) for x in (core.get("shortlist", []) or []))
        prior_by_oid = {
            int(object_id): float(score)
            for object_id, score in ((core.get("prior_by_oid", {}) or {}).items())
        }
        class_prior_by_cid = {
            int(class_id): float(score)
            for class_id, score in ((core.get("class_prior_by_cid", {}) or {}).items())
        }
        selective_classes = sorted(int(x) for x in (core.get("selective_classes", []) or []))
        n_hypotheses = int(core.get("n_hypotheses", len(hypotheses)) or 0)
        best_score = float(core.get("best_score", meta.get("best_score", 0.0)) or 0.0)
        second_score = float(core.get("second_score", meta.get("second_score", 0.0)) or 0.0)
        gap_best = float(core.get("gap_best", meta.get("gap_best", 0.0)) or 0.0)
        k_best = int(meta.get("k_best", 0) or 0)
        coverage_eff_best = float(meta.get("coverage_eff_best", 0.0) or 0.0)
        density_best = float(meta.get("density_best", 0.0) or 0.0)
        mean_maturity_best = float(core.get("mean_maturity_best", meta.get("mean_maturity_best", 0.0)) or 0.0)
        shortlist_threshold = float(core.get("thr_shortlist", 0.0) or 0.0)
        total_dets = int(meta.get("n_dets", 0) or 0)
        total_classes = int(meta.get("n_classes", 0) or 0)
        neigh_sets = getattr(self.scores, "neigh_sets", None)
        beam_width = int(getattr(neigh_sets, "beam_width", 0) or 0) if neigh_sets is not None else 0
        topk_sets = int(getattr(neigh_sets, "topk_sets", 0) or 0) if neigh_sets is not None else 0
        context_k = int(getattr(neigh_sets, "context_k", 0) or 0) if neigh_sets is not None else 0
        support_sum_by_oid = {
            int(object_id): float(score)
            for object_id, score in (((ns_ctx or {}).get("support_sum_by_oid", {}) or {}).items())
        } if isinstance(ns_ctx, dict) else {}
        class_ctx_by_id = dict((ns_ctx or {}).get("class_ctx", {}) or {}) if isinstance(ns_ctx, dict) else {}
        maturity_fn = getattr(getattr(self.scores, "neigh_sets", None), "object_maturity_score", None)

        for class_id, spec in sorted(specs.items()):
            class_object_ids = {int(x) for x in (spec.get("snapshot_object_ids", []) or [])}
            class_det_ids = {int(x) for x in (spec.get("det_ids", []) or [])}
            class_shortlist = sorted(int(x) for x in global_shortlist if int(x) in class_object_ids)
            class_anchors = sorted(int(x) for x in global_anchors if int(x) in class_object_ids)
            class_prior_by_oid = {
                int(object_id): float(score)
                for object_id, score in prior_by_oid.items()
                if int(object_id) in class_object_ids
            }
            class_support_sum_by_oid = {
                int(object_id): float(score)
                for object_id, score in support_sum_by_oid.items()
                if int(object_id) in class_object_ids
            }
            class_pack = class_ctx_by_id.get(int(class_id), None)
            if not isinstance(class_pack, dict):
                class_pack = {}

            scope_key = self.trace_scope_key(class_id=int(class_id))
            participants = {
                "det_ids": list(spec.get("det_ids", []) or []),
                "object_ids": list(spec.get("snapshot_object_ids", []) or []),
            }

            self.trace_collector.enter_node(
                "context.neighbor_sets_hypotheses",
                scope_key,
                participants=participants,
            )
            self.trace_collector.add_check(
                "context.neighbor_sets_hypotheses",
                scope_key,
                {
                    "id": "sets.hypotheses_computed",
                    "label": "neighbor_sets_output_available",
                    "lhs": bool(computed),
                    "op": "==",
                    "rhs": True,
                    "passed": bool(computed),
                    "reason": "COMPUTED" if computed else "NOT_REQUESTED",
                    "effect": "inspect_hypotheses" if computed else "skip_hypotheses",
                },
            )
            if computed:
                self.trace_collector.add_check(
                    "context.neighbor_sets_hypotheses",
                    scope_key,
                    {
                        "id": "sets.hypotheses_found",
                        "label": "neighbor_sets_found_hypotheses",
                        "lhs": int(n_hypotheses),
                        "op": ">",
                        "rhs": 0,
                        "passed": bool(int(n_hypotheses) > 0),
                        "reason": "OK" if int(n_hypotheses) > 0 else "NO_HYPOTHESES",
                        "effect": "build_relational_context" if int(n_hypotheses) > 0 else "empty_relational_context",
                    },
                )

            self.trace_collector.set_values(
                "context.neighbor_sets_hypotheses",
                scope_key,
                {
                    "class_id": int(class_id),
                    "class_name": spec.get("class_name"),
                    "computed": bool(computed),
                    "n_hypotheses": int(n_hypotheses),
                    "best_score": float(best_score),
                    "second_score": float(second_score),
                    "gap_best": float(gap_best),
                    "k_best": int(k_best),
                    "coverage_eff_best": float(coverage_eff_best),
                    "density_best": float(density_best),
                    "mean_maturity_best": float(mean_maturity_best),
                    "shortlist_threshold": float(shortlist_threshold),
                    "beam_width": int(beam_width),
                    "topk_sets_limit": int(topk_sets),
                    "context_k": int(context_k),
                    "retained_hypotheses": int(n_hypotheses),
                    "shortlist_size": int(len(class_shortlist)),
                    "anchor_count": int(len(class_anchors)),
                    "prior_count": int(len(class_prior_by_oid)),
                    "shortlist_object_ids": list(class_shortlist),
                    "anchor_object_ids": list(class_anchors),
                    "prior_object_ids": sorted(int(x) for x in class_prior_by_oid.keys()),
                    "class_prior": float(class_prior_by_cid.get(int(class_id), 0.0)),
                    "class_is_selective": bool(int(class_id) in set(selective_classes)),
                    "global_anchor_count": int(len(global_anchors)),
                    "global_shortlist_size": int(len(global_shortlist)),
                    "global_prior_count": int(len(prior_by_oid)),
                    "total_detections_considered": int(total_dets),
                    "total_classes_considered": int(total_classes),
                },
            )

            relevant_rows = []
            for rank, hyp in enumerate(hypotheses, start=1):
                if not isinstance(hyp, dict):
                    continue
                hyp_object_ids = [int(x) for x in (hyp.get("object_ids", []) or []) if x is not None]
                hyp_det_ids = [int(x) for x in (hyp.get("det_ids_explained", []) or []) if x is not None]
                overlap_objects = sorted(int(x) for x in hyp_object_ids if int(x) in class_object_ids)
                overlap_dets = sorted(int(x) for x in hyp_det_ids if int(x) in class_det_ids)
                if not overlap_objects and not overlap_dets:
                    continue
                relevant_rows.append(
                    {
                        "row_type": "hypothesis",
                        "rank": int(rank),
                        "score_sets": float(hyp.get("score_sets", 0.0) or 0.0),
                        "object_ids": list(hyp_object_ids),
                        "det_ids_explained": list(hyp_det_ids),
                        "class_object_overlap": list(overlap_objects),
                        "class_det_overlap": list(overlap_dets),
                        "k": int(hyp.get("k", len(hyp_object_ids)) or len(hyp_object_ids)),
                        "coverage_eff": float(hyp.get("coverage_eff", 0.0) or 0.0),
                        "density": float(hyp.get("density", 0.0) or 0.0),
                        "mean_maturity": float(hyp.get("mean_maturity", 0.0) or 0.0),
                    }
                )

            for row in relevant_rows:
                self.trace_collector.add_global_row(
                    "context.neighbor_sets_hypotheses",
                    scope_key,
                    row,
                )

            for object_id in sorted(int(x) for x in class_object_ids):
                obj = self.memory_store.get(int(object_id))
                neighbors = getattr(obj, "neighbors", None) if obj is not None else None
                maturity_score = float(maturity_fn(int(object_id))) if callable(maturity_fn) else 0.0
                self.trace_collector.add_global_row(
                    "context.neighbor_sets_hypotheses",
                    scope_key,
                    {
                        "row_type": "object_support",
                        "object_id": int(object_id),
                        "prior": float(class_prior_by_oid.get(int(object_id), 0.0)),
                        "support_sum": float(class_support_sum_by_oid.get(int(object_id), 0.0)),
                        "maturity_score": float(maturity_score),
                        "hits": int(getattr(obj, "hits", 0) or 0) if obj is not None else 0,
                        "state": str(getattr(obj, "state", "") or "") if obj is not None else "",
                        "neighbor_episode_count": int(getattr(neighbors, "episode_count", 0) or 0) if neighbors is not None else 0,
                        "shortlist_hit": bool(int(object_id) in set(class_pack.get("shortlist", set()) or set())),
                        "supported_hit": bool(int(object_id) in set(class_pack.get("supported", set()) or set())),
                        "soft_supported_hit": bool(int(object_id) in set(class_pack.get("soft_supported", set()) or set())),
                        "coverage_ok": bool((class_pack.get("coverage_ok_by_oid", {}) or {}).get(int(object_id), False)),
                        "compat_rel": float((class_pack.get("compat_rel_by_oid", {}) or {}).get(int(object_id), 0.0)),
                        "kernel_raw": float((class_pack.get("kernel_raw_by_oid", {}) or {}).get(int(object_id), 0.0)),
                        "kernel_hit_count": int((class_pack.get("kernel_hit_count_by_oid", {}) or {}).get(int(object_id), 0) or 0),
                        "kernel_hit_ratio": float((class_pack.get("kernel_hit_ratio_by_oid", {}) or {}).get(int(object_id), 0.0)),
                        "kernel_rel": float((class_pack.get("kernel_rel_by_oid", {}) or {}).get(int(object_id), 0.0)),
                        "hyp_rel": float((class_pack.get("hyp_rel_by_oid", {}) or {}).get(int(object_id), 0.0)),
                    },
                )

            self.trace_collector.set_decision(
                "context.neighbor_sets_hypotheses",
                scope_key,
                {
                    "status": "PASS" if int(n_hypotheses) > 0 else ("SOFT" if computed else "N/A"),
                    "branch": "hypotheses_ready" if int(n_hypotheses) > 0 else ("empty" if computed else "inactive"),
                },
            )
            self.trace_collector.leave_node("context.neighbor_sets_hypotheses", scope_key)

    def trace_class_partition(self, *, detections: list | None = None, reports_by_det_id: dict | None = None) -> None:
        specs = self.build_trace_class_specs(detections=detections, reports_by_det_id=reports_by_det_id)
        for class_id, spec in sorted(specs.items()):
            scope_key = self.trace_scope_key(class_id=int(class_id))
            participants = {
                "det_ids": list(spec.get("det_ids", []) or []),
                "object_ids": list(spec.get("snapshot_object_ids", []) or []),
            }
            det_ids = list(spec.get("det_ids", []) or [])
            snapshot_object_ids = list(spec.get("snapshot_object_ids", []) or [])
            self.trace_collector.enter_node(
                "prepare.class_partition",
                scope_key,
                participants=participants,
            )
            self.trace_collector.add_check(
                "prepare.class_partition",
                scope_key,
                {
                    "id": "class_partition.has_detections",
                    "label": "class_has_detections",
                    "lhs": int(len(det_ids)),
                    "op": ">",
                    "rhs": 0,
                    "passed": bool(det_ids),
                    "reason": "CLASS_PARTITION_READY" if det_ids else "CLASS_WITHOUT_DETECTIONS",
                    "effect": "continue_class_path" if det_ids else "skip_class_path",
                },
            )
            self.trace_collector.set_values(
                "prepare.class_partition",
                scope_key,
                {
                    "class_id": int(class_id),
                    "class_name": spec.get("class_name"),
                    "detection_count": int(len(det_ids)),
                    "snapshot_object_count": int(len(snapshot_object_ids)),
                    "det_ids": list(det_ids),
                    "snapshot_object_ids": list(snapshot_object_ids),
                },
            )
            self.trace_collector.set_decision(
                "prepare.class_partition",
                scope_key,
                {
                    "status": "PASS" if det_ids else "N/A",
                    "branch": "class_partition_ready" if det_ids else "empty_class_partition",
                },
            )
            self.trace_collector.leave_node("prepare.class_partition", scope_key)

    def trace_visual_report_diagnosis(self, *, out: FrameAssociationOutput) -> None:
        specs = self.build_trace_class_specs(reports_by_det_id=out.reports_by_det_id)
        for class_id, spec in sorted(specs.items()):
            scope_key = self.trace_scope_key(class_id=int(class_id))
            participants = {
                "det_ids": list(spec.get("det_ids", []) or []),
                "object_ids": list(spec.get("snapshot_object_ids", []) or []),
            }
            rows = []
            counts = {"STRONG": 0, "AMBIGUOUS": 0, "WEAK": 0}
            self.trace_collector.enter_node(
                "visual.report_diagnosis",
                scope_key,
                participants=participants,
            )
            for det_id in list(spec.get("det_ids", []) or []):
                rep = (out.reports_by_det_id or {}).get(int(det_id), None)
                diag = getattr(rep, "match_diag_sim", None) or {}
                status = str(diag.get("status", "WEAK") or "WEAK").upper()
                counts[status] = int(counts.get(status, 0)) + 1
                row = {
                    "det_id": int(det_id),
                    "status": str(status),
                    "reason": str(diag.get("reason", "") or ""),
                    "s1": float(diag.get("s1", 0.0) or 0.0),
                    "s2": float(diag.get("s2", 0.0) or 0.0),
                    "gap": float(diag.get("gap", 0.0) or 0.0),
                    "n_close": int(diag.get("n_close", 0) or 0),
                    "confidence": float(diag.get("confidence", 0.0) or 0.0),
                    "checks": [
                        {
                            "id": "diag.strong_min_score",
                            "label": "strong_min_score",
                            "lhs": float(diag.get("s1", 0.0) or 0.0),
                            "op": ">=",
                            "rhs": float(diag.get("strong_min_score", 0.0) or 0.0),
                            "passed": bool(
                                float(diag.get("s1", 0.0) or 0.0) >= float(diag.get("strong_min_score", 0.0) or 0.0)
                            ),
                            "logic_group": "strong_gate",
                            "logic_group_label": "compuerta strong",
                            "logic_op": "AND",
                            "logic_order": 1,
                            "reason": str(diag.get("reason", "") or ""),
                            "effect": "eligible_strong" if str(status) == "STRONG" else "still_not_strong",
                        },
                        {
                            "id": "diag.strong_gap",
                            "label": "strong_gap",
                            "lhs": float(diag.get("gap", 0.0) or 0.0),
                            "op": ">=",
                            "rhs": float(diag.get("strong_gap", 0.0) or 0.0),
                            "passed": bool(
                                float(diag.get("gap", 0.0) or 0.0) >= float(diag.get("strong_gap", 0.0) or 0.0)
                            ),
                            "logic_group": "strong_gate",
                            "logic_group_label": "compuerta strong",
                            "logic_op": "AND",
                            "logic_order": 2,
                            "reason": str(diag.get("reason", "") or ""),
                            "effect": "eligible_strong" if str(status) == "STRONG" else "still_not_strong",
                        },
                        {
                            "id": "diag.strong_unique_close",
                            "label": "strong_unique_close",
                            "lhs": int(diag.get("n_close", 0) or 0),
                            "op": "<=",
                            "rhs": 1,
                            "passed": bool(int(diag.get("n_close", 0) or 0) <= 1),
                            "logic_group": "strong_gate",
                            "logic_group_label": "compuerta strong",
                            "logic_op": "AND",
                            "logic_order": 3,
                            "reason": str(diag.get("reason", "") or ""),
                            "effect": "eligible_strong" if str(status) == "STRONG" else "still_not_strong",
                        },
                        {
                            "id": "diag.ambiguous_min_score",
                            "label": "ambiguous_min_score",
                            "lhs": float(diag.get("s1", 0.0) or 0.0),
                            "op": ">=",
                            "rhs": float(diag.get("ambiguous_min_score", 0.0) or 0.0),
                            "passed": bool(
                                float(diag.get("s1", 0.0) or 0.0) >= float(diag.get("ambiguous_min_score", 0.0) or 0.0)
                            ),
                            "logic_group": "ambiguous_fallback",
                            "logic_group_label": "vía fallback ambiguous",
                            "logic_op": "AND",
                            "logic_order": 1,
                            "reason": str(diag.get("reason", "") or ""),
                            "effect": "eligible_ambiguous" if str(status) == "AMBIGUOUS" else "not_ambiguous_fallback",
                        },
                    ],
                }
                rows.append(row)
                self.trace_collector.add_detection_row(
                    "visual.report_diagnosis",
                    scope_key,
                    row,
                )
            self.trace_collector.set_values(
                "visual.report_diagnosis",
                scope_key,
                {
                    "class_id": int(class_id),
                    "class_name": spec.get("class_name"),
                    "n_strong": int(counts.get("STRONG", 0)),
                    "n_ambiguous": int(counts.get("AMBIGUOUS", 0)),
                    "n_weak": int(counts.get("WEAK", 0)),
                },
            )
            self.trace_collector.set_decision(
                "visual.report_diagnosis",
                scope_key,
                {
                    "status": "PASS",
                    "branch": "diagnosed",
                },
            )
            self.trace_collector.leave_node("visual.report_diagnosis", scope_key)

    def trace_final_ambiguity(self, *, out: FrameAssociationOutput) -> None:
        specs = self.build_trace_class_specs(reports_by_det_id=out.reports_by_det_id)
        for class_id, spec in sorted(specs.items()):
            scope_key = self.trace_scope_key(class_id=int(class_id))
            participants = {
                "det_ids": list(spec.get("det_ids", []) or []),
                "object_ids": list(spec.get("snapshot_object_ids", []) or []),
            }
            counts = {"STRONG": 0, "AMBIGUOUS": 0, "WEAK": 0}
            self.trace_collector.enter_node(
                "outcome.final_ambiguity",
                scope_key,
                participants=participants,
            )
            for det_id in list(spec.get("det_ids", []) or []):
                rep = (out.reports_by_det_id or {}).get(int(det_id), None)
                diag = getattr(rep, "match_diag_final", None) or {}
                status = str(diag.get("status", "WEAK") or "WEAK").upper()
                counts[status] = int(counts.get(status, 0)) + 1
                self.trace_collector.add_detection_row(
                    "outcome.final_ambiguity",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "status": str(status),
                        "reason": str(diag.get("reason", "") or ""),
                        "s1": float(diag.get("s1", 0.0) or 0.0),
                        "s2": float(diag.get("s2", 0.0) or 0.0),
                        "gap": float(diag.get("gap", 0.0) or 0.0),
                        "n_close": int(diag.get("n_close", 0) or 0),
                        "confidence": float(diag.get("confidence", 0.0) or 0.0),
                        "final_decision": str(getattr(rep, "final_decision", "") or ""),
                        "final_reason": str(getattr(rep, "final_reason", "") or ""),
                        "checks": [
                            {
                                "id": "diag_final.strong_min_score",
                                "label": "final_strong_min_score",
                                "lhs": float(diag.get("s1", 0.0) or 0.0),
                                "op": ">=",
                                "rhs": float(diag.get("strong_min_score", 0.0) or 0.0),
                                "passed": bool(
                                    float(diag.get("s1", 0.0) or 0.0) >= float(diag.get("strong_min_score", 0.0) or 0.0)
                                ),
                                "logic_group": "final_strong_gate",
                                "logic_group_label": "compuerta final strong",
                                "logic_op": "AND",
                                "logic_order": 1,
                                "reason": str(diag.get("reason", "") or ""),
                                "effect": "eligible_strong" if str(status) == "STRONG" else "still_not_strong",
                            },
                            {
                                "id": "diag_final.strong_gap",
                                "label": "final_strong_gap",
                                "lhs": float(diag.get("gap", 0.0) or 0.0),
                                "op": ">=",
                                "rhs": float(diag.get("strong_gap", 0.0) or 0.0),
                                "passed": bool(
                                    float(diag.get("gap", 0.0) or 0.0) >= float(diag.get("strong_gap", 0.0) or 0.0)
                                ),
                                "logic_group": "final_strong_gate",
                                "logic_group_label": "compuerta final strong",
                                "logic_op": "AND",
                                "logic_order": 2,
                                "reason": str(diag.get("reason", "") or ""),
                                "effect": "eligible_strong" if str(status) == "STRONG" else "still_not_strong",
                            },
                            {
                                "id": "diag_final.ambiguous_min_score",
                                "label": "final_ambiguous_min_score",
                                "lhs": float(diag.get("s1", 0.0) or 0.0),
                                "op": ">=",
                                "rhs": float(diag.get("ambiguous_min_score", 0.0) or 0.0),
                                "passed": bool(
                                    float(diag.get("s1", 0.0) or 0.0) >= float(diag.get("ambiguous_min_score", 0.0) or 0.0)
                                ),
                                "logic_group": "final_ambiguous_fallback",
                                "logic_group_label": "vía final ambiguous",
                                "logic_op": "AND",
                                "logic_order": 1,
                                "reason": str(diag.get("reason", "") or ""),
                                "effect": "eligible_ambiguous" if str(status) == "AMBIGUOUS" else "not_ambiguous_fallback",
                            },
                        ],
                    },
                )
            self.trace_collector.set_values(
                "outcome.final_ambiguity",
                scope_key,
                {
                    "class_id": int(class_id),
                    "class_name": spec.get("class_name"),
                    "n_strong": int(counts.get("STRONG", 0)),
                    "n_ambiguous": int(counts.get("AMBIGUOUS", 0)),
                    "n_weak": int(counts.get("WEAK", 0)),
                },
            )
            self.trace_collector.set_decision(
                "outcome.final_ambiguity",
                scope_key,
                {
                    "status": "PASS",
                    "branch": "diagnosed",
                },
            )
            self.trace_collector.leave_node("outcome.final_ambiguity", scope_key)

    def trace_visual_build_candidates(self, *, out: FrameAssociationOutput) -> None:
        specs = self.build_trace_class_specs(reports_by_det_id=out.reports_by_det_id)
        for class_id, spec in sorted(specs.items()):
            scope_key = self.trace_scope_key(class_id=int(class_id))
            participants = {
                "det_ids": list(spec.get("det_ids", []) or []),
                "object_ids": list(spec.get("snapshot_object_ids", []) or []),
            }
            total_candidate_count = 0
            self.trace_collector.enter_node(
                "visual.build_candidates",
                scope_key,
                participants=participants,
            )
            for det_id in list(spec.get("det_ids", []) or []):
                rep = (out.reports_by_det_id or {}).get(int(det_id), None)
                candidates = list(getattr(rep, "candidates", None) or []) if rep is not None else []
                ranked_candidates = sorted(
                    [
                        c
                        for c in candidates
                        if isinstance(c, dict) and c.get("object_id", None) is not None
                    ],
                    key=lambda item: (
                        float(item.get("score_sim", 0.0) or 0.0),
                        int(item.get("object_id", -1) or -1),
                    ),
                    reverse=True,
                )
                total_candidate_count += int(len(ranked_candidates))
                best = ranked_candidates[0] if ranked_candidates else None
                second = ranked_candidates[1] if len(ranked_candidates) > 1 else None

                self.trace_collector.add_detection_row(
                    "visual.build_candidates",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "candidate_count": int(len(ranked_candidates)),
                        "best_object_id": None if best is None else int(best.get("object_id", -1)),
                        "best_score_sim": None if best is None else float(best.get("score_sim", 0.0) or 0.0),
                        "second_object_id": None if second is None else int(second.get("object_id", -1)),
                        "second_score_sim": None if second is None else float(second.get("score_sim", 0.0) or 0.0),
                        "checks": [
                            {
                                "id": "build_candidates.has_candidates",
                                "label": "has_similarity_candidates",
                                "lhs": int(len(ranked_candidates)),
                                "op": ">",
                                "rhs": 0,
                                "passed": bool(ranked_candidates),
                                "reason": "CANDIDATES_FOUND" if ranked_candidates else "NO_CANDIDATES",
                                "effect": "keep_detection_in_matching" if ranked_candidates else "no_candidate_path",
                            }
                        ],
                    },
                )

                for rank, candidate in enumerate(ranked_candidates, start=1):
                    self.trace_collector.add_candidate_row(
                        "visual.build_candidates",
                        scope_key,
                        {
                            "det_id": int(det_id),
                            "object_id": int(candidate.get("object_id", -1)),
                            "rank": int(rank),
                            "score_sim": float(candidate.get("score_sim", 0.0) or 0.0),
                            "score_obj": (
                                None if candidate.get("score_obj_collapsed", None) is None
                                else float(candidate.get("score_obj_collapsed", 0.0) or 0.0)
                            ),
                            "score_bg": (
                                None if candidate.get("score_bg_collapsed", None) is None
                                else float(candidate.get("score_bg_collapsed", 0.0) or 0.0)
                            ),
                            "score_bg_partial": (
                                None if candidate.get("score_bgp_collapsed", None) is None
                                else float(candidate.get("score_bgp_collapsed", 0.0) or 0.0)
                            ),
                            "score_parts": (
                                None if candidate.get("score_parts_collapsed", None) is None
                                else float(candidate.get("score_parts_collapsed", 0.0) or 0.0)
                            ),
                            "weight_eff_obj": float(candidate.get("weight_eff_obj", 0.0) or 0.0),
                            "weight_eff_bg": float(candidate.get("weight_eff_bg", 0.0) or 0.0),
                            "weight_eff_bg_partial": float(candidate.get("weight_eff_bgp", 0.0) or 0.0),
                            "weight_eff_parts": float(candidate.get("weight_eff_parts", 0.0) or 0.0),
                            "quality_eff_obj": float(candidate.get("quality_eff_obj", 0.0) or 0.0),
                            "quality_eff_bg": float(candidate.get("quality_eff_bg", 0.0) or 0.0),
                            "quality_eff_parts": float(candidate.get("quality_eff_parts", 0.0) or 0.0),
                            "quality_obj": float(candidate.get("quality_obj", 0.0) or 0.0),
                            "quality_bg": float(candidate.get("quality_bg", 0.0) or 0.0),
                            "quality_parts": float(candidate.get("quality_parts", 0.0) or 0.0),
                            "obj_patch_density": float(
                                (((candidate.get("scores", {}) or {}).get("_quality", {}) or {}).get("meta", {}) or {}).get("obj_patch_density", 0.0) or 0.0
                            ),
                            "parts_support_frac": float(
                                (((candidate.get("scores", {}) or {}).get("_quality", {}) or {}).get("meta", {}) or {}).get("parts_support_frac", 0.0) or 0.0
                            ),
                            "n_parts_valid": int(
                                (((candidate.get("scores", {}) or {}).get("_quality", {}) or {}).get("meta", {}) or {}).get("n_parts_valid", 0) or 0
                            ),
                            "bg_mask_quality": float(
                                (((candidate.get("scores", {}) or {}).get("_quality", {}) or {}).get("meta", {}) or {}).get("bg_mask_quality", 0.0) or 0.0
                            ),
                            "score_source_policy": dict(candidate.get("score_source_policy", {}) or {}),
                        },
                    )

            self.trace_collector.set_values(
                "visual.build_candidates",
                scope_key,
                {
                    "class_id": int(class_id),
                    "class_name": spec.get("class_name"),
                    "detection_count": int(len(spec.get("det_ids", []) or [])),
                    "candidate_count": int(total_candidate_count),
                },
            )
            self.trace_collector.set_decision(
                "visual.build_candidates",
                scope_key,
                {
                    "status": "PASS" if total_candidate_count > 0 else "N/A",
                    "branch": "candidates_built" if total_candidate_count > 0 else "no_candidates",
                },
            )
            self.trace_collector.leave_node("visual.build_candidates", scope_key)

    def trace_sets_activation(self, *, out: FrameAssociationOutput) -> None:
        specs = self.build_trace_class_specs(reports_by_det_id=out.reports_by_det_id)
        ns_ctx = self.sets_provider.build_context(getattr(out, "neighbor_sets_out", None))
        influence = self.neighbor_sets_influence
        enabled = bool(isinstance(ns_ctx, dict) and ns_ctx.get("enabled", False))
        global_ok = bool(isinstance(ns_ctx, dict) and ns_ctx.get("global_ok", False))
        quality = float((ns_ctx or {}).get("quality", 0.0) or 0.0) if isinstance(ns_ctx, dict) else 0.0
        reason = str((ns_ctx or {}).get("reason", "") or ("NO_CONTEXT" if not enabled else "UNKNOWN")) if isinstance(ns_ctx, dict) else "NO_CONTEXT"
        best = float((ns_ctx or {}).get("best", 0.0) or 0.0) if isinstance(ns_ctx, dict) else 0.0
        coverage_eff = float((ns_ctx or {}).get("coverage_eff", 0.0) or 0.0) if isinstance(ns_ctx, dict) else 0.0
        maturity = float((ns_ctx or {}).get("maturity", 0.0) or 0.0) if isinstance(ns_ctx, dict) else 0.0
        density = float((ns_ctx or {}).get("density", 0.0) or 0.0) if isinstance(ns_ctx, dict) else 0.0
        k_best = int((ns_ctx or {}).get("k_best", 0) or 0) if isinstance(ns_ctx, dict) else 0
        n_hypotheses = int((ns_ctx or {}).get("n_hypotheses", 0) or 0) if isinstance(ns_ctx, dict) else 0
        shortlist = sorted(int(x) for x in ((ns_ctx or {}).get("shortlist", set()) or set())) if isinstance(ns_ctx, dict) else []
        anchors = [int(x) for x in ((ns_ctx or {}).get("anchors", []) or [])] if isinstance(ns_ctx, dict) else []
        prior_by_oid = {
            int(object_id): float(score)
            for object_id, score in (((ns_ctx or {}).get("prior_by_oid", {}) or {}).items())
        } if isinstance(ns_ctx, dict) else {}
        quality_terms = dict((ns_ctx or {}).get("quality_terms", {}) or {}) if isinstance(ns_ctx, dict) else {}
        global_shortlist_size = int(len(shortlist))
        global_anchor_count = int(len(anchors))
        global_prior_count = int(len(prior_by_oid))

        for class_id, spec in sorted(specs.items()):
            class_object_ids = {int(x) for x in (spec.get("snapshot_object_ids", []) or [])}
            class_shortlist = sorted(int(x) for x in shortlist if int(x) in class_object_ids)
            class_anchors = sorted(int(x) for x in anchors if int(x) in class_object_ids)
            class_prior_by_oid = {
                int(object_id): float(score)
                for object_id, score in prior_by_oid.items()
                if int(object_id) in class_object_ids
            }
            scope_key = self.trace_scope_key(class_id=int(class_id))
            participants = {
                "det_ids": list(spec.get("det_ids", []) or []),
                "object_ids": list(spec.get("snapshot_object_ids", []) or []),
            }
            self.trace_collector.enter_node(
                "context.sets_activation",
                scope_key,
                participants=participants,
            )
            self.trace_collector.add_check(
                "context.sets_activation",
                scope_key,
                {
                    "id": "sets.enabled",
                    "label": "sets_context_built",
                    "lhs": bool(enabled),
                    "op": "==",
                    "rhs": True,
                    "passed": bool(enabled),
                    "reason": str(reason),
                    "effect": "continue" if enabled else "disable_sets_context",
                },
            )
            if enabled:
                self.trace_collector.add_check(
                    "context.sets_activation",
                    scope_key,
                    {
                        "id": "sets.has_hypotheses",
                        "label": "sets_has_hypotheses",
                        "lhs": int(n_hypotheses),
                        "op": ">",
                        "rhs": 0,
                        "passed": bool(int(n_hypotheses) > 0),
                        "reason": "OK" if int(n_hypotheses) > 0 else "NO_HYPOTHESES",
                        "effect": "continue" if int(n_hypotheses) > 0 else "disable_sets_context",
                    },
                )
                self.trace_collector.add_check(
                    "context.sets_activation",
                    scope_key,
                    {
                        "id": "sets.min_size",
                        "label": "sets_best_group_reaches_min_size",
                        "lhs": int(k_best),
                        "op": ">=",
                        "rhs": int(influence.min_size),
                        "passed": bool(int(k_best) >= int(influence.min_size)),
                        "reason": "OK" if int(k_best) >= int(influence.min_size) else "SMALL_SET",
                        "effect": "continue" if int(k_best) >= int(influence.min_size) else "degrade_sets_context",
                    },
                )
                self.trace_collector.add_check(
                    "context.sets_activation",
                    scope_key,
                    {
                        "id": "sets.min_best_score",
                        "label": "sets_best_score_reaches_threshold",
                        "lhs": float(best),
                        "op": ">=",
                        "rhs": float(influence.min_best_score),
                        "passed": bool(float(best) >= float(influence.min_best_score)),
                        "reason": "OK" if float(best) >= float(influence.min_best_score) else "LOW_SCORE",
                        "effect": "continue" if float(best) >= float(influence.min_best_score) else "degrade_sets_context",
                    },
                )
                self.trace_collector.add_check(
                    "context.sets_activation",
                    scope_key,
                    {
                        "id": "sets.min_coverage_eff",
                        "label": "sets_coverage_reaches_threshold",
                        "lhs": float(coverage_eff),
                        "op": ">=",
                        "rhs": float(influence.min_coverage_eff),
                        "passed": bool(float(coverage_eff) >= float(influence.min_coverage_eff)),
                        "reason": "OK" if float(coverage_eff) >= float(influence.min_coverage_eff) else "LOW_COVERAGE",
                        "effect": "continue" if float(coverage_eff) >= float(influence.min_coverage_eff) else "degrade_sets_context",
                    },
                )
                self.trace_collector.add_check(
                    "context.sets_activation",
                    scope_key,
                    {
                        "id": "sets.min_quality",
                        "label": "sets_quality_reaches_threshold",
                        "lhs": float(quality),
                        "op": ">=",
                        "rhs": float(influence.min_quality),
                        "passed": bool(float(quality) >= float(influence.min_quality)),
                        "reason": "OK" if float(quality) >= float(influence.min_quality) else str(reason),
                        "effect": "enable_sets_context" if float(quality) >= float(influence.min_quality) else "degrade_sets_context",
                    },
                )
                self.trace_collector.add_check(
                    "context.sets_activation",
                    scope_key,
                    {
                        "id": "sets.global_ok",
                        "label": "sets_quality_gate",
                        "lhs": float(quality),
                        "op": "global_ok",
                        "rhs": True,
                        "passed": bool(global_ok),
                        "reason": str(reason),
                        "effect": "enable_sets_context" if global_ok else "degraded_sets_context",
                    },
                )
            self.trace_collector.set_values(
                "context.sets_activation",
                scope_key,
                {
                    "class_id": int(class_id),
                    "enabled": bool(enabled),
                    "global_ok": bool(global_ok),
                    "reason": str(reason),
                    "quality": float(quality),
                    "best": float(best),
                    "coverage_eff": float(coverage_eff),
                    "maturity": float(maturity),
                    "density": float(density),
                    "k_best": int(k_best),
                    "n_hypotheses": int(n_hypotheses),
                    "min_size_threshold": int(influence.min_size),
                    "best_score_threshold": float(influence.min_best_score),
                    "coverage_eff_threshold": float(influence.min_coverage_eff),
                    "quality_threshold": float(influence.min_quality),
                    "shortlist_size": int(len(class_shortlist)),
                    "anchor_count": int(len(class_anchors)),
                    "prior_count": int(len(class_prior_by_oid)),
                    "shortlist_object_ids": list(class_shortlist),
                    "anchor_object_ids": list(class_anchors),
                    "prior_object_ids": sorted(int(x) for x in class_prior_by_oid.keys()),
                    "global_shortlist_size": int(global_shortlist_size),
                    "global_anchor_count": int(global_anchor_count),
                    "global_prior_count": int(global_prior_count),
                    "quality_terms": dict(quality_terms),
                },
            )
            self.trace_collector.set_decision(
                "context.sets_activation",
                scope_key,
                {
                    "status": "PASS" if global_ok else ("SOFT" if enabled else "N/A"),
                    "branch": "active" if global_ok else ("degraded" if enabled else "inactive"),
                },
            )
            self.trace_collector.leave_node("context.sets_activation", scope_key)

    def trace_final_outcomes(self, *, out: FrameAssociationOutput) -> None:
        specs = self.build_trace_class_specs(reports_by_det_id=out.reports_by_det_id)
        match_source_by_det_id = {
            int(item.get("det_id", -1)): str(item.get("source", "") or "association")
            for item in (out.decided_matches or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        for class_id, spec in sorted(specs.items()):
            scope_key = self.trace_scope_key(class_id=int(class_id))
            participants = {
                "det_ids": list(spec.get("det_ids", []) or []),
                "object_ids": list(spec.get("snapshot_object_ids", []) or []),
            }
            summary: dict[str, int] = {}
            self.trace_collector.enter_node(
                "outcome.finalize",
                scope_key,
                participants=participants,
            )
            for det_id in list(spec.get("det_ids", []) or []):
                rep = (out.reports_by_det_id or {}).get(int(det_id), None)
                if rep is None:
                    continue
                final_decision = str(getattr(rep, "final_decision", "UNASSIGNED") or "UNASSIGNED")
                summary[final_decision] = int(summary.get(final_decision, 0)) + 1
                self.trace_collector.add_detection_row(
                    "outcome.finalize",
                    scope_key,
                    {
                        "det_id": int(det_id),
                        "final_decision": str(final_decision),
                        "final_reason": str(getattr(rep, "final_reason", "") or ""),
                        "final_object_id": getattr(rep, "final_object_id", None),
                        "final_score": float(getattr(rep, "final_score", 0.0) or 0.0),
                        "match_source": match_source_by_det_id.get(int(det_id), ""),
                        "ambiguous_candidate_ids": [
                            int(x) for x in (getattr(rep, "ambiguous_candidate_ids", []) or [])
                        ],
                        "ambiguous_candidate_scores": {
                            int(k): float(v)
                            for k, v in ((getattr(rep, "ambiguous_candidate_scores", {}) or {}).items())
                            if k is not None and v is not None
                        },
                        "provisional_support_ids": [
                            int(x) for x in (getattr(rep, "provisional_support_ids", []) or [])
                        ],
                        "provisional_support_scores": {
                            int(k): float(v)
                            for k, v in ((getattr(rep, "provisional_support_scores", {}) or {}).items())
                            if k is not None and v is not None
                        },
                        "provisional_blocked_known_ids": [
                            int(x) for x in (getattr(rep, "provisional_blocked_known_ids", []) or [])
                        ],
                        "provisional_blocked_known_scores": {
                            int(k): float(v)
                            for k, v in ((getattr(rep, "provisional_blocked_known_scores", {}) or {}).items())
                            if k is not None and v is not None
                        },
                        "provisional_related_known_ids": [
                            int(x) for x in (getattr(rep, "provisional_related_known_ids", []) or [])
                        ],
                        "provisional_related_known_scores": {
                            int(k): float(v)
                            for k, v in ((getattr(rep, "provisional_related_known_scores", {}) or {}).items())
                            if k is not None and v is not None
                        },
                    },
                )
            self.trace_collector.set_values(
                "outcome.finalize",
                scope_key,
                {
                    "class_id": int(class_id),
                    "class_name": spec.get("class_name"),
                    "decision_counts": {str(k): int(v) for k, v in sorted(summary.items())},
                },
            )
            self.trace_collector.set_decision(
                "outcome.finalize",
                scope_key,
                {
                    "status": "PASS",
                    "branch": "finalized",
                },
            )
            self.trace_collector.leave_node("outcome.finalize", scope_key)
