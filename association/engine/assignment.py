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
    det->obj assignment using Hungarian with dummies, locks, and neighbor-sets influence.

    Score convention:
      - score_sim   : pure similarity (gating + locks)
      - score_assign: stable score for Hungarian; does not include sets_rescue
      - bonus_sets  : signed contextual adjustment from neighbor-sets, when applicable
      - score_final : score_sim + bonus_sets
    """

    def __init__(self, config: dict, memory_store):
        self.config = config or {}
        self.memory_store = memory_store
        self.assignment_path_support = AssignmentPathSupport()

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
            raise RuntimeError("Hungarian requires scipy. Install: pip install scipy") from e

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

        if not valid_det_ids:
            return [], create_from_missing_features

        step = (lambda suffix: f"{timer_prefix}{suffix}" if timer_prefix else str(suffix))
        used_det_ids: set[int] = set()
        used_obj_ids: set[int] = set()

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
        if not table_sim:
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
        remaining_det_ids = [int(did) for did in valid_det_ids if int(did) not in used_det_ids]
        if not remaining_det_ids:
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
