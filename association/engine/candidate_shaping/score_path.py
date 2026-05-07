from __future__ import annotations


class CandidateScoreShaper:
    """
    Fachada semántica para reglas que construyen tablas y restringen filas.

    Agrupa score base/final, veto contextual y dummy score sin alterar la
    implementación real de la policy.
    """

    def __init__(self, score_policy):
        self.score_policy = score_policy

    def report_status(self, report) -> str:
        return self.score_policy.report_status(report)

    def default_sets_trace(self, report) -> dict:
        return self.score_policy.default_sets_trace(report)

    def format_sets_trace_summary(self, trace: dict) -> tuple[str, str, str]:
        return self.score_policy.format_sets_trace_summary(trace)

    def attach_sets_trace_fields(self, candidate: dict, trace: dict) -> None:
        self.score_policy.attach_sets_trace_fields(candidate, trace)

    def resolve_report_confidence(self, report) -> float | None:
        return self.score_policy.resolve_report_confidence(report)

    def resolve_dummy_score(self, report) -> float:
        return self.score_policy.resolve_dummy_score(report)

    def build_score_tables(
        self,
        *,
        det_ids: list[int],
        detections_by_id: dict[int, object],
        reports: dict,
        snapshot_ids: set[int],
        used_obj_ids: set[int],
        use_neighbor_sets: bool,
        ns_ctx: dict | None,
        neighbor_sets_influence,
        match_thr: float,
        min_match_score: float,
        min_score: float | None = None,
        gate_by_match_thr: bool | None = None,
    ):
        return self.score_policy.build_score_tables(
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

    def context_veto_reason(
        self,
        *,
        det_class_id: int,
        object_id: int,
        candidate: dict,
        ns_ctx: dict | None,
        neighbor_sets_influence,
    ) -> str:
        return self.score_policy.candidate_context_veto_reason(
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
        return self.score_policy.candidate_vetoed_by_context(
            det_class_id=det_class_id,
            object_id=object_id,
            candidate=candidate,
            ns_ctx=ns_ctx,
            neighbor_sets_influence=neighbor_sets_influence,
        )
