from __future__ import annotations

from association.models import AssignmentResult


class AssignmentExecutor:
    def __init__(
        self,
        *,
        assigner,
        memory_store,
        match_thr: float,
        min_match_score: float,
        neighbor_sets_influence,
    ):
        self.assigner = assigner
        self.memory_store = memory_store
        self.match_thr = float(match_thr)
        self.min_match_score = float(min_match_score)
        self.neighbor_sets_influence = neighbor_sets_influence

    def run(
        self,
        *,
        out,
        detections: list,
        det_features_by_id: dict,
        timer,
    ) -> AssignmentResult:
        with timer.measure("hungarian"):
            snapshot_ids = set(int(obj.object_id) for obj in self.memory_store.all_objects())
            use_neighbor_sets = bool(getattr(self.neighbor_sets_influence, "enabled", False))
            decided_matches, to_create = self.assigner.assign(
                detections=detections,
                det_features_by_id=det_features_by_id,
                reports=out.reports_by_det_id,
                snapshot_ids=snapshot_ids,
                association_output=out,
                use_neighbor_sets=use_neighbor_sets,
                match_thr=float(self.match_thr),
                min_match_score=float(self.min_match_score),
                neighbor_sets_influence=self.neighbor_sets_influence if use_neighbor_sets else None,
                ns_ctx_override=None,
                timer=timer,
                timer_prefix="hungarian/",
            )
        return AssignmentResult(
            decided_matches=list(decided_matches or []),
            to_create=list(to_create or []),
        )
