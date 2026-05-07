from __future__ import annotations

from association.flow.contracts import AssociationFrameRequest
from association.reports import FrameAssociationOutput
from utils.time import ExecutionTimer


class FrameAssociationFlow:
    """
    Orquesta la asociación siguiendo un camino de decisión explícito.

    La intención de esta capa es hacer visible el orden conceptual del
    pipeline sin alterar las reglas ni los componentes ya existentes.
    """

    def __init__(self, engine):
        self.engine = engine

    def run(self, request: AssociationFrameRequest) -> FrameAssociationOutput:
        out, timer = self.prepare_frame(request)

        detections = list(request.detections or [])
        if not detections:
            return self.engine.finalize_empty_frame(
                out=out,
                timer=timer,
                frame_context=request.frame_context,
            )

        self.build_visual_evidence(
            out=out,
            detections=detections,
            det_features_by_id=request.det_features_by_id,
            frame_context=request.frame_context,
            timer=timer,
        )
        self.select_reliable_visual_anchors(out=out, timer=timer)

        runtime_flags = self.engine.resolve_runtime_flags()
        self.activate_context_layers(
            out=out,
            detections=detections,
            timestamp=float(request.timestamp),
            runtime_flags=runtime_flags,
            timer=timer,
        )
        self.diagnose_reports(out=out, timer=timer)

        assignment = self.resolve_global_assignment(
            out=out,
            detections=detections,
            det_features_by_id=request.det_features_by_id,
            timer=timer,
        )
        self.apply_post_assignment_guards(
            out=out,
            detections=detections,
            assignment=assignment,
            timer=timer,
        )
        self.finalize_outcomes(
            out=out,
            frame_context=request.frame_context,
            timer=timer,
        )
        return out

    def prepare_frame(
        self,
        request: AssociationFrameRequest,
    ) -> tuple[FrameAssociationOutput, ExecutionTimer]:
        self.engine.reset_runtime_caches()
        out = FrameAssociationOutput(timestamp=float(request.timestamp))
        timer = ExecutionTimer()
        self.engine.observability.initialize_frame_output(out)
        self.engine.init_frame_summary(out, request.detections)
        self.engine.observability.start_frame(
            frame_context=request.frame_context,
            detections=request.detections,
        )
        return out, timer

    def build_visual_evidence(
        self,
        *,
        out: FrameAssociationOutput,
        detections: list,
        det_features_by_id: dict,
        frame_context,
        timer: ExecutionTimer,
    ) -> None:
        self.engine.compute_similarity_reports(
            out=out,
            detections=detections,
            det_features_by_id=det_features_by_id,
            frame_context=frame_context,
            timer=timer,
        )

    def select_reliable_visual_anchors(
        self,
        *,
        out: FrameAssociationOutput,
        timer: ExecutionTimer,
    ) -> None:
        anchor_pairs = timer.run("reliable_anchor_ids", self.engine.build_reliable_anchor_pairs, out)
        out.reliable_anchor_det_by_object_id = {
            int(object_id): int(det_id)
            for object_id, det_id in ((anchor_pairs or {}).items())
        }
        out.reliable_anchor_object_ids = sorted(list(out.reliable_anchor_det_by_object_id.keys()))
        out.frame_summary["n_reliable_anchors"] = int(len(out.reliable_anchor_object_ids))
        self.engine.observability.trace_after_reliable_anchor_selection(out=out)

    def activate_context_layers(
        self,
        *,
        out: FrameAssociationOutput,
        detections: list,
        timestamp: float,
        runtime_flags: dict,
        timer: ExecutionTimer,
    ) -> None:
        self.engine.compute_neighbor_sets_if_needed(
            out=out,
            detections=detections,
            timestamp=timestamp,
            runtime=runtime_flags,
            timer=timer,
        )
        self.engine.observability.trace_after_context_activation(out=out)

    def diagnose_reports(
        self,
        *,
        out: FrameAssociationOutput,
        timer: ExecutionTimer,
    ) -> None:
        self.engine.annotate_similarity_ambiguity(out=out, timer=timer)

    def resolve_global_assignment(
        self,
        *,
        out: FrameAssociationOutput,
        detections: list,
        det_features_by_id: dict,
        timer: ExecutionTimer,
    ):
        return self.engine.run_assignment(
            out=out,
            detections=detections,
            det_features_by_id=det_features_by_id,
            timer=timer,
        )

    def apply_post_assignment_guards(
        self,
        *,
        out: FrameAssociationOutput,
        detections: list,
        assignment,
        timer: ExecutionTimer,
    ) -> None:
        self.engine.apply_assignment_results(
            out=out,
            detections=detections,
            decided_matches=assignment.decided_matches,
            to_create=assignment.to_create,
            timer=timer,
        )

    def finalize_outcomes(
        self,
        *,
        out: FrameAssociationOutput,
        frame_context,
        timer: ExecutionTimer,
    ) -> None:
        self.update_frame_summary_after_assignment(out=out)

        self.engine.annotate_final_ambiguity(out=out, timer=timer)
        self.engine.observability.trace_finalize_outcomes(out=out)
        self.engine.observability.build_debug_view_if_enabled(
            out=out,
            frame_context=frame_context,
        )

        out.timings_seconds = timer.snapshot_seconds()
        self.engine.observability.finish_frame(frame_context=frame_context)

    def update_frame_summary_after_assignment(self, *, out: FrameAssociationOutput) -> None:
        out.frame_summary["n_matches"] = int(len(out.decided_matches))
        out.frame_summary["n_created"] = int(len(out.to_create))
        out.frame_summary["n_ambiguous_tracks"] = int(len(out.to_ambiguous))
        out.frame_summary["n_provisional_new"] = int(len(out.to_provisional_new))
        out.frame_summary["n_reports"] = int(len(out.reports_by_det_id))
