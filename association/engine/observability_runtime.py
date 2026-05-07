from __future__ import annotations

from association.reports import FrameAssociationOutput


class DataAssociationObservabilityRuntime:
    """Fachada de observabilidad (debug + trazas) desacoplada del pipeline."""

    def __init__(
        self,
        *,
        trace_runtime,
        debug_view_builder,
        debug_enabled: bool,
        trace_enabled: bool | None = None,
    ):
        self.trace_runtime = trace_runtime
        self.debug_view_builder = debug_view_builder
        self.debug_enabled = bool(debug_enabled)
        if trace_enabled is None:
            trace_enabled = bool(getattr(getattr(trace_runtime, "trace_collector", None), "enabled", False))
        self.trace_enabled = bool(trace_enabled)

    def initialize_frame_output(self, out: FrameAssociationOutput) -> dict:
        return self.debug_view_builder.ensure_out_debug_schema(out)

    def start_frame(self, *, frame_context, detections: list) -> None:
        if not self.trace_enabled:
            return
        self.trace_runtime.start_trace_frame(
            frame_context=frame_context,
            detections=detections,
        )

    def finish_frame(self, *, frame_context) -> None:
        if not self.trace_enabled:
            return
        self.trace_runtime.finish_trace_frame(frame_context=frame_context)

    def trace_after_similarity_reports(
        self,
        *,
        out: FrameAssociationOutput,
        detections: list | None = None,
    ) -> None:
        if not self.trace_enabled:
            return
        self.trace_runtime.start_trace_classes(
            detections=detections,
            reports_by_det_id=out.reports_by_det_id,
        )
        self.trace_runtime.trace_class_partition(
            detections=detections,
            reports_by_det_id=out.reports_by_det_id,
        )
        self.trace_runtime.trace_visual_build_candidates(out=out)

    def trace_after_reliable_anchor_selection(self, *, out: FrameAssociationOutput) -> None:
        if not self.trace_enabled:
            return
        self.trace_runtime.trace_reliable_visual_anchors(out=out)

    def trace_after_context_activation(self, *, out: FrameAssociationOutput) -> None:
        if not self.trace_enabled:
            return
        self.trace_runtime.trace_neighbor_sets_hypotheses(out=out)
        self.trace_runtime.trace_sets_activation(out=out)

    def trace_after_similarity_diagnosis(self, *, out: FrameAssociationOutput) -> None:
        if not self.trace_enabled:
            return
        self.trace_runtime.trace_visual_report_diagnosis(out=out)

    def trace_after_final_ambiguity(self, *, out: FrameAssociationOutput) -> None:
        if not self.trace_enabled:
            return
        self.trace_runtime.trace_final_ambiguity(out=out)

    def trace_finalize_outcomes(self, *, out: FrameAssociationOutput) -> None:
        if not self.trace_enabled:
            return
        self.trace_runtime.trace_final_outcomes(out=out)
        self.trace_runtime.finish_trace_classes(reports_by_det_id=out.reports_by_det_id)

    def build_debug_view(self, *, out: FrameAssociationOutput, frame_id: int | None) -> None:
        self.debug_view_builder.build(out=out, frame_id=frame_id)

    def build_debug_view_if_enabled(self, *, out: FrameAssociationOutput, frame_context) -> None:
        if not self.debug_enabled:
            return
        frame_id = getattr(frame_context, "frame_id", None) if frame_context is not None else None
        self.build_debug_view(
            out=out,
            frame_id=None if frame_id is None else int(frame_id),
        )
