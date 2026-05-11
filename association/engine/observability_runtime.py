from __future__ import annotations

from association.reports import FrameAssociationOutput


class DataAssociationObservabilityRuntime:
    """Association observability facade."""

    def __init__(
        self,
        *,
        debug_view_builder,
        debug_enabled: bool,
    ):
        self.debug_view_builder = debug_view_builder
        self.debug_enabled = bool(debug_enabled)

    def initialize_frame_output(self, out: FrameAssociationOutput) -> dict:
        return self.debug_view_builder.ensure_out_debug_schema(out)

    def start_frame(self, *, frame_context, detections: list) -> None:
        return None

    def finish_frame(self, *, frame_context) -> None:
        return None

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
