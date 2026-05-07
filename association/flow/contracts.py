from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AssociationFrameRequest:
    detections: list = field(default_factory=list)
    det_features_by_id: dict = field(default_factory=dict)
    frame_context: object | None = None
    timestamp: float = 0.0
