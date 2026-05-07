from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AssignmentPartition:
    detections_by_id: dict[int, object] = field(default_factory=dict)
    det_ids_by_class: dict[int, list[int]] = field(default_factory=dict)


@dataclass(slots=True)
class PreparedAssignmentInputs:
    snapshot_ids: set[int] = field(default_factory=set)
    context: object | None = None
