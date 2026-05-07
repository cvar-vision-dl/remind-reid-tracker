from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class NormalizedAssignmentState:
    decided_matches: list[tuple[int, int, float]] = field(default_factory=list)
    assigned_by_det_id: dict[int, int] = field(default_factory=dict)


@dataclass(slots=True)
class TemporalResolutionState:
    decided_matches: list[tuple[int, int, float]] = field(default_factory=list)
    create_entries: list[dict] = field(default_factory=list)
    ambiguous_entries: list[dict] = field(default_factory=list)
    provisional_entries: list[dict] = field(default_factory=list)
    resolved_source_by_det_id: dict[int, str] = field(default_factory=dict)
    known_set_distance_disambiguation: dict = field(default_factory=dict)
    postcreate_debug_entries: list[dict] = field(default_factory=list)
