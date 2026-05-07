from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AssignmentContext:
    ns_ctx: dict = field(default_factory=dict)
    use_sets_bonus: bool = False


@dataclass(slots=True)
class AssignmentResult:
    decided_matches: list[tuple[int, int, float]] = field(default_factory=list)
    to_create: list[tuple[int, int]] = field(default_factory=list)
