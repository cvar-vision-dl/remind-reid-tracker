from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from association_trace.collector import (
    AssociationTraceCollector,
    JsonAssociationTraceCollector,
    NoOpAssociationTraceCollector,
)
from association_trace.config import AssociationTraceSettings


def create_association_trace_collector(
    *,
    config: dict | None,
    output_dir: str | Path | None,
) -> AssociationTraceCollector:
    settings = AssociationTraceSettings.from_config(config)
    if not settings.enabled:
        return NoOpAssociationTraceCollector(settings=settings)
    if output_dir is None:
        # No output destination means trace cannot be materialized; treat as disabled.
        return NoOpAssociationTraceCollector(settings=replace(settings, enabled=False, mode="off"))
    return JsonAssociationTraceCollector(settings=settings, output_dir=output_dir)
