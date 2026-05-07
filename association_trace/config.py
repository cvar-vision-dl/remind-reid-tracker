from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssociationTraceSettings:
    enabled: bool = False
    mode: str = "off"
    write_json: bool = True
    output_group: str = "association_trace"
    write_schema_once: bool = True
    write_per_frame_class_trace: bool = True
    write_frame_previews: bool = True

    @classmethod
    def from_config(cls, config: dict | None) -> "AssociationTraceSettings":
        dbg = ((config or {}).get("debug", {}) or {})
        trace_cfg = (dbg.get("association_trace", {}) or {})

        enabled = bool(trace_cfg.get("enabled", False))
        raw_mode = str(trace_cfg.get("mode", "full" if enabled else "off")).strip().lower()
        mode = raw_mode if raw_mode in {"off", "summary", "full"} else "full"

        return cls(
            enabled=bool(enabled),
            mode=str(mode),
            write_json=bool(trace_cfg.get("write_json", True)),
            output_group=str(trace_cfg.get("output_group", "association_trace")),
            write_schema_once=bool(trace_cfg.get("write_schema_once", True)),
            write_per_frame_class_trace=bool(trace_cfg.get("write_per_frame_class_trace", True)),
            write_frame_previews=bool(trace_cfg.get("write_frame_previews", True)),
        )
