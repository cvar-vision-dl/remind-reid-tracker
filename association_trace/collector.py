from __future__ import annotations

import json

import cv2
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from association_trace.config import AssociationTraceSettings
from association_trace.schema import build_node_order_map, build_pipeline_schema
from utils.logging import default_run_artifact_dir, default_run_timestamp


def _sorted_serialized_list(values) -> list:
    return sorted((_serialize_value(x) for x in values), key=lambda x: json.dumps(x, sort_keys=True))


def _serialize_value(value):
    if is_dataclass(value):
        return _serialize_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(x) for x in value]
    if isinstance(value, set):
        return _sorted_serialized_list(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _default_node_run(node_id: str, scope_key: dict, participants=None) -> dict:
    return {
        "node_id": str(node_id),
        "entered": True,
        "skipped_reason": "",
        "scope_key": _serialize_value(scope_key or {}),
        "participants": _serialize_value(participants or {}),
        "checks": [],
        "values": {},
        "decision": {},
        "candidate_rows": [],
        "detection_rows": [],
        "global_rows": [],
    }


class AssociationTraceCollector:
    def __init__(self, settings: AssociationTraceSettings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return bool(self.settings.enabled)

    def start_frame(self, frame_id: int, timestamp: float, det_ids: list[int] | None = None) -> None:
        del frame_id, timestamp, det_ids

    def start_class(
        self,
        frame_id: int,
        class_id: int,
        class_name: str | None,
        det_ids: list[int] | None = None,
        snapshot_object_ids: list[int] | None = None,
    ) -> None:
        del frame_id, class_id, class_name, det_ids, snapshot_object_ids

    def finish_class(self, frame_id: int, class_id: int) -> None:
        del frame_id, class_id

    def finish_frame(self, frame_id: int) -> None:
        del frame_id

    def enter_node(self, node_id: str, scope_key: dict, participants=None) -> None:
        del node_id, scope_key, participants

    def skip_node(self, node_id: str, scope_key: dict, reason: str, participants=None) -> None:
        del node_id, scope_key, reason, participants

    def add_check(self, node_id: str, scope_key: dict, check: dict) -> None:
        del node_id, scope_key, check

    def add_value(self, node_id: str, scope_key: dict, key: str, value) -> None:
        del node_id, scope_key, key, value

    def set_values(self, node_id: str, scope_key: dict, values: dict) -> None:
        del node_id, scope_key, values

    def set_decision(self, node_id: str, scope_key: dict, decision: dict) -> None:
        del node_id, scope_key, decision

    def add_candidate_row(self, node_id: str, scope_key: dict, row: dict) -> None:
        del node_id, scope_key, row

    def add_detection_row(self, node_id: str, scope_key: dict, row: dict) -> None:
        del node_id, scope_key, row

    def add_global_row(self, node_id: str, scope_key: dict, row: dict) -> None:
        del node_id, scope_key, row

    def leave_node(self, node_id: str, scope_key: dict) -> None:
        del node_id, scope_key

    def save_frame_preview(self, frame_id: int, image_bgr) -> None:
        del frame_id, image_bgr

    def save_memory_snapshot(self, frame_id: int, snapshot: dict) -> None:
        del frame_id, snapshot

    def flush(self) -> None:
        return None


class NoOpAssociationTraceCollector(AssociationTraceCollector):
    pass


class JsonAssociationTraceCollector(AssociationTraceCollector):
    def __init__(self, *, settings: AssociationTraceSettings, output_dir: str | Path):
        super().__init__(settings=settings)
        self.output_dir = Path(output_dir)
        self.run_id = f"run_{default_run_timestamp()}"
        self.run_dir = Path(
            default_run_artifact_dir(
                str(self.output_dir),
                group=self.settings.output_group,
                prefix="run",
                timestamp=self.run_id.removeprefix("run_"),
            )
        )
        self.frames_dir = self.run_dir / "frames"
        self.frame_previews_dir = self.run_dir / "frame_previews"
        self.memory_snapshots_dir = self.run_dir / "memory_snapshots"
        self.pipeline_schema = build_pipeline_schema()
        self.node_order_map = build_node_order_map(self.pipeline_schema)
        self.trace_version = "1.0"
        self._manifest = {
            "run_id": str(self.run_id),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "module": "association",
            "schema_version": str(self.pipeline_schema.get("schema_version", "1.0")),
            "trace_version": str(self.trace_version),
            "frame_ids": [],
            "class_entries": [],
            "frame_previews": [],
            "memory_snapshots": [],
        }
        self._frame_trace_meta: dict[int, dict[str, Any]] = {}
        self._class_traces: dict[tuple[int, int], dict] = {}
        self._schema_written = False

    def _ensure_output_dirs(self) -> None:
        if not self.settings.write_json:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.frame_previews_dir.mkdir(parents=True, exist_ok=True)
        self.memory_snapshots_dir.mkdir(parents=True, exist_ok=True)

    def _write_json(self, path: Path, payload: dict) -> None:
        if not self.settings.write_json:
            return
        self._ensure_output_dirs()
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    def _normalize_node_run(self, node_run: dict) -> dict:
        payload = dict(node_run or {})
        payload["participants"] = _serialize_value(payload.get("participants", {}) or {})
        payload["checks"] = list(payload.get("checks", []) or [])
        payload["candidate_rows"] = list(payload.get("candidate_rows", []) or [])
        payload["detection_rows"] = list(payload.get("detection_rows", []) or [])
        payload["global_rows"] = list(payload.get("global_rows", []) or [])
        payload["values"] = _serialize_value(payload.get("values", {}) or {})
        payload["decision"] = _serialize_value(payload.get("decision", {}) or {})
        return payload

    def _normalize_trace_for_write(self, trace: dict) -> dict:
        payload = dict(trace or {})
        payload["node_runs"] = sorted(
            [self._normalize_node_run(node_run) for node_run in (payload.get("node_runs", []) or [])],
            key=lambda node_run: (
                self.node_order_map.get(str(node_run.get("node_id")), (10**9, 10**9, str(node_run.get("node_id")))),
                json.dumps(node_run.get("scope_key", {}), sort_keys=True),
            ),
        )
        return payload

    def _normalize_manifest_for_write(self) -> dict:
        payload = dict(self._manifest)
        payload["frame_ids"] = sorted(int(x) for x in (payload.get("frame_ids", []) or []))
        payload["class_entries"] = sorted(
            [dict(item) for item in (payload.get("class_entries", []) or []) if isinstance(item, dict)],
            key=lambda item: (
                int(item.get("frame_id", -1)),
                int(item.get("class_id", -1)),
                str(item.get("path", "")),
            ),
        )
        payload["frame_previews"] = sorted(
            [dict(item) for item in (payload.get("frame_previews", []) or []) if isinstance(item, dict)],
            key=lambda item: (int(item.get("frame_id", -1)), str(item.get("path", ""))),
        )
        payload["memory_snapshots"] = sorted(
            [dict(item) for item in (payload.get("memory_snapshots", []) or []) if isinstance(item, dict)],
            key=lambda item: (int(item.get("frame_id", -1)), str(item.get("path", ""))),
        )
        return payload

    def _manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    def _schema_path(self) -> Path:
        return self.run_dir / "pipeline_schema.json"

    def _class_trace_path(self, frame_id: int, class_id: int) -> Path:
        return self.frames_dir / f"frame_{int(frame_id):06d}_class_{int(class_id):03d}.json"

    def _frame_preview_path(self, frame_id: int) -> Path:
        return self.frame_previews_dir / f"frame_{int(frame_id):06d}.png"

    def _memory_snapshot_path(self, frame_id: int) -> Path:
        return self.memory_snapshots_dir / f"frame_{int(frame_id):06d}.json"

    def _sorted_unique_ints(self, values) -> list[int]:
        return sorted({int(x) for x in (values or [])})

    def _ensure_class_trace(
        self,
        *,
        frame_id: int,
        class_id: int,
        class_name: str | None = None,
        det_ids: list[int] | None = None,
        snapshot_object_ids: list[int] | None = None,
        timestamp: float | None = None,
    ) -> dict:
        key = (int(frame_id), int(class_id))
        trace = self._class_traces.get(key, None)
        if trace is None:
            frame_meta = self._frame_trace_meta.get(int(frame_id), {})
            trace = {
                "trace_version": str(self.trace_version),
                "schema_version": str(self.pipeline_schema.get("schema_version", "1.0")),
                "module": "association",
                "run_id": str(self.run_id),
                "frame_id": int(frame_id),
                "class_id": int(class_id),
                "class_name": None if class_name is None else str(class_name),
                "timestamp": float(frame_meta.get("timestamp", timestamp or 0.0)),
                "det_ids": self._sorted_unique_ints(det_ids),
                "snapshot_object_ids": self._sorted_unique_ints(snapshot_object_ids),
                "node_runs": [],
            }
            self._class_traces[key] = trace
        else:
            if class_name is not None and not trace.get("class_name"):
                trace["class_name"] = str(class_name)
            if det_ids:
                trace["det_ids"] = self._sorted_unique_ints(list(trace.get("det_ids", [])) + list(det_ids or []))
            if snapshot_object_ids:
                trace["snapshot_object_ids"] = self._sorted_unique_ints(
                    list(trace.get("snapshot_object_ids", [])) + list(snapshot_object_ids or [])
                )
            if timestamp is not None:
                trace["timestamp"] = float(timestamp)
        return trace

    def _find_or_create_node_run(self, *, node_id: str, scope_key: dict, participants=None) -> dict | None:
        frame_id = scope_key.get("frame_id", None)
        class_id = scope_key.get("class_id", None)
        if frame_id is None or class_id is None:
            return None
        trace = self._ensure_class_trace(frame_id=int(frame_id), class_id=int(class_id))
        for node_run in trace["node_runs"]:
            if str(node_run.get("node_id")) == str(node_id) and node_run.get("scope_key", {}) == _serialize_value(scope_key):
                if participants and not node_run.get("participants"):
                    node_run["participants"] = _serialize_value(participants)
                return node_run
        node_run = _default_node_run(node_id=str(node_id), scope_key=scope_key, participants=participants)
        trace["node_runs"].append(node_run)
        return node_run

    def _register_class_entry(self, frame_id: int, class_id: int, class_name: str | None, path: str) -> None:
        key = (int(frame_id), int(class_id), str(path))
        existing = {
            (int(item.get("frame_id", -1)), int(item.get("class_id", -1)), str(item.get("path", "")))
            for item in (self._manifest.get("class_entries", []) or [])
            if isinstance(item, dict)
        }
        if key in existing:
            return
        self._manifest["class_entries"].append(
            {
                "frame_id": int(frame_id),
                "class_id": int(class_id),
                "class_name": None if class_name is None else str(class_name),
                "path": str(path),
            }
        )

    def _register_frame_preview(self, frame_id: int, path: str) -> None:
        key = (int(frame_id), str(path))
        existing = {
            (int(item.get("frame_id", -1)), str(item.get("path", "")))
            for item in (self._manifest.get("frame_previews", []) or [])
            if isinstance(item, dict)
        }
        if key in existing:
            return
        self._manifest["frame_previews"].append(
            {
                "frame_id": int(frame_id),
                "path": str(path),
            }
        )

    def _register_memory_snapshot(self, frame_id: int, path: str) -> None:
        key = (int(frame_id), str(path))
        existing = {
            (int(item.get("frame_id", -1)), str(item.get("path", "")))
            for item in (self._manifest.get("memory_snapshots", []) or [])
            if isinstance(item, dict)
        }
        if key in existing:
            return
        self._manifest["memory_snapshots"].append(
            {
                "frame_id": int(frame_id),
                "path": str(path),
            }
        )

    def start_frame(self, frame_id: int, timestamp: float, det_ids: list[int] | None = None) -> None:
        frame_id = int(frame_id)
        self._frame_trace_meta[frame_id] = {
            "timestamp": float(timestamp),
            "det_ids": self._sorted_unique_ints(det_ids),
        }
        frame_ids = set(int(x) for x in (self._manifest.get("frame_ids", []) or []))
        frame_ids.add(int(frame_id))
        self._manifest["frame_ids"] = sorted(frame_ids)

        if self.settings.write_schema_once and not self._schema_written:
            self._write_json(self._schema_path(), self.pipeline_schema)
            self._schema_written = True

    def start_class(
        self,
        frame_id: int,
        class_id: int,
        class_name: str | None,
        det_ids: list[int] | None = None,
        snapshot_object_ids: list[int] | None = None,
    ) -> None:
        trace = self._ensure_class_trace(
            frame_id=int(frame_id),
            class_id=int(class_id),
            class_name=class_name,
            det_ids=det_ids,
            snapshot_object_ids=snapshot_object_ids,
        )
        rel_path = self._class_trace_path(int(frame_id), int(class_id)).relative_to(self.run_dir)
        self._register_class_entry(
            frame_id=int(frame_id),
            class_id=int(class_id),
            class_name=trace.get("class_name"),
            path=str(rel_path),
        )

    def finish_class(self, frame_id: int, class_id: int) -> None:
        key = (int(frame_id), int(class_id))
        trace = self._class_traces.get(key, None)
        if trace is None or not self.settings.write_per_frame_class_trace:
            return
        self._write_json(
            self._class_trace_path(int(frame_id), int(class_id)),
            self._normalize_trace_for_write(trace),
        )

    def finish_frame(self, frame_id: int) -> None:
        del frame_id
        self.flush()

    def enter_node(self, node_id: str, scope_key: dict, participants=None) -> None:
        self._find_or_create_node_run(node_id=str(node_id), scope_key=scope_key, participants=participants)

    def skip_node(self, node_id: str, scope_key: dict, reason: str, participants=None) -> None:
        node_run = self._find_or_create_node_run(
            node_id=str(node_id),
            scope_key=scope_key,
            participants=participants,
        )
        if node_run is None:
            return
        node_run["entered"] = False
        node_run["skipped_reason"] = str(reason or "")

    def add_check(self, node_id: str, scope_key: dict, check: dict) -> None:
        node_run = self._find_or_create_node_run(node_id=str(node_id), scope_key=scope_key)
        if node_run is None:
            return
        node_run["checks"].append(_serialize_value(check or {}))

    def add_value(self, node_id: str, scope_key: dict, key: str, value) -> None:
        node_run = self._find_or_create_node_run(node_id=str(node_id), scope_key=scope_key)
        if node_run is None:
            return
        node_run["values"][str(key)] = _serialize_value(value)

    def set_values(self, node_id: str, scope_key: dict, values: dict) -> None:
        node_run = self._find_or_create_node_run(node_id=str(node_id), scope_key=scope_key)
        if node_run is None:
            return
        node_run["values"] = _serialize_value(values or {})

    def set_decision(self, node_id: str, scope_key: dict, decision: dict) -> None:
        node_run = self._find_or_create_node_run(node_id=str(node_id), scope_key=scope_key)
        if node_run is None:
            return
        node_run["decision"] = _serialize_value(decision or {})

    def add_candidate_row(self, node_id: str, scope_key: dict, row: dict) -> None:
        node_run = self._find_or_create_node_run(node_id=str(node_id), scope_key=scope_key)
        if node_run is None:
            return
        node_run["candidate_rows"].append(_serialize_value(row or {}))

    def add_detection_row(self, node_id: str, scope_key: dict, row: dict) -> None:
        node_run = self._find_or_create_node_run(node_id=str(node_id), scope_key=scope_key)
        if node_run is None:
            return
        node_run["detection_rows"].append(_serialize_value(row or {}))

    def add_global_row(self, node_id: str, scope_key: dict, row: dict) -> None:
        node_run = self._find_or_create_node_run(node_id=str(node_id), scope_key=scope_key)
        if node_run is None:
            return
        node_run["global_rows"].append(_serialize_value(row or {}))

    def leave_node(self, node_id: str, scope_key: dict) -> None:
        del node_id, scope_key

    def save_frame_preview(self, frame_id: int, image_bgr) -> None:
        if not self.settings.write_frame_previews or image_bgr is None:
            return
        self._ensure_output_dirs()
        path = self._frame_preview_path(int(frame_id))
        cv2.imwrite(str(path), image_bgr)
        rel_path = path.relative_to(self.run_dir)
        self._register_frame_preview(int(frame_id), str(rel_path))
        self._write_json(self._manifest_path(), self._normalize_manifest_for_write())

    def save_memory_snapshot(self, frame_id: int, snapshot: dict) -> None:
        if not self.settings.write_json or snapshot is None:
            return
        self._ensure_output_dirs()
        path = self._memory_snapshot_path(int(frame_id))
        self._write_json(path, _serialize_value(snapshot or {}))
        rel_path = path.relative_to(self.run_dir)
        self._register_memory_snapshot(int(frame_id), str(rel_path))
        self._write_json(self._manifest_path(), self._normalize_manifest_for_write())

    def flush(self) -> None:
        if self.settings.write_schema_once and not self._schema_written:
            self._write_json(self._schema_path(), self.pipeline_schema)
            self._schema_written = True
        self._write_json(self._manifest_path(), self._normalize_manifest_for_write())
