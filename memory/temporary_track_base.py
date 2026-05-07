from __future__ import annotations

from memory.background_model import LocalBackgroundModel
from memory.object_appearance import ObjectAppearanceModel
from memory.part_model import PartModel


class TemporaryTrackBase:
    """Base ligera para tracks temporales con continuidad propia entre frames."""

    def __init__(
        self,
        *,
        temp_id: int,
        class_id: int,
        class_name: str | None,
        timestamp: float,
        config: dict,
        track_kind: str,
    ):
        self.temp_id = int(temp_id)
        self.class_id = int(class_id)
        self.class_name = str(class_name or f"CLASS{int(class_id)}").upper()
        self.track_kind = str(track_kind or "temporary").lower()
        self.object_id = int(self.synthetic_object_id(self.track_kind, self.temp_id))

        self.created_at = float(timestamp)
        self.first_seen = float(timestamp)
        self.last_seen = float(timestamp)
        self.state = "TEMPORARY"
        # `temp_id` da continuidad temporal al hipotesis-track, pero no
        # significa identidad semantica confirmada del objeto.
        self.identity_label = f"T_{self.class_name}_{int(self.temp_id)}"
        self.instance_label = str(self.identity_label)
        self.resolution_state = "UNRESOLVED"
        self.resolution_target = None
        self.identity_confidence = 0.0
        self.novelty_score = 0.0
        self.related_known_ids: list[int] = []
        self.related_known_scores: dict[int, float] = {}
        self.related_known_stats: dict[int, dict] = {}

        self.hits = 1
        self.misses = 0
        self.age = 1
        self.observation_count = 0

        self.last_geom = None
        self.last_bbox = None
        self.last_center = None
        self.last_area = None
        self.observation_history: list[dict] = []

        tmp_cfg = (((config or {}).get("update", {}) or {}).get("temporary_tracks", {}) or {})
        self.max_observation_history = max(1, int(tmp_cfg.get("max_observation_history", 12)))

        mem_cfg = (config.get("memory", {}) or {}) if isinstance(config, dict) else {}
        self.appearance = ObjectAppearanceModel(config=(mem_cfg.get("appearance", {}) or {}))
        self.parts = PartModel(config=(mem_cfg.get("parts", {}) or {}))
        self.background = LocalBackgroundModel(config=config or {})

    @staticmethod
    def synthetic_object_id(track_kind: str, temp_id: int) -> int:
        if str(track_kind) == "ambiguous":
            return -1000000 - int(temp_id)
        if str(track_kind) == "provisional":
            return -2000000 - int(temp_id)
        return -3000000 - int(temp_id)

    @staticmethod
    def normalize_ids(ids: list[int] | None) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for value in ids or []:
            try:
                value_i = int(value)
            except Exception:
                continue
            if value_i in seen:
                continue
            seen.add(value_i)
            out.append(int(value_i))
        return out

    @staticmethod
    def normalize_scores(scores: dict[int, float] | None) -> dict[int, float]:
        out: dict[int, float] = {}
        for key, value in (scores or {}).items():
            if key is None or value is None:
                continue
            try:
                out[int(key)] = float(value)
            except Exception:
                continue
        return out

    def sync_related_known_state(
        self,
        *,
        ids: list[int] | None,
        scores: dict[int, float] | None,
        stats: dict[int, dict] | None = None,
    ) -> None:
        self.related_known_ids = list(ids or [])
        self.related_known_scores = dict(scores or {})
        self.related_known_stats = dict(stats or {})

    def set_resolution_state(
        self,
        *,
        state: str,
        target=None,
        confidence: float | None = None,
        novelty_score: float | None = None,
    ) -> None:
        self.resolution_state = str(state or "UNRESOLVED").upper()
        self.resolution_target = target
        if confidence is not None:
            self.identity_confidence = float(max(0.0, min(1.0, float(confidence))))
        if novelty_score is not None:
            self.novelty_score = float(max(0.0, min(1.0, float(novelty_score))))

    def display_label(self, memory_store=None) -> str:
        _ = memory_store
        return str(self.identity_label)

    def mark_visible(self, timestamp: float) -> None:
        self.last_seen = float(timestamp)
        self.misses = 0
        self.hits = int(getattr(self, "hits", 0)) + 1
        self.age = int(getattr(self, "age", 0)) + 1

    def record_observation(
        self,
        *,
        timestamp: float,
        geom: dict | None = None,
        score: float | None = None,
        metadata: dict | None = None,
    ) -> None:
        self.observation_count = int(getattr(self, "observation_count", 0)) + 1
        self.last_seen = float(timestamp)
        if isinstance(geom, dict):
            self.last_geom = dict(geom)
            self.last_bbox = geom.get("bbox", None)
            self.last_center = geom.get("center", None)
            self.last_area = geom.get("area", None)

        entry = {
            "timestamp": float(timestamp),
            "score": None if score is None else float(score),
        }
        if isinstance(geom, dict):
            entry["geom"] = {
                "center": geom.get("center", None),
                "area": geom.get("area", None),
                "bbox": geom.get("bbox", None),
            }
        if isinstance(metadata, dict) and metadata:
            entry["metadata"] = dict(metadata)

        self.observation_history.append(entry)
        max_hist = int(getattr(self, "max_observation_history", 12))
        if max_hist > 0 and len(self.observation_history) > max_hist:
            self.observation_history = self.observation_history[-max_hist:]
