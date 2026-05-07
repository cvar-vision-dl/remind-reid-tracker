# memory/tracked_object.py

from memory.object_appearance import ObjectAppearanceModel
from memory.part_model import PartModel
from memory.neighbor_graph import NeighborGraph
from memory.background_model import LocalBackgroundModel
from memory.cross_view_identity import CrossViewIdentity
from memory.neighbor_distance_graph import NeighborDistanceGraph


class TrackedObject:
    def __init__(
        self,
        object_id: int,
        class_id: int,
        timestamp: float,
        config: dict,
        class_name=None,
    ):
        self.object_id = int(object_id)
        self.class_id = int(class_id)

        self.class_name = str(class_name)

        self.first_seen = float(timestamp)
        self.last_seen = float(timestamp)

        self.state = "NEW"

        self.instance_label = None
        self.origin = {
            "mode": "DIRECT_NEW",
            "reason": "",
            "context_mode": "none",
            "competition_mode": "none",
            "provisional_temp_id": None,
            "parent_object_ids": [],
            "parent_object_scores": {},
            "related_known_ids": [],
            "related_known_scores": {},
            "support_known_ids": [],
            "support_known_scores": {},
        }

        self.hits = 0
        self.misses = 0
        self.age = 0

        mem_cfg = (config.get("memory", {}) or {}) if isinstance(config, dict) else {}

        app_cfg = mem_cfg.get("appearance", {}) or {}
        self.appearance = ObjectAppearanceModel(config=app_cfg)

        parts_mem_cfg = mem_cfg.get("parts", {}) or {}
        self.parts = PartModel(config=parts_mem_cfg)

        neigh_cfg = mem_cfg.get("neighbors", {}) or {}
        self.neighbors = NeighborGraph(config=neigh_cfg)

        dist_cfg = mem_cfg.get("neighbors_distance", {}) or {}
        self.neighbor_dist = NeighborDistanceGraph(config=dist_cfg)

        cv_cfg = mem_cfg.get("cross_view_identity", {}) or {}
        self.cross_view = CrossViewIdentity(config=cv_cfg)

        self.background = LocalBackgroundModel(config=config)
