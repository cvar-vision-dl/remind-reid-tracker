# memory/part_model.py

from __future__ import annotations

import numpy as np


class PartPrototype:
    """Prototipo persistente de una parte."""

    def __init__(self, embedding: np.ndarray, timestamp: float):
        self.embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self.count = 1
        self.last_seen = float(timestamp)


class PartChannel:
    """
    Per-channel part memory with two banks:
      - work_protos: protos recientes / de trabajo
      - stable_protos: protos consolidados
    """

    def __init__(self, name: str, max_prototypes: int | None = None):
        self.name = str(name)
        self.max_prototypes = None if max_prototypes is None else int(max_prototypes)

        self.work_protos: list[PartPrototype] = []
        self.stable_protos: list[PartPrototype] = []

    def num_parts(self) -> int:
        return int(len(self.work_protos))

    def num_stable_parts(self) -> int:
        return int(len(self.stable_protos))

    def get_embeddings(self) -> list[np.ndarray]:
        if self.stable_protos:
            return [p.embedding for p in self.stable_protos]
        return [p.embedding for p in self.work_protos]

    def get_work_embeddings(self) -> list[np.ndarray]:
        return [p.embedding for p in self.work_protos]

    def get_stable_embeddings(self) -> list[np.ndarray]:
        return [p.embedding for p in self.stable_protos]

    def add_work_prototype(self, embedding: np.ndarray, timestamp: float) -> None:
        self.work_protos.append(PartPrototype(embedding=embedding, timestamp=timestamp))

    def add_stable_prototype(self, embedding: np.ndarray, timestamp: float) -> None:
        self.stable_protos.append(PartPrototype(embedding=embedding, timestamp=timestamp))


class PartModel:
    """Part-channel memory container for a TrackedObject."""

    DEFAULT_CHANNELS = ["kmeans", "attention"]

    def __init__(self, config: dict | None = None):
        cfg = config or {}

        self.enabled = bool(cfg.get("enabled", True))

        mpc = cfg.get("max_prototypes_per_channel", None)
        self.max_prototypes_per_channel = None if mpc is None else int(mpc)

        names = self.resolve_channel_names(cfg)
        if not names:
            names = list(self.DEFAULT_CHANNELS)

        self.channels: dict[str, PartChannel] = {}
        if self.enabled:
            for name in names:
                self.channels[str(name)] = PartChannel(
                    name=str(name),
                    max_prototypes=self.max_prototypes_per_channel,
                )

    def resolve_channel_names(self, cfg: dict) -> list[str]:
        ch_cfg = cfg.get("channels", {}) or {}
        names: list[str] = []
        for name, sub in ch_cfg.items():
            if isinstance(sub, dict) and sub.get("enabled", False):
                names.append(str(name))
        return names

    def channel_names(self) -> list[str]:
        return list(self.channels.keys())

    def get_channel(self, name: str) -> PartChannel | None:
        return self.channels.get(str(name), None)

    def get_channel_embeddings(self, name: str) -> list[np.ndarray]:
        ch = self.get_channel(name)
        return [] if ch is None else ch.get_embeddings()

    def get_channel_work_embeddings(self, name: str) -> list[np.ndarray]:
        ch = self.get_channel(name)
        return [] if ch is None else ch.get_work_embeddings()

    def get_channel_stable_embeddings(self, name: str) -> list[np.ndarray]:
        ch = self.get_channel(name)
        return [] if ch is None else ch.get_stable_embeddings()
