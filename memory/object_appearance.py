# memory/object_appearance.py

from __future__ import annotations

import numpy as np

from utils.math import l2_normalize_vector


class AppearancePrototype:
    """Prototipo individual de apariencia."""

    def __init__(
        self,
        embedding: np.ndarray,
        timestamp: float,
        count: int = 1,
        quality_ema: float = 1.0,
        quality_sum: float = 1.0,
        n_obs: int = 1,
        first_seen: float | None = None,
    ):
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        self.embedding = l2_normalize_vector(emb)
        self.count = max(1, int(count))
        self.last_seen = float(timestamp)
        self.first_seen = float(timestamp if first_seen is None else first_seen)
        self.quality_ema = float(max(0.0, min(1.0, quality_ema)))
        self.quality_sum = float(max(0.0, quality_sum))
        self.n_obs = max(1, int(n_obs))
        self.stability = float(self.quality_ema)

    def copy(self) -> "AppearancePrototype":
        return AppearancePrototype(
            embedding=self.embedding.copy(),
            timestamp=self.last_seen,
            count=self.count,
            quality_ema=self.quality_ema,
            quality_sum=self.quality_sum,
            n_obs=self.n_obs,
            first_seen=self.first_seen,
        )


class AppearanceChannel:
    """
    Canal de memoria de apariencia con dos bancos:
      - work_protos: protos recientes / de trabajo
      - stable_protos: protos consolidados
    """

    def __init__(self, name: str, max_prototypes: int | None = None, max_stable: int | None = None):
        self.name = str(name)
        self.max_prototypes = None if max_prototypes is None else int(max_prototypes)
        self.max_stable = None if max_stable is None else int(max_stable)

        self.work_protos: list[AppearancePrototype] = []
        self.stable_protos: list[AppearancePrototype] = []

    def num_work(self) -> int:
        return int(len(self.work_protos))

    def num_stable(self) -> int:
        return int(len(self.stable_protos))

    def get_work_embeddings(self) -> list[np.ndarray]:
        return [p.embedding for p in self.work_protos]

    def get_stable_embeddings(self) -> list[np.ndarray]:
        return [p.embedding for p in self.stable_protos]

    def get_embeddings(self) -> list[np.ndarray]:
        if self.stable_protos:
            return [p.embedding for p in self.stable_protos]
        return [p.embedding for p in self.work_protos]

    def add_work_prototype(self, embedding: np.ndarray, timestamp: float, **proto_kwargs) -> int:
        proto = AppearancePrototype(embedding=embedding, timestamp=timestamp, **proto_kwargs)
        self.work_protos.append(proto)
        return int(len(self.work_protos) - 1)

    def add_stable_prototype(self, embedding: np.ndarray, timestamp: float, **proto_kwargs) -> int:
        proto = AppearancePrototype(embedding=embedding, timestamp=timestamp, **proto_kwargs)
        self.stable_protos.append(proto)
        return int(len(self.stable_protos) - 1)

    def append_work_proto(self, proto: AppearancePrototype) -> int:
        self.work_protos.append(proto)
        return int(len(self.work_protos) - 1)

    def append_stable_proto(self, proto: AppearancePrototype) -> int:
        self.stable_protos.append(proto)
        return int(len(self.stable_protos) - 1)


class ObjectAppearanceModel:
    """Modelo de apariencia a nivel objeto: organiza múltiples canales."""

    DEFAULT_CHANNELS = ["global", "global_trimmed", "patch"]

    def __init__(self, config: dict | None = None):
        cfg = config or {}

        self.enabled = bool(cfg.get("enabled", True))

        mpc = cfg.get("max_prototypes_per_channel", None)
        self.max_prototypes_per_channel = None if mpc is None else int(mpc)

        max_stable = cfg.get("max_stable_prototypes_per_channel", None)
        self.max_stable_prototypes_per_channel = None if max_stable is None else int(max_stable)

        names = self.resolve_channel_names(cfg)
        if not names:
            names = list(self.DEFAULT_CHANNELS)

        self.channels: dict[str, AppearanceChannel] = {}
        if self.enabled:
            for name in names:
                self.channels[str(name)] = AppearanceChannel(
                    name=str(name),
                    max_prototypes=self.max_prototypes_per_channel,
                    max_stable=self.max_stable_prototypes_per_channel,
                )

    def resolve_channel_names(self, cfg: dict) -> list[str]:
        if not isinstance(cfg, dict):
            return []

        names = []
        ch_cfg = cfg.get("channels", None)

        if isinstance(ch_cfg, dict):
            for name, sub in ch_cfg.items():
                if isinstance(sub, dict) and bool(sub.get("enabled", False)):
                    names.append(str(name))
            return names

        for name in self.DEFAULT_CHANNELS:
            sub = cfg.get(name, {}) or {}
            if isinstance(sub, dict) and bool(sub.get("enabled", False)):
                names.append(str(name))

        return names

    def channel_names(self) -> list[str]:
        return list(self.channels.keys())

    def get_channel(self, name: str) -> AppearanceChannel | None:
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
