# memory/background_model.py

from __future__ import annotations


class BackgroundPrototype:
    def __init__(
        self,
        embedding,
        timestamp,
        weight: float = 1.0,
        count: int = 1,
        quality_ema: float = 1.0,
        quality_sum: float = 1.0,
        n_obs: int = 1,
    ):
        self.embedding = embedding
        self.weight = float(weight)
        self.count = max(1, int(count))
        self.first_seen = float(timestamp)
        self.last_seen = float(timestamp)
        self.quality_ema = float(max(0.0, min(1.0, quality_ema)))
        self.quality_sum = float(max(0.0, quality_sum))
        self.n_obs = max(1, int(n_obs))

    def copy(self):
        return BackgroundPrototype(
            embedding=self.embedding.copy(),
            timestamp=self.last_seen,
            weight=self.weight,
            count=self.count,
            quality_ema=self.quality_ema,
            quality_sum=self.quality_sum,
            n_obs=self.n_obs,
        )


class PrototypeBank:
    """Banco FIFO (por first_seen) si max_size > 0."""

    def __init__(self, max_size: int):
        self.max_size = int(max_size)
        self.prototypes = []

    def __len__(self):
        return int(len(self.prototypes))

    def get_embeddings(self):
        return [p.embedding for p in self.prototypes]

    def get_weights(self):
        return [float(getattr(p, "weight", 1.0)) for p in self.prototypes]

    def add(self, embedding, timestamp: float, weight: float = 1.0, **proto_kwargs):
        proto = BackgroundPrototype(
            embedding=embedding,
            timestamp=float(timestamp),
            weight=float(weight),
            **proto_kwargs,
        )
        self.prototypes.append(proto)
        self.trim(now=float(timestamp))
        return proto

    def trim(self, now: float):
        if self.max_size <= 0:
            return
        if len(self.prototypes) <= self.max_size:
            return

        self.prototypes.sort(key=lambda p: float(p.first_seen))
        while len(self.prototypes) > self.max_size:
            self.prototypes.pop(0)


class StableFirstBank:
    """Vista de banco: usa stable si hay protos; si no, usa work."""

    def __init__(self, stable_bank: PrototypeBank, work_bank: PrototypeBank):
        self.stable_bank = stable_bank
        self.work_bank = work_bank

    def __len__(self):
        if len(self.stable_bank) > 0:
            return int(len(self.stable_bank))
        return int(len(self.work_bank))

    def get_embeddings(self):
        if len(self.stable_bank) > 0:
            return self.stable_bank.get_embeddings()
        return self.work_bank.get_embeddings()

    def get_weights(self):
        if len(self.stable_bank) > 0:
            return self.stable_bank.get_weights()
        return self.work_bank.get_weights()


class LocalBackgroundModel:
    def __init__(self, config: dict):
        bg_cfg = (config.get("memory", {}) or {}).get("background", {}) or {}

        self.enabled = bool(bg_cfg.get("enabled", True))

        max_inner = int(bg_cfg.get("max_inner", 20))
        max_outer = int(bg_cfg.get("max_outer", 30))

        max_inner_partials = int(bg_cfg.get("max_inner_partials", max_inner))
        max_outer_partials = int(bg_cfg.get("max_outer_partials", max_outer))

        self.inner_global_work = PrototypeBank(max_size=max_inner)
        self.outer_global_work = PrototypeBank(max_size=max_outer)

        self.inner_global_stable = PrototypeBank(max_size=int(bg_cfg.get("max_inner_global_stable", 0)))
        self.outer_global_stable = PrototypeBank(max_size=int(bg_cfg.get("max_outer_global_stable", 0)))

        self.inner_partials_work = PrototypeBank(max_size=max_inner_partials)
        self.outer_partials_work = PrototypeBank(max_size=max_outer_partials)

        self.inner_partials_stable = PrototypeBank(max_size=int(bg_cfg.get("max_inner_partials_stable", 0)))
        self.outer_partials_stable = PrototypeBank(max_size=int(bg_cfg.get("max_outer_partials_stable", 0)))

        self.inner_global = StableFirstBank(self.inner_global_stable, self.inner_global_work)
        self.outer_global = StableFirstBank(self.outer_global_stable, self.outer_global_work)

        self.inner_partials = StableFirstBank(self.inner_partials_stable, self.inner_partials_work)
        self.outer_partials = StableFirstBank(self.outer_partials_stable, self.outer_partials_work)

        w = bg_cfg.get("combine_weights", {}) or {}
        self.w_inner = float(w.get("inner", 0.5))
        self.w_outer = float(w.get("outer", 0.5))

        s = float(self.w_inner + self.w_outer)
        if s <= 1e-12:
            self.w_inner = 0.5
            self.w_outer = 0.5
        else:
            self.w_inner /= s
            self.w_outer /= s
