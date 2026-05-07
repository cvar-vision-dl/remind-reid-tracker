# memory/neighbor_graph.py

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set

class NeighborEdge:
    """
    Arista dirigida A -> B, contada por episodios (no por frame).
    """

    def __init__(self, dst_id: int, timestamp: float, episode_idx: int):
        self.dst_id = int(dst_id)

        self.cooc_count = 1
        self.weight = 1.0

        self.first_seen_ts = float(timestamp)
        self.last_seen_ts = float(timestamp)
        self.last_seen_episode = int(episode_idx)

    def bump(self, timestamp: float, episode_idx: int, inc: float = 1.0) -> None:
        """
        Incrementa conteos/weight.
        """
        self.last_seen_ts = float(timestamp)
        self.last_seen_episode = int(episode_idx)
        self.cooc_count += 1
        self.weight += float(inc)


class NeighborGraph:
    """
    Modelo (memoria) de vecinos para un objeto A (dirigido A -> B)..
    """

    def __init__(self, config: dict | None = None):
        cfg = config or {}

        self.enabled = bool(cfg.get("enabled", True))

        self.smoothing_alpha = float(cfg.get("smoothing_alpha", 0.5))
        self.smoothing_alpha = max(0.0, self.smoothing_alpha)

        self.trim_strategy = str(cfg.get("trim_strategy", "weight")).strip().lower()
        if self.trim_strategy not in ("weight", "recent"):
            self.trim_strategy = "weight"

        self.edges: Dict[int, NeighborEdge] = {}

        self.episode_count = 0
        self.episode_idx = -1
        self.last_episode_frame_id: int | None = None

        self.stable_context: Optional[Set[int]] = None
        self.pending_context: Optional[Set[int]] = None
        self.pending_hits = 0

    # ------------------------------------------------------------------
    # Queries / métricas (no policy)
    # ------------------------------------------------------------------

    def self_episode_count(self) -> int:
        return int(self.episode_count)

    def cooc_count(self, other_id: int) -> int:
        e = self.edges.get(int(other_id), None)
        return int(e.cooc_count) if e is not None else 0

    def p_conditional(self, other_id: int, vocab_size: int | None = None) -> float:
        """
        P(B|A) ≈ (cAB + alpha) / (cA + alpha*V)
        """
        cA = float(max(0, self.episode_count))
        cAB = float(max(0, self.cooc_count(other_id)))

        a = float(self.smoothing_alpha)
        V = max(1, int(vocab_size)) if vocab_size is not None else max(1, len(self.edges))

        denom = cA + a * float(V)
        if denom <= 1e-12:
            return 0.0
        return float((cAB + a) / denom)

    def pmi(self, other_id: int, other_self_episode_count: int | None, total_episodes: int | None) -> float:
        """
        PMI(A,B) ≈ log( P(A,B) / (P(A)P(B)) )
        """
        if other_self_episode_count is None or total_episodes is None:
            return 0.0

        T = float(max(1, int(total_episodes)))

        cAB = float(self.cooc_count(other_id))
        cA = float(max(0, self.episode_count))
        cB = float(max(0, int(other_self_episode_count)))

        eps = 1e-12
        pAB = max(eps, cAB / T)
        pA = max(eps, cA / T)
        pB = max(eps, cB / T)

        return float(math.log(pAB / (pA * pB)))

    def neighbors(self) -> List[dict]:
        """
        Devuelve vecinos ordenados según trim_strategy.
        """
        items = list(self.edges.items())
        if self.trim_strategy == "recent":
            items.sort(key=lambda kv: float(kv[1].last_seen_ts), reverse=True)
        else:
            items.sort(key=lambda kv: (float(kv[1].weight), float(kv[1].last_seen_ts)), reverse=True)

        out = []
        for dst_id, e in items:
            out.append(
                {
                    "dst_id": int(dst_id),
                    "cooc_count": int(e.cooc_count),
                    "weight": float(e.weight),
                    "last_seen_ts": float(e.last_seen_ts),
                    "last_seen_episode": int(e.last_seen_episode),
                }
            )
        return out

    def topk(self, k: int = 5) -> List[dict]:
        k = int(max(0, k))
        if k == 0:
            return []
        return self.neighbors()[:k]
