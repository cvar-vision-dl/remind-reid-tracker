# association/score_aggregator.py

from __future__ import annotations

from association.scores.base_scores import BaseScores
from association.scores.neighbor_sets_score import NeighborSetsScore


class ScoreAggregator:
    """
    Orquesta la computación de módulos de scoring.

    - BaseScores: obj/bg/parts
    - NeighborSetsScore: hipótesis de sets (contexto global)
    - El módulo relacional de distancias ya no participa aquí en scoring base.
    """

    def __init__(self, config: dict, memory_store=None):
        self.config = config
        self.memory_store = memory_store

        self.base = BaseScores(config)
        self.neigh_sets = NeighborSetsScore(config, memory_store=memory_store)

    def compute_base(self, det_feats: dict | None, tracked_object) -> dict:
        return self.base.compute(det_feats, tracked_object)

    def compute_neighbor_sets(
        self,
        detections: list,
        anchor_object_ids: list[int] | None = None,
        timestamp: float | None = None,
        vocab_size: int | None = None,
    ) -> dict:
        return self.neigh_sets.compute(
            detections=detections,
            anchor_object_ids=anchor_object_ids,
            timestamp=timestamp,
            vocab_size=vocab_size,
        )

