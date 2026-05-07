from __future__ import annotations

from memory.temporary_track_base import TemporaryTrackBase


class ProvisionalNewTrack(TemporaryTrackBase):
    """Entidad temporal para posibles objetos nuevos aún no comprometidos."""

    def __init__(
        self,
        temp_id: int,
        class_id: int,
        class_name: str | None,
        support_known_ids: list[int] | None,
        support_known_scores: dict[int, float] | None,
        context_mode: str,
        timestamp: float,
        ttl: int,
        reason: str = "UNCERTAIN_NEW",
        config: dict | None = None,
    ):
        super().__init__(
            temp_id=int(temp_id),
            class_id=int(class_id),
            class_name=class_name,
            timestamp=float(timestamp),
            config=config or {},
            track_kind="provisional",
        )
        self.support_known_ids = self.normalize_ids(support_known_ids)
        self.support_known_scores = self.normalize_scores(support_known_scores)
        self.context_mode = str(context_mode or "none")
        self.reason = str(reason or "UNCERTAIN_NEW")
        self.ttl_left = int(max(1, int(ttl)))
        self.support_known_stats: dict[int, dict] = {}
        self._update_support_known_stats(
            support_known_ids=self.support_known_ids,
            support_known_scores=self.support_known_scores,
            timestamp=float(timestamp),
        )
        self.sync_related_known_state(
            ids=self.support_known_ids,
            scores=self.support_known_scores,
            stats=self.support_known_stats,
        )
        self.set_resolution_state(
            state=self.resolution_state_from_reason(self.reason),
            confidence=self.best_related_known_confidence(),
            novelty_score=self.compute_novelty_score(),
        )

    def refresh(
        self,
        *,
        support_known_ids: list[int] | None,
        support_known_scores: dict[int, float] | None,
        context_mode: str,
        timestamp: float,
        ttl: int,
        reason: str | None = None,
    ) -> None:
        self.support_known_ids = self.normalize_ids(support_known_ids)
        self.support_known_scores = self.normalize_scores(support_known_scores)
        self.context_mode = str(context_mode or self.context_mode or "none")
        if reason:
            self.reason = str(reason)
        self.last_seen = float(timestamp)
        self.ttl_left = int(max(1, int(ttl)))
        self._update_support_known_stats(
            support_known_ids=self.support_known_ids,
            support_known_scores=self.support_known_scores,
            timestamp=float(timestamp),
        )
        self.sync_related_known_state(
            ids=self.support_known_ids,
            scores=self.support_known_scores,
            stats=self.support_known_stats,
        )
        self.set_resolution_state(
            state=self.resolution_state_from_reason(self.reason),
            confidence=self.best_related_known_confidence(),
            novelty_score=self.compute_novelty_score(),
        )

    @staticmethod
    def resolution_state_from_reason(reason: str | None) -> str:
        reason_txt = str(reason or "").upper()
        if reason_txt.startswith("UNCERTAIN_PARENT"):
            return "UNRESOLVED_PARENT"
        return "UNRESOLVED_NOVEL"

    def _update_support_known_stats(
        self,
        *,
        support_known_ids: list[int],
        support_known_scores: dict[int, float],
        timestamp: float,
    ) -> None:
        for oid in support_known_ids or []:
            oid_i = int(oid)
            score = float(support_known_scores.get(int(oid_i), 0.0) or 0.0)
            prev = dict((self.support_known_stats or {}).get(int(oid_i), {}) or {})
            count_prev = int(prev.get("seen_count", 0) or 0)
            score_sum_prev = float(prev.get("score_sum", 0.0) or 0.0)
            ema_prev = float(prev.get("score_ema", score) or score)
            best_prev = float(prev.get("best_score", score) or score)
            alpha = 0.5

            seen_count = int(count_prev + 1)
            score_sum = float(score_sum_prev + score)
            score_avg = float(score_sum / max(1, seen_count))
            score_ema = float((alpha * score) + ((1.0 - alpha) * ema_prev)) if count_prev > 0 else float(score)
            best_score = float(max(best_prev, score))

            self.support_known_stats[int(oid_i)] = {
                "seen_count": int(seen_count),
                "score_sum": float(score_sum),
                "score_avg": float(score_avg),
                "score_ema": float(score_ema),
                "best_score": float(best_score),
                "last_score": float(score),
                "last_seen": float(timestamp),
            }

    def best_related_known_confidence(self) -> float:
        if not self.support_known_scores:
            return 0.0
        return float(max(float(v) for v in self.support_known_scores.values()))

    def compute_novelty_score(self) -> float:
        best_known = float(self.best_related_known_confidence())
        if best_known <= 0.0:
            return 1.0
        return float(max(0.0, min(1.0, 1.0 - best_known)))

    def display_label(self, memory_store) -> str:
        if str(self.reason or "").upper().startswith("UNCERTAIN_PARENT"):
            if memory_store is not None and self.support_known_ids:
                parent = memory_store.get(int(self.support_known_ids[0]))
                parent_lbl = getattr(parent, "instance_label", None) if parent is not None else None
                if parent_lbl:
                    return f"T_{self.class_name}_PARENT->{parent_lbl}"
            return f"T_{self.class_name}_PARENT"

        _ = memory_store
        # Un provisional puede estar relacionado con conocidos ya ocupados en el
        # mismo frame; mostrar esos IDs en la etiqueta visual induce a leerlo
        # como "podria ser ese conocido", cuando semanticamente sigue siendo
        # una hipotesis de novedad.
        return f"T_{self.class_name}_NEW"
