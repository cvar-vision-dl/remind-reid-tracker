from __future__ import annotations

from memory.temporary_track_base import TemporaryTrackBase


class AmbiguousTrack(TemporaryTrackBase):
    """Temporary entity for known detections that cannot be resolved yet."""

    def __init__(
        self,
        temp_id: int,
        class_id: int,
        class_name: str | None,
        candidate_ids: list[int],
        candidate_scores: dict[int, float] | None,
        timestamp: float,
        ttl: int,
        config: dict | None = None,
    ):
        super().__init__(
            temp_id=int(temp_id),
            class_id=int(class_id),
            class_name=class_name,
            timestamp=float(timestamp),
            config=config or {},
            track_kind="ambiguous",
        )
        self.current_candidate_ids = self._normalize_candidate_ids(candidate_ids)
        self.current_candidate_scores = self._normalize_candidate_scores(candidate_scores)
        self.candidate_ids = list(self.current_candidate_ids)
        self.candidate_scores = dict(self.current_candidate_scores)
        self.candidate_stats: dict[int, dict] = {}
        self.ttl_left = int(max(1, int(ttl)))
        self._update_candidate_stats(
            candidate_ids=self.current_candidate_ids,
            candidate_scores=self.current_candidate_scores,
            timestamp=float(timestamp),
        )
        self._sync_relationship_state()
        self.set_resolution_state(
            state="UNRESOLVED_KNOWN",
            confidence=self.best_candidate_confidence(),
            novelty_score=0.0,
        )

    def refresh(
        self,
        candidate_ids: list[int],
        candidate_scores: dict[int, float] | None,
        timestamp: float,
        ttl: int,
    ) -> None:
        self.current_candidate_ids = self._normalize_candidate_ids(candidate_ids)
        self.current_candidate_scores = self._normalize_candidate_scores(candidate_scores)
        self._merge_persistent_candidates(
            candidate_ids=self.current_candidate_ids,
            candidate_scores=self.current_candidate_scores,
        )
        self._update_candidate_stats(
            candidate_ids=self.current_candidate_ids,
            candidate_scores=self.current_candidate_scores,
            timestamp=float(timestamp),
        )
        self.last_seen = float(timestamp)
        self.ttl_left = int(max(1, int(ttl)))
        self._sync_relationship_state()
        self.set_resolution_state(
            state="UNRESOLVED_KNOWN",
            confidence=self.best_candidate_confidence(),
            novelty_score=0.0,
        )

    def _normalize_candidate_ids(self, candidate_ids: list[int] | None) -> list[int]:
        return self.normalize_ids(candidate_ids)

    def _normalize_candidate_scores(self, candidate_scores: dict[int, float] | None) -> dict[int, float]:
        return self.normalize_scores(candidate_scores)

    def _merge_persistent_candidates(
        self,
        *,
        candidate_ids: list[int],
        candidate_scores: dict[int, float],
    ) -> None:
        seen = set(int(x) for x in (self.candidate_ids or []))
        merged_ids = list(self.candidate_ids or [])
        for oid in candidate_ids:
            oid_i = int(oid)
            if oid_i in seen:
                continue
            seen.add(oid_i)
            merged_ids.append(int(oid_i))

        merged_scores = dict(self.candidate_scores or {})
        for oid, score in (candidate_scores or {}).items():
            oid_i = int(oid)
            score_f = float(score)
            prev = merged_scores.get(int(oid_i), None)
            if prev is None or score_f > float(prev):
                merged_scores[int(oid_i)] = float(score_f)

        self.candidate_ids = list(merged_ids)
        self.candidate_scores = dict(merged_scores)

    def _update_candidate_stats(
        self,
        *,
        candidate_ids: list[int],
        candidate_scores: dict[int, float],
        timestamp: float,
    ) -> None:
        for oid in candidate_ids:
            oid_i = int(oid)
            score = float(candidate_scores.get(int(oid_i), 0.0) or 0.0)
            prev = dict(self.candidate_stats.get(int(oid_i), {}) or {})
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

            self.candidate_stats[int(oid_i)] = {
                "seen_count": int(seen_count),
                "score_sum": float(score_sum),
                "score_avg": float(score_avg),
                "score_ema": float(score_ema),
                "best_score": float(best_score),
                "last_score": float(score),
                "last_seen": float(timestamp),
            }

    def _sync_relationship_state(self) -> None:
        self.sync_related_known_state(
            ids=self.current_candidate_ids,
            scores=self.current_candidate_scores,
            stats=self.candidate_stats,
        )

    def best_candidate_confidence(self) -> float:
        if not self.current_candidate_scores:
            return 0.0
        return float(max(float(v) for v in self.current_candidate_scores.values()))

    def candidate_quality(self, object_id: int) -> float:
        st = dict((self.candidate_stats or {}).get(int(object_id), {}) or {})
        count = float(st.get("seen_count", 0) or 0.0)
        avg = float(st.get("score_avg", 0.0) or 0.0)
        ema = float(st.get("score_ema", avg) or avg)
        best = float(st.get("best_score", ema) or ema)
        # Frequency and basic visual/contextual quality in one smooth signal.
        freq = float(min(1.0, count / 5.0))
        return float((0.45 * ema) + (0.35 * avg) + (0.10 * best) + (0.10 * freq))

    def persistent_ranked_candidate_ids(self) -> list[int]:
        ids = [int(x) for x in (self.candidate_ids or [])]
        ids.sort(
            key=lambda oid: (
                float(self.candidate_quality(int(oid))),
                int(((self.candidate_stats or {}).get(int(oid), {}) or {}).get("seen_count", 0) or 0),
                float((self.candidate_scores or {}).get(int(oid), 0.0) or 0.0),
            ),
            reverse=True,
        )
        return ids

    def age_seconds(self, timestamp: float) -> float:
        return float(max(0.0, float(timestamp) - float(self.created_at)))

    def display_label(self, memory_store, *, mode: str = "current") -> str:
        mode = str(mode or "current").strip().lower()
        if mode == "persistent":
            ids = self.persistent_ranked_candidate_ids()
        else:
            ids = [int(x) for x in (self.current_candidate_ids or [])]

        parts: list[str] = []
        for oid in ids:
            obj = memory_store.get(int(oid)) if memory_store is not None else None
            lbl = str(getattr(obj, "instance_label", "") or "")
            if "_" in lbl:
                tail = lbl.rsplit("_", 1)[-1]
                if tail.isdigit():
                    parts.append(str(int(tail)))
                    continue
            parts.append(str(int(oid)))

        body = "|".join(parts) if parts else "?"
        return f"T_{self.class_name}[{body}]"
