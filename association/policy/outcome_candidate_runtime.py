from __future__ import annotations


class OutcomeCandidateRuntime:
    """Runtime helper for candidate filtering and comparable score."""

    def __init__(self, *, combiner):
        self.combiner = combiner
        self.reset_runtime_caches()

    def reset_runtime_caches(self) -> None:
        self._runtime_iter_candidates_cache: dict[tuple[int, int, int, str], list[dict]] = {}
        self._runtime_comparable_pack_cache: dict[tuple[int, tuple[str, ...] | None], float] = {}
        self._runtime_comparable_term_cache: dict[int, set[str]] = {}
        self._runtime_comparable_score_map_cache: dict[tuple[int, ...], dict[int, float]] = {}

    def _candidate_comparable_score(
        self,
        candidate: dict,
        *,
        allowed_terms: set[str] | None = None,
    ) -> float:
        cid = int(id(candidate))
        terms_sig = None if allowed_terms is None else tuple(sorted(str(x) for x in (allowed_terms or set())))
        cache_key = (int(cid), terms_sig)
        cached = self._runtime_comparable_pack_cache.get(cache_key, None)
        if cached is not None:
            return float(cached)

        pack = self.combiner.combine_comparable_pack(
            candidate.get("scores", {}) or {},
            allowed_terms=allowed_terms,
        )
        score_sim = float((pack.get("core", {}) or {}).get("score_sim", 0.0) or 0.0)
        self._runtime_comparable_pack_cache[cache_key] = float(score_sim)
        if allowed_terms is None:
            collapsed = (((pack.get("debug", {}) or {}).get("collapsed", {}) or {}))
            candidate_terms = {
                str(name)
                for name in ("object", "bg_global", "bg_partials", "parts")
                if collapsed.get(str(name), None) is not None
            }
            self._runtime_comparable_term_cache[int(cid)] = set(candidate_terms)
        return float(score_sim)

    def iter_candidates(self, rep, *, scope: str = "raw") -> list[dict]:
        cands = getattr(rep, "candidates", None)
        if not isinstance(cands, list):
            return []
        scope = str(scope or "raw").strip().lower()
        cache_key = (int(id(rep)), int(id(cands)), int(len(cands)), str(scope))
        cached = self._runtime_iter_candidates_cache.get(cache_key, None)
        if cached is not None:
            return cached
        if scope == "raw":
            out = [c for c in cands if isinstance(c, dict)]
        elif scope == "eligible":
            # `decision_keep` = candidate eligible for final decision.
            out = [
                c for c in cands
                if isinstance(c, dict) and int(c.get("decision_keep", 0) or 0) == 1
            ]
        elif scope == "ambiguity":
            # `known_plausible_keep` = candidato conocido todavia plausible
            # for temporal ambiguity reasoning.
            out = [
                c for c in cands
                if isinstance(c, dict) and int(c.get("known_plausible_keep", 0) or 0) == 1
            ]
        else:
            out = [
                c for c in cands
                if isinstance(c, dict)
            ]
        self._runtime_iter_candidates_cache[cache_key] = out
        return out

    def compute_comparable_score_map(self, candidates: list[dict]) -> dict[int, float]:
        valid = [c for c in (candidates or []) if isinstance(c, dict)]
        if not valid:
            return {}

        map_cache_key = tuple(int(id(candidate)) for candidate in valid)
        cached_map = self._runtime_comparable_score_map_cache.get(map_cache_key, None)
        if cached_map is not None:
            return cached_map

        common_terms = {"object", "bg_global", "bg_partials", "parts"}
        first_terms: set[str] | None = None
        same_terms = True
        full_map: dict[int, float] = {}
        for candidate in valid:
            candidate_id = int(id(candidate))
            full_map[candidate_id] = float(self._candidate_comparable_score(candidate, allowed_terms=None))
            candidate_terms = set(self._runtime_comparable_term_cache.get(candidate_id, set()))
            common_terms &= candidate_terms
            if first_terms is None:
                first_terms = set(candidate_terms)
            elif candidate_terms != first_terms:
                same_terms = False

        if not common_terms:
            out = {
                id(candidate): float(candidate.get("score_sim", 0.0) or 0.0)
                for candidate in valid
            }
            self._runtime_comparable_score_map_cache[map_cache_key] = out
            return out

        if bool(same_terms and first_terms == common_terms):
            self._runtime_comparable_score_map_cache[map_cache_key] = full_map
            return full_map

        out = {}
        for candidate in valid:
            out[int(id(candidate))] = float(
                self._candidate_comparable_score(
                    candidate,
                    allowed_terms=common_terms,
                )
            )
        self._runtime_comparable_score_map_cache[map_cache_key] = out
        return out
