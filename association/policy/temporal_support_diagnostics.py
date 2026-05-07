from __future__ import annotations


class TemporalSupportDiagnostics:
    """Diagnosticos de soporte temporal y reglas de gap."""

    def __init__(
        self,
        *,
        amb_track_min_top_score: float,
        prov_new_min_top_score: float,
        amb_track_min_candidate_score: float,
        prov_new_min_candidate_score: float,
        amb_track_gap_max: float,
        amb_track_gap_rel_max: float,
        amb_track_gap_eps: float,
        prov_new_gap_max: float,
        prov_new_gap_rel_max: float,
        prov_new_gap_eps: float,
        temporal_candidate_score_fn,
    ) -> None:
        self.amb_track_min_top_score = float(amb_track_min_top_score)
        self.prov_new_min_top_score = float(prov_new_min_top_score)
        self.amb_track_min_candidate_score = float(amb_track_min_candidate_score)
        self.prov_new_min_candidate_score = float(prov_new_min_candidate_score)
        self.amb_track_gap_max = float(amb_track_gap_max)
        self.amb_track_gap_rel_max = float(amb_track_gap_rel_max)
        self.amb_track_gap_eps = float(amb_track_gap_eps)
        self.prov_new_gap_max = float(prov_new_gap_max)
        self.prov_new_gap_rel_max = float(prov_new_gap_rel_max)
        self.prov_new_gap_eps = float(prov_new_gap_eps)
        self.temporal_candidate_score_fn = temporal_candidate_score_fn

    def amb_track_gap_allows(self, top_score: float, cand_score: float) -> bool:
        return bool(
            self.candidate_gap_allows(
                top_score=top_score,
                cand_score=cand_score,
                gap_max=self.amb_track_gap_max,
                gap_rel_max=self.amb_track_gap_rel_max,
                gap_eps=self.amb_track_gap_eps,
            )
        )

    def prov_new_gap_allows(self, top_score: float, cand_score: float) -> bool:
        return bool(
            self.candidate_gap_allows(
                top_score=top_score,
                cand_score=cand_score,
                gap_max=self.prov_new_gap_max,
                gap_rel_max=self.prov_new_gap_rel_max,
                gap_eps=self.prov_new_gap_eps,
            )
        )

    @staticmethod
    def candidate_gap_allows(
        *,
        top_score: float,
        cand_score: float,
        gap_max: float,
        gap_rel_max: float,
        gap_eps: float,
    ) -> bool:
        s1 = float(top_score)
        s2 = float(cand_score)
        gap_abs = float(max(0.0, s1 - s2))
        if gap_abs <= float(gap_max + gap_eps):
            return True

        den = max(1e-12, abs(float(s1)))
        gap_rel = float(gap_abs / den)
        return bool(gap_rel <= float(gap_rel_max + gap_eps))

    def compute_temporal_support_diag_from_candidates(
        self,
        candidates: list[dict],
        *,
        mode: str,
        score_map: dict[int, float] | None = None,
    ) -> dict:
        scored: list[tuple[float, dict]] = []
        for candidate in (candidates or []):
            if not isinstance(candidate, dict):
                continue
            scored.append((float(self.temporal_candidate_score_fn(candidate, score_map=score_map)), candidate))

        if not scored:
            return {
                "s1": 0.0,
                "s2": 0.0,
                "gap": 0.0,
                "status": "WEAK",
                "reason": "NO_TEMPORAL_SUPPORT",
                "n_plausible": 0,
                "plausible_candidates": [],
            }

        scored.sort(key=lambda item: float(item[0]), reverse=True)
        s1 = float(scored[0][0])
        s2 = float(scored[1][0]) if len(scored) > 1 else 0.0
        is_ambiguous_mode = str(mode).lower() == "ambiguous"
        min_top = float(self.amb_track_min_top_score if is_ambiguous_mode else self.prov_new_min_top_score)
        min_candidate = float(
            self.amb_track_min_candidate_score if is_ambiguous_mode else self.prov_new_min_candidate_score
        )
        gap_allows = self.amb_track_gap_allows if is_ambiguous_mode else self.prov_new_gap_allows

        plausible_candidates: list[dict] = []
        for score, candidate in scored:
            if float(score) < float(min_candidate):
                continue
            if not gap_allows(s1, float(score)):
                continue
            plausible_candidates.append(candidate)

        gap = float(max(0.0, s1 - s2))
        status = "WEAK"
        reason = "LOW_TEMPORAL_SUPPORT"
        if s1 >= float(min_top):
            if len(plausible_candidates) >= 2:
                status = "AMBIGUOUS"
                reason = "MULTI_PLAUSIBLE_TEMPORAL"
            else:
                status = "STRONG"
                reason = "SINGLE_PLAUSIBLE_TEMPORAL"

        return {
            "s1": float(s1),
            "s2": float(s2),
            "gap": float(gap),
            "status": str(status),
            "reason": str(reason),
            "n_plausible": int(len(plausible_candidates)),
            "plausible_candidates": list(plausible_candidates),
        }
