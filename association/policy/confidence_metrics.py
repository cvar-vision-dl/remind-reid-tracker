from __future__ import annotations

import math

from association.reports import SimilarityReport


class ConfidenceMetrics:
    """Cálculo de confianza y factores auxiliares para asociación."""

    def __init__(
        self,
        *,
        combiner,
        conf_enabled: bool,
        conf_T: float,
        conf_cov_target: float,
        conf_eps: float,
        conf_min_temperature: float,
        conf_min_gap_k: float,
        conf_margin_mode: str,
        conf_gap_center: float,
        conf_gap_k: float,
    ) -> None:
        self.combiner = combiner
        self.conf_enabled = bool(conf_enabled)
        self.conf_T = float(conf_T)
        self.conf_cov_target = float(conf_cov_target)
        self.conf_eps = float(conf_eps)
        self.conf_min_temperature = float(conf_min_temperature)
        self.conf_min_gap_k = float(conf_min_gap_k)
        self.conf_margin_mode = str(conf_margin_mode)
        self.conf_gap_center = float(conf_gap_center)
        self.conf_gap_k = float(conf_gap_k)

    def compute_confidence(
        self,
        rep: SimilarityReport,
        s1: float,
        s2: float,
        gap: float,
        scores_sorted: list[float],
    ) -> dict:
        if not self.conf_enabled:
            return {"confidence": 0.0}

        eps = float(max(1e-12, self.conf_eps))
        T = float(self.conf_T) if float(self.conf_T) > eps else float(self.conf_min_temperature)

        p1 = self.softmax_top2_p1(s1, s2, T=T)
        margin_factor, margin_raw = self.compute_margin_factor(s1=s1, gap=gap, eps=eps)

        mean_others = 0.0
        if len(scores_sorted) > 1:
            mean_others = float(sum(scores_sorted[1:]) / float(len(scores_sorted) - 1))

        distinct = float(max(0.0, s1 - mean_others))
        distinct_norm = float(distinct / max(1.0 - mean_others, eps))
        distinct_norm = float(max(0.0, min(1.0, distinct_norm)))

        coverage = self.compute_coverage_factor(rep)
        conf = float(s1) * float(p1) * float(margin_factor) * float(coverage) * float(distinct_norm)
        conf = float(max(0.0, min(1.0, conf)))

        return {
            "confidence": float(conf),
            "conf_p1": float(p1),
            "conf_margin": float(margin_factor),
            "conf_margin_raw": float(margin_raw),
            "conf_coverage": float(coverage),
            "conf_distinct": float(distinct_norm),
        }

    def compute_margin_factor(self, s1: float, gap: float, eps: float) -> tuple[float, float]:
        mode = str(self.conf_margin_mode or "sigmoid_abs")

        abs_gap = float(max(0.0, gap))
        rel = float(abs_gap / max(float(s1), eps)) if s1 > 0.0 else 0.0

        if mode == "linear_abs":
            return float(max(0.0, min(1.0, abs_gap))), float(abs_gap)

        if mode == "linear_rel":
            return float(max(0.0, min(1.0, rel))), float(rel)

        center = float(self.conf_gap_center)
        k = float(self.conf_gap_k)
        if k <= eps:
            k = float(self.conf_min_gap_k)

        if mode == "sigmoid_rel":
            x = float((rel - center) / k)
            return float(self.sigmoid(x)), float(rel)

        x = float((abs_gap - center) / k)
        return float(self.sigmoid(x)), float(abs_gap)

    @staticmethod
    def sigmoid(x: float) -> float:
        z = float(max(-50.0, min(50.0, float(x))))
        return float(1.0 / (1.0 + math.exp(-z)))

    def compute_coverage_factor(self, rep: SimilarityReport) -> float:
        best = getattr(rep, "best", None)
        if not isinstance(best, dict):
            return 0.0

        scores = best.get("scores", None)
        if not isinstance(scores, dict):
            return 0.0

        s_obj = self.combiner.collapse_object_score(scores.get("object", {}) or {}, self.combiner.object_mode)
        s_bg_g = self.combiner.collapse_background_score(scores.get("background", {}) or {}, self.combiner.background_mode)
        s_bg_p = self.combiner.collapse_background_partials_score(scores.get("background", {}) or {})
        s_parts = self.combiner.collapse_parts_score(scores.get("parts", {}) or {}, self.combiner.parts_mode)

        present = 0
        present += 1 if s_obj is not None else 0
        present += 1 if s_bg_g is not None else 0
        present += 1 if s_bg_p is not None else 0
        present += 1 if s_parts is not None else 0

        frac = float(present) / 4.0
        target = float(max(1e-6, self.conf_cov_target))
        cov = float(min(1.0, frac / target))
        return float(max(0.0, min(1.0, cov)))

    @staticmethod
    def softmax_top2_p1(s1: float, s2: float, T: float) -> float:
        z1 = float(s1) / float(T)
        z2 = float(s2) / float(T)

        zmax = float(max(z1, z2))
        e1 = math.exp(float(z1 - zmax))
        e2 = math.exp(float(z2 - zmax))
        den = float(e1 + e2)
        return float(e1 / den) if den > 1e-12 else 0.5
