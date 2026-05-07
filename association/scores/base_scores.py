# association/scores/base_scores.py

from __future__ import annotations

import math

from association.similarity_computer import SimilarityComputer
from utils.config import bg_partials_enabled


class BaseScores:
    """Scores directos (obj/bg/parts) delegando en SimilarityComputer."""

    def __init__(self, config: dict):
        self.config = config or {}
        self.sim = SimilarityComputer(config)

        sim_cfg = ((self.config.get("association", {}) or {}).get("similarity", {}) or {})
        q_cfg = (sim_cfg.get("quality", {}) or {})
        self.quality_enabled = bool(q_cfg.get("enabled", True))

        obj_cfg = (q_cfg.get("object", {}) or {})
        self.obj_min_patches = max(1, int(obj_cfg.get("min_patches", 8)))
        self.obj_full_patches = max(self.obj_min_patches, int(obj_cfg.get("full_patches", 32)))
        self.obj_min_coverage_density = max(0.0, min(1.0, float(obj_cfg.get("min_coverage_density", 0.15))))
        self.obj_full_coverage_density = max(
            self.obj_min_coverage_density,
            min(1.0, float(obj_cfg.get("full_coverage_density", 0.60))),
        )
        self.obj_weight_floor = max(0.0, min(1.0, float(obj_cfg.get("weight_floor", 0.65))))

        parts_cfg = (q_cfg.get("parts", {}) or {})
        self.parts_disable_below_obj_patches = max(0, int(parts_cfg.get("disable_below_obj_patches", 8)))
        self.parts_min_obj_patches = max(1, int(parts_cfg.get("min_obj_patches", 12)))
        self.parts_full_obj_patches = max(self.parts_min_obj_patches, int(parts_cfg.get("full_obj_patches", 32)))
        self.parts_min_parts = max(1, int(parts_cfg.get("min_parts", 2)))
        self.parts_full_parts = max(self.parts_min_parts, int(parts_cfg.get("full_parts", 4)))
        self.parts_min_support_frac = max(0.0, float(parts_cfg.get("min_support_frac", 0.15)))
        self.parts_full_support_frac = max(self.parts_min_support_frac, float(parts_cfg.get("full_support_frac", 0.45)))
        self.parts_min_coverage_density = max(0.0, min(1.0, float(parts_cfg.get("min_coverage_density", 0.15))))
        self.parts_full_coverage_density = max(
            self.parts_min_coverage_density,
            min(1.0, float(parts_cfg.get("full_coverage_density", 0.60))),
        )

        bg_cfg = (q_cfg.get("background", {}) or {})
        self.bg_min_inner_patches = max(1, int(bg_cfg.get("min_inner_patches", 8)))
        self.bg_full_inner_patches = max(self.bg_min_inner_patches, int(bg_cfg.get("full_inner_patches", 20)))
        self.bg_min_outer_patches = max(1, int(bg_cfg.get("min_outer_patches", 12)))
        self.bg_full_outer_patches = max(self.bg_min_outer_patches, int(bg_cfg.get("full_outer_patches", 32)))
        self.bg_min_mask_quality = max(0.0, min(1.0, float(bg_cfg.get("min_mask_quality", 0.5))))
        self.bg_weight_floor = max(0.0, min(1.0, float(bg_cfg.get("weight_floor", 0.35))))

    def compute(self, det_feats: dict | None, tracked_object) -> dict:
        det_feats = det_feats or {}
        out = {
            "object": self.sim.object_score(det_feats, tracked_object),
            "background": self.sim.background_score(det_feats, tracked_object),
            "parts": self.sim.parts_score(det_feats, tracked_object),
        }
        out["_quality"] = self.build_quality(det_feats)
        return out

    def ramp(self, x: float | int | None, x0: float, x1: float) -> float:
        if x is None:
            return 0.0
        xv = float(x)
        if x1 <= x0:
            return 1.0 if xv >= x1 else 0.0
        return float(max(0.0, min(1.0, (xv - x0) / (x1 - x0))))

    def compute_patch_density(
        self,
        *,
        effective_patches: float | int | None,
        n_patches: float | int | None,
    ) -> float:
        if effective_patches is None or n_patches is None:
            return 1.0
        den = float(n_patches)
        if den <= 1e-12:
            return 1.0
        return float(max(0.0, min(1.0, float(effective_patches) / den)))

    def build_quality(self, det_feats: dict | None) -> dict:
        if not self.quality_enabled:
            return {"object": 1.0, "parts": 1.0, "background": 1.0}

        meta = ((det_feats or {}).get("meta", {}) or {})

        n_obj_patches = meta.get("n_obj_patches", None)
        effective_obj_patches = meta.get("effective_obj_patches", None)
        n_parts_valid = int(meta.get("n_parts_valid", 0) or 0)
        parts_support = float(meta.get("parts_support", 0.0) or 0.0)
        n_bg_inner = int(meta.get("n_bg_inner_patches", 0) or 0)
        n_bg_outer = int(meta.get("n_bg_outer_patches", 0) or 0)
        bg_mask_quality = float(meta.get("bg_mask_quality", 1.0) or 1.0)

        obj_support = effective_obj_patches if effective_obj_patches is not None else n_obj_patches
        obj_patch_density = self.compute_patch_density(
            effective_patches=effective_obj_patches,
            n_patches=n_obj_patches,
        )
        q_obj = self.ramp(obj_support, self.obj_min_patches, self.obj_full_patches)

        q_parts_obj = self.ramp(obj_support, self.parts_min_obj_patches, self.parts_full_obj_patches)
        q_parts_count = self.ramp(n_parts_valid, self.parts_min_parts, self.parts_full_parts)
        support_denom = float(obj_support or 0.0)
        support_frac = float(parts_support / max(1.0, support_denom)) if support_denom > 0.0 else 0.0
        q_parts_support = self.ramp(support_frac, self.parts_min_support_frac, self.parts_full_support_frac)
        q_parts = float((q_parts_obj + q_parts_count + q_parts_support) / 3.0)
        if obj_support is not None and float(obj_support) < float(self.parts_disable_below_obj_patches):
            q_parts = 0.0

        q_bg_inner = self.ramp(n_bg_inner, self.bg_min_inner_patches, self.bg_full_inner_patches)
        q_bg_outer = self.ramp(n_bg_outer, self.bg_min_outer_patches, self.bg_full_outer_patches)
        q_bg_mask = self.ramp(bg_mask_quality, self.bg_min_mask_quality, 1.0)
        q_bg = float((q_bg_inner + q_bg_outer + q_bg_mask) / 3.0)

        return {
            "object": float(max(0.0, min(1.0, q_obj))),
            "parts": float(max(0.0, min(1.0, q_parts))),
            "background": float(max(0.0, min(1.0, q_bg))),
            "meta": {
                "n_obj_patches": None if n_obj_patches is None else int(n_obj_patches),
                "effective_obj_patches": None if effective_obj_patches is None else float(effective_obj_patches),
                "obj_patch_density": float(obj_patch_density),
                "n_parts_valid": int(n_parts_valid),
                "parts_support_frac": float(support_frac),
                "n_bg_inner_patches": int(n_bg_inner),
                "n_bg_outer_patches": int(n_bg_outer),
                "bg_mask_quality": float(bg_mask_quality),
            },
        }


class SimilarityCombiner:
    """
    Combina términos de similitud (obj/bg/parts) en un escalar.

    - Si un término es None, se ignora y se renormalizan pesos (opcional).
    - work/stable: colapsa dicts usando "max" si existe o max(work, stable).
    """

    def __init__(self, config: dict):
        self.config = config or {}

        assoc_cfg = self.config.get("association", {}) or {}
        match_cfg = assoc_cfg.get("matching", {}) or {}

        self.object_mode = str(match_cfg.get("object_mode", "max")).lower()
        self.background_mode = str(match_cfg.get("background_mode", "combined")).lower()
        self.parts_mode = str(match_cfg.get("parts_mode", "max")).lower()

        self.renormalize_missing = bool(match_cfg.get("renormalize_missing", True))

        w_cfg = match_cfg.get("weights", {}) or {}
        self.w_obj = float(w_cfg.get("object", 1.0))
        self.w_bg_global = float(w_cfg.get("background_global", 0.0))
        self.w_bg_partials = float(w_cfg.get("background_partials", 0.0))
        self.w_parts = float(w_cfg.get("parts", 0.0))
        self.bg_partials_enabled = bool(bg_partials_enabled(self.config))
        if not self.bg_partials_enabled:
            self.w_bg_partials = 0.0

        sim_cfg = ((assoc_cfg.get("similarity", {}) or {}).get("quality", {}) or {})
        obj_qcfg = (sim_cfg.get("object", {}) or {})
        bg_qcfg = (sim_cfg.get("background", {}) or {})
        parts_qcfg = (sim_cfg.get("parts", {}) or {})
        self.obj_min_coverage_density = max(0.0, min(1.0, float(obj_qcfg.get("min_coverage_density", 0.15))))
        self.obj_full_coverage_density = max(
            self.obj_min_coverage_density,
            min(1.0, float(obj_qcfg.get("full_coverage_density", 0.60))),
        )
        self.parts_min_coverage_density = max(0.0, min(1.0, float(parts_qcfg.get("min_coverage_density", 0.15))))
        self.parts_full_coverage_density = max(
            self.parts_min_coverage_density,
            min(1.0, float(parts_qcfg.get("full_coverage_density", 0.60))),
        )
        self.q_floor_object = max(0.0, min(1.0, float(obj_qcfg.get("weight_floor", 0.65))))
        self.q_floor_background = max(0.0, min(1.0, float(bg_qcfg.get("weight_floor", 0.35))))
        self.q_floor_parts = max(0.0, min(1.0, float(parts_qcfg.get("weight_floor", 0.0))))

    def combine(self, scores: dict, det_feats: dict | None = None) -> float:
        return float(self.combine_pack(scores, det_feats=det_feats)["core"]["score_sim"])

    def combine_pack(self, scores: dict, det_feats: dict | None = None) -> dict:
        scores = scores or {}
        _ = det_feats

        collapsed = self.build_collapsed_scores(scores)
        quality = self.build_quality_pack(scores, object_proto_quality=collapsed["object_proto"])
        terms = self.build_base_terms(collapsed=collapsed, quality=quality)
        base = float(self.combine_base_terms(terms))
        score_sim = float(base)

        return {
            "core": {"score_sim": float(score_sim)},
            "debug": {
                "base": float(base),
                "ctx": 0.0,
                "ctx_raw": 0.0,
                "ctx_scale": 1.0,
                "collapsed": {
                    "object": collapsed["object"],
                    "bg_global": collapsed["bg_global"],
                    "bg_partials": collapsed["bg_partials"],
                    "parts": collapsed["parts"],
                },
                "quality": {
                    "object": float(quality["object"]),
                    "object_proto": float(quality["object_proto"]),
                    "object_joint": float(quality["object_joint"]),
                    "background": float(quality["background"]),
                    "parts": float(quality["parts"]),
                },
                "quality_effective": {
                    "object": float(quality["effective_object"]),
                    "background": float(quality["effective_background"]),
                    "parts": float(quality["effective_parts"]),
                },
                "effective_weights": {
                    "object": float(quality["weight_object"]),
                    "bg_global": float(quality["weight_bg_global"]),
                    "bg_partials": float(quality["weight_bg_partials"]),
                    "parts": float(quality["weight_parts"]),
                },
            },
        }

    def combine_consistent_pack(
        self,
        scores: dict,
        *,
        source_policy: dict[str, str] | None = None,
    ) -> dict:
        scores = scores or {}

        collapsed = self.build_collapsed_scores(scores, source_policy=source_policy)
        object_scores = scores.get("object", {}) or {}
        object_source = None if not isinstance(source_policy, dict) else source_policy.get("object", None)
        quality = self.build_quality_pack(
            scores,
            object_proto_quality=self.collapse_object_proto_quality(
                object_scores,
                self.object_mode,
                forced_source=object_source,
            ),
        )
        terms = self.build_base_terms(collapsed=collapsed, quality=quality)
        base = float(self.combine_base_terms(terms))

        return {
            "core": {"score_sim": float(base)},
            "debug": {
                "base": float(base),
                "collapsed": {
                    "object": collapsed["object"],
                    "bg_global": collapsed["bg_global"],
                    "bg_partials": collapsed["bg_partials"],
                    "parts": collapsed["parts"],
                },
                "quality": {
                    "object": float(quality["object"]),
                    "object_proto": float(quality["object_proto"]),
                    "object_joint": float(quality["object_joint"]),
                    "background": float(quality["background"]),
                    "parts": float(quality["parts"]),
                },
                "quality_effective": {
                    "object": float(quality["effective_object"]),
                    "background": float(quality["effective_background"]),
                    "parts": float(quality["effective_parts"]),
                },
                "effective_weights": {
                    "object": float(quality["weight_object"]),
                    "bg_global": float(quality["weight_bg_global"]),
                    "bg_partials": float(quality["weight_bg_partials"]),
                    "parts": float(quality["weight_parts"]),
                },
                "source_policy": dict(source_policy or {}),
            },
        }

    def combine_comparable_pack(
        self,
        scores: dict,
        *,
        allowed_terms: set[str] | None = None,
    ) -> dict:
        scores = scores or {}

        collapsed = self.build_collapsed_scores(scores, source_neutral=True)
        if allowed_terms is not None:
            allowed = {str(x) for x in (allowed_terms or set())}
            collapsed = {
                "object": collapsed["object"] if "object" in allowed else None,
                "bg_global": collapsed["bg_global"] if "bg_global" in allowed else None,
                "bg_partials": collapsed["bg_partials"] if "bg_partials" in allowed else None,
                "parts": collapsed["parts"] if "parts" in allowed else None,
                "object_proto": None,
            }

        quality = self.build_comparable_quality_pack(scores)
        terms = self.build_base_terms(collapsed=collapsed, quality=quality)
        base = float(self.combine_base_terms(terms))

        return {
            "core": {"score_sim": float(base)},
            "debug": {
                "base": float(base),
                "collapsed": {
                    "object": collapsed["object"],
                    "bg_global": collapsed["bg_global"],
                    "bg_partials": collapsed["bg_partials"],
                    "parts": collapsed["parts"],
                },
            },
        }

    def build_collapsed_scores(
        self,
        scores: dict,
        *,
        source_neutral: bool = False,
        source_policy: dict[str, str] | None = None,
    ) -> dict:
        object_scores = scores.get("object", {}) or {}
        background_scores = scores.get("background", {}) or {}
        parts_scores = scores.get("parts", {}) or {}
        policy = source_policy or {}
        return {
            "object": self.collapse_object_score(
                object_scores,
                self.object_mode,
                source_neutral=source_neutral,
                forced_source=policy.get("object", None),
            ),
            "object_proto": self.collapse_object_proto_quality(
                object_scores,
                self.object_mode,
                forced_source=policy.get("object", None),
            ),
            "bg_global": self.collapse_background_score(
                background_scores,
                self.background_mode,
                source_neutral=source_neutral,
                forced_source=policy.get("bg_global", None),
            ),
            "bg_partials": self.collapse_background_partials_score(
                background_scores,
                source_neutral=source_neutral,
                forced_source=policy.get("bg_partials", None),
            ) if self.bg_partials_enabled else None,
            "parts": self.collapse_parts_score(
                parts_scores,
                self.parts_mode,
                source_neutral=source_neutral,
                forced_source=policy.get("parts", None),
            ),
        }

    def build_quality_pack(self, scores: dict, *, object_proto_quality: float | None) -> dict:
        q_pack = (scores.get("_quality", {}) or {}) if isinstance(scores, dict) else {}
        q_meta = (q_pack.get("meta", {}) or {}) if isinstance(q_pack, dict) else {}
        q_obj = self.clamp01(q_pack.get("object", 1.0))
        q_bg = self.clamp01(q_pack.get("background", 1.0))
        q_parts = self.clamp01(q_pack.get("parts", 1.0))
        q_obj_proto = 1.0 if object_proto_quality is None else self.clamp01(object_proto_quality)
        q_obj_joint = float(math.sqrt(max(0.0, q_obj * q_obj_proto)))
        q_obj_density = self.ramp(
            q_meta.get("obj_patch_density", 1.0),
            self.obj_min_coverage_density,
            self.obj_full_coverage_density,
        )
        q_parts_density = self.ramp(
            q_meta.get("obj_patch_density", 1.0),
            self.parts_min_coverage_density,
            self.parts_full_coverage_density,
        )

        q_obj_eff = self.apply_quality_floor(q_obj_joint, self.q_floor_object) * float(q_obj_density)
        q_bg_eff = self.apply_quality_floor(q_bg, self.q_floor_background)
        q_parts_eff = self.apply_quality_floor(q_parts, self.q_floor_parts) * float(q_parts_density)

        return {
            "object": float(q_obj),
            "object_proto": float(q_obj_proto),
            "object_joint": float(q_obj_joint),
            "background": float(q_bg),
            "parts": float(q_parts),
            "object_density": float(q_obj_density),
            "parts_density": float(q_parts_density),
            "effective_object": float(q_obj_eff),
            "effective_background": float(q_bg_eff),
            "effective_parts": float(q_parts_eff),
            "weight_object": float(self.w_obj * q_obj_eff),
            "weight_bg_global": float(self.w_bg_global * q_bg_eff),
            "weight_bg_partials": float(self.w_bg_partials * q_bg_eff),
            "weight_parts": float(self.w_parts * q_parts_eff),
        }

    def build_comparable_quality_pack(self, scores: dict) -> dict:
        q_pack = (scores.get("_quality", {}) or {}) if isinstance(scores, dict) else {}
        q_meta = (q_pack.get("meta", {}) or {}) if isinstance(q_pack, dict) else {}
        q_obj = self.clamp01(q_pack.get("object", 1.0))
        q_bg = self.clamp01(q_pack.get("background", 1.0))
        q_parts = self.clamp01(q_pack.get("parts", 1.0))
        q_obj_density = self.ramp(
            q_meta.get("obj_patch_density", 1.0),
            self.obj_min_coverage_density,
            self.obj_full_coverage_density,
        )
        q_parts_density = self.ramp(
            q_meta.get("obj_patch_density", 1.0),
            self.parts_min_coverage_density,
            self.parts_full_coverage_density,
        )

        q_obj_eff = self.apply_quality_floor(q_obj, self.q_floor_object) * float(q_obj_density)
        q_bg_eff = self.apply_quality_floor(q_bg, self.q_floor_background)
        q_parts_eff = self.apply_quality_floor(q_parts, self.q_floor_parts) * float(q_parts_density)

        return {
            "object": float(q_obj),
            "object_proto": 1.0,
            "object_joint": float(q_obj),
            "background": float(q_bg),
            "parts": float(q_parts),
            "object_density": float(q_obj_density),
            "parts_density": float(q_parts_density),
            "effective_object": float(q_obj_eff),
            "effective_background": float(q_bg_eff),
            "effective_parts": float(q_parts_eff),
            "weight_object": float(self.w_obj * q_obj_eff),
            "weight_bg_global": float(self.w_bg_global * q_bg_eff),
            "weight_bg_partials": float(self.w_bg_partials * q_bg_eff),
            "weight_parts": float(self.w_parts * q_parts_eff),
        }

    def build_base_terms(self, *, collapsed: dict, quality: dict) -> list[tuple[str, float, float, float | None]]:
        return [
            ("object", self.w_obj, float(quality["weight_object"]), collapsed["object"]),
            ("bg_global", self.w_bg_global, float(quality["weight_bg_global"]), collapsed["bg_global"]),
            ("bg_partials", self.w_bg_partials, float(quality["weight_bg_partials"]), collapsed["bg_partials"]),
            ("parts", self.w_parts, float(quality["weight_parts"]), collapsed["parts"]),
        ]

    def combine_base_terms(self, terms: list[tuple[str, float, float, float | None]]) -> float:
        used = [(w_nom, w_eff, s) for _, w_nom, w_eff, s in terms if (w_nom > 0.0 and s is not None)]
        if not used:
            return 0.0

        sum_eff_w = float(sum(w_eff for _, w_eff, _ in used))
        raw = float(sum(w_eff * float(s) for _, w_eff, s in used))

        if not self.renormalize_missing:
            return float(raw)

        if sum_eff_w <= 1e-12:
            return 0.0

        return float(raw / sum_eff_w)

    def collapse_object_score(
        self,
        s_obj: dict,
        mode: str,
        *,
        source_neutral: bool = False,
        forced_source: str | None = None,
    ) -> float | None:
        if not s_obj:
            return None

        if mode == "global":
            return self.collapse_ws_value(
                s_obj.get("global", None),
                prefer_combined=not source_neutral,
                forced_source=forced_source,
            )
        if mode == "global_trimmed":
            return self.collapse_ws_value(
                s_obj.get("global_trimmed", None),
                prefer_combined=not source_neutral,
                forced_source=forced_source,
            )

        vals = [
            self.collapse_ws_value(v, prefer_combined=not source_neutral, forced_source=forced_source)
            for v in s_obj.values()
        ]
        vals = [float(v) for v in vals if v is not None]
        return float(max(vals)) if vals else None

    def collapse_object_proto_quality(self, s_obj: dict, mode: str, *, forced_source: str | None = None) -> float | None:
        if not s_obj:
            return None

        if mode == "global":
            return self.collapse_ws_proto_quality(s_obj.get("global", None), forced_source=forced_source)
        if mode == "global_trimmed":
            return self.collapse_ws_proto_quality(s_obj.get("global_trimmed", None), forced_source=forced_source)

        best_score = None
        best_q = None
        for v in s_obj.values():
            s = self.collapse_ws_value(v, forced_source=forced_source)
            if s is None:
                continue
            q = self.collapse_ws_proto_quality(v, forced_source=forced_source)
            if best_score is None or float(s) > float(best_score):
                best_score = float(s)
                best_q = q
        return None if best_q is None else float(best_q)

    def collapse_background_score(
        self,
        s_bg: dict,
        mode: str,
        *,
        source_neutral: bool = False,
        forced_source: str | None = None,
    ) -> float | None:
        if not s_bg:
            return None

        if mode == "inner":
            return self.collapse_ws_value(
                s_bg.get("inner", None),
                prefer_combined=not source_neutral,
                forced_source=forced_source,
            )
        if mode == "outer":
            return self.collapse_ws_value(
                s_bg.get("outer", None),
                prefer_combined=not source_neutral,
                forced_source=forced_source,
            )

        v = s_bg.get("combined", None)
        if v is not None:
            return self.collapse_ws_value(v, prefer_combined=not source_neutral, forced_source=forced_source)

        vals = []
        if s_bg.get("inner", None) is not None:
            vals.append(
                self.collapse_ws_value(
                    s_bg["inner"],
                    prefer_combined=not source_neutral,
                    forced_source=forced_source,
                )
            )
        if s_bg.get("outer", None) is not None:
            vals.append(
                self.collapse_ws_value(
                    s_bg["outer"],
                    prefer_combined=not source_neutral,
                    forced_source=forced_source,
                )
            )
        vals = [float(x) for x in vals if x is not None]
        return float(max(vals)) if vals else None

    def collapse_background_partials_score(
        self,
        s_bg: dict,
        *,
        source_neutral: bool = False,
        forced_source: str | None = None,
    ) -> float | None:
        if not s_bg:
            return None

        v = s_bg.get("partials", None)
        if v is not None:
            return self.collapse_ws_value(v, prefer_combined=not source_neutral, forced_source=forced_source)

        vals = []
        if s_bg.get("partials_inner", None) is not None:
            vals.append(
                self.collapse_ws_value(
                    s_bg["partials_inner"],
                    prefer_combined=not source_neutral,
                    forced_source=forced_source,
                )
            )
        if s_bg.get("partials_outer", None) is not None:
            vals.append(
                self.collapse_ws_value(
                    s_bg["partials_outer"],
                    prefer_combined=not source_neutral,
                    forced_source=forced_source,
                )
            )
        vals = [float(x) for x in vals if x is not None]
        return float(max(vals)) if vals else None

    def collapse_parts_score(
        self,
        s_parts: dict,
        mode: str,
        *,
        source_neutral: bool = False,
        forced_source: str | None = None,
    ) -> float | None:
        if not s_parts:
            return None

        if mode == "kmeans":
            return self.collapse_ws_value(
                s_parts.get("kmeans", None),
                prefer_combined=not source_neutral,
                forced_source=forced_source,
            )
        if mode == "attention":
            return self.collapse_ws_value(
                s_parts.get("attention", None),
                prefer_combined=not source_neutral,
                forced_source=forced_source,
            )

        vals = [
            self.collapse_ws_value(v, prefer_combined=not source_neutral, forced_source=forced_source)
            for v in s_parts.values()
        ]
        vals = [float(v) for v in vals if v is not None]
        return float(max(vals)) if vals else None

    def collapse_ws_value(
        self,
        v,
        *,
        prefer_combined: bool = True,
        forced_source: str | None = None,
    ) -> float | None:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, dict):
            if forced_source in {"work", "stable"}:
                chosen = v.get(str(forced_source), None)
                return None if chosen is None else float(chosen)
            c = v.get("combined", None)
            if prefer_combined and c is not None:
                return float(c)
            m = v.get("max", None)
            if m is not None:
                return float(m)
            w = v.get("work", None)
            s = v.get("stable", None)
            if w is None and s is None:
                return None
            if w is None:
                return float(s)
            if s is None:
                return float(w)
            return float(max(float(w), float(s)))
        return None

    def collapse_ws_proto_quality(self, v, *, forced_source: str | None = None) -> float | None:
        if v is None or not isinstance(v, dict):
            return None

        if forced_source in {"work", "stable"}:
            q = v.get(f"{str(forced_source)}_proto_q", None)
            return None if q is None else float(q)

        c = v.get("combined", None)
        if c is not None:
            q = v.get("combined_proto_q", None)
            return None if q is None else float(q)

        m = v.get("max", None)
        if m is not None:
            q = v.get("max_proto_q", None)
            return None if q is None else float(q)

        w = v.get("work", None)
        s = v.get("stable", None)
        if w is None and s is None:
            return None
        if w is None:
            q = v.get("stable_proto_q", None)
            return None if q is None else float(q)
        if s is None:
            q = v.get("work_proto_q", None)
            return None if q is None else float(q)

        if float(w) > float(s):
            q = v.get("work_proto_q", None)
            return None if q is None else float(q)
        q = v.get("stable_proto_q", None)
        return None if q is None else float(q)

    def has_source_for_term(self, scores: dict, term: str, source: str) -> bool:
        collapsed = self.build_collapsed_scores(scores or {}, source_policy={str(term): str(source)})
        return bool(collapsed.get(str(term), None) is not None)

    def clamp01(self, value) -> float:
        return float(max(0.0, min(1.0, float(value))))

    def ramp(self, x: float | int | None, x0: float, x1: float) -> float:
        if x is None:
            return 0.0
        xv = float(x)
        if x1 <= x0:
            return 1.0 if xv >= x1 else 0.0
        return float(max(0.0, min(1.0, (xv - x0) / (x1 - x0))))

    def compute_patch_density(
        self,
        *,
        effective_patches: float | int | None,
        n_patches: float | int | None,
    ) -> float:
        if effective_patches is None or n_patches is None:
            return 1.0
        den = float(n_patches)
        if den <= 1e-12:
            return 1.0
        return self.clamp01(float(effective_patches) / den)

    def apply_quality_floor(self, quality: float, floor: float) -> float:
        q = self.clamp01(quality)
        f = self.clamp01(floor)
        return float(f + (1.0 - f) * q)
