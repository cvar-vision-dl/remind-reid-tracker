from __future__ import annotations

from association.reports import SimilarityReport


class CandidateGenerator:
    """Construye reports de similitud por deteccion sin aplicar policy contextual."""

    def __init__(self, *, scores, combiner, memory_store):
        self.scores = scores
        self.combiner = combiner
        self.memory_store = memory_store

    def process_one_detection(
        self,
        det_id: int,
        detection,
        det_feats: dict | None,
        frame_context=None,
    ) -> SimilarityReport:
        ts = self.resolve_timestamp(detection, frame_context)
        rep = SimilarityReport(det_id=det_id, class_id=detection.class_id, timestamp=ts)

        geom = getattr(detection, "geom", None)
        if not isinstance(geom, dict):
            raise RuntimeError(f"Detection {det_id} no tiene geom (dict) en det.geom.")

        c = geom.get("center", None)
        a = geom.get("area", None)
        if c is None or a is None or not isinstance(c, (tuple, list)) or len(c) != 2:
            raise RuntimeError(f"Detection {det_id} geom invalida: {geom}")

        rep.det_geom = {"center": (float(c[0]), float(c[1])), "area": float(a)}

        if det_feats is None:
            return rep

        candidates = self.memory_store.get_by_class(detection.class_id)

        for obj in candidates:
            rep.candidates.append(
                self.build_similarity_candidate(
                    det_feats=det_feats,
                    tracked_object=obj,
                    defer_similarity_pack=True,
                )
            )

        self.apply_consistent_similarity_policy(rep.candidates)
        rep.best = self.pick_best(rep.candidates, key="score_sim")
        return rep

    def build_similarity_candidate(
        self,
        *,
        det_feats: dict,
        tracked_object,
        defer_similarity_pack: bool = False,
    ) -> dict:
        scores_sim = self.scores.compute_base(det_feats, tracked_object)

        candidate = {
            "object_id": int(tracked_object.object_id),
            "class_id": int(tracked_object.class_id),
            "score_sim": 0.0,
            "score_sets": 0.0,
            "bonus_sets_raw": 0.0,
            "bonus_sets": 0.0,
            "penalty_sets": 0.0,
            "support_sets": 0.0,
            "support_local_sets": 0.0,
            "support_global_sets": 0.0,
            "quality_sets": 0.0,
            "score_ctx_local": 0.0,
            "score_ctx_global": 0.0,
            "compat_rel": 0.0,
            "compat_band": 0,
            "kernel_raw": 0.0,
            "kernel_rel": 0.0,
            "hyp_rel": 0.0,
            "score_final": 0.0,
            "score_known": 0.0,
            "score_sim_base": 0.0,
            "score_obj_collapsed": None,
            "score_bg_collapsed": None,
            "score_bgp_collapsed": None,
            "score_parts_collapsed": None,
            "quality_obj": 0.0,
            "quality_obj_proto": 0.0,
            "quality_obj_joint": 0.0,
            "quality_bg": 0.0,
            "quality_parts": 0.0,
            "quality_eff_obj": 0.0,
            "quality_eff_bg": 0.0,
            "quality_eff_parts": 0.0,
            "weight_eff_obj": 0.0,
            "weight_eff_bg": 0.0,
            "weight_eff_bgp": 0.0,
            "weight_eff_parts": 0.0,
            "score_source_policy": {},
            "scores": scores_sim,
        }
        if not bool(defer_similarity_pack):
            sim_pack = self.combiner.combine_pack(scores_sim, det_feats=det_feats)
            self.apply_similarity_pack(candidate=candidate, sim_pack=sim_pack)
        return candidate

    def apply_similarity_pack(self, *, candidate: dict, sim_pack: dict | None) -> None:
        sim_pack = sim_pack if isinstance(sim_pack, dict) else {}
        total_sim = float((sim_pack.get("core", {}) or {}).get("score_sim", 0.0))
        sim_dbg = (sim_pack.get("debug", {}) or {})
        sim_collapsed = (sim_dbg.get("collapsed", {}) or {})
        sim_quality = (sim_dbg.get("quality", {}) or {})
        sim_quality_eff = (sim_dbg.get("quality_effective", {}) or {})
        sim_weights = (sim_dbg.get("effective_weights", {}) or {})

        candidate["score_sim"] = float(total_sim)
        candidate["score_final"] = float(total_sim)
        candidate["score_known"] = float(total_sim)
        candidate["score_sim_base"] = float(sim_dbg.get("base", 0.0) or 0.0)
        candidate["score_obj_collapsed"] = sim_collapsed.get("object", None)
        candidate["score_bg_collapsed"] = sim_collapsed.get("bg_global", None)
        candidate["score_bgp_collapsed"] = sim_collapsed.get("bg_partials", None)
        candidate["score_parts_collapsed"] = sim_collapsed.get("parts", None)
        candidate["quality_obj"] = float(sim_quality.get("object", 0.0) or 0.0)
        candidate["quality_obj_proto"] = float(sim_quality.get("object_proto", 0.0) or 0.0)
        candidate["quality_obj_joint"] = float(sim_quality.get("object_joint", 0.0) or 0.0)
        candidate["quality_bg"] = float(sim_quality.get("background", 0.0) or 0.0)
        candidate["quality_parts"] = float(sim_quality.get("parts", 0.0) or 0.0)
        candidate["quality_eff_obj"] = float(sim_quality_eff.get("object", 0.0) or 0.0)
        candidate["quality_eff_bg"] = float(sim_quality_eff.get("background", 0.0) or 0.0)
        candidate["quality_eff_parts"] = float(sim_quality_eff.get("parts", 0.0) or 0.0)
        candidate["weight_eff_obj"] = float(sim_weights.get("object", 0.0) or 0.0)
        candidate["weight_eff_bg"] = float(sim_weights.get("bg_global", 0.0) or 0.0)
        candidate["weight_eff_bgp"] = float(sim_weights.get("bg_partials", 0.0) or 0.0)
        candidate["weight_eff_parts"] = float(sim_weights.get("parts", 0.0) or 0.0)

    def apply_consistent_similarity_policy(self, candidates: list[dict]) -> None:
        if not candidates:
            return

        policy = self.resolve_similarity_source_policy(candidates)
        for candidate in candidates:
            scores_sim = candidate.get("scores", {}) if isinstance(candidate.get("scores", None), dict) else {}
            sim_pack = self.combiner.combine_consistent_pack(scores_sim, source_policy=policy)
            self.apply_similarity_pack(candidate=candidate, sim_pack=sim_pack)
            candidate["score_source_policy"] = dict(policy)

    def resolve_similarity_source_policy(self, candidates: list[dict]) -> dict[str, str]:
        policy: dict[str, str] = {}
        terms = ("object", "bg_global", "bg_partials", "parts")
        for term in terms:
            if all(self.combiner.has_source_for_term(c.get("scores", {}) or {}, term, "stable") for c in candidates):
                policy[str(term)] = "stable"
            elif all(self.combiner.has_source_for_term(c.get("scores", {}) or {}, term, "work") for c in candidates):
                policy[str(term)] = "work"
            else:
                policy[str(term)] = "auto"
        return policy

    def pick_best(self, candidates: list[dict], key: str) -> dict | None:
        if not candidates:
            return None
        best = candidates[0]
        best_s = float(best.get(key, 0.0))
        for c in candidates[1:]:
            s = float(c.get(key, 0.0))
            if s > best_s:
                best_s = s
                best = c
        return best

    def pick_second_best(self, candidates: list[dict], best: dict, key: str) -> dict | None:
        best_id = int(best.get("object_id", -1))
        second = None
        second_s = -1e18
        for c in candidates:
            if int(c.get("object_id", -1)) == best_id:
                continue
            s = float(c.get(key, 0.0))
            if s > second_s:
                second_s = s
                second = c
        return second

    def resolve_timestamp(self, detection, frame_context) -> float:
        if frame_context is not None and hasattr(frame_context, "timestamp"):
            return float(frame_context.timestamp)
        if hasattr(detection, "timestamp"):
            return float(detection.timestamp)
        return 0.0
