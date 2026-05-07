from __future__ import annotations

import itertools
import math


class AssignmentScoringMixin:
    def solve_component(
        self,
        *,
        component: dict,
        reports_by_det_id: dict,
        det_geom_by_id: dict[int, dict],
        anchor_ids: list[int],
        anchor_geom_by_oid: dict[int, dict],
    ) -> dict | None:
        det_ids = [int(x) for x in (component.get("det_ids", []) or [])]
        candidates_by_det = {
            int(k): [int(x) for x in (v or [])]
            for k, v in ((component.get("candidates_by_det", {}) or {}).items())
        }
        policy_mode = self.component_policy_mode(
            det_ids=det_ids,
            candidates_by_det=candidates_by_det,
        )
        if str(policy_mode) != "full_sibling":
            return None

        anchor_quality_by_id = self.build_component_anchor_quality_map(
            component=component,
            anchor_ids=anchor_ids,
            det_geom_by_id=det_geom_by_id,
            anchor_geom_by_oid=anchor_geom_by_oid,
        )

        anchor_order_by_det = self.build_anchor_order_by_det(
            det_ids=det_ids,
            anchor_ids=anchor_ids,
            det_geom_by_id=det_geom_by_id,
            anchor_geom_by_oid=anchor_geom_by_oid,
        )

        solutions = []
        for assignment in self.enumerate_assignments(det_ids=det_ids, candidates_by_det=candidates_by_det):
            score = self.score_assignment(
                assignment=assignment,
                det_ids=det_ids,
                candidates_by_det=candidates_by_det,
                anchor_ids=anchor_ids,
                reports_by_det_id=reports_by_det_id,
                det_geom_by_id=det_geom_by_id,
                anchor_order_by_det=anchor_order_by_det,
                anchor_geom_by_oid=anchor_geom_by_oid,
                anchor_quality_by_id=anchor_quality_by_id,
                include_breakdown=False,
            )
            if score is None:
                continue
            solutions.append(
                {
                    "assignment": dict(assignment),
                    "best_score": float(score["score"]),
                    "core_score": float(score.get("core_score", 0.0)),
                    "evidence": float(score["evidence"]),
                    "anchor_term": float(score.get("anchor_term", 0.0)),
                    "history_term": float(score.get("history_term", 0.0)),
                    "frame_term": float(score.get("frame_term", 0.0)),
                    "anchor_quality_term": float(score.get("anchor_quality_term", 0.0)),
                    "order_term": float(score.get("order_term", 0.0)),
                    "peer_term": float(score.get("peer_term", 0.0)),
                    "sibling_term": float(score.get("sibling_term", 0.0)),
                    "visual_term": float(score.get("visual_term", 0.0)),
                    "known_assignments": int(score.get("scored_known_assignments", 0) or 0),
                }
            )

        if not solutions:
            return None

        solutions.sort(key=lambda item: float(item["best_score"]), reverse=True)
        best = solutions[0]
        second_score = float(solutions[1]["best_score"]) if len(solutions) > 1 else -1.0
        core_ranked = sorted(solutions, key=lambda item: float(item.get("core_score", 0.0)), reverse=True)
        core_best = core_ranked[0]
        core_second_score = float(core_ranked[1].get("core_score", 0.0)) if len(core_ranked) > 1 else -1.0
        competitive = self.competitive_frontier(
            solutions=solutions,
            best_score=float(best["best_score"]),
            second_score=float(second_score),
        )
        detailed_top_solutions = []
        for item in solutions[:3]:
            assignment = {
                int(det_id): int(oid)
                for det_id, oid in ((item.get("assignment", {}) or {}).items())
                if det_id is not None and oid is not None
            }
            detailed_score = self.score_assignment(
                assignment=assignment,
                det_ids=det_ids,
                candidates_by_det=candidates_by_det,
                anchor_ids=anchor_ids,
                reports_by_det_id=reports_by_det_id,
                det_geom_by_id=det_geom_by_id,
                anchor_order_by_det=anchor_order_by_det,
                anchor_geom_by_oid=anchor_geom_by_oid,
                anchor_quality_by_id=anchor_quality_by_id,
                include_breakdown=True,
            )
            if detailed_score is None:
                detailed_score = {}
            detailed_top_solutions.append(
                {
                    "assignment": dict(assignment),
                    "score": float(item.get("best_score", 0.0)),
                    "core_score": float(item.get("core_score", 0.0)),
                    "evidence": float(item.get("evidence", 0.0)),
                    "anchor_term": float(item.get("anchor_term", 0.0)),
                    "history_term": float(item.get("history_term", 0.0)),
                    "frame_term": float(item.get("frame_term", 0.0)),
                    "anchor_quality_term": float(item.get("anchor_quality_term", 0.0)),
                    "order_term": float(item.get("order_term", 0.0)),
                    "peer_term": float(item.get("peer_term", 0.0)),
                    "sibling_term": float(item.get("sibling_term", 0.0)),
                    "visual_term": float(item.get("visual_term", 0.0)),
                    "known_assignments": int(item.get("known_assignments", 0) or 0),
                    "anchor_breakdown": list(detailed_score.get("anchor_breakdown", []) or []),
                }
            )
        return {
            "assignment": dict(best["assignment"]),
            "best_score": float(best["best_score"]),
            "second_score": float(second_score),
            "gap": float(best["best_score"] - second_score) if second_score >= 0.0 else float(best["best_score"]),
            "core_score": float(best.get("core_score", 0.0)),
            "core_second_score": float(core_second_score),
            "core_gap": float(best.get("core_score", 0.0) - core_second_score) if core_second_score >= 0.0 else float(best.get("core_score", 0.0)),
            "core_rank_status": "match" if dict(best.get("assignment", {}) or {}) == dict(core_best.get("assignment", {}) or {}) else "visual_rank_flip",
            "policy_mode": str(policy_mode),
            "evidence": float(best["evidence"]),
            "anchor_term": float(best.get("anchor_term", 0.0)),
            "history_term": float(best.get("history_term", 0.0)),
            "frame_term": float(best.get("frame_term", 0.0)),
            "anchor_quality_term": float(best.get("anchor_quality_term", 0.0)),
            "order_term": float(best.get("order_term", 0.0)),
            "peer_term": float(best.get("peer_term", 0.0)),
            "sibling_term": float(best.get("sibling_term", 0.0)),
            "visual_term": float(best.get("visual_term", 0.0)),
            "known_assignments": int(best.get("known_assignments", 0) or 0),
            "top_solutions": list(detailed_top_solutions),
            "frontier_size": int(len(competitive)),
            "stable_det_assignments": self.stable_det_assignments(competitive),
        }

    def build_complete_sibling_groups(
        self,
        *,
        det_ids: list[int],
        candidates_by_det: dict[int, list[int]],
    ) -> list[dict]:
        groups = {}
        for det_id in det_ids or []:
            real_candidate_ids = []
            class_ids = set()
            for oid in (candidates_by_det.get(int(det_id), []) or []):
                obj = None if self.memory_store is None else self.memory_store.get(int(oid))
                if obj is None:
                    continue
                real_candidate_ids.append(int(oid))
                class_ids.add(int(getattr(obj, "class_id", -1)))
            if len(class_ids) != 1:
                continue

            class_id = next(iter(class_ids))
            pack = groups.setdefault(
                int(class_id),
                {"class_id": int(class_id), "det_ids": [], "candidate_ids": set()},
            )
            pack["det_ids"].append(int(det_id))
            pack["candidate_ids"].update(int(x) for x in real_candidate_ids)

        out = []
        for pack in groups.values():
            group_det_ids = sorted(set(int(x) for x in (pack.get("det_ids", []) or [])))
            group_candidate_ids = sorted(set(int(x) for x in (pack.get("candidate_ids", set()) or [])))
            if len(group_det_ids) < 2:
                continue
            if len(group_candidate_ids) != len(group_det_ids):
                continue
            out.append(
                {
                    "class_id": int(pack.get("class_id", -1)),
                    "det_ids": list(group_det_ids),
                    "candidate_ids": list(group_candidate_ids),
                }
            )
        return out

    def component_policy_mode(
        self,
        *,
        det_ids: list[int],
        candidates_by_det: dict[int, list[int]],
    ) -> str:
        complete_groups = self.build_complete_sibling_groups(
            det_ids=det_ids,
            candidates_by_det=candidates_by_det,
        )
        if complete_groups:
            covered = set()
            for group in complete_groups:
                covered.update(int(x) for x in (group.get("det_ids", []) or []))
            if covered == set(int(x) for x in (det_ids or [])):
                return "full_sibling"

        same_class_groups = {}
        for det_id in det_ids or []:
            real_candidate_ids = []
            class_ids = set()
            for oid in (candidates_by_det.get(int(det_id), []) or []):
                obj = None if self.memory_store is None else self.memory_store.get(int(oid))
                if obj is None:
                    continue
                real_candidate_ids.append(int(oid))
                class_ids.add(int(getattr(obj, "class_id", -1)))
            if len(class_ids) != 1 or len(real_candidate_ids) < 2:
                continue
            class_id = next(iter(class_ids))
            pack = same_class_groups.setdefault(
                int(class_id),
                {"det_ids": set(), "candidate_ids": set()},
            )
            pack["det_ids"].add(int(det_id))
            pack["candidate_ids"].update(int(x) for x in real_candidate_ids)

        if not same_class_groups:
            return "singleton" if len(det_ids or []) <= 1 else "generic"

        for pack in same_class_groups.values():
            det_count = len(pack.get("det_ids", set()) or set())
            cand_count = len(pack.get("candidate_ids", set()) or set())
            if cand_count > det_count:
                return "partial_same_class"

        return "singleton_same_class" if len(det_ids or []) <= 1 else "generic"

    def policy_core_thresholds(self, *, policy_mode: str, det_count: int) -> tuple[float, float]:
        mode = str(policy_mode or "")
        if mode in ("full_sibling", "coupled_full_sibling"):
            return float(self.min_full_sibling_core_assignment_score), float(self.min_full_sibling_core_gap)
        if mode in ("partial_same_class", "singleton_same_class"):
            return float(self.min_partial_same_class_core_assignment_score), float(self.min_partial_same_class_core_gap)
        if int(det_count) <= 1 or mode == "singleton":
            return float(self.min_singleton_core_assignment_score), float(self.min_singleton_core_gap)
        return float(self.min_core_assignment_score), float(self.min_core_gap)

    def enumerate_assignments(self, *, det_ids: list[int], candidates_by_det: dict[int, list[int]]):
        ordered = sorted(det_ids, key=lambda did: len(candidates_by_det.get(int(did), [])))
        used = set()
        cur = {}

        def backtrack(idx: int):
            if idx >= len(ordered):
                yield dict(cur)
                return

            det_id = int(ordered[idx])
            for oid in candidates_by_det.get(int(det_id), []):
                if int(oid) in used:
                    continue
                used.add(int(oid))
                cur[int(det_id)] = int(oid)
                yield from backtrack(idx + 1)
                cur.pop(int(det_id), None)
                used.remove(int(oid))

        yield from backtrack(0)

    def build_anchor_order_by_det(
        self,
        *,
        det_ids: list[int],
        anchor_ids: list[int],
        det_geom_by_id: dict[int, dict],
        anchor_geom_by_oid: dict[int, dict],
    ) -> dict[int, list[int]]:
        out = {}
        for det_id in det_ids:
            det_geom = det_geom_by_id.get(int(det_id), None)
            if not isinstance(det_geom, dict):
                continue
            ranked = []
            for anchor_id in anchor_ids or []:
                anchor_geom = anchor_geom_by_oid.get(int(anchor_id), None)
                if not isinstance(anchor_geom, dict):
                    continue
                obs = self.relation_observation_cached(
                    det_geom,
                    anchor_geom,
                    scale_min=40.0,
                    geom_a_key=("det", int(det_id)),
                    geom_b_key=("anchor", int(anchor_id)),
                )
                if not isinstance(obs, dict):
                    continue
                ranked.append((self.primary_observed_distance(obs), int(anchor_id)))
            ranked.sort(key=lambda kv: (float(kv[0]), int(kv[1])))
            out[int(det_id)] = [int(anchor_id) for _, anchor_id in ranked]
        return out

    def score_assignment(
        self,
        *,
        assignment: dict[int, int],
        det_ids: list[int],
        candidates_by_det: dict[int, list[int]],
        anchor_ids: list[int],
        reports_by_det_id: dict,
        det_geom_by_id: dict[int, dict],
        anchor_order_by_det: dict[int, list[int]],
        anchor_geom_by_oid: dict[int, dict],
        anchor_quality_by_id: dict[int, dict] | None = None,
        include_breakdown: bool = True,
    ) -> dict | None:
        anchor_scores = []
        anchor_weights = []
        history_terms = []
        frame_terms = []
        quality_terms = []
        order_terms = []
        anchor_breakdown = [] if include_breakdown else None
        visual_scores = []
        scored_known_assignments = 0

        for det_id in det_ids:
            oid = assignment.get(int(det_id), None)
            rep = reports_by_det_id.get(int(det_id), None)
            obj = None if oid is None or self.memory_store is None else self.memory_store.get(int(oid))
            if oid is not None and obj is not None:
                visual_scores.append(float(self.resolve_candidate_score(rep, int(oid))))
                scored_known_assignments += 1
        del candidates_by_det
        del anchor_order_by_det
        anchor_quality_by_id = dict(anchor_quality_by_id or {})
        for anchor_id in (anchor_ids or []):
            anchor_id = int(anchor_id)
            anchor_geom = anchor_geom_by_oid.get(int(anchor_id), None)
            if not isinstance(anchor_geom, dict):
                continue
            quality_pack = dict(anchor_quality_by_id.get(int(anchor_id), {}) or {})
            anchor_quality = float(quality_pack.get("combined_usefulness", 0.0) or 0.0)
            if anchor_quality <= 1e-12:
                continue
            per_anchor_scores = []
            per_anchor_weights = []
            for det_id in det_ids:
                oid = assignment.get(int(det_id), None)
                if oid is None:
                    continue
                det_geom = det_geom_by_id.get(int(det_id), None)
                if not isinstance(det_geom, dict):
                    continue
                obj = None if self.memory_store is None else self.memory_store.get(int(oid))
                dg = getattr(obj, "neighbor_dist", None) if obj is not None else None
                if dg is None:
                    continue
                edge = dg.get_edge(int(anchor_id))
                if edge is None:
                    continue
                obs = self.relation_observation_cached(
                    det_geom,
                    anchor_geom,
                    scale_min=float(getattr(dg, "scale_min", 40.0)),
                    contact_margin_px=float(getattr(dg, "contact_margin_px", 2.0)),
                    near_thresh_n=float(getattr(dg, "near_thresh_n", 1.25)),
                    exact_gap_max_n=float(getattr(dg, "exact_gap_max_n", 1.75)),
                    geom_a_key=("det", int(det_id)),
                    geom_b_key=("anchor", int(anchor_id)),
                )
                score, weight = self.relation_similarity(obs=obs, edge=edge)
                if float(weight) <= 0.0:
                    continue
                per_anchor_scores.append(float(score))
                per_anchor_weights.append(float(weight))
            if not per_anchor_weights:
                continue
            order_pack = self.anchor_assignment_pair_order_score(
                det_ids=det_ids,
                assignment=assignment,
                anchor_id=int(anchor_id),
                det_geom_by_id=det_geom_by_id,
                anchor_geom_by_oid=anchor_geom_by_oid,
                include_detail=bool(include_breakdown),
            )
            pair_fit = float(order_pack.get("score", 0.0) or 0.0)
            pair_evidence = float(order_pack.get("evidence", 0.0) or 0.0)
            if float(pair_evidence) <= 1e-12:
                continue
            assignment_match = float(pair_fit)
            assignment_evidence = float(pair_evidence)
            anchor_evidence = float(anchor_quality * assignment_evidence)
            if anchor_evidence <= 1e-12:
                continue
            anchor_scores.append(float(assignment_match))
            anchor_weights.append(float(anchor_evidence))
            history_terms.append(float(quality_pack.get("history_usefulness", 0.0) or 0.0))
            frame_terms.append(float(quality_pack.get("frame_strength", 0.0) or 0.0))
            quality_terms.append(float(anchor_quality))
            order_terms.append(float(pair_fit))
            if include_breakdown and isinstance(anchor_breakdown, list):
                anchor_breakdown.append(
                    {
                        "anchor_id": int(anchor_id),
                        "history_usefulness": float(quality_pack.get("history_usefulness", 0.0) or 0.0),
                        "frame_strength": float(quality_pack.get("frame_strength", 0.0) or 0.0),
                        "anchor_quality": float(anchor_quality),
                        "order_fit": float(pair_fit),
                        "order_evidence": float(pair_evidence),
                        "assignment_match": float(assignment_match),
                        "assignment_evidence": float(assignment_evidence),
                        "anchor_evidence": float(anchor_evidence),
                        "pair_gap": quality_pack.get("pair_gap", None),
                        "order_detail": dict(order_pack.get("detail", {}) or {}),
                    }
                )

        anchor_term = self.weighted_average(anchor_scores, anchor_weights)
        history_term = self.weighted_average(history_terms, anchor_weights)
        frame_term = self.weighted_average(frame_terms, anchor_weights)
        anchor_quality_term = self.weighted_average(quality_terms, anchor_weights)
        order_term = self.weighted_average(order_terms, anchor_weights)
        peer_term = 0.0
        sibling_term = 0.0
        visual_term = float(sum(visual_scores) / float(max(1, len(visual_scores)))) if visual_scores else 0.0
        evidence = float(sum(anchor_weights))
        total_weight = 0.0
        weighted_sum = 0.0
        if anchor_weights:
            anchor_core = float(anchor_term * anchor_quality_term)
            total_weight += float(self.anchor_weight)
            weighted_sum += float(self.anchor_weight) * float(anchor_core)
        if visual_scores:
            total_weight += float(self.visual_weight)
            weighted_sum += float(self.visual_weight) * float(visual_term)
        if total_weight <= 1e-12:
            return None

        score = float(weighted_sum / total_weight)
        core_score = float(anchor_term * anchor_quality_term) if anchor_weights else 0.0
        return {
            "score": float(score),
            "core_score": float(core_score),
            "evidence": float(evidence),
            "anchor_term": float(anchor_term),
            "history_term": float(history_term),
            "frame_term": float(frame_term),
            "anchor_quality_term": float(anchor_quality_term),
            "order_term": float(order_term),
            "peer_term": float(peer_term),
            "sibling_term": float(sibling_term),
            "visual_term": float(visual_term),
            "scored_known_assignments": int(scored_known_assignments),
            "anchor_breakdown": list(anchor_breakdown) if isinstance(anchor_breakdown, list) else [],
        }

    def anchor_assignment_pair_order_score(
        self,
        *,
        det_ids: list[int],
        assignment: dict[int, int],
        anchor_id: int,
        det_geom_by_id: dict[int, dict],
        anchor_geom_by_oid: dict[int, dict],
        include_detail: bool = True,
    ) -> dict:
        anchor_geom = anchor_geom_by_oid.get(int(anchor_id), None)
        if not isinstance(anchor_geom, dict):
            return {"score": 0.0, "evidence": 0.0, "detail": {}}

        observed_dist_by_det = {}
        expected_dist_by_det = {}
        reliability_by_det = {}
        for det_id in (det_ids or []):
            oid = assignment.get(int(det_id), None)
            if oid is None:
                continue
            obj = None if self.memory_store is None else self.memory_store.get(int(oid))
            dg = getattr(obj, "neighbor_dist", None) if obj is not None else None
            det_geom = det_geom_by_id.get(int(det_id), None)
            if dg is None or not isinstance(det_geom, dict):
                continue
            edge = dg.get_edge(int(anchor_id))
            if edge is None:
                continue
            reliability = float(edge.reliability())
            if reliability < float(self.min_edge_reliability):
                continue

            obs = self.relation_observation_cached(
                det_geom,
                anchor_geom,
                scale_min=float(getattr(dg, "scale_min", 40.0)),
                contact_margin_px=float(getattr(dg, "contact_margin_px", 2.0)),
                near_thresh_n=float(getattr(dg, "near_thresh_n", 1.25)),
                exact_gap_max_n=float(getattr(dg, "exact_gap_max_n", 1.75)),
                geom_a_key=("det", int(det_id)),
                geom_b_key=("anchor", int(anchor_id)),
            )
            obs_dist = self.primary_observed_distance(obs)
            exp_dist = edge.primary_distance()
            if not math.isfinite(float(obs_dist)) or exp_dist is None or not math.isfinite(float(exp_dist)):
                continue

            observed_dist_by_det[int(det_id)] = float(obs_dist)
            expected_dist_by_det[int(det_id)] = float(exp_dist)
            reliability_by_det[int(det_id)] = float(reliability)

        if len(observed_dist_by_det) < 2:
            return {"score": 0.0, "evidence": 0.0, "detail": {}}

        pair_scores = []
        pair_weights = []
        pair_details = [] if include_detail else None
        for det_i, det_j in itertools.combinations([int(x) for x in (det_ids or [])], 2):
            obs_i = float(observed_dist_by_det.get(int(det_i), float("inf")))
            obs_j = float(observed_dist_by_det.get(int(det_j), float("inf")))
            exp_i = float(expected_dist_by_det.get(int(det_i), float("inf")))
            exp_j = float(expected_dist_by_det.get(int(det_j), float("inf")))
            if not all(math.isfinite(x) for x in (obs_i, obs_j, exp_i, exp_j)):
                continue

            obs_sep = abs(float(obs_i) - float(obs_j))
            exp_sep = abs(float(exp_i) - float(exp_j))
            sep_strength = float(
                min(
                    1.0,
                    max(float(obs_sep), float(exp_sep)) / max(1e-6, float(self.anchor_span_ref)),
                )
            )
            if sep_strength <= 1e-12:
                continue

            obs_i_closer = bool(float(obs_i) < float(obs_j))
            exp_i_closer = bool(float(exp_i) < float(exp_j))
            pair_score = 1.0 if obs_i_closer == exp_i_closer else 0.0
            order_prob = None
            margin_sim = None
            robustness = None
            oid_i = assignment.get(int(det_i), None)
            oid_j = assignment.get(int(det_j), None)
            if oid_i is not None and oid_j is not None:
                closer_oid = int(oid_i) if bool(obs_i_closer) else int(oid_j)
                farther_oid = int(oid_j) if bool(obs_i_closer) else int(oid_i)
                order_prob = self.pair_anchor_discriminator.pair_order_probability(
                    anchor_id=int(anchor_id),
                    closer_oid=int(closer_oid),
                    farther_oid=int(farther_oid),
                )
                order_stats = self.pair_anchor_discriminator.pair_order_stats(
                    anchor_id=int(anchor_id),
                    oid_a=int(oid_i),
                    oid_b=int(oid_j),
                )
                if order_prob is not None and isinstance(order_stats, dict):
                    margin_ref = float(order_stats.get("mean_margin", 0.0) or 0.0)
                    margin_sigma = max(1e-6, float(self.order_margin_ref))
                    margin_sim = float(math.exp(-abs(float(obs_sep) - float(margin_ref)) / margin_sigma))
                    robustness = float(order_stats.get("robustness", 0.0) or 0.0)
                    pair_score = float(
                        ((1.0 - float(self.anchor_pair_order_weight)) * float(pair_score))
                        + (float(self.anchor_pair_order_weight) * float(order_prob))
                    )
                    pair_score = float(
                        ((1.0 - float(self.anchor_pair_margin_weight)) * float(pair_score))
                        + (float(self.anchor_pair_margin_weight) * float(margin_sim))
                    )
                    sep_strength *= float(max(0.10, robustness))
            pair_rel = 0.5 * (
                float(reliability_by_det.get(int(det_i), 0.0))
                + float(reliability_by_det.get(int(det_j), 0.0))
            )
            pair_weight = float(pair_rel * sep_strength)
            if pair_weight <= 1e-12:
                continue
            pair_scores.append(float(pair_score))
            pair_weights.append(float(pair_weight))
            if include_detail and isinstance(pair_details, list):
                pair_details.append(
                    {
                        "det_i": int(det_i),
                        "det_j": int(det_j),
                        "oid_i": int(oid_i) if oid_i is not None else None,
                        "oid_j": int(oid_j) if oid_j is not None else None,
                        "obs_i": float(obs_i),
                        "obs_j": float(obs_j),
                        "obs_sep": float(obs_sep),
                        "obs_closer_det": int(det_i) if bool(obs_i_closer) else int(det_j),
                        "closer_oid": int(closer_oid) if oid_i is not None and oid_j is not None else None,
                        "farther_oid": int(farther_oid) if oid_i is not None and oid_j is not None else None,
                        "exp_i": float(exp_i),
                        "exp_j": float(exp_j),
                        "exp_closer_oid": int(oid_i) if bool(exp_i_closer) and oid_i is not None else (int(oid_j) if oid_j is not None else None),
                        "order_prob": order_prob,
                        "margin_sim": margin_sim,
                        "robustness": robustness,
                        "pair_score": float(pair_score),
                        "pair_weight": float(pair_weight),
                    }
                )

        if not pair_weights:
            return {"score": 0.0, "evidence": 0.0, "detail": {}}
        best_detail = max(pair_details, key=lambda item: float(item.get("pair_weight", 0.0))) if isinstance(pair_details, list) and pair_details else {}
        return {
            "score": float(self.weighted_average(pair_scores, pair_weights)),
            "evidence": float(sum(pair_weights)),
            "detail": dict(best_detail),
        }

    def relation_similarity(self, *, obs: dict | None, edge) -> tuple[float, float]:
        if edge is None or not isinstance(obs, dict):
            return 0.0, 0.0
        reliability = float(edge.reliability())
        if reliability < float(self.min_edge_reliability):
            return 0.0, 0.0

        primary_expected = edge.primary_distance()
        primary_observed = self.primary_observed_distance(obs)
        if primary_expected is None or not math.isfinite(float(primary_observed)):
            return 0.0, 0.0

        gap_like = edge.mean_gap()
        sigma = float(self.gap_sigma) if gap_like is not None else float(self.center_sigma)
        sim_dist = math.exp(-abs(float(primary_observed) - float(primary_expected)) / sigma)

        sim_contact = float(edge.contact_probability(str(obs.get("contact_state", "separate"))))
        support_diff = abs(float(obs.get("support_like", 0.0)) - float(edge.mean_support_like()))
        sim_support = float(max(0.0, 1.0 - support_diff))

        score = float(sim_dist)

        gap_quality = float(max(0.0, min(1.0, obs.get("gap_quality", 0.0))))
        truncation_risk = float(max(0.0, min(1.0, obs.get("truncation_risk", 0.0))))
        obs_quality = float(
            (1.0 - float(self.obs_gap_quality_weight))
            + (float(self.obs_gap_quality_weight) * float(gap_quality))
        )
        obs_quality *= float(max(0.10, 1.0 - (float(self.obs_truncation_penalty) * float(truncation_risk))))

        informativeness = float(
            max(
                0.10,
                float(edge.informativeness()) * max(0.10, 1.0 - float(self.support_penalty) * float(obs.get("support_like", 0.0))),
            )
        )
        consistency = float(0.5 + (0.25 * float(sim_contact)) + (0.25 * float(sim_support)))
        weight = float(reliability * informativeness * consistency * obs_quality)
        return float(score), float(weight)

    def rank_similarity(self, *, observed_rank: int | None, expected_rank: float | None) -> float | None:
        if observed_rank is None or expected_rank is None:
            return None
        if not math.isfinite(float(expected_rank)) or float(expected_rank) <= 0.0:
            return None
        return float(math.exp(-abs(float(observed_rank) - float(expected_rank)) / float(self.rank_sigma)))

    def primary_observed_distance(self, obs: dict) -> float:
        if bool(obs.get("gap_valid", False)):
            return float(obs.get("mask_gap_n", float("inf")))
        return float(obs.get("center_dist_n", float("inf")))

    def weighted_average(self, values: list[float], weights: list[float]) -> float:
        if not values or not weights:
            return 0.0
        denom = float(sum(float(w) for w in weights))
        if denom <= 1e-12:
            return 0.0
        return float(sum(float(v) * float(w) for v, w in zip(values, weights)) / denom)

    def resolve_candidate_score(self, rep, oid: int) -> float:
        if rep is None:
            return 0.0
        for candidate in (getattr(rep, "candidates", None) or []):
            if not isinstance(candidate, dict):
                continue
            if int(candidate.get("object_id", -1)) != int(oid):
                continue
            return float(candidate.get("score_known", candidate.get("score_final", candidate.get("score_sim", 0.0))) or 0.0)
        score_map = getattr(rep, "ambiguous_candidate_scores", None)
        if isinstance(score_map, dict):
            return float(score_map.get(int(oid), 0.0))
        return 0.0

    def stable_det_assignments(self, solutions: list[dict]) -> dict[int, int]:
        valid = [dict(item.get("assignment", {}) or {}) for item in (solutions or []) if isinstance(item, dict)]
        if not valid:
            return {}
        common_det_ids = set(int(det_id) for det_id in valid[0].keys())
        for assignment in valid[1:]:
            common_det_ids &= set(int(det_id) for det_id in assignment.keys())
        out = {}
        for det_id in sorted(common_det_ids):
            vals = {int(assignment.get(int(det_id), -1)) for assignment in valid}
            if len(vals) == 1:
                val = next(iter(vals))
                if int(val) >= 0:
                    out[int(det_id)] = int(val)
        return out

    def competitive_frontier(self, *, solutions: list[dict], best_score: float, second_score: float) -> list[dict]:
        if not solutions:
            return []

        if float(second_score) >= 0.0:
            cutoff = float(second_score) - 1e-9
        else:
            cutoff = float(best_score) - 1e-9

        frontier = [
            dict(item)
            for item in (solutions or [])
            if float(item.get("best_score", -1.0)) >= float(cutoff)
        ]
        if frontier:
            return frontier
        return [dict(solutions[0])]

    def accepted_det_assignments(self, *, solution: dict | None, det_count: int, policy_mode: str = "") -> dict[int, int]:
        if not isinstance(solution, dict):
            return {}
        if float(solution.get("evidence", 0.0) or 0.0) < float(self.min_total_evidence):
            return {}
        if float(solution.get("best_score", 0.0) or 0.0) < float(self.min_assignment_score):
            return {}
        assignment = {
            int(det_id): int(oid)
            for det_id, oid in ((solution.get("assignment", {}) or {}).items())
            if det_id is not None and oid is not None
        }
        if str(policy_mode or "") in ("full_sibling", "coupled_full_sibling"):
            return dict(assignment)
        if int(det_count) <= 1:
            return dict(assignment)
        stable = {
            int(det_id): int(oid)
            for det_id, oid in ((solution.get("stable_det_assignments", {}) or {}).items())
            if det_id is not None and oid is not None
        }
        return dict(stable)
