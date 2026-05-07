from __future__ import annotations

import math


class ComponentBuildingMixin:
    def build_det_geom_by_id(self, detections: list) -> dict[int, dict]:
        out = {}
        for det in detections or []:
            det_id = getattr(det, "detection_id", None)
            geom = getattr(det, "geom", None)
            if det_id is None or not isinstance(geom, dict):
                continue
            center = geom.get("center", None)
            area = geom.get("area", None)
            if center is None or area is None:
                continue
            bbox = getattr(det, "bbox", None)
            mask = getattr(det, "mask", None)
            out[int(det_id)] = {
                "center": (float(center[0]), float(center[1])),
                "area": float(area),
                "bbox": tuple(float(x) for x in bbox[:4]) if bbox is not None and len(bbox) >= 4 else None,
                "mask": mask,
            }
        return out

    def build_components(self, *, ambiguous_entries: list[dict], det_geom_by_id: dict[int, dict]) -> list[dict]:
        det_to_candidates = {}
        for item in ambiguous_entries or []:
            if not isinstance(item, dict):
                continue
            det_id = int(item.get("det_id", -1))
            candidate_ids = [int(x) for x in (item.get("candidate_ids", []) or []) if x is not None]
            candidate_ids = sorted(set(candidate_ids))
            if det_id < 0 or len(candidate_ids) < 2:
                continue
            det_to_candidates[int(det_id)] = candidate_ids

        components = []
        adjacency = {int(det_id): set() for det_id in det_to_candidates.keys()}
        det_ids = sorted(det_to_candidates.keys())
        for idx, det_id_a in enumerate(det_ids):
            for det_id_b in det_ids[idx + 1 :]:
                if self.component_pair_is_linked(
                    det_id_a=int(det_id_a),
                    det_id_b=int(det_id_b),
                    candidates_a=det_to_candidates.get(int(det_id_a), []),
                    candidates_b=det_to_candidates.get(int(det_id_b), []),
                    det_geom_by_id=det_geom_by_id,
                ):
                    adjacency[int(det_id_a)].add(int(det_id_b))
                    adjacency[int(det_id_b)].add(int(det_id_a))

        seen_dets = set()
        for det_id in det_ids:
            if int(det_id) in seen_dets:
                continue
            det_queue = [int(det_id)]
            comp_dets = set()
            while det_queue:
                cur_det = int(det_queue.pop())
                if cur_det in comp_dets:
                    continue
                comp_dets.add(cur_det)
                seen_dets.add(cur_det)
                for other_det in adjacency.get(int(cur_det), set()):
                    if int(other_det) not in comp_dets:
                        det_queue.append(int(other_det))

            comp_oids = set()
            for comp_det_id in comp_dets:
                comp_oids.update(int(oid) for oid in det_to_candidates.get(int(comp_det_id), []))
            components.append(
                {
                    "det_ids": sorted(int(x) for x in comp_dets),
                    "candidate_union": sorted(int(x) for x in comp_oids),
                    "candidates_by_det": {
                        int(did): list(det_to_candidates.get(int(did), []))
                        for did in sorted(comp_dets)
                    },
                }
            )
        return components

    def component_resolvability_reason(self, component: dict) -> str | None:
        det_ids = list(component.get("det_ids", []) or [])
        candidate_union = list(component.get("candidate_union", []) or [])
        if len(det_ids) < 1:
            return "empty_component"
        if len(det_ids) > int(self.max_group_size):
            return f"too_many_dets({len(det_ids)}>{int(self.max_group_size)})"
        if len(candidate_union) > int(self.max_candidate_union):
            return f"too_many_candidates({len(candidate_union)}>{int(self.max_candidate_union)})"
        if len(candidate_union) < len(det_ids):
            return f"candidate_union_smaller_than_dets({len(candidate_union)}<{len(det_ids)})"
        for det_id in det_ids:
            if len(component.get("candidates_by_det", {}).get(int(det_id), []) or []) < 1:
                return f"det_without_candidates({int(det_id)})"
        return None

    def component_pair_is_linked(
        self,
        *,
        det_id_a: int,
        det_id_b: int,
        candidates_a: list[int],
        candidates_b: list[int],
        det_geom_by_id: dict[int, dict],
    ) -> bool:
        set_a = set(int(x) for x in (candidates_a or []))
        set_b = set(int(x) for x in (candidates_b or []))
        class_ids_a = {
            int(getattr(obj, "class_id", -1))
            for oid in set_a
            for obj in [None if self.memory_store is None else self.memory_store.get(int(oid))]
            if obj is not None and int(getattr(obj, "class_id", -1)) >= 0
        }
        class_ids_b = {
            int(getattr(obj, "class_id", -1))
            for oid in set_b
            for obj in [None if self.memory_store is None else self.memory_store.get(int(oid))]
            if obj is not None and int(getattr(obj, "class_id", -1)) >= 0
        }
        if class_ids_a and class_ids_b and not (class_ids_a & class_ids_b):
            return False
        if set_a & set_b:
            return True
        affinity = self.component_pair_affinity(
            det_id_a=int(det_id_a),
            det_id_b=int(det_id_b),
            candidates_a=list(set_a),
            candidates_b=list(set_b),
            det_geom_by_id=det_geom_by_id,
        )
        return bool(float(affinity) >= float(self.min_component_pair_score))

    def component_pair_affinity(
        self,
        *,
        det_id_a: int,
        det_id_b: int,
        candidates_a: list[int],
        candidates_b: list[int],
        det_geom_by_id: dict[int, dict],
    ) -> float:
        geom_a = det_geom_by_id.get(int(det_id_a), None)
        geom_b = det_geom_by_id.get(int(det_id_b), None)
        if not isinstance(geom_a, dict) or not isinstance(geom_b, dict):
            return 0.0

        obs = self.relation_observation_cached(
            geom_a,
            geom_b,
            scale_min=40.0,
            geom_a_key=("det", int(det_id_a)),
            geom_b_key=("det", int(det_id_b)),
        )
        if not isinstance(obs, dict):
            return 0.0

        obs_distance = self.primary_observed_distance(obs)
        if not math.isfinite(float(obs_distance)):
            return 0.0
        if float(obs_distance) > float(self.max_component_pair_distance):
            return 0.0

        best_strength = 0.0
        for oid_a in candidates_a or []:
            obj_a = self.memory_store.get(int(oid_a)) if self.memory_store is not None else None
            dg_a = getattr(obj_a, "neighbor_dist", None) if obj_a is not None else None
            for oid_b in candidates_b or []:
                obj_b = self.memory_store.get(int(oid_b)) if self.memory_store is not None else None
                dg_b = getattr(obj_b, "neighbor_dist", None) if obj_b is not None else None
                if dg_a is None and dg_b is None:
                    continue

                score_ab, weight_ab = self.relation_similarity(
                    obs=obs,
                    edge=None if dg_a is None else dg_a.get_edge(int(oid_b)),
                )
                score_ba, weight_ba = self.relation_similarity(
                    obs=obs,
                    edge=None if dg_b is None else dg_b.get_edge(int(oid_a)),
                )
                pair_weight = float(weight_ab + weight_ba)
                if pair_weight <= 0.0:
                    continue
                pair_score = float(
                    ((float(score_ab) * float(weight_ab)) + (float(score_ba) * float(weight_ba)))
                    / max(1e-12, pair_weight)
                )
                strength = float(pair_score * (pair_weight / 2.0))
                if float(strength) > float(best_strength):
                    best_strength = float(strength)
        return float(best_strength)

    def component_class_ids(self, *, component: dict) -> set[int]:
        class_ids = set()
        candidates_by_det = dict(component.get("candidates_by_det", {}) or {})
        for candidate_ids in candidates_by_det.values():
            for oid in candidate_ids or []:
                obj = None if self.memory_store is None else self.memory_store.get(int(oid))
                if obj is None:
                    continue
                class_id = int(getattr(obj, "class_id", -1))
                if class_id >= 0:
                    class_ids.add(int(class_id))
        return set(int(x) for x in class_ids)

    def is_real_object_id(self, object_id: int) -> bool:
        return bool(self.memory_store is not None and self.memory_store.get(int(object_id)) is not None)
