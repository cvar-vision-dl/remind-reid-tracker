# update/memory_manager.py

from __future__ import annotations

from association.engine.candidate_generation import CandidateGenerator
from association.score_aggregator import ScoreAggregator
from association.scores.base_scores import SimilarityCombiner
from update.update_general import UpdatePolicies
from utils.time import ExecutionTimer



class FrameUpdateOutput:
    def __init__(self, timestamp: float):
        self.timestamp = float(timestamp)
        self.visible_object_ids = []
        self.matches = []
        self.created = []
        self.ambiguous = []
        self.provisional = []
        self.inactive = []
        self.removed = []

        self.proto_events = []

        self.summary = {
            "n_matches": 0,
            "n_created": 0,
            "n_ambiguous": 0,
            "n_provisional": 0,
            "n_visible": 0,
            "n_inactive": 0,
            "n_removed": 0,
        }
        self.timings_seconds = {}


class MemoryManager:
    """
    Per-frame update consuming ONLY association output:
      - decided_matches / to_create
      - geom_by_object_id (si existe)
      - reports_by_det_id for update rules (neighbor episodes and robust updates)
    """

    def __init__(self, config: dict, memory_store, class_id_to_name=None):
        self.config = config
        self.memory_store = memory_store

        self.policies = UpdatePolicies(
            config=config,
            memory_store=memory_store,
            class_id_to_name=class_id_to_name,
        )
        self.temp_scores = ScoreAggregator(config=config, memory_store=memory_store)
        self.temp_combiner = SimilarityCombiner(config)
        self.temp_candidate_generator = CandidateGenerator(
            scores=self.temp_scores,
            combiner=self.temp_combiner,
            memory_store=memory_store,
        )

    def apply_frame(
        self,
        detections: list,
        det_features_by_id: dict,
        association_output,
        timestamp: float,
        frame_id: int | None = None,
    ) -> FrameUpdateOutput:
        out = FrameUpdateOutput(timestamp=timestamp)
        timer = ExecutionTimer()

        dets = detections or []
        reports = getattr(association_output, "reports_by_det_id", {}) or {}
        decided_matches = getattr(association_output, "decided_matches", []) or []
        to_create = getattr(association_output, "to_create", []) or []
        to_ambiguous = getattr(association_output, "to_ambiguous", []) or []
        to_provisional_new = getattr(association_output, "to_provisional_new", []) or []
        assigned_by_det_id = dict(getattr(association_output, "assigned_by_det_id", {}) or {})
        det_by_id = self.index_detections(dets)
        visible_object_ids = []

        timer.run(
            "matches",
            self.apply_decided_matches,
            out=out,
            decided_matches=decided_matches,
            det_features_by_id=det_features_by_id,
            reports=reports,
            timestamp=float(timestamp),
            visible_object_ids=visible_object_ids,
        )
        timer.run(
            "creates",
            self.create_new_objects,
            out=out,
            to_create=to_create,
            det_features_by_id=det_features_by_id,
            reports=reports,
            timestamp=float(timestamp),
            visible_object_ids=visible_object_ids,
            assigned_by_det_id=assigned_by_det_id,
        )

        protected_ambiguous_object_ids = timer.run(
            "ambiguous_tracks",
            self.handle_ambiguous_tracks,
            out=out,
            to_ambiguous=to_ambiguous,
            timestamp=float(timestamp),
            det_features_by_id=det_features_by_id,
            det_by_id=det_by_id,
        )
        protected_provisional_object_ids = timer.run(
            "provisional_tracks",
            self.handle_provisional_new_tracks,
            out=out,
            to_provisional_new=to_provisional_new,
            timestamp=float(timestamp),
            det_features_by_id=det_features_by_id,
            det_by_id=det_by_id,
        )

        visible_object_ids = list(dict.fromkeys(int(x) for x in visible_object_ids))
        geom_by_object_id = self.base_geom_by_object_id(association_output)
        self.extend_geom_for_created_objects(
            geom_by_object_id=geom_by_object_id,
            created_entries=out.created,
            det_by_id=det_by_id,
        )

        timer.run(
            "neighbor_graphs",
            self.policies.update_neighbor_graphs,
            visible_object_ids=visible_object_ids,
            timestamp=float(timestamp),
            frame_id=frame_id,
            geom_by_object_id=geom_by_object_id,
            reports_by_det_id=reports,
            assigned_by_det_id=assigned_by_det_id,
            timer=timer,
            timer_prefix="neighbor_graphs/",
        )

        miss_out = timer.run(
            "misses",
            self.policies.apply_misses_for_non_visible,
            visible_object_ids,
            float(timestamp),
            protected_object_ids=(set(protected_ambiguous_object_ids) | set(protected_provisional_object_ids)),
        )
        out.inactive = list(miss_out.get("inactive_ids", []) or [])
        out.removed = list(miss_out.get("removed_ids", []) or [])

        timer.run("finalize", self.finalize_output, out=out, visible_object_ids=visible_object_ids)
        out.timings_seconds = timer.snapshot_seconds()
        return out

    def index_detections(self, detections: list) -> dict[int, object]:
        out = {}
        for det in detections or []:
            det_id = getattr(det, "detection_id", None)
            if det_id is None:
                continue
            out[int(det_id)] = det
        return out

    @staticmethod
    def duplicate_known_ref_payload(
        *,
        ids: list[int] | None = None,
        scores: dict[int, float] | None = None,
    ) -> dict:
        norm_ids = [int(x) for x in (ids or []) if x is not None]
        norm_scores = {
            int(k): float(v)
            for k, v in ((scores or {}).items())
            if k is not None and v is not None
        }
        return {
            "related_known_ids": list(norm_ids),
            "related_known_scores": dict(norm_scores),
            "support_known_ids": list(norm_ids),
            "support_known_scores": dict(norm_scores),
        }

    @classmethod
    def prefixed_known_ref_payload(
        cls,
        *,
        prefix: str,
        ids: list[int] | None = None,
        scores: dict[int, float] | None = None,
    ) -> dict:
        payload = cls.duplicate_known_ref_payload(ids=ids, scores=scores)
        return {f"{prefix}_{key}": value for key, value in payload.items()}

    @classmethod
    def track_known_ref_payload(cls, track) -> dict:
        if track is None:
            return cls.duplicate_known_ref_payload()
        ids = getattr(track, "support_known_ids", None)
        scores = getattr(track, "support_known_scores", None)
        return cls.duplicate_known_ref_payload(ids=ids, scores=scores)

    def apply_decided_matches(
        self,
        *,
        out: FrameUpdateOutput,
        decided_matches: list,
        det_features_by_id: dict,
        reports: dict,
        timestamp: float,
        visible_object_ids: list[int],
    ) -> None:
        for match in decided_matches or []:
            det_id = int(match.get("det_id", -1))
            obj_id = int(match.get("object_id", -1))
            score_final = float(match.get("score_final", 0.0))
            source = str(match.get("source", "association") or "association")

            obj = self.memory_store.get(int(obj_id))
            det_feats = det_features_by_id.get(int(det_id), None)
            if obj is None or det_feats is None:
                continue

            self.policies.apply_match_update(
                obj=obj,
                det_feats=det_feats,
                timestamp=float(timestamp),
                proto_events=out.proto_events,
                report=reports.get(int(det_id), None),
            )
            visible_object_ids.append(int(obj.object_id))
            out.matches.append(
                {
                    "det_id": int(det_id),
                    "object_id": int(obj.object_id),
                    "score_final": float(score_final),
                    "source": str(source),
                }
            )

    def create_new_objects(
        self,
        *,
        out: FrameUpdateOutput,
        to_create: list,
        det_features_by_id: dict,
        reports: dict,
        timestamp: float,
        visible_object_ids: list[int],
        assigned_by_det_id: dict[int, int],
    ) -> None:
        prov_cfg = ((self.config.get("update", {}) or {}).get("provisional_new", {}) or {})
        prov_min_overlap = float(prov_cfg.get("min_support_overlap", 0.5))
        for item in to_create or []:
            det_id = int(item.get("det_id", -1))
            class_id = int(item.get("class_id", -1))
            det_feats = det_features_by_id.get(int(det_id), None)
            if det_feats is None:
                continue

            origin = self.resolve_new_object_origin(
                det_id=int(det_id),
                class_id=int(class_id),
                reports=reports,
                min_overlap=float(prov_min_overlap),
                assigned_by_det_id=assigned_by_det_id,
            )

            class_name = self.policies.resolve_class_name(int(class_id))
            new_obj = self.memory_store.create_tracked_object(
                class_id=int(class_id),
                timestamp=float(timestamp),
                class_name=class_name,
            )
            self.policies.bootstrap_object_from_observation(
                tracked_object=new_obj,
                det_feats=det_feats,
                timestamp=float(timestamp),
                proto_events=out.proto_events,
            )
            self.policies.mark_visible(new_obj, float(timestamp))
            new_obj.origin = dict(origin)

            visible_object_ids.append(int(new_obj.object_id))
            out.created.append(
                {
                    "det_id": int(det_id),
                    "object_id": int(new_obj.object_id),
                    "origin_mode": str(origin.get("mode", "DIRECT_NEW") or "DIRECT_NEW"),
                    "origin_reason": str(origin.get("reason", "") or ""),
                    "origin_provisional_temp_id": origin.get("provisional_temp_id", None),
                    "origin_parent_object_ids": list(origin.get("parent_object_ids", []) or []),
                    **self.prefixed_known_ref_payload(
                        prefix="origin",
                        ids=origin.get("support_known_ids", []),
                        scores=origin.get("support_known_scores", {}),
                    ),
                }
            )
            assigned_by_det_id[int(det_id)] = int(new_obj.object_id)

    def resolve_new_object_origin(
        self,
        *,
        det_id: int,
        class_id: int,
        reports: dict,
        min_overlap: float,
        assigned_by_det_id: dict[int, int] | None = None,
    ) -> dict:
        rep = (reports or {}).get(int(det_id), None)
        support_known_ids, support_known_scores = self.support_knowns_from_report(rep)
        track = self.memory_store.find_best_provisional_origin(
            class_id=int(class_id),
            support_known_ids=support_known_ids,
            min_overlap=float(min_overlap),
        )
        if track is not None:
            temp_id = int(getattr(track, "temp_id", -1))
            origin = {
                "mode": "FROM_PROVISIONAL_NEW",
                "reason": str(getattr(track, "reason", "UNCERTAIN_NEW") or "UNCERTAIN_NEW"),
                "context_mode": str(getattr(track, "context_mode", "none") or "none"),
                "competition_mode": "none",
                "provisional_temp_id": int(temp_id) if temp_id >= 0 else None,
                "parent_object_ids": [],
                "parent_object_scores": {},
                **self.track_known_ref_payload(track),
            }
            if temp_id >= 0:
                self.memory_store.remove_provisional(int(temp_id))
            return origin

        committed_parent_origin = self.resolve_competitive_parent_new_origin(
            report=rep,
            assigned_by_det_id=assigned_by_det_id,
        )
        if committed_parent_origin is not None:
            return committed_parent_origin

        explicit_support_known_ids, explicit_support_known_scores = self.support_knowns_from_report(
            rep,
            include_plausible_fallback=False,
        )
        if explicit_support_known_ids:
            return {
                "mode": "DIRECT_NEW_WITH_KNOWN_PLAUSIBLE",
                "reason": str(getattr(rep, "final_reason", "") or ""),
                "context_mode": "known_plausible",
                "competition_mode": "none",
                "provisional_temp_id": None,
                "parent_object_ids": [],
                "parent_object_scores": {},
                **self.duplicate_known_ref_payload(
                    ids=explicit_support_known_ids,
                    scores=explicit_support_known_scores,
                ),
            }

        return {
            "mode": "DIRECT_NEW",
            "reason": str(getattr(rep, "final_reason", "") or ""),
            "context_mode": "none",
            "competition_mode": "none",
            "provisional_temp_id": None,
            "parent_object_ids": [],
            "parent_object_scores": {},
            **self.duplicate_known_ref_payload(),
        }

    def resolve_competitive_parent_new_origin(
        self,
        *,
        report,
        assigned_by_det_id: dict[int, int] | None,
    ) -> dict | None:
        if report is None:
            return None

        assigned_object_ids = {int(x) for x in ((assigned_by_det_id or {}).values()) if x is not None}
        if not assigned_object_ids:
            return None

        assoc_cfg = (self.config.get("association", {}) or {})
        match_cfg = (assoc_cfg.get("matching", {}) or {})
        match_thr = float(match_cfg.get("match_thr", 0.0))
        candidates = getattr(report, "candidates", None) or []
        ranked: list[tuple[float, dict]] = []
        for c in candidates:
            if not isinstance(c, dict):
                continue
            oid = c.get("object_id", None)
            if oid is None:
                continue
            oid = int(oid)
            if oid not in assigned_object_ids:
                continue
            score = c.get("score_known", None)
            if score is None:
                score = c.get("score_final", c.get("score_sim", 0.0))
            ranked.append((float(score or 0.0), c))

        if not ranked:
            return None

        ranked.sort(key=lambda item: float(item[0]), reverse=True)
        top_score, top_candidate = ranked[0]
        top_oid = int(top_candidate.get("object_id", -1))
        if top_oid < 0:
            return None
        if float(top_score) < float(match_thr):
            return None

        parent_ids = [int(top_oid)]
        parent_scores = {int(top_oid): float(top_score)}
        return {
            "mode": "COMMITTED_NEW_FROM_PARENT_COMPETITION",
            "reason": str(getattr(report, "final_reason", "") or "CREATED_NEW"),
            "context_mode": "parent_new_competition",
            "competition_mode": "known_vs_new",
            "provisional_temp_id": None,
            "parent_object_ids": list(parent_ids),
            "parent_object_scores": dict(parent_scores),
            **self.duplicate_known_ref_payload(ids=parent_ids, scores=parent_scores),
        }

    def support_knowns_from_report(
        self,
        report,
        *,
        topk: int = 3,
        include_plausible_fallback: bool = True,
    ) -> tuple[list[int], dict[int, float]]:
        if report is None:
            return [], {}

        support_ids = [int(x) for x in (getattr(report, "provisional_support_ids", None) or []) if x is not None]
        support_scores = {
            int(k): float(v)
            for k, v in (((getattr(report, "provisional_support_scores", None) or {}) or {}).items())
            if k is not None and v is not None
        }
        if support_ids:
            ordered = []
            seen = set()
            for oid in support_ids:
                if oid in seen:
                    continue
                seen.add(int(oid))
                ordered.append(int(oid))
                if len(ordered) >= int(max(1, topk)):
                    break
            return ordered, {int(k): float(v) for k, v in support_scores.items() if int(k) in ordered}

        if not include_plausible_fallback:
            return [], {}

        cands = getattr(report, "candidates", None) or []
        ranked = []
        for c in cands:
            if not isinstance(c, dict):
                continue
            oid = c.get("object_id", None)
            if oid is None:
                continue
            if int(c.get("known_plausible_keep", 0) or 0) != 1:
                continue
            score = c.get("score_known", None)
            if score is None:
                score = c.get("score_final", c.get("score_sim", 0.0))
            ranked.append((float(score or 0.0), int(oid)))

        ranked.sort(key=lambda x: float(x[0]), reverse=True)
        out_ids: list[int] = []
        out_scores: dict[int, float] = {}
        seen = set()
        for score, oid in ranked:
            if oid in seen:
                continue
            seen.add(int(oid))
            out_ids.append(int(oid))
            out_scores[int(oid)] = float(score)
            if len(out_ids) >= int(max(1, topk)):
                break
        return out_ids, out_scores

    def handle_ambiguous_tracks(
        self,
        *,
        out: FrameUpdateOutput,
        to_ambiguous: list,
        timestamp: float,
        det_features_by_id: dict,
        det_by_id: dict[int, object],
    ) -> set[int]:
        amb_cfg = ((self.config.get("update", {}) or {}).get("ambiguous_tracks", {}) or {})
        amb_enabled = bool(amb_cfg.get("enabled", True))
        amb_ttl = max(1, int(amb_cfg.get("ttl_frames", 6)))
        amb_min_overlap = float(amb_cfg.get("min_candidate_overlap", 0.5))
        match_cfg = (amb_cfg.get("temporal_matching", {}) or {})
        match_enabled = bool(match_cfg.get("enabled", True))
        match_min_score = float(match_cfg.get("min_score_sim", 0.55))
        match_overlap_weight = float(match_cfg.get("candidate_overlap_weight", 0.18))
        match_distribution_weight = float(match_cfg.get("candidate_distribution_weight", 0.10))
        match_geom_weight = float(match_cfg.get("geom_weight", 0.08))
        match_min_total = float(match_cfg.get("min_total_score", match_min_score))
        protected_object_ids: set[int] = set()
        seen_ambiguous_ids: set[int] = set()
        items = []

        if amb_enabled:
            for item in to_ambiguous or []:
                if not isinstance(item, dict):
                    continue
                det_id = int(item.get("det_id", -1))
                class_id = int(item.get("class_id", -1))
                candidate_ids = [int(x) for x in (item.get("candidate_ids", []) or [])]
                candidate_scores = {
                    int(key): float(value)
                    for key, value in ((item.get("candidate_scores", {}) or {}).items())
                    if key is not None and value is not None
                }
                if det_id < 0 or class_id < 0 or len(candidate_ids) < 2:
                    continue

                for oid in candidate_ids:
                    if self.memory_store.get(int(oid)) is not None:
                        protected_object_ids.add(int(oid))
                items.append(
                    {
                        "det_id": int(det_id),
                        "class_id": int(class_id),
                        "candidate_ids": list(candidate_ids),
                        "candidate_scores": dict(candidate_scores),
                        "related_known_ids": list(candidate_ids),
                        "related_known_scores": dict(candidate_scores),
                        "best_score": float(item.get("best_score", 0.0) or 0.0),
                        "reason": str(item.get("reason", "KNOWN_BUT_AMBIGUOUS") or "KNOWN_BUT_AMBIGUOUS"),
                        "committed_new_object_id": item.get("committed_new_object_id", None),
                        "committed_new_parent_ids": [int(x) for x in (item.get("committed_new_parent_ids", []) or [])],
                        "committed_new_parent_scores": {
                            int(key): float(value)
                            for key, value in ((item.get("committed_new_parent_scores", {}) or {}).items())
                            if key is not None and value is not None
                        },
                        "committed_new_seed_det_id": item.get("committed_new_seed_det_id", None),
                    }
                )

        self.materialize_committed_new_objects_for_ambiguous_items(
            items=items,
            timestamp=float(timestamp),
            det_features_by_id=det_features_by_id,
            protected_object_ids=protected_object_ids,
        )

        matched_track_by_det_id: dict[int, object] = {}
        if amb_enabled and match_enabled and items:
            matched_track_by_det_id = self.match_ambiguous_tracks_hungarian(
                ambiguous_items=items,
                det_features_by_id=det_features_by_id,
                det_by_id=det_by_id,
                min_score_sim=float(match_min_score),
                min_candidate_overlap=float(amb_min_overlap),
                overlap_weight=float(match_overlap_weight),
                distribution_weight=float(match_distribution_weight),
                geom_weight=float(match_geom_weight),
                min_total_score=float(match_min_total),
            )

        if amb_enabled:
            for item in items:
                det_id = int(item["det_id"])
                class_id = int(item["class_id"])
                candidate_ids = list(item.get("candidate_ids", []) or [])
                candidate_scores = dict(item.get("candidate_scores", {}) or {})

                track = matched_track_by_det_id.get(int(det_id), None)
                if track is None:
                    track = self.memory_store.create_ambiguous_track(
                        class_id=int(class_id),
                        timestamp=float(timestamp),
                        candidate_ids=candidate_ids,
                        candidate_scores=candidate_scores,
                        class_name=self.policies.resolve_class_name(int(class_id)),
                        ttl=int(amb_ttl),
                    )
                else:
                    track.refresh(
                        candidate_ids=candidate_ids,
                        candidate_scores=candidate_scores,
                        timestamp=float(timestamp),
                        ttl=int(amb_ttl),
                    )
                track.set_resolution_state(
                    state="UNRESOLVED_KNOWN",
                    confidence=float(item.get("best_score", 0.0) or 0.0),
                    novelty_score=0.0,
                )

                det_feats = det_features_by_id.get(int(det_id), None)
                det = det_by_id.get(int(det_id), None)
                geom = self.geom_from_detection(det) if det is not None else None
                if det_feats is not None:
                    self.policies.apply_temporary_observation(
                        track,
                        det_feats,
                        float(timestamp),
                        geom=geom,
                        score=float(item.get("best_score", 0.0) or 0.0),
                        metadata={
                            "mode": "ambiguous",
                            "candidate_ids": list(candidate_ids),
                            "candidate_scores": dict(candidate_scores),
                            "related_known_ids": list(candidate_ids),
                            "related_known_scores": dict(candidate_scores),
                        },
                        proto_events=None,
                    )

                seen_ambiguous_ids.add(int(track.temp_id))
                out.ambiguous.append(
                    {
                        "det_id": int(det_id),
                        "temp_id": int(track.temp_id),
                        "temp_label": str(getattr(track, "identity_label", f"T_ID{int(track.temp_id)}")),
                        "class_id": int(class_id),
                        "candidate_ids": list(candidate_ids),
                        "related_known_ids": [int(x) for x in (getattr(track, "current_candidate_ids", []) or [])],
                        "best_score": float(item.get("best_score", 0.0) or 0.0),
                        "source": str(item.get("source", "association") or "association"),
                    }
                )

        self.expire_stale_ambiguous_tracks(seen_ambiguous_ids)
        return protected_object_ids

    def materialize_committed_new_objects_for_ambiguous_items(
        self,
        *,
        items: list[dict],
        timestamp: float,
        det_features_by_id: dict,
        protected_object_ids: set[int],
    ) -> None:
        specs_by_object_id: dict[int, dict] = {}
        for item in items or []:
            if not isinstance(item, dict):
                continue
            object_id = item.get("committed_new_object_id", None)
            if object_id is None:
                continue
            object_id = int(object_id)
            if object_id in specs_by_object_id:
                continue
            seed_det_id = item.get("committed_new_seed_det_id", None)
            if seed_det_id is None:
                continue
            specs_by_object_id[int(object_id)] = {
                "object_id": int(object_id),
                "class_id": int(item.get("class_id", -1)),
                "seed_det_id": int(seed_det_id),
                "parent_object_ids": [int(x) for x in (item.get("committed_new_parent_ids", []) or [])],
                "parent_object_scores": {
                    int(key): float(value)
                    for key, value in ((item.get("committed_new_parent_scores", {}) or {}).items())
                    if key is not None and value is not None
                },
                "reason": str(item.get("reason", "KNOWN_AND_COMMITTED_NEW_COMPETITION") or "KNOWN_AND_COMMITTED_NEW_COMPETITION"),
            }

        for object_id in sorted(specs_by_object_id.keys()):
            if self.memory_store.get(int(object_id)) is not None:
                protected_object_ids.add(int(object_id))
                continue
            spec = specs_by_object_id[int(object_id)]
            det_feats = det_features_by_id.get(int(spec["seed_det_id"]), None)
            if det_feats is None or int(spec["class_id"]) < 0:
                continue
            new_obj = self.memory_store.create_tracked_object_with_id(
                object_id=int(object_id),
                class_id=int(spec["class_id"]),
                timestamp=float(timestamp),
                class_name=self.policies.resolve_class_name(int(spec["class_id"])),
            )
            self.policies.bootstrap_object_from_observation(
                tracked_object=new_obj,
                det_feats=det_feats,
                timestamp=float(timestamp),
                proto_events=None,
            )
            self.policies.mark_visible(new_obj, float(timestamp))
            new_obj.origin = {
                "mode": "COMMITTED_NEW_FROM_PARENT_COMPETITION",
                "reason": str(spec["reason"]),
                "context_mode": "parent_new_competition",
                "competition_mode": "known_vs_new",
                "provisional_temp_id": None,
                "parent_object_ids": list(spec["parent_object_ids"]),
                "parent_object_scores": dict(spec["parent_object_scores"]),
                **self.duplicate_known_ref_payload(
                    ids=spec["parent_object_ids"],
                    scores=spec["parent_object_scores"],
                ),
            }
            protected_object_ids.add(int(new_obj.object_id))

    def expire_stale_ambiguous_tracks(self, seen_ambiguous_ids: set[int]) -> None:
        _ = seen_ambiguous_ids
        return None

    def handle_provisional_new_tracks(
        self,
        *,
        out: FrameUpdateOutput,
        to_provisional_new: list,
        timestamp: float,
        det_features_by_id: dict,
        det_by_id: dict[int, object],
    ) -> set[int]:
        prov_cfg = ((self.config.get("update", {}) or {}).get("provisional_new", {}) or {})
        prov_enabled = bool(prov_cfg.get("enabled", True))
        prov_ttl = max(1, int(prov_cfg.get("ttl_frames", 6)))
        protected_object_ids: set[int] = set()
        seen_provisional_ids: set[int] = set()
        match_cfg = (prov_cfg.get("temporal_matching", {}) or {})
        match_enabled = bool(match_cfg.get("enabled", True))
        match_min_score = float(match_cfg.get("min_score_sim", 0.60))
        match_parent_bonus = float(match_cfg.get("parent_overlap_bonus", 0.08))
        match_parent_weight = float(match_cfg.get("parent_overlap_weight", match_parent_bonus))
        match_distribution_weight = float(match_cfg.get("parent_distribution_weight", 0.08))
        match_geom_weight = float(match_cfg.get("geom_weight", 0.06))
        match_min_total = float(match_cfg.get("min_total_score", match_min_score))
        match_empty_parent_enabled = bool(match_cfg.get("allow_empty_parent_match", True))

        items = []
        if prov_enabled:
            for item in to_provisional_new or []:
                if not isinstance(item, dict):
                    continue
                det_id = int(item.get("det_id", -1))
                class_id = int(item.get("class_id", -1))
                support_known_ids = [int(x) for x in (item.get("support_known_ids", []) or [])]
                support_known_scores = {
                    int(key): float(value)
                    for key, value in ((item.get("support_known_scores", {}) or {}).items())
                    if key is not None and value is not None
                }
                if det_id < 0 or class_id < 0:
                    continue

                for oid in support_known_ids:
                    if self.memory_store.get(int(oid)) is not None:
                        protected_object_ids.add(int(oid))
                items.append(
                    {
                        "det_id": int(det_id),
                        "class_id": int(class_id),
                        "best_score": float(item.get("best_score", 0.0) or 0.0),
                        "context_mode": str(item.get("context_mode", "none") or "none"),
                        "reason": str(item.get("reason", "UNCERTAIN_NEW") or "UNCERTAIN_NEW"),
                        **self.duplicate_known_ref_payload(
                            ids=support_known_ids,
                            scores=support_known_scores,
                        ),
                    }
                )

        matched_track_by_det_id: dict[int, object] = {}
        if prov_enabled and match_enabled and items:
            matched_track_by_det_id = self.match_provisional_tracks_hungarian(
                provisional_items=items,
                det_features_by_id=det_features_by_id,
                det_by_id=det_by_id,
                min_score_sim=float(match_min_score),
                parent_overlap_weight=float(match_parent_weight),
                distribution_weight=float(match_distribution_weight),
                geom_weight=float(match_geom_weight),
                min_total_score=float(match_min_total),
                allow_empty_parent_match=bool(match_empty_parent_enabled),
            )

        if prov_enabled:
            for item in items:
                det_id = int(item["det_id"])
                class_id = int(item["class_id"])
                support_known_ids = [int(x) for x in (item.get("support_known_ids", []) or [])]
                support_known_scores = {
                    int(key): float(value)
                    for key, value in ((item.get("support_known_scores", {}) or {}).items())
                    if key is not None and value is not None
                }

                track = matched_track_by_det_id.get(int(det_id), None)
                if track is None:
                    track = self.memory_store.create_provisional_new_track(
                        class_id=int(class_id),
                        timestamp=float(timestamp),
                        support_known_ids=support_known_ids,
                        support_known_scores=support_known_scores,
                        class_name=self.policies.resolve_class_name(int(class_id)),
                        ttl=int(prov_ttl),
                        context_mode=str(item.get("context_mode", "none") or "none"),
                        reason=str(item.get("reason", "UNCERTAIN_NEW") or "UNCERTAIN_NEW"),
                    )
                else:
                    track.refresh(
                        support_known_ids=support_known_ids,
                        support_known_scores=support_known_scores,
                        context_mode=str(item.get("context_mode", getattr(track, "context_mode", "none")) or "none"),
                        timestamp=float(timestamp),
                        ttl=int(prov_ttl),
                        reason=str(item.get("reason", getattr(track, "reason", "UNCERTAIN_NEW")) or "UNCERTAIN_NEW"),
                    )
                track.set_resolution_state(
                    state=track.resolution_state_from_reason(getattr(track, "reason", "UNCERTAIN_NEW")),
                    confidence=float(item.get("best_score", 0.0) or 0.0),
                    novelty_score=float(max(0.0, 1.0 - float(item.get("best_score", 0.0) or 0.0))),
                )

                det_feats = det_features_by_id.get(int(det_id), None)
                det = det_by_id.get(int(det_id), None)
                geom = self.geom_from_detection(det) if det is not None else None
                if det_feats is not None:
                    self.policies.apply_temporary_observation(
                        track,
                        det_feats,
                        float(timestamp),
                        geom=geom,
                        score=float(item.get("best_score", 0.0) or 0.0),
                        metadata={
                            "mode": "provisional",
                            "context_mode": str(item.get("context_mode", "none") or "none"),
                            **self.duplicate_known_ref_payload(
                                ids=support_known_ids,
                                scores=support_known_scores,
                            ),
                        },
                        proto_events=None,
                    )

                seen_provisional_ids.add(int(track.temp_id))
                out.provisional.append(
                    {
                        "det_id": int(det_id),
                        "temp_id": int(track.temp_id),
                        "temp_label": str(getattr(track, "identity_label", f"T_ID{int(track.temp_id)}")),
                        "class_id": int(class_id),
                        **self.duplicate_known_ref_payload(ids=support_known_ids),
                        "best_score": float(item.get("best_score", 0.0) or 0.0),
                        "context_mode": str(item.get("context_mode", "none") or "none"),
                    }
                )

        self.expire_stale_provisional_new_tracks(seen_provisional_ids)
        return protected_object_ids

    def match_provisional_tracks_hungarian(
        self,
        *,
        provisional_items: list[dict],
        det_features_by_id: dict,
        det_by_id: dict[int, object],
        min_score_sim: float,
        parent_overlap_weight: float,
        distribution_weight: float,
        geom_weight: float,
        min_total_score: float,
        allow_empty_parent_match: bool,
    ) -> dict[int, object]:
        return self.match_temporary_tracks_hungarian(
            items=provisional_items,
            existing_tracks=list(self.memory_store.all_provisional_new_tracks()),
            det_features_by_id=det_features_by_id,
            det_by_id=det_by_id,
            affinity_fn=lambda item, track, det_feats, det: self.provisional_temporal_affinity(
                item=item,
                track=track,
                det_feats=det_feats,
                det=det,
                min_score_sim=float(min_score_sim),
                parent_overlap_weight=float(parent_overlap_weight),
                distribution_weight=float(distribution_weight),
                geom_weight=float(geom_weight),
                allow_empty_parent_match=bool(allow_empty_parent_match),
                min_total_score=float(min_total_score),
            ),
        )

    def match_ambiguous_tracks_hungarian(
        self,
        *,
        ambiguous_items: list[dict],
        det_features_by_id: dict,
        det_by_id: dict[int, object],
        min_score_sim: float,
        min_candidate_overlap: float,
        overlap_weight: float,
        distribution_weight: float,
        geom_weight: float,
        min_total_score: float,
    ) -> dict[int, object]:
        return self.match_temporary_tracks_hungarian(
            items=ambiguous_items,
            existing_tracks=list(self.memory_store.all_ambiguous_tracks()),
            det_features_by_id=det_features_by_id,
            det_by_id=det_by_id,
            affinity_fn=lambda item, track, det_feats, det: self.ambiguous_temporal_affinity(
                item=item,
                track=track,
                det_feats=det_feats,
                det=det,
                min_score_sim=float(min_score_sim),
                min_candidate_overlap=float(min_candidate_overlap),
                overlap_weight=float(overlap_weight),
                distribution_weight=float(distribution_weight),
                geom_weight=float(geom_weight),
                min_total_score=float(min_total_score),
            ),
        )

    def match_temporary_tracks_hungarian(
        self,
        *,
        items: list[dict],
        existing_tracks: list,
        det_features_by_id: dict,
        det_by_id: dict[int, object],
        affinity_fn,
    ) -> dict[int, object]:
        if not items or not existing_tracks:
            return {}

        try:
            from scipy.optimize import linear_sum_assignment
        except Exception as e:
            raise RuntimeError("Temporal Hungarian requires scipy. Install: pip install scipy") from e

        items_by_class: dict[int, list[dict]] = {}
        tracks_by_class: dict[int, list] = {}
        for item in items:
            class_id = int(item.get("class_id", -1))
            if class_id < 0:
                continue
            items_by_class.setdefault(int(class_id), []).append(item)
        for track in existing_tracks:
            class_id = int(getattr(track, "class_id", -1))
            if class_id < 0:
                continue
            tracks_by_class.setdefault(int(class_id), []).append(track)

        out: dict[int, object] = {}
        for class_id, class_items in items_by_class.items():
            class_tracks = list(tracks_by_class.get(int(class_id), []))
            if not class_items or not class_tracks:
                continue

            score_rows: list[list[float | None]] = []
            for item in class_items:
                det_id = int(item.get("det_id", -1))
                det_feats = det_features_by_id.get(int(det_id), None)
                det = det_by_id.get(int(det_id), None)
                row: list[float | None] = []
                for track in class_tracks:
                    row.append(affinity_fn(item, track, det_feats, det))
                score_rows.append(row)

            if not score_rows:
                continue

            n_rows = len(class_items)
            n_cols = len(class_tracks)
            cost = [[1e6 for _ in range(n_cols)] for _ in range(n_rows)]
            for i in range(n_rows):
                for j in range(n_cols):
                    score = score_rows[i][j]
                    if score is None:
                        continue
                    cost[i][j] = -float(score)

            row_ind, col_ind = linear_sum_assignment(cost)
            for i, j in zip(row_ind, col_ind):
                score = score_rows[int(i)][int(j)]
                if score is None:
                    continue
                if float(cost[int(i)][int(j)]) >= 1e5:
                    continue
                det_id = int(class_items[int(i)].get("det_id", -1))
                if det_id < 0:
                    continue
                out[int(det_id)] = class_tracks[int(j)]
        return out

    def provisional_temporal_affinity(
        self,
        *,
        item: dict,
        track,
        det_feats,
        det,
        min_score_sim: float,
        parent_overlap_weight: float,
        distribution_weight: float,
        geom_weight: float,
        allow_empty_parent_match: bool,
        min_total_score: float,
    ) -> float | None:
        if det_feats is None or track is None:
            return None
        if int(item.get("class_id", -1)) != int(getattr(track, "class_id", -1)):
            return None

        item_parent_ids = [int(x) for x in (item.get("support_known_ids", []) or [])]
        track_parent_ids = [int(x) for x in (getattr(track, "support_known_ids", []) or [])]
        parent_overlap = self.provisional_parent_compatibility(
            item_parent_ids=item_parent_ids,
            track_parent_ids=track_parent_ids,
            allow_empty_parent_match=allow_empty_parent_match,
        )
        if parent_overlap is None:
            return None

        candidate = self.temp_candidate_generator.build_similarity_candidate(
            det_feats=det_feats,
            tracked_object=track,
        )
        score_sim = float(candidate.get("score_sim", 0.0) or 0.0)
        if score_sim < float(min_score_sim):
            return None

        distribution_similarity = self.weighted_relation_similarity(
            item_scores=dict(item.get("support_known_scores", {}) or {}),
            track_scores=dict(getattr(track, "support_known_scores", {}) or {}),
            allow_empty_match=bool(allow_empty_parent_match),
        )
        geom_score = self.temporal_geom_similarity(track=track, det=det)
        score_total = float(
            score_sim
            + (float(parent_overlap_weight) * float(parent_overlap))
            + (float(distribution_weight) * float(distribution_similarity))
            + (float(geom_weight) * float(geom_score))
        )
        if score_total < float(min_total_score):
            return None
        return float(score_total)

    def ambiguous_temporal_affinity(
        self,
        *,
        item: dict,
        track,
        det_feats,
        det,
        min_score_sim: float,
        min_candidate_overlap: float,
        overlap_weight: float,
        distribution_weight: float,
        geom_weight: float,
        min_total_score: float,
    ) -> float | None:
        if det_feats is None or track is None:
            return None
        if int(item.get("class_id", -1)) != int(getattr(track, "class_id", -1)):
            return None

        item_candidate_ids = [int(x) for x in (item.get("candidate_ids", []) or [])]
        track_candidate_ids = [int(x) for x in (getattr(track, "current_candidate_ids", []) or [])]
        overlap = self.dice_overlap(item_candidate_ids, track_candidate_ids)
        if overlap < float(min_candidate_overlap):
            return None

        candidate = self.temp_candidate_generator.build_similarity_candidate(
            det_feats=det_feats,
            tracked_object=track,
        )
        score_sim = float(candidate.get("score_sim", 0.0) or 0.0)
        if score_sim < float(min_score_sim):
            return None

        distribution_similarity = self.weighted_relation_similarity(
            item_scores=dict(item.get("candidate_scores", {}) or {}),
            track_scores=dict(getattr(track, "current_candidate_scores", {}) or {}),
            allow_empty_match=False,
        )
        geom_score = self.temporal_geom_similarity(track=track, det=det)
        score_total = float(
            score_sim
            + (float(overlap_weight) * float(overlap))
            + (float(distribution_weight) * float(distribution_similarity))
            + (float(geom_weight) * float(geom_score))
        )
        if score_total < float(min_total_score):
            return None
        return float(score_total)

    def provisional_parent_compatibility(
        self,
        *,
        item_parent_ids: list[int],
        track_parent_ids: list[int],
        allow_empty_parent_match: bool,
    ) -> float | None:
        item_set = set(int(x) for x in (item_parent_ids or []))
        track_set = set(int(x) for x in (track_parent_ids or []))
        if item_set and track_set:
            inter = int(len(item_set & track_set))
            if inter <= 0:
                return None
            return float((2.0 * inter) / float(len(item_set) + len(track_set)))
        if item_set or track_set:
            return None
        if not bool(allow_empty_parent_match):
            return None
        return 0.0

    def dice_overlap(self, ids_a: list[int], ids_b: list[int]) -> float:
        set_a = set(int(x) for x in (ids_a or []))
        set_b = set(int(x) for x in (ids_b or []))
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        inter = int(len(set_a & set_b))
        if inter <= 0:
            return 0.0
        return float((2.0 * inter) / float(len(set_a) + len(set_b)))

    def weighted_relation_similarity(
        self,
        *,
        item_scores: dict[int, float],
        track_scores: dict[int, float],
        allow_empty_match: bool,
    ) -> float:
        item_map = {
            int(k): max(0.0, float(v))
            for k, v in (item_scores or {}).items()
            if k is not None and v is not None
        }
        track_map = {
            int(k): max(0.0, float(v))
            for k, v in (track_scores or {}).items()
            if k is not None and v is not None
        }
        if not item_map and not track_map:
            return 1.0 if bool(allow_empty_match) else 0.0
        if not item_map or not track_map:
            return 0.0
        shared = set(item_map.keys()) | set(track_map.keys())
        inter = 0.0
        total = 0.0
        for oid in shared:
            a = float(item_map.get(int(oid), 0.0))
            b = float(track_map.get(int(oid), 0.0))
            inter += float(min(a, b))
            total += float(a + b)
        if total <= 1e-12:
            return 0.0
        return float((2.0 * inter) / float(total))

    def temporal_geom_similarity(self, *, track, det) -> float:
        if track is None or det is None:
            return 0.0
        det_geom = self.geom_from_detection(det)
        track_geom = getattr(track, "last_geom", None)
        if not isinstance(det_geom, dict) or not isinstance(track_geom, dict):
            return 0.0

        det_center = det_geom.get("center", None)
        track_center = track_geom.get("center", None)
        det_area = float(det_geom.get("area", 0.0) or 0.0)
        track_area = float(track_geom.get("area", 0.0) or 0.0)
        if det_center is None or track_center is None or det_area <= 0.0 or track_area <= 0.0:
            return 0.0

        dx = float(det_center[0]) - float(track_center[0])
        dy = float(det_center[1]) - float(track_center[1])
        dist = float((dx * dx + dy * dy) ** 0.5)
        scale = float(max(1.0, ((det_area + track_area) * 0.5) ** 0.5))
        center_score = float(max(0.0, 1.0 - min(1.0, dist / (2.5 * scale))))

        larger = float(max(det_area, track_area))
        smaller = float(min(det_area, track_area))
        area_score = float(smaller / larger) if larger > 1e-12 else 0.0
        return float((0.65 * center_score) + (0.35 * area_score))

    def expire_stale_provisional_new_tracks(self, seen_provisional_ids: set[int]) -> None:
        _ = seen_provisional_ids
        return None

    def base_geom_by_object_id(self, association_output) -> dict:
        geom_src = getattr(association_output, "geom_by_object_id", None)
        return dict(geom_src) if isinstance(geom_src, dict) else {}

    def extend_geom_for_created_objects(
        self,
        *,
        geom_by_object_id: dict,
        created_entries: list[dict],
        det_by_id: dict[int, object],
    ) -> None:
        for created in created_entries or []:
            det_id = int(created["det_id"])
            obj_id = int(created["object_id"])
            det = det_by_id.get(int(det_id), None)
            if det is None:
                continue
            geom = self.geom_from_detection(det)
            if geom is not None:
                geom_by_object_id[int(obj_id)] = geom

    def geom_from_detection(self, det) -> dict | None:
        bbox = getattr(det, "bbox", None)
        geom = getattr(det, "geom", None)
        if isinstance(geom, dict):
            center = geom.get("center", None)
            area = geom.get("area", None)
            if center is not None and area is not None and isinstance(center, (tuple, list)) and len(center) == 2:
                return {
                    "center": (float(center[0]), float(center[1])),
                    "area": float(area),
                    "bbox": tuple(float(x) for x in bbox[:4]) if bbox is not None and len(bbox) >= 4 else None,
                    "mask": getattr(det, "mask", None),
                }

        if not bbox or len(bbox) < 4:
            return None

        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
        w = float(max(0.0, float(x2) - float(x1)))
        h = float(max(0.0, float(y2) - float(y1)))
        return {
            "center": (0.5 * (float(x1) + float(x2)), 0.5 * (float(y1) + float(y2))),
            "area": float(w * h),
            "bbox": tuple(float(x) for x in bbox[:4]),
            "mask": getattr(det, "mask", None),
        }

    def finalize_output(self, *, out: FrameUpdateOutput, visible_object_ids: list[int]) -> None:
        out.visible_object_ids = list(visible_object_ids or [])
        out.summary["n_matches"] = int(len(out.matches))
        out.summary["n_created"] = int(len(out.created))
        out.summary["n_ambiguous"] = int(len(out.ambiguous))
        out.summary["n_provisional"] = int(len(out.provisional))
        out.summary["n_visible"] = int(len(out.visible_object_ids))
        out.summary["n_inactive"] = int(len(out.inactive))
        out.summary["n_removed"] = int(len(out.removed))
