# update/update_general.py

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from memory.cross_view_identity import relation_affinity, zone_key_from_support_bbox
from memory.neighbor_distance_graph import compute_relation_observation, prepare_relation_mask_runtime
from update.descriptors.update_background import BackgroundUpdater
from update.descriptors.update_neighbors import NeighborUpdater
from update.descriptors.update_object import ObjectUpdater
from update.descriptors.update_parts import PartsUpdater


class UpdateDecision:
    """Control which update operation types are allowed for a match."""

    def __init__(
        self,
        mode: str = "FULL",
        allow_insert: bool = True,
        allow_merge: bool = True,
        allow_promote: bool = True,
        allow_ema: bool = True,
        alpha_scale: float = 1.0,
    ):
        self.mode = str(mode)
        self.allow_insert = bool(allow_insert)
        self.allow_merge = bool(allow_merge)
        self.allow_promote = bool(allow_promote)
        self.allow_ema = bool(allow_ema)
        self.alpha_scale = float(alpha_scale)


class UpdatePolicies:
    """Update orchestrator: lifecycle + delegation to object/background/parts/neighbors."""

    def __init__(self, config: dict, memory_store, class_id_to_name=None):
        self.config = config
        self.memory_store = memory_store
        self.class_id_to_name = class_id_to_name if isinstance(class_id_to_name, dict) else None

        upd = (config.get("update", {}) or {})
        self.confirm_hits = int(upd.get("confirm_hits", 2))

        self.remove_enabled = bool(upd.get("remove_enabled", True))

        max_misses = int(upd.get("max_misses", 10))
        self.max_misses_tentative = int(upd.get("max_misses_tentative", max_misses))
        self.max_misses_confirmed = int(upd.get("max_misses_confirmed", max_misses))
        self.inactive_ttl = int(upd.get("inactive_ttl", 0))

        robust = (upd.get("robust_updates", {}) or {})
        self.robust_updates_enabled = bool(robust.get("enabled", True))

        self.safe_alpha_scale = float(robust.get("safe_alpha_scale", 0.2))
        self.safe_alpha_scale = max(0.0, min(1.0, self.safe_alpha_scale))

        ng_cfg = (upd.get("neighbor_graphs", {}) or {})
        dist_parallel_cfg = (ng_cfg.get("dist_observations_parallel", {}) or {})
        self.dist_obs_parallel_enabled = bool(dist_parallel_cfg.get("enabled", True))
        self.dist_obs_parallel_min_pairs = max(2, int(dist_parallel_cfg.get("min_pairs", 48)))
        self.dist_obs_parallel_workers = int(dist_parallel_cfg.get("workers", 0))
        self.dist_obs_parallel_max_auto_workers = max(1, int(dist_parallel_cfg.get("max_auto_workers", 8)))
        graph_parallel_cfg = (ng_cfg.get("graph_updates_parallel", {}) or {})
        self.graph_updates_parallel_enabled = bool(graph_parallel_cfg.get("enabled", True))
        self.graph_updates_parallel_min_objects = max(2, int(graph_parallel_cfg.get("min_objects", 8)))
        self.graph_updates_parallel_workers = int(graph_parallel_cfg.get("workers", 0))
        self.graph_updates_parallel_max_auto_workers = max(1, int(graph_parallel_cfg.get("max_auto_workers", 8)))

        self.obj_upd = ObjectUpdater(config)
        self.bg_upd = BackgroundUpdater(config)
        self.parts_upd = PartsUpdater(config)
        self.neigh_upd = NeighborUpdater(config)

        mem_cfg = (config.get("memory", {}) or {}) if isinstance(config, dict) else {}
        cv_cfg = (mem_cfg.get("cross_view_identity", {}) or {}) if isinstance(mem_cfg, dict) else {}
        self.cross_view_enabled = bool(cv_cfg.get("enabled", True))
        self.cross_view_min_support_like = max(0.0, min(1.0, float(cv_cfg.get("min_support_like", 0.35))))
        self.cross_view_grid_rows = max(1, int(cv_cfg.get("grid_rows", 3)))
        self.cross_view_grid_cols = max(1, int(cv_cfg.get("grid_cols", 3)))
        self.cross_view_anchor_topk = max(1, int(cv_cfg.get("max_anchor_rank", 3)))
        self.cross_view_support_neighbor_topk = max(1, int(cv_cfg.get("max_support_neighbor_rank", 3)))

    def resolve_class_name(self, class_id: int):
        if self.class_id_to_name is not None:
            return self.class_id_to_name.get(int(class_id), None)
        return None

    def bootstrap_object_from_observation(
        self,
        tracked_object,
        det_feats: dict,
        timestamp: float,
        proto_events: list | None = None,
    ):
        self.obj_upd.update_from_det_feats(
            tracked_object, det_feats, timestamp, proto_events=proto_events, decision=None
        )
        self.bg_upd.update_from_det_feats(
            tracked_object, det_feats, timestamp, proto_events=proto_events, decision=None
        )
        self.parts_upd.update_from_det_feats(
            tracked_object, det_feats, timestamp, proto_events=proto_events, decision=None
        )

    def build_update_decision(self, report) -> UpdateDecision:
        if not self.robust_updates_enabled:
            return UpdateDecision()

        if report is None:
            return UpdateDecision()

        final_decision = str(getattr(report, "final_decision", "MATCH") or "MATCH").upper()
        if final_decision != "MATCH":
            return UpdateDecision(
                mode="OFF",
                allow_insert=False,
                allow_merge=False,
                allow_promote=False,
                allow_ema=False,
                alpha_scale=0.0,
            )

        diag = getattr(report, "match_diag_final", None)
        if not isinstance(diag, dict):
            diag = getattr(report, "match_diag_sim", None)
        status = str((diag or {}).get("status", "STRONG")).upper() if isinstance(diag, dict) else "STRONG"

        if status == "WEAK":
            return UpdateDecision(
                mode="OFF",
                allow_insert=False,
                allow_merge=False,
                allow_promote=False,
                allow_ema=False,
                alpha_scale=0.0,
            )

        if status == "AMBIGUOUS":
            return UpdateDecision(
                mode="SAFE",
                allow_insert=False,
                allow_merge=False,
                allow_promote=False,
                allow_ema=True,
                alpha_scale=float(self.safe_alpha_scale),
            )

        return UpdateDecision()

    def apply_match_update(
        self,
        obj,
        det_feats: dict,
        timestamp: float,
        proto_events: list | None = None,
        report=None,
    ) -> None:
        self.mark_visible(obj, timestamp)

        decision = self.build_update_decision(report)
        if str(decision.mode).upper() == "OFF":
            return

        self.obj_upd.update_from_det_feats(obj, det_feats, timestamp, proto_events=proto_events, decision=decision)
        self.bg_upd.update_from_det_feats(obj, det_feats, timestamp, proto_events=proto_events, decision=decision)
        self.parts_upd.update_from_det_feats(obj, det_feats, timestamp, proto_events=proto_events, decision=decision)

    def apply_temporary_observation(
        self,
        obj,
        det_feats: dict,
        timestamp: float,
        *,
        geom: dict | None = None,
        score: float | None = None,
        metadata: dict | None = None,
        proto_events: list | None = None,
    ) -> None:
        if obj is None or det_feats is None:
            return

        if hasattr(obj, "mark_visible"):
            obj.mark_visible(float(timestamp))
        if hasattr(obj, "record_observation"):
            obj.record_observation(
                timestamp=float(timestamp),
                geom=geom,
                score=score,
                metadata=metadata,
            )

        self.obj_upd.update_from_det_feats(obj, det_feats, timestamp, proto_events=proto_events, decision=None)
        self.bg_upd.update_from_det_feats(obj, det_feats, timestamp, proto_events=proto_events, decision=None)
        self.parts_upd.update_from_det_feats(obj, det_feats, timestamp, proto_events=proto_events, decision=None)

    def mark_visible(self, obj, timestamp: float) -> None:
        obj.last_seen = float(timestamp)
        obj.misses = 0
        obj.hits = int(getattr(obj, "hits", 0)) + 1
        obj.age = int(getattr(obj, "age", 0)) + 1

        state = str(getattr(obj, "state", "NEW"))

        if state in ("NEW", "TENTATIVE"):
            if int(getattr(obj, "hits", 0)) >= self.confirm_hits:
                obj.state = "CONFIRMED"
            else:
                obj.state = "TENTATIVE"
            return

        if state == "INACTIVE":
            obj.state = "CONFIRMED"
            return

        if state != "CONFIRMED":
            obj.state = "CONFIRMED"

    def apply_misses_for_non_visible(
        self,
        visible_object_ids,
        timestamp: float,
        protected_object_ids=None,
    ) -> dict:
        visible = set(int(x) for x in (visible_object_ids or []))
        protected = set(int(x) for x in (protected_object_ids or []))

        inactive_ids: list[int] = []
        removed_ids: list[int] = []

        for obj in list(self.memory_store.all_objects()):
            oid = int(obj.object_id)
            if oid in visible:
                continue
            if oid in protected:
                continue

            prev_state = str(getattr(obj, "state", "NEW"))
            prev_misses = int(getattr(obj, "misses", 0))

            obj.misses = prev_misses + 1
            obj.age = int(getattr(obj, "age", 0)) + 1

            if prev_state in ("NEW", "TENTATIVE"):
                if self.max_misses_tentative > 0 and obj.misses >= self.max_misses_tentative:
                    if self.remove_enabled:
                        self.memory_store.remove(oid)
                        removed_ids.append(oid)
                    else:
                        obj.state = "TENTATIVE"
                else:
                    obj.state = "TENTATIVE"
                continue

            if prev_state == "CONFIRMED":
                if self.max_misses_confirmed > 0 and obj.misses >= self.max_misses_confirmed:
                    obj.state = "INACTIVE"
                    inactive_ids.append(oid)
                continue

            if prev_state == "INACTIVE":
                if self.inactive_ttl > 0 and obj.misses >= self.inactive_ttl:
                    if self.remove_enabled:
                        self.memory_store.remove(oid)
                        removed_ids.append(oid)

        return {
            "inactive_ids": list(dict.fromkeys(int(x) for x in inactive_ids)),
            "removed_ids": list(dict.fromkeys(int(x) for x in removed_ids)),
        }

    def allow_neighbors_episode_from_report(self, report) -> bool:
        if not self.robust_updates_enabled:
            return True

        if report is None:
            return True

        final_decision = str(getattr(report, "final_decision", "")).upper()
        if final_decision and final_decision != "MATCH":
            return False

        diag = getattr(report, "match_diag_final", None)
        if not isinstance(diag, dict):
            diag = getattr(report, "match_diag_sim", None)
        status = str((diag or {}).get("status", "STRONG")).upper() if isinstance(diag, dict) else "STRONG"
        if status in ("WEAK", "AMBIGUOUS"):
            return False

        return True

    def _support_assignment_for_object(
        self,
        oid: int,
        ids: list[int],
        geom: dict[int, dict],
        dist_obs_by_object_id: dict[int, dict[int, dict]],
    ) -> tuple[int | None, float, dict | None]:
        geom_self = geom.get(int(oid), None)
        if not isinstance(geom_self, dict):
            return None, 0.0, None
        area_self = float(max(1e-6, geom_self.get("area", 0.0)))

        best_oid = None
        best_score = 0.0
        best_obs = None
        for other_id in ids:
            other_id = int(other_id)
            if other_id == int(oid):
                continue
            geom_other = geom.get(int(other_id), None)
            if not isinstance(geom_other, dict):
                continue
            area_other = float(max(1e-6, geom_other.get("area", 0.0)))
            if area_other <= float(area_self * 1.15):
                continue
            obs = ((dist_obs_by_object_id.get(int(oid), {}) or {}).get(int(other_id), None))
            if not isinstance(obs, dict):
                continue
            support_like = float(max(0.0, min(1.0, obs.get("support_like", 0.0))))
            if support_like < float(self.cross_view_min_support_like):
                continue
            if support_like > float(best_score):
                best_oid = int(other_id)
                best_score = float(support_like)
                best_obs = obs
        return best_oid, float(best_score), best_obs

    def _rank_context_ids(
        self,
        oid: int,
        context_ids: list[int],
        dist_obs_by_object_id: dict[int, dict[int, dict]],
        *,
        topk: int,
    ) -> list[int]:
        scored = []
        for other_id in context_ids or []:
            other_id = int(other_id)
            if other_id == int(oid):
                continue
            obs = ((dist_obs_by_object_id.get(int(oid), {}) or {}).get(int(other_id), None))
            if not isinstance(obs, dict):
                continue
            scored.append((float(relation_affinity(obs)), int(other_id)))
        scored.sort(key=lambda kv: float(kv[0]), reverse=True)
        return [int(oid2) for score, oid2 in scored[: max(1, int(topk))] if float(score) > 0.0]

    def _update_cross_view_identity(
        self,
        ids: list[int],
        geom: dict[int, dict],
        dist_obs_by_object_id: dict[int, dict[int, dict]],
        allow_episode_by_object_id: dict[int, bool],
        episode_idx: int,
    ) -> None:
        if not self.cross_view_enabled or not geom:
            return

        support_by_oid: dict[int, tuple[int | None, float, dict | None]] = {}
        for oid in ids:
            support_by_oid[int(oid)] = self._support_assignment_for_object(
                oid=int(oid),
                ids=ids,
                geom=geom,
                dist_obs_by_object_id=dist_obs_by_object_id,
            )

        confirmed_ids = []
        for oid in ids:
            obj = self.memory_store.get(int(oid))
            if obj is None:
                continue
            if str(getattr(obj, "state", "")).upper() == "CONFIRMED":
                confirmed_ids.append(int(oid))

        for oid in ids:
            if not bool(allow_episode_by_object_id.get(int(oid), True)):
                continue
            obj = self.memory_store.get(int(oid))
            if obj is None:
                continue
            cv = getattr(obj, "cross_view", None)
            if cv is None or not getattr(cv, "enabled", False):
                continue
            if str(getattr(obj, "state", "")).upper() not in ("CONFIRMED", "TENTATIVE"):
                continue

            support_oid, support_like, support_obs = support_by_oid.get(int(oid), (None, 0.0, None))
            geom_self = geom.get(int(oid), None)
            if not isinstance(geom_self, dict):
                continue

            support_bbox = None
            if support_oid is not None:
                support_geom = geom.get(int(support_oid), None)
                if isinstance(support_geom, dict):
                    support_bbox = support_geom.get("bbox", None)
            zone_key = zone_key_from_support_bbox(
                support_bbox,
                geom_self.get("center", None),
                rows=int(self.cross_view_grid_rows),
                cols=int(self.cross_view_grid_cols),
            )

            anchor_context_ids = [int(x) for x in confirmed_ids if int(x) != int(oid)]
            anchor_order = self._rank_context_ids(
                oid=int(oid),
                context_ids=anchor_context_ids,
                dist_obs_by_object_id=dist_obs_by_object_id,
                topk=int(self.cross_view_anchor_topk),
            )

            support_neighbor_ids = []
            if support_oid is not None:
                for other_id in confirmed_ids:
                    other_id = int(other_id)
                    if other_id in (int(oid), int(support_oid)):
                        continue
                    other_support_oid, other_support_like, _ = support_by_oid.get(int(other_id), (None, 0.0, None))
                    if other_support_oid is None or int(other_support_oid) != int(support_oid):
                        continue
                    if float(other_support_like) < float(self.cross_view_min_support_like):
                        continue
                    support_neighbor_ids.append(int(other_id))
            support_neighbor_order = self._rank_context_ids(
                oid=int(oid),
                context_ids=support_neighbor_ids,
                dist_obs_by_object_id=dist_obs_by_object_id,
                topk=int(self.cross_view_support_neighbor_topk),
            )

            on_support_like = float(max(0.0, min(1.0, support_like)))
            inside_support_like = 0.0
            if isinstance(support_obs, dict):
                inside_support_like = float(on_support_like * (1.0 if bool(support_obs.get("center_inside", False)) else 0.0))

            cv.observe(
                support_oid=int(support_oid) if support_oid is not None else None,
                support_like=float(support_like),
                zone_key=zone_key,
                anchor_order=anchor_order,
                support_neighbor_order=support_neighbor_order,
                on_support_like=float(on_support_like),
                inside_support_like=float(inside_support_like),
                episode_idx=int(episode_idx),
            )

    def update_neighbor_graphs(
        self,
        visible_object_ids: list,
        timestamp: float,
        frame_id: int | None = None,
        geom_by_object_id: dict | None = None,
        reports_by_det_id: dict | None = None,
        assigned_by_det_id: dict | None = None,
        timer=None,
        timer_prefix: str = "",
    ) -> None:
        ids = [int(x) for x in (visible_object_ids or [])]
        if not ids:
            return

        geom = geom_by_object_id if isinstance(geom_by_object_id, dict) else {}
        rep_by_det = reports_by_det_id if isinstance(reports_by_det_id, dict) else {}
        asg = assigned_by_det_id if isinstance(assigned_by_det_id, dict) else {}
        allow_episode_by_object_id = self.build_allow_episode_map(asg=asg, reports_by_det_id=rep_by_det)
        step = (lambda suffix: f"{timer_prefix}{suffix}" if timer_prefix else str(suffix))
        if timer is not None:
            dist_obs_by_object_id = timer.run(
                step("dist_observations"),
                self.build_dist_observations,
                ids=ids,
                geom=geom,
                allow_episode_by_object_id=allow_episode_by_object_id,
            )
        else:
            dist_obs_by_object_id = self.build_dist_observations(
                ids=ids,
                geom=geom,
                allow_episode_by_object_id=allow_episode_by_object_id,
            )
        view_id = self.resolve_anchor_view_id(
            ids=ids,
            geom=geom,
            allow_episode_by_object_id=allow_episode_by_object_id,
        )

        if timer is not None:
            timer.run(
                step("cross_view"),
                self._update_cross_view_identity,
                ids=ids,
                geom=geom,
                dist_obs_by_object_id=dist_obs_by_object_id,
                allow_episode_by_object_id=allow_episode_by_object_id,
                episode_idx=int(frame_id if frame_id is not None else round(float(timestamp))),
            )
        else:
            self._update_cross_view_identity(
                ids=ids,
                geom=geom,
                dist_obs_by_object_id=dist_obs_by_object_id,
                allow_episode_by_object_id=allow_episode_by_object_id,
                episode_idx=int(frame_id if frame_id is not None else round(float(timestamp))),
            )

        def _apply_graph_updates() -> None:
            object_ids = [int(oid) for oid in ids]
            workers = self._resolve_graph_updates_workers(object_count=int(len(object_ids)))

            def _update_one(oid: int) -> None:
                obj = self.memory_store.get(int(oid))
                if obj is None:
                    return

                ng = getattr(obj, "neighbors", None)
                if ng is None:
                    return

                dist_graph = getattr(obj, "neighbor_dist", None)
                allow = bool(allow_episode_by_object_id.get(int(oid), True))
                dist_obs = dist_obs_by_object_id.get(int(oid), None) if allow else None

                self.neigh_upd.update(
                    graph=ng,
                    dist_graph=dist_graph,
                    self_id=int(oid),
                    visible_object_ids=object_ids,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    geom_by_object_id=geom,
                    dist_obs_by_other_id=dist_obs,
                    allow_episode=allow,
                    view_id=view_id,
                )

            if workers <= 1:
                for oid in object_ids:
                    _update_one(int(oid))
                return

            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                for _ in ex.map(_update_one, object_ids):
                    pass

        if timer is not None:
            timer.run(step("graph_updates"), _apply_graph_updates)
        else:
            _apply_graph_updates()

    def build_allow_episode_map(self, *, asg: dict, reports_by_det_id: dict) -> dict[int, bool]:
        out: dict[int, bool] = {}
        for det_id, obj_id in (asg or {}).items():
            rep = reports_by_det_id.get(int(det_id), None)
            out[int(obj_id)] = bool(self.allow_neighbors_episode_from_report(rep))
        return out

    def build_dist_observations(
        self,
        *,
        ids: list[int],
        geom: dict[int, dict],
        allow_episode_by_object_id: dict[int, bool],
    ) -> dict[int, dict[int, dict]]:
        out: dict[int, dict[int, dict]] = {}
        if not geom or len(ids) < 2:
            return out

        pair_specs: list[tuple[int, dict, float, float, float, dict | None]] = []
        for oid in ids:
            if not bool(allow_episode_by_object_id.get(int(oid), True)):
                continue
            obj = self.memory_store.get(int(oid))
            if obj is None:
                continue
            dg = getattr(obj, "neighbor_dist", None)
            if dg is None or not getattr(dg, "enabled", False):
                continue

            geom_self = geom.get(int(oid), None)
            if not isinstance(geom_self, dict):
                continue

            pair_specs.append(
                (
                    int(oid),
                    geom_self,
                    float(getattr(dg, "scale_min", 40.0)),
                    float(getattr(dg, "contact_margin_px", 2.0)),
                    float(getattr(dg, "near_thresh_n", 1.25)),
                    prepare_relation_mask_runtime(geom_self.get("mask", None), compute_bbox=False),
                )
            )

        if len(pair_specs) < 2:
            return out

        # Prepare immutable runtime fields once so worker threads only read.
        for _, geom_self, _, _, _, runtime_self in pair_specs:
            if not isinstance(runtime_self, dict):
                continue
            bbox_self = geom_self.get("bbox", None) if isinstance(geom_self, dict) else None
            if (
                runtime_self.get("bbox", None) is None
                and isinstance(bbox_self, (list, tuple))
                and len(bbox_self) >= 4
            ):
                runtime_self["bbox"] = (
                    int(bbox_self[0]),
                    int(bbox_self[1]),
                    int(bbox_self[2]),
                    int(bbox_self[3]),
                )
            if runtime_self.get("mask_bool", None) is None:
                mask_arr = runtime_self.get("mask", None)
                if hasattr(mask_arr, "astype"):
                    runtime_self["mask_bool"] = mask_arr.astype(bool, copy=False)

        pair_jobs = []
        for idx, (oid, geom_self, scale_min, contact_margin_px, near_thresh_n, mask_runtime_self) in enumerate(pair_specs):
            for other_id, geom_other, _, _, _, mask_runtime_other in pair_specs[idx + 1 :]:
                pair_jobs.append(
                    (
                        int(oid),
                        int(other_id),
                        geom_self,
                        geom_other,
                        float(scale_min),
                        float(contact_margin_px),
                        float(near_thresh_n),
                        mask_runtime_self,
                        mask_runtime_other,
                    )
                )

        if not pair_jobs:
            return out

        def _compute_pair(job):
            (
                oid,
                other_id,
                geom_self,
                geom_other,
                scale_min,
                contact_margin_px,
                near_thresh_n,
                mask_runtime_self,
                mask_runtime_other,
            ) = job
            obs = compute_relation_observation(
                geom_self,
                geom_other,
                scale_min=float(scale_min),
                contact_margin_px=float(contact_margin_px),
                near_thresh_n=float(near_thresh_n),
                mask_runtime_a=mask_runtime_self,
                mask_runtime_b=mask_runtime_other,
            )
            if not isinstance(obs, dict):
                return None
            return int(oid), int(other_id), obs

        workers = self._resolve_dist_observations_workers(pair_count=int(len(pair_jobs)))
        if workers <= 1:
            computed = map(_compute_pair, pair_jobs)
        else:
            with ThreadPoolExecutor(max_workers=int(workers)) as ex:
                computed = ex.map(_compute_pair, pair_jobs)
                for item in computed:
                    if item is None:
                        continue
                    oid, other_id, obs = item
                    out.setdefault(int(oid), {})[int(other_id)] = obs
                    out.setdefault(int(other_id), {})[int(oid)] = obs
            return out

        for item in computed:
            if item is None:
                continue
            oid, other_id, obs = item
            out.setdefault(int(oid), {})[int(other_id)] = obs
            out.setdefault(int(other_id), {})[int(oid)] = obs

        return out

    def _resolve_dist_observations_workers(self, *, pair_count: int) -> int:
        if not bool(self.dist_obs_parallel_enabled):
            return 1
        if int(pair_count) < int(self.dist_obs_parallel_min_pairs):
            return 1
        fixed = int(self.dist_obs_parallel_workers)
        if fixed > 0:
            return max(1, int(fixed))
        cpu = max(1, int(os.cpu_count() or 1))
        return max(1, min(int(cpu), int(self.dist_obs_parallel_max_auto_workers)))

    def _resolve_graph_updates_workers(self, *, object_count: int) -> int:
        if not bool(self.graph_updates_parallel_enabled):
            return 1
        if int(object_count) < int(self.graph_updates_parallel_min_objects):
            return 1
        fixed = int(self.graph_updates_parallel_workers)
        if fixed > 0:
            return max(1, int(fixed))
        cpu = max(1, int(os.cpu_count() or 1))
        return max(1, min(int(cpu), int(self.graph_updates_parallel_max_auto_workers)))

    def resolve_anchor_view_id(
        self,
        *,
        ids: list[int],
        geom: dict[int, dict],
        allow_episode_by_object_id: dict[int, bool],
    ) -> int | None:
        del ids, geom, allow_episode_by_object_id
        return None
