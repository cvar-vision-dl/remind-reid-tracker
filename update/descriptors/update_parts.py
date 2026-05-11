# update/update_parts.py

from __future__ import annotations

import numpy as np

from utils.math import l2_normalize_vector
from update.descriptors.proto_ops import (
    choose_evict_index,
    compute_sims_to_protos,
    ensure_channel_lists,
    find_best_internal_pair,
    make_proto_event,
    merge_two_protos_inplace,
    update_proto_dup_gated_ema,
)


class PartsUpdater:
    def __init__(self, config: dict):
        self.config = config or {}

        upd = (self.config.get("update", {}) or {})
        pm = (upd.get("parts_memory", {}) or {})

        self.enabled = bool(pm.get("enabled", True))

        self.dup_thr = float(pm.get("dup_thr", 0.93))
        self.novel_thr = float(pm.get("novel_thr", 0.78))
        self.merge_thr = float(pm.get("merge_thr", 0.92))

        self.insert_policy = str(pm.get("insert_policy", "always")).lower().strip()
        self.evict_strategy = str(pm.get("evict_strategy", "lru")).lower().strip()

        self.greedy_1to1 = bool(pm.get("greedy_1to1", True))

        self.promote_hits = int(pm.get("promote_hits", 0))
        self.stable_dup_thr = float(pm.get("stable_dup_thr", self.dup_thr))
        self.stable_merge_thr = float(pm.get("stable_merge_thr", self.merge_thr))

        dup_upd = (pm.get("dup_update", {}) or {})
        self.dup_margin = float(dup_upd.get("margin", 0.02))
        self.alpha_min = float(dup_upd.get("alpha_min", 0.03))
        self.alpha_max = float(dup_upd.get("alpha_max", 0.15))

        merge_cfg = (pm.get("merge", {}) or {})
        self.merge_weight = str(merge_cfg.get("merge_weight", "count_capped")).lower().strip()
        self.count_cap = int(merge_cfg.get("count_cap", 10))

    def parts_channel_capacity(self, obj, channel_name: str) -> int:
        ch = obj.parts.get_channel(channel_name)
        if ch is None:
            return 0

        k = getattr(ch, "max_prototypes", None)
        if k is None:
            k = getattr(obj.parts, "max_prototypes_per_channel", None)
        if k is None:
            return 0

        return int(k)

    def ensure_channel_lists(self, ch):
        return ensure_channel_lists(ch)

    def try_promote_work_proto(
        self,
        stable_protos: list,
        work_proto,
        timestamp: float,
        K: int,
        label: str | None,
        obj_id: int,
        channel_name: str,
        proto_events: list | None,
        decision,
    ) -> None:
        if decision is not None and not getattr(decision, "allow_promote", True):
            return
        if self.promote_hits <= 0:
            return
        if int(getattr(work_proto, "count", 1)) < int(self.promote_hits):
            return

        n_before = int(len(stable_protos))
        x = getattr(work_proto, "embedding", None)
        if x is None:
            return

        if n_before == 0:
            if decision is not None and not getattr(decision, "allow_insert", True):
                return
            stable_protos.append(type(work_proto)(embedding=x, timestamp=timestamp))
            if proto_events is not None:
                proto_events.append(make_proto_event("parts", label, obj_id, str(channel_name), "PROMOTE_INSERT", None, 0, int(len(stable_protos))))
            return

        sims = compute_sims_to_protos(x, stable_protos)
        nn_idx = int(np.argmax(sims))
        s_max = float(sims[nn_idx])

        if s_max >= self.stable_dup_thr:
            p = stable_protos[nn_idx]
            act = update_proto_dup_gated_ema(
                proto=p,
                x_norm=x,
                timestamp=timestamp,
                s_max=s_max,
                dup_thr=self.stable_dup_thr,
                margin=self.dup_margin,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                alpha_scale=float(getattr(decision, "alpha_scale", 1.0)) if decision is not None else 1.0,
            )
            if proto_events is not None:
                proto_events.append(make_proto_event("parts", label, obj_id, str(channel_name), f"STABLE_{act}", s_max, n_before, n_before))
            return

        if decision is not None and not getattr(decision, "allow_insert", True):
            return

        if K <= 0 or n_before < K:
            stable_protos.append(type(work_proto)(embedding=x, timestamp=timestamp))
            if proto_events is not None:
                proto_events.append(make_proto_event("parts", label, obj_id, str(channel_name), "PROMOTE_INSERT", s_max, n_before, int(len(stable_protos))))
            return

        if decision is not None and not getattr(decision, "allow_merge", True):
            return

        best_pair_sim, bi, bj = find_best_internal_pair(stable_protos)
        if best_pair_sim >= self.stable_merge_thr and bi >= 0 and bj >= 0:
            merge_two_protos_inplace(
                stable_protos[int(bi)],
                stable_protos[int(bj)],
                timestamp,
                merge_weight=self.merge_weight,
                count_cap=self.count_cap,
            )
            stable_protos.pop(int(bj))
            stable_protos.append(type(work_proto)(embedding=x, timestamp=timestamp))
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "parts",
                        label,
                        obj_id,
                        str(channel_name),
                        "PROMOTE_MERGE_INSERT",
                        s_max,
                        n_before,
                        int(len(stable_protos)),
                        merge_pair_sim=float(best_pair_sim),
                    )
                )
            return

        ev = choose_evict_index(stable_protos, strategy=self.evict_strategy)
        if ev >= 0:
            stable_protos.pop(int(ev))
        stable_protos.append(type(work_proto)(embedding=x, timestamp=timestamp))
        if proto_events is not None:
            proto_events.append(
                make_proto_event(
                    "parts",
                    label,
                    obj_id,
                    str(channel_name),
                    "PROMOTE_EVICT_INSERT",
                    s_max,
                    n_before,
                    int(len(stable_protos)),
                    merge_pair_sim=float(best_pair_sim) if best_pair_sim > -1.0 else None,
                    evict_strategy=self.evict_strategy,
                    evicted_index=int(ev),
                )
            )

    def update_parts_channel(
        self,
        obj,
        channel_name: str,
        part_descs: list,
        timestamp: float,
        proto_events: list | None = None,
        decision=None,
    ) -> None:
        ch = obj.parts.get_channel(channel_name)
        if ch is None:
            return

        work_protos, stable_protos = self.ensure_channel_lists(ch)

        label = getattr(obj, "instance_label", None)
        obj_id = int(getattr(obj, "object_id", -1))
        K = self.parts_channel_capacity(obj, channel_name)

        if decision is not None and str(getattr(decision, "mode", "FULL")) == "OFF":
            return

        used_work_idx = set()

        for pd in (part_descs or []):
            x = l2_normalize_vector(np.asarray(pd, dtype=np.float32).reshape(-1))
            n_before = int(len(work_protos))

            if not self.enabled:
                ch.add_work_prototype(x, timestamp)
                if proto_events is not None:
                    proto_events.append(make_proto_event("parts", label, obj_id, str(channel_name), "ADD_RAW", None, n_before, int(len(work_protos))))
                continue

            if n_before == 0:
                if decision is not None and not getattr(decision, "allow_insert", True):
                    continue
                ch.add_work_prototype(x, timestamp)
                if proto_events is not None:
                    proto_events.append(make_proto_event("parts", label, obj_id, str(channel_name), "INSERT", None, 0, int(len(work_protos))))
                continue

            sims = compute_sims_to_protos(x, work_protos)
            nn_idx = int(np.argmax(sims))
            s_max = float(sims[nn_idx])

            if self.greedy_1to1 and nn_idx in used_work_idx and len(work_protos) > 1:
                order = np.argsort(-sims)
                for j in order:
                    j = int(j)
                    if j not in used_work_idx:
                        nn_idx = j
                        s_max = float(sims[nn_idx])
                        break

            if s_max >= self.dup_thr:
                used_work_idx.add(int(nn_idx))
                p = work_protos[int(nn_idx)]

                act = update_proto_dup_gated_ema(
                    proto=p,
                    x_norm=x,
                    timestamp=timestamp,
                    s_max=s_max,
                    dup_thr=self.dup_thr,
                    margin=self.dup_margin,
                    alpha_min=self.alpha_min,
                    alpha_max=self.alpha_max,
                    alpha_scale=float(getattr(decision, "alpha_scale", 1.0)) if decision is not None else 1.0,
                )

                if decision is not None and not getattr(decision, "allow_ema", True):
                    act = "DUP_COUNT"

                if proto_events is not None:
                    proto_events.append(make_proto_event("parts", label, obj_id, str(channel_name), act, s_max, n_before, n_before))

                self.try_promote_work_proto(
                    stable_protos=stable_protos,
                    work_proto=p,
                    timestamp=float(timestamp),
                    K=K,
                    label=label,
                    obj_id=obj_id,
                    channel_name=str(channel_name),
                    proto_events=proto_events,
                    decision=decision,
                )
                continue

            if decision is not None and str(getattr(decision, "mode", "FULL")) == "SAFE":
                continue

            if K > 0 and n_before >= K and self.insert_policy == "novel_only":
                if s_max > self.novel_thr:
                    if proto_events is not None:
                        proto_events.append(make_proto_event("parts", label, obj_id, str(channel_name), "SKIP_NOT_NOVEL", s_max, n_before, n_before))
                    continue

            if decision is not None and not getattr(decision, "allow_insert", True):
                continue

            if K <= 0 or n_before < K:
                ch.add_work_prototype(x, timestamp)
                if proto_events is not None:
                    proto_events.append(make_proto_event("parts", label, obj_id, str(channel_name), "INSERT", s_max, n_before, int(len(work_protos))))
                continue

            if decision is not None and not getattr(decision, "allow_merge", True):
                continue

            best_pair_sim, bi, bj = find_best_internal_pair(work_protos)
            if best_pair_sim >= self.merge_thr and bi >= 0 and bj >= 0:
                merge_two_protos_inplace(
                    work_protos[int(bi)],
                    work_protos[int(bj)],
                    timestamp,
                    merge_weight=self.merge_weight,
                    count_cap=self.count_cap,
                )
                work_protos.pop(int(bj))
                ch.add_work_prototype(x, timestamp)

                if proto_events is not None:
                    proto_events.append(
                        make_proto_event(
                            "parts",
                            label,
                            obj_id,
                            str(channel_name),
                            "MERGE_INSERT",
                            s_max,
                            n_before,
                            int(len(work_protos)),
                            merge_pair_sim=float(best_pair_sim),
                        )
                    )
                continue

            ev = choose_evict_index(work_protos, strategy=self.evict_strategy)
            if ev >= 0:
                work_protos.pop(int(ev))
            ch.add_work_prototype(x, timestamp)

            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "parts",
                        label,
                        obj_id,
                        str(channel_name),
                        "EVICT_INSERT",
                        s_max,
                        n_before,
                        int(len(work_protos)),
                        merge_pair_sim=float(best_pair_sim) if best_pair_sim > -1.0 else None,
                        evict_strategy=self.evict_strategy,
                        evicted_index=int(ev),
                    )
                )

    def update_from_det_feats(self, obj, det_feats: dict, timestamp: float, proto_events: list | None = None, decision=None) -> None:
        if obj is None or det_feats is None:
            return
        if not getattr(obj, "parts", None) or not getattr(obj.parts, "enabled", False):
            return

        obs_parts = det_feats.get("parts", None)
        if not isinstance(obs_parts, dict):
            return

        for ch_name in obj.parts.channel_names():
            pack = obs_parts.get(ch_name, None)
            if not isinstance(pack, dict):
                continue
            part_descs = pack.get("part_descs", []) or []
            self.update_parts_channel(
                obj,
                str(ch_name),
                part_descs,
                float(timestamp),
                proto_events=proto_events,
                decision=decision,
            )
