# update/update_object.py

from __future__ import annotations

import numpy as np

from utils.math import l2_normalize_vector
from update.descriptors.proto_ops import (
    choose_evict_index,
    compute_sims_to_protos,
    find_best_internal_pair,
    make_proto_event,
    update_proto_dup_gated_ema,
)


class ObjectUpdater:
    def __init__(self, config: dict):
        upd = (config.get("update", {}) or {})
        ap = (upd.get("appearance_memory", {}) or {})

        self.enabled = bool(ap.get("enabled", True))

        self.dup_thr = float(ap.get("dup_thr", 0.92))
        self.novel_thr = float(ap.get("novel_thr", 0.78))
        self.merge_thr = float(ap.get("merge_thr", 0.90))

        self.insert_policy = str(ap.get("insert_policy", "always")).lower().strip()
        self.evict_strategy = str(ap.get("evict_strategy", "redundant")).lower().strip()
        self.update_nn_on_skip = bool(ap.get("update_nn_on_skip", True))

        self.merge_weight = str(ap.get("merge_weight", "equal")).lower().strip()
        self.count_cap = int(ap.get("count_cap", 10))

        dup_upd = (ap.get("dup_update", {}) or {})
        self.ema_on_dup = bool(dup_upd.get("ema_on_dup", False))
        self.dup_margin = float(dup_upd.get("margin", 0.02))
        self.alpha_min = float(dup_upd.get("alpha_min", 0.02))
        self.alpha_max = float(dup_upd.get("alpha_max", 0.08))

        self.promote_hits = int(ap.get("promote_hits", 0))
        self.stable_dup_thr = float(ap.get("stable_dup_thr", self.dup_thr))
        self.stable_merge_thr = float(ap.get("stable_merge_thr", self.merge_thr))
        self.stable_evict_strategy = str(ap.get("stable_evict_strategy", self.evict_strategy)).lower().strip()

        self.quality_min_patches = max(1, int(ap.get("quality_min_patches", 8)))
        self.quality_full_patches = max(self.quality_min_patches, int(ap.get("quality_full_patches", 24)))
        self.create_min_quality = max(0.0, min(1.0, float(ap.get("create_min_quality", 0.35))))
        self.promote_min_quality = max(0.0, min(1.0, float(ap.get("promote_min_quality", 0.55))))
        self.work_update_min_quality = max(0.0, min(1.0, float(ap.get("work_update_min_quality", 0.15))))
        self.stable_update_min_quality = max(0.0, min(1.0, float(ap.get("stable_update_min_quality", 0.20))))

    def appearance_work_capacity(self, obj, channel_name: str) -> int:
        ch = obj.appearance.get_channel(channel_name)
        if ch is None:
            return 0
        k = getattr(ch, "max_prototypes", None)
        if k is None:
            k = getattr(obj.appearance, "max_prototypes_per_channel", None)
        return 0 if k is None else int(k)

    def appearance_stable_capacity(self, obj, channel_name: str) -> int:
        ch = obj.appearance.get_channel(channel_name)
        if ch is None:
            return 0
        k = getattr(ch, "max_stable", None)
        if k is None:
            k = getattr(obj.appearance, "max_stable_prototypes_per_channel", None)
        return 0 if k is None else int(k)

    def ensure_channel_lists(self, ch):
        work = getattr(ch, "work_protos", None)
        stable = getattr(ch, "stable_protos", None)

        if work is None:
            work = []
            ch.work_protos = work

        if stable is None:
            stable = []
            ch.stable_protos = stable

        return work, stable

    def ramp(self, x: float | int | None, x0: float, x1: float) -> float:
        if x is None:
            return 0.0
        xv = float(x)
        if x1 <= x0:
            return 1.0 if xv >= x1 else 0.0
        return float(max(0.0, min(1.0, (xv - x0) / (x1 - x0))))

    def observation_quality(self, det_feats: dict | None, channel_name: str) -> float:
        meta = ((det_feats or {}).get("meta", {}) or {})
        obj_support = meta.get("effective_obj_patches", None)
        if obj_support is None:
            obj_support = meta.get("n_obj_patches", None)
        q = self.ramp(obj_support, self.quality_min_patches, self.quality_full_patches)
        # global_trimmed tolera un poco mejor observaciones parciales.
        if str(channel_name) == "global_trimmed":
            q = min(1.0, 0.10 + 0.90 * q)
        return float(max(0.0, min(1.0, q)))

    def proto_mean_quality(self, proto) -> float:
        n_obs = max(1, int(getattr(proto, "n_obs", 1)))
        q_sum = float(getattr(proto, "quality_sum", getattr(proto, "quality_ema", 1.0)))
        return float(max(0.0, min(1.0, q_sum / float(n_obs))))

    def proto_confidence(self, proto) -> float:
        support_ref = max(1, int(self.promote_hits if self.promote_hits > 0 else 4))
        support = min(1.0, float(getattr(proto, "count", 1)) / float(support_ref))
        quality = max(self.proto_mean_quality(proto), float(getattr(proto, "quality_ema", 0.0)))
        return float(max(0.0, min(1.0, support * quality)))

    def refresh_proto_quality(self, proto, obs_quality: float) -> None:
        q = float(max(0.0, min(1.0, obs_quality)))
        prev_n = max(1, int(getattr(proto, "n_obs", 1)))
        proto.quality_sum = float(getattr(proto, "quality_sum", getattr(proto, "quality_ema", 1.0))) + q
        proto.n_obs = int(prev_n + 1)
        beta = max(0.08, min(0.30, 1.0 / float(proto.n_obs)))
        prev_q = float(getattr(proto, "quality_ema", q))
        proto.quality_ema = float((1.0 - beta) * prev_q + beta * q)
        proto.stability = float(self.proto_confidence(proto))

    def merge_object_protos_inplace(self, dst, src, timestamp: float) -> None:
        a = dst.embedding
        b = src.embedding

        mw = str(self.merge_weight).lower().strip()
        if mw == "count_capped":
            wa = float(min(int(getattr(dst, "count", 1)), int(max(1, self.count_cap))))
            wb = float(min(int(getattr(src, "count", 1)), int(max(1, self.count_cap))))
        else:
            wa, wb = 1.0, 1.0

        dst.embedding = l2_normalize_vector((wa * a) + (wb * b))
        dst.count = int(getattr(dst, "count", 1)) + int(getattr(src, "count", 1))
        dst.last_seen = float(
            max(
                float(getattr(dst, "last_seen", 0.0)),
                float(getattr(src, "last_seen", 0.0)),
                float(timestamp),
            )
        )
        dst.first_seen = float(
            min(
                float(getattr(dst, "first_seen", dst.last_seen)),
                float(getattr(src, "first_seen", dst.last_seen)),
            )
        )
        dst.quality_sum = float(getattr(dst, "quality_sum", 1.0)) + float(getattr(src, "quality_sum", 1.0))
        dst.n_obs = int(getattr(dst, "n_obs", 1)) + int(getattr(src, "n_obs", 1))
        denom = max(1.0, float(getattr(dst, "n_obs", 1)))
        dst.quality_ema = float(max(0.0, min(1.0, dst.quality_sum / denom)))
        dst.stability = float(self.proto_confidence(dst))

    def update_proto_on_dup(
        self,
        proto,
        x_norm: np.ndarray,
        timestamp: float,
        s_max: float,
        dup_thr: float,
        decision,
        obs_quality: float,
        min_update_quality: float,
    ) -> str:
        allow_embed = bool(decision is None or getattr(decision, "allow_ema", True))
        allow_embed = allow_embed and float(obs_quality) >= float(min_update_quality)

        if not allow_embed:
            proto.count = int(getattr(proto, "count", 1)) + 1
            proto.last_seen = float(timestamp)
            self.refresh_proto_quality(proto, obs_quality)
            return "DUP_COUNT"

        if self.ema_on_dup:
            act = update_proto_dup_gated_ema(
                proto=proto,
                x_norm=x_norm,
                timestamp=timestamp,
                s_max=s_max,
                dup_thr=dup_thr,
                margin=self.dup_margin,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                alpha_scale=float(getattr(decision, "alpha_scale", 1.0)) * float(obs_quality) if decision is not None else float(obs_quality),
            )
            self.refresh_proto_quality(proto, obs_quality)
            return act

        proto.count = int(getattr(proto, "count", 1)) + 1
        proto.last_seen = float(timestamp)
        proto.embedding = l2_normalize_vector(
            ((1.0 - float(obs_quality)) * proto.embedding) + (float(obs_quality) * x_norm)
        )
        self.refresh_proto_quality(proto, obs_quality)
        return "DUP_EMA"

    def clone_proto(self, proto):
        clone = proto.copy()
        clone.stability = float(self.proto_confidence(clone))
        return clone

    def insert_work_proto(self, ch, x: np.ndarray, timestamp: float, obs_quality: float) -> None:
        ch.add_work_prototype(
            x,
            timestamp,
            quality_ema=float(obs_quality),
            quality_sum=float(obs_quality),
            n_obs=1,
            count=1,
            first_seen=float(timestamp),
        )
        ch.work_protos[-1].stability = float(self.proto_confidence(ch.work_protos[-1]))

    def event_extra(self, proto=None, obs_quality: float | None = None) -> dict:
        extra = {}
        if obs_quality is not None:
            extra["obs_q"] = round(float(obs_quality), 3)
        if proto is not None:
            extra["proto_q"] = round(float(self.proto_mean_quality(proto)), 3)
            extra["proto_c"] = round(float(self.proto_confidence(proto)), 3)
        return extra

    def try_promote_work_proto(
        self,
        obj,
        channel_name: str,
        work_proto,
        work_idx: int,
        work_protos: list,
        stable_protos: list,
        timestamp: float,
        proto_events: list | None,
        decision,
    ) -> bool:
        if decision is not None and not getattr(decision, "allow_promote", True):
            return False
        if self.promote_hits <= 0:
            return False
        if int(getattr(work_proto, "count", 1)) < int(self.promote_hits):
            return False
        if float(self.proto_mean_quality(work_proto)) < float(self.promote_min_quality):
            return False

        x = getattr(work_proto, "embedding", None)
        if x is None:
            return False

        label = getattr(obj, "instance_label", None)
        obj_id = int(getattr(obj, "object_id", -1))
        ch = obj.appearance.get_channel(channel_name)
        if ch is None:
            return False

        n_before = int(len(stable_protos))
        K = self.appearance_stable_capacity(obj, channel_name)
        promoted = self.clone_proto(work_proto)

        if n_before == 0:
            ch.append_stable_proto(promoted)
            work_protos.pop(int(work_idx))
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "obj",
                        label,
                        obj_id,
                        channel_name,
                        "PROMOTE_INSERT",
                        None,
                        0,
                        int(len(stable_protos)),
                        extra=self.event_extra(promoted),
                    )
                )
            return True

        sims = compute_sims_to_protos(x, stable_protos)
        nn_idx = int(np.argmax(sims))
        s_max = float(sims[nn_idx])

        if s_max >= self.stable_dup_thr:
            self.merge_object_protos_inplace(stable_protos[nn_idx], promoted, timestamp)
            work_protos.pop(int(work_idx))
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "obj",
                        label,
                        obj_id,
                        channel_name,
                        "PROMOTE_MERGE_STABLE",
                        s_max,
                        n_before,
                        n_before,
                        extra=self.event_extra(stable_protos[nn_idx]),
                    )
                )
            return True

        if decision is not None and not getattr(decision, "allow_insert", True):
            return False

        if K <= 0 or n_before < K:
            ch.append_stable_proto(promoted)
            work_protos.pop(int(work_idx))
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "obj",
                        label,
                        obj_id,
                        channel_name,
                        "PROMOTE_INSERT",
                        s_max,
                        n_before,
                        int(len(stable_protos)),
                        extra=self.event_extra(promoted),
                    )
                )
            return True

        if decision is not None and not getattr(decision, "allow_merge", True):
            return False

        best_pair_sim, bi, bj = find_best_internal_pair(stable_protos)
        if best_pair_sim >= self.stable_merge_thr and bi >= 0 and bj >= 0:
            self.merge_object_protos_inplace(stable_protos[bi], stable_protos[bj], timestamp)
            stable_protos.pop(bj)
            ch.append_stable_proto(promoted)
            work_protos.pop(int(work_idx))
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "obj",
                        label,
                        obj_id,
                        channel_name,
                        "PROMOTE_MERGE_INSERT",
                        s_max,
                        n_before,
                        int(len(stable_protos)),
                        merge_pair_sim=float(best_pair_sim),
                        extra=self.event_extra(promoted),
                    )
                )
            return True

        ev = choose_evict_index(stable_protos, strategy=self.stable_evict_strategy)
        if ev >= 0:
            stable_protos.pop(int(ev))
        ch.append_stable_proto(promoted)
        work_protos.pop(int(work_idx))

        if proto_events is not None:
            proto_events.append(
                make_proto_event(
                    "obj",
                    label,
                    obj_id,
                    channel_name,
                    "PROMOTE_EVICT_INSERT",
                    s_max,
                    n_before,
                    int(len(stable_protos)),
                    merge_pair_sim=float(best_pair_sim) if best_pair_sim > -1.0 else None,
                    evict_strategy=self.stable_evict_strategy,
                    evicted_index=int(ev),
                    extra=self.event_extra(promoted),
                )
            )
        return True

    def update_appearance_channel(
        self,
        obj,
        channel_name: str,
        embedding: np.ndarray,
        timestamp: float,
        det_feats: dict | None,
        proto_events: list | None,
        decision,
    ) -> None:
        ch = obj.appearance.get_channel(channel_name)
        if ch is None:
            return

        work_protos, stable_protos = self.ensure_channel_lists(ch)

        x = l2_normalize_vector(np.asarray(embedding, dtype=np.float32).reshape(-1))
        obs_quality = self.observation_quality(det_feats, channel_name)
        label = getattr(obj, "instance_label", None)
        obj_id = int(getattr(obj, "object_id", -1))

        if decision is not None and str(getattr(decision, "mode", "FULL")) == "OFF":
            return

        if not self.enabled:
            self.insert_work_proto(ch, x, timestamp, obs_quality)
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "obj",
                        label,
                        obj_id,
                        channel_name,
                        "ADD_RAW",
                        None,
                        int(len(work_protos) - 1),
                        int(len(work_protos)),
                        extra=self.event_extra(ch.work_protos[-1], obs_quality),
                    )
                )
            return

        s_stable_max = None
        if stable_protos:
            stable_sims = compute_sims_to_protos(x, stable_protos)
            stable_idx = int(np.argmax(stable_sims))
            s_stable_max = float(stable_sims[stable_idx])
            if s_stable_max >= self.stable_dup_thr:
                act = self.update_proto_on_dup(
                    stable_protos[stable_idx],
                    x,
                    timestamp,
                    s_stable_max,
                    self.stable_dup_thr,
                    decision,
                    obs_quality=obs_quality,
                    min_update_quality=self.stable_update_min_quality,
                )
                if proto_events is not None:
                    proto_events.append(
                        make_proto_event(
                            "obj",
                            label,
                            obj_id,
                            channel_name,
                            f"STABLE_{act}",
                            s_stable_max,
                            int(len(stable_protos)),
                            int(len(stable_protos)),
                            extra=self.event_extra(stable_protos[stable_idx], obs_quality),
                        )
                    )
                return

        n_before = int(len(work_protos))
        s_work_max = None
        if work_protos:
            sims = compute_sims_to_protos(x, work_protos)
            nn_idx = int(np.argmax(sims))
            s_work_max = float(sims[nn_idx])

            if s_work_max >= self.dup_thr:
                if not self.update_nn_on_skip:
                    if proto_events is not None:
                        proto_events.append(
                            make_proto_event(
                                "obj",
                                label,
                                obj_id,
                                channel_name,
                                "DUP_SKIP",
                                s_work_max,
                                n_before,
                                n_before,
                                extra=self.event_extra(work_protos[nn_idx], obs_quality),
                            )
                        )
                    return

                act = self.update_proto_on_dup(
                    work_protos[nn_idx],
                    x,
                    timestamp,
                    s_work_max,
                    self.dup_thr,
                    decision,
                    obs_quality=obs_quality,
                    min_update_quality=self.work_update_min_quality,
                )
                if proto_events is not None:
                    proto_events.append(
                        make_proto_event(
                            "obj",
                            label,
                            obj_id,
                            channel_name,
                            act,
                            s_work_max,
                            n_before,
                            n_before,
                            extra=self.event_extra(work_protos[nn_idx], obs_quality),
                        )
                    )

                self.try_promote_work_proto(
                    obj=obj,
                    channel_name=channel_name,
                    work_proto=work_protos[nn_idx],
                    work_idx=nn_idx,
                    work_protos=work_protos,
                    stable_protos=stable_protos,
                    timestamp=float(timestamp),
                    proto_events=proto_events,
                    decision=decision,
                )
                return

        if decision is not None and str(getattr(decision, "mode", "FULL")) == "SAFE":
            return

        if float(obs_quality) < float(self.create_min_quality):
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "obj",
                        label,
                        obj_id,
                        channel_name,
                        "SKIP_LOW_QUALITY",
                        max(s for s in [s_work_max, s_stable_max] if s is not None) if any(s is not None for s in [s_work_max, s_stable_max]) else None,
                        n_before,
                        n_before,
                        extra={"obs_q": round(float(obs_quality), 3)},
                    )
                )
            return

        if decision is not None and not getattr(decision, "allow_insert", True):
            return

        known_sims = [float(s) for s in [s_work_max, s_stable_max] if s is not None]
        best_known = max(known_sims) if known_sims else None
        if self.insert_policy == "novel_only" and best_known is not None and float(best_known) > self.novel_thr:
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "obj",
                        label,
                        obj_id,
                        channel_name,
                        "SKIP_NOT_NOVEL",
                        best_known,
                        n_before,
                        n_before,
                        extra={"obs_q": round(float(obs_quality), 3)},
                    )
                )
            return

        K = self.appearance_work_capacity(obj, channel_name)

        if K <= 0 or n_before < K:
            self.insert_work_proto(ch, x, timestamp, obs_quality)
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "obj",
                        label,
                        obj_id,
                        channel_name,
                        "INSERT",
                        best_known,
                        n_before,
                        int(len(work_protos)),
                        extra=self.event_extra(work_protos[-1], obs_quality),
                    )
                )
            return

        if decision is not None and not getattr(decision, "allow_merge", True):
            return

        best_pair_sim, bi, bj = find_best_internal_pair(work_protos)
        if best_pair_sim >= self.merge_thr and bi >= 0 and bj >= 0:
            self.merge_object_protos_inplace(work_protos[bi], work_protos[bj], timestamp)
            work_protos.pop(bj)
            self.insert_work_proto(ch, x, timestamp, obs_quality)
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "obj",
                        label,
                        obj_id,
                        channel_name,
                        "MERGE_INSERT",
                        best_known,
                        n_before,
                        int(len(work_protos)),
                        merge_pair_sim=float(best_pair_sim),
                        extra=self.event_extra(work_protos[-1], obs_quality),
                    )
                )
            return

        ev = choose_evict_index(work_protos, strategy=self.evict_strategy)
        if ev >= 0:
            work_protos.pop(int(ev))
        self.insert_work_proto(ch, x, timestamp, obs_quality)

        if proto_events is not None:
            proto_events.append(
                make_proto_event(
                    "obj",
                    label,
                    obj_id,
                    channel_name,
                    "EVICT_INSERT",
                    best_known,
                    n_before,
                    int(len(work_protos)),
                    merge_pair_sim=float(best_pair_sim) if best_pair_sim > -1.0 else None,
                    evict_strategy=self.evict_strategy,
                    evicted_index=int(ev),
                    extra=self.event_extra(work_protos[-1], obs_quality),
                )
            )

    def update_from_det_feats(self, obj, det_feats: dict, timestamp: float, proto_events: list | None = None, decision=None):
        obs_obj = det_feats.get("obj", None)
        if not isinstance(obs_obj, dict):
            return
        if not getattr(obj.appearance, "enabled", False):
            return

        for ch_name in obj.appearance.channel_names():
            pack = obs_obj.get(ch_name, None)
            if not isinstance(pack, dict):
                continue
            desc = pack.get("desc", None)
            if desc is None:
                continue
            self.update_appearance_channel(
                obj,
                str(ch_name),
                desc,
                float(timestamp),
                det_feats=det_feats,
                proto_events=proto_events,
                decision=decision,
            )
