# update/update_background.py

from __future__ import annotations

import numpy as np

from utils.math import l2_normalize_vector
from utils.math import ramp as linear_ramp
from utils.config import bg_partials_enabled
from update.descriptors.proto_ops import (
    compute_sims_to_protos,
    find_best_internal_pair,
    choose_evict_index,
    merge_two_protos_inplace,
    update_proto_dup_gated_ema,
    make_proto_event,
)


class BackgroundUpdater:
    def __init__(self, config: dict):
        upd = (config.get("update", {}) or {})
        bg = (upd.get("background_memory", {}) or {})

        self.enabled = bool(bg.get("enabled", True))
        self.partials_enabled = bool(bg_partials_enabled(config))

        self.work_dup_thr = float(bg.get("work_dup_thr", 0.93))
        self.work_merge_thr = float(bg.get("work_merge_thr", 0.94))
        self.work_evict_strategy = str(bg.get("work_evict_strategy", "redundant")).lower().strip()

        self.promote_hits_global = int(bg.get("promote_hits_global", bg.get("promote_hits", 5)))
        self.promote_hits_partials = int(bg.get("promote_hits_partials", bg.get("promote_hits", 5)))

        self.stable_dup_thr = float(bg.get("stable_dup_thr", 0.95))
        self.stable_merge_thr = float(bg.get("stable_merge_thr", 0.96))
        self.stable_update = str(bg.get("stable_update", "count_capped")).lower().strip()

        self.merge_weight = "count_capped"
        self.count_cap = int(bg.get("count_cap", 10))

        dup_upd = (bg.get("dup_update", {}) or {})
        self.dup_margin = float(dup_upd.get("margin", 0.02))
        self.alpha_min = float(dup_upd.get("alpha_min", 0.02))
        self.alpha_max = float(dup_upd.get("alpha_max", 0.10))

        dup_upd_s = (bg.get("stable_dup_update", {}) or {})
        self.stable_dup_margin = float(dup_upd_s.get("margin", self.dup_margin))
        self.stable_alpha_min = float(dup_upd_s.get("alpha_min", 0.01))
        self.stable_alpha_max = float(dup_upd_s.get("alpha_max", 0.05))

        self.global_create_min_quality = max(0.0, min(1.0, float(bg.get("global_create_min_quality", 0.40))))
        self.global_promote_min_quality = max(0.0, min(1.0, float(bg.get("global_promote_min_quality", 0.60))))
        self.global_work_update_min_quality = max(0.0, min(1.0, float(bg.get("global_work_update_min_quality", 0.20))))
        self.global_stable_update_min_quality = max(0.0, min(1.0, float(bg.get("global_stable_update_min_quality", 0.25))))
        self.global_inner_min_patches = max(1, int(bg.get("global_inner_min_patches", 8)))
        self.global_inner_full_patches = max(self.global_inner_min_patches, int(bg.get("global_inner_full_patches", 20)))
        self.global_outer_min_patches = max(1, int(bg.get("global_outer_min_patches", 12)))
        self.global_outer_full_patches = max(self.global_outer_min_patches, int(bg.get("global_outer_full_patches", 28)))
        self.global_min_mask_quality = max(0.0, min(1.0, float(bg.get("global_min_mask_quality", 0.5))))

    def ensure_bank_list(self, bank):
        protos = getattr(bank, "prototypes", None)
        if protos is None:
            protos = []
            bank.prototypes = protos
        return protos

    def ramp(self, x: float | int | None, x0: float, x1: float) -> float:
        return linear_ramp(x, x0, x1)

    def bank_capacity(self, bank) -> int:
        return int(getattr(bank, "max_size", 0))

    def scope(self, bank: str, region: str, ring: str) -> dict:
        return {"bank": str(bank), "region": str(region), "ring": str(ring)}

    def global_observation_quality(self, det_feats: dict | None, ring: str) -> float:
        meta = ((det_feats or {}).get("meta", {}) or {})
        bg_mask_quality = float(meta.get("bg_mask_quality", 1.0) or 1.0)
        if str(ring).lower().strip() == "inner":
            n_patches = meta.get("n_bg_inner_patches", None)
            q_ring = self.ramp(n_patches, self.global_inner_min_patches, self.global_inner_full_patches)
        else:
            n_patches = meta.get("n_bg_outer_patches", None)
            q_ring = self.ramp(n_patches, self.global_outer_min_patches, self.global_outer_full_patches)
        q_mask = self.ramp(bg_mask_quality, self.global_min_mask_quality, 1.0)
        return float(max(0.0, min(1.0, 0.7 * q_ring + 0.3 * q_mask)))

    def proto_mean_quality(self, proto) -> float:
        n_obs = max(1, int(getattr(proto, "n_obs", 1)))
        q_sum = float(getattr(proto, "quality_sum", getattr(proto, "quality_ema", 1.0)))
        return float(max(0.0, min(1.0, q_sum / float(n_obs))))

    def refresh_proto_quality(self, proto, obs_quality: float) -> None:
        q = float(max(0.0, min(1.0, obs_quality)))
        proto.quality_sum = float(getattr(proto, "quality_sum", getattr(proto, "quality_ema", 1.0))) + q
        proto.n_obs = int(max(1, int(getattr(proto, "n_obs", 1))) + 1)
        beta = max(0.08, min(0.30, 1.0 / float(proto.n_obs)))
        prev_q = float(getattr(proto, "quality_ema", q))
        proto.quality_ema = float((1.0 - beta) * prev_q + beta * q)

    def merge_bg_protos_inplace(self, dst, src, timestamp: float) -> None:
        merge_two_protos_inplace(dst, src, timestamp, merge_weight=self.merge_weight, count_cap=self.count_cap)
        dst.weight = float(max(float(getattr(dst, "weight", 1.0)), float(getattr(src, "weight", 1.0))))
        dst.quality_sum = float(getattr(dst, "quality_sum", 1.0)) + float(getattr(src, "quality_sum", 1.0))
        dst.n_obs = int(getattr(dst, "n_obs", 1)) + int(getattr(src, "n_obs", 1))
        dst.quality_ema = float(max(0.0, min(1.0, dst.quality_sum / max(1.0, float(dst.n_obs)))))

    def event_extra(self, proto=None, obs_quality: float | None = None) -> dict:
        extra = {}
        if obs_quality is not None:
            extra["obs_q"] = round(float(obs_quality), 3)
        if proto is not None:
            extra["proto_q"] = round(float(self.proto_mean_quality(proto)), 3)
        return extra

    def insert_global_work_proto(self, bank, x_norm: np.ndarray, timestamp: float, obs_quality: float, weight: float = 1.0):
        return bank.add(
            x_norm,
            timestamp,
            weight=float(weight),
            quality_ema=float(obs_quality),
            quality_sum=float(obs_quality),
            n_obs=1,
            count=1,
        )

    def clone_proto(self, proto):
        return proto.copy()

    def update_bank_work(
        self,
        bank,
        x_norm: np.ndarray,
        timestamp: float,
        label: str | None,
        obj_id: int,
        region: str,
        ring: str,
        proto_events: list | None,
        decision,
    ):
        protos = self.ensure_bank_list(bank)
        n_before = int(len(protos))

        ch = str(region).lower().strip()
        sc = self.scope("work", ch, str(ring).lower().strip())

        if decision is not None and str(getattr(decision, "mode", "FULL")) == "OFF":
            return

        if n_before == 0:
            if decision is not None and not getattr(decision, "allow_insert", True):
                return
            bank.add(x_norm, timestamp, weight=1.0)
            if proto_events is not None:
                proto_events.append(make_proto_event("bg", label, obj_id, ch, "INSERT", None, 0, int(len(protos)), scope=sc))
            return

        sims = compute_sims_to_protos(x_norm, protos)
        nn_idx = int(np.argmax(sims))
        s_max = float(sims[nn_idx])

        if s_max >= self.work_dup_thr:
            p = protos[nn_idx]
            act = update_proto_dup_gated_ema(
                proto=p,
                x_norm=x_norm,
                timestamp=timestamp,
                s_max=s_max,
                dup_thr=self.work_dup_thr,
                margin=self.dup_margin,
                alpha_min=self.alpha_min,
                alpha_max=self.alpha_max,
                alpha_scale=float(getattr(decision, "alpha_scale", 1.0)) if decision is not None else 1.0,
            )
            if decision is not None and not getattr(decision, "allow_ema", True):
                act = "DUP_COUNT"
            if proto_events is not None:
                proto_events.append(make_proto_event("bg", label, obj_id, ch, act, s_max, n_before, n_before, scope=sc))
            return

        if decision is not None and str(getattr(decision, "mode", "FULL")) == "SAFE":
            return

        K = self.bank_capacity(bank)

        if K <= 0 or n_before < K:
            bank.add(x_norm, timestamp, weight=1.0)
            if proto_events is not None:
                proto_events.append(make_proto_event("bg", label, obj_id, ch, "INSERT", s_max, n_before, int(len(protos)), scope=sc))
            return

        best_pair_sim, bi, bj = find_best_internal_pair(protos)
        if best_pair_sim >= self.work_merge_thr and bi >= 0 and bj >= 0:
            merge_two_protos_inplace(protos[bi], protos[bj], timestamp, merge_weight=self.merge_weight, count_cap=self.count_cap)
            protos.pop(bj)
            bank.add(x_norm, timestamp, weight=1.0)
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "bg",
                        label,
                        obj_id,
                        ch,
                        "MERGE_INSERT",
                        s_max,
                        n_before,
                        int(len(protos)),
                        scope=sc,
                        merge_pair_sim=float(best_pair_sim),
                    )
                )
            return

        ev = choose_evict_index(protos, strategy=self.work_evict_strategy)
        if ev >= 0:
            protos.pop(int(ev))
        bank.add(x_norm, timestamp, weight=1.0)

        if proto_events is not None:
            proto_events.append(
                make_proto_event(
                    "bg",
                    label,
                    obj_id,
                    ch,
                    "EVICT_INSERT",
                    s_max,
                    n_before,
                    int(len(protos)),
                    scope=sc,
                    merge_pair_sim=float(best_pair_sim) if best_pair_sim > -1.0 else None,
                    evict_strategy=self.work_evict_strategy,
                    evicted_index=ev,
                )
            )

    def try_promote_to_stable(
        self,
        stable_bank,
        work_proto,
        timestamp: float,
        promote_hits: int,
        label: str | None,
        obj_id: int,
        region: str,
        ring: str,
        proto_events: list | None,
        decision,
    ):
        if decision is not None and not getattr(decision, "allow_promote", True):
            return
        if int(getattr(work_proto, "count", 1)) < int(promote_hits):
            return

        stable_protos = self.ensure_bank_list(stable_bank)
        x = work_proto.embedding

        ch = str(region).lower().strip()
        sc = self.scope("stable", ch, str(ring).lower().strip())
        n_before = int(len(stable_protos))

        if stable_protos:
            sims = compute_sims_to_protos(x, stable_protos)
            nn_idx = int(np.argmax(sims))
            s_max = float(sims[nn_idx])

            if s_max >= self.stable_dup_thr:
                p = stable_protos[nn_idx]

                if self.stable_update in ("none", "off", "skip") or (decision is not None and not getattr(decision, "allow_ema", True)):
                    p.count = int(getattr(p, "count", 1)) + 1
                    p.last_seen = float(timestamp)
                    act = "DUP_COUNT"
                else:
                    act = update_proto_dup_gated_ema(
                        proto=p,
                        x_norm=x,
                        timestamp=timestamp,
                        s_max=s_max,
                        dup_thr=self.stable_dup_thr,
                        margin=self.stable_dup_margin,
                        alpha_min=self.stable_alpha_min,
                        alpha_max=self.stable_alpha_max,
                        alpha_scale=float(getattr(decision, "alpha_scale", 1.0)) if decision is not None else 1.0,
                    )

                if proto_events is not None:
                    proto_events.append(
                        make_proto_event(
                            "bg",
                            label,
                            obj_id,
                            ch,
                            act,
                            s_max,
                            n_before,
                            n_before,
                            scope=sc,
                            extra={"promote_hits": int(promote_hits)},
                        )
                    )
                return

        if decision is not None and not getattr(decision, "allow_insert", True):
            return

        stable_bank.add(x, timestamp, weight=float(getattr(work_proto, "weight", 1.0)))
        if proto_events is not None:
            proto_events.append(
                make_proto_event(
                    "bg",
                    label,
                    obj_id,
                    ch,
                    "PROMOTE_INSERT",
                    None,
                    n_before,
                    int(len(stable_protos)),
                    scope=sc,
                    extra={"promote_hits": int(promote_hits)},
                )
            )

        stable_protos = self.ensure_bank_list(stable_bank)
        best_pair_sim, bi, bj = find_best_internal_pair(stable_protos)
        if best_pair_sim >= self.stable_merge_thr and bi >= 0 and bj >= 0:
            if decision is not None and not getattr(decision, "allow_merge", True):
                return
            merge_two_protos_inplace(stable_protos[bi], stable_protos[bj], timestamp, merge_weight=self.merge_weight, count_cap=self.count_cap)
            stable_protos.pop(bj)
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "bg",
                        label,
                        obj_id,
                        ch,
                        "MERGE",
                        None,
                        int(len(stable_protos) + 1),
                        int(len(stable_protos)),
                        scope=sc,
                        merge_pair_sim=float(best_pair_sim),
                    )
                )

    def promote_from_obs_to_stable(
        self,
        work_bank,
        stable_bank,
        x_norm: np.ndarray,
        timestamp: float,
        promote_hits: int,
        label,
        obj_id,
        region,
        ring,
        proto_events,
        decision,
    ):
        work_protos = self.ensure_bank_list(work_bank)
        if not work_protos:
            return
        sims = compute_sims_to_protos(x_norm, work_protos)
        nn_idx = int(np.argmax(sims))
        self.try_promote_to_stable(
            stable_bank=stable_bank,
            work_proto=work_protos[nn_idx],
            timestamp=timestamp,
            promote_hits=promote_hits,
            label=label,
            obj_id=obj_id,
            region=region,
            ring=ring,
            proto_events=proto_events,
            decision=decision,
        )

    def update_global_bank(
        self,
        *,
        work_bank,
        stable_bank,
        x_norm: np.ndarray,
        timestamp: float,
        promote_hits: int,
        label: str | None,
        obj_id: int,
        ring: str,
        det_feats: dict | None,
        proto_events: list | None,
        decision,
    ):
        ch = "global"
        ring_l = str(ring).lower().strip()
        work_scope = self.scope("work", ch, ring_l)
        stable_scope = self.scope("stable", ch, ring_l)
        obs_quality = self.global_observation_quality(det_feats, ring_l)
        work_protos = self.ensure_bank_list(work_bank)
        stable_protos = self.ensure_bank_list(stable_bank)
        x_norm = l2_normalize_vector(np.asarray(x_norm, dtype=np.float32).reshape(-1))

        if decision is not None and str(getattr(decision, "mode", "FULL")) == "OFF":
            return

        if stable_protos:
            sims = compute_sims_to_protos(x_norm, stable_protos)
            nn_idx = int(np.argmax(sims))
            s_max = float(sims[nn_idx])
            if s_max >= self.stable_dup_thr:
                p = stable_protos[nn_idx]
                allow_ema = bool(decision is None or getattr(decision, "allow_ema", True))
                allow_ema = allow_ema and float(obs_quality) >= float(self.global_stable_update_min_quality)
                if self.stable_update in ("none", "off", "skip") or not allow_ema:
                    p.count = int(getattr(p, "count", 1)) + 1
                    p.last_seen = float(timestamp)
                    act = "DUP_COUNT"
                else:
                    act = update_proto_dup_gated_ema(
                        proto=p,
                        x_norm=x_norm,
                        timestamp=timestamp,
                        s_max=s_max,
                        dup_thr=self.stable_dup_thr,
                        margin=self.stable_dup_margin,
                        alpha_min=self.stable_alpha_min,
                        alpha_max=self.stable_alpha_max,
                        alpha_scale=float(getattr(decision, "alpha_scale", 1.0)) * float(obs_quality) if decision is not None else float(obs_quality),
                    )
                self.refresh_proto_quality(p, obs_quality)
                if proto_events is not None:
                    proto_events.append(
                        make_proto_event(
                            "bg",
                            label,
                            obj_id,
                            ch,
                            f"STABLE_{act}",
                            s_max,
                            int(len(stable_protos)),
                            int(len(stable_protos)),
                            scope=stable_scope,
                            extra=self.event_extra(p, obs_quality),
                        )
                    )
                return

        n_before = int(len(work_protos))
        s_max = None
        if work_protos:
            sims = compute_sims_to_protos(x_norm, work_protos)
            nn_idx = int(np.argmax(sims))
            s_max = float(sims[nn_idx])
            if s_max >= self.work_dup_thr:
                p = work_protos[nn_idx]
                allow_ema = bool(decision is None or getattr(decision, "allow_ema", True))
                allow_ema = allow_ema and float(obs_quality) >= float(self.global_work_update_min_quality)
                if allow_ema:
                    act = update_proto_dup_gated_ema(
                        proto=p,
                        x_norm=x_norm,
                        timestamp=timestamp,
                        s_max=s_max,
                        dup_thr=self.work_dup_thr,
                        margin=self.dup_margin,
                        alpha_min=self.alpha_min,
                        alpha_max=self.alpha_max,
                        alpha_scale=float(getattr(decision, "alpha_scale", 1.0)) * float(obs_quality) if decision is not None else float(obs_quality),
                    )
                else:
                    p.count = int(getattr(p, "count", 1)) + 1
                    p.last_seen = float(timestamp)
                    act = "DUP_COUNT"
                self.refresh_proto_quality(p, obs_quality)
                if proto_events is not None:
                    proto_events.append(
                        make_proto_event(
                            "bg",
                            label,
                            obj_id,
                            ch,
                            act,
                            s_max,
                            n_before,
                            n_before,
                            scope=work_scope,
                            extra=self.event_extra(p, obs_quality),
                        )
                    )

                can_promote = bool(decision is None or getattr(decision, "allow_promote", True))
                if can_promote and int(getattr(p, "count", 1)) >= int(promote_hits) and float(self.proto_mean_quality(p)) >= float(self.global_promote_min_quality):
                    promoted = self.clone_proto(p)
                    if stable_protos:
                        s2 = compute_sims_to_protos(promoted.embedding, stable_protos)
                        stable_idx = int(np.argmax(s2))
                        stable_best = float(s2[stable_idx])
                        if stable_best >= self.stable_dup_thr:
                            self.merge_bg_protos_inplace(stable_protos[stable_idx], promoted, timestamp)
                            work_protos.pop(int(nn_idx))
                            if proto_events is not None:
                                proto_events.append(
                                    make_proto_event(
                                        "bg",
                                        label,
                                        obj_id,
                                        ch,
                                        "PROMOTE_MERGE_STABLE",
                                        stable_best,
                                        int(len(stable_protos)),
                                        int(len(stable_protos)),
                                        scope=stable_scope,
                                        extra=self.event_extra(stable_protos[stable_idx], obs_quality),
                                    )
                                )
                            return

                    if decision is not None and not getattr(decision, "allow_insert", True):
                        return
                    K = self.bank_capacity(stable_bank)
                    if K <= 0 or len(stable_protos) < K:
                        stable_bank.add(
                            promoted.embedding,
                            timestamp,
                            weight=float(getattr(promoted, "weight", 1.0)),
                            count=int(getattr(promoted, "count", 1)),
                            quality_ema=float(getattr(promoted, "quality_ema", 1.0)),
                            quality_sum=float(getattr(promoted, "quality_sum", 1.0)),
                            n_obs=int(getattr(promoted, "n_obs", 1)),
                        )
                        work_protos.pop(int(nn_idx))
                        if proto_events is not None:
                            proto_events.append(
                                make_proto_event(
                                    "bg",
                                    label,
                                    obj_id,
                                    ch,
                                    "PROMOTE_INSERT",
                                    s_max,
                                    int(len(stable_protos) - 1),
                                    int(len(stable_protos)),
                                    scope=stable_scope,
                                    extra=self.event_extra(stable_protos[-1], obs_quality),
                                )
                            )
                        return
                return

        if decision is not None and str(getattr(decision, "mode", "FULL")) == "SAFE":
            return
        if float(obs_quality) < float(self.global_create_min_quality):
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "bg",
                        label,
                        obj_id,
                        ch,
                        "SKIP_LOW_QUALITY",
                        s_max,
                        n_before,
                        n_before,
                        scope=work_scope,
                        extra={"obs_q": round(float(obs_quality), 3)},
                    )
                )
            return
        if decision is not None and not getattr(decision, "allow_insert", True):
            return

        K = self.bank_capacity(work_bank)
        if K <= 0 or n_before < K:
            p = self.insert_global_work_proto(work_bank, x_norm, timestamp, obs_quality)
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "bg",
                        label,
                        obj_id,
                        ch,
                        "INSERT",
                        s_max,
                        n_before,
                        int(len(work_protos)),
                        scope=work_scope,
                        extra=self.event_extra(p, obs_quality),
                    )
                )
            return

        if decision is not None and not getattr(decision, "allow_merge", True):
            return

        best_pair_sim, bi, bj = find_best_internal_pair(work_protos)
        if best_pair_sim >= self.work_merge_thr and bi >= 0 and bj >= 0:
            self.merge_bg_protos_inplace(work_protos[bi], work_protos[bj], timestamp)
            work_protos.pop(bj)
            p = self.insert_global_work_proto(work_bank, x_norm, timestamp, obs_quality)
            if proto_events is not None:
                proto_events.append(
                    make_proto_event(
                        "bg",
                        label,
                        obj_id,
                        ch,
                        "MERGE_INSERT",
                        s_max,
                        n_before,
                        int(len(work_protos)),
                        scope=work_scope,
                        merge_pair_sim=float(best_pair_sim),
                        extra=self.event_extra(p, obs_quality),
                    )
                )
            return

        ev = choose_evict_index(work_protos, strategy=self.work_evict_strategy)
        if ev >= 0:
            work_protos.pop(int(ev))
        p = self.insert_global_work_proto(work_bank, x_norm, timestamp, obs_quality)
        if proto_events is not None:
            proto_events.append(
                make_proto_event(
                    "bg",
                    label,
                    obj_id,
                    ch,
                    "EVICT_INSERT",
                    s_max,
                    n_before,
                    int(len(work_protos)),
                    scope=work_scope,
                    merge_pair_sim=float(best_pair_sim) if best_pair_sim > -1.0 else None,
                    evict_strategy=self.work_evict_strategy,
                    evicted_index=ev,
                    extra=self.event_extra(p, obs_quality),
                )
            )

    def update_from_det_feats(self, obj, det_feats: dict, timestamp: float, proto_events: list | None = None, decision=None):
        if not self.enabled:
            return

        if decision is not None and str(getattr(decision, "mode", "FULL")) == "OFF":
            return

        bg_model = getattr(obj, "background", None)
        if bg_model is None or not getattr(bg_model, "enabled", False):
            return

        obs_bg = det_feats.get("bg", None)
        if not isinstance(obs_bg, dict):
            return

        t = float(timestamp)
        label = getattr(obj, "instance_label", None)
        obj_id = int(getattr(obj, "object_id", -1))

        inner = obs_bg.get("inner", None)
        if inner is not None:
            x = l2_normalize_vector(np.asarray(inner, dtype=np.float32).reshape(-1))
            self.update_global_bank(
                work_bank=bg_model.inner_global_work,
                stable_bank=bg_model.inner_global_stable,
                x_norm=x,
                timestamp=t,
                promote_hits=self.promote_hits_global,
                label=label,
                obj_id=obj_id,
                ring="inner",
                det_feats=det_feats,
                proto_events=proto_events,
                decision=decision,
            )

        outer = obs_bg.get("outer", None)
        if outer is not None:
            x = l2_normalize_vector(np.asarray(outer, dtype=np.float32).reshape(-1))
            self.update_global_bank(
                work_bank=bg_model.outer_global_work,
                stable_bank=bg_model.outer_global_stable,
                x_norm=x,
                timestamp=t,
                promote_hits=self.promote_hits_global,
                label=label,
                obj_id=obj_id,
                ring="outer",
                det_feats=det_feats,
                proto_events=proto_events,
                decision=decision,
            )

        if not self.partials_enabled:
            return

        inner_protos = obs_bg.get("inner_protos", []) or []
        inner_w = obs_bg.get("inner_proto_weights", None)
        if not isinstance(inner_w, list) or len(inner_w) != len(inner_protos):
            inner_w = None

        for i, p in enumerate(inner_protos):
            if p is None:
                continue
            x = l2_normalize_vector(np.asarray(p, dtype=np.float32).reshape(-1))
            self.update_bank_work(bg_model.inner_partials_work, x, t, label, obj_id, region="partials", ring="inner", proto_events=proto_events, decision=decision)

            if inner_w is not None:
                work_protos = self.ensure_bank_list(bg_model.inner_partials_work)
                if work_protos:
                    sims = compute_sims_to_protos(x, work_protos)
                    nn_idx = int(np.argmax(sims))
                    work_protos[nn_idx].weight = float(inner_w[i])

            self.promote_from_obs_to_stable(
                bg_model.inner_partials_work,
                bg_model.inner_partials_stable,
                x,
                t,
                self.promote_hits_partials,
                label,
                obj_id,
                region="partials",
                ring="inner",
                proto_events=proto_events,
                decision=decision,
            )

        outer_protos = obs_bg.get("outer_protos", []) or []
        outer_w = obs_bg.get("outer_proto_weights", None)
        if not isinstance(outer_w, list) or len(outer_w) != len(outer_protos):
            outer_w = None

        for i, p in enumerate(outer_protos):
            if p is None:
                continue
            x = l2_normalize_vector(np.asarray(p, dtype=np.float32).reshape(-1))
            self.update_bank_work(bg_model.outer_partials_work, x, t, label, obj_id, region="partials", ring="outer", proto_events=proto_events, decision=decision)

            if outer_w is not None:
                work_protos = self.ensure_bank_list(bg_model.outer_partials_work)
                if work_protos:
                    sims = compute_sims_to_protos(x, work_protos)
                    nn_idx = int(np.argmax(sims))
                    work_protos[nn_idx].weight = float(outer_w[i])

            self.promote_from_obs_to_stable(
                bg_model.outer_partials_work,
                bg_model.outer_partials_stable,
                x,
                t,
                self.promote_hits_partials,
                label,
                obj_id,
                region="partials",
                ring="outer",
                proto_events=proto_events,
                decision=decision,
            )
