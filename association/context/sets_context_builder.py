from __future__ import annotations

import math


class SetsContextBuilder:
    def __init__(self, *, memory_store, influence) -> None:
        self.memory_store = memory_store
        self.influence = influence

    def build_context(self, neighbor_sets_out) -> dict:
        if not bool(getattr(self.influence, "enabled", False)) or not isinstance(neighbor_sets_out, dict):
            return {"enabled": False}

        payload = self.extract_context_payload(neighbor_sets_out)
        if payload is None:
            return {"enabled": False}

        class_ctx = self.build_class_context(
            shortlist=payload["shortlist"],
            prior_by_oid=payload["prior_by_oid"],
            support_sum_by_oid=payload["support_sum_by_oid"],
            anchors=payload["anchors"],
            hypotheses=payload["hypotheses"],
            vocab_size=payload["vocab_size"],
        )

        quality_terms, quality, quality_ok, reason = self.build_global_quality(
            best_score=float(payload["best_score"]),
            coverage_eff=float(payload["coverage_eff"]),
            maturity=float(payload["maturity"]),
            density=float(payload["density"]),
            k_best=int(payload["k_best"]),
            n_hypotheses=int(payload["n_hypotheses"]),
            class_ctx=class_ctx,
        )

        return {
            "enabled": True,
            "global_ok": bool(quality_ok),
            "reason": str(reason),
            "quality": float(quality),
            "best": float(payload["best_score"]),
            "coverage_eff": float(payload["coverage_eff"]),
            "maturity": float(payload["maturity"]),
            "density": float(payload["density"]),
            "k_best": int(payload["k_best"]),
            "n_hypotheses": int(payload["n_hypotheses"]),
            "shortlist": set(int(x) for x in payload["shortlist"]),
            "anchors": list(payload["anchors"]),
            "prior_by_oid": dict(payload["prior_by_oid"] or {}),
            "support_sum_by_oid": dict(payload["support_sum_by_oid"] or {}),
            "class_ctx": class_ctx,
            "quality_terms": dict(quality_terms),
        }

    def extract_context_payload(self, neighbor_sets_out) -> dict | None:
        core = neighbor_sets_out.get("core", None)
        debug = neighbor_sets_out.get("debug", None)
        if not isinstance(core, dict) or not isinstance(debug, dict):
            return None

        meta = debug.get("meta", None)
        if not isinstance(meta, dict):
            meta = {}

        hyps = debug.get("set_hypotheses", None)
        if not isinstance(hyps, list):
            hyps = []

        return {
            "shortlist": set(int(x) for x in (core.get("shortlist", []) or [])),
            "prior_by_oid": self.as_float_map(core.get("prior_by_oid", None)),
            "support_sum_by_oid": self.as_float_map(debug.get("object_support_sum", None)),
            "anchors": [int(x) for x in (core.get("anchors", []) or meta.get("anchors", []) or []) if x is not None],
            "hypotheses": [h for h in hyps if isinstance(h, dict)],
            "vocab_size": int(len(self.memory_store.all_objects())) if self.memory_store is not None else None,
            "best_score": float(core.get("best_score", 0.0)),
            "coverage_eff": float(meta.get("coverage_eff_best", 0.0)),
            "maturity": float(core.get("mean_maturity_best", meta.get("mean_maturity_best", 0.0))),
            "density": float(meta.get("density_best", 0.0)),
            "k_best": int(meta.get("k_best", 0)),
            "n_hypotheses": int(core.get("n_hypotheses", 0)),
        }

    def build_global_quality(
        self,
        *,
        best_score: float,
        coverage_eff: float,
        maturity: float,
        density: float,
        k_best: int,
        n_hypotheses: int,
        class_ctx: dict[int, dict],
    ) -> tuple[dict, float, bool, str]:
        quality_terms = self.build_quality_terms(
            best_score=float(best_score),
            coverage_eff=float(coverage_eff),
            maturity=float(maturity),
            density=float(density),
            k_best=int(k_best),
            class_ctx=class_ctx,
        )
        quality = self.weighted_mean(
            [
                (self.influence.qw_best, quality_terms["best"]),
                (self.influence.qw_cov, quality_terms["coverage_eff"]),
                (self.influence.qw_maturity, quality_terms["maturity"]),
                (self.influence.qw_density, quality_terms["density"]),
                (self.influence.qw_size, quality_terms["size"]),
                (self.influence.qw_pruning, quality_terms["pruning"]),
            ]
        )

        quality_ok = bool(
            int(n_hypotheses) > 0
            and int(k_best) >= int(self.influence.min_size)
            and float(quality_terms["best"]) >= float(self.influence.min_best_score)
            and float(quality_terms["coverage_eff"]) >= float(self.influence.min_coverage_eff)
            and float(quality) >= float(self.influence.min_quality)
        )
        reason = self.quality_reason(
            n_hypotheses=int(n_hypotheses),
            k_best=int(k_best),
            best_term=float(quality_terms["best"]),
            coverage_term=float(quality_terms["coverage_eff"]),
            quality_ok=bool(quality_ok),
        )
        return quality_terms, float(quality), bool(quality_ok), str(reason)

    def build_quality_terms(
        self,
        *,
        best_score: float,
        coverage_eff: float,
        maturity: float,
        density: float,
        k_best: int,
        class_ctx: dict[int, dict],
    ) -> dict:
        active_pruning = [
            float(pack.get("pruning_power", 0.0))
            for pack in class_ctx.values()
            if int(pack.get("kept_count", 0)) > 0
        ]
        q_pruning = float(sum(active_pruning) / float(len(active_pruning))) if active_pruning else 0.0
        return {
            "best": self.clamp01(best_score),
            "coverage_eff": self.clamp01(coverage_eff),
            "maturity": self.clamp01(maturity),
            "density": self.clamp01(density),
            "size": self.clamp01(1.0 - math.exp(-max(0.0, float(k_best) - 1.0) / float(self.influence.size_tau))),
            "pruning": self.clamp01(q_pruning),
        }

    def quality_reason(
        self,
        *,
        n_hypotheses: int,
        k_best: int,
        best_term: float,
        coverage_term: float,
        quality_ok: bool,
    ) -> str:
        if quality_ok:
            return "OK"
        if int(n_hypotheses) <= 0:
            return "NO_HYPOTHESES"
        if int(k_best) < int(self.influence.min_size):
            return "SMALL_SET"
        if float(best_term) < float(self.influence.min_best_score):
            return "LOW_SCORE"
        if float(coverage_term) < float(self.influence.min_coverage_eff):
            return "LOW_COVERAGE"
        return "LOW_CONF"

    def build_class_context(
        self,
        *,
        shortlist: set[int],
        prior_by_oid: dict[int, float],
        support_sum_by_oid: dict[int, float],
        anchors: list[int],
        hypotheses: list[dict],
        vocab_size: int | None,
    ) -> dict[int, dict]:
        out: dict[int, dict] = {}
        for cid, all_oids in self.objects_by_class().items():
            pack = self.build_class_pack(
                class_id=int(cid),
                all_oids=all_oids,
                shortlist=shortlist,
                prior_by_oid=prior_by_oid,
                support_sum_by_oid=support_sum_by_oid,
                anchors=anchors,
                hypotheses=hypotheses,
                vocab_size=vocab_size,
            )
            if pack is not None:
                out[int(cid)] = pack
        return out

    def objects_by_class(self) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for obj in self.memory_store.all_objects() if self.memory_store is not None else []:
            oid = int(getattr(obj, "object_id", -1))
            cid = int(getattr(obj, "class_id", -1))
            if oid < 0 or cid < 0:
                continue
            out.setdefault(int(cid), []).append(int(oid))
        return out

    def build_class_pack(
        self,
        *,
        class_id: int,
        all_oids: list[int],
        shortlist: set[int],
        prior_by_oid: dict[int, float],
        support_sum_by_oid: dict[int, float],
        anchors: list[int],
        hypotheses: list[dict],
        vocab_size: int | None,
    ) -> dict | None:
        all_ids = sorted(int(x) for x in (all_oids or []))
        total = int(len(all_ids))
        if total <= 0:
            return None

        kernel_ids = self.class_kernel_ids(
            class_id=int(class_id),
            anchors=anchors,
            hypotheses=hypotheses,
        )
        kernel_stats = self.kernel_support_stats_by_oid(all_ids=all_ids, kernel_ids=kernel_ids, vocab_size=vocab_size)
        kernel_raw_by_oid = dict(kernel_stats["kernel_raw_by_oid"])
        rel_pack = self.class_relative_support_pack(
            all_ids=all_ids,
            prior_by_oid=prior_by_oid,
            support_sum_by_oid=support_sum_by_oid,
            kernel_raw_by_oid=kernel_raw_by_oid,
        )
        support_pack = self.class_support_pack(
            all_ids=all_ids,
            shortlist=shortlist,
            kernel_raw_by_oid=kernel_raw_by_oid,
            kernel_hit_count_by_oid=kernel_stats["kernel_hit_count_by_oid"],
            kernel_hit_ratio_by_oid=kernel_stats["kernel_hit_ratio_by_oid"],
            kernel_rel_by_oid=rel_pack["kernel_rel_by_oid"],
            hyp_rel_by_oid=rel_pack["hyp_rel_by_oid"],
        )

        pruning = 1.0 - (float(len(support_pack["supported"])) / float(total))
        return {
            "all_oids": all_ids,
            "shortlist": set(int(x) for x in support_pack["shortlist"]),
            "supported": set(int(x) for x in support_pack["supported"]),
            "soft_supported": set(int(x) for x in support_pack["soft_supported"]),
            "coverage_ok_by_oid": {int(k): bool(v) for k, v in support_pack["coverage_ok_by_oid"].items()},
            "total_count": int(total),
            "kept_count": int(len(support_pack["supported"])),
            "pruning_power": float(self.clamp01(pruning)),
            "compat_rel_by_oid": {int(k): float(v) for k, v in rel_pack["compat_rel_by_oid"].items()},
            "kernel_raw_by_oid": {int(k): float(v) for k, v in kernel_raw_by_oid.items()},
            "kernel_hit_count_by_oid": {int(k): int(v) for k, v in kernel_stats["kernel_hit_count_by_oid"].items()},
            "kernel_hit_ratio_by_oid": {int(k): float(v) for k, v in kernel_stats["kernel_hit_ratio_by_oid"].items()},
            "kernel_rel_by_oid": {int(k): float(v) for k, v in rel_pack["kernel_rel_by_oid"].items()},
            "hyp_rel_by_oid": {int(k): float(v) for k, v in rel_pack["hyp_rel_by_oid"].items()},
            "selectivity": float(support_pack["selectivity"]),
            "class_strength": float(rel_pack["class_strength"]),
            "top_support_abs": float(support_pack["top_support_abs"]),
            "top_support_rel": float(support_pack["top_support_rel"]),
            "band_cutoff_rel": float(support_pack["band_cutoff_rel"]),
            "soft_band_cutoff_rel": float(support_pack["soft_band_cutoff_rel"]),
            "kernel_ids": list(int(x) for x in kernel_ids),
        }

    def kernel_support_stats_by_oid(
        self,
        *,
        all_ids: list[int],
        kernel_ids: list[int],
        vocab_size: int | None,
    ) -> dict:
        raw_out: dict[int, float] = {}
        hit_count_out: dict[int, int] = {}
        hit_ratio_out: dict[int, float] = {}
        for oid in all_ids:
            stats = self.object_support_stats_to_kernel(int(oid), kernel_ids, vocab_size=vocab_size)
            raw_out[int(oid)] = float(stats["mean_support"])
            hit_count_out[int(oid)] = int(stats["hit_count"])
            hit_ratio_out[int(oid)] = float(stats["hit_ratio"])
        return {
            "kernel_raw_by_oid": raw_out,
            "kernel_hit_count_by_oid": hit_count_out,
            "kernel_hit_ratio_by_oid": hit_ratio_out,
        }

    def class_relative_support_pack(
        self,
        *,
        all_ids: list[int],
        prior_by_oid: dict[int, float],
        support_sum_by_oid: dict[int, float],
        kernel_raw_by_oid: dict[int, float],
    ) -> dict:
        max_top = max((float(prior_by_oid.get(int(oid), 0.0)) for oid in all_ids), default=0.0)
        max_sum = max((float(support_sum_by_oid.get(int(oid), 0.0)) for oid in all_ids), default=0.0)
        max_kernel = max((float(kernel_raw_by_oid.get(int(oid), 0.0)) for oid in all_ids), default=0.0)

        den = float(self.influence.support_top_weight + self.influence.support_sum_weight)
        if den <= 1e-12:
            den = 1.0
        mix_den = float(self.influence.support_kernel_weight + self.influence.support_hyp_weight)
        if mix_den <= 1e-12:
            mix_den = 1.0

        compat_rel_by_oid: dict[int, float] = {}
        kernel_rel_by_oid: dict[int, float] = {}
        hyp_rel_by_oid: dict[int, float] = {}
        class_strength = 0.0

        for oid in all_ids:
            t = float(prior_by_oid.get(int(oid), 0.0))
            s = float(support_sum_by_oid.get(int(oid), 0.0))
            t_rel = float(t / max_top) if max_top > 1e-12 else 0.0
            s_rel = float(s / max_sum) if max_sum > 1e-12 else 0.0
            t_rel = float(self.influence.compress_rel(t_rel, gamma=self.influence.support_hyp_rel_gamma))
            s_rel = float(self.influence.compress_rel(s_rel, gamma=self.influence.support_hyp_rel_gamma))
            hyp_rel = (float(self.influence.support_top_weight) * float(t_rel) + float(self.influence.support_sum_weight) * float(s_rel)) / den
            k_rel = float(kernel_raw_by_oid.get(int(oid), 0.0) / max_kernel) if max_kernel > 1e-12 else 0.0
            k_rel = float(self.influence.compress_rel(k_rel, gamma=self.influence.support_kernel_rel_gamma))
            compat_rel_by_oid[int(oid)] = float(self.clamp01(k_rel))
            kernel_rel_by_oid[int(oid)] = float(self.clamp01(k_rel))
            hyp_rel_by_oid[int(oid)] = float(self.clamp01(hyp_rel))
            abs_strength = (
                float(self.influence.support_kernel_weight) * float(kernel_raw_by_oid.get(int(oid), 0.0))
                + float(self.influence.support_hyp_weight) * float(t)
            ) / mix_den
            class_strength = max(float(class_strength), float(abs_strength))

        return {
            "compat_rel_by_oid": compat_rel_by_oid,
            "kernel_rel_by_oid": kernel_rel_by_oid,
            "hyp_rel_by_oid": hyp_rel_by_oid,
            "class_strength": self.clamp01(class_strength),
        }

    def class_support_pack(
        self,
        *,
        all_ids: list[int],
        shortlist: set[int],
        kernel_raw_by_oid: dict[int, float],
        kernel_hit_count_by_oid: dict[int, int],
        kernel_hit_ratio_by_oid: dict[int, float],
        kernel_rel_by_oid: dict[int, float],
        hyp_rel_by_oid: dict[int, float],
    ) -> dict:
        coverage_ok_by_oid = {
            int(oid): bool(
                int(kernel_hit_count_by_oid.get(int(oid), 0) or 0) >= int(self.influence.support_min_kernel_hits)
                and float(kernel_hit_ratio_by_oid.get(int(oid), 0.0) or 0.0) >= float(self.influence.support_min_kernel_hit_ratio)
            )
            for oid in all_ids
        }
        ranked_kernel = [
            (float(kernel_rel_by_oid.get(int(oid), 0.0)), int(oid))
            for oid in all_ids
            if float(kernel_rel_by_oid.get(int(oid), 0.0)) > 0.0 and bool(coverage_ok_by_oid.get(int(oid), False))
        ]
        ranked_kernel.sort(key=lambda x: float(x[0]), reverse=True)

        soft_mix_den = float(self.influence.support_soft_mix_kernel_weight + self.influence.support_soft_mix_hyp_weight)
        if soft_mix_den <= 1e-12:
            soft_mix_den = 1.0
        ranked_soft = []
        for oid in all_ids:
            k_rel = float(kernel_rel_by_oid.get(int(oid), 0.0))
            h_rel = float(hyp_rel_by_oid.get(int(oid), 0.0))
            soft_mix = (
                float(self.influence.support_soft_mix_kernel_weight) * float(k_rel)
                + float(self.influence.support_soft_mix_hyp_weight) * float(h_rel)
            ) / float(soft_mix_den)
            if float(soft_mix) > 0.0 and bool(coverage_ok_by_oid.get(int(oid), False)):
                ranked_soft.append((float(self.clamp01(soft_mix)), int(oid)))
        ranked_soft.sort(key=lambda x: float(x[0]), reverse=True)

        top1 = float(ranked_kernel[0][0]) if ranked_kernel else 0.0
        top2 = float(ranked_kernel[1][0]) if len(ranked_kernel) > 1 else 0.0
        selectivity = float((top1 - top2) / max(1e-12, top1)) if top1 > 0.0 else 0.0
        selectivity = self.clamp01(selectivity)

        band_cutoff = float(self.influence.support_band_rel) * float(top1) if top1 > 0.0 else 0.0
        soft_top1 = float(ranked_soft[0][0]) if ranked_soft else 0.0
        soft_band_cutoff = float(self.influence.support_soft_band_rel) * float(soft_top1) if soft_top1 > 0.0 else 0.0
        supported = set(int(oid) for raw, oid in ranked_kernel if float(raw) >= float(band_cutoff))
        soft_supported = set(int(oid) for raw, oid in ranked_soft if float(raw) >= float(soft_band_cutoff))
        if not supported and ranked_kernel and float(ranked_kernel[0][0]) > 0.0:
            supported.add(int(ranked_kernel[0][1]))
        if not soft_supported and ranked_soft and float(ranked_soft[0][0]) > 0.0:
            soft_supported.add(int(ranked_soft[0][1]))

        return {
            "shortlist": set(int(oid) for oid in all_ids if int(oid) in shortlist),
            "supported": supported,
            "soft_supported": soft_supported,
            "coverage_ok_by_oid": {int(k): bool(v) for k, v in coverage_ok_by_oid.items()},
            "selectivity": float(selectivity),
            "top_support_abs": float(max((float(kernel_raw_by_oid.get(int(oid), 0.0)) for oid in all_ids), default=0.0)),
            "top_support_rel": float(top1),
            "band_cutoff_rel": float(band_cutoff),
            "soft_band_cutoff_rel": float(soft_band_cutoff),
        }

    def class_pack(self, det_class_id: int, ctx: dict) -> dict | None:
        packs = ctx.get("class_ctx", None)
        if not isinstance(packs, dict):
            return None
        pack = packs.get(int(det_class_id), None)
        return pack if isinstance(pack, dict) else None

    def as_float_map(self, raw) -> dict[int, float]:
        if not isinstance(raw, dict):
            return {}
        out = {}
        for k, v in raw.items():
            try:
                out[int(k)] = float(v)
            except Exception:
                continue
        return out

    def clamp01(self, x: float) -> float:
        return float(max(0.0, min(1.0, float(x))))

    def weighted_mean(self, items: list[tuple[float, float]]) -> float:
        used = [(max(0.0, float(w)), self.clamp01(v)) for w, v in (items or []) if float(w) > 0.0]
        if not used:
            return 0.0
        den = float(sum(w for w, _ in used))
        if den <= 1e-12:
            return 0.0
        num = float(sum(w * v for w, v in used))
        return self.clamp01(num / den)

    def class_kernel_ids(self, class_id: int, anchors: list[int], hypotheses: list[dict]) -> list[int]:
        kernel = []
        seen = set()

        for aid in anchors or []:
            try:
                aid_i = int(aid)
            except Exception:
                continue
            if aid_i in seen:
                continue
            seen.add(aid_i)
            kernel.append(aid_i)

        for hyp in (hypotheses or [])[: max(1, self.influence.context_k)]:
            if not isinstance(hyp, dict):
                continue
            for oid in (hyp.get("object_ids", []) or []):
                try:
                    oid_i = int(oid)
                except Exception:
                    continue
                obj = self.memory_store.get(int(oid_i)) if self.memory_store is not None else None
                if obj is None:
                    continue
                if int(getattr(obj, "class_id", -1)) == int(class_id):
                    continue
                if oid_i in seen:
                    continue
                seen.add(oid_i)
                kernel.append(oid_i)
                if self.influence.kernel_max > 0 and len(kernel) >= int(self.influence.kernel_max):
                    return [int(x) for x in kernel]

        return [int(x) for x in kernel]

    def object_support_to_kernel(self, object_id: int, kernel_obj_ids: list[int], vocab_size: int | None = None) -> float:
        stats = self.object_support_stats_to_kernel(object_id, kernel_obj_ids, vocab_size=vocab_size)
        return float(stats["mean_support"])

    def object_support_stats_to_kernel(self, object_id: int, kernel_obj_ids: list[int], vocab_size: int | None = None) -> dict:
        if not kernel_obj_ids or self.memory_store is None:
            return {"mean_support": 0.0, "hit_count": 0, "kernel_count": 0, "hit_ratio": 0.0}

        vals = []
        kernel_count = 0
        for kid in kernel_obj_ids:
            kobj = self.memory_store.get(int(kid))
            g = getattr(kobj, "neighbors", None) if kobj is not None else None
            if g is None or not getattr(g, "enabled", False):
                continue
            kernel_count += 1
            try:
                p = float(g.p_conditional(int(object_id), vocab_size=vocab_size))
            except Exception:
                p = 0.0
            p = self.clamp01(p)
            if p >= float(self.influence.min_edge_p):
                vals.append(float(p))

        hit_count = int(len(vals))
        mean_support = float(sum(vals) / float(len(vals))) if vals else 0.0
        hit_ratio = float(hit_count / float(max(1, kernel_count))) if kernel_count > 0 else 0.0
        return {
            "mean_support": float(mean_support),
            "hit_count": int(hit_count),
            "kernel_count": int(kernel_count),
            "hit_ratio": float(self.clamp01(hit_ratio)),
        }
