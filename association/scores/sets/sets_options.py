from __future__ import annotations

from contextlib import nullcontext
from itertools import combinations


class SetsOptionsBuilder:
    def __init__(self, *, score) -> None:
        self.score = score

    def frame_timer(self):
        timer = getattr(self.score, "_frame_timer", None)
        return timer

    def _filter_class_options_by_used_obj_ids(
        self,
        opts: list[dict],
        used_obj_ids: set[int],
        used_obj_mask: int = 0,
    ) -> list[dict]:
        if not opts:
            return []
        used_mask_int = int(used_obj_mask or 0) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0
        if used_mask_int:
            used = set(int(x) for x in (used_obj_ids or set()))
            if not used:
                return [
                    opt
                    for opt in opts
                    if (int(opt.get("object_mask", 0) or 0) & used_mask_int) == 0
                ]
            return [
                opt
                for opt in opts
                if (int(opt.get("object_mask", 0) or 0) & used_mask_int) == 0
                and not any(int(oid) in used for oid in (opt.get("object_ids", ()) or ()))
            ]

        used = set(int(x) for x in (used_obj_ids or set()))
        if not used:
            return list(opts)
        return [
            opt
            for opt in opts
            if not any(int(oid) in used for oid in (opt.get("object_ids", ()) or ()))
        ]

    def select_top_dets_per_class(self, dets_by_class: dict[int, list]) -> dict[int, list]:
        class_items: dict[int, list] = {}
        for cid, class_dets in dets_by_class.items():
            scored = []
            for d in class_dets:
                conf = float(getattr(d, "confidence", 0.0))
                did = int(getattr(d, "detection_id", -1))
                scored.append((conf, did, d))
            scored.sort(key=lambda x: (float(x[0]), int(x[1])), reverse=True)
            class_items[int(cid)] = [x[2] for x in scored[: self.score.per_class_det_k]]
        return class_items

    def build_pools_for_classes(self, class_items: dict[int, list], anchors: list[int], vocab_size: int | None = None) -> tuple[dict[int, list[int]], dict[int, dict]]:
        pools: dict[int, list[int]] = {}
        meta: dict[int, dict] = {}
        timer = self.frame_timer()
        for cid in sorted(int(x) for x in class_items.keys()):
            if timer is None:
                pool = self.build_pool_for_class(int(cid), anchors=anchors, vocab_size=vocab_size)
            else:
                pool = timer.run(
                    "build_pools_for_classes/build_pool_for_class",
                    self.build_pool_for_class,
                    int(cid),
                    anchors=anchors,
                    vocab_size=vocab_size,
                )
            pool = [int(x) for x in (pool or [])]
            if not pool:
                continue
            pools[int(cid)] = pool[: self.score.per_class_pool_k]
            meta[int(cid)] = {"pool_n": int(len(pool))}
        return pools, meta

    def build_pool_for_class(self, class_id: int, anchors: list[int], vocab_size: int | None = None) -> list[int]:
        objs = self.score.memory_store.get_by_class(int(class_id)) or []
        if not objs:
            return []

        base_pool = [int(o.object_id) for o in objs]
        anchor_in_class = []
        for aid in anchors or []:
            obj = self.score.memory_store.get(int(aid))
            if obj is None:
                continue
            if int(getattr(obj, "class_id", -1)) == int(class_id):
                anchor_in_class.append(int(aid))

        if anchors:
            cand = {}
            for aid in anchors:
                a = self.score.memory_store.get(int(aid))
                if a is None:
                    continue
                g = getattr(a, "neighbors", None)
                if g is None or not getattr(g, "enabled", False):
                    continue

                for oid in base_pool:
                    if oid == int(aid):
                        continue
                    p = float(self.score.p_conditional_cached(g, int(aid), int(oid), vocab_size=vocab_size))
                    if p < self.score.min_edge_p:
                        continue
                    prev = float(cand.get(int(oid), 0.0))
                    if p > prev:
                        cand[int(oid)] = float(p)

            ranked = sorted(cand.items(), key=lambda kv: float(kv[1]), reverse=True)
            out = [int(kv[0]) for kv in ranked]

            if len(out) < self.score.per_class_pool_k:
                seen = set(out)
                rest = []
                for oid in base_pool:
                    if int(oid) in seen:
                        continue
                    rest.append((self.score.object_maturity_score(int(oid)), int(oid)))
                rest.sort(key=lambda x: (float(x[0]), int(x[1])), reverse=True)
                for _, oid in rest:
                    out.append(int(oid))
                    if len(out) >= self.score.per_class_pool_k:
                        break

            final = []
            seen = set()
            for oid in (anchor_in_class + out):
                if int(oid) in seen:
                    continue
                seen.add(int(oid))
                final.append(int(oid))
                if len(final) >= self.score.per_class_pool_k:
                    break

            return final

        ranked = [(self.score.object_maturity_score(int(oid)), int(oid)) for oid in base_pool]
        ranked.sort(key=lambda x: (float(x[0]), int(x[1])), reverse=True)

        final = []
        seen = set()
        for oid in (anchor_in_class + [int(x[1]) for x in ranked]):
            if int(oid) in seen:
                continue
            seen.add(int(oid))
            final.append(int(oid))
            if len(final) >= self.score.per_class_pool_k:
                break

        return final

    def build_kernel(self, anchors: list[int], selected_obj_ids: list[int]) -> list[int]:
        kernel = [int(x) for x in (anchors or [])]
        for x in selected_obj_ids or []:
            if int(x) not in kernel:
                kernel.append(int(x))
        if self.score.kernel_max > 0 and len(kernel) > self.score.kernel_max:
            kernel = kernel[: self.score.kernel_max]
        return [int(x) for x in kernel]

    def build_class_options(
        self,
        class_id: int,
        det_ids: list[int],
        pool_obj_ids: list[int],
        used_obj_ids: set[int],
        kernel_obj_ids: list[int],
        vocab_size: int | None = None,
    ) -> list[dict]:
        timer = self.frame_timer()
        ctx = timer.measure("build_class_options") if timer is not None else nullcontext()
        with ctx:
            det_ids = [int(x) for x in (det_ids or [])]
            pool = [int(x) for x in (pool_obj_ids or []) if int(x) not in used_obj_ids]

            if not det_ids:
                return []
            if not pool:
                return (
                    [{
                        "class_id": int(class_id),
                        "det_ids": (),
                        "object_ids": (),
                        "det_mask": 0,
                        "object_mask": 0,
                        "opt_rank": 0.0,
                        "class_info": 0.0,
                        "class_logC": 0.0,
                        "class_stability": 0.0,
                        "class_exclusivity": 0.0,
                        "class_exclusivity_valid": False,
                        "k": 0,
                    }]
                    if self.score.allow_partial_coverage
                    else []
                )

            max_k = min(len(det_ids), len(pool))
            ks = list(range(1, max_k + 1))
            if self.score.allow_partial_coverage:
                ks = [0] + ks

            obj_scored = []
            obj_support_debug = []
            kernel_set = set(int(x) for x in (kernel_obj_ids or []))
            supp_cache: dict[int, float] = {}

            if timer is None:
                for oid in pool:
                    support_detail = None
                    if kernel_set:
                        support_detail = self.object_support_detail_to_kernel(int(oid), kernel_obj_ids, vocab_size=vocab_size)
                        s = float(support_detail.get("mean_support", 0.0))
                    else:
                        s = float(self.score.object_maturity_score(int(oid)))
                    supp_cache[int(oid)] = float(s)
                    obj_scored.append((float(s), int(oid)))
                    obj_support_debug.append(
                        {
                            "object_id": int(oid),
                            "support": float(s),
                            "detail": dict(support_detail or {}),
                        }
                    )
            else:
                with timer.measure("build_class_options/support_rank"):
                    for oid in pool:
                        support_detail = None
                        if kernel_set:
                            support_detail = self.object_support_detail_to_kernel(int(oid), kernel_obj_ids, vocab_size=vocab_size)
                            s = float(support_detail.get("mean_support", 0.0))
                        else:
                            s = float(self.score.object_maturity_score(int(oid)))
                        supp_cache[int(oid)] = float(s)
                        obj_scored.append((float(s), int(oid)))
                        obj_support_debug.append(
                            {
                                "object_id": int(oid),
                                "support": float(s),
                                "detail": dict(support_detail or {}),
                            }
                        )
            obj_scored.sort(key=lambda x: (float(x[0]), int(x[1])), reverse=True)

            obj_ranked = [int(x[1]) for x in obj_scored[: self.score.per_class_pool_k]]
            obj_ranked_set = set(obj_ranked)
            top_support = float(obj_scored[0][0]) if obj_scored else 0.0
            plausible_thr = float(top_support) * float(self.score.class_plausible_rel)
            plausible_n = 0
            for s, _ in obj_scored:
                if float(s) + 1e-12 < plausible_thr:
                    break
                plausible_n += 1
            target_k = int(min(len(det_ids), max(1, plausible_n))) if det_ids else 0

            out = []
            pool_n = int(len(pool_obj_ids))
            stability = float(self.score.class_stability_from_pool(obj_ranked_set, kernel_obj_ids, vocab_size=vocab_size))

            if bool(getattr(self.score, "collect_class_options_debug", False)):
                dbg_key = f"{int(class_id)}|{','.join(str(int(x)) for x in (kernel_obj_ids or []))}"
                debug_rows = []
                for pack in sorted(obj_support_debug, key=lambda x: (float(x.get("support", 0.0)), int(x.get("object_id", -1))), reverse=True):
                    stats = dict(pack.get("detail", {}) or {})
                    debug_rows.append(
                        {
                            "object_id": int(pack.get("object_id", -1)),
                            "support": float(pack.get("support", 0.0)),
                            "hit_count": int(stats.get("hit_count", 0) or 0),
                            "kernel_count": int(stats.get("kernel_count", 0) or 0),
                            "hit_ratio": float(stats.get("hit_ratio", 0.0) or 0.0),
                            "details": list(stats.get("details", []) or []),
                        }
                    )
                self.score._frame_class_opts_debug[dbg_key] = {
                    "class_id": int(class_id),
                    "kernel_obj_ids": [int(x) for x in (kernel_obj_ids or [])],
                    "det_ids": [int(x) for x in (det_ids or [])],
                    "pool_obj_ids": [int(x) for x in (pool or [])],
                    "rows": list(debug_rows),
                }

            def _generate_options() -> None:
                for k in ks:
                    k_int = int(k)
                    if k_int == 0:
                        out.append({
                            "class_id": int(class_id),
                            "det_ids": (),
                            "object_ids": (),
                            "det_mask": 0,
                            "object_mask": 0,
                            "opt_rank": 0.0,
                            "class_info": 1.0,
                            "class_support": 0.0,
                            "class_logC": 0.0,
                            "class_stability": 0.0,
                            "class_exclusivity": 0.0,
                            "target_k": int(target_k),
                            "class_exclusivity_valid": False,
                            "k": 0,
                        })
                        continue

                    L = min(len(obj_ranked), max(k_int, min(len(obj_ranked), self.score.options_obj_Lmax)))
                    obj_source = obj_ranked[:L]
                    if len(obj_source) < k_int:
                        continue

                    supp_vals = [float(supp_cache.get(int(oid), 0.0)) for oid in obj_source]
                    pref = [0.0]
                    for v in supp_vals:
                        pref.append(pref[-1] + float(v))
                    idx_by_oid = {int(oid): int(i) for i, oid in enumerate(obj_source)}

                    obj_variants: list[tuple[int, ...]] = []
                    obj_variants.append(tuple(int(x) for x in obj_source[:k_int]))
                    if self.score.options_obj_variants > 1 and k_int >= 1:
                        base = [int(x) for x in obj_source[: max(0, k_int - 1)]]
                        max_extra = min(int(self.score.options_obj_variants) - 1, max(0, len(obj_source) - k_int))
                        for t in range(1, max_extra + 1):
                            obj_variants.append(tuple(base + [int(obj_source[k_int - 1 + t])]))

                    variant_ranks: list[tuple[tuple[int, ...], float, float]] = []
                    for osub in obj_variants:
                        if k_int <= 0:
                            supp = 0.0
                        else:
                            if osub == tuple(int(x) for x in obj_source[:k_int]):
                                ssum = float(pref[k_int])
                            else:
                                last = int(osub[-1])
                                idx = idx_by_oid.get(int(last), -1)
                                if idx >= 0:
                                    ssum = float(pref[max(0, k_int - 1)]) + float(supp_vals[int(idx)])
                                else:
                                    ssum = float(sum(float(supp_cache.get(int(oid), 0.0)) for oid in osub))
                            supp = float(ssum / float(k_int))
                        fill = 1.0
                        if target_k > 0:
                            fill = float(min(1.0, float(k_int) / float(target_k)))
                        class_support = float(supp) * (float(fill) ** float(self.score.class_fill_gamma))
                        variant_ranks.append((tuple(int(x) for x in osub), float(supp), float(class_support)))

                    logC = float(self.score.log_n_choose_k(pool_n, k_int))
                    info = float(self.score.class_info_from_logC(logC))
                    det_cap = int(min(int(self.score.options_det_combo_max), int(self.score.options_det_variants)))
                    det_n = 0
                    for dsub in combinations(det_ids, k_int):
                        if det_n >= det_cap:
                            break
                        det_n += 1
                        for osub, supp, class_support in variant_ranks:
                            osub_tuple = tuple(int(x) for x in osub)
                            dsub_tuple = tuple(int(x) for x in dsub)
                            out.append({
                                "class_id": int(class_id),
                                "det_ids": dsub_tuple,
                                "object_ids": osub_tuple,
                                "det_mask": int(self.score.det_mask(dsub_tuple)) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0,
                                "object_mask": int(self.score.object_mask(osub_tuple)) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0,
                                "opt_rank": float(class_support),
                                "class_info": float(info),
                                "class_support": float(class_support),
                                "class_logC": float(logC),
                                "class_stability": float(stability),
                                "class_exclusivity": 0.0,
                                "target_k": int(target_k),
                                "class_exclusivity_valid": False,
                                "k": int(k_int),
                            })

            if timer is None:
                _generate_options()
                out = self.attach_class_exclusivity(out)
                out.sort(key=lambda o: float(o.get("opt_rank", 0.0)), reverse=True)
            else:
                timer.run("build_class_options/generate_options", _generate_options)
                out = timer.run("build_class_options/attach_exclusivity", self.attach_class_exclusivity, out)
                with timer.measure("build_class_options/sort"):
                    out.sort(key=lambda o: float(o.get("opt_rank", 0.0)), reverse=True)

            for o in out:
                oids_raw = o.get("object_ids", ()) or ()
                dids_raw = o.get("det_ids", ()) or ()
                oids = oids_raw if isinstance(oids_raw, tuple) else tuple(int(x) for x in oids_raw)
                dids = dids_raw if isinstance(dids_raw, tuple) else tuple(int(x) for x in dids_raw)
                o["_pair"] = {
                    "class_id": int(class_id),
                    "det_ids": dids,
                    "object_ids": oids,
                }

                kc = int(o.get("k", len(oids)))
                target_k = int(o.get("target_k", kc))
                support_w = float(max(1, target_k))
                support_val = float(max(0.0, min(1.0, o.get("class_support", 0.0))))
                stab_val = float(o.get("class_stability", 0.0))
                excl_w = float(kc) if kc > 0 else 0.0
                excl_val = float(max(0.0, min(1.0, o.get("class_exclusivity", 0.0)))) if kc > 0 else 0.0
                excl_v = bool(o.get("class_exclusivity_valid", False))
                o["_transition_pack"] = (
                    int(kc),
                    float(support_w),
                    float(support_val),
                    float(stab_val),
                    float(excl_w),
                    float(excl_val),
                    bool(excl_v),
                )
            return out[: self.score.max_class_options]

    def build_class_options_cached(
        self,
        class_id: int,
        det_ids: list[int],
        pool_obj_ids: list[int],
        used_obj_ids: set[int],
        kernel_obj_ids: list[int],
        vocab_size: int | None = None,
        used_obj_mask: int = 0,
    ) -> list[dict]:
        used = set(int(x) for x in (used_obj_ids or set()))
        vs = int(vocab_size) if vocab_size is not None else -1
        kernel_sig = tuple(int(x) for x in (kernel_obj_ids or []))
        key = (int(class_id), kernel_sig, int(vs))

        cached = self.score._frame_class_opts_cache.get(key, None)
        if cached is None:
            cached = self.build_class_options(
                class_id=int(class_id),
                det_ids=det_ids,
                pool_obj_ids=pool_obj_ids,
                used_obj_ids=set(),
                kernel_obj_ids=list(kernel_sig),
                vocab_size=vocab_size,
            )
            self.score._frame_class_opts_cache[key] = cached

        used_mask_int = int(used_obj_mask or 0) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0
        if not used and used_mask_int == 0:
            return cached

        used_sig = int(used_mask_int) if used_mask_int else tuple(sorted(int(x) for x in used))
        filtered_key = (int(class_id), kernel_sig, used_sig, int(vs))
        filtered_cached = self.score._frame_filtered_class_opts_cache.get(filtered_key, None)
        if filtered_cached is not None:
            return filtered_cached

        filtered = self._filter_class_options_by_used_obj_ids(
            cached,
            used,
            used_obj_mask=used_mask_int,
        )
        self.score._frame_filtered_class_opts_cache[filtered_key] = filtered
        return filtered

    def attach_class_exclusivity(self, opts: list[dict]) -> list[dict]:
        if not self.score.excl_enabled or not opts:
            return opts

        by_k: dict[int, list[dict]] = {}
        for o in opts:
            k = int(o.get("k", len(o.get("object_ids", ()) or ())))
            by_k.setdefault(int(k), []).append(o)

        for k, group in by_k.items():
            if int(k) <= 0:
                for o in group:
                    o["class_exclusivity"] = 0.0
                    o["class_exclusivity_valid"] = False
                continue

            ranks = [float(o.get("opt_rank", 0.0)) for o in group]
            ranks.sort(reverse=True)
            top1 = float(ranks[0]) if ranks else 0.0
            top2 = 0.0
            for r in ranks[1:]:
                if r < (top1 - 1e-15):
                    top2 = float(r)
                    break

            if top1 <= self.score.excl_eps:
                exc = 0.0
                valid = False
            else:
                exc = float(max(0.0, top1 - top2) / max(self.score.excl_eps, top1))
                valid = bool(top2 > 0.0 or (top1 - top2) > 1e-12)

            for o in group:
                o["class_exclusivity"] = float(max(0.0, min(1.0, exc)))
                o["class_exclusivity_valid"] = bool(valid)

        return opts

    def object_support_to_kernel(self, object_id: int, kernel_obj_ids: list[int], vocab_size: int | None = None) -> float:
        if not kernel_obj_ids:
            return 0.0
        oid = int(object_id)
        vals = []
        for kid in kernel_obj_ids:
            kobj = self.score.memory_store.get(int(kid))
            g = getattr(kobj, "neighbors", None) if kobj is not None else None
            if g is None or not getattr(g, "enabled", False):
                continue
            p = float(self.score.p_conditional_cached(g, int(kid), int(oid), vocab_size=vocab_size))
            if p >= self.score.min_edge_p:
                vals.append(float(p))
        return float(sum(vals) / float(len(vals))) if vals else 0.0

    def object_support_detail_to_kernel(self, object_id: int, kernel_obj_ids: list[int], vocab_size: int | None = None) -> dict:
        if not kernel_obj_ids:
            return {"mean_support": 0.0, "hit_count": 0, "kernel_count": 0, "hit_ratio": 0.0, "details": []}

        oid = int(object_id)
        vals = []
        details = []
        kernel_count = 0
        for kid in kernel_obj_ids:
            kobj = self.score.memory_store.get(int(kid))
            g = getattr(kobj, "neighbors", None) if kobj is not None else None
            if g is None or not getattr(g, "enabled", False):
                continue
            kernel_count += 1
            p = float(self.score.p_conditional_cached(g, int(kid), int(oid), vocab_size=vocab_size))
            hit = bool(float(p) >= float(self.score.min_edge_p))
            if hit:
                vals.append(float(p))
            details.append(
                {
                    "kernel_id": int(kid),
                    "p": float(p),
                    "hit": bool(hit),
                }
            )

        hit_count = int(len(vals))
        mean_support = float(sum(vals) / float(hit_count)) if hit_count > 0 else 0.0
        hit_ratio = float(hit_count / float(max(1, kernel_count))) if kernel_count > 0 else 0.0
        return {
            "mean_support": float(mean_support),
            "hit_count": int(hit_count),
            "kernel_count": int(kernel_count),
            "hit_ratio": float(hit_ratio),
            "details": list(details),
        }
