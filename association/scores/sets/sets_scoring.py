from __future__ import annotations

import math
from time import perf_counter


class SetsScoring:
    def __init__(self, *, score) -> None:
        self.score = score

    def frame_timer(self):
        timer = getattr(self.score, "_frame_timer", None)
        return timer

    def state_class_aggregates(self, st: dict) -> dict:
        support_wsum = float(st.get("class_support_wsum", 0.0))
        stab_count = int(st.get("class_stability_count", 0))
        excl_wsum = float(st.get("excl_wsum", 0.0))
        return {
            "class_info": float(st.get("class_info", 1.0)),
            "class_support": float(st.get("class_support_sum", 0.0)) / support_wsum if support_wsum > 0.0 else 0.0,
            "class_support_valid": bool(support_wsum > 0.0),
            "class_stability": float(st.get("class_stability_sum", 0.0)) / float(stab_count) if stab_count > 0 else 0.0,
            "class_stability_valid": bool(stab_count > 0),
            "class_logC_sum": float(st.get("class_logC_sum", 0.0)),
            "exclusivity": float(st.get("excl_sum", 0.0)) / excl_wsum if excl_wsum > 0.0 else 0.0,
            "excl_any_valid": bool(st.get("excl_any_valid", False)),
        }

    def state_object_ids(self, st: dict) -> tuple[int, ...]:
        obj_mask = int(st.get("obj_mask", 0) or 0) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0
        if obj_mask:
            return self.score.object_ids_from_mask(obj_mask)
        return tuple(int(x) for x in (st.get("obj_ids_sorted", ()) or ()))

    def state_explained_det_ids(self, st: dict) -> tuple[int, ...]:
        det_mask = int(st.get("explained_det_mask", 0) or 0) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0
        if det_mask:
            return self.score.det_ids_from_mask(det_mask)
        return tuple(int(x) for x in (st.get("explained_det_sorted", ()) or ()))

    def score_state_quick(
        self,
        *,
        st: dict,
        total_dets: int,
        anchors: list[int],
        vocab_size: int | None = None,
    ) -> float:
        use_masks = bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)())
        obj_mask = int(st.get("obj_mask", 0) or 0) if use_masks else 0
        if obj_mask > 0:
            k = int(obj_mask.bit_count())
            obj_ids = []
        else:
            obj_ids = list(self.state_object_ids(st))
            k = int(len(obj_ids))

        explained_n = int(st.get("explained_n", len(st.get("explained_det_sorted", ()) or ())))
        cov_raw = float(explained_n) / float(max(1, int(total_dets)))
        cov_raw = float(max(0.0, min(1.0, cov_raw)))
        cov_eff = float(self.coverage_effective(cov_raw, explained_n, int(total_dets)))

        size_util = float(self.size_utility(k))
        timer = self.frame_timer()
        if timer is None:
            if obj_mask > 0:
                dens, dens_valid, edge_cov, node_cov, min_deg = self.score.density_score_cached_by_mask(obj_mask, vocab_size=vocab_size)
            else:
                dens, dens_valid, edge_cov, node_cov, min_deg = self.score.density_score_cached(obj_ids, vocab_size=vocab_size)
        else:
            t0 = perf_counter()
            if obj_mask > 0:
                dens, dens_valid, edge_cov, node_cov, min_deg = self.score.density_score_cached_by_mask(obj_mask, vocab_size=vocab_size)
            else:
                dens, dens_valid, edge_cov, node_cov, min_deg = self.score.density_score_cached(obj_ids, vocab_size=vocab_size)
            timer.add(
                "run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick/density_score_cached",
                perf_counter() - t0,
            )

        class_info = float(st.get("class_info", 1.0))
        support_wsum = float(st.get("class_support_wsum", 0.0))
        class_support_valid = bool(support_wsum > 0.0)
        class_support = float(st.get("class_support_sum", 0.0)) / support_wsum if class_support_valid else 0.0
        stab_count = int(st.get("class_stability_count", 0))
        class_stability_valid = bool(stab_count > 0)
        class_stability = float(st.get("class_stability_sum", 0.0)) / float(stab_count) if class_stability_valid else 0.0
        excl_wsum = float(st.get("excl_wsum", 0.0))
        exclusivity = float(st.get("excl_sum", 0.0)) / excl_wsum if excl_wsum > 0.0 else 0.0
        excl_any_valid = bool(st.get("excl_any_valid", False))

        if timer is None:
            if obj_mask > 0:
                mean_maturity, maturity_coh, maturity_rel = self.score.maturity_pack_cached_by_mask(obj_mask)
            else:
                mean_maturity, maturity_coh, maturity_rel = self.score.maturity_pack_cached(obj_ids)
        else:
            t0 = perf_counter()
            if obj_mask > 0:
                mean_maturity, maturity_coh, maturity_rel = self.score.maturity_pack_cached_by_mask(obj_mask)
            else:
                mean_maturity, maturity_coh, maturity_rel = self.score.maturity_pack_cached(obj_ids)
            timer.add(
                "run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick/maturity_pack_cached",
                perf_counter() - t0,
            )
        exclusivity_valid = bool(
            self.score.excl_enabled
            and bool(excl_any_valid)
            and int(k) >= int(self.score.excl_k_min)
            and float(mean_maturity) >= float(self.score.excl_maturity_min)
        )

        score_sets = float(
            self.combine_score(
                k=int(k),
                coverage_eff=cov_eff,
                size_util=size_util,
                density=dens,
                density_valid=dens_valid,
                class_info=class_info,
                class_support=class_support,
                class_support_valid=class_support_valid,
                class_stability=class_stability,
                class_stability_valid=class_stability_valid,
                exclusivity=exclusivity,
                exclusivity_valid=exclusivity_valid,
            )
        )

        conn_factor = 1.0
        if int(k) >= 2 and float(self.score.min_edge_p) > 0.0:
            if self.score.connectivity_require_min_degree > 0 and int(min_deg) < int(self.score.connectivity_require_min_degree):
                conn_factor = 0.0
            else:
                if self.score.connectivity_node_gamma > 0.0:
                    conn_factor *= float(max(0.0, min(1.0, node_cov))) ** float(self.score.connectivity_node_gamma)
                if self.score.connectivity_edge_gamma > 0.0:
                    conn_factor *= float(max(0.0, min(1.0, edge_cov))) ** float(self.score.connectivity_edge_gamma)

        score_sets = float(score_sets) * float(conn_factor) * float(maturity_rel)
        return float(max(0.0, min(1.0, score_sets)))

    def score_state_as_hypothesis(
        self,
        st: dict,
        total_dets: int,
        all_det_ids: list[int],
        anchors: list[int],
        vocab_size: int | None = None,
    ) -> dict:
        obj_ids = list(self.state_object_ids(st))
        explained = list(self.state_explained_det_ids(st))

        explained_n = int(len(explained))
        cov_raw = float(explained_n) / float(max(1, int(total_dets)))
        cov_raw = float(max(0.0, min(1.0, cov_raw)))
        cov_eff = float(self.coverage_effective(cov_raw, explained_n, int(total_dets)))

        k = int(len(obj_ids))
        size_util = float(self.size_utility(k))
        density, dens_valid, edge_cov, node_cov, min_deg = self.score.density_score(obj_ids, vocab_size=vocab_size)

        aggr = self.state_class_aggregates(st)
        class_info = float(aggr.get("class_info", 1.0))
        class_support = float(aggr.get("class_support", 0.0))
        class_support_valid = bool(aggr.get("class_support_valid", False))
        class_stability = float(aggr.get("class_stability", 0.0))
        class_stability_valid = bool(aggr.get("class_stability_valid", False))
        logC_sum = float(aggr.get("class_logC_sum", 0.0))
        exclusivity = float(aggr.get("exclusivity", 0.0))

        mean_maturity = self.score.mean_maturity(obj_ids)
        exclusivity_valid = bool(
            self.score.excl_enabled
            and bool(aggr.get("excl_any_valid", False))
            and int(k) >= int(self.score.excl_k_min)
            and float(mean_maturity) >= float(self.score.excl_maturity_min)
        )

        maturity_coh = float(self.score.maturity_coherence(obj_ids))
        maturity_rel = 1.0
        if self.score.maturity_enabled and self.score.maturity_gamma > 0.0:
            maturity_rel = float(max(0.0, min(1.0, maturity_coh))) ** float(self.score.maturity_gamma)

        score_sets = float(
            self.combine_score(
                k=int(k),
                coverage_eff=cov_eff,
                size_util=size_util,
                density=density,
                density_valid=dens_valid,
                class_info=class_info,
                class_support=class_support,
                class_support_valid=class_support_valid,
                class_stability=class_stability,
                class_stability_valid=class_stability_valid,
                exclusivity=exclusivity,
                exclusivity_valid=exclusivity_valid,
            )
        )

        conn_factor = 1.0
        if int(k) >= 2 and float(self.score.min_edge_p) > 0.0:
            if self.score.connectivity_require_min_degree > 0 and int(min_deg) < int(self.score.connectivity_require_min_degree):
                conn_factor = 0.0
            else:
                if self.score.connectivity_node_gamma > 0.0:
                    conn_factor *= float(max(0.0, min(1.0, node_cov))) ** float(self.score.connectivity_node_gamma)
                if self.score.connectivity_edge_gamma > 0.0:
                    conn_factor *= float(max(0.0, min(1.0, edge_cov))) ** float(self.score.connectivity_edge_gamma)

        ctx_cov_eff = 1.0
        if self.score.context_enabled and self.score.context_k > 0 and int(k) >= 2:
            ctx_cov_eff = float(self.context_coverage_effective(obj_ids, vocab_size=vocab_size))

        score_sets = float(score_sets) * float(conn_factor) * float(maturity_rel)
        score_sets = float(max(0.0, min(1.0, score_sets)))

        return {
            "object_ids": obj_ids,
            "det_ids_explained": explained,
            "det_ids_unexplained": self.score.build_unexplained_det_ids(all_det_ids, explained),
            "pairs": list(st.get("pairs", []) or []),
            "coverage": float(cov_raw),
            "coverage_eff": float(cov_eff),
            "k": int(k),
            "size_util": float(size_util),
            "density": float(density),
            "density_valid": bool(dens_valid),
            "edge_cov": float(edge_cov),
            "node_cov": float(node_cov),
            "conn_factor": float(conn_factor),
            "conn_min_deg": int(min_deg),
            "ctx_cov_eff": float(ctx_cov_eff),
            "class_info": float(class_info),
            "class_logC_sum": float(logC_sum),
            "class_support": float(class_support),
            "class_support_valid": bool(class_support_valid),
            "class_stability": float(class_stability),
            "class_stability_valid": bool(class_stability_valid),
            "exclusivity": float(exclusivity),
            "exclusivity_valid": bool(exclusivity_valid),
            "mean_maturity": float(mean_maturity),
            "maturity_coh": float(maturity_coh),
            "maturity_rel": float(maturity_rel),
            "score_sets": float(score_sets),
            "anchors": [int(x) for x in anchors],
        }

    def context_coverage_effective(self, object_ids: list[int], vocab_size: int | None = None) -> float:
        ids = [int(x) for x in (object_ids or [])]
        if len(ids) < 2:
            return 1.0

        S = set(ids)
        total_w = 0.0
        sum_cov = 0.0

        for oid in ids:
            obj = self.score.memory_store.get(int(oid))
            g = getattr(obj, "neighbors", None) if obj is not None else None
            if g is None or not getattr(g, "enabled", False):
                continue

            e = float(max(0, int(getattr(g, "episode_count", 0))))
            w = float(1.0 - math.exp(-e / float(self.score.context_maturity_tau)))
            if w <= 1e-12:
                continue

            exp_ids = self.score.expected_neighbors_topk(int(oid), topk=self.score.context_k, vocab_size=vocab_size)
            if not exp_ids:
                continue

            present = sum(1 for nid in exp_ids if int(nid) in S)
            cov = float(present) / float(max(1, len(exp_ids)))
            cov = float(max(0.0, min(1.0, cov)))

            sum_cov += float(w) * float(cov)
            total_w += float(w)

        if total_w <= 1e-12:
            return 1.0
        return float(max(0.0, min(1.0, sum_cov / total_w)))

    def coverage_effective(self, coverage_raw: float, explained_n: int, total_dets: int) -> float:
        cov = float(max(0.0, min(1.0, float(coverage_raw))))
        cov = float(cov ** self.score.coverage_gamma)

        if total_dets > 1 and self.score.coverage_size_boost > 0.0:
            t = float(max(0, int(total_dets) - 1))
            boost = float(1.0 - math.exp(-t / float(self.score.coverage_size_tau)))
            cov = float(cov * (1.0 + float(self.score.coverage_size_boost) * boost))

        if self.score.coverage_explained_beta > 1e-12:
            e = float(max(0, int(explained_n)))
            damp = float(1.0 - math.exp(-e / float(self.score.coverage_explained_beta)))
            cov = float(cov * damp)

        return float(max(0.0, min(1.0, cov)))

    def size_utility(self, k: int) -> float:
        kk = int(max(0, int(k)))
        if kk <= int(self.score.size_k_min):
            return 0.0
        x = float(kk - int(self.score.size_k_min))
        return float(max(0.0, min(1.0, 1.0 - math.exp(-x / float(self.score.size_tau)))))

    def combine_score(
        self,
        k: int,
        coverage_eff: float,
        size_util: float,
        density: float,
        density_valid: bool,
        class_info: float,
        class_support: float,
        class_support_valid: bool,
        class_stability: float,
        class_stability_valid: bool,
        exclusivity: float,
        exclusivity_valid: bool,
    ) -> float:
        w_info = self.scaled_weight(float(self.score.w_class_info), int(k), kind="info")
        parts = [
            (float(self.score.w_coverage), float(max(0.0, min(1.0, coverage_eff)))),
            (float(self.score.w_size), float(max(0.0, min(1.0, size_util)))),
            (float(w_info), float(max(0.0, min(1.0, class_info)))),
        ]
        if density_valid:
            parts.append((float(self.score.w_density), float(max(0.0, min(1.0, density)))))
        if class_support_valid:
            parts.append((float(self.score.w_class_support), float(max(0.0, min(1.0, class_support)))))
        if class_stability_valid:
            parts.append((float(self.score.w_class_stability), float(max(0.0, min(1.0, class_stability)))))
        if exclusivity_valid:
            w_excl = self.scaled_weight(float(self.score.w_exclusivity), int(k), kind="excl")
            parts.append((float(w_excl), float(max(0.0, min(1.0, exclusivity)))))

        used = [(w, v) for w, v in parts if w > 0.0]
        wsum = float(sum(w for w, _ in used))
        if wsum <= 1e-12:
            return 0.0

        s = 0.0
        for w, v in used:
            s += float(w) * float(v)
        return float(max(0.0, min(1.0, s / wsum)))

    def scaled_weight(self, base_weight: float, k: int, kind: str) -> float:
        bw = float(base_weight)
        if bw <= 0.0:
            return 0.0

        kk = int(max(1, int(k)))
        if kind == "info":
            mult = self.k_weight_multiplier(
                kk,
                ref=float(self.score.info_k_ref),
                gamma=float(self.score.info_k_gamma),
                min_mult=float(self.score.info_k_min_mult),
                max_mult=float(self.score.info_k_max_mult),
            )
            return float(bw * float(mult))

        if kind == "excl":
            mult = self.k_weight_multiplier(
                kk,
                ref=float(self.score.excl_k_ref),
                gamma=float(self.score.excl_k_gamma),
                min_mult=float(self.score.excl_k_min_mult),
                max_mult=float(self.score.excl_k_max_mult),
            )
            return float(bw * float(mult))

        return float(bw)

    def k_weight_multiplier(self, k: int, ref: float, gamma: float, min_mult: float, max_mult: float) -> float:
        kk = int(max(1, int(k)))
        r = float(max(1e-6, float(ref)))
        g = float(max(0.0, float(gamma)))
        lo = float(max(0.0, float(min_mult)))
        hi = float(max(lo, float(max_mult)))
        base = (r / float(kk)) ** float(g) if g > 0.0 else 1.0
        return float(max(lo, min(hi, float(base))))

    def log_n_choose_k(self, n: int, k: int) -> float:
        nn = int(max(0, int(n)))
        kk = int(max(0, int(k)))
        if kk > nn or kk == 0 or kk == nn:
            return 0.0
        return float(math.lgamma(nn + 1) - math.lgamma(kk + 1) - math.lgamma(nn - kk + 1))

    def class_info_from_logC(self, logC: float) -> float:
        x = float(max(0.0, float(logC)))
        return float(1.0 / (1.0 + self.score.class_ambig_beta * x))
