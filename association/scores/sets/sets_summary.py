from __future__ import annotations


class SetsSummaryBuilder:
    def __init__(self, *, memory_store, shortlist_rel: float, priors_top_m: int, selective_min_prior: float, selective_rel_gap: float) -> None:
        self.memory_store = memory_store
        self.shortlist_rel = float(shortlist_rel)
        self.priors_top_m = int(priors_top_m)
        self.selective_min_prior = float(selective_min_prior)
        self.selective_rel_gap = float(selective_rel_gap)

    def build_result(
        self,
        *,
        hypotheses: list[dict],
        anchors: list[int],
        total_dets: int,
        dets_by_class: dict[int, list],
        timestamp: float | None,
    ) -> dict:
        object_support_top, object_support_sum = self.build_object_support(hypotheses)
        best_pack = self.best_second_summary(hypotheses)

        shortlist, thr_shortlist = self.build_shortlist(hypotheses)
        prior_by_oid = {int(k): float(v) for k, v in (object_support_top or {}).items()}
        class_prior_by_cid = self.class_priors_from_shortlist(shortlist=shortlist, prior_by_oid=prior_by_oid)
        selective_classes = self.selective_classes_from_priors(class_prior_by_cid)

        core = {
            "enabled": True,
            "anchors": list(anchors),
            "n_hypotheses": int(len(hypotheses)),
            "best_score": float(best_pack.get("best_score", 0.0)),
            "second_score": float(best_pack.get("second_score", 0.0)),
            "gap_best": float(best_pack.get("gap_best", 0.0)),
            "mean_maturity_best": float(best_pack.get("mean_maturity_best", 0.0)),
            "thr_shortlist": float(thr_shortlist),
            "shortlist": sorted(int(x) for x in shortlist),
            "prior_by_oid": prior_by_oid,
            "class_prior_by_cid": {int(k): float(v) for k, v in (class_prior_by_cid or {}).items()},
            "selective_classes": sorted(int(x) for x in (selective_classes or set())),
        }

        meta = {
            "score_key": "score_sets",
            "n_dets": int(total_dets),
            "n_classes": int(len(dets_by_class)),
            "classes": sorted(int(x) for x in dets_by_class.keys()),
            "anchors": list(anchors),
            "n_hypotheses": int(len(hypotheses)),
        }
        meta.update(best_pack)

        return {
            "core": core,
            "debug": {
                "meta": meta,
                "set_hypotheses": list(hypotheses or []),
                "object_support_sum": {int(k): float(v) for k, v in (object_support_sum or {}).items()},
                "timestamp": float(timestamp) if timestamp is not None else None,
            },
        }

    def build_unexplained_det_ids(self, all_det_ids: list[int], explained_det_ids: list[int]) -> list[int]:
        all_ids = [int(x) for x in (all_det_ids or [])]
        expl = set(int(x) for x in (explained_det_ids or []))
        return [int(did) for did in all_ids if int(did) not in expl]

    def build_object_support(self, hypotheses: list[dict]) -> tuple[dict[int, float], dict[int, float]]:
        top_m = int(self.priors_top_m)
        if top_m <= 0:
            top = list(hypotheses or [])
        else:
            top = (hypotheses or [])[:top_m]

        support_top: dict[int, float] = {}
        support_sum: dict[int, float] = {}

        for h in top:
            s = float(h.get("score_sets", 0.0))
            for oid in h.get("object_ids", []) or []:
                oid = int(oid)
                prev = float(support_top.get(oid, 0.0))
                if s > prev:
                    support_top[oid] = float(s)
                support_sum[oid] = float(support_sum.get(oid, 0.0) + s)

        return support_top, support_sum

    def best_second_summary(self, hypotheses: list[dict]) -> dict:
        if not hypotheses:
            return {
                "best_score": 0.0,
                "second_score": 0.0,
                "gap_best": 0.0,
                "k_best": 0,
                "coverage_best": 0.0,
                "coverage_eff_best": 0.0,
                "size_best": 0.0,
                "density_best": 0.0,
                "class_info_best": 0.0,
                "class_stability_best": 0.0,
                "exclusivity_best": 0.0,
                "density_valid_best": False,
                "class_stability_valid_best": False,
                "exclusivity_valid_best": False,
                "mean_maturity_best": 0.0,
            }

        best = hypotheses[0]
        second = hypotheses[1] if len(hypotheses) > 1 else None

        s1 = float(best.get("score_sets", 0.0))
        s2 = float(second.get("score_sets", 0.0)) if second is not None else 0.0
        gap = float(max(0.0, s1 - s2))

        return {
            "best_score": float(s1),
            "second_score": float(s2),
            "gap_best": float(gap),
            "k_best": int(best.get("k", len(best.get("object_ids", []) or []))),
            "coverage_best": float(best.get("coverage", 0.0)),
            "coverage_eff_best": float(best.get("coverage_eff", best.get("coverage", 0.0))),
            "size_best": float(best.get("size_util", 0.0)),
            "density_best": float(best.get("density", 0.0)),
            "class_info_best": float(best.get("class_info", 0.0)),
            "class_stability_best": float(best.get("class_stability", 0.0)),
            "exclusivity_best": float(best.get("exclusivity", 0.0)),
            "density_valid_best": bool(best.get("density_valid", False)),
            "class_stability_valid_best": bool(best.get("class_stability_valid", False)),
            "exclusivity_valid_best": bool(best.get("exclusivity_valid", False)),
            "mean_maturity_best": float(best.get("mean_maturity", 0.0)),
        }

    def build_shortlist(self, hyps: list[dict]) -> tuple[set[int], float]:
        if not hyps:
            return set(), 0.0

        top = hyps[0] if isinstance(hyps[0], dict) else {}
        best_score = float(top.get("score_sets", 0.0))
        thr = float(best_score * (1.0 - float(self.shortlist_rel)))

        shortlist: set[int] = set()
        for h in hyps:
            if not isinstance(h, dict):
                continue
            s = float(h.get("score_sets", 0.0))
            if s < thr:
                continue
            for oid in h.get("object_ids", []) or []:
                shortlist.add(int(oid))

        return shortlist, float(thr)

    def class_priors_from_shortlist(self, shortlist: set[int], prior_by_oid: dict[int, float]) -> dict[int, float]:
        by_class: dict[int, float] = {}
        for oid in shortlist or set():
            obj = self.memory_store.get(int(oid)) if self.memory_store is not None else None
            if obj is None:
                continue
            cid = int(getattr(obj, "class_id", -1))
            p = float((prior_by_oid or {}).get(int(oid), 0.0))
            prev = float(by_class.get(cid, 0.0))
            if p > prev:
                by_class[cid] = float(p)
        return {int(cid): float(p) for cid, p in by_class.items()}

    def selective_classes_from_priors(self, class_prior_by_cid: dict[int, float]) -> set[int]:
        by_class: dict[int, list[float]] = {}
        for cid, p in (class_prior_by_cid or {}).items():
            by_class.setdefault(int(cid), []).append(float(p))

        selective: set[int] = set()
        for cid, ps in by_class.items():
            ps = [float(x) for x in ps if x is not None]
            if not ps:
                continue
            ps.sort(reverse=True)
            p1 = float(ps[0])
            p2 = float(ps[1]) if len(ps) > 1 else 0.0

            if p1 < self.selective_min_prior:
                continue
            rel_gap = float((p1 - p2) / max(1e-12, p1))
            if rel_gap >= self.selective_rel_gap:
                selective.add(int(cid))

        return selective
