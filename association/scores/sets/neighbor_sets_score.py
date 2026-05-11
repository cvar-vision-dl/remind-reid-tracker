# association/scores/sets/neighbor_sets_score.py

from __future__ import annotations

from association.scores.sets.sets_graph_utils import SetsGraphUtils
from association.scores.sets.sets_options import SetsOptionsBuilder
from association.scores.sets.sets_scoring import SetsScoring
from association.scores.sets.sets_search import SetsSearchEngine
from association.scores.sets.sets_summary import SetsSummaryBuilder
from utils.time import ExecutionTimer


class NeighborSetsScore:
    """Generate set hypotheses (detection subsets <-> ID subsets) using the ONLINE graph."""

    def __init__(self, config: dict | None = None, memory_store=None):
        self.config = config or {}
        cfg = (config or {}).get("association", {}) or {}
        sc = (cfg.get("scores", {}) or {}).get("neighbor_sets", {}) or {}

        self.memory_store = memory_store

        self.topk_sets = max(1, int(sc.get("topk_sets", 10)))
        self.beam_width = max(1, int(sc.get("beam_width", 64)))

        self.per_class_det_k = max(1, int(sc.get("per_class_det_k", 10)))
        self.per_class_pool_k = max(1, int(sc.get("per_class_pool_k", 25)))
        self.max_class_options = max(1, int(sc.get("max_class_options", 200)))

        self.max_set_size = max(0, int(sc.get("max_set_size", 0)))
        self.min_set_score = float(sc.get("min_set_score", 0.0))
        self.allow_partial_coverage = bool(sc.get("allow_partial_coverage", True))

        self.min_edge_p = float(sc.get("min_edge_p", 0.0))
        self.min_edge_p = max(0.0, min(1.0, self.min_edge_p))

        self.density_gate_min_edge_p = bool(sc.get("density_gate_min_edge_p", False))
        self.density_edge_cov_gamma = float(sc.get("density_edge_cov_gamma", 0.0))
        self.density_edge_cov_gamma = max(0.0, self.density_edge_cov_gamma)

        self.connectivity_require_min_degree = int(sc.get("connectivity_require_min_degree", 0))
        self.connectivity_require_min_degree = max(0, self.connectivity_require_min_degree)
        self.connectivity_node_gamma = float(sc.get("connectivity_node_gamma", 0.0))
        self.connectivity_node_gamma = max(0.0, self.connectivity_node_gamma)
        self.connectivity_edge_gamma = float(sc.get("connectivity_edge_gamma", 0.0))
        self.connectivity_edge_gamma = max(0.0, self.connectivity_edge_gamma)

        self.use_mutual = bool(sc.get("use_mutual", True))
        self.kernel_max = max(0, int(sc.get("kernel_max", 6)))

        self.coverage_gamma = float(sc.get("coverage_gamma", 1.0))
        self.coverage_gamma = max(1e-6, self.coverage_gamma)

        self.coverage_size_tau = float(sc.get("coverage_size_tau", 3.0))
        self.coverage_size_tau = max(1e-6, self.coverage_size_tau)

        self.coverage_size_boost = float(sc.get("coverage_size_boost", 0.35))
        self.coverage_size_boost = max(0.0, min(1.0, self.coverage_size_boost))

        self.coverage_explained_beta = float(sc.get("coverage_explained_beta", 0.0))
        self.coverage_explained_beta = max(0.0, self.coverage_explained_beta)

        size_cfg = (sc.get("size", {}) or {})
        self.size_k_min = int(size_cfg.get("k_min", 2))
        self.size_tau = float(size_cfg.get("tau", 2.0))
        self.size_tau = max(1e-6, self.size_tau)

        cls_cfg = (sc.get("class_terms", {}) or {})
        self.class_ambig_beta = float(cls_cfg.get("ambig_beta", 1.0))
        self.class_ambig_beta = max(1e-12, self.class_ambig_beta)

        self.class_stability_eps = float(cls_cfg.get("stability_eps", 1e-12))
        self.class_stability_eps = max(1e-15, self.class_stability_eps)
        self.class_plausible_rel = float(cls_cfg.get("plausible_rel", 0.75))
        self.class_plausible_rel = max(0.0, min(1.0, self.class_plausible_rel))
        self.class_fill_gamma = float(cls_cfg.get("fill_gamma", 1.0))
        self.class_fill_gamma = max(0.0, self.class_fill_gamma)

        # Reweighting by set size (k):
        # - small sets: class "rarity"/information matters more
        # - sets grandes: esa rareza importa menos (el propio set ya es discriminativo)
        self.info_k_ref = float(cls_cfg.get("info_k_ref", 2.0))
        self.info_k_gamma = float(cls_cfg.get("info_k_gamma", 0.5))
        self.info_k_min_mult = float(cls_cfg.get("info_k_min_mult", 0.35))
        self.info_k_max_mult = float(cls_cfg.get("info_k_max_mult", 1.0))

        self.excl_k_ref = float(cls_cfg.get("excl_k_ref", 3.0))
        self.excl_k_gamma = float(cls_cfg.get("excl_k_gamma", 0.5))
        self.excl_k_min_mult = float(cls_cfg.get("excl_k_min_mult", 0.35))
        self.excl_k_max_mult = float(cls_cfg.get("excl_k_max_mult", 1.0))

        self.info_k_ref = max(1.0, self.info_k_ref)
        self.info_k_gamma = max(0.0, self.info_k_gamma)
        self.info_k_min_mult = max(0.0, self.info_k_min_mult)
        self.info_k_max_mult = max(self.info_k_min_mult, self.info_k_max_mult)

        self.excl_k_ref = max(1.0, self.excl_k_ref)
        self.excl_k_gamma = max(0.0, self.excl_k_gamma)
        self.excl_k_min_mult = max(0.0, self.excl_k_min_mult)
        self.excl_k_max_mult = max(self.excl_k_min_mult, self.excl_k_max_mult)

        self.maturity_enabled = bool(cls_cfg.get("maturity_enabled", True))
        self.maturity_softmin_p = float(cls_cfg.get("maturity_softmin_p", -4.0))
        self.maturity_gamma = float(cls_cfg.get("maturity_gamma", 1.0))
        self.maturity_gamma = max(0.0, self.maturity_gamma)

        exc_cfg = (sc.get("exclusivity", {}) or {})
        self.excl_enabled = bool(exc_cfg.get("enabled", True))
        self.excl_k_min = int(exc_cfg.get("k_min", 3))
        self.excl_maturity_min = float(exc_cfg.get("maturity_min", 0.35))
        self.excl_maturity_min = max(0.0, min(1.0, self.excl_maturity_min))
        self.excl_eps = float(exc_cfg.get("eps", 1e-12))
        self.excl_eps = max(1e-15, self.excl_eps)

        w = (sc.get("weights", {}) or {})
        self.w_coverage = float(w.get("coverage", 0.40))
        self.w_size = float(w.get("size", 0.20))
        self.w_density = float(w.get("density", 0.20))
        self.w_class_info = float(w.get("class_info", 0.10))
        self.w_class_support = float(w.get("class_support", 0.10))
        self.w_class_stability = float(w.get("class_stability", 0.10))
        self.w_exclusivity = float(w.get("exclusivity", 0.05))

        self.conf_lambda = float(sc.get("conf_lambda", 0.25))
        self.conf_lambda = max(1e-12, self.conf_lambda)

        core_cfg = (sc.get("core", {}) or {})
        self.shortlist_rel = float(core_cfg.get("shortlist_rel", 0.10))
        self.shortlist_rel = max(0.0, min(0.95, self.shortlist_rel))

        self.priors_top_m = int(core_cfg.get("priors_top_m", 0))
        self.priors_top_m = max(0, self.priors_top_m)

        sel_cfg = (core_cfg.get("selective_classes", {}) or {})
        self.selective_min_prior = float(sel_cfg.get("min_prior", 0.50))
        self.selective_rel_gap = float(sel_cfg.get("rel_gap", 0.12))
        self.selective_min_prior = max(0.0, min(1.0, self.selective_min_prior))
        self.selective_rel_gap = max(0.0, min(1.0, self.selective_rel_gap))

        ctx_cfg = (sc.get("context", {}) or {})
        self.context_enabled = bool(ctx_cfg.get("enabled", True))
        self.context_k = int(ctx_cfg.get("k", 6))
        self.context_k = max(0, self.context_k)
        self.context_min_p = float(ctx_cfg.get("min_p", self.min_edge_p))
        self.context_min_p = max(0.0, min(1.0, self.context_min_p))
        self.context_gamma = float(ctx_cfg.get("gamma", 2.0))
        self.context_gamma = max(0.0, self.context_gamma)
        self.context_maturity_tau = float(ctx_cfg.get("maturity_tau", 8.0))
        self.context_maturity_tau = max(1e-6, self.context_maturity_tau)

        matrix_cfg = (sc.get("candidate_pair_matrix", {}) or {})
        self.candidate_pair_matrix_enabled = bool(matrix_cfg.get("enabled", True))
        self.candidate_pair_matrix_max_objects = int(matrix_cfg.get("max_objects", 160))
        self.candidate_pair_matrix_max_objects = max(0, self.candidate_pair_matrix_max_objects)

        self.graph_utils = SetsGraphUtils(score=self)
        self.options_builder = SetsOptionsBuilder(score=self)
        self.scoring = SetsScoring(score=self)
        self.search_engine = SetsSearchEngine(score=self)
        self.summary_builder = SetsSummaryBuilder(
            memory_store=memory_store,
            shortlist_rel=self.shortlist_rel,
            priors_top_m=self.priors_top_m,
            selective_min_prior=self.selective_min_prior,
            selective_rel_gap=self.selective_rel_gap,
        )
        self._frame_class_opts_debug = {}

        dens_cfg = (sc.get("density_factor", {}) or {})
        self.density_factor_enabled = bool(dens_cfg.get("enabled", True))
        self.density_factor_gamma = float(dens_cfg.get("gamma", 2.0))
        self.density_factor_gamma = max(0.0, self.density_factor_gamma)

        opt_cfg = (sc.get("options", {}) or {})
        self.options_obj_Lmax = int(opt_cfg.get("obj_Lmax", 8))
        self.options_obj_Lmax = max(1, self.options_obj_Lmax)
        self.options_det_combo_max = int(opt_cfg.get("det_combo_max", 32))
        self.options_det_combo_max = max(1, self.options_det_combo_max)
        # Number of detection-combination variants per k (per class).
        # Note: set score does NOT depend on the detection subset, only on how many are explained.
        # Keeping few variants reduces state explosion without affecting score_sets.
        self.options_det_variants = int(opt_cfg.get("det_variants", 2))
        self.options_det_variants = max(1, self.options_det_variants)
        # Number of object-combination variants per k (per class).
        # Motivation: avoid the combinatorial explosion (C(L,k)) that dominates runtime.
        # 1 => top-1 only; 2 => top-1 and top-2 (recommended).
        self.options_obj_variants = int(opt_cfg.get("obj_variants", 2))
        self.options_obj_variants = max(1, self.options_obj_variants)
        raw_beam_state_mode = sc.get("beam_state_mode", "classic")
        if isinstance(raw_beam_state_mode, bool):
            self.beam_state_mode = "bitmask_full" if bool(raw_beam_state_mode) else "classic"
        else:
            mode = str(raw_beam_state_mode or "classic").strip().lower()
            if mode not in ("classic", "bitmask_full", "bitmask_used"):
                mode = "classic"
            self.beam_state_mode = str(mode)

        dbg_cfg = (self.config.get("debug", {}) or {}) if isinstance(self.config, dict) else {}
        assoc_dbg = (dbg_cfg.get("association", {}) or {}) if isinstance(dbg_cfg, dict) else {}
        self.collect_class_options_debug = bool(dbg_cfg.get("enabled", False)) and bool(assoc_dbg.get("enabled", True)) and bool(assoc_dbg.get("show_neighbor_sets_table", False))
        self.last_timings_seconds: dict[str, float] = {}
        self._frame_timer: ExecutionTimer | None = None

    def compute(
        self,
        detections: list,
        anchor_object_ids: list[int] | None = None,
        timestamp: float | None = None,
        vocab_size: int | None = None,
    ) -> dict:
        timer = ExecutionTimer()
        self._frame_timer = timer
        self.last_timings_seconds = {}
        try:
            timer.run("init_frame_caches", self.init_frame_caches)

            dets = detections or []
            total_dets = int(len(dets))
            anchors = [int(x) for x in (anchor_object_ids or []) if x is not None]

            if self.memory_store is None or total_dets <= 0:
                return self.empty_result(total_dets=total_dets, anchors=anchors, timestamp=timestamp)

            all_det_ids = timer.run("all_detection_ids", self.all_detection_ids, dets)

            dets_by_class = timer.run("group_dets_by_class", self.group_dets_by_class, dets)
            if not dets_by_class:
                return self.empty_result(total_dets=total_dets, anchors=anchors, timestamp=timestamp)

            class_items = timer.run("select_top_dets_per_class", self.select_top_dets_per_class, dets_by_class)
            pools, pools_meta = timer.run(
                "build_pools_for_classes",
                self.build_pools_for_classes,
                class_items,
                anchors,
                vocab_size=vocab_size,
            )
            if not pools:
                return self.empty_result(
                    total_dets=total_dets,
                    anchors=anchors,
                    timestamp=timestamp,
                    n_classes=int(len(class_items)),
                )

            if self.use_beam_bitmask_mode():
                timer.run(
                    "setup_frame_bitmasks",
                    self.setup_frame_bitmasks,
                    all_det_ids=all_det_ids,
                    pools=pools,
                    anchors=anchors,
                )

            timer.run(
                "build_candidate_pair_matrix",
                self.build_candidate_pair_matrix,
                pools=pools,
                vocab_size=vocab_size,
            )

            ordered_cids = timer.run("ordered_pool_classes", self.ordered_pool_classes, pools)
            beam = timer.run(
                "run_beam_search",
                self.run_beam_search,
                ordered_cids=ordered_cids,
                class_items=class_items,
                pools=pools,
                pools_meta=pools_meta,
                anchors=anchors,
                total_dets=total_dets,
                vocab_size=vocab_size,
            )
            hypotheses = timer.run(
                "collect_hypotheses",
                self.collect_hypotheses,
                beam=beam,
                total_dets=total_dets,
                all_det_ids=all_det_ids,
                anchors=anchors,
                vocab_size=vocab_size,
            )
            return timer.run(
                "build_result",
                self.build_result,
                hypotheses=hypotheses,
                anchors=anchors,
                total_dets=total_dets,
                dets_by_class=dets_by_class,
                timestamp=timestamp,
            )
        finally:
            self.last_timings_seconds = timer.snapshot_seconds()
            self._frame_timer = None

    def init_frame_caches(self) -> None:
        self.graph_utils.init_frame_caches()
        self._frame_class_opts_debug = {}
        self._frame_object_index_by_id = {}
        self._frame_det_index_by_id = {}
        self._frame_object_ids_by_index = ()
        self._frame_det_ids_by_index = ()
        self._frame_object_ids_from_mask_cache = {0: ()}
        self._frame_det_ids_from_mask_cache = {0: ()}

    def empty_result(
        self,
        *,
        total_dets: int,
        anchors: list[int],
        timestamp: float | None,
        n_classes: int = 0,
    ) -> dict:
        core = self.empty_core(anchors=anchors)
        meta = {
            "score_key": "score_sets",
            "n_dets": int(total_dets),
            "n_hypotheses": 0,
        }
        if int(n_classes) > 0:
            meta["n_classes"] = int(n_classes)
            meta["anchors"] = list(anchors)
        return {
            "core": core,
            "debug": {
                "meta": meta,
                "set_hypotheses": [],
                "class_options_debug": {},
                "timestamp": float(timestamp) if timestamp is not None else None,
            },
        }

    def all_detection_ids(self, dets: list) -> list[int]:
        return [
            int(did)
            for did in (int(getattr(det, "detection_id", -1)) for det in (dets or []))
            if int(did) >= 0
        ]

    def ordered_pool_classes(self, pools: dict[int, list[int]]) -> list[int]:
        ordered_cids = sorted((int(cid) for cid in pools.keys()), key=lambda cid: (len(pools.get(int(cid), []) or []), int(cid)))
        if self.max_set_size > 0 and len(ordered_cids) > self.max_set_size:
            return ordered_cids[: int(self.max_set_size)]
        return ordered_cids

    def initial_beam_state(self) -> list[dict]:
        return self.search_engine.initial_beam_state()

    def run_beam_search(
        self,
        *,
        ordered_cids: list[int],
        class_items: dict[int, list],
        pools: dict[int, list[int]],
        pools_meta: dict[int, dict],
        anchors: list[int],
        total_dets: int,
        vocab_size: int | None,
    ) -> list[dict]:
        return self.search_engine.run_beam_search(
            ordered_cids=ordered_cids,
            class_items=class_items,
            pools=pools,
            pools_meta=pools_meta,
            anchors=anchors,
            total_dets=total_dets,
            vocab_size=vocab_size,
        )

    def class_detection_ids(self, class_items: dict[int, list], class_id: int) -> list[int]:
        return self.search_engine.class_detection_ids(class_items, class_id)

    def expand_beam_for_class(
        self,
        *,
        beam: list[dict],
        class_id: int,
        det_ids: list[int],
        pool: list[int],
        pool_n: int,
        anchors: list[int],
        total_dets: int,
        vocab_size: int | None,
    ) -> list[dict]:
        return self.search_engine.expand_beam_for_class(
            beam=beam,
            class_id=class_id,
            det_ids=det_ids,
            pool=pool,
            pool_n=pool_n,
            anchors=anchors,
            total_dets=total_dets,
            vocab_size=vocab_size,
        )

    def transition_state(self, *, st: dict, class_id: int, opt: dict, pool_n: int) -> dict | None:
        return self.search_engine.transition_state(st=st, class_id=class_id, opt=opt, pool_n=pool_n)

    def class_selection_pack(self, *, opt: dict, k_c: int, pool_n: int) -> dict:
        return self.search_engine.class_selection_pack(opt=opt, k_c=k_c, pool_n=pool_n)

    def score_beam_states(
        self,
        *,
        states: list[dict],
        total_dets: int,
        anchors: list[int],
        vocab_size: int | None,
    ) -> list[tuple[float, dict]]:
        return self.search_engine.score_beam_states(
            states=states,
            total_dets=total_dets,
            anchors=anchors,
            vocab_size=vocab_size,
        )

    def collect_hypotheses(
        self,
        *,
        beam: list[dict],
        total_dets: int,
        all_det_ids: list[int],
        anchors: list[int],
        vocab_size: int | None,
    ) -> list[dict]:
        return self.search_engine.collect_hypotheses(
            beam=beam,
            total_dets=total_dets,
            all_det_ids=all_det_ids,
            anchors=anchors,
            vocab_size=vocab_size,
        )

    def build_result(
        self,
        *,
        hypotheses: list[dict],
        anchors: list[int],
        total_dets: int,
        dets_by_class: dict[int, list],
        timestamp: float | None,
    ) -> dict:
        result = self.summary_builder.build_result(
            hypotheses=hypotheses,
            anchors=anchors,
            total_dets=total_dets,
            dets_by_class=dets_by_class,
            timestamp=timestamp,
        )
        dbg = result.get("debug", None)
        if isinstance(dbg, dict):
            dbg["class_options_debug"] = dict(self._frame_class_opts_debug or {})
        return result

    def score_state_quick(
        self,
        *,
        st: dict,
        total_dets: int,
        anchors: list[int],
        vocab_size: int | None = None,
    ) -> float:
        return self.scoring.score_state_quick(
            st=st,
            total_dets=total_dets,
            anchors=anchors,
            vocab_size=vocab_size,
        )

    def select_beam_diverse_by_class_k(
        self,
        *,
        scored: list[tuple[float, dict]],
        class_id: int,
        n_class_dets: int,
    ) -> list[dict]:
        return self.search_engine.select_beam_diverse_by_class_k(
            scored=scored,
            class_id=class_id,
            n_class_dets=n_class_dets,
        )

    def empty_core(self, anchors: list[int]) -> dict:
        return {
            "enabled": False,
            "anchors": list(anchors or []),
            "n_hypotheses": 0,
            "best_score": 0.0,
            "second_score": 0.0,
            "gap_best": 0.0,
            "mean_maturity_best": 0.0,
            "thr_shortlist": 0.0,
            "shortlist": [],
            "prior_by_oid": {},
            "class_prior_by_cid": {},
            "selective_classes": [],
        }

    def group_dets_by_class(self, dets: list) -> dict[int, list]:
        dets_by_class: dict[int, list] = {}
        for det in dets:
            cid = getattr(det, "class_id", None)
            did = getattr(det, "detection_id", None)
            if cid is None or did is None:
                continue
            dets_by_class.setdefault(int(cid), []).append(det)
        return dets_by_class

    def select_top_dets_per_class(self, dets_by_class: dict[int, list]) -> dict[int, list]:
        return self.options_builder.select_top_dets_per_class(dets_by_class)

    def build_pools_for_classes(
        self,
        class_items: dict[int, list],
        anchors: list[int],
        vocab_size: int | None = None,
    ) -> tuple[dict[int, list[int]], dict[int, dict]]:
        return self.options_builder.build_pools_for_classes(class_items, anchors, vocab_size=vocab_size)

    def build_pool_for_class(self, class_id: int, anchors: list[int], vocab_size: int | None = None) -> list[int]:
        return self.options_builder.build_pool_for_class(class_id, anchors, vocab_size=vocab_size)

    def build_candidate_pair_matrix(
        self,
        *,
        pools: dict[int, list[int]],
        vocab_size: int | None = None,
    ) -> dict:
        if not self.candidate_pair_matrix_enabled:
            return {"enabled": False, "executed": False, "n_objects": 0, "n_pairs": 0}

        candidate_ids = self.graph_utils.candidate_object_ids_from_pools(pools)
        n_objects = int(len(candidate_ids))
        if n_objects <= 0:
            return {"enabled": True, "executed": False, "n_objects": 0, "n_pairs": 0}

        if self.candidate_pair_matrix_max_objects > 0 and n_objects > self.candidate_pair_matrix_max_objects:
            return {
                "enabled": True,
                "executed": False,
                "n_objects": int(n_objects),
                "n_pairs": 0,
                "skip_reason": "max_objects",
            }

        out = self.graph_utils.build_candidate_pair_matrices(
            candidate_ids,
            vocab_size=vocab_size,
        )
        return {
            "enabled": True,
            "executed": bool(out.get("executed", False)),
            "n_objects": int(out.get("n_objects", n_objects) or 0),
            "n_pairs": int(out.get("n_pairs", 0) or 0),
        }

    def setup_frame_bitmasks(
        self,
        *,
        all_det_ids: list[int],
        pools: dict[int, list[int]],
        anchors: list[int] | None = None,
    ) -> None:
        det_ids = tuple(sorted({int(x) for x in (all_det_ids or []) if x is not None and int(x) >= 0}))
        obj_ids = set(self.graph_utils.candidate_object_ids_from_pools(pools))
        obj_ids.update(int(x) for x in (anchors or []) if x is not None and int(x) >= 0)
        obj_ids_sorted = tuple(sorted(obj_ids))

        self._frame_det_ids_by_index = det_ids
        self._frame_det_index_by_id = {int(det_id): int(idx) for idx, det_id in enumerate(det_ids)}
        self._frame_object_ids_by_index = obj_ids_sorted
        self._frame_object_index_by_id = {int(obj_id): int(idx) for idx, obj_id in enumerate(obj_ids_sorted)}
        self._frame_det_ids_from_mask_cache = {0: ()}
        self._frame_object_ids_from_mask_cache = {0: ()}

    def object_mask(self, object_ids) -> int:
        if not self.use_beam_bitmask_mode():
            return 0
        index_by_id = getattr(self, "_frame_object_index_by_id", None)
        if not isinstance(index_by_id, dict) or not index_by_id:
            return 0
        mask = 0
        for raw_id in (object_ids or ()):
            idx = index_by_id.get(int(raw_id), None)
            if idx is not None:
                mask |= 1 << int(idx)
        return int(mask)

    def det_mask(self, det_ids) -> int:
        if not self.use_beam_bitmask_mode():
            return 0
        index_by_id = getattr(self, "_frame_det_index_by_id", None)
        if not isinstance(index_by_id, dict) or not index_by_id:
            return 0
        mask = 0
        for raw_id in (det_ids or ()):
            idx = index_by_id.get(int(raw_id), None)
            if idx is not None:
                mask |= 1 << int(idx)
        return int(mask)

    def _ids_from_mask(
        self,
        *,
        mask: int,
        ids_by_index: tuple[int, ...],
        cache_name: str,
    ) -> tuple[int, ...]:
        mask_int = int(mask or 0)
        if mask_int == 0:
            return ()

        cache = getattr(self, cache_name, None)
        if isinstance(cache, dict):
            cached = cache.get(mask_int, None)
            if cached is not None:
                return cached

        out = []
        bit_idx = 0
        cur = int(mask_int)
        while cur:
            if (cur & 1) and bit_idx < len(ids_by_index):
                out.append(int(ids_by_index[bit_idx]))
            cur >>= 1
            bit_idx += 1
        out_tuple = tuple(out)
        if isinstance(cache, dict):
            cache[mask_int] = out_tuple
        return out_tuple

    def object_ids_from_mask(self, mask: int) -> tuple[int, ...]:
        return self._ids_from_mask(
            mask=int(mask or 0),
            ids_by_index=tuple(getattr(self, "_frame_object_ids_by_index", ()) or ()),
            cache_name="_frame_object_ids_from_mask_cache",
        )

    def det_ids_from_mask(self, mask: int) -> tuple[int, ...]:
        return self._ids_from_mask(
            mask=int(mask or 0),
            ids_by_index=tuple(getattr(self, "_frame_det_ids_by_index", ()) or ()),
            cache_name="_frame_det_ids_from_mask_cache",
        )

    def use_beam_bitmask_mode(self) -> bool:
        return bool(self.beam_state_mode in ("bitmask_full", "bitmask_used"))

    def use_beam_bitmask_full_mode(self) -> bool:
        return bool(self.beam_state_mode == "bitmask_full")

    def build_kernel(self, anchors: list[int], selected_obj_ids: list[int]) -> list[int]:
        return self.options_builder.build_kernel(anchors, selected_obj_ids)

    def build_class_options(
        self,
        class_id: int,
        det_ids: list[int],
        pool_obj_ids: list[int],
        used_obj_ids: set[int],
        kernel_obj_ids: list[int],
        vocab_size: int | None = None,
    ) -> list[dict]:
        return self.options_builder.build_class_options(
            class_id,
            det_ids,
            pool_obj_ids,
            used_obj_ids,
            kernel_obj_ids,
            vocab_size=vocab_size,
        )

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
        return self.options_builder.build_class_options_cached(
            class_id,
            det_ids,
            pool_obj_ids,
            used_obj_ids,
            kernel_obj_ids,
            vocab_size=vocab_size,
            used_obj_mask=used_obj_mask,
        )

    def attach_class_exclusivity(self, opts: list[dict]) -> list[dict]:
        return self.options_builder.attach_class_exclusivity(opts)

    def object_support_to_kernel(self, object_id: int, kernel_obj_ids: list[int], vocab_size: int | None = None) -> float:
        return self.options_builder.object_support_to_kernel(object_id, kernel_obj_ids, vocab_size=vocab_size)

    def score_state_as_hypothesis(
        self,
        st: dict,
        total_dets: int,
        all_det_ids: list[int],
        anchors: list[int],
        vocab_size: int | None = None,
    ) -> dict:
        return self.scoring.score_state_as_hypothesis(
            st=st,
            total_dets=total_dets,
            all_det_ids=all_det_ids,
            anchors=anchors,
            vocab_size=vocab_size,
        )

    def context_coverage_effective(self, object_ids: list[int], vocab_size: int | None = None) -> float:
        return self.scoring.context_coverage_effective(object_ids, vocab_size=vocab_size)

    def expected_neighbors_topk(self, object_id: int, topk: int, vocab_size: int | None = None) -> list[int]:
        return self.graph_utils.expected_neighbors_topk(object_id, topk, vocab_size=vocab_size)

    def mean_maturity(self, object_ids: list[int]) -> float:
        return self.graph_utils.mean_maturity(object_ids)

    def coverage_effective(self, coverage_raw: float, explained_n: int, total_dets: int) -> float:
        return self.scoring.coverage_effective(coverage_raw, explained_n, total_dets)

    def size_utility(self, k: int) -> float:
        return self.scoring.size_utility(k)

    def density_score(self, object_ids: list[int], vocab_size: int | None = None) -> tuple[float, bool, float, float, int]:
        return self.graph_utils.density_score(object_ids, vocab_size=vocab_size)

    def density_score_cached(
        self,
        object_ids_sorted: list[int],
        vocab_size: int | None = None,
    ) -> tuple[float, bool, float, float, int]:
        return self.graph_utils.density_score_cached(object_ids_sorted, vocab_size=vocab_size)

    def density_score_cached_by_mask(
        self,
        object_mask: int,
        vocab_size: int | None = None,
    ) -> tuple[float, bool, float, float, int]:
        return self.graph_utils.density_score_cached_by_mask(object_mask, vocab_size=vocab_size)

    def maturity_pack_cached(self, object_ids_sorted: list[int]) -> tuple[float, float, float]:
        return self.graph_utils.maturity_pack_cached(object_ids_sorted)

    def maturity_pack_cached_by_mask(self, object_mask: int) -> tuple[float, float, float]:
        return self.graph_utils.maturity_pack_cached_by_mask(object_mask)

    def p_conditional_cached(self, g, src_id: int, dst_id: int, vocab_size: int | None = None) -> float:
        return self.graph_utils.p_conditional_cached(g, src_id, dst_id, vocab_size=vocab_size)

    def cooc_count_cached(self, g, src_id: int, dst_id: int) -> int:
        return self.graph_utils.cooc_count_cached(g, src_id, dst_id)

    def max_spanning_tree_mean(self, ids: list[int], edges: list[tuple[float, int, int]]) -> float:
        return self.graph_utils.max_spanning_tree_mean(ids, edges)

    def maturity_coherence(self, object_ids: list[int]) -> float:
        return self.graph_utils.maturity_coherence(object_ids)
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
        return self.scoring.combine_score(
            k=k,
            coverage_eff=coverage_eff,
            size_util=size_util,
            density=density,
            density_valid=density_valid,
            class_info=class_info,
            class_support=class_support,
            class_support_valid=class_support_valid,
            class_stability=class_stability,
            class_stability_valid=class_stability_valid,
            exclusivity=exclusivity,
            exclusivity_valid=exclusivity_valid,
        )

    def scaled_weight(self, base_weight: float, k: int, kind: str) -> float:
        return self.scoring.scaled_weight(base_weight, k, kind)

    def k_weight_multiplier(self, k: int, ref: float, gamma: float, min_mult: float, max_mult: float) -> float:
        return self.scoring.k_weight_multiplier(k, ref, gamma, min_mult, max_mult)

    def log_n_choose_k(self, n: int, k: int) -> float:
        return self.scoring.log_n_choose_k(n, k)

    def class_info_from_logC(self, logC: float) -> float:
        return self.scoring.class_info_from_logC(logC)

    def class_stability_from_pool(
        self,
        pool_obj_ids: set[int],
        kernel_obj_ids: list[int],
        vocab_size: int | None = None,
    ) -> float:
        return self.graph_utils.class_stability_from_pool(pool_obj_ids, kernel_obj_ids, vocab_size=vocab_size)

    def build_unexplained_det_ids(self, all_det_ids: list[int], explained_det_ids: list[int]) -> list[int]:
        return self.summary_builder.build_unexplained_det_ids(all_det_ids, explained_det_ids)

    def build_object_support(self, hypotheses: list[dict]) -> tuple[dict[int, float], dict[int, float]]:
        return self.summary_builder.build_object_support(hypotheses)

    def best_second_summary(self, hypotheses: list[dict]) -> dict:
        return self.summary_builder.best_second_summary(hypotheses)

    def build_shortlist(self, hyps: list[dict]) -> tuple[set[int], float]:
        return self.summary_builder.build_shortlist(hyps)

    def class_priors_from_shortlist(self, shortlist: set[int], prior_by_oid: dict[int, float]) -> dict[int, float]:
        return self.summary_builder.class_priors_from_shortlist(shortlist, prior_by_oid)

    def selective_classes_from_priors(self, class_prior_by_cid: dict[int, float]) -> set[int]:
        return self.summary_builder.selective_classes_from_priors(class_prior_by_cid)

    def object_maturity_score(self, object_id: int) -> float:
        return self.graph_utils.object_maturity_score(object_id)
