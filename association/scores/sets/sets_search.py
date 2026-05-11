from __future__ import annotations

from time import perf_counter


class SetsSearchEngine:
    def __init__(self, *, score) -> None:
        self.score = score

    def frame_timer(self):
        timer = getattr(self.score, "_frame_timer", None)
        return timer

    def _normalize_sorted_unique_ids(self, ids) -> tuple[int, ...]:
        vals = tuple(int(x) for x in (ids or ()))
        if len(vals) <= 1:
            return vals

        prev = vals[0]
        monotonic_unique = True
        for cur in vals[1:]:
            if cur <= prev:
                monotonic_unique = False
                break
            prev = cur
        if monotonic_unique:
            return vals
        return tuple(sorted(set(vals)))

    def initial_beam_state(self) -> list[dict]:
        return [{
            "used_obj": set(),
            "used_obj_mask": 0,
            "explained_det_sorted": (),
            "explained_det_mask": 0,
            "obj_ids_sorted": (),
            "obj_mask": 0,
            "parent_state": None,
            "last_pair": None,
            "last_class_id": None,
            "last_class_k": 0,
            "explained_n": 0,
            "class_info": 1.0,
            "class_support_sum": 0.0,
            "class_support_wsum": 0.0,
            "class_stability_sum": 0.0,
            "class_stability_count": 0,
            "class_logC_sum": 0.0,
            "excl_sum": 0.0,
            "excl_wsum": 0.0,
            "excl_any_valid": False,
        }]

    def state_object_ids_sorted(self, st: dict) -> tuple[int, ...]:
        obj_mask = int(st.get("obj_mask", 0) or 0) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0
        if obj_mask:
            return self.score.object_ids_from_mask(obj_mask)
        return tuple(int(x) for x in (st.get("obj_ids_sorted", ()) or ()))

    def state_explained_det_sorted(self, st: dict) -> tuple[int, ...]:
        det_mask = int(st.get("explained_det_mask", 0) or 0) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0
        if det_mask:
            return self.score.det_ids_from_mask(det_mask)
        return tuple(int(x) for x in (st.get("explained_det_sorted", ()) or ()))

    def materialize_pairs(self, st: dict | None) -> list[dict]:
        chain = []
        cur = st if isinstance(st, dict) else None
        while isinstance(cur, dict):
            pair = cur.get("last_pair", None)
            if isinstance(pair, dict):
                chain.append(pair)
            cur = cur.get("parent_state", None)
        if not chain:
            return []
        chain.reverse()
        return [dict(item) for item in chain]

    def sorted_union_tuple(self, base_ids, new_ids) -> tuple[int, ...]:
        base = tuple(int(x) for x in (base_ids or ()))
        extra = self._normalize_sorted_unique_ids(new_ids)
        if not base:
            return extra
        if not extra:
            return base

        out: list[int] = []
        i = 0
        j = 0
        base_n = len(base)
        extra_n = len(extra)
        last = None
        have_last = False

        while i < base_n and j < extra_n:
            bi = int(base[i])
            ej = int(extra[j])
            if bi <= ej:
                cur = bi
                i += 1
                if bi == ej:
                    j += 1
            else:
                cur = ej
                j += 1

            if not have_last or int(cur) != int(last):
                out.append(int(cur))
                last = int(cur)
                have_last = True

        while i < base_n:
            cur = int(base[i])
            i += 1
            if not have_last or int(cur) != int(last):
                out.append(int(cur))
                last = int(cur)
                have_last = True

        while j < extra_n:
            cur = int(extra[j])
            j += 1
            if not have_last or int(cur) != int(last):
                out.append(int(cur))
                last = int(cur)
                have_last = True

        return tuple(out)

    def state_score_cache_key(self, *, st: dict, total_dets: int, vocab_size: int | None) -> tuple:
        vs = int(vocab_size) if vocab_size is not None else -1
        use_masks = bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)())
        obj_mask = int(st.get("obj_mask", 0) or 0) if use_masks else 0
        return (
            int(obj_mask) if use_masks and int(obj_mask) else self.state_object_ids_sorted(st),
            int(st.get("explained_n", len(st.get("explained_det_sorted", ()) or ()))),
            float(st.get("class_info", 1.0)),
            float(st.get("class_support_sum", 0.0)),
            float(st.get("class_support_wsum", 0.0)),
            float(st.get("class_stability_sum", 0.0)),
            int(st.get("class_stability_count", 0)),
            float(st.get("excl_sum", 0.0)),
            float(st.get("excl_wsum", 0.0)),
            bool(st.get("excl_any_valid", False)),
            int(total_dets),
            int(vs),
        )

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
        beam = self.initial_beam_state()
        timer = self.frame_timer()
        for cid in ordered_cids:
            det_ids = self.class_detection_ids(class_items, int(cid))
            if not det_ids:
                continue

            if timer is None:
                next_beam = self.expand_beam_for_class(
                    beam=beam,
                    class_id=int(cid),
                    det_ids=det_ids,
                    pool=(pools.get(int(cid), []) or []),
                    pool_n=int(pools_meta.get(int(cid), {}).get("pool_n", len(pools.get(int(cid), []) or []))),
                    anchors=anchors,
                    total_dets=total_dets,
                    vocab_size=vocab_size,
                )
            else:
                next_beam = timer.run(
                    "run_beam_search/expand_beam_for_class",
                    self.expand_beam_for_class,
                    beam=beam,
                    class_id=int(cid),
                    det_ids=det_ids,
                    pool=(pools.get(int(cid), []) or []),
                    pool_n=int(pools_meta.get(int(cid), {}).get("pool_n", len(pools.get(int(cid), []) or []))),
                    anchors=anchors,
                    total_dets=total_dets,
                    vocab_size=vocab_size,
                )
            if not next_beam:
                if self.score.allow_partial_coverage:
                    continue
                break
            beam = next_beam
        return beam

    def class_detection_ids(self, class_items: dict[int, list], class_id: int) -> list[int]:
        return [
            int(did)
            for did in (int(getattr(det, "detection_id", -1)) for det in (class_items.get(int(class_id), []) or []))
            if int(did) >= 0
        ]

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
        next_beam: list[dict] = []
        timer = self.frame_timer()
        if timer is None:
            for st in beam:
                kernel = self.score.build_kernel(anchors=anchors, selected_obj_ids=list(self.state_object_ids_sorted(st)))
                opts = self.score.build_class_options_cached(
                    class_id=int(class_id),
                    det_ids=det_ids,
                    pool_obj_ids=pool,
                    used_obj_ids=st["used_obj"],
                    used_obj_mask=int(st.get("used_obj_mask", 0) or 0) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0,
                    kernel_obj_ids=kernel,
                    vocab_size=vocab_size,
                )
                for opt in opts:
                    next_state = self.transition_state(
                        st=st,
                        class_id=int(class_id),
                        opt=opt,
                        pool_n=int(pool_n),
                    )
                    if next_state is not None:
                        next_beam.append(next_state)
        else:
            with timer.measure("run_beam_search/expand_beam_for_class/class_options"):
                state_opts: list[tuple[dict, list[dict]]] = []
                for st in beam:
                    kernel = self.score.build_kernel(anchors=anchors, selected_obj_ids=list(self.state_object_ids_sorted(st)))
                    opts = self.score.build_class_options_cached(
                        class_id=int(class_id),
                        det_ids=det_ids,
                        pool_obj_ids=pool,
                        used_obj_ids=st["used_obj"],
                        used_obj_mask=int(st.get("used_obj_mask", 0) or 0) if bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)()) else 0,
                        kernel_obj_ids=kernel,
                        vocab_size=vocab_size,
                    )
                    state_opts.append((st, opts))
            with timer.measure("run_beam_search/expand_beam_for_class/transition"):
                for st, opts in state_opts:
                    for opt in opts:
                        next_state = self.transition_state(
                            st=st,
                            class_id=int(class_id),
                            opt=opt,
                            pool_n=int(pool_n),
                        )
                        if next_state is not None:
                            next_beam.append(next_state)

        if not next_beam:
            return []

        if timer is None:
            scored = self.score_beam_states(
                states=next_beam,
                total_dets=total_dets,
                anchors=anchors,
                vocab_size=vocab_size,
            )
            return self.select_beam_diverse_by_class_k(
                scored=scored,
                class_id=int(class_id),
                n_class_dets=int(len(det_ids)),
            )

        scored = timer.run(
            "run_beam_search/expand_beam_for_class/score_beam_states",
            self.score_beam_states,
            states=next_beam,
            total_dets=total_dets,
            anchors=anchors,
            vocab_size=vocab_size,
        )
        return timer.run(
            "run_beam_search/expand_beam_for_class/select_beam_diverse_by_class_k",
            self.select_beam_diverse_by_class_k,
            scored=scored,
            class_id=int(class_id),
            n_class_dets=int(len(det_ids)),
        )

    def transition_state(self, *, st: dict, class_id: int, opt: dict, pool_n: int) -> dict | None:
        oids_raw = opt.get("object_ids", ()) or ()
        dids_raw = opt.get("det_ids", ()) or ()
        oids = oids_raw if isinstance(oids_raw, tuple) else tuple(int(x) for x in oids_raw)
        dids = dids_raw if isinstance(dids_raw, tuple) else tuple(int(x) for x in dids_raw)
        use_overlap_masks = bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)())
        used_obj_mask = int(st.get("used_obj_mask", 0) or 0) if use_overlap_masks else 0
        opt_obj_mask = int(opt.get("object_mask", 0) or 0) if use_overlap_masks else 0
        opt_det_mask = int(opt.get("det_mask", 0) or 0) if use_overlap_masks else 0

        if use_overlap_masks and opt_obj_mask:
            if used_obj_mask & opt_obj_mask:
                return None
            new_used = st["used_obj"]
            new_used_mask = int(used_obj_mask | opt_obj_mask)
            new_det_mask = int(st.get("explained_det_mask", 0) or 0) | int(opt_det_mask)
            new_obj_mask = int(st.get("obj_mask", 0) or 0) | int(opt_obj_mask)
            # In mask modes, avoid rebuilding sorted tuples at each transition;
            # materialize them from obj_mask/det_mask only when needed.
            new_det_sorted = ()
            new_obj_sorted = ()
        else:
            if any(int(oid) in st["used_obj"] for oid in oids):
                return None
            new_used = set(st["used_obj"])
            new_used.update(int(oid) for oid in oids)
            new_used_mask = int(used_obj_mask)
            new_det_mask = int(st.get("explained_det_mask", 0) or 0) | int(opt_det_mask) if use_overlap_masks else 0
            new_obj_mask = int(st.get("obj_mask", 0) or 0) | int(opt_obj_mask) if use_overlap_masks else 0
            new_det_sorted = self.sorted_union_tuple(st.get("explained_det_sorted", ()), dids)
            new_obj_sorted = self.sorted_union_tuple(st.get("obj_ids_sorted", ()), oids)

        new_pair = opt.get("_pair", None)
        if not isinstance(new_pair, dict):
            new_pair = {
                "class_id": int(class_id),
                "det_ids": tuple(int(x) for x in dids),
                "object_ids": tuple(int(x) for x in oids),
            }

        pack_fast = opt.get("_transition_pack", None)
        if isinstance(pack_fast, tuple) and len(pack_fast) == 7:
            kc = int(pack_fast[0])
            support_w = float(pack_fast[1])
            support_val = float(pack_fast[2])
            stab_val = float(pack_fast[3])
            excl_w = float(pack_fast[4])
            excl_val = float(pack_fast[5])
            excl_v = bool(pack_fast[6])
            class_info_mul = float(opt.get("class_info", 1.0))
            class_logC = float(opt.get("class_logC", 0.0))
        else:
            pack = self.class_selection_pack(
                opt=opt,
                k_c=int(len(oids)),
                pool_n=int(pool_n),
            )
            target_k = int(pack.get("target_k", pack.get("k_c", 0)))
            support_w = float(max(1, target_k))
            support_val = float(max(0.0, min(1.0, pack.get("support", 0.0))))
            stab_val = float(pack.get("stability", 0.0))
            kc = int(pack.get("k_c", 0))
            excl_w = float(kc) if kc > 0 else 0.0
            excl_val = float(max(0.0, min(1.0, pack.get("excl", 0.0)))) if kc > 0 else 0.0
            excl_v = bool(pack.get("excl_v", False))
            class_info_mul = float(max(0.0, min(1.0, pack.get("info", 1.0))))
            class_logC = float(pack.get("logC", 0.0))

        excl_any_valid = bool(st.get("excl_any_valid", False) or (kc > 0 and excl_v))

        return {
            "used_obj": new_used,
            "used_obj_mask": int(new_used_mask),
            "explained_det_sorted": new_det_sorted,
            "explained_det_mask": int(new_det_mask),
            "obj_ids_sorted": new_obj_sorted,
            "obj_mask": int(new_obj_mask),
            "parent_state": st,
            "last_pair": new_pair,
            "last_class_id": int(class_id),
            "last_class_k": int(kc),
            "explained_n": int(st.get("explained_n", 0)) + int(len(dids)),
            "class_info": float(st.get("class_info", 1.0)) * float(max(0.0, min(1.0, class_info_mul))),
            "class_support_sum": float(st.get("class_support_sum", 0.0)) + float(support_w) * float(support_val),
            "class_support_wsum": float(st.get("class_support_wsum", 0.0)) + float(support_w),
            "class_stability_sum": float(st.get("class_stability_sum", 0.0)) + float(stab_val),
            "class_stability_count": int(st.get("class_stability_count", 0)) + 1,
            "class_logC_sum": float(st.get("class_logC_sum", 0.0)) + float(class_logC),
            "excl_sum": float(st.get("excl_sum", 0.0)) + float(excl_w) * float(excl_val),
            "excl_wsum": float(st.get("excl_wsum", 0.0)) + float(excl_w),
            "excl_any_valid": bool(excl_any_valid),
        }

    def class_selection_pack(self, *, opt: dict, k_c: int, pool_n: int) -> dict:
        return {
            "k_c": int(k_c),
            "pool_n": int(pool_n),
            "target_k": int(opt.get("target_k", 0)),
            "support": float(opt.get("class_support", 0.0)),
            "stability": float(opt.get("class_stability", 0.0)),
            "info": float(opt.get("class_info", 0.0)),
            "logC": float(opt.get("class_logC", 0.0)),
            "excl": float(opt.get("class_exclusivity", 0.0)),
            "excl_v": bool(opt.get("class_exclusivity_valid", False)),
        }

    def score_beam_states(
        self,
        *,
        states: list[dict],
        total_dets: int,
        anchors: list[int],
        vocab_size: int | None,
    ) -> list[tuple[float, dict]]:
        scored = []
        timer = self.frame_timer()
        frame_score_cache = getattr(self.score, "_frame_score_state_quick_cache", None)
        if not isinstance(frame_score_cache, dict):
            frame_score_cache = {}
            self.score._frame_score_state_quick_cache = frame_score_cache
        for st in states:
            cache_key = self.state_score_cache_key(
                st=st,
                total_dets=total_dets,
                vocab_size=vocab_size,
            )
            cached_score = frame_score_cache.get(cache_key, None)
            if cached_score is None:
                if timer is None:
                    score = float(
                        self.score.score_state_quick(
                            st=st,
                            total_dets=total_dets,
                            anchors=anchors,
                            vocab_size=vocab_size,
                        )
                    )
                else:
                    t0 = perf_counter()
                    score = float(
                        self.score.score_state_quick(
                            st=st,
                            total_dets=total_dets,
                            anchors=anchors,
                            vocab_size=vocab_size,
                        )
                    )
                    timer.add(
                        "run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick",
                        perf_counter() - t0,
                    )
                frame_score_cache[cache_key] = float(score)
            else:
                score = float(cached_score)
            scored.append((score, st))
        scored.sort(key=lambda x: float(x[0]), reverse=True)
        return scored

    def select_beam_diverse_by_class_k(
        self,
        *,
        scored: list[tuple[float, dict]],
        class_id: int,
        n_class_dets: int,
    ) -> list[dict]:
        if not scored:
            return []

        limit = max(1, int(self.score.beam_width))
        if int(n_class_dets) <= 1:
            return [st for _, st in scored[:limit]]

        selected: list[dict] = []
        seen_state_ids: set[int] = set()
        best_by_k: dict[int, dict] = {}

        for _, st in scored:
            if int(st.get("last_class_id", -1)) != int(class_id):
                continue
            kc = int(st.get("last_class_k", 0))
            if kc in best_by_k:
                continue
            best_by_k[int(kc)] = st

        for kc in sorted(best_by_k.keys(), reverse=True):
            st = best_by_k[int(kc)]
            sid = id(st)
            if sid in seen_state_ids:
                continue
            selected.append(st)
            seen_state_ids.add(sid)
            if len(selected) >= limit:
                return selected[:limit]

        for _, st in scored:
            sid = id(st)
            if sid in seen_state_ids:
                continue
            selected.append(st)
            seen_state_ids.add(sid)
            if len(selected) >= limit:
                break

        return selected

    def collect_hypotheses(
        self,
        *,
        beam: list[dict],
        total_dets: int,
        all_det_ids: list[int],
        anchors: list[int],
        vocab_size: int | None,
    ) -> list[dict]:
        hypotheses = []
        seen = set()

        for st in beam:
            use_masks = bool(getattr(self.score, "use_beam_bitmask_mode", lambda: False)())
            obj_mask = int(st.get("obj_mask", 0) or 0) if use_masks else 0
            det_mask = int(st.get("explained_det_mask", 0) or 0) if use_masks else 0
            obj_ids = list(self.state_object_ids_sorted(st))
            explained = list(self.state_explained_det_sorted(st))
            key = (int(obj_mask), int(det_mask)) if use_masks and (obj_mask or det_mask) else (tuple(obj_ids), tuple(explained))
            if key in seen:
                continue
            seen.add(key)

            if not obj_ids and not self.score.allow_partial_coverage:
                continue

            st_for_hyp = dict(st)
            st_for_hyp["obj_ids_sorted"] = tuple(obj_ids)
            st_for_hyp["explained_det_sorted"] = tuple(explained)
            st_for_hyp["pairs"] = self.materialize_pairs(st)
            hyp = self.score.score_state_as_hypothesis(
                st=st_for_hyp,
                total_dets=total_dets,
                all_det_ids=all_det_ids,
                anchors=anchors,
                vocab_size=vocab_size,
            )
            if int(total_dets) > 1 and int(hyp.get("k", 0)) < 2:
                continue
            if float(hyp.get("score_sets", 0.0)) < float(self.score.min_set_score):
                continue
            hypotheses.append(hyp)

        hypotheses.sort(key=lambda h: float(h.get("score_sets", 0.0)), reverse=True)
        return hypotheses[: self.score.topk_sets]
