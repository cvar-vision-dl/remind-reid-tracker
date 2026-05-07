from __future__ import annotations

import math

import numpy as np


class SetsGraphUtils:
    def __init__(self, *, score) -> None:
        self.score = score

    def init_frame_caches(self) -> None:
        self.score._frame_pcond_cache = {}
        self.score._frame_cooc_cache = {}
        self.score._frame_graph_cache = {}
        self.score._frame_pair_density_cache = {}
        self.score._frame_obj_maturity_cache = {}
        self.score._frame_density_cache = {}
        self.score._frame_density_cache_by_mask = {}
        self.score._frame_maturity_pack_cache = {}
        self.score._frame_maturity_pack_cache_by_mask = {}
        self.score._frame_score_state_quick_cache = {}
        self.score._frame_class_opts_cache = {}
        self.score._frame_filtered_class_opts_cache = {}
        self.score._frame_candidate_pair_ids = ()
        self.score._frame_candidate_pair_index = {}
        self.score._frame_candidate_pair_weights = None
        self.score._frame_candidate_pair_edges = None
        self.score._frame_candidate_pair_vocab_size = -1
        self.score._frame_candidate_bit_to_matrix_idx = None
        self.score._frame_candidate_idx_from_mask_cache = {0: ()}
        self.score._frame_object_maturity_by_bit_index = None

    def object_graph_cached(self, object_id: int):
        oid = int(object_id)
        cached = self.score._frame_graph_cache.get(oid, None)
        if cached is not None:
            return cached

        obj = self.score.memory_store.get(int(oid))
        g = getattr(obj, "neighbors", None) if obj is not None else None
        out = g if (g is not None and getattr(g, "enabled", False)) else None
        self.score._frame_graph_cache[oid] = out
        return out

    def candidate_object_ids_from_pools(self, pools: dict[int, list[int]]) -> list[int]:
        ids: set[int] = set()
        for obj_ids in (pools or {}).values():
            for oid in obj_ids or []:
                ids.add(int(oid))
        if not ids:
            return []
        return sorted(int(x) for x in ids)

    def build_candidate_pair_matrices(
        self,
        object_ids: list[int],
        *,
        vocab_size: int | None = None,
    ) -> dict:
        ids = [int(x) for x in (object_ids or [])]
        n = int(len(ids))
        vs = int(vocab_size) if vocab_size is not None else -1
        if n <= 0:
            self.score._frame_candidate_pair_ids = ()
            self.score._frame_candidate_pair_index = {}
            self.score._frame_candidate_pair_weights = None
            self.score._frame_candidate_pair_edges = None
            self.score._frame_candidate_pair_vocab_size = int(vs)
            self.score._frame_candidate_bit_to_matrix_idx = None
            self.score._frame_candidate_idx_from_mask_cache = {0: ()}
            return {"n_objects": 0, "n_pairs": 0, "executed": False}

        weights = np.zeros((n, n), dtype=np.float32)
        edges = np.zeros((n, n), dtype=np.uint8)

        pair_count = 0
        for i in range(n):
            oi = int(ids[i])
            for j in range(i + 1, n):
                oj = int(ids[j])
                val, edge_present = self.pair_density_metrics_cached(
                    int(oi),
                    int(oj),
                    vocab_size=vocab_size,
                )
                w = float(val)
                e = 1 if bool(edge_present) else 0
                weights[i, j] = w
                weights[j, i] = w
                edges[i, j] = e
                edges[j, i] = e
                pair_count += 1

        self.score._frame_candidate_pair_ids = tuple(ids)
        self.score._frame_candidate_pair_index = {int(oid): int(i) for i, oid in enumerate(ids)}
        self.score._frame_candidate_pair_weights = weights
        self.score._frame_candidate_pair_edges = edges
        self.score._frame_candidate_pair_vocab_size = int(vs)
        bit_to_matrix = None
        object_ids_by_index = tuple(getattr(self.score, "_frame_object_ids_by_index", ()) or ())
        object_index_by_id = getattr(self.score, "_frame_object_index_by_id", None)
        if object_ids_by_index and isinstance(object_index_by_id, dict):
            bit_to_matrix = np.full((len(object_ids_by_index),), -1, dtype=np.int32)
            for oid, matrix_idx in self.score._frame_candidate_pair_index.items():
                bit_idx = object_index_by_id.get(int(oid), None)
                if bit_idx is None:
                    continue
                b = int(bit_idx)
                if b < 0 or b >= int(bit_to_matrix.size):
                    continue
                bit_to_matrix[b] = int(matrix_idx)
        self.score._frame_candidate_bit_to_matrix_idx = bit_to_matrix
        self.score._frame_candidate_idx_from_mask_cache = {0: ()}

        return {
            "n_objects": int(n),
            "n_pairs": int(pair_count),
            "executed": bool(n > 0),
        }

    def _density_score_from_candidate_indices(
        self,
        idx_tuple: tuple[int, ...],
        *,
        vocab_size: int | None = None,
    ) -> tuple[float, bool, float, float, int] | None:
        weights = getattr(self.score, "_frame_candidate_pair_weights", None)
        edges = getattr(self.score, "_frame_candidate_pair_edges", None)
        vs = int(vocab_size) if vocab_size is not None else -1
        cached_vs = int(getattr(self.score, "_frame_candidate_pair_vocab_size", -2))
        if weights is None or edges is None or cached_vs != int(vs):
            return None

        n = int(len(idx_tuple))
        if n <= 1:
            return 0.0, False, 0.0, 0.0, 0
        if n <= 10:
            return self._density_score_from_candidate_indices_small(idx_tuple, weights, edges)
        idx = np.asarray(idx_tuple, dtype=np.int32)

        subw = weights[np.ix_(idx, idx)]
        dens = float(self.max_spanning_tree_mean_dense_np(subw))

        edge_cov = 0.0
        node_cov = 0.0
        min_deg = 0
        if float(self.score.min_edge_p) > 0.0:
            sube = edges[np.ix_(idx, idx)]
            deg = np.sum(sube, axis=1, dtype=np.int32)
            m = int(n * (n - 1) // 2)
            if m > 0:
                edges_present = int(np.sum(np.triu(sube, k=1), dtype=np.int32))
                edge_cov = float(edges_present) / float(m)
                edge_cov = float(max(0.0, min(1.0, edge_cov)))

            if deg.size > 0:
                min_deg = int(np.min(deg))
                node_cov = float(np.count_nonzero(deg > 0)) / float(max(1, n))
                node_cov = float(max(0.0, min(1.0, node_cov)))

        if float(self.score.min_edge_p) > 0.0 and self.score.density_edge_cov_gamma > 0.0:
            dens = float(dens * (edge_cov ** float(self.score.density_edge_cov_gamma)))

        return float(max(0.0, min(1.0, dens))), True, float(edge_cov), float(node_cov), int(min_deg)

    def _density_score_from_candidate_indices_small(
        self,
        idx_tuple: tuple[int, ...],
        weights: np.ndarray,
        edges: np.ndarray,
    ) -> tuple[float, bool, float, float, int]:
        n = int(len(idx_tuple))
        if n <= 1:
            return 0.0, False, 0.0, 0.0, 0
        if n == 2:
            w = float(weights[int(idx_tuple[0]), int(idx_tuple[1])])
            dens = float(max(0.0, min(1.0, w)))
            edge_cov = 0.0
            node_cov = 0.0
            min_deg = 0
            if float(self.score.min_edge_p) > 0.0:
                e = 1 if int(edges[int(idx_tuple[0]), int(idx_tuple[1])]) != 0 else 0
                edge_cov = float(e)
                node_cov = float(e)
                min_deg = int(1 if e > 0 else 0)
                if self.score.density_edge_cov_gamma > 0.0:
                    dens = float(dens * (edge_cov ** float(self.score.density_edge_cov_gamma)))
            return float(max(0.0, min(1.0, dens))), True, float(edge_cov), float(node_cov), int(min_deg)

        selected = [False] * n
        best = [0.0] * n
        selected[0] = True
        row0 = weights[int(idx_tuple[0])]
        for j in range(1, n):
            best[j] = float(row0[int(idx_tuple[j])])

        total = 0.0
        chosen = 0
        for _ in range(n - 1):
            next_idx = -1
            next_w = -1.0
            for j in range(n):
                if selected[j]:
                    continue
                w = float(best[j])
                if w > next_w:
                    next_w = float(w)
                    next_idx = int(j)
            if next_idx < 0:
                break
            selected[next_idx] = True
            total += float(max(0.0, next_w))
            chosen += 1
            row = weights[int(idx_tuple[next_idx])]
            for j in range(n):
                if selected[j]:
                    continue
                w = float(row[int(idx_tuple[j])])
                if w > float(best[j]):
                    best[j] = float(w)

        dens = float(total / float(chosen)) if chosen > 0 else 0.0

        edge_cov = 0.0
        node_cov = 0.0
        min_deg = 0
        if float(self.score.min_edge_p) > 0.0:
            deg = [0] * n
            edges_present = 0
            m = int(n * (n - 1) // 2)
            for i in range(n):
                row_e = edges[int(idx_tuple[i])]
                for j in range(i + 1, n):
                    if int(row_e[int(idx_tuple[j])]) != 0:
                        edges_present += 1
                        deg[i] += 1
                        deg[j] += 1
            if m > 0:
                edge_cov = float(edges_present) / float(m)
                edge_cov = float(max(0.0, min(1.0, edge_cov)))
            if deg:
                min_deg = int(min(deg))
                node_cov = float(sum(1 for d in deg if int(d) > 0)) / float(max(1, n))
                node_cov = float(max(0.0, min(1.0, node_cov)))

        if float(self.score.min_edge_p) > 0.0 and self.score.density_edge_cov_gamma > 0.0:
            dens = float(dens * (edge_cov ** float(self.score.density_edge_cov_gamma)))

        return float(max(0.0, min(1.0, dens))), True, float(edge_cov), float(node_cov), int(min_deg)

    def density_score_from_candidate_matrix(
        self,
        object_ids: list[int],
        *,
        vocab_size: int | None = None,
    ) -> tuple[float, bool, float, float, int] | None:
        weights = getattr(self.score, "_frame_candidate_pair_weights", None)
        edges = getattr(self.score, "_frame_candidate_pair_edges", None)
        index = getattr(self.score, "_frame_candidate_pair_index", None)
        vs = int(vocab_size) if vocab_size is not None else -1
        cached_vs = int(getattr(self.score, "_frame_candidate_pair_vocab_size", -2))

        if weights is None or edges is None or not isinstance(index, dict) or cached_vs != int(vs):
            return None

        ids = [int(x) for x in (object_ids or [])]
        n = int(len(ids))
        if n <= 1:
            return 0.0, False, 0.0, 0.0, 0

        try:
            idx_tuple = tuple(int(index[int(oid)]) for oid in ids)
        except KeyError:
            return None

        return self._density_score_from_candidate_indices(idx_tuple, vocab_size=vocab_size)

    def _candidate_indices_from_object_mask(self, object_mask: int) -> tuple[int, ...] | None:
        mask = int(object_mask or 0)
        if mask <= 0:
            return ()

        cache = getattr(self.score, "_frame_candidate_idx_from_mask_cache", None)
        if isinstance(cache, dict):
            cached = cache.get(mask, None)
            if cached is not None:
                return tuple(int(x) for x in cached)

        bit_to_matrix = getattr(self.score, "_frame_candidate_bit_to_matrix_idx", None)
        if bit_to_matrix is None:
            return None

        idx_out: list[int] = []
        bit_idx = 0
        cur = int(mask)
        max_bits = int(bit_to_matrix.size)
        while cur:
            if (cur & 1) != 0:
                if bit_idx < 0 or bit_idx >= max_bits:
                    return None
                matrix_idx = int(bit_to_matrix[bit_idx])
                if matrix_idx < 0:
                    return None
                idx_out.append(int(matrix_idx))
            cur >>= 1
            bit_idx += 1
        out = tuple(idx_out)
        if isinstance(cache, dict):
            cache[mask] = out
        return out

    def pair_density_metrics_cached(
        self,
        object_id_a: int,
        object_id_b: int,
        *,
        vocab_size: int | None = None,
    ) -> tuple[float, bool]:
        oa = int(object_id_a)
        ob = int(object_id_b)
        if oa == ob:
            return 0.0, False

        if oa < ob:
            key_pair = (oa, ob)
        else:
            key_pair = (ob, oa)

        vs = int(vocab_size) if vocab_size is not None else -1
        key = (key_pair, int(vs))
        cached = self.score._frame_pair_density_cache.get(key, None)
        if cached is not None:
            return float(cached[0]), bool(cached[1])

        gi = self.object_graph_cached(int(oa))
        gj = self.object_graph_cached(int(ob))

        val = 0.0
        if gi is not None and gj is not None:
            pij = float(self.p_conditional_cached(gi, int(oa), int(ob), vocab_size=vocab_size))
            pji = float(self.p_conditional_cached(gj, int(ob), int(oa), vocab_size=vocab_size))
            if self.score.use_mutual:
                val = math.sqrt(max(0.0, pij) * max(0.0, pji))
            else:
                val = 0.5 * (max(0.0, pij) + max(0.0, pji))

        val = float(max(0.0, min(1.0, float(val))))
        if self.score.density_gate_min_edge_p and val < float(self.score.min_edge_p):
            val = 0.0

        cij = int(self.cooc_count_cached(gi, int(oa), int(ob))) if gi is not None else 0
        cji = int(self.cooc_count_cached(gj, int(ob), int(oa))) if gj is not None else 0
        observed = bool(cij > 0 or cji > 0)
        edge_present = bool(
            observed
            and float(val) >= float(self.score.min_edge_p)
            and float(self.score.min_edge_p) > 0.0
        )

        out = (float(val), bool(edge_present))
        self.score._frame_pair_density_cache[key] = out
        return out

    def expected_neighbors_topk(self, object_id: int, topk: int, vocab_size: int | None = None) -> list[int]:
        g = self.object_graph_cached(int(object_id))
        if g is None:
            return []

        out = []
        for pack in (g.neighbors() or []):
            nid = int(pack.get("dst_id", -1))
            if nid < 0:
                continue
            cooc = int(pack.get("cooc_count", 0))
            if cooc <= 0:
                continue
            p = float(self.p_conditional_cached(g, int(object_id), int(nid), vocab_size=vocab_size))
            if p < float(self.score.context_min_p):
                continue
            out.append(int(nid))
            if len(out) >= int(topk):
                break
        return out

    def mean_maturity(self, object_ids: list[int]) -> float:
        ids = [int(x) for x in (object_ids or [])]
        if not ids:
            return 0.0
        vals = [float(self.object_maturity_score(int(oid))) for oid in ids]
        return float(sum(vals) / float(len(vals))) if vals else 0.0

    def density_score(self, object_ids: list[int], vocab_size: int | None = None) -> tuple[float, bool, float, float, int]:
        ids = [int(x) for x in (object_ids or [])]
        n = len(ids)
        if n <= 1:
            return 0.0, False, 0.0, 0.0, 0

        matrix_out = self.density_score_from_candidate_matrix(ids, vocab_size=vocab_size)
        if matrix_out is not None:
            return matrix_out

        m = 0
        edges_present = 0
        deg = [0] * n
        weights = [[0.0] * n for _ in range(n)]

        for i in range(n):
            oi = int(ids[i])
            for j in range(i + 1, n):
                oj = int(ids[j])
                val, edge_present = self.pair_density_metrics_cached(
                    int(oi),
                    int(oj),
                    vocab_size=vocab_size,
                )
                weights[i][j] = float(val)
                weights[j][i] = float(val)

                if edge_present:
                    edges_present += 1
                    deg[i] += 1
                    deg[j] += 1

                m += 1

        if m <= 0:
            return 0.0, False, 0.0, 0.0, 0

        dens = float(self.max_spanning_tree_mean_dense(weights))

        edge_cov = 0.0
        node_cov = 0.0
        min_deg = 0
        if float(self.score.min_edge_p) > 0.0:
            edge_cov = float(edges_present) / float(m)
            edge_cov = float(max(0.0, min(1.0, edge_cov)))

            deg_vals = [int(x) for x in deg]
            min_deg = int(min(deg_vals)) if deg_vals else 0
            node_cov = float(sum(1 for d in deg_vals if int(d) > 0)) / float(max(1, len(deg_vals)))
            node_cov = float(max(0.0, min(1.0, node_cov)))

        if self.score.density_edge_cov_gamma > 0.0 and m > 0 and float(self.score.min_edge_p) > 0.0:
            dens = float(dens * (edge_cov ** float(self.score.density_edge_cov_gamma)))

        return float(max(0.0, min(1.0, dens))), True, float(edge_cov), float(node_cov), int(min_deg)

    def density_score_cached(self, object_ids_sorted: list[int], vocab_size: int | None = None) -> tuple[float, bool, float, float, int]:
        ids = tuple(int(x) for x in (object_ids_sorted or []))
        vs = int(vocab_size) if vocab_size is not None else -1
        key = (ids, int(vs))
        cached = self.score._frame_density_cache.get(key, None)
        if cached is not None:
            dens, valid, edge_cov, node_cov, min_deg = cached
            return float(dens), bool(valid), float(edge_cov), float(node_cov), int(min_deg)
        out = self.density_score(list(ids), vocab_size=vocab_size)
        self.score._frame_density_cache[key] = (float(out[0]), bool(out[1]), float(out[2]), float(out[3]), int(out[4]))
        return out

    def density_score_cached_by_mask(self, object_mask: int, vocab_size: int | None = None) -> tuple[float, bool, float, float, int]:
        mask = int(object_mask or 0)
        if mask <= 0:
            return 0.0, False, 0.0, 0.0, 0
        vs = int(vocab_size) if vocab_size is not None else -1
        key = (int(mask), int(vs))
        cached = self.score._frame_density_cache_by_mask.get(key, None)
        if cached is not None:
            dens, valid, edge_cov, node_cov, min_deg = cached
            return float(dens), bool(valid), float(edge_cov), float(node_cov), int(min_deg)
        out = None
        idx_tuple = self._candidate_indices_from_object_mask(mask)
        if idx_tuple is not None:
            out = self._density_score_from_candidate_indices(idx_tuple, vocab_size=vocab_size)
        if out is None:
            ids = list(self.score.object_ids_from_mask(int(mask)))
            out = self.density_score(ids, vocab_size=vocab_size)
        self.score._frame_density_cache_by_mask[key] = (float(out[0]), bool(out[1]), float(out[2]), float(out[3]), int(out[4]))
        return out

    def maturity_pack_cached(self, object_ids_sorted: list[int]) -> tuple[float, float, float]:
        ids = tuple(int(x) for x in (object_ids_sorted or []))
        cached = self.score._frame_maturity_pack_cache.get(ids, None)
        if cached is not None:
            return float(cached[0]), float(cached[1]), float(cached[2])

        mean_maturity = self.mean_maturity(list(ids))
        maturity_coh = float(self.maturity_coherence(list(ids)))
        maturity_rel = 1.0
        if self.score.maturity_enabled and self.score.maturity_gamma > 0.0:
            maturity_rel = float(max(0.0, min(1.0, maturity_coh))) ** float(self.score.maturity_gamma)

        pack = (float(mean_maturity), float(maturity_coh), float(maturity_rel))
        self.score._frame_maturity_pack_cache[ids] = pack
        return pack

    def maturity_pack_cached_by_mask(self, object_mask: int) -> tuple[float, float, float]:
        mask = int(object_mask or 0)
        if mask <= 0:
            return 0.0, 0.0, 1.0
        cached = self.score._frame_maturity_pack_cache_by_mask.get(int(mask), None)
        if cached is not None:
            return float(cached[0]), float(cached[1]), float(cached[2])
        ms = self._maturity_values_from_object_mask(int(mask))
        if ms is None:
            ids = self.score.object_ids_from_mask(int(mask))
            pack = self.maturity_pack_cached(list(ids))
            self.score._frame_maturity_pack_cache_by_mask[int(mask)] = (float(pack[0]), float(pack[1]), float(pack[2]))
            return float(pack[0]), float(pack[1]), float(pack[2])

        n = int(len(ms))
        mean_maturity = float(sum(ms) / float(n)) if n > 0 else 0.0
        maturity_coh = float(self._maturity_coherence_from_values(ms))
        maturity_rel = 1.0
        if self.score.maturity_enabled and self.score.maturity_gamma > 0.0:
            maturity_rel = float(max(0.0, min(1.0, maturity_coh))) ** float(self.score.maturity_gamma)

        pack = (float(mean_maturity), float(maturity_coh), float(maturity_rel))
        self.score._frame_maturity_pack_cache_by_mask[int(mask)] = pack
        return pack

    def _maturity_values_from_object_mask(self, object_mask: int) -> list[float] | None:
        mask = int(object_mask or 0)
        if mask <= 0:
            return []
        maturity_by_bit = getattr(self.score, "_frame_object_maturity_by_bit_index", None)
        if maturity_by_bit is None:
            obj_ids = tuple(getattr(self.score, "_frame_object_ids_by_index", ()) or ())
            if not obj_ids:
                return None
            maturity_by_bit = [float(self.object_maturity_score(int(oid))) for oid in obj_ids]
            self.score._frame_object_maturity_by_bit_index = maturity_by_bit

        out: list[float] = []
        bit_idx = 0
        cur = int(mask)
        max_bits = int(len(maturity_by_bit))
        while cur:
            if (cur & 1) != 0:
                if bit_idx < 0 or bit_idx >= max_bits:
                    return None
                out.append(float(maturity_by_bit[bit_idx]))
            cur >>= 1
            bit_idx += 1
        return out

    def _maturity_coherence_from_values(self, maturities: list[float]) -> float:
        ms = [float(max(0.0, min(1.0, float(m)))) for m in (maturities or [])]
        if not ms:
            return 0.0
        if len(ms) == 1:
            return float(ms[0])

        p = float(self.score.maturity_softmin_p)
        if abs(p) <= 1e-12:
            prod = 1.0
            for m in ms:
                prod *= float(max(1e-12, m))
            return float(max(0.0, min(1.0, prod ** (1.0 / float(len(ms))))))

        s = 0.0
        for m in ms:
            s += float(max(1e-12, m)) ** float(p)
        mean = float(s) / float(len(ms))
        out = float(mean) ** (1.0 / float(p))
        return float(max(0.0, min(1.0, out)))

    def p_conditional_cached(self, g, src_id: int, dst_id: int, vocab_size: int | None = None) -> float:
        if g is None or not getattr(g, "enabled", False):
            return 0.0
        vs = int(vocab_size) if vocab_size is not None else -1
        key = (int(src_id), int(dst_id), int(vs))
        cached = self.score._frame_pcond_cache.get(key, None)
        if cached is not None:
            return float(cached)
        try:
            p = float(g.p_conditional(int(dst_id), vocab_size=vocab_size))
        except Exception:
            p = 0.0
        p = float(max(0.0, min(1.0, p)))
        self.score._frame_pcond_cache[key] = float(p)
        return float(p)

    def cooc_count_cached(self, g, src_id: int, dst_id: int) -> int:
        if g is None or not getattr(g, "enabled", False):
            return 0
        key = (int(src_id), int(dst_id))
        cached = self.score._frame_cooc_cache.get(key, None)
        if cached is not None:
            return int(cached)
        try:
            c = int(g.cooc_count(int(dst_id)))
        except Exception:
            c = 0
        self.score._frame_cooc_cache[key] = int(c)
        return int(c)

    def max_spanning_tree_mean_dense(self, weights: list[list[float]]) -> float:
        n = int(len(weights or []))
        if n <= 1:
            return 0.0
        if n == 2:
            return float(max(0.0, min(1.0, float(weights[0][1]))))

        selected = [False] * n
        best = [0.0] * n
        selected[0] = True
        for j in range(1, n):
            best[j] = float(weights[0][j])

        total = 0.0
        chosen = 0
        for _ in range(n - 1):
            next_idx = -1
            next_w = -1.0
            for j in range(n):
                if selected[j]:
                    continue
                w = float(best[j])
                if w > next_w:
                    next_w = float(w)
                    next_idx = int(j)

            if next_idx < 0:
                break

            selected[next_idx] = True
            total += float(max(0.0, next_w))
            chosen += 1

            row = weights[next_idx]
            for j in range(n):
                if selected[j]:
                    continue
                w = float(row[j])
                if w > float(best[j]):
                    best[j] = float(w)

        if chosen <= 0:
            return 0.0
        return float(total / float(chosen))

    def max_spanning_tree_mean_dense_np(self, weights: np.ndarray) -> float:
        w = np.asarray(weights, dtype=np.float32)
        if w.ndim != 2:
            return 0.0
        n = int(w.shape[0])
        if n <= 1:
            return 0.0
        if n == 2:
            return float(max(0.0, min(1.0, float(w[0, 1]))))

        selected = np.zeros(n, dtype=bool)
        selected[0] = True
        best = np.asarray(w[0], dtype=np.float32).copy()
        best[0] = 0.0

        total = 0.0
        chosen = 0
        for _ in range(n - 1):
            masked = np.where(selected, np.float32(-1.0), best)
            next_idx = int(np.argmax(masked))
            next_w = float(masked[next_idx])
            if next_idx < 0 or next_w < 0.0:
                break

            selected[next_idx] = True
            total += float(max(0.0, next_w))
            chosen += 1
            best = np.maximum(best, w[next_idx])
            best[selected] = 0.0

        if chosen <= 0:
            return 0.0
        return float(total / float(chosen))

    def max_spanning_tree_mean(self, ids: list[int], edges: list[tuple[float, int, int]]) -> float:
        nodes = [int(x) for x in (ids or [])]
        n = len(nodes)
        if n <= 1:
            return 0.0
        if n == 2:
            if not edges:
                return 0.0
            return float(max(0.0, min(1.0, float(edges[0][0]))))

        parent = {int(x): int(x) for x in nodes}
        rank = {int(x): 0 for x in nodes}

        def find(x: int) -> int:
            x = int(x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> bool:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True

        chosen = []
        for w, a, b in sorted(edges or [], key=lambda t: float(t[0]), reverse=True):
            if union(int(a), int(b)):
                chosen.append(float(w))
                if len(chosen) >= (n - 1):
                    break

        if not chosen:
            return 0.0
        return float(sum(chosen) / float(len(chosen)))

    def maturity_coherence(self, object_ids: list[int]) -> float:
        ids = [int(x) for x in (object_ids or [])]
        if not ids:
            return 0.0
        ms = [float(self.object_maturity_score(int(oid))) for oid in ids]
        ms = [float(max(0.0, min(1.0, m))) for m in ms]
        if not ms:
            return 0.0
        if len(ms) == 1:
            return float(ms[0])

        p = float(self.score.maturity_softmin_p)
        if abs(p) <= 1e-12:
            prod = 1.0
            for m in ms:
                prod *= float(max(1e-12, m))
            return float(max(0.0, min(1.0, prod ** (1.0 / float(len(ms))))))

        s = 0.0
        for m in ms:
            s += float(max(1e-12, m)) ** float(p)
        mean = float(s) / float(len(ms))
        out = float(mean) ** (1.0 / float(p))
        return float(max(0.0, min(1.0, out)))

    def class_stability_from_pool(self, pool_obj_ids: set[int], kernel_obj_ids: list[int], vocab_size: int | None = None) -> float:
        if not pool_obj_ids or not kernel_obj_ids:
            return 0.0

        weights = []
        for oid in pool_obj_ids:
            w = float(self.score.object_support_to_kernel(int(oid), kernel_obj_ids, vocab_size=vocab_size))
            if w < self.score.min_edge_p:
                w = 0.0
            weights.append(float(max(0.0, w)))

        z = float(sum(weights))
        if z <= float(self.score.class_stability_eps):
            return 0.0

        ps = [float(w) / z for w in weights if w > 0.0]
        if not ps:
            return 0.0

        H = 0.0
        for p in ps:
            H += -float(p) * float(math.log(max(self.score.class_stability_eps, p)))

        Hmax = float(math.log(max(2.0, float(len(ps)))))
        if Hmax <= float(self.score.class_stability_eps):
            return 0.0

        Hn = float(max(0.0, min(1.0, H / Hmax)))
        return float(max(0.0, min(1.0, 1.0 - Hn)))

    def object_maturity_score(self, object_id: int) -> float:
        cache = getattr(self.score, "_frame_obj_maturity_cache", None)
        if isinstance(cache, dict):
            cached = cache.get(int(object_id), None)
            if cached is not None:
                return float(cached)

        obj = self.score.memory_store.get(int(object_id))
        if obj is None:
            return 0.0

        g = getattr(obj, "neighbors", None)
        if g is None or not getattr(g, "enabled", False):
            return 0.0

        e = int(getattr(g, "episode_count", 0))
        out = float(1.0 - math.exp(-self.score.conf_lambda * float(max(0, e))))
        if isinstance(cache, dict):
            cache[int(object_id)] = float(out)
        return float(out)
