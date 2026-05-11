from __future__ import annotations

import numpy as np
from utils.math import l2_normalize_rows, l2_normalize_vector, kmeans_np
from utils.time import ExecutionTimer


class PartFeatureExtractor:
    """
    Extract a descriptor set (multi-prototype) per object using internal parts.

    Backends (por config):
      - kmeans (propose several prototypes and then do basic intra-frame merge)
      - attention (part proposals from patch->patch attention)

    Output by method:
      {
        "part_descs": [np.ndarray(D,), ...]
        "part_masks": [np.ndarray(Hp,Wp), ...] | None
        "part_stats": [dict, ...]
      }
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self.last_timings_seconds: dict[str, float] = {}

        # ----------------------------
        # KMEANS PARTS
        # ----------------------------
        k_cfg = self.config.get("kmeans", {}) or {}
        self.enable_kmeans = bool(k_cfg.get("enabled", False))

        self.kmeans_k = int(k_cfg.get("k", 4))
        self.kmeans_iters = int(k_cfg.get("iters", 10))
        self.kmeans_n_init = int(k_cfg.get("n_init", 3))
        self.kmeans_seed = int(k_cfg.get("seed", 0))

        self.kmeans_weighted = bool(k_cfg.get("weighted", True))
        self.kmeans_use_trimmed = bool(k_cfg.get("use_trimmed_mean", False))
        self.kmeans_trimmed_keep_frac = float(k_cfg.get("trimmed_keep_frac", 0.5))
        self.kmeans_trimmed_min_patches = int(k_cfg.get("trimmed_min_patches", 8))

        self.kmeans_return_masks = bool(k_cfg.get("return_masks", False))

        self.kmeans_min_cluster_patches = int(k_cfg.get("min_cluster_patches", 6))
        self.kmeans_min_support = float(k_cfg.get("min_support", 0.0))

        m_cfg = k_cfg.get("merge", {}) or {}
        self.kmeans_enable_merge = bool(m_cfg.get("enabled", True))
        self.kmeans_merge_sim_thr = float(m_cfg.get("sim_thr", 0.94))
        self.kmeans_merge_mode = str(m_cfg.get("mode", "keep_best")).lower()

        # ----------------------------
        # ATTENTION PARTS
        # ----------------------------
        a_cfg = self.config.get("attention", {}) or {}
        self.enable_attention = bool(a_cfg.get("enabled", False))
        self.attn_head_ids = a_cfg.get("head_ids", None)

        self.attn_weighted = bool(a_cfg.get("weighted", True))
        self.attn_use_trimmed = bool(a_cfg.get("use_trimmed_mean", False))
        self.attn_trimmed_keep_frac = float(a_cfg.get("trimmed_keep_frac", 0.5))
        self.attn_trimmed_min_patches = int(a_cfg.get("trimmed_min_patches", 8))

        self.attn_return_masks = bool(a_cfg.get("return_masks", False))

        self.attn_max_seeds = int(a_cfg.get("max_seeds", 5))
        self.attn_region_frac = float(a_cfg.get("region_frac", 0.25))
        self.attn_min_region_patches = int(a_cfg.get("min_region_patches", 6))
        self.attn_max_region_frac = float(a_cfg.get("max_region_frac", 0.60))
        self.attn_seed_score = str(a_cfg.get("seed_score", "in_degree")).lower()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        dino,
        fmap,
        obj_mask_px,
        attn=None,
        obj_patch_descs=None,
        patch_cache: dict | None = None,
        frame_cache: dict | None = None,
    ) -> dict:
        """
        {
          "kmeans":    {...} | None
          "attention": {...} | None
        }

        attn:
          - None: attention parts are not computed
          - (N,N): average patch->patch attention
          - (H,N,N): per-head attention (averaged across layers)
        """
        parts_out = {"kmeans": None, "attention": None}
        if fmap is None or obj_mask_px is None:
            self.last_timings_seconds = {}
            return parts_out
        timer = ExecutionTimer()

        if self.enable_kmeans:
            parts_out["kmeans"] = timer.run(
                "parts_kmeans",
                self.extract_kmeans_parts,
                dino=dino,
                fmap=fmap,
                obj_mask_px=obj_mask_px,
                obj_patch_descs=obj_patch_descs,
                patch_cache=patch_cache,
                frame_cache=frame_cache,
            )

        if self.enable_attention and attn is not None:
            parts_out["attention"] = timer.run(
                "parts_attention",
                self.extract_attention_parts,
                dino=dino,
                fmap=fmap,
                attn=attn,
                obj_mask_px=obj_mask_px,
                patch_cache=patch_cache,
                frame_cache=frame_cache,
            )

        self.last_timings_seconds = timer.snapshot_seconds()
        return parts_out

    # ------------------------------------------------------------------
    # KMEANS PARTS
    # ------------------------------------------------------------------

    def pool_selected_patch_features(
        self,
        *,
        dino,
        feats_sel: np.ndarray,
        weights_sel: np.ndarray | None,
        weighted: bool,
        use_trimmed_mean: bool,
        trimmed_keep_frac: float,
        trimmed_min_patches: int,
    ) -> np.ndarray | None:
        if feats_sel is None or feats_sel.ndim != 2 or int(feats_sel.shape[0]) <= 0:
            return None

        sel_feats = feats_sel.astype(np.float32, copy=False)
        sel_w = None
        if bool(weighted) and weights_sel is not None:
            sel_w = weights_sel.astype(np.float32, copy=False)
            if float(np.sum(sel_w, dtype=np.float32)) <= 1e-12:
                return None

        if bool(weighted) and sel_w is not None:
            wsum = float(np.sum(sel_w, dtype=np.float32))
            v0 = (sel_feats * sel_w[:, None]).sum(axis=0) / wsum
        else:
            v0 = sel_feats.mean(axis=0)

        if bool(use_trimmed_mean):
            sel_feats, sel_w = dino.apply_trimmed_selection(
                sel_feats,
                sel_w,
                v0,
                keep_frac=trimmed_keep_frac,
                min_patches=trimmed_min_patches,
            )

        if bool(weighted) and sel_w is not None:
            wsum = float(np.sum(sel_w, dtype=np.float32))
            if wsum <= 1e-12:
                return None
            desc = (sel_feats * sel_w[:, None]).sum(axis=0) / wsum
        else:
            desc = sel_feats.mean(axis=0)

        return l2_normalize_vector(desc)

    def extract_kmeans_parts(
        self,
        dino,
        fmap,
        obj_mask_px,
        obj_patch_descs=None,
        patch_cache: dict | None = None,
        frame_cache: dict | None = None,
    ):
        hp, wp, dim = fmap.shape

        cov = patch_cache.get("cov") if isinstance(patch_cache, dict) else None
        if cov is None:
            cov = dino.mask_px_to_patch_coverage(obj_mask_px.astype(np.uint8, copy=False), hp, wp)
        patch_mask_obj = patch_cache.get("patch_mask") if isinstance(patch_cache, dict) else None
        if patch_mask_obj is None:
            patch_mask_obj = dino.patch_mask_from_coverage(cov)

        if not np.any(patch_mask_obj):
            return {"part_descs": [], "part_masks": [] if self.kmeans_return_masks else None, "part_stats": []}

        feats = frame_cache.get("flat_feats") if isinstance(frame_cache, dict) else None
        if feats is None:
            feats = fmap.reshape(-1, dim).astype(np.float32, copy=False)
            if isinstance(frame_cache, dict):
                frame_cache["flat_feats"] = feats
        mask_flat = patch_mask_obj.reshape(-1)
        x = feats[mask_flat]
        cov_obj = cov.reshape(-1)[mask_flat].astype(np.float32, copy=False)
        n_obj = int(x.shape[0])
        if n_obj <= 0:
            return {"part_descs": [], "part_masks": [] if self.kmeans_return_masks else None, "part_stats": []}

        use_precomputed = (
            obj_patch_descs is not None
            and isinstance(obj_patch_descs, np.ndarray)
            and obj_patch_descs.ndim == 2
            and int(obj_patch_descs.shape[0]) == int(n_obj)
        )
        if use_precomputed:
            x_n = obj_patch_descs.astype(np.float32, copy=False)
        else:
            feats_n = frame_cache.get("flat_feats_n") if isinstance(frame_cache, dict) else None
            if feats_n is None:
                x_n = l2_normalize_rows(x)
            else:
                x_n = feats_n[mask_flat]

        k = int(max(1, self.kmeans_k))
        k = int(min(k, n_obj))

        obj_patch_indices = None
        rr_obj = None
        cc_obj = None
        if self.kmeans_return_masks:
            obj_patch_indices = np.flatnonzero(mask_flat)
            rr_obj = obj_patch_indices // wp
            cc_obj = obj_patch_indices % wp

        if k <= 1:
            desc = self.pool_selected_patch_features(
                dino=dino,
                feats_sel=x,
                weights_sel=cov_obj if self.kmeans_weighted else None,
                weighted=self.kmeans_weighted,
                use_trimmed_mean=self.kmeans_use_trimmed,
                trimmed_keep_frac=self.kmeans_trimmed_keep_frac,
                trimmed_min_patches=self.kmeans_trimmed_min_patches,
            )
            if desc is None:
                return {"part_descs": [], "part_masks": [] if self.kmeans_return_masks else None, "part_stats": []}

            out_masks = [patch_mask_obj.copy()] if self.kmeans_return_masks else None
            out_stats = [{
                "support": float(np.sum(cov_obj, dtype=np.float32)) if self.kmeans_weighted else float(n_obj),
                "n_patches": int(n_obj),
                "coherence": 1.0
            }]
            return {"part_descs": [desc], "part_masks": out_masks, "part_stats": out_stats}

        labels_obj, _centers, _inertia = kmeans_np(
            x_n,
            k=k,
            n_iter=self.kmeans_iters,
            n_init=self.kmeans_n_init,
            seed=self.kmeans_seed,
        )
        labels_obj = labels_obj.astype(np.int32, copy=False)

        part_descs = []
        part_masks = [] if self.kmeans_return_masks else None
        part_stats = []
        counts = np.bincount(labels_obj, minlength=int(k))
        order = np.argsort(labels_obj, kind="stable")
        use_fast_cluster_pool = bool(not self.kmeans_use_trimmed)
        cov_weights = cov_obj if self.kmeans_weighted else None

        start = 0
        for ci in range(int(k)):
            n_ci = int(counts[ci]) if ci < counts.size else 0
            if n_ci <= 0:
                continue

            stop = start + n_ci
            idxs = order[start:stop]
            start = stop

            if n_ci < int(self.kmeans_min_cluster_patches):
                continue

            weights_ci = cov_weights[idxs] if cov_weights is not None else None
            support = float(np.sum(weights_ci, dtype=np.float32)) if weights_ci is not None else float(n_ci)
            if support < float(self.kmeans_min_support):
                continue

            if use_fast_cluster_pool:
                x_ci = x[idxs]
                if weights_ci is not None:
                    wsum = float(np.sum(weights_ci, dtype=np.float32))
                    if wsum <= 1e-12:
                        continue
                    desc = (x_ci * weights_ci[:, None]).sum(axis=0, dtype=np.float32) / wsum
                else:
                    desc = x_ci.mean(axis=0, dtype=np.float32)
                desc = l2_normalize_vector(desc)
            else:
                desc = self.pool_selected_patch_features(
                    dino=dino,
                    feats_sel=x[idxs],
                    weights_sel=weights_ci,
                    weighted=self.kmeans_weighted,
                    use_trimmed_mean=self.kmeans_use_trimmed,
                    trimmed_keep_frac=self.kmeans_trimmed_keep_frac,
                    trimmed_min_patches=self.kmeans_trimmed_min_patches,
                )
                if desc is None:
                    continue

            x_ci_n = x_n[idxs]
            coherence = float(np.mean(x_ci_n @ desc)) if x_ci_n.size else 0.0

            part_descs.append(desc)
            part_stats.append(
                {
                    "support": float(support),
                    "n_patches": int(n_ci),
                    "coherence": float(coherence),
                    "cluster_id": int(ci),
                }
            )
            if self.kmeans_return_masks:
                pm = np.zeros((hp, wp), dtype=bool)
                pm[rr_obj[idxs], cc_obj[idxs]] = True
                part_masks.append(pm)

        if self.kmeans_enable_merge and len(part_descs) >= 2:
            part_descs, part_stats, part_masks = self.merge_basic_greedy(
                part_descs=part_descs,
                part_stats=part_stats,
                part_masks=part_masks,
                sim_thr=self.kmeans_merge_sim_thr,
                mode=self.kmeans_merge_mode,
                return_masks=self.kmeans_return_masks,
            )

        return {
            "part_descs": part_descs,
            "part_masks": part_masks if self.kmeans_return_masks else None,
            "part_stats": part_stats,
        }

    def merge_basic_greedy(
        self,
        part_descs,
        part_stats,
        part_masks,
        sim_thr: float,
        mode: str,
        return_masks: bool,
    ):
        supports = np.asarray([float(s.get("support", 1.0)) for s in part_stats], dtype=np.float32)
        order = np.argsort(-supports)

        kept_descs = []
        kept_stats = []
        kept_masks = [] if return_masks else None

        for idx in order.tolist():
            d = l2_normalize_vector(part_descs[idx])
            s = dict(part_stats[idx])
            sup = float(s.get("support", 1.0))
            m = part_masks[idx] if return_masks else None

            if not kept_descs:
                kept_descs.append(d)
                kept_stats.append(s)
                if return_masks:
                    kept_masks.append(m)
                continue

            sims = np.asarray([float(np.dot(d, kd)) for kd in kept_descs], dtype=np.float32)
            j = int(np.argmax(sims))
            if float(sims[j]) <= float(sim_thr):
                kept_descs.append(d)
                kept_stats.append(s)
                if return_masks:
                    kept_masks.append(m)
                continue

            if str(mode).lower() == "weighted_mean":
                sup_j = float(kept_stats[j].get("support", 1.0))
                wsum = sup_j + sup
                if wsum > 1e-12:
                    new_d = (kept_descs[j] * sup_j + d * sup) / wsum
                    new_d = l2_normalize_vector(new_d)
                    kept_descs[j] = new_d
                    kept_stats[j]["support"] = float(wsum)
                    kept_stats[j]["n_patches"] = int(kept_stats[j].get("n_patches", 0)) + int(s.get("n_patches", 0))
                    if return_masks and kept_masks[j] is not None and m is not None:
                        kept_masks[j] = (kept_masks[j] | m)

        return kept_descs, kept_stats, kept_masks

    # ------------------------------------------------------------------
    # ATTENTION PARTS
    # ------------------------------------------------------------------

    def extract_attention_parts(
        self,
        dino,
        fmap,
        attn,
        obj_mask_px,
        patch_cache: dict | None = None,
        frame_cache: dict | None = None,
    ):
        """
        Attention-based part proposals (intra-frame).

        attn can be:
          - (N,N): average patch->patch attention
          - (H,N,N): per-head attention (averaged across layers)
        """
        hp, wp, dim = fmap.shape
        n_total = hp * wp

        cov = patch_cache.get("cov") if isinstance(patch_cache, dict) else None
        if cov is None:
            cov = dino.mask_px_to_patch_coverage(obj_mask_px.astype(np.uint8, copy=False), hp, wp)
        patch_mask_obj = patch_cache.get("patch_mask") if isinstance(patch_cache, dict) else None
        if patch_mask_obj is None:
            patch_mask_obj = dino.patch_mask_from_coverage(cov)
        mask_flat = patch_mask_obj.reshape(-1)

        if not np.any(mask_flat):
            return {
                "part_descs": [],
                "part_masks": [] if self.attn_return_masks else None,
                "part_stats": [],
            }

        idx_obj = np.flatnonzero(mask_flat)

        attn_mat = self.resolve_attention_matrix(attn, n_total)

        row_sum = attn_mat.sum(axis=1, keepdims=True) + 1e-12
        attn_mat = attn_mat / row_sum

        attn_obj = attn_mat[np.ix_(idx_obj, idx_obj)]

        if self.attn_seed_score == "out_degree":
            scores = attn_obj.sum(axis=1)
        else:
            scores = attn_obj.sum(axis=0)

        max_seeds = int(min(self.attn_max_seeds, scores.size))
        if max_seeds <= 0:
            return {
                "part_descs": [],
                "part_masks": [] if self.attn_return_masks else None,
                "part_stats": [],
            }

        seed_ids = np.argsort(-scores)[:max_seeds]

        feats = frame_cache.get("flat_feats") if isinstance(frame_cache, dict) else None
        if feats is None:
            feats = fmap.reshape(-1, dim).astype(np.float32, copy=False)
            if isinstance(frame_cache, dict):
                frame_cache["flat_feats"] = feats
        x_obj = feats[mask_flat]
        feats_n = frame_cache.get("flat_feats_n") if isinstance(frame_cache, dict) else None
        x_obj_n = feats_n[mask_flat] if feats_n is not None else l2_normalize_rows(x_obj)

        true_pos = np.flatnonzero(mask_flat)

        part_descs = []
        part_stats = []
        part_masks = [] if self.attn_return_masks else None

        k_min = int(self.attn_min_region_patches)
        k_frac = int(np.ceil(float(self.attn_region_frac) * idx_obj.size))
        k_max = int(np.ceil(float(self.attn_max_region_frac) * idx_obj.size))
        if k_max < 1:
            k_max = idx_obj.size

        for sid in seed_ids:
            seed_patch = int(idx_obj[sid])

            a = attn_mat[seed_patch].astype(np.float32, copy=True)
            a[seed_patch] = 0.0

            k = max(k_min, k_frac)
            k = min(k, k_max)
            k = min(k, idx_obj.size)

            if k <= 0:
                continue

            region_idx = idx_obj[np.argpartition(a[idx_obj], -k)[-k:]]

            pm_flat = np.zeros((n_total,), dtype=bool)
            pm_flat[region_idx] = True
            pm = pm_flat.reshape(hp, wp)

            desc = dino.pool_descriptor_from_patch_mask(
                fmap=fmap,
                patch_mask=pm,
                patch_weights=cov if self.attn_weighted else None,
                weighted=self.attn_weighted,
                use_trimmed_mean=self.attn_use_trimmed,
                trimmed_keep_frac=self.attn_trimmed_keep_frac,
                trimmed_min_patches=self.attn_trimmed_min_patches,
            )
            if desc is None:
                continue

            desc = l2_normalize_vector(desc)

            support = float(np.sum(cov[pm])) if self.attn_weighted else float(np.sum(pm_flat))

            loc = np.searchsorted(true_pos, region_idx)
            loc = loc[(loc >= 0) & (loc < true_pos.size)]
            if loc.size:
                coherence = float(np.mean(x_obj_n[loc] @ desc.reshape(-1, 1)))
            else:
                coherence = 0.0

            part_descs.append(desc)
            part_stats.append(
                {
                    "seed_patch": int(seed_patch),
                    "n_patches": int(np.sum(pm_flat)),
                    "support": float(support),
                    "coherence": float(coherence),
                }
            )
            if self.attn_return_masks:
                part_masks.append(pm)

        return {
            "part_descs": part_descs,
            "part_masks": part_masks if self.attn_return_masks else None,
            "part_stats": part_stats,
        }

    def resolve_attention_matrix(self, attn, n_total: int) -> np.ndarray:
        """
        Convert attn to an (N,N) float32 matrix.
        - If attn is (H,N,N) and head_ids exist, average only those heads.
        - If attn is (H,N,N) without head_ids, average all heads.
        - If attn is (N,N), use it as-is.
        """
        a = np.asarray(attn)
        if a.ndim == 2:
            if a.shape != (n_total, n_total):
                raise ValueError(f"attn (N,N) shape mismatch: {a.shape} vs {(n_total, n_total)}")
            return a.astype(np.float32, copy=False)

        if a.ndim == 3:
            h, n1, n2 = a.shape
            if (n1, n2) != (n_total, n_total):
                raise ValueError(f"attn (H,N,N) shape mismatch: {a.shape} vs {(h, n_total, n_total)}")

            head_ids = self.attn_head_ids
            if head_ids is None:
                return a.mean(axis=0).astype(np.float32, copy=False)

            ids = np.asarray(head_ids, dtype=np.int64).reshape(-1)
            ids = ids[(ids >= 0) & (ids < h)]
            if ids.size == 0:
                return a.mean(axis=0).astype(np.float32, copy=False)

            return a[ids].mean(axis=0).astype(np.float32, copy=False)

        raise ValueError(f"attn must be (N,N) or (H,N,N), got {a.shape}")
