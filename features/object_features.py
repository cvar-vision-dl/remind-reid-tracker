from __future__ import annotations

import numpy as np

from utils.math import l2_normalize_rows


class ObjectFeatureExtractor:
    """
    Extract object representations:
      - descriptor global
      - descriptor global trimmed
      - descriptores por patch (N,D)
    """

    def __init__(self, config: dict):
        self.config = config or {}

        g = self.config.get("global", {}) or {}
        self.enable_global = bool(g.get("enabled", False))
        self.global_weighted = bool(g.get("weighted", True))

        gt = self.config.get("global_trimmed", {}) or {}
        self.enable_global_trimmed = bool(gt.get("enabled", False))
        self.trimmed_weighted = bool(gt.get("weighted", True))
        self.trimmed_keep_frac = float(gt.get("keep_frac", 0.5))
        self.trimmed_min_patches = int(gt.get("min_patches", 8))

        p = self.config.get("patch", {}) or {}
        self.enable_patch_descs = bool(p.get("enabled", False))
        self.patch_max_patches = int(p.get("max_patches", 0))

        self.patch_return_coverage = bool(p.get("return_coverage", True))
        self.patch_l2_normalize = bool(p.get("l2_normalize", True))

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def extract(
        self,
        dino,
        fmap: np.ndarray,
        obj_mask_px: np.ndarray,
        patch_cache: dict | None = None,
        frame_cache: dict | None = None,
    ) -> dict:
        out = {
            "desc_global": None,
            "desc_global_trimmed": None,
            "patch_descs": None,
            "patch_cov": None,
            "effective_patches": None,
        }

        if fmap is None or obj_mask_px is None:
            return out

        hp, wp, dim = fmap.shape
        cov = patch_cache.get("cov") if isinstance(patch_cache, dict) else None
        if cov is None:
            cov = dino.mask_px_to_patch_coverage(obj_mask_px.astype(np.uint8, copy=False), hp, wp)
        out["effective_patches"] = float(np.sum(cov, dtype=np.float32))
        patch_mask = patch_cache.get("patch_mask") if isinstance(patch_cache, dict) else None
        if patch_mask is None:
            patch_mask = dino.patch_mask_from_coverage(cov)
        if not np.any(patch_mask):
            return out

        flat_feats = None
        if isinstance(frame_cache, dict):
            flat_feats = frame_cache.get("flat_feats")
        if flat_feats is None:
            flat_feats = fmap.reshape(-1, dim).astype(np.float32, copy=False)
            if isinstance(frame_cache, dict):
                frame_cache["flat_feats"] = flat_feats

        # 1) Global (pool normal)
        if self.enable_global:
            out["desc_global"] = dino.pool_descriptor_from_patch_mask(
                fmap,
                patch_mask,
                patch_weights=cov if self.global_weighted else None,
                weighted=self.global_weighted,
                use_trimmed_mean=False,
                trimmed_keep_frac=None,
                trimmed_min_patches=None,
            )

        # 2) Global trimmed
        if self.enable_global_trimmed:
            out["desc_global_trimmed"] = dino.pool_descriptor_from_patch_mask(
                fmap,
                patch_mask,
                patch_weights=cov if self.trimmed_weighted else None,
                weighted=self.trimmed_weighted,
                use_trimmed_mean=True,
                trimmed_keep_frac=self.trimmed_keep_frac,
                trimmed_min_patches=self.trimmed_min_patches,
            )

        # 3) Patch descriptors (all object patches, individually)
        if self.enable_patch_descs:
            patch_descs, patch_cov = self.extract_patch_descriptors(
                dino,
                fmap,
                obj_mask_px,
                max_patches=self.patch_max_patches,
                return_coverage=self.patch_return_coverage,
                l2_normalize=self.patch_l2_normalize,
                cov=cov,
                patch_mask=patch_mask,
                flat_feats=flat_feats,
                flat_feats_n=frame_cache.get("flat_feats_n") if isinstance(frame_cache, dict) else None,
            )
            out["patch_descs"] = patch_descs
            out["patch_cov"] = patch_cov

        return out

    # ------------------------------------------------------------
    # Patch descriptor extraction
    # ------------------------------------------------------------

    def extract_patch_descriptors(
        self,
        dino,
        fmap: np.ndarray,
        mask_px: np.ndarray,
        max_patches: int = 0,
        return_coverage: bool = True,
        l2_normalize: bool = True,
        cov: np.ndarray | None = None,
        patch_mask: np.ndarray | None = None,
        flat_feats: np.ndarray | None = None,
        flat_feats_n: np.ndarray | None = None,
    ):
        """
        Returns:
          - patch_descs: (N,D) object patch embeddings
          - patch_cov: (N,) coverage per patch (or None if return_coverage=False)

        N = number of patches that pass the policy (any/threshold).
        """
        if fmap is None or fmap.ndim != 3:
            return None, None

        hp, wp, dim = fmap.shape

        if cov is None:
            cov = dino.mask_px_to_patch_coverage(mask_px.astype(np.uint8, copy=False), hp, wp)
        if patch_mask is None:
            patch_mask = dino.patch_mask_from_coverage(cov)

        if not np.any(patch_mask):
            return None, None

        feats = flat_feats
        if feats is None:
            feats = fmap.reshape(-1, dim).astype(np.float32, copy=False)
        m = patch_mask.reshape(-1).astype(bool)

        already_normalized = False
        if l2_normalize and flat_feats_n is not None:
            sel = flat_feats_n[m]
            already_normalized = True
        else:
            sel = feats[m]
        if sel.shape[0] == 0:
            return None, None

        cov_sel = None
        if return_coverage:
            cov_sel = cov.reshape(-1)[m].astype(np.float32)

        max_patches = int(max(0, max_patches))
        if max_patches > 0 and sel.shape[0] > max_patches:
            idx = np.random.choice(sel.shape[0], size=max_patches, replace=False)
            sel = sel[idx]
            if cov_sel is not None:
                cov_sel = cov_sel[idx]

        if l2_normalize and not already_normalized:
            sel = l2_normalize_rows(sel)

        return sel, cov_sel
