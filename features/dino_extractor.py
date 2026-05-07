# features/dino_extractor.py

from __future__ import annotations

import numpy as np
import cv2
import torch

from transformers import AutoImageProcessor, AutoModel


class DinoExtractor:
    """Extractor DINO (contenedor general de métodos)."""

    def __init__(self, config: dict, device: str):
        self.config = config
        self.device = device

        self.processor = None
        self.model = None

        self.patch_size = int(config.get("default_patch_size", 16))
        self.normalize = bool(config.get("normalize_embeddings", True))

        self.patch_selection = str(config.get("patch_selection", "any")).lower()
        self.patch_threshold = float(config.get("patch_threshold", 0.0))
        self.patch_coverage_mode = str(config.get("patch_coverage_mode", "resize_area") or "resize_area").strip().lower()

    def load_model(self):
        model_id = self.resolve_model_id(self.config)

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, low_cpu_mem_usage=True)
        self.model.to(self.device).eval()

        ps = getattr(getattr(self.model, "config", None), "patch_size", self.patch_size)
        self.patch_size = int(ps)

    def resolve_model_id(self, config_dino: dict) -> str:
        label = config_dino.get("model_label", None)
        if label is None:
            raise ValueError("En config['dino'] falta 'model_label' (S/B/L/7B).")

        models_map = config_dino.get("models", None)
        if not isinstance(models_map, dict) or not models_map:
            raise ValueError("En config['dino'] falta 'models' (label -> hf_model_id).")

        key = str(label).strip().upper()
        if key not in models_map:
            raise ValueError(f"DINO model_label='{label}' no está en config['dino']['models'].")

        return str(models_map[key])

    def set_attn_impl_if_possible(self, impl: str):
        try:
            if hasattr(self.model, "set_attn_implementation"):
                self.model.set_attn_implementation(str(impl))
        except Exception:
            return

    @torch.inference_mode()
    def extract_patches(self, image_rgb: np.ndarray) -> np.ndarray:
        """Devuelve fmap [Hp, Wp, D] para una imagen."""
        self.set_attn_impl_if_possible("sdpa")

        batch = self.processor(
            images=image_rgb,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(**batch)
        tokens = outputs.last_hidden_state
        dim = tokens.shape[-1]

        hp = batch["pixel_values"].shape[-2] // self.patch_size
        wp = batch["pixel_values"].shape[-1] // self.patch_size
        n_patches = hp * wp

        n_special = tokens.shape[1] - n_patches
        patch_tokens = tokens[:, n_special:, :]

        fmap = patch_tokens[0].reshape(hp, wp, dim).detach().cpu().float().numpy()
        return fmap

    @torch.inference_mode()
    def extract_patches_and_attn(self, image_rgb: np.ndarray, head_ids=None):
        """
        Devuelve:
        - fmap: (Hp, Wp, D)
        - attn_mean: (N, N) atención patch->patch promedio
        - attn_heads: (Hsel, N, N) solo si head_ids != None, si no None
        """
        self.set_attn_impl_if_possible("eager")

        batch = self.processor(
            images=image_rgb,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        outputs = self.model(
            **batch,
            output_attentions=True,
            return_dict=True,
        )

        tokens = outputs.last_hidden_state
        attentions = outputs.attentions
        dim = tokens.shape[-1]

        if attentions is None:
            raise RuntimeError(
                "DINO no devolvió 'attentions' (outputs.attentions == None). "
                "Necesitas una implementación compatible para capturar attentions."
            )

        hp = batch["pixel_values"].shape[-2] // self.patch_size
        wp = batch["pixel_values"].shape[-1] // self.patch_size
        n_patches = hp * wp
        n_special = tokens.shape[1] - n_patches

        patch_tokens = tokens[:, n_special:, :]
        fmap = patch_tokens[0].reshape(hp, wp, dim).detach().cpu().float().numpy()

        attn_acc = None
        n_layers = 0

        for a in attentions:
            a = a[0]
            a = a[:, n_special:, n_special:]

            if head_ids is not None:
                a = a[head_ids, :, :]

            a = a.float()
            attn_acc = a if attn_acc is None else (attn_acc + a)
            n_layers += 1

        attn_acc = attn_acc / float(max(1, n_layers))

        attn_mean_t = attn_acc.mean(dim=0)

        if head_ids is None:
            attn_heads = None
        else:
            attn_heads = attn_acc.detach().cpu().float().numpy()

        attn_mean = attn_mean_t.detach().cpu().float().numpy()
        return fmap, attn_mean, attn_heads

    def mask_px_to_patch_coverage(self, mask_px: np.ndarray, hp: int, wp: int) -> np.ndarray:
        """Devuelve (Hp, Wp) con fracción de píxeles True por patch."""
        if mask_px.ndim != 2:
            raise ValueError("mask_px debe ser 2D (H, W)")

        p = int(self.patch_size)
        h_eff = hp * p
        w_eff = wp * p

        if h_eff <= 0 or w_eff <= 0:
            return np.zeros((hp, wp), dtype=np.float32)

        mode = self.patch_coverage_mode
        if mode not in ("reshape_mean", "resize_area"):
            mode = "resize_area"

        if mode == "resize_area":
            m = mask_px[:h_eff, :w_eff].astype(np.float32, copy=False)
            cov = cv2.resize(m, (wp, hp), interpolation=cv2.INTER_AREA)
            return cov.astype(np.float32, copy=False)

        m = mask_px[:h_eff, :w_eff].astype(np.uint8, copy=False)
        m4 = m.reshape(hp, p, wp, p)
        return m4.mean(axis=(1, 3)).astype(np.float32)

    def patch_mask_from_coverage(self, cov: np.ndarray) -> np.ndarray:
        """Devuelve patch_mask bool según self.patch_selection."""
        if cov.ndim != 2:
            raise ValueError("cov debe ser 2D (Hp, Wp)")

        if self.patch_selection == "threshold":
            return cov > float(self.patch_threshold)
        return cov > 0.0

    def cosine_sim_to_vector(self, feats: np.ndarray, v: np.ndarray) -> np.ndarray:
        """feats: (N,D), v: (D,) -> sims coseno (N,)."""
        v = v.astype(np.float32, copy=False)
        feats = feats.astype(np.float32, copy=False)

        vnorm = float(np.linalg.norm(v))
        if vnorm <= 1e-12:
            return np.zeros((feats.shape[0],), dtype=np.float32)

        fnorm = np.linalg.norm(feats, axis=1)
        denom = (fnorm * vnorm) + 1e-12
        sim = (feats @ v) / denom
        return sim.astype(np.float32)

    def apply_trimmed_selection(
        self,
        sel_feats: np.ndarray,
        sel_weights: np.ndarray | None,
        v0: np.ndarray,
        keep_frac: float | None,
        min_patches: int | None,
    ):
        if keep_frac is None:
            keep_frac = 1.0
        if min_patches is None:
            min_patches = 8

        n = int(sel_feats.shape[0])
        if n < int(min_patches):
            return sel_feats, sel_weights

        keep = float(keep_frac)
        keep = 1.0 if keep <= 0.0 else keep
        keep = 1.0 if keep > 1.0 else keep
        if keep >= 1.0:
            return sel_feats, sel_weights

        k = int(np.ceil(keep * n))
        k = 1 if k < 1 else k
        if k >= n:
            return sel_feats, sel_weights

        sim = self.cosine_sim_to_vector(sel_feats, v0)
        idx = np.argpartition(sim, -k)[-k:]

        sel_feats2 = sel_feats[idx]
        sel_weights2 = sel_weights[idx] if sel_weights is not None else None
        return sel_feats2, sel_weights2

    def pool_descriptor_from_mask(
        self,
        fmap: np.ndarray,
        mask_px: np.ndarray | None,
        weighted: bool = True,
        use_trimmed_mean: bool = False,
        trimmed_keep_frac: float | None = None,
        trimmed_min_patches: int | None = None,
    ) -> np.ndarray | None:
        if fmap.ndim != 3:
            raise ValueError("fmap debe ser (Hp, Wp, D)")

        hp, wp, dim = fmap.shape
        feats = fmap.reshape(-1, dim).astype(np.float32, copy=False)

        if mask_px is None:
            v = feats.mean(axis=0)
            return self.normalize_if_needed(v.astype(np.float32))

        cov = self.mask_px_to_patch_coverage(mask_px.astype(np.uint8, copy=False), hp, wp)
        cov_flat = cov.reshape(-1)

        if self.patch_selection == "threshold":
            valid = cov_flat > float(self.patch_threshold)
        else:
            valid = cov_flat > 0.0

        if not np.any(valid):
            return None

        sel_feats = feats[valid]

        sel_w = None
        if weighted:
            sel_w = cov_flat[valid].astype(np.float32, copy=False)
            wsum = float(np.sum(sel_w, dtype=np.float32))
            if wsum <= 1e-12:
                return None

        if weighted:
            v0 = (sel_feats * sel_w[:, None]).sum(axis=0) / wsum
        else:
            v0 = sel_feats.mean(axis=0)

        if use_trimmed_mean:
            sel_feats, sel_w = self.apply_trimmed_selection(
                sel_feats,
                sel_w,
                v0,
                keep_frac=trimmed_keep_frac,
                min_patches=trimmed_min_patches,
            )

        if weighted:
            wsum = float(np.sum(sel_w, dtype=np.float32))
            if wsum <= 1e-12:
                return None
            v = (sel_feats * sel_w[:, None]).sum(axis=0) / wsum
        else:
            v = sel_feats.mean(axis=0)

        return self.normalize_if_needed(v.astype(np.float32))

    def pool_descriptor_from_patch_mask(
        self,
        fmap: np.ndarray,
        patch_mask: np.ndarray,
        patch_weights: np.ndarray | None = None,
        weighted: bool = True,
        use_trimmed_mean: bool = False,
        trimmed_keep_frac: float | None = None,
        trimmed_min_patches: int | None = None,
    ) -> np.ndarray | None:
        if fmap.ndim != 3 or patch_mask.ndim != 2:
            raise ValueError("Dimensiones inválidas")

        hp, wp, dim = fmap.shape
        if patch_mask.shape != (hp, wp):
            raise ValueError("patch_mask shape mismatch")

        feats = fmap.reshape(-1, dim).astype(np.float32, copy=False)
        m = patch_mask.reshape(-1).astype(bool, copy=False)

        if not np.any(m):
            return None

        sel_feats = feats[m]

        sel_w = None
        if weighted and patch_weights is not None:
            sel_w = patch_weights.reshape(-1)[m].astype(np.float32, copy=False)
            wsum = float(np.sum(sel_w, dtype=np.float32))
            if wsum <= 1e-12:
                return None

        if weighted and sel_w is not None:
            v0 = (sel_feats * sel_w[:, None]).sum(axis=0) / wsum
        else:
            v0 = sel_feats.mean(axis=0)

        if use_trimmed_mean:
            sel_feats, sel_w = self.apply_trimmed_selection(
                sel_feats,
                sel_w,
                v0,
                keep_frac=trimmed_keep_frac,
                min_patches=trimmed_min_patches,
            )

        if weighted and sel_w is not None:
            wsum = float(np.sum(sel_w, dtype=np.float32))
            if wsum <= 1e-12:
                return None
            v = (sel_feats * sel_w[:, None]).sum(axis=0) / wsum
        else:
            v = sel_feats.mean(axis=0)

        return self.normalize_if_needed(v.astype(np.float32))

    def normalize_if_needed(self, v: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return v
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-12 else v
