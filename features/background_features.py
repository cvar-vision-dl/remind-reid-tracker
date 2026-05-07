# features/background_features.py

from __future__ import annotations

import numpy as np
import cv2

from utils.math import kmeans_np, l2_normalize_rows, l2_normalize_vector
from utils.time import ExecutionTimer


class BackgroundFeatureExtractor:
    """
    Extractor de fondo local (inner/outer) a partir de anillos en patch-space.

    Devuelve (para cada detection):
      - global por anillo: inner/outer
      - global combinado: combined (mezcla inner+outer)
      - prototipos estables por anillo: inner_protos / outer_protos (y weights)
      - rings: máscaras/coverage para debug
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.last_timings_seconds: dict[str, float] = {}

    def extract(self, dino, fmap, obj_mask_px, patch_cache: dict | None = None, frame_cache: dict | None = None):
        if not self.enabled:
            self.last_timings_seconds = {}
            return None

        timer = ExecutionTimer()
        out = self.compute_local_background_descriptors(
            dino_extractor=dino,
            fmap=fmap,
            obj_mask_px=obj_mask_px,
            bg_local_cfg=self.config,
            timer=timer,
            patch_cache=patch_cache,
            frame_cache=frame_cache,
        )
        self.last_timings_seconds = timer.snapshot_seconds()

        return {
            "inner": out.get("bg_inner_desc", None),
            "outer": out.get("bg_outer_desc", None),
            "combined": out.get("bg_desc", None),
            "inner_protos": out.get("bg_inner_protos", None),
            "inner_proto_weights": out.get("bg_inner_proto_weights", None),
            "outer_protos": out.get("bg_outer_protos", None),
            "outer_proto_weights": out.get("bg_outer_proto_weights", None),
            "rings": out.get("rings", None),
            "quality": out.get("quality", None),
        }

    def dilate_patch_mask(self, mask_patch: np.ndarray, radius_patches: int) -> np.ndarray:
        """
        Dilata una máscara en patch-space (Hp,Wp) usando kernel cuadrado.
        radius_patches=R => kernel (2R+1).
        """
        if mask_patch.ndim != 2:
            raise ValueError("mask_patch debe ser 2D (Hp, Wp)")

        r = int(max(0, radius_patches))
        if r == 0:
            return mask_patch.astype(bool, copy=False)

        k = 2 * r + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        out = cv2.dilate(mask_patch.astype(np.uint8, copy=False), kernel, iterations=1)
        return out.astype(bool)

    def fill_holes_mask(self, mask_px: np.ndarray) -> np.ndarray:
        mask_u8 = (mask_px.astype(np.uint8, copy=False) > 0).astype(np.uint8)
        padded = cv2.copyMakeBorder(mask_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        inv = (1 - padded).astype(np.uint8)
        h, w = inv.shape[:2]
        flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
        cv2.floodFill(inv, flood, (0, 0), 2)
        holes = inv == 1
        filled = padded.copy()
        filled[holes] = 1
        out = filled[1:-1, 1:-1]
        return out.astype(bool)

    def keep_largest_component(self, mask_px: np.ndarray) -> np.ndarray:
        mask_u8 = (mask_px.astype(np.uint8, copy=False) > 0).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n_labels <= 1:
            return mask_u8.astype(bool)
        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        return (labels == best)

    def convex_hull_mask(self, mask_px: np.ndarray) -> np.ndarray:
        mask_u8 = (mask_px.astype(np.uint8, copy=False) > 0).astype(np.uint8)
        pts = cv2.findNonZero(mask_u8)
        if pts is None:
            return mask_u8.astype(bool)

        hull = cv2.convexHull(pts)
        out = np.zeros_like(mask_u8)
        cv2.fillConvexPoly(out, hull, 1)
        return out.astype(bool)

    def sanitize_object_mask(self, obj_mask_px: np.ndarray, bg_local_cfg: dict) -> tuple[np.ndarray, dict]:
        raw = obj_mask_px.astype(bool, copy=False)
        raw_area = int(np.count_nonzero(raw))

        san_cfg = (bg_local_cfg.get("sanitize", {}) or {}) if isinstance(bg_local_cfg, dict) else {}
        enabled = bool(san_cfg.get("enabled", True))
        if not enabled:
            return raw.astype(bool), {
                "raw_area_px": float(raw_area),
                "sanitized_area_px": float(raw_area),
                "hole_fill_area_px": 0.0,
                "hole_fill_ratio": 0.0,
                "mask_quality": 1.0,
            }

        cur = raw.astype(np.uint8, copy=False)
        mode = str(san_cfg.get("mode", "morphology") or "morphology").strip().lower()
        if mode in ("hull", "convex", "convexhull"):
            mode = "convex_hull"
        if mode not in ("morphology", "convex_hull"):
            mode = "morphology"

        if bool(san_cfg.get("keep_largest_component", False)):
            cur = self.keep_largest_component(cur).astype(np.uint8)

        if mode == "convex_hull":
            cur = self.convex_hull_mask(cur).astype(np.uint8)
        else:
            if bool(san_cfg.get("fill_holes", True)):
                cur = self.fill_holes_mask(cur).astype(np.uint8)

            close_px = max(0, int(san_cfg.get("close_px", 3)))
            if close_px > 0:
                k = np.ones((close_px, close_px), dtype=np.uint8)
                cur = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, k)

            open_px = max(0, int(san_cfg.get("open_px", 0)))
            if open_px > 0:
                k = np.ones((open_px, open_px), dtype=np.uint8)
                cur = cv2.morphologyEx(cur, cv2.MORPH_OPEN, k)

        san = cur.astype(bool)
        san_area = int(np.count_nonzero(san))
        hole_fill = int(max(0, san_area - raw_area))
        hole_ratio = float(hole_fill / max(1.0, float(san_area))) if san_area > 0 else 0.0
        mask_quality = float(max(0.0, min(1.0, 1.0 - hole_ratio)))

        return san, {
            "raw_area_px": float(raw_area),
            "sanitized_area_px": float(san_area),
            "hole_fill_area_px": float(hole_fill),
            "hole_fill_ratio": float(hole_ratio),
            "mask_quality": float(mask_quality),
            "sanitize_mode": str(mode),
        }

    def build_local_rings_patch_masks(
        self,
        dino_extractor,
        fmap: np.ndarray,
        obj_mask_px: np.ndarray,
        bg_local_cfg: dict,
        patch_cache: dict | None = None,
        timer: ExecutionTimer | None = None,
    ) -> dict:
        """
        Construye máscaras en patch-space alrededor del objeto (inner/outer rings).

        - obj_mask_px: máscara en píxeles (H,W) alineada con el fmap.
        - fmap: (Hp,Wp,D)

        bg_local_cfg:
          inner_radius_patches: int
          outer_radius_patches: int
          ring_mode: "disjoint" | "nested"
        """
        if fmap.ndim != 3:
            raise ValueError("fmap debe ser (Hp, Wp, D)")
        if obj_mask_px is None:
            raise ValueError("obj_mask_px no puede ser None")
        if not isinstance(bg_local_cfg, dict):
            raise ValueError("bg_local_cfg debe ser dict")

        hp, wp, _ = fmap.shape

        obj_mask_raw = obj_mask_px.astype(bool, copy=False)
        if timer is None:
            obj_mask_sane, mask_stats = self.sanitize_object_mask(obj_mask_px=obj_mask_px, bg_local_cfg=bg_local_cfg)
        else:
            obj_mask_sane, mask_stats = timer.run(
                "bg_rings/sanitize_mask",
                self.sanitize_object_mask,
                obj_mask_px=obj_mask_px,
                bg_local_cfg=bg_local_cfg,
            )

        cov_obj = None
        obj_patch = None
        if isinstance(patch_cache, dict) and np.array_equal(obj_mask_sane, obj_mask_raw):
            cov_cached = patch_cache.get("cov", None)
            patch_cached = patch_cache.get("patch_mask", None)
            if isinstance(cov_cached, np.ndarray) and cov_cached.shape == (hp, wp):
                cov_obj = cov_cached.astype(np.float32, copy=False)
            if isinstance(patch_cached, np.ndarray) and patch_cached.shape == (hp, wp):
                obj_patch = patch_cached.astype(bool, copy=False)

        if cov_obj is None:
            if timer is None:
                cov_obj = dino_extractor.mask_px_to_patch_coverage(
                    obj_mask_sane.astype(np.uint8, copy=False),
                    hp,
                    wp,
                )
            else:
                cov_obj = timer.run(
                    "bg_rings/mask_to_patch_coverage",
                    dino_extractor.mask_px_to_patch_coverage,
                    obj_mask_sane.astype(np.uint8, copy=False),
                    hp,
                    wp,
                )
        if obj_patch is None:
            cov_thr = float(bg_local_cfg.get("obj_patch_min_coverage", 0.0))
            cov_thr = max(0.0, min(1.0, cov_thr))
            obj_patch = cov_obj > cov_thr

        r_in = int(bg_local_cfg.get("inner_radius_patches", 2))
        r_out = int(bg_local_cfg.get("outer_radius_patches", 4))
        r_in = max(0, r_in)
        r_out = max(r_in, r_out)

        dilated_cache: dict[int, np.ndarray] = {0: obj_patch.astype(bool, copy=False)}

        def get_dilated(radius: int) -> np.ndarray:
            r = int(max(0, radius))
            cached = dilated_cache.get(r, None)
            if cached is not None:
                return cached
            out = self.dilate_patch_mask(obj_patch, r)
            dilated_cache[r] = out
            return out

        adapt_cfg = (bg_local_cfg.get("adaptive", {}) or {}) if isinstance(bg_local_cfg, dict) else {}
        adapt_enabled = bool(adapt_cfg.get("enabled", True))
        min_inner = max(0, int(adapt_cfg.get("min_inner_patches", 12)))
        min_outer = max(0, int(adapt_cfg.get("min_outer_patches", 24)))
        max_radius = max(r_out, int(adapt_cfg.get("max_radius_patches", max(r_out, r_in) + 6)))
        border_excl = max(0, int(adapt_cfg.get("border_exclusion_patches", 1)))

        mode = str(bg_local_cfg.get("ring_mode", "disjoint")).strip().lower()
        if mode not in ("disjoint", "nested"):
            mode = "disjoint"

        def _build_rings():
            obj_excl = get_dilated(border_excl)

            r_in_used = int(max(r_in, border_excl))
            d_in = get_dilated(r_in_used)
            ring_inner = d_in & (~obj_excl)

            if adapt_enabled:
                while int(np.count_nonzero(ring_inner)) < min_inner and r_in_used < max_radius:
                    r_in_used += 1
                    d_in = get_dilated(r_in_used)
                    ring_inner = d_in & (~obj_excl)

            r_out_used = int(max(r_out, r_in_used))
            d_out = get_dilated(r_out_used)

            if mode == "nested":
                ring_outer = d_out & (~obj_excl)
            else:
                ring_outer = d_out & (~d_in) & (~obj_excl)

            if adapt_enabled:
                while int(np.count_nonzero(ring_outer)) < min_outer and r_out_used < max_radius:
                    r_out_used += 1
                    d_out = get_dilated(r_out_used)
                    if mode == "nested":
                        ring_outer = d_out & (~obj_excl)
                    else:
                        ring_outer = d_out & (~d_in) & (~obj_excl)

            return obj_excl, d_in, d_out, ring_inner, ring_outer, r_in_used, r_out_used

        if timer is None:
            obj_excl, d_in, d_out, ring_inner, ring_outer, r_in_used, r_out_used = _build_rings()
        else:
            obj_excl, d_in, d_out, ring_inner, ring_outer, r_in_used, r_out_used = timer.run(
                "bg_rings/adaptive_rings",
                _build_rings,
            )

        return {
            "coverage_obj": cov_obj.astype(np.float32),
            "obj_patch": obj_patch.astype(bool),
            "obj_mask_sanitized_px": obj_mask_sane.astype(bool),
            "ring_inner": ring_inner.astype(bool),
            "ring_outer": ring_outer.astype(bool),
            "stats": {
                **mask_stats,
                "obj_patch_count": int(np.count_nonzero(obj_patch)),
                "inner_patch_count": int(np.count_nonzero(ring_inner)),
                "outer_patch_count": int(np.count_nonzero(ring_outer)),
                "inner_radius_used": int(r_in_used),
                "outer_radius_used": int(r_out_used),
                "border_exclusion_patches": int(border_excl),
            },
        }

    def estimate_k_from_patches(self, p_count: int, proto_cfg: dict) -> int:
        """
        K adaptativo en base a nº de patches válidos P del anillo.
        """
        p = int(max(0, p_count))
        if p <= 0:
            return 0

        k_min = int(proto_cfg.get("k_min", 4))
        k_max = int(proto_cfg.get("k_max", 24))
        k_min = max(1, k_min)
        k_max = max(k_min, k_max)

        mode = str(proto_cfg.get("k_mode", "sqrt")).strip().lower()

        if mode == "ppc":
            ppc = float(proto_cfg.get("patches_per_cluster", 15))
            ppc = max(1.0, ppc)
            k_raw = int(np.round(p / ppc))
        else:
            c_sqrt = float(proto_cfg.get("c_sqrt", 1.0))
            c_sqrt = max(0.01, c_sqrt)
            k_raw = int(np.round(c_sqrt * np.sqrt(float(p))))

        k = int(np.clip(k_raw, k_min, k_max))

        min_pts = int(proto_cfg.get("min_pts_per_cluster", 4))
        min_pts = max(1, min_pts)
        k_cap = int(max(1, p // min_pts))
        k = int(min(k, k_cap))

        k = int(min(k, p))
        return int(max(1, k))

    def compute_cluster_stats(
        self,
        feats: np.ndarray,
        weights: np.ndarray | None,
        labels: np.ndarray,
        centers: np.ndarray,
        feats_are_normalized: bool = False,
    ) -> list[dict]:
        """
        Devuelve stats por cluster:
          - idxs: índices de puntos
          - mass: suma de pesos (o count)
          - centroid: centro normalizado (cosine-friendly)
          - cohesion: mean cosine(punto, centroid)
        """
        n = int(feats.shape[0])
        if n <= 0:
            return []

        feats_n = feats if feats_are_normalized else l2_normalize_rows(feats)
        k = int(centers.shape[0])
        labels_i = np.asarray(labels, dtype=np.int32).reshape(-1)
        if labels_i.size != n or k <= 0:
            return []

        counts = np.bincount(labels_i, minlength=k)
        order = np.argsort(labels_i, kind="stable")
        weights_arr = None if weights is None else np.asarray(weights, dtype=np.float32).reshape(-1)

        out = []
        start = 0
        for ci in range(k):
            count = int(counts[ci]) if ci < counts.size else 0
            if count <= 0:
                continue

            stop = start + count
            sel = order[start:stop].astype(np.int32, copy=False)
            start = stop

            sum_vec = np.sum(feats_n[sel], axis=0, dtype=np.float32)
            if weights_arr is None:
                mass = float(count)
                weighted_sum_vec = None
            else:
                weights_sel = weights_arr[sel]
                mass = float(np.sum(weights_sel, dtype=np.float32))
                weighted_sum_vec = np.sum(feats_n[sel] * weights_sel[:, None], axis=0, dtype=np.float32)

            c = centers[ci].astype(np.float32, copy=False)
            c = l2_normalize_vector(c)
            cohesion = float(np.dot(sum_vec, c) / float(count)) if count > 0 else 0.0

            out.append(
                {
                    "cluster_id": int(ci),
                    "idxs": sel,
                    "count": int(count),
                    "mass": float(mass),
                    "centroid": c.astype(np.float32, copy=False),
                    "cohesion": float(cohesion),
                    "sum_vec": sum_vec.astype(np.float32, copy=False),
                    "weighted_sum_vec": None if weighted_sum_vec is None else weighted_sum_vec.astype(np.float32, copy=False),
                }
            )

        return out

    def union_find_build(self, n: int):
        parent = np.arange(n, dtype=np.int32)
        rank = np.zeros(n, dtype=np.int32)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = int(parent[i])
            return int(i)

        def union(a: int, b: int):
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        return find, union, parent

    def merge_clusters_by_similarity(self, clusters: list[dict], sim_thr: float) -> list[list[int]]:
        """
        Agrupa clusters por conectividad: si cos(centroid_i, centroid_j) > sim_thr,
        los unimos. Devuelve lista de grupos (índices sobre `clusters`).
        """
        if not clusters:
            return []

        thr = float(sim_thr)
        n = len(clusters)
        find, union, parent = self.union_find_build(n)

        centroids = np.stack([c["centroid"] for c in clusters], axis=0).astype(np.float32)
        centroids = l2_normalize_rows(centroids)

        sim = centroids @ centroids.T
        for i in range(n):
            for j in range(i + 1, n):
                if float(sim[i, j]) > thr:
                    union(i, j)

        groups = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)

        return list(groups.values())

    def build_merged_clusters(
        self,
        feats: np.ndarray,
        weights: np.ndarray | None,
        clusters: list[dict],
        groups: list[list[int]],
        feats_are_normalized: bool = False,
    ) -> list[dict]:
        """
        Construye clusters merged (concatenando índices) y recomputa stats.
        """
        if not groups:
            return []
        del feats, weights, feats_are_normalized

        merged = []
        for gi, ids in enumerate(groups):
            idxs_list = [clusters[i]["idxs"] for i in ids]
            idxs = np.concatenate(idxs_list, axis=0).astype(np.int32, copy=False)

            count = int(sum(int(clusters[i].get("count", clusters[i]["idxs"].size)) for i in ids))
            mass = float(sum(float(clusters[i].get("mass", 0.0)) for i in ids))
            sum_vec = np.sum(
                np.stack([clusters[i]["sum_vec"] for i in ids], axis=0).astype(np.float32, copy=False),
                axis=0,
                dtype=np.float32,
            )
            weighted_terms = [
                clusters[i].get("weighted_sum_vec", None)
                for i in ids
                if clusters[i].get("weighted_sum_vec", None) is not None
            ]
            weighted_sum_vec = None
            if len(weighted_terms) == len(ids) and weighted_terms:
                weighted_sum_vec = np.sum(
                    np.stack(weighted_terms, axis=0).astype(np.float32, copy=False),
                    axis=0,
                    dtype=np.float32,
                )

            if weighted_sum_vec is not None and mass > 1e-12:
                c = weighted_sum_vec / float(mass)
            elif count > 0:
                c = sum_vec / float(count)
            else:
                c = sum_vec

            c = l2_normalize_vector(c.astype(np.float32, copy=False))
            cohesion = float(np.dot(sum_vec, c) / float(count)) if count > 0 else 0.0

            merged.append(
                {
                    "cluster_id": int(gi),
                    "idxs": idxs,
                    "count": int(count),
                    "mass": float(mass),
                    "centroid": c.astype(np.float32, copy=False),
                    "cohesion": float(cohesion),
                    "sum_vec": sum_vec.astype(np.float32, copy=False),
                    "weighted_sum_vec": None if weighted_sum_vec is None else weighted_sum_vec.astype(np.float32, copy=False),
                }
            )

        return merged

    def select_stable_prototypes(
        self,
        feats: np.ndarray,
        weights: np.ndarray | None,
        clusters: list[dict],
        proto_cfg: dict,
        feats_are_normalized: bool = False,
    ) -> tuple[list[np.ndarray], list[float], dict]:
        """
        Selecciona top-N clusters estables (masa alta + cohesión alta),
        y devuelve prototipos + pesos (normalizados a 1).

        proto_mode:
          - medoid: punto del cluster con mayor cosine a centroid
          - centroid: el centroid normalizado
        """
        if not clusters:
            return [], [], {"selected": [], "clusters": []}

        total_mass = float(sum(float(c["mass"]) for c in clusters)) + 1e-12
        min_mass_frac = float(proto_cfg.get("min_mass_frac", 0.06))
        top_n = int(proto_cfg.get("top_n", 3))
        top_n = max(0, top_n)

        power = float(proto_cfg.get("cohesion_power", 2.0))
        power = max(0.0, power)

        proto_mode = str(proto_cfg.get("proto_mode", "medoid")).strip().lower()
        if proto_mode not in ("medoid", "centroid"):
            proto_mode = "medoid"

        feats_n = feats if feats_are_normalized else l2_normalize_rows(feats)

        scored = []
        for ci, c in enumerate(clusters):
            mass = float(c["mass"])
            mass_frac = mass / total_mass
            if mass_frac < min_mass_frac:
                continue

            cohesion = float(c["cohesion"])
            stability = mass * (cohesion ** power)

            scored.append(
                {
                    "idx": int(ci),
                    "mass": mass,
                    "mass_frac": mass_frac,
                    "cohesion": cohesion,
                    "stability": float(stability),
                }
            )

        scored.sort(key=lambda x: x["stability"], reverse=True)
        if top_n > 0:
            scored = scored[:top_n]

        protos = []
        proto_w = []
        selected_dbg = []

        for s in scored:
            c = clusters[int(s["idx"])]
            idxs = c["idxs"]
            centroid = c["centroid"]

            if proto_mode == "centroid":
                proto = centroid.astype(np.float32, copy=False)
            else:
                sims = feats_n[idxs] @ centroid
                best_local = int(np.argmax(sims))
                best_idx = int(idxs[best_local])
                proto = feats_n[best_idx].astype(np.float32, copy=False)

            protos.append(proto)
            proto_w.append(float(c["mass"]))

            selected_dbg.append(
                {
                    "mass": float(s["mass"]),
                    "mass_frac": float(s["mass_frac"]),
                    "cohesion": float(s["cohesion"]),
                    "stability": float(s["stability"]),
                    "n_pts": int(c["idxs"].size),
                }
            )

        wsum = float(sum(proto_w)) + 1e-12
        proto_w = [float(w / wsum) for w in proto_w]

        dbg = {
            "clusters": [
                {
                    "mass": float(c["mass"]),
                    "cohesion": float(c["cohesion"]),
                    "n_pts": int(c["idxs"].size),
                }
                for c in clusters
            ],
            "selected": selected_dbg,
        }
        return protos, proto_w, dbg

    def compute_ring_prototypes(
        self,
        fmap: np.ndarray,
        ring_mask: np.ndarray,
        patch_weights: np.ndarray | None,
        proto_cfg: dict,
        seed: int,
        frame_cache: dict | None = None,
    ) -> tuple[list[np.ndarray], list[float], dict]:
        """
        Extrae prototipos estables de un anillo.

        - fmap: (Hp,Wp,D)
        - ring_mask: (Hp,Wp) bool
        - patch_weights: (Hp,Wp) float o None (recomendado: w_bg = 1 - cov_obj)
        """
        hp, wp, dim = fmap.shape
        if ring_mask.shape != (hp, wp):
            raise ValueError("ring_mask shape mismatch")

        m = ring_mask.reshape(-1).astype(bool)
        p_count = int(np.count_nonzero(m))
        if p_count <= 0:
            return [], [], {"k": 0, "note": "no_patches"}

        feats = frame_cache.get("flat_feats") if isinstance(frame_cache, dict) else None
        if feats is None:
            feats = fmap.reshape(-1, dim).astype(np.float32, copy=False)
            if isinstance(frame_cache, dict):
                frame_cache["flat_feats"] = feats
        feats_sel = feats[m]

        weights_sel = None
        if patch_weights is not None:
            w = patch_weights.reshape(-1).astype(np.float32, copy=False)
            weights_sel = w[m]
            if float(np.sum(weights_sel)) <= 1e-12:
                weights_sel = None

        feats_n = frame_cache.get("flat_feats_n") if isinstance(frame_cache, dict) else None
        feats_sel_n = feats_n[m] if feats_n is not None else l2_normalize_rows(feats_sel)

        k = self.estimate_k_from_patches(p_count=p_count, proto_cfg=proto_cfg)
        if k <= 1 or feats_sel_n.shape[0] < k:
            # fallback: un proto (centroid) si hay señal
            centroid = l2_normalize_vector(np.mean(feats_sel_n, axis=0))
            protos = [centroid.astype(np.float32)]
            weights = [1.0]
            dbg = {"k": int(k), "note": "fallback_centroid", "selected": [{"n_pts": int(p_count)}]}
            return protos, weights, dbg

        iters = int(proto_cfg.get("iters", 10))
        n_init = int(proto_cfg.get("n_init", 1))

        labels, centers, _inertia = kmeans_np(
            x=feats_sel_n.astype(np.float32, copy=False),
            k=int(k),
            n_iter=int(iters),
            n_init=int(n_init),
            seed=int(seed),
        )

        clusters = self.compute_cluster_stats(
            feats=feats_sel_n,
            weights=weights_sel,
            labels=labels,
            centers=centers.astype(np.float32, copy=False),
            feats_are_normalized=True,
        )

        sim_thr = float(proto_cfg.get("merge_sim_thr", 0.92))
        groups = self.merge_clusters_by_similarity(clusters=clusters, sim_thr=sim_thr)
        merged_clusters = self.build_merged_clusters(
            feats=feats_sel_n,
            weights=weights_sel,
            clusters=clusters,
            groups=groups,
            feats_are_normalized=True,
        )

        protos, proto_w, dbg_sel = self.select_stable_prototypes(
            feats=feats_sel_n,
            weights=weights_sel,
            clusters=merged_clusters,
            proto_cfg=proto_cfg,
            feats_are_normalized=True,
        )

        dbg = {
            "k": int(k),
            "p_count": int(p_count),
            "n_clusters_raw": int(len(clusters)),
            "n_clusters_merged": int(len(merged_clusters)),
            "merge_sim_thr": float(sim_thr),
            "selection": dbg_sel,
        }
        return protos, proto_w, dbg

    def compute_local_background_descriptors(
        self,
        dino_extractor,
        fmap: np.ndarray,
        obj_mask_px: np.ndarray | None,
        bg_local_cfg: dict,
        timer: ExecutionTimer | None = None,
        patch_cache: dict | None = None,
        frame_cache: dict | None = None,
    ) -> dict:
        """
        Calcula:
          - bg_inner_desc / bg_outer_desc: global por anillo
          - bg_desc: global combinado
          - bg_*_protos / bg_*_proto_weights: prototipos estables por anillo (opcional)
          - rings: masks/coverage + debug de protos
        """
        enabled = bool(bg_local_cfg.get("enabled", False))
        if not enabled:
            return {
                "enabled": False,
                "bg_inner_desc": None,
                "bg_outer_desc": None,
                "bg_desc": None,
                "bg_inner_protos": None,
                "bg_inner_proto_weights": None,
                "bg_outer_protos": None,
                "bg_outer_proto_weights": None,
                "rings": None,
                "quality": None,
            }

        if obj_mask_px is None:
            return {
                "enabled": True,
                "bg_inner_desc": None,
                "bg_outer_desc": None,
                "bg_desc": None,
                "bg_inner_protos": None,
                "bg_inner_proto_weights": None,
                "bg_outer_protos": None,
                "bg_outer_proto_weights": None,
                "rings": None,
                "quality": None,
            }

        if timer is None:
            rings = self.build_local_rings_patch_masks(
                dino_extractor=dino_extractor,
                fmap=fmap,
                obj_mask_px=obj_mask_px,
                bg_local_cfg=bg_local_cfg,
                patch_cache=patch_cache,
            )
        else:
            rings = timer.run(
                "bg_rings",
                self.build_local_rings_patch_masks,
                dino_extractor=dino_extractor,
                fmap=fmap,
                obj_mask_px=obj_mask_px,
                bg_local_cfg=bg_local_cfg,
                patch_cache=patch_cache,
                timer=timer,
            )

        cov_obj = rings["coverage_obj"]
        w_bg = (1.0 - cov_obj).astype(np.float32)

        # 1) Global descriptors (como antes)
        if timer is None:
            bg_inner_desc = dino_extractor.pool_descriptor_from_patch_mask(
                fmap=fmap,
                patch_mask=rings["ring_inner"],
                patch_weights=w_bg,
            )
            bg_outer_desc = dino_extractor.pool_descriptor_from_patch_mask(
                fmap=fmap,
                patch_mask=rings["ring_outer"],
                patch_weights=w_bg,
            )
        else:
            bg_inner_desc = timer.run(
                "bg_inner_global",
                dino_extractor.pool_descriptor_from_patch_mask,
                fmap=fmap,
                patch_mask=rings["ring_inner"],
                patch_weights=w_bg,
            )
            bg_outer_desc = timer.run(
                "bg_outer_global",
                dino_extractor.pool_descriptor_from_patch_mask,
                fmap=fmap,
                patch_mask=rings["ring_outer"],
                patch_weights=w_bg,
            )

        # 2) Combined global descriptor (mezcla inner+outer)
        bg_desc = None
        cw = bg_local_cfg.get("combine_weights", {})
        if not isinstance(cw, dict):
            cw = {}

        wi = float(cw.get("inner", 0.7))
        wo = float(cw.get("outer", 0.3))

        acc = None
        wsum = 0.0

        if bg_inner_desc is not None and wi > 0.0:
            acc = bg_inner_desc * wi if acc is None else (acc + bg_inner_desc * wi)
            wsum += wi

        if bg_outer_desc is not None and wo > 0.0:
            acc = bg_outer_desc * wo if acc is None else (acc + bg_outer_desc * wo)
            wsum += wo

        if acc is not None and wsum > 1e-12:
            bg_desc = (acc / float(wsum)).astype(np.float32)
            if getattr(dino_extractor, "normalize", True):
                n = float(np.linalg.norm(bg_desc))
                if n > 1e-12:
                    bg_desc = bg_desc / n

        # 3) Prototypes (NEW)
        proto_cfg = bg_local_cfg.get("prototypes", {})
        if not isinstance(proto_cfg, dict):
            proto_cfg = {}

        proto_enabled = bool(proto_cfg.get("enabled", False))

        bg_inner_protos = None
        bg_inner_proto_weights = None
        bg_outer_protos = None
        bg_outer_proto_weights = None

        dbg_protos = None
        if proto_enabled:
            seed = int(proto_cfg.get("seed", 0))

            if timer is None:
                inner_protos, inner_w, inner_dbg = self.compute_ring_prototypes(
                    fmap=fmap,
                    ring_mask=rings["ring_inner"],
                    patch_weights=w_bg,
                    proto_cfg=proto_cfg,
                    seed=seed,
                    frame_cache=frame_cache,
                )
                outer_protos, outer_w, outer_dbg = self.compute_ring_prototypes(
                    fmap=fmap,
                    ring_mask=rings["ring_outer"],
                    patch_weights=w_bg,
                    proto_cfg=proto_cfg,
                    seed=seed + 1,
                    frame_cache=frame_cache,
                )
            else:
                inner_protos, inner_w, inner_dbg = timer.run(
                    "bg_proto_inner",
                    self.compute_ring_prototypes,
                    fmap=fmap,
                    ring_mask=rings["ring_inner"],
                    patch_weights=w_bg,
                    proto_cfg=proto_cfg,
                    seed=seed,
                    frame_cache=frame_cache,
                )
                outer_protos, outer_w, outer_dbg = timer.run(
                    "bg_proto_outer",
                    self.compute_ring_prototypes,
                    fmap=fmap,
                    ring_mask=rings["ring_outer"],
                    patch_weights=w_bg,
                    proto_cfg=proto_cfg,
                    seed=seed + 1,
                    frame_cache=frame_cache,
                )

            bg_inner_protos = inner_protos if inner_protos else []
            bg_inner_proto_weights = inner_w if inner_w else []
            bg_outer_protos = outer_protos if outer_protos else []
            bg_outer_proto_weights = outer_w if outer_w else []

            dbg_protos = {
                "inner": inner_dbg,
                "outer": outer_dbg,
            }

        rings_dbg = dict(rings)
        if dbg_protos is not None:
            rings_dbg["prototypes_dbg"] = dbg_protos

        stats = (rings_dbg.get("stats", {}) or {}) if isinstance(rings_dbg, dict) else {}
        quality = {
            "inner_patch_count": int(stats.get("inner_patch_count", 0)),
            "outer_patch_count": int(stats.get("outer_patch_count", 0)),
            "mask_quality": float(stats.get("mask_quality", 1.0)),
        }

        return {
            "enabled": True,
            "bg_inner_desc": bg_inner_desc,
            "bg_outer_desc": bg_outer_desc,
            "bg_desc": bg_desc,
            "bg_inner_protos": bg_inner_protos,
            "bg_inner_proto_weights": bg_inner_proto_weights,
            "bg_outer_protos": bg_outer_protos,
            "bg_outer_proto_weights": bg_outer_proto_weights,
            "rings": rings_dbg,
            "quality": quality,
        }
