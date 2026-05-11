# utils/math.py

import numpy as np

def l2_normalize_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(x))
    if n < eps:
        return x * 0.0
    return x / n

def l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    nrm = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / nrm

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = l2_normalize_vector(a)
    b = l2_normalize_vector(b)
    return float(np.dot(a, b))

def kmeans_np(x: np.ndarray, k: int, n_iter: int = 50, n_init: int = 3, seed: int = 0):
    """
    Simple k-means (numpy) returning (labels, centers, inertia).
    Simplified numpy implementation for the current pipeline.
    """
    if x.ndim != 2:
        raise ValueError("kmeans_np: x must be (N,D)")
    x = np.asarray(x)
    n, d = x.shape
    if n < k:
        raise ValueError(f"kmeans_np: N={n} < K={k}")

    # Keep float32 path when input is float32 to avoid costly promotion and
    # improve throughput in the hot per-object kmeans path.
    dtype = np.float32 if x.dtype == np.float32 else np.float64
    x_work = x.astype(dtype, copy=False)
    x_sq = np.sum(x_work * x_work, axis=1, dtype=dtype)

    rng = np.random.default_rng(int(seed))

    best_inertia = float("inf")
    best_labels = None
    best_centers = None

    for _ in range(int(n_init)):
        centers = np.empty((k, d), dtype=dtype)

        idx0 = int(rng.integers(0, n))
        centers[0] = x_work[idx0]
        center_sq0 = float(np.sum(centers[0] * centers[0], dtype=dtype))
        dist2 = x_sq - (dtype(2.0) * (x_work @ centers[0])) + dtype(center_sq0)

        for ci in range(1, k):
            if np.any(dist2 > 0):
                idx = int(np.argmax(dist2))
            else:
                idx = int(rng.integers(0, n))
            centers[ci] = x_work[idx]
            center_sq = float(np.sum(centers[ci] * centers[ci], dtype=dtype))
            cur_dist2 = x_sq - (dtype(2.0) * (x_work @ centers[ci])) + dtype(center_sq)
            np.minimum(dist2, cur_dist2, out=dist2)

        labels = np.full(n, -1, dtype=np.int32)
        best_dist = np.empty((n,), dtype=dtype)
        best_lab = np.empty((n,), dtype=np.int32)

        for _it in range(int(n_iter)):
            best_dist.fill(np.inf)
            best_lab.fill(0)
            for ci in range(int(k)):
                c = centers[ci]
                c_sq = float(np.sum(c * c, dtype=dtype))
                d_ci = x_sq - (dtype(2.0) * (x_work @ c)) + dtype(c_sq)
                take = d_ci < best_dist
                if np.any(take):
                    best_dist[take] = d_ci[take]
                    best_lab[take] = int(ci)

            unchanged = bool(np.array_equal(best_lab, labels))
            np.copyto(labels, best_lab)
            if unchanged:
                break

            counts = np.bincount(labels, minlength=k)
            nonempty = counts > 0
            order = np.argsort(labels, kind="stable")
            x_ord = x_work[order]
            start = 0
            for ci in range(int(k)):
                cnt = int(counts[ci]) if ci < counts.size else 0
                if cnt > 0:
                    stop = start + cnt
                    centers[ci] = x_ord[start:stop].mean(axis=0, dtype=dtype)
                    start = stop

            empty_ids = np.flatnonzero(~nonempty)
            for ci in empty_ids.tolist():
                idx = int(rng.integers(0, n))
                centers[ci] = x_work[idx]

        if labels[0] < 0:
            # Guard for pathological n_iter<=0 paths.
            best_dist.fill(np.inf)
            for ci in range(int(k)):
                c = centers[ci]
                c_sq = float(np.sum(c * c, dtype=dtype))
                d_ci = x_sq - (dtype(2.0) * (x_work @ c)) + dtype(c_sq)
                np.minimum(best_dist, d_ci, out=best_dist)

        inertia = float(np.sum(best_dist, dtype=np.float64))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centers = centers.copy()

    return best_labels, best_centers, float(best_inertia)

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    uni = len(a | b)
    if uni <= 0:
        return 1.0
    return float(inter) / float(uni)
