# update/proto_ops.py

from __future__ import annotations

import numpy as np

from utils.math import l2_normalize_vector


def make_proto_event(
    kind: str,
    label: str | None,
    object_id: int,
    channel: str,
    action: str,
    s_max: float | None,
    n_before: int,
    n_after: int,
    scope: dict | None = None,
    merge_pair_sim: float | None = None,
    evict_strategy: str | None = None,
    evicted_index: int | None = None,
    extra: dict | None = None,
) -> dict:
    d = {
        "kind": str(kind),
        "scope": scope if isinstance(scope, dict) else None,
        "label": label,
        "object_id": int(object_id),
        "channel": str(channel),
        "action": str(action),
        "s_max": None if s_max is None else float(s_max),
        "n_before": int(n_before),
        "n_after": int(n_after),
        "merge_pair_sim": None if merge_pair_sim is None else float(merge_pair_sim),
        "evict_strategy": evict_strategy,
        "evicted_index": None if evicted_index is None else int(evicted_index),
    }
    if isinstance(extra, dict) and extra:
        d.update(extra)
    return d


def ensure_channel_lists(ch):
    work = getattr(ch, "work_protos", None)
    stable = getattr(ch, "stable_protos", None)

    if work is None:
        work = []
        ch.work_protos = work

    if stable is None:
        stable = []
        ch.stable_protos = stable

    return work, stable


def stack_embeddings(protos) -> np.ndarray:
    return np.stack([p.embedding for p in protos], axis=0).astype(np.float32, copy=False)


def compute_sims_to_protos(x_norm: np.ndarray, protos) -> np.ndarray:
    if not protos:
        return np.zeros((0,), dtype=np.float32)
    P = stack_embeddings(protos)
    return (P @ x_norm.astype(np.float32, copy=False)).astype(np.float32, copy=False)


def find_best_internal_pair(protos) -> tuple[float, int, int]:
    n = len(protos)
    if n < 2:
        return -1.0, -1, -1

    best_sim = -1.0
    best_i, best_j = -1, -1

    for i in range(n):
        pi = protos[i].embedding
        for j in range(i + 1, n):
            s = float(np.dot(pi, protos[j].embedding))
            if s > best_sim:
                best_sim = s
                best_i, best_j = i, j

    return float(best_sim), int(best_i), int(best_j)


def compute_redundancy_scores(protos) -> list[float]:
    n = len(protos)
    if n <= 1:
        return [0.0] * n

    out = []
    for i in range(n):
        pi = protos[i].embedding
        r = -1.0
        for j in range(n):
            if i == j:
                continue
            r = max(r, float(np.dot(pi, protos[j].embedding)))
        out.append(float(r if r > -1.0 else 0.0))
    return out


def choose_evict_index(protos, strategy: str) -> int:
    n = len(protos)
    if n <= 0:
        return -1

    s = str(strategy).lower().strip()

    if s == "lru":
        return int(min(range(n), key=lambda i: float(getattr(protos[i], "last_seen", 0.0))))

    redundancies = compute_redundancy_scores(protos)
    max_r = max(redundancies)
    cands = [i for i, r in enumerate(redundancies) if abs(r - max_r) < 1e-6]
    if len(cands) == 1:
        return int(cands[0])

    return int(min(cands, key=lambda i: float(getattr(protos[i], "last_seen", 0.0))))


def merge_two_protos_inplace(
    p_a,
    p_b,
    timestamp: float,
    merge_weight: str = "equal",
    count_cap: int = 10,
):
    a = p_a.embedding
    b = p_b.embedding

    mw = str(merge_weight).lower().strip()
    if mw == "count_capped":
        wa = float(min(int(getattr(p_a, "count", 1)), int(max(1, count_cap))))
        wb = float(min(int(getattr(p_b, "count", 1)), int(max(1, count_cap))))
    else:
        wa, wb = 1.0, 1.0

    merged = l2_normalize_vector((wa * a) + (wb * b))

    p_a.embedding = merged
    p_a.count = int(getattr(p_a, "count", 1)) + int(getattr(p_b, "count", 1))
    p_a.last_seen = float(
        max(
            float(getattr(p_a, "last_seen", 0.0)),
            float(getattr(p_b, "last_seen", 0.0)),
            float(timestamp),
        )
    )
    return p_a


def update_proto_dup_gated_ema(
    proto,
    x_norm: np.ndarray,
    timestamp: float,
    s_max: float,
    dup_thr: float,
    margin: float,
    alpha_min: float,
    alpha_max: float,
    alpha_scale: float = 1.0,
) -> str:
    """
    DUP update:
      - always bumps count/last_seen
      - only updates embedding if s_max >= dup_thr + margin
      - EMA alpha grows with s_max (clamped) and is scaled by alpha_scale
    Returns action: "DUP_COUNT" or "DUP_EMA"
    """
    proto.count = int(getattr(proto, "count", 1)) + 1
    proto.last_seen = float(timestamp)

    gate = float(dup_thr + margin)
    if float(s_max) < gate:
        return "DUP_COUNT"

    denom = max(1e-6, 1.0 - gate)
    t = (float(s_max) - gate) / denom
    t = max(0.0, min(1.0, t))

    a0 = max(0.0, min(1.0, float(alpha_min)))
    a1 = max(0.0, min(1.0, float(alpha_max)))
    if a1 < a0:
        a0, a1 = a1, a0

    scale = max(0.0, min(1.0, float(alpha_scale)))
    alpha = (a0 + (a1 - a0) * t) * scale
    if alpha <= 0.0:
        return "DUP_COUNT"

    proto.embedding = l2_normalize_vector(((1.0 - alpha) * proto.embedding) + (alpha * x_norm))
    return "DUP_EMA"

