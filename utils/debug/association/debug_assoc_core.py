# utils/debug/association/debug_assoc_core.py

from __future__ import annotations

import math

import pandas as pd

from ..debug_format import safe_float
from .debug_assoc_helpers import (
    candidate_debug_reason,
    candidate_score_final,
)
from ..debug_entity_helpers import (
    get_track_label_short,
)


def _column_all_close_to(df: pd.DataFrame, col: str, target: float, tol: float = 1e-9) -> bool:
    if col not in df.columns or df.empty:
        return True
    vals = []
    for v in df[col].tolist():
        x = safe_float(v, default=None)
        if x is None:
            return False
        vals.append(float(x))
    if not vals:
        return True
    return all(abs(float(v) - float(target)) <= float(tol) for v in vals)


def _drop_uninformative_assoc_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    out = list(cols or [])

    for col in ("S_sets", "B_sets", "P_sets", "U_sets", "Q_sets", "C_rel", "K_abs", "K_rel", "H_rel", "B_dbonus", "S_dist", "B_dist"):
        if col in out and _column_all_close_to(df, col, 0.0):
            out.remove(col)
    if "C_band" in out and _column_all_close_to(df, "C_band", 1.0):
        out.remove("C_band")

    if "KP_keep" in out and _column_all_close_to(df, "KP_keep", 1.0):
        out.remove("KP_keep")

    if "D_keep" in out and _column_all_close_to(df, "D_keep", 1.0):
        out.remove("D_keep")

    if "S_known" in out and "S_final" in df.columns and "S_known" in df.columns:
        same = True
        for a, b in zip(df["S_final"].tolist(), df["S_known"].tolist()):
            if safe_float(a, default=None) != safe_float(b, default=None):
                same = False
                break
        if same:
            out.remove("S_known")

    if "S_final2" in out and "S_final" in df.columns and "S_final2" in df.columns:
        same = True
        for a, b in zip(df["S_final"].tolist(), df["S_final2"].tolist()):
            if safe_float(a, default=None) != safe_float(b, default=None):
                same = False
                break
        if same:
            out.remove("S_final2")

    return out


def _alt_candidate_summary(ranked_candidates, selected_track_id, memory_store) -> str:
    if not ranked_candidates:
        return ""

    sel_oid = None if selected_track_id is None else int(selected_track_id)
    selected_score = None
    best_alt = None

    for c in ranked_candidates:
        if not isinstance(c, dict):
            continue
        oid = c.get("object_id", None)
        if oid is None:
            continue
        oid = int(oid)
        score = float(candidate_score_final(c))
        if sel_oid is not None and oid == sel_oid:
            selected_score = float(score)
            continue
        if best_alt is None:
            best_alt = (oid, score)

    if best_alt is None:
        return ""

    oid_alt, score_alt = best_alt
    lbl = get_track_label_short(memory_store, oid_alt, n=3) or f"ID{int(oid_alt)}"
    if selected_score is None:
        return f"{lbl}@{score_alt:.2f}"
    gap = max(0.0, float(selected_score) - float(score_alt))
    return f"{lbl}@{score_alt:.2f},gap={gap:.2f}"


def _report_decision_summary(rep, *, selected_candidate: dict | None = None, alt_summary: str = "") -> str:
    if rep is None:
        return ""

    decision = str(getattr(rep, "final_decision", "") or "").upper().strip()
    reason = str(getattr(rep, "final_reason", "") or "").strip().lower()
    parts: list[str] = []

    if decision == "MATCH":
        parts.append("match")
    elif decision == "NEW":
        parts.append("new")
    elif decision == "AMBIGUOUS_TRACK":
        k = len(getattr(rep, "ambiguous_candidate_ids", []) or [])
        parts.append(f"ambiguous_track(k={int(k)})")
    elif decision == "PROVISIONAL_NEW":
        k = len(getattr(rep, "provisional_support_ids", []) or [])
        if k > 0:
            parts.append(f"provisional_new(known={int(k)})")
        else:
            parts.append("provisional_new")
    elif decision == "PROVISIONAL_PARENT":
        k = len(getattr(rep, "provisional_support_ids", []) or [])
        if k > 0:
            parts.append(f"provisional_parent(known={int(k)})")
        else:
            parts.append("provisional_parent")
    elif decision == "UNASSIGNED":
        parts.append("unassigned")
    elif decision:
        parts.append(str(decision.lower()))

    if reason and reason not in ("assigned", "created_new", "no_assignment"):
        parts.append(str(reason))
    elif reason in ("assigned", "created_new", "no_assignment"):
        parts.append(str(reason))

    if isinstance(selected_candidate, dict):
        pick_reason = str(candidate_debug_reason(selected_candidate, selected_track_id=selected_candidate.get("object_id", None)) or "")
        if pick_reason and pick_reason != "selected":
            parts.append(str(pick_reason))

    if alt_summary:
        parts.append(f"alt={alt_summary}")

    return ",".join(str(x) for x in parts if str(x))


def _candidate_sets_trace(candidate: dict) -> dict:
    if not isinstance(candidate, dict):
        return {}
    trace = candidate.get("sets_trace", None)
    return dict(trace) if isinstance(trace, dict) else {}


def _resolve_neighbor_sets_ctx(assoc_out) -> dict:
    dbg = getattr(assoc_out, "debug", None)
    if not isinstance(dbg, dict):
        return {}
    extra = dbg.get("extra", {}) or {}
    ns = extra.get("ns", {}) or {}
    ctx = ns.get("ctx", None)
    return dict(ctx) if isinstance(ctx, dict) else {}


def _neighbor_sets_context_cfg(config: dict | None = None) -> tuple[int, float, float, float, float]:
    assoc_cfg = ((config or {}).get("association", {}) or {})
    scores_cfg = (assoc_cfg.get("scores", {}) or {})
    ns_cfg = (scores_cfg.get("neighbor_sets", {}) or {})
    ctx_cfg = (ns_cfg.get("context", {}) or {})
    match_cfg = (assoc_cfg.get("matching", {}) or {})
    veto_cfg = (match_cfg.get("neighbor_sets_context_veto", {}) or {})
    local_cfg = (veto_cfg.get("local", {}) or {})

    context_k = max(1, int(ctx_cfg.get("k", 6)))
    context_min_p = float(ctx_cfg.get("min_p", ns_cfg.get("min_edge_p", 0.08)))
    context_min_p = max(0.0, min(1.0, context_min_p))
    conf_lambda = float(ns_cfg.get("conf_lambda", 0.25))
    conf_lambda = max(1e-12, conf_lambda)
    expected_mass_target = float(local_cfg.get("expected_mass_target", 0.75))
    expected_mass_target = max(0.0, min(1.0, expected_mass_target))
    expected_topk_scale = float(local_cfg.get("expected_topk_scale", 2.0))
    expected_topk_scale = max(1.0, expected_topk_scale)
    return int(context_k), float(context_min_p), float(conf_lambda), float(expected_mass_target), float(expected_topk_scale)


def _label_list(memory_store, ids: list[int], *, limit: int = 3) -> str:
    out = []
    seen = set()
    for oid in ids or []:
        oid = int(oid)
        if oid in seen:
            continue
        seen.add(oid)
        out.append(get_track_label_short(memory_store, oid, n=4) or f"ID{oid}")
        if len(out) >= int(limit):
            break
    return ",".join(str(x) for x in out if str(x))


def _expected_neighbors_for_object(memory_store, object_id: int, *, config: dict | None = None) -> tuple[list[int], float, int, float]:
    obj = memory_store.get(int(object_id)) if memory_store is not None else None
    g = getattr(obj, "neighbors", None) if obj is not None else None
    if g is None or not getattr(g, "enabled", False):
        return [], 0.0, 0, 0.0

    context_k, context_min_p, conf_lambda, expected_mass_target, expected_topk_scale = _neighbor_sets_context_cfg(config)
    vocab_size = len(memory_store.all_objects()) if memory_store is not None and hasattr(memory_store, "all_objects") else None
    episode_count = int(max(0, int(getattr(g, "episode_count", 0) or 0)))
    maturity = float(1.0 - math.exp(-float(conf_lambda) * float(episode_count)))

    expected: list[int] = []
    probs: list[float] = []
    max_count = max(int(context_k), int(math.ceil(float(context_k) * float(expected_topk_scale))))
    cum_p = 0.0
    for pack in (g.neighbors() or []):
        nid = int(pack.get("dst_id", -1))
        if nid < 0:
            continue
        cooc = int(pack.get("cooc_count", 0) or 0)
        if cooc <= 0:
            continue
        try:
            p = float(g.p_conditional(int(nid), vocab_size=vocab_size))
        except Exception:
            p = 0.0
        p = max(0.0, min(1.0, p))
        if p < float(context_min_p):
            continue
        expected.append(int(nid))
        probs.append(float(p))
        cum_p += float(p)
        if len(expected) >= int(max_count):
            break
        if len(expected) >= int(context_k) and cum_p >= float(expected_mass_target):
            break

    richness = float(1.0 - math.exp(-float(len(expected)) / 2.0)) if expected else 0.0
    mean_p = float(sum(probs) / float(len(probs))) if probs else 0.0
    richness = float(max(0.0, min(1.0, 0.5 * float(richness) + 0.5 * float(mean_p))))
    return expected, float(maturity), int(episode_count), float(richness)


def _candidate_local_context_diagnostics(assoc_out, memory_store, candidate: dict, *, config: dict | None = None) -> dict:
    oid = candidate.get("object_id", None) if isinstance(candidate, dict) else None
    if oid is None or memory_store is None:
        return {
            "ep": 0,
            "mat": 0.0,
            "exp": 0,
            "ker": 0,
            "hit": "0/0",
            "ctx": "off",
            "why": "no_object",
        }

    oid = int(oid)
    obj = memory_store.get(int(oid))
    class_id = int(getattr(obj, "class_id", -1)) if obj is not None else -1
    ns_ctx = _resolve_neighbor_sets_ctx(assoc_out)
    class_pack = (((ns_ctx.get("class_ctx", {}) or {}).get(int(class_id), None)) if isinstance(ns_ctx, dict) else None)

    frame_local_kernel_ids = [
        int(x)
        for x in (candidate.get("frame_local_ctx_kernel_ids", []) or [])
        if x is not None
    ]
    use_frame_local_kernel = bool(frame_local_kernel_ids or ("frame_local_ctx_kernel_ids" in candidate))
    kernel_ids = (
        list(frame_local_kernel_ids)
        if use_frame_local_kernel
        else [int(x) for x in ((class_pack or {}).get("kernel_ids", []) or [])]
    )
    kernel_source = "frame_visible" if use_frame_local_kernel else "sets_kernel"

    expected_ids, maturity, episode_count, richness = _expected_neighbors_for_object(
        memory_store,
        int(oid),
        config=config,
    )
    exp_set = set(int(x) for x in expected_ids)
    ker_set = set(int(x) for x in kernel_ids)
    hit_ids = [int(x) for x in expected_ids if int(x) in ker_set]
    hit_ratio = float(len(hit_ids) / float(len(expected_ids))) if expected_ids else 0.0

    exp_labels = _label_list(memory_store, expected_ids, limit=3)
    ker_labels = _label_list(memory_store, kernel_ids, limit=3)

    if episode_count <= 1 or maturity < 0.35:
        state = "unknown"
        why = f"immature ep={int(episode_count)}"
    elif not expected_ids:
        state = "sparse"
        why = "no_expected_neighbors"
    elif not kernel_ids:
        state = "no_frame_ctx"
        why = f"{kernel_source} expected={exp_labels or '-'}"
    elif not hit_ids:
        state = "contradicts"
        why = f"expected={exp_labels or '-'} ker={ker_labels or '-'} src={kernel_source}"
    elif hit_ratio >= 0.50:
        state = "matches"
        why = f"hit={len(hit_ids)}/{len(expected_ids)} exp={exp_labels or '-'} src={kernel_source}"
    else:
        state = "partial"
        why = f"hit={len(hit_ids)}/{len(expected_ids)} ker={ker_labels or '-'} src={kernel_source}"

    return {
        "ep": int(episode_count),
        "mat": float(maturity),
        "rich": float(richness),
        "exp": int(len(expected_ids)),
        "ker": int(len(kernel_ids)),
        "hit": f"{int(len(hit_ids))}/{int(len(expected_ids))}",
        "ctx": str(state),
        "why": str(why),
    }


def _candidate_veto_diagnostics(candidate: dict, config: dict | None = None) -> tuple[str, str]:
    trace = _candidate_sets_trace(candidate)
    if not trace or not bool(trace.get("enabled", False)):
        return ("off", "sets_off")

    g = trace.get("global", {}) or {}
    c = trace.get("class", {}) or {}
    p = trace.get("policy", {}) or {}

    if str(p.get("veto_reason", "") or ""):
        return ("yes", str(p.get("veto_reason", "") or ""))

    match_cfg = (((config or {}).get("association", {}) or {}).get("matching", {}) or {})
    veto_cfg = (match_cfg.get("neighbor_sets_context_veto", {}) or {})
    veto_enabled = bool(veto_cfg.get("enabled", False))
    if not veto_enabled:
        return ("no", "veto_disabled")

    quality = float(g.get("quality", 0.0) or 0.0)

    if bool(c.get("supported_hit", False)) or bool(c.get("soft_supported_hit", False)):
        return ("no", "supported_candidate")
    if bool(c.get("shortlist_hit", False)):
        return ("no", "shortlist_candidate")

    compat_band = int(bool(candidate.get("compat_band", 0)))
    if compat_band != 0:
        return ("no", "compat_band")

    compat_rel = float(candidate.get("compat_rel", 0.0) or 0.0)
    if compat_rel > float(veto_cfg.get("max_compat_rel", 0.10)):
        return ("no", "compat_rel_too_high")

    score_sets = float(candidate.get("score_sets", 0.0) or 0.0)
    if score_sets > float(veto_cfg.get("max_score_sets", 0.05)):
        return ("no", "sets_score_too_high")

    local_cfg = (veto_cfg.get("local", {}) or {})
    if bool(local_cfg.get("enabled", True)):
        if quality < float(local_cfg.get("min_quality", 0.45)):
            return ("no", "low_local_quality")
        ep = int(candidate.get("local_ctx_episode_count", 0) or 0)
        kernel_size = int(candidate.get("local_ctx_kernel_size", 0) or 0)
        exp_n = int(candidate.get("local_ctx_expected_count", 0) or 0)
        hit_ratio = float(candidate.get("local_ctx_hit_ratio", 1.0) or 1.0)
        if ep >= int(local_cfg.get("min_episodes", 4)) and exp_n >= int(local_cfg.get("min_expected_neighbors", 3)):
            if kernel_size < int(local_cfg.get("min_kernel_size", 3)):
                return ("no", "small_kernel")
            if hit_ratio <= float(local_cfg.get("max_hit_ratio", 0.10)):
                return ("ready", "local_ctx_contradiction")
        if kernel_size < int(local_cfg.get("min_kernel_size", 3)):
            return ("no", "small_kernel")

    if not bool(g.get("global_ok", False)):
        return ("no", "global_not_ok")
    if quality < float(veto_cfg.get("min_quality", 0.60)):
        return ("no", "low_quality")

    kept_count = int(c.get("kept_count", 0) or 0)
    if kept_count <= 0:
        return ("no", "no_kept")
    if kept_count > int(veto_cfg.get("supported_max", 3)):
        return ("no", "too_many_supported")

    pruning = float(c.get("pruning_power", 0.0) or 0.0)
    if pruning < float(veto_cfg.get("min_pruning", 0.35)):
        return ("no", "low_pruning")

    class_strength = float(c.get("class_strength", 0.0) or 0.0)
    if class_strength < float(veto_cfg.get("min_class_strength", 0.50)):
        return ("no", "low_class_strength")

    return ("ready", "would_veto")
