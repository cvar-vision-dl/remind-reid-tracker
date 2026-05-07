from __future__ import annotations

from ..debug_format import fmt, safe_float


def safe_best_score_map(score_map: dict | None) -> float:
    """Devuelve el máximo score de un mapa {k: score}, con safe_float sobre valores."""
    if not score_map:
        return 0.0
    vals = []
    for v in score_map.values():
        if v is None:
            continue
        vals.append(safe_float(v, default=None))
    vals = [float(x) for x in vals if x is not None]
    return float(max(vals)) if vals else 0.0


def candidate_score_sim(candidate: dict) -> float:
    """Extrae score_sim de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("score_sim", 0.0), default=0.0)


def candidate_score_sets(candidate: dict) -> float:
    """Extrae score_sets de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("score_sets", 0.0), default=0.0)


def candidate_bonus_sets(candidate: dict) -> float:
    """Extrae bonus_sets de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("bonus_sets", 0.0), default=0.0)


def candidate_penalty_sets(candidate: dict) -> float:
    """Extrae penalty_sets de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("penalty_sets", 0.0), default=0.0)


def candidate_support_sets(candidate: dict) -> float:
    """Extrae support_sets de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("support_sets", 0.0), default=0.0)


def candidate_support_local_sets(candidate: dict) -> float:
    """Extrae support_local_sets de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("support_local_sets", 0.0), default=0.0)


def candidate_support_global_sets(candidate: dict) -> float:
    """Extrae support_global_sets de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("support_global_sets", 0.0), default=0.0)


def candidate_quality_sets(candidate: dict) -> float:
    """Extrae quality_sets de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("quality_sets", 0.0), default=0.0)


def candidate_compat_rel(candidate: dict) -> float:
    """Extrae compat_rel de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("compat_rel", 0.0), default=0.0)


def candidate_kernel_rel(candidate: dict) -> float:
    """Extrae kernel_rel de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("kernel_rel", 0.0), default=0.0)


def candidate_kernel_raw(candidate: dict) -> float:
    """Extrae kernel_raw absoluto de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("kernel_raw", 0.0), default=0.0)


def candidate_kernel_hit_ratio(candidate: dict) -> float:
    """Extrae fracción de kernel que realmente soporta al candidato."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("kernel_hit_ratio", 0.0), default=0.0)


def candidate_kernel_hit_count(candidate: dict) -> int:
    """Extrae nº de hits del kernel sobre el candidato."""
    if not isinstance(candidate, dict):
        return 0
    try:
        return int(candidate.get("kernel_hit_count", 0) or 0)
    except Exception:
        return 0


def candidate_hyp_rel(candidate: dict) -> float:
    """Extrae hyp_rel de candidate."""
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("hyp_rel", 0.0), default=0.0)


def candidate_compat_band(candidate: dict) -> int:
    """Indica si el candidato está dentro de la compatibility band contextual (1/0)."""
    if not isinstance(candidate, dict):
        return 0
    try:
        return int(bool(candidate.get("compat_band", 0)))
    except Exception:
        return 0


def candidate_score_sim_base(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("score_sim_base", 0.0), default=0.0)


def candidate_quality_obj(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("quality_obj", 0.0), default=0.0)


def candidate_quality_bg(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("quality_bg", 0.0), default=0.0)


def candidate_quality_parts(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("quality_parts", 0.0), default=0.0)


def candidate_quality_eff_obj(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("quality_eff_obj", 0.0), default=0.0)


def candidate_quality_eff_bg(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("quality_eff_bg", 0.0), default=0.0)


def candidate_quality_eff_parts(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("quality_eff_parts", 0.0), default=0.0)


def candidate_weight_eff_obj(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("weight_eff_obj", 0.0), default=0.0)


def candidate_weight_eff_bg(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("weight_eff_bg", 0.0), default=0.0)


def candidate_weight_eff_bgp(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("weight_eff_bgp", 0.0), default=0.0)


def candidate_weight_eff_parts(candidate: dict) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return safe_float(candidate.get("weight_eff_parts", 0.0), default=0.0)


def candidate_score_final(candidate: dict) -> float:
    """Extrae score_final; si no existe, cae a score_sim."""
    if not isinstance(candidate, dict):
        return 0.0
    if candidate.get("score_final", None) is not None:
        return safe_float(candidate.get("score_final", 0.0), default=0.0)
    return safe_float(candidate.get("score_sim", 0.0), default=0.0)


def candidate_score_known(candidate: dict) -> float:
    """
    Score "conjunto conocido" usado para temporales:
    max(score_final, score_sim + bonus_sets_raw).
    """
    if not isinstance(candidate, dict):
        return 0.0
    score_known = candidate.get("score_known", None)
    if score_known is not None:
        return safe_float(score_known, default=0.0)
    score_sim = candidate_score_sim(candidate)
    bonus_sets = safe_float(candidate.get("bonus_sets_raw", candidate.get("bonus_sets", 0.0)), default=0.0)
    score_final = candidate_score_final(candidate)
    return float(max(score_final, score_sim + bonus_sets))


def candidate_decision_keep(candidate: dict) -> int:
    """Indica si el candidato es elegible para decision final (1/0)."""
    if not isinstance(candidate, dict):
        return 0
    v = candidate.get("decision_keep", None)
    if v is None:
        return 0
    try:
        return int(bool(v))
    except Exception:
        return 0


def candidate_ctx_keep(candidate: dict) -> int:
    """Indica si el candidato está permitido por gating de contexto (neighbor-sets shortlist) (1/0)."""
    if not isinstance(candidate, dict):
        return 1
    v = candidate.get("ctx_keep", None)
    if v is None:
        return 1
    try:
        return int(bool(v))
    except Exception:
        return 1


def candidate_policy_trace(candidate: dict) -> dict:
    if not isinstance(candidate, dict):
        return {}
    trace = candidate.get("sets_trace", None)
    if not isinstance(trace, dict):
        return {}
    policy = trace.get("policy", None)
    return dict(policy) if isinstance(policy, dict) else {}


def candidate_ctx_reason(candidate: dict) -> str:
    policy = candidate_policy_trace(candidate)
    return str(policy.get("ctx_reason", "") or "")


def candidate_gate_reason(candidate: dict) -> str:
    policy = candidate_policy_trace(candidate)
    return str(policy.get("gate_reason", "") or "")


def candidate_veto_reason(candidate: dict) -> str:
    policy = candidate_policy_trace(candidate)
    return str(policy.get("veto_reason", "") or "")


def candidate_known_plausible_reason(candidate: dict) -> str:
    policy = candidate_policy_trace(candidate)
    return str(policy.get("known_plausible_reason", "") or "")


def candidate_known_plausible_keep(candidate: dict) -> int:
    """Indica si el candidato permanece plausible para ambiguedad temporal (1/0)."""
    if not isinstance(candidate, dict):
        return 1
    v = candidate.get("known_plausible_keep", None)
    if v is None:
        return 1
    try:
        return int(bool(v))
    except Exception:
        return 1


def _pretty_rule_name(raw: str) -> str:
    s = str(raw or "").strip().upper()
    if not s:
        return ""

    mapping = {
        "KNOWN_OK": "known_ok",
        "OUTSIDE_CTX": "ctx_veto",
        "BLOCK_MIN_MATCH": "below_min_match",
        "BLOCK_MATCH_THR": "below_match_thr",
        "PASS_MATCH_THR": "pass_match_thr",
        "SETS_RESCUE": "sets_rescue",
        "CLASS_SHORTLIST_HIT": "class_shortlist",
        "CLASS_SHORTLIST_BLOCK": "outside_class_shortlist",
    }
    if s in mapping:
        return str(mapping[s])
    return str(s.lower())


def candidate_debug_state(candidate: dict, *, selected_track_id=None) -> str:
    if not isinstance(candidate, dict):
        return ""

    oid = candidate.get("object_id", None)
    if selected_track_id is not None and oid is not None and int(oid) == int(selected_track_id):
        return "SEL"

    if int(candidate_known_plausible_keep(candidate)) != 1:
        return "VETO"

    if int(candidate_decision_keep(candidate)) == 1:
        gate_reason = str(candidate_gate_reason(candidate))
        if gate_reason == "SETS_RESCUE":
            return "RESC"
        return "ELIG"

    gate_reason = str(candidate_gate_reason(candidate))
    if gate_reason == "BLOCK_MIN_MATCH":
        return "MIN"
    if gate_reason == "BLOCK_MATCH_THR":
        return "THR"
    return "DROP"


def candidate_debug_reason(candidate: dict, *, selected_track_id=None) -> str:
    if not isinstance(candidate, dict):
        return ""

    oid = candidate.get("object_id", None)
    gate_reason = _pretty_rule_name(candidate_gate_reason(candidate))
    veto_reason = _pretty_rule_name(candidate_veto_reason(candidate))
    known_reason = _pretty_rule_name(candidate_known_plausible_reason(candidate))
    ctx_reason = _pretty_rule_name(candidate_ctx_reason(candidate))

    if selected_track_id is not None and oid is not None and int(oid) == int(selected_track_id):
        extras = []
        if gate_reason in ("soft_shortlist", "sets_rescue"):
            extras.append(gate_reason)
        bonus_sets = float(candidate_bonus_sets(candidate))
        if abs(bonus_sets) > 1e-3:
            extras.append(f"sets{bonus_sets:+.2f}")
        return "selected" if not extras else ("selected," + ",".join(extras))

    if int(candidate_known_plausible_keep(candidate)) != 1:
        return str(veto_reason or known_reason or "ctx_veto")

    if int(candidate_decision_keep(candidate)) == 1:
        if gate_reason in ("soft_shortlist", "sets_rescue"):
            return str(gate_reason)
        if ctx_reason == "outside_class_shortlist":
            return "eligible_outside_shortlist"
        return "eligible_not_selected"

    if gate_reason:
        return str(gate_reason)
    if ctx_reason:
        return str(ctx_reason)
    return "not_selected"


def candidate_sets_ctx_summary(candidate: dict) -> str:
    if not isinstance(candidate, dict):
        return ""
    return str(candidate.get("sets_ctx_summary", "") or "")


def candidate_sets_class_summary(candidate: dict) -> str:
    if not isinstance(candidate, dict):
        return ""
    return str(candidate.get("sets_class_summary", "") or "")


def candidate_sets_policy_summary(candidate: dict) -> str:
    if not isinstance(candidate, dict):
        return ""
    return str(candidate.get("sets_policy_summary", "") or "")


def rank_key(candidate: dict):
    """Clave de ranking para candidates: (final, best_obj, bg_combined)."""
    scores = candidate.get("scores", {}) or {}
    obj = scores.get("object", {}) or {}
    bg = scores.get("background", {}) or {}

    s_final = candidate_score_final(candidate)
    best_obj = safe_best_score_map(obj)
    bg_c = safe_float(bg.get("combined", None), default=0.0)
    return (float(s_final), float(best_obj), float(bg_c))


def diag_pack(rep, which: str = "sim") -> dict:
    """Empaqueta diagnósticos de matching del report (sim/final) con valores formateados."""
    attr = "match_diag_sim" if str(which).lower().strip() == "sim" else "match_diag_final"
    d = getattr(rep, attr, None)

    if not isinstance(d, dict):
        return {
            "s1": "NA",
            "s2": "NA",
            "gap": "NA",
            "n_close": "NA",
            "status": "",
            "reason": "",
            "confidence": "NA",
        }

    return {
        "s1": fmt(d.get("s1", None)),
        "s2": fmt(d.get("s2", None)),
        "gap": fmt(d.get("gap", None)),
        "n_close": int(d.get("n_close", 0)),
        "status": str(d.get("status", "")),
        "reason": str(d.get("reason", "")),
        "confidence": fmt(d.get("confidence", None)),
    }


def pick_selected_track_id(rep):
    """Devuelve el track_id realmente elegido si la decision final selecciona uno."""
    v = getattr(rep, "final_object_id", None)
    if v is not None:
        return int(v)

    return None


def pick_focus_track_id(rep, ranked_candidates):
    """Devuelve el candidato de referencia para inspeccion visual del reporte."""
    v = pick_selected_track_id(rep)
    if v is not None:
        return int(v)

    d = str(getattr(rep, "final_decision", "") or "").upper().strip()
    if d == "AMBIGUOUS_TRACK":
        score_map = getattr(rep, "ambiguous_candidate_scores", None)
        if isinstance(score_map, dict) and score_map:
            scored = []
            for oid, s in score_map.items():
                try:
                    scored.append((float(s), int(oid)))
                except Exception:
                    continue
            if scored:
                scored.sort(key=lambda x: float(x[0]), reverse=True)
                return int(scored[0][1])

        cand_ids = getattr(rep, "ambiguous_candidate_ids", None) or []
        cand_set = set(int(x) for x in cand_ids if x is not None)
        if cand_set and ranked_candidates:
            supported = []
            for c in ranked_candidates:
                oid = c.get("object_id", None)
                if oid is None:
                    continue
                if int(oid) in cand_set:
                    supported.append(c)
            if supported:
                supported.sort(key=lambda c: float(candidate_score_known(c)), reverse=True)
                oid = supported[0].get("object_id", None)
                return int(oid) if oid is not None else None

    if d in ("PROVISIONAL_NEW", "PROVISIONAL_PARENT"):
        score_map = getattr(rep, "provisional_related_known_scores", None)
        if not isinstance(score_map, dict) or not score_map:
            score_map = getattr(rep, "provisional_support_scores", None)
        if isinstance(score_map, dict) and score_map:
            scored = []
            for oid, s in score_map.items():
                try:
                    scored.append((float(s), int(oid)))
                except Exception:
                    continue
            if scored:
                scored.sort(key=lambda x: float(x[0]), reverse=True)
                return int(scored[0][1])

        cand_ids = getattr(rep, "provisional_related_known_ids", None) or []
        if not cand_ids:
            cand_ids = getattr(rep, "provisional_support_ids", None) or []
        cand_set = set(int(x) for x in cand_ids if x is not None)
        if cand_set and ranked_candidates:
            supported = []
            for c in ranked_candidates:
                oid = c.get("object_id", None)
                if oid is None:
                    continue
                if int(oid) in cand_set:
                    supported.append(c)
            if supported:
                supported.sort(key=lambda c: float(candidate_score_known(c)), reverse=True)
                oid = supported[0].get("object_id", None)
                return int(oid) if oid is not None else None

    if ranked_candidates:
        oid = ranked_candidates[0].get("object_id", None)
        return int(oid) if oid is not None else None

    return None


def find_candidate_by_track_id(cands, track_id):
    """Devuelve el candidate cuyo object_id coincide con track_id."""
    if track_id is None:
        return None
    for c in (cands or []):
        oid = c.get("object_id", None)
        if oid is not None and int(oid) == int(track_id):
            return c
    return None


def decision_letter(dec: str) -> str:
    """Codifica decisión final en una letra para tabla."""
    d = str(dec or "").upper().strip()
    if d == "MATCH":
        return "M"
    if d == "NEW":
        return "N"
    if d == "AMBIGUOUS_TRACK":
        return "T"
    if d == "PROVISIONAL_NEW":
        return "P"
    if d == "PROVISIONAL_PARENT":
        return "P"
    if d == "UNASSIGNED":
        return "U"
    return ""
