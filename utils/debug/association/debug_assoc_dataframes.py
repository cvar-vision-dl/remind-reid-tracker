# utils/debug/association/debug_assoc_dataframes.py

from __future__ import annotations

import pandas as pd

from ..debug_format import fmt, fmt_ws, safe_float
from .debug_assoc_helpers import (
    candidate_bonus_sets,
    candidate_compat_band,
    candidate_compat_rel,
    candidate_debug_reason,
    candidate_debug_state,
    candidate_known_plausible_keep,
    candidate_hyp_rel,
    candidate_kernel_raw,
    candidate_kernel_hit_count,
    candidate_kernel_hit_ratio,
    candidate_kernel_rel,
    candidate_penalty_sets,
    candidate_quality_sets,
    candidate_score_final,
    candidate_score_known,
    candidate_score_sets,
    candidate_score_sim,
    candidate_support_global_sets,
    candidate_support_local_sets,
    candidate_support_sets,
    candidate_sets_class_summary,
    candidate_sets_ctx_summary,
    candidate_sets_policy_summary,
    decision_letter,
    diag_pack,
    find_candidate_by_track_id,
    pick_focus_track_id,
    pick_selected_track_id,
    rank_key,
    safe_best_score_map,
)
from ..debug_entity_helpers import (
    assignment_tokens,
    det_local,
    get_track_class_name,
    get_track_label_short,
    pair_label,
)
from .debug_assoc_core import (
    _alt_candidate_summary,
    _candidate_local_context_diagnostics,
    _candidate_sets_trace,
    _candidate_veto_diagnostics,
    _report_decision_summary,
)


def _final_score_by_object_id(candidates: list[dict]) -> dict[int, float]:
    out = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        oid = candidate.get("object_id", None)
        if oid is None:
            continue
        oid = int(oid)
        out[oid] = float(candidate_score_final(candidate))
    return {int(oid): float(score) for oid, score in out.items()}


def _rank_key_post_dist(candidate: dict, final_score_by_object_id: dict[int, float]):
    scores = candidate.get("scores", {}) or {}
    obj = scores.get("object", {}) or {}
    bg = scores.get("background", {}) or {}

    oid = candidate.get("object_id", None)
    oid = int(oid) if oid is not None else None
    s_final2 = (
        float(final_score_by_object_id.get(int(oid), candidate_score_final(candidate)))
        if oid is not None
        else 0.0
    )
    best_obj = safe_best_score_map(obj)
    bg_c = safe_float(bg.get("combined", None), default=0.0)
    return (float(s_final2), float(best_obj), float(bg_c))


def assoc_output_to_dataframe(assoc_out, memory_store, det_id_to_local=None, topk=5, config: dict | None = None):
    """Main candidates table by detection."""
    rows = []
    reports = getattr(assoc_out, "reports_by_det_id", {}) or {}

    cfg = (config or {}) if isinstance(config, dict) else {}
    assoc_cfg = (cfg.get("association", {}) or {}) if isinstance(cfg, dict) else {}
    match_cfg = (assoc_cfg.get("matching", {}) or {}) if isinstance(assoc_cfg, dict) else {}

    hung_cfg = (match_cfg.get("hungarian", {}) or {}) if isinstance(match_cfg, dict) else {}
    gate_by_match_thr = bool(hung_cfg.get("gate_by_match_thr", True))
    gate_by_min_match = bool(hung_cfg.get("gate_by_min_match_score", True))

    match_thr = float((match_cfg.get("match_thr", 0.0)) if isinstance(match_cfg, dict) else 0.0)
    upd_cfg = (cfg.get("update", {}) or {}) if isinstance(cfg, dict) else {}
    min_match_score = float(upd_cfg.get("min_match_score", 0.0))

    # Cuando global_ok es true, Hungarian desactiva el gate por match_thr
        # and uses a softer minimum threshold.
    soft_mode = False
    soft_thr = 0.0

    for det_id, rep in reports.items():
        det_id = int(det_id)
        cands = getattr(rep, "candidates", None) or []

        d_final = diag_pack(rep, which="final")
        conf = d_final.get("confidence", "NA")

        if not cands:
            rows.append(
                {
                    "class": None,
                    "det": det_local(det_id, det_id_to_local),
                    "pair": f"{det_local(det_id, det_id_to_local)}-?",
                }
            )
            continue

        final2_by_oid = _final_score_by_object_id(cands)
        ranked_all = sorted(cands, key=lambda c: _rank_key_post_dist(c, final2_by_oid), reverse=True)
        selected_track_id = pick_selected_track_id(rep)
        ranked = ranked_all[: max(1, int(topk))]

        for c in ranked:
            scores = c.get("scores", {}) or {}
            obj = scores.get("object", {}) or {}
            bg = scores.get("background", {}) or {}
            parts = scores.get("parts", {}) or {}

            track_id = c.get("object_id", None)
            oid_int = int(track_id) if track_id is not None else None
            is_selected = bool(
                selected_track_id is not None
                and oid_int is not None
                and int(oid_int) == int(selected_track_id)
            )

            row = {
                "class": (get_track_class_name(memory_store, track_id) or "?").upper(),
                "det": det_local(det_id, det_id_to_local),
                "pair": pair_label(det_id, track_id, memory_store, det_id_to_local),
                "sel": "*" if is_selected else "",
                "st": str(candidate_debug_state(c, selected_track_id=selected_track_id)),
                "why": str(candidate_debug_reason(c, selected_track_id=selected_track_id)),
                "f": decision_letter(getattr(rep, "final_decision", "")),
                "S_sim": fmt(candidate_score_sim(c)),
                "S_sets": fmt(candidate_score_sets(c)),
                "B_sets": fmt(candidate_bonus_sets(c)),
                "P_sets": fmt(candidate_penalty_sets(c)),
                "U_sets": fmt(candidate_support_sets(c)),
                "U_loc": fmt(candidate_support_local_sets(c)),
                "U_glob": fmt(candidate_support_global_sets(c)),
                "Q_sets": fmt(candidate_quality_sets(c)),
                "C_rel": fmt(candidate_compat_rel(c)),
                "C_band": int(candidate_compat_band(c)),
                "K_rel": fmt(candidate_kernel_rel(c)),
                "H_rel": fmt(candidate_hyp_rel(c)),
                "S_final": fmt(candidate_score_final(c)),
                "S_known": fmt(candidate_score_known(c)),
                "S_final2": fmt(final2_by_oid.get(int(oid_int), candidate_score_final(c)) if oid_int is not None else 0.0),
                "KP_keep": int(candidate_known_plausible_keep(c)),
                # Quick indicators: would pure similarity pass Hungarian gating?
                "G_thr": int((not gate_by_match_thr) or soft_mode or (candidate_score_sim(c) >= match_thr)),
                "G_min": int((not gate_by_min_match) or (candidate_score_sim(c) >= (soft_thr if soft_mode else min_match_score))),
                "conf_f": conf,
                "bg": fmt_ws(bg.get("combined", None)),
                "bgp": fmt_ws(bg.get("partials", None)),
                "bg_i": fmt_ws(bg.get("inner", None)),
                "bg_o": fmt_ws(bg.get("outer", None)),
                "bgp_i": fmt_ws(bg.get("partials_inner", None)),
                "bgp_o": fmt_ws(bg.get("partials_outer", None)),
            }

            if "global" in obj:
                row["og"] = fmt_ws(obj.get("global"))
            if "global_trimmed" in obj:
                row["ogt"] = fmt_ws(obj.get("global_trimmed"))

            if "kmeans" in parts:
                row["pk"] = fmt_ws(parts.get("kmeans"))
            if "attention" in parts:
                row["pa"] = fmt_ws(parts.get("attention"))

            rows.append(row)

    return pd.DataFrame(rows)


def assoc_diagnostics_output_to_dataframe(assoc_out, memory_store, det_id_to_local=None, config: dict | None = None):
    """Diagnostics table by detection: SIM vs FINAL + final decision."""
    rows = []
    reports = getattr(assoc_out, "reports_by_det_id", {}) or {}

    cfg = (config or {}) if isinstance(config, dict) else {}
    assoc_cfg = (cfg.get("association", {}) or {}) if isinstance(cfg, dict) else {}
    match_cfg = (assoc_cfg.get("matching", {}) or {}) if isinstance(assoc_cfg, dict) else {}
    for det_id, rep in reports.items():
        det_id = int(det_id)
        cands = getattr(rep, "candidates", None) or []

        d_sim = diag_pack(rep, which="sim")
        d_final = diag_pack(rep, which="final")

        f_dec = decision_letter(getattr(rep, "final_decision", ""))
        f_oid = getattr(rep, "final_object_id", None)

        s_sel_final = None
        s_sel_known = None
        s_sel_sim = None
        s_sel_sim_base = 0.0
        st_sel = None
        b_sel = 0.0
        p_sel = 0.0
        u_sel = 0.0
        q_sel = 0.0
        c_rel_sel = 0.0
        c_band_sel = 0
        k_rel_sel = 0.0
        h_rel_sel = 0.0
        q_obj_sel = 0.0
        q_bg_sel = 0.0
        q_pt_sel = 0.0
        qe_obj_sel = 0.0
        qe_bg_sel = 0.0
        qe_pt_sel = 0.0
        w_obj_sel = 0.0
        w_bg_sel = 0.0
        w_bgp_sel = 0.0
        w_pt_sel = 0.0
        bdb_sel = 0.0
        d_sel = 0.0
        bd_sel = 0.0
        sf2_sel = 0.0

        if cands:
            final2_by_oid = _final_score_by_object_id(cands)
            ranked = sorted(cands, key=lambda c: _rank_key_post_dist(c, final2_by_oid), reverse=True)
            selected_track_id = pick_selected_track_id(rep)
            focus_track_id = pick_focus_track_id(rep, ranked)
            sel_cand = find_candidate_by_track_id(cands, focus_track_id) or ranked[0]
            alt_summary = _alt_candidate_summary(ranked, selected_track_id, memory_store)
            decision_summary = _report_decision_summary(
                rep,
                selected_candidate=find_candidate_by_track_id(cands, selected_track_id),
                alt_summary=alt_summary,
            )
            s_sel_final = candidate_score_final(sel_cand)
            s_sel_known = candidate_score_known(sel_cand)
            s_sel_sim = candidate_score_sim(sel_cand)
            st_sel = candidate_score_sets(sel_cand)
            b_sel = candidate_bonus_sets(sel_cand)
            p_sel = candidate_penalty_sets(sel_cand)
            u_sel = candidate_support_sets(sel_cand)
            q_sel = candidate_quality_sets(sel_cand)
            c_rel_sel = candidate_compat_rel(sel_cand)
            c_band_sel = candidate_compat_band(sel_cand)
            k_rel_sel = candidate_kernel_rel(sel_cand)
            h_rel_sel = candidate_hyp_rel(sel_cand)
            track_id = sel_cand.get("object_id", None)
            tid = int(track_id) if track_id is not None else None
            sf2_sel = float(final2_by_oid.get(int(tid), float(s_sel_final))) if tid is not None else float(s_sel_final)

            rows.append(
                {
                    "class": (get_track_class_name(memory_store, track_id) or "?").upper(),
                    "det": det_local(det_id, det_id_to_local),
                    "pair": pair_label(det_id, track_id, memory_store, det_id_to_local),
                    "f": f_dec,
                    "why": str(decision_summary),
                    "pick": str(candidate_debug_reason(sel_cand, selected_track_id=selected_track_id)),
                    "alt": str(alt_summary),
                    "S_sim": fmt(s_sel_sim),
                    "S_sets": fmt(st_sel),
                    "B_sets": fmt(b_sel),
                    "P_sets": fmt(p_sel),
                    "U_sets": fmt(u_sel),
                    "Q_sets": fmt(q_sel),
                    "C_rel": fmt(c_rel_sel),
                    "C_band": int(c_band_sel),
                    "K_rel": fmt(k_rel_sel),
                    "H_rel": fmt(h_rel_sel),
                    "S_final": fmt(s_sel_final),
                    "S_known": fmt(s_sel_known),
                    "S_final2": fmt(sf2_sel),
                    "KP_keep": int(candidate_known_plausible_keep(sel_cand)),
                    "sim_status": str(d_sim.get("status", "")),
                    "sim_reason": str(d_sim.get("reason", "")),
                    "sim_gap": d_sim.get("gap", "NA"),
                    "sim_conf": d_sim.get("confidence", "NA"),
                    "fin_status": str(d_final.get("status", "")),
                    "fin_reason": str(d_final.get("reason", "")),
                    "fin_gap": d_final.get("gap", "NA"),
                    "fin_conf": d_final.get("confidence", "NA"),
                    "f_oid": int(f_oid) if f_oid is not None else "NA",
                }
            )
            continue

        rows.append(
            {
                "class": None,
                "det": det_local(det_id, det_id_to_local),
                "pair": f"{det_local(det_id, det_id_to_local)}-?",
                "f": f_dec,
                "why": str(_report_decision_summary(rep, selected_candidate=None, alt_summary="")),
                "pick": "",
                "alt": "",
                "S_sim": fmt(s_sel_sim),
                "S_sets": fmt(st_sel),
                "B_sets": fmt(b_sel),
                "P_sets": fmt(p_sel),
                "U_sets": fmt(u_sel),
                "Q_sets": fmt(q_sel),
                "C_rel": fmt(c_rel_sel),
                "C_band": int(c_band_sel),
                "K_rel": fmt(k_rel_sel),
                "H_rel": fmt(h_rel_sel),
                "S_final": fmt(s_sel_final),
                "S_known": fmt(s_sel_known),
                "S_final2": fmt(sf2_sel),
                "KP_keep": 1,
                "sim_status": str(d_sim.get("status", "")),
                "sim_reason": str(d_sim.get("reason", "")),
                "sim_gap": d_sim.get("gap", "NA"),
                "sim_conf": d_sim.get("confidence", "NA"),
                "fin_status": str(d_final.get("status", "")),
                "fin_reason": str(d_final.get("reason", "")),
                "fin_gap": d_final.get("gap", "NA"),
                "fin_conf": d_final.get("confidence", "NA"),
                "f_oid": "NA",
            }
        )

    return pd.DataFrame(rows)


def assoc_similarity_details_to_dataframe(assoc_out, memory_store, det_id_to_local=None, topk=3):
    """Separate table for inspecting the score_sim breakdown by channel."""
    rows = []
    reports = getattr(assoc_out, "reports_by_det_id", {}) or {}

    for det_id, rep in reports.items():
        det_id = int(det_id)
        cands = getattr(rep, "candidates", None) or []
        if not cands:
            continue

        ranked = sorted(
            [c for c in cands if isinstance(c, dict)],
            key=lambda c: float(c.get("score_final", c.get("score_sim", 0.0))),
            reverse=True,
        )[: max(1, int(topk))]

        for c in ranked:
            track_id = c.get("object_id", None)
            scores = c.get("scores", {}) if isinstance(c.get("scores", None), dict) else {}
            obj = scores.get("object", {}) if isinstance(scores.get("object", None), dict) else {}
            bg = scores.get("background", {}) if isinstance(scores.get("background", None), dict) else {}
            parts = scores.get("parts", {}) if isinstance(scores.get("parts", None), dict) else {}
            row = {
                "class": (get_track_class_name(memory_store, track_id) or "?").upper(),
                "det": det_local(det_id, det_id_to_local),
                "pair": pair_label(det_id, track_id, memory_store, det_id_to_local),
                "S_sim": fmt(candidate_score_sim(c)),
                "Qobj": fmt(c.get("quality_obj", 0.0)),
                "Qproto": fmt(c.get("quality_obj_proto", 0.0)),
                "Qjoint": fmt(c.get("quality_obj_joint", 0.0)),
                "Qeff": fmt(c.get("quality_eff_obj", 0.0)),
                "Wobj": fmt(c.get("weight_eff_obj", 0.0)),
                "Obj": fmt(c.get("score_obj_collapsed", None)),
                "Bg": fmt(c.get("score_bg_collapsed", None)),
                "BgP": fmt(c.get("score_bgp_collapsed", None)),
                "Parts": fmt(c.get("score_parts_collapsed", None)),
                "Base": fmt(c.get("score_sim_base", None)),
                "ObjG": fmt_ws(obj.get("global")),
                "ObjGT": fmt_ws(obj.get("global_trimmed")),
                "BgWS": fmt_ws(bg.get("combined", None)),
                "BgPWS": fmt_ws(bg.get("partials", None)),
                "BgI": fmt_ws(bg.get("inner", None)),
                "BgO": fmt_ws(bg.get("outer", None)),
                "BgPI": fmt_ws(bg.get("partials_inner", None)),
                "BgPO": fmt_ws(bg.get("partials_outer", None)),
                "PkWS": fmt_ws(parts.get("kmeans")),
                "PaWS": fmt_ws(parts.get("attention")),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def neighbor_sets_candidates_to_dataframe(assoc_out, memory_store, det_id_to_local=None, topk=5):
    """Dedicated table for how `sets` affects candidates by detection."""
    rows = []
    reports = getattr(assoc_out, "reports_by_det_id", {}) or {}

    for det_id, rep in reports.items():
        det_id = int(det_id)
        cands = [c for c in (getattr(rep, "candidates", None) or []) if isinstance(c, dict)]
        if not cands:
            continue

        ranked = sorted(cands, key=rank_key, reverse=True)[: max(1, int(topk))]
        sel_track_id = pick_focus_track_id(rep, ranked)
        fin_dec = decision_letter(getattr(rep, "final_decision", ""))

        for c in ranked:
            track_id = c.get("object_id", None)
            oid = int(track_id) if track_id is not None else None
            rows.append(
                {
                    "class": (get_track_class_name(memory_store, track_id) or "?").upper(),
                    "det": det_local(det_id, det_id_to_local),
                    "pair": pair_label(det_id, track_id, memory_store, det_id_to_local),
                    "fin": fin_dec,
                    "sel": int(oid is not None and sel_track_id is not None and int(oid) == int(sel_track_id)),
                    "S_sim": fmt(candidate_score_sim(c)),
                    "S_sets": fmt(candidate_score_sets(c)),
                    "B_sets": fmt(candidate_bonus_sets(c)),
                    "P_sets": fmt(candidate_penalty_sets(c)),
                    "U_sets": fmt(candidate_support_sets(c)),
                    "U_loc": fmt(candidate_support_local_sets(c)),
                    "U_glob": fmt(candidate_support_global_sets(c)),
                    "Q_sets": fmt(candidate_quality_sets(c)),
                    "C_rel": fmt(candidate_compat_rel(c)),
                    "C_band": int(candidate_compat_band(c)),
                    "K_abs": fmt(candidate_kernel_raw(c)),
                    "K_hit": int(candidate_kernel_hit_count(c)),
                    "K_cov": fmt(candidate_kernel_hit_ratio(c)),
                    "K_rel": fmt(candidate_kernel_rel(c)),
                    "H_rel": fmt(candidate_hyp_rel(c)),
                    "S_final": fmt(candidate_score_final(c)),
                    "NS_ctx": candidate_sets_ctx_summary(c),
                    "NS_cls": candidate_sets_class_summary(c),
                    "NS_pol": candidate_sets_policy_summary(c),
                }
            )

    return pd.DataFrame(rows)


def context_veto_candidates_to_dataframe(
    assoc_out,
    memory_store,
    det_id_to_local=None,
    topk=5,
    config: dict | None = None,
):
    rows = []
    reports = getattr(assoc_out, "reports_by_det_id", {}) or {}
    for det_id, rep in reports.items():
        det_id = int(det_id)
        cands = [c for c in (getattr(rep, "candidates", None) or []) if isinstance(c, dict)]
        if not cands:
            continue

        ranked = sorted(cands, key=rank_key, reverse=True)[: max(1, int(topk))]
        sel_track_id = pick_focus_track_id(rep, ranked)
        fin_dec = decision_letter(getattr(rep, "final_decision", ""))

        for c in ranked:
            track_id = c.get("object_id", None)
            oid = int(track_id) if track_id is not None else None
            trace = _candidate_sets_trace(c)
            g = trace.get("global", {}) or {}
            cls = trace.get("class", {}) or {}
            veto_flag, veto_why = _candidate_veto_diagnostics(
                c,
                config=config,
            )
            rows.append(
                {
                    "class": (get_track_class_name(memory_store, track_id) or "?").upper(),
                    "det": det_local(det_id, det_id_to_local),
                    "pair": pair_label(det_id, track_id, memory_store, det_id_to_local),
                    "fin": fin_dec,
                    "sel": int(oid is not None and sel_track_id is not None and int(oid) == int(sel_track_id)),
                    "veto": str(veto_flag),
                    "why": str(veto_why),
                    "kp": int(candidate_known_plausible_keep(c)),
                    "sup": int(bool(cls.get("supported_hit", False))),
                    "ss": int(bool(cls.get("soft_supported_hit", False))),
                    "sh": int(bool(cls.get("shortlist_hit", False))),
                    "keep": f"{int(cls.get('kept_count', 0) or 0)}/{int(cls.get('total_count', 0) or 0)}",
                    "q": fmt(g.get("quality", None)),
                    "pr": fmt(cls.get("pruning_power", None)),
                    "cs": fmt(cls.get("class_strength", None)),
                    "cr": fmt(c.get("compat_rel", None)),
                    "s_sets": fmt(c.get("score_sets", None)),
                    "gate": str((trace.get("policy", {}) or {}).get("gate_reason", "") or ""),
                    "veto_raw": str((trace.get("policy", {}) or {}).get("veto_reason", "") or ""),
                }
            )

    return pd.DataFrame(rows)


def local_context_candidates_to_dataframe(
    assoc_out,
    memory_store,
    det_id_to_local=None,
    topk=5,
    config: dict | None = None,
):
    rows = []
    reports = getattr(assoc_out, "reports_by_det_id", {}) or {}

    for det_id, rep in reports.items():
        det_id = int(det_id)
        cands = [c for c in (getattr(rep, "candidates", None) or []) if isinstance(c, dict)]
        if not cands:
            continue

        ranked = sorted(cands, key=rank_key, reverse=True)[: max(1, int(topk))]
        sel_track_id = pick_focus_track_id(rep, ranked)
        fin_dec = decision_letter(getattr(rep, "final_decision", ""))

        for c in ranked:
            track_id = c.get("object_id", None)
            oid = int(track_id) if track_id is not None else None
            diag = _candidate_local_context_diagnostics(
                assoc_out,
                memory_store,
                c,
                config=config,
            )
            rows.append(
                {
                    "class": (get_track_class_name(memory_store, track_id) or "?").upper(),
                    "det": det_local(det_id, det_id_to_local),
                    "pair": pair_label(det_id, track_id, memory_store, det_id_to_local),
                    "fin": fin_dec,
                    "sel": int(oid is not None and sel_track_id is not None and int(oid) == int(sel_track_id)),
                    "ep": int(diag.get("ep", 0)),
                    "mat": fmt(diag.get("mat", 0.0)),
                    "rich": fmt(diag.get("rich", 0.0)),
                    "exp": int(diag.get("exp", 0)),
                    "ker": int(diag.get("ker", 0)),
                    "hit": str(diag.get("hit", "0/0")),
                    "ctx": str(diag.get("ctx", "")),
                    "why": str(diag.get("why", "")),
                }
            )

    return pd.DataFrame(rows)


def neighbor_sets_output_to_dataframes(neigh_sets_out, memory_store, det_id_to_local=None):
    """Convert neighbor-sets output to (df_sets, df_obj_priors)."""
    out = neigh_sets_out if isinstance(neigh_sets_out, dict) else {}
    core = out.get("core", {}) if isinstance(out.get("core", None), dict) else {}
    dbg = out.get("debug", {}) if isinstance(out.get("debug", None), dict) else {}

    hyps = dbg.get("set_hypotheses", None)
    if not isinstance(hyps, list):
        hyps = out.get("set_hypotheses", []) or []

    pri = core.get("prior_by_oid", {}) or {}
    if not isinstance(pri, dict):
        pri = {}

    supp_sum = dbg.get("object_support_sum", {}) or {}
    if not isinstance(supp_sum, dict):
        supp_sum = {}

    rows_sets = []
    for i, h in enumerate(hyps, start=1):
        if not isinstance(h, dict):
            continue

        det_oid = []
        for p in (h.get("pairs", []) or []):
            if not isinstance(p, dict):
                continue
            det_ids = p.get("det_ids", ()) or ()
            obj_ids = p.get("object_ids", ()) or ()
            try:
                det_ids = [int(x) for x in det_ids]
                obj_ids = [int(x) for x in obj_ids]
            except Exception:
                continue
            if len(det_ids) != len(obj_ids) or not det_ids:
                continue
            det_oid.extend(list(zip(det_ids, obj_ids)))

        obj_ids_all = [int(x) for x in (h.get("object_ids", []) or [])]
        labels = []
        for oid in obj_ids_all:
            lbl = get_track_label_short(memory_store, int(oid), n=3)
            labels.append(lbl if lbl else f"ID{int(oid)}")

        rows_sets.append(
            {
                "rank": int(i),
                "score_sets": fmt(h.get("score_sets", 0.0)),
                "cov": fmt(h.get("coverage", 0.0)),
                "cov_eff": fmt(h.get("coverage_eff", h.get("coverage", 0.0))),
                "k": int(h.get("k", len(obj_ids_all))),
                "size": fmt(h.get("size_util", None)),
                "info": fmt(h.get("class_info", None)),
                "stab": fmt(h.get("class_stability", None)),
                "dens": fmt(h.get("density", None)),
                "dens_v": int(bool(h.get("density_valid", False))),
                "edge_cov": fmt(h.get("edge_cov", None)),
                "node_cov": fmt(h.get("node_cov", None)),
                "conn_f": fmt(h.get("conn_factor", None)),
                "ctx": fmt(h.get("ctx_cov_eff", None)),
                "mat_c": fmt(h.get("maturity_coh", None)),
                "mat_r": fmt(h.get("maturity_rel", None)),
                "logC": fmt(h.get("class_logC_sum", None)),
                "excl": fmt(h.get("exclusivity", None)),
                "excl_v": int(bool(h.get("exclusivity_valid", False))),
                "maturity": fmt(h.get("mean_maturity", None)),
                "assign": assignment_tokens(det_oid, memory_store, det_id_to_local=det_id_to_local),
                "objs": ", ".join(labels),
                # Packed columns (same info, less width)
                "cov2": f"{fmt(h.get('coverage', 0.0))}/{fmt(h.get('coverage_eff', h.get('coverage', 0.0)))}",
                "mat": f"{fmt(h.get('maturity_rel', None))}({fmt(h.get('maturity_coh', None))})",
                "covg": f"{fmt(h.get('edge_cov', None))}/{fmt(h.get('node_cov', None))}v{int(bool(h.get('density_valid', False)))}",
                "excl2": f"{fmt(h.get('exclusivity', None))}v{int(bool(h.get('exclusivity_valid', False)))}",
            }
        )

    rows_obj = []
    items = [(int(k), safe_float(v, 0.0)) for k, v in pri.items()]
    items.sort(key=lambda kv: float(kv[1]), reverse=True)

    for oid, s in items:
        rows_obj.append(
            {
                "obj_id": int(oid),
                "label": get_track_label_short(memory_store, int(oid), n=3) or f"ID{int(oid)}",
                "prior": fmt(s),
                "supp_sum": fmt(supp_sum.get(int(oid), None)),
            }
        )

    df_sets = pd.DataFrame(rows_sets)
    df_obj = pd.DataFrame(rows_obj)
    return df_sets, df_obj


def neighbor_sets_class_options_to_dataframes(neigh_sets_out, memory_store):
    out = neigh_sets_out if isinstance(neigh_sets_out, dict) else {}
    dbg = out.get("debug", {}) if isinstance(out.get("debug", None), dict) else {}
    raw = dbg.get("class_options_debug", {}) if isinstance(dbg.get("class_options_debug", None), dict) else {}

    rows = []
    for pack in raw.values():
        if not isinstance(pack, dict):
            continue
        class_id = int(pack.get("class_id", -1))
        kernel_ids = [int(x) for x in (pack.get("kernel_obj_ids", []) or [])]
        det_ids = [int(x) for x in (pack.get("det_ids", []) or [])]
        kernel_lbls = [get_track_label_short(memory_store, int(oid), n=4) or f"ID{int(oid)}" for oid in kernel_ids]
        for row in (pack.get("rows", []) or []):
            if not isinstance(row, dict):
                continue
            oid = int(row.get("object_id", -1))
            details = []
            for item in (row.get("details", []) or []):
                if not isinstance(item, dict):
                    continue
                kid = int(item.get("kernel_id", -1))
                klbl = get_track_label_short(memory_store, int(kid), n=4) or f"ID{int(kid)}"
                p = fmt(item.get("p", 0.0))
                mark = "*" if bool(item.get("hit", False)) else ""
                details.append(f"{klbl}:{p}{mark}")
            rows.append(
                {
                    "class": (get_track_class_name(memory_store, oid) or "?").upper(),
                    "dets": ",".join(str(int(x)) for x in det_ids),
                    "obj": get_track_label_short(memory_store, int(oid), n=4) or f"ID{int(oid)}",
                    "supp": fmt(row.get("support", 0.0)),
                    "hit": f"{int(row.get('hit_count', 0) or 0)}/{int(row.get('kernel_count', 0) or 0)}",
                    "cov": fmt(row.get("hit_ratio", 0.0)),
                    "kernel": ",".join(kernel_lbls),
                    "detail": ",".join(details),
                    "class_id": int(class_id),
                }
            )

    return pd.DataFrame(rows)
