from __future__ import annotations

from ..debug_format import fmt
from ..debug_table_utils import compact_columns, print_table_auto
from .debug_assoc_core import _drop_uninformative_assoc_cols
from .debug_assoc_dataframes import (
    assoc_diagnostics_output_to_dataframe,
    assoc_output_to_dataframe,
    assoc_similarity_details_to_dataframe,
)


def print_assoc_table(config, frame_id, assoc_out, memory_store, det_id_to_local=None):
    """Imprime tabla de asociación (candidates) por clase."""
    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    assoc_dbg = (dbg.get("association", {}) or {})
    if not assoc_dbg.get("enabled", True):
        return

    every = max(1, int(assoc_dbg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    show_candidates = bool(assoc_dbg.get("show_candidates_table", True))
    if not show_candidates:
        return

    topk = int(assoc_dbg.get("candidates_topk", 5))
    df = assoc_output_to_dataframe(assoc_out, memory_store, det_id_to_local=det_id_to_local, topk=topk, config=config)

    anchors = getattr(assoc_out, "reliable_anchor_object_ids", None) or []
    show_anchors = bool(assoc_dbg.get("show_reliable_anchor_ids", True))

    if df.empty:
        print(f"\n[DEBUG][Association] frame={frame_id} (empty)")
        return

    if show_anchors:
        extra = ""
        out_dbg = getattr(assoc_out, "debug", None)
        if isinstance(out_dbg, dict):
            ns_ctx = (((out_dbg.get("extra", {}) or {}).get("ns", {}) or {}).get("ctx", {}) or {})
            if isinstance(ns_ctx, dict) and ns_ctx.get("enabled", False):
                extra += (
                    f" ns_ok={int(bool(ns_ctx.get('global_ok', False)))}"
                    f" ns_q={fmt(ns_ctx.get('quality', None))}"
                    f" ns_reason={str(ns_ctx.get('reason',''))}"
                )
        print(f"\n[DEBUG][Association] frame={frame_id} anchors={list(anchors)}{extra}")
    else:
        print(f"\n[DEBUG][Association] frame={frame_id}")

    compact_tables = bool(assoc_dbg.get("compact_tables", True))
    max_text = int(assoc_dbg.get("max_text_chars", 14))
    max_pair = int(assoc_dbg.get("max_pair_chars", 14))

    for cname in sorted(df["class"].dropna().unique()):
        sub = df[df["class"] == cname]
        if compact_tables:
            order = [
                "det",
                "pair",
                "sel",
                "st",
                "f",
                "why",
                "S_sim",
                "S_sets",
                "S_final",
                "S_known",
                "KP_keep",
                "G_thr",
                "G_min",
                "conf_f",
            ]
        else:
            order = [
                "det",
                "pair",
                "sel",
                "st",
                "f",
                "why",
                "S_sim",
                "S_sets",
                "B_sets",
                "C_rel",
                "C_band",
                "P_sets",
                "Q_sets",
                "S_final",
                "S_known",
                "S_final2",
                "KP_keep",
                "G_thr",
                "G_min",
                "conf_f",
                "og",
                "ogt",
                "bg",
                "bgp",
                "pk",
                "pa",
                "bg_i",
                "bg_o",
                "bgp_i",
                "bgp_o",
            ]
        cols = [c for c in order if c in sub.columns]
        cols = _drop_uninformative_assoc_cols(sub, cols)
        if compact_tables:
            sub = compact_columns(
                sub,
                {
                    "pair": max_pair,
                    "class": max_text,
                },
            )
        print(f"\n[CLASS {cname}]")
        print_table_auto(
            sub,
            cols,
            pinned_cols=["det", "pair", "sel", "st", "f"],
            wrap_col="why" if "why" in cols else None,
            col_space=2,
            wrap_sep=",",
        )


def print_assoc_diagnostics_table(config, frame_id, assoc_out, memory_store, det_id_to_local=None):
    """Imprime tabla de diagnósticos (SIM vs FINAL) por clase."""
    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    assoc_dbg = (dbg.get("association", {}) or {})
    if not assoc_dbg.get("enabled", True):
        return

    every = max(1, int(assoc_dbg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    if not assoc_dbg.get("show_diagnostics_table", True):
        return

    df = assoc_diagnostics_output_to_dataframe(assoc_out, memory_store, det_id_to_local=det_id_to_local, config=config)

    anchors = getattr(assoc_out, "reliable_anchor_object_ids", None) or []
    show_anchors = bool(assoc_dbg.get("show_reliable_anchor_ids", True))

    summ = getattr(assoc_out, "frame_summary", {}) or {}
    n_strong = int(summ.get("n_strong", 0))
    n_amb = int(summ.get("n_ambiguous", 0))
    n_weak = int(summ.get("n_weak", 0))

    if df.empty:
        print(f"\n[DEBUG][AssociationDiag] frame={frame_id} (empty)")
        return

    diag_hdr = f"diag strong={n_strong} amb={n_amb} weak={n_weak}"
    if show_anchors:
        print(f"\n[DEBUG][AssociationDiag] frame={frame_id} anchors={list(anchors)} | {diag_hdr}")
    else:
        print(f"\n[DEBUG][AssociationDiag] frame={frame_id} | {diag_hdr}")

    compact_tables = bool(assoc_dbg.get("compact_tables", True))
    max_text = int(assoc_dbg.get("max_text_chars", 14))
    max_pair = int(assoc_dbg.get("max_pair_chars", 14))

    for cname in sorted(df["class"].dropna().unique()):
        sub = df[df["class"] == cname]
        order = [
            "det",
            "pair",
            "f",
            "why",
            "pick",
            "alt",
            "S_sim",
            "S_sets",
            "S_final",
            "S_known",
            "KP_keep",
            "sim_status",
            "sim_gap",
            "sim_conf",
            "fin_status",
            "fin_gap",
            "fin_conf",
            "f_oid",
        ]
        cols = [c for c in order if c in sub.columns]
        cols = _drop_uninformative_assoc_cols(sub, cols)
        if compact_tables:
            sub = compact_columns(
                sub,
                {
                    "pair": max_pair,
                    "sim_status": max_text,
                    "fin_status": max_text,
                    "pick": max_text,
                    "alt": max_text,
                },
            )
        print(f"\n[CLASS {cname}]")
        print_table_auto(
            sub,
            cols,
            pinned_cols=["det", "pair", "f"],
            wrap_col="why" if "why" in cols else None,
            col_space=2,
            wrap_sep=",",
        )


def print_assoc_similarity_details_table(config, frame_id, assoc_out, memory_store, det_id_to_local=None):
    """Imprime tabla separada con el desglose de score_sim por canal."""
    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    assoc_dbg = (dbg.get("association", {}) or {})
    if not assoc_dbg.get("enabled", True):
        return

    every = max(1, int(assoc_dbg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    if not assoc_dbg.get("show_similarity_details_table", False):
        return

    topk = int(assoc_dbg.get("candidates_topk", 3))
    df = assoc_similarity_details_to_dataframe(assoc_out, memory_store, det_id_to_local=det_id_to_local, topk=topk)
    if df.empty:
        return

    print(f"\n[DEBUG][AssociationSim] frame={frame_id}")
    for cname in sorted(df["class"].dropna().unique()):
        sub = df[df["class"] == cname]
        sub = compact_columns(sub, {"pair": 18, "class": 16})
        cols_show = [
            c
            for c in (
                "det",
                "pair",
                "S_sim",
                "Obj",
                "Bg",
                "BgP",
                "Parts",
                "Base",
            )
            if c in sub.columns
        ]
        print(f"\n[CLASS {cname}]")
        print_table_auto(sub, cols_show, pinned_cols=["det", "pair"], col_space=2)
