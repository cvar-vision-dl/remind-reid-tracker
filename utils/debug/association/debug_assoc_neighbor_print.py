from __future__ import annotations

from ..debug_format import fmt
from ..debug_table_utils import compact_columns, print_table_auto, print_table_wrap_column
from .debug_assoc_dataframes import (
    context_veto_candidates_to_dataframe,
    local_context_candidates_to_dataframe,
    neighbor_sets_candidates_to_dataframe,
    neighbor_sets_class_options_to_dataframes,
    neighbor_sets_output_to_dataframes,
)


def print_neighbor_sets_table(config, frame_id, assoc_out, memory_store, det_id_to_local=None):
    """Imprime tabla de neighbor-sets (sets + priors)."""
    neigh_sets_out = getattr(assoc_out, "neighbor_sets_out", None) if assoc_out is not None else None

    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    assoc_dbg = (dbg.get("association", {}) or {})
    if not assoc_dbg.get("enabled", True):
        return

    every = max(1, int(assoc_dbg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    if not assoc_dbg.get("show_neighbor_sets_table", False):
        return

    topk = max(1, int(assoc_dbg.get("neighbor_sets_topk", 10)))
    compact_tables = bool(assoc_dbg.get("compact_tables", True))
    max_assign = int(assoc_dbg.get("max_assign_chars", 20))
    max_objs = int(assoc_dbg.get("max_objs_chars", 20))
    wrap_assign = bool(assoc_dbg.get("neighbor_sets_wrap_assign", True))
    wrap_width = int(assoc_dbg.get("neighbor_sets_table_width", 0))
    compact_headers = bool(assoc_dbg.get("neighbor_sets_compact_headers", True))

    df_sets, df_obj = neighbor_sets_output_to_dataframes(neigh_sets_out, memory_store, det_id_to_local=det_id_to_local)
    if df_sets is None or df_sets.empty:
        print(f"\n[DEBUG][NeighborSets] frame={frame_id} (empty)")
        return

    df_sets = df_sets.head(topk)

    out = neigh_sets_out if isinstance(neigh_sets_out, dict) else {}
    core = out.get("core", {}) if isinstance(out.get("core", None), dict) else {}
    dbg_pack = out.get("debug", {}) if isinstance(out.get("debug", None), dict) else {}
    meta = dbg_pack.get("meta", {}) if isinstance(dbg_pack.get("meta", None), dict) else {}

    anchors_raw = meta.get("anchors_raw", []) or []
    anchors_filtered = meta.get("anchors_filtered", []) or []

    ctx = {}
    out_dbg = getattr(assoc_out, "debug", None)
    if isinstance(out_dbg, dict):
        ctx = (((out_dbg.get("extra", {}) or {}).get("ns", {}) or {}).get("ctx", {}) or {})
    global_ok = int(bool(ctx.get("global_ok", False)))
    reason = str(ctx.get("reason", ""))
    quality = ctx.get("quality", None)

    hdr_bits = []
    for k, label in (
        ("best_score", "best"),
        ("coverage_eff", "cov"),
        ("mean_maturity_best", "mat"),
        ("k_best", "k"),
    ):
        v = ctx.get(k, None)
        if v is None:
            v = core.get(k, None)
        if v is not None:
            hdr_bits.append(f"{label}={fmt(v)}")
    if quality is not None:
        hdr_bits.insert(0, f"q={fmt(quality)}")

    extra_hdr = (" | " + " ".join(hdr_bits)) if hdr_bits else ""
    print(
        f"\n[DEBUG][NeighborSets] frame={frame_id} ok={global_ok} reason={reason} "
        f"anchors_raw={list(anchors_raw)} anchors={list(anchors_filtered)}{extra_hdr}"
    )

    cols_sets = [
        "rank",
        "score_sets",
        "cov",
        "cov_eff",
        "k",
        "size",
        "info",
        "stab",
        "dens",
        "ctx",
        "mat_c",
        "mat_r",
        "conn_f",
        "edge_cov",
        "node_cov",
        "dens_v",
        "excl",
        "excl_v",
        "logC",
        "maturity",
        "assign",
    ]

    packed = bool(assoc_dbg.get("neighbor_sets_packed", True))
    if packed:
        cols_sets = [
            "rank",
            "score_sets",
            "cov2",
            "k",
            "size",
            "info",
            "stab",
            "dens",
            "ctx",
            "mat",
            "conn_f",
            "covg",
            "excl2",
            "logC",
            "assign",
        ]
    cols_sets = [c for c in cols_sets if c in df_sets.columns]

    hide_density = bool(assoc_dbg.get("neighbor_sets_hide_density", False))
    hide_logc = bool(assoc_dbg.get("neighbor_sets_hide_logc", True))
    hide_excl = bool(assoc_dbg.get("neighbor_sets_hide_exclusivity", False))

    if hide_density:
        for c in ("dens", "ctx", "mat_c", "mat_r", "mat", "conn_f", "edge_cov", "node_cov", "covg", "dens_v"):
            if c in cols_sets:
                cols_sets.remove(c)
    if hide_logc and "logC" in cols_sets:
        cols_sets.remove("logC")
    if hide_excl:
        for c in ("excl", "excl_v", "excl2"):
            if c in cols_sets:
                cols_sets.remove(c)

    print("\n[SETS]")
    df_sets_show = df_sets
    if compact_tables:
        limits = {"objs": max_objs}
        if not wrap_assign:
            limits["assign"] = max_assign
        df_sets_show = compact_columns(df_sets_show, limits)
    col_space = int(assoc_dbg.get("neighbor_sets_col_space", 1))
    col_space = max(0, col_space)
    if wrap_assign and "assign" in cols_sets:
        hdr_map = {}
        if compact_headers and packed:
            hdr_map = {
                "rank": "r",
                "score_sets": "S",
                "cov2": "cov",
                "k": "k",
                "size": "sz",
                "info": "inf",
                "stab": "stb",
                "dens": "den",
                "ctx": "ctx",
                "mat": "mat",
                "conn_f": "conn",
                "covg": "covg",
                "excl2": "excl",
                "logC": "logC",
                "assign": "assign",
            }
        print_table_wrap_column(
            df_sets_show,
            cols_sets,
            wrap_col="assign",
            table_width=wrap_width,
            col_space=max(1, col_space),
            headers=hdr_map,
            wrap_sep=",",
        )
    else:
        print_table_auto(
            df_sets_show,
            cols_sets,
            pinned_cols=["rank"],
            col_space=col_space,
        )

    show_cands = bool(assoc_dbg.get("show_neighbor_sets_candidates_table", True))
    if show_cands:
        cand_topk = int(assoc_dbg.get("neighbor_sets_candidates_topk", assoc_dbg.get("candidates_topk", 5)))
        cand_topk = max(1, cand_topk)
        df_cands = neighbor_sets_candidates_to_dataframe(
            assoc_out,
            memory_store,
            det_id_to_local=det_id_to_local,
            topk=cand_topk,
        )
        if df_cands is not None and not df_cands.empty:
            print("\n[CANDIDATE CONTEXT]")
            for cname in sorted(df_cands["class"].dropna().unique()):
                sub = df_cands[df_cands["class"] == cname]
                cols = [
                    "det",
                    "pair",
                    "fin",
                    "sel",
                    "S_sim",
                    "S_sets",
                    "B_sets",
                    "P_sets",
                    "U_sets",
                    "Q_sets",
                    "C_rel",
                    "C_band",
                    "K_abs",
                    "K_hit",
                    "K_cov",
                    "K_rel",
                    "H_rel",
                    "S_final",
                    "NS_ctx",
                    "NS_cls",
                    "NS_pol",
                ]
                cols = [c for c in cols if c in sub.columns]
                if compact_tables:
                    sub = compact_columns(
                        sub,
                        {
                            "pair": max_assign,
                            "NS_ctx": 22,
                            "NS_cls": 28,
                            "NS_pol": 28,
                        },
                    )
                print(f"\n[CLASS {cname}]")
                print_table_auto(
                    sub,
                    cols,
                    pinned_cols=["det", "pair", "fin", "sel"],
                    col_space=2,
                )

    if bool(assoc_dbg.get("show_neighbor_sets_veto_table", True)):
        veto_topk = int(assoc_dbg.get("neighbor_sets_veto_topk", assoc_dbg.get("candidates_topk", 5)))
        veto_topk = max(1, veto_topk)
        df_veto = context_veto_candidates_to_dataframe(
            assoc_out,
            memory_store,
            det_id_to_local=det_id_to_local,
            topk=veto_topk,
            config=config,
        )
        if df_veto is not None and not df_veto.empty:
            print("\n[VETO CHECK]")
            for cname in sorted(df_veto["class"].dropna().unique()):
                sub = df_veto[df_veto["class"] == cname]
                cols = [
                    "det",
                    "pair",
                    "fin",
                    "sel",
                    "veto",
                    "why",
                    "kp",
                    "sup",
                    "ss",
                    "sh",
                    "keep",
                    "q",
                    "pr",
                    "cs",
                    "cr",
                    "s_sets",
                    "s_dist",
                    "gate",
                    "veto_raw",
                ]
                cols = [c for c in cols if c in sub.columns]
                if compact_tables:
                    sub = compact_columns(
                        sub,
                        {
                            "pair": max_assign,
                            "gate": 18,
                            "veto_raw": 18,
                        },
                    )
                print(f"\n[CLASS {cname}]")
                print_table_auto(
                    sub,
                    cols,
                    pinned_cols=["det", "pair", "fin", "sel", "veto"],
                    wrap_col="why" if "why" in cols else None,
                    col_space=2,
                    wrap_sep=",",
                )

    if bool(assoc_dbg.get("show_local_context_table", True)):
        local_topk = int(assoc_dbg.get("local_context_topk", assoc_dbg.get("candidates_topk", 5)))
        local_topk = max(1, local_topk)
        df_local = local_context_candidates_to_dataframe(
            assoc_out,
            memory_store,
            det_id_to_local=det_id_to_local,
            topk=local_topk,
            config=config,
        )
        if df_local is not None and not df_local.empty:
            print("\n[LOCAL CONTEXT]")
            for cname in sorted(df_local["class"].dropna().unique()):
                sub = df_local[df_local["class"] == cname]
                cols = [
                    "det",
                    "pair",
                    "fin",
                    "sel",
                    "ep",
                    "mat",
                    "rich",
                    "exp",
                    "ker",
                    "hit",
                    "ctx",
                    "why",
                ]
                cols = [c for c in cols if c in sub.columns]
                if compact_tables:
                    sub = compact_columns(
                        sub,
                        {
                            "pair": max_assign,
                            "why": 42,
                        },
                    )
                print(f"\n[CLASS {cname}]")
                print_table_auto(
                    sub,
                    cols,
                    pinned_cols=["det", "pair", "fin", "sel", "ctx"],
                    wrap_col="why" if "why" in cols else None,
                    col_space=2,
                    wrap_sep=",",
                )

    df_opts = neighbor_sets_class_options_to_dataframes(neigh_sets_out, memory_store)
    if df_opts is not None and not df_opts.empty:
        print("\n[CLASS OPTIONS]")
        for cname in sorted(df_opts["class"].dropna().unique()):
            sub = df_opts[df_opts["class"] == cname]
            if compact_tables:
                sub = compact_columns(
                    sub,
                    {
                        "obj": 16,
                        "kernel": max_assign,
                        "detail": 84,
                    },
                )
            print(f"\n[CLASS {cname}]")
            print_table_auto(
                sub,
                ["dets", "obj", "supp", "hit", "cov", "kernel", "detail"],
                pinned_cols=["dets", "obj"],
                wrap_col="detail",
                col_space=2,
                wrap_sep=",",
            )

    if not bool(assoc_dbg.get("show_neighbor_sets_object_priors", True)):
        return
    if df_obj is None or df_obj.empty:
        return

    obj_topk = int(assoc_dbg.get("neighbor_sets_object_topk", 20))
    obj_topk = max(1, obj_topk)
    cols_obj = ["obj_id", "label", "prior", "supp_sum"]
    cols_obj = [c for c in cols_obj if c in df_obj.columns]

    print("\n[OBJECT PRIORS]")
    print_table_auto(
        df_obj.head(obj_topk),
        cols_obj,
        pinned_cols=["obj_id", "label"],
        col_space=2,
    )
