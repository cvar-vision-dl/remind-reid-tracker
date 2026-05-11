# utils/debug/update/debug_update.py

from __future__ import annotations

import pandas as pd

from ..debug_format import fmt
from ..debug_entity_helpers import det_local
from ..debug_table_utils import clamp_dataframe_rows, print_table_auto, sort_dataframe_if_possible
from .debug_update_helpers import memory_store_to_dataframe, neighbor_graph_topk_rows, proto_events_to_domain_dataframes


def print_proto_update_domain_table(
    title: str,
    frame_id: int,
    df: pd.DataFrame,
    cols: list,
    sort_by: list,
    max_rows: int,
):
    """Print a proto-update table for one domain."""
    if df is None or df.empty:
        print(f"\n[DEBUG][{title}] frame={frame_id} (no events)")
        return

    df = sort_dataframe_if_possible(df, sort_by=sort_by)
    df = clamp_dataframe_rows(df, max_rows=max_rows)

    cols = [c for c in (cols or []) if c in df.columns]
    print(f"\n[DEBUG][{title}] frame={frame_id}")
    print_table_auto(df, cols, pinned_cols=["label", "obj"], col_space=2)


def print_proto_update_table(config, frame_id, update_out):
    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    pcfg = (dbg.get("prototype_update", {}) or {})
    if not pcfg.get("enabled", False):
        return

    every = max(1, int(pcfg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    if update_out is None:
        return

    events = getattr(update_out, "proto_events", None) or []
    dfs = proto_events_to_domain_dataframes(events)

    obj_cfg = (pcfg.get("objects", {}) or {})
    bg_cfg = (pcfg.get("background", {}) or {})
    parts_cfg = (pcfg.get("parts", {}) or {})

    obj_enabled = bool(obj_cfg.get("enabled", True))
    bg_enabled = bool(bg_cfg.get("enabled", True))
    parts_enabled = bool(parts_cfg.get("enabled", False))

    fallback_max = int(pcfg.get("max_rows", 60))
    obj_max = int(obj_cfg.get("max_rows", fallback_max))
    bg_max = int(bg_cfg.get("max_rows", fallback_max))
    parts_max = int(parts_cfg.get("max_rows", fallback_max))

    if obj_enabled:
        print_proto_update_domain_table(
            title="ProtoUpdate-OBJ",
            frame_id=frame_id,
            df=dfs["OBJ"],
            cols=["label", "obj", "ch", "act", "s_max", "n_b", "n_a", "merge", "evict", "evict_s", "extra"],
            sort_by=["label", "obj", "ch"],
            max_rows=obj_max,
        )

    if bg_enabled:
        print_proto_update_domain_table(
            title="ProtoUpdate-BG",
            frame_id=frame_id,
            df=dfs["BG"],
            cols=[
                "label",
                "obj",
                "bank",
                "kind",
                "ring",
                "ch",
                "act",
                "s_max",
                "n_b",
                "n_a",
                "merge",
                "evict",
                "evict_s",
                "extra",
            ],
            sort_by=["label", "bank", "kind", "ring", "ch"],
            max_rows=bg_max,
        )

    if parts_enabled:
        print_proto_update_domain_table(
            title="ProtoUpdate-PARTS",
            frame_id=frame_id,
            df=dfs["PARTS"],
            cols=["label", "obj", "ch", "act", "s_max", "n_b", "n_a", "merge", "evict", "evict_s", "extra"],
            sort_by=["label", "obj", "ch"],
            max_rows=parts_max,
        )


def print_memory_table(config, frame_id, memory_store):
    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    mem_dbg = (dbg.get("memory", {}) or {})
    if not mem_dbg.get("enabled", True):
        return

    every = max(1, int(mem_dbg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    df = memory_store_to_dataframe(memory_store)
    if df.empty:
        print(f"\n[DEBUG][Memory] frame={frame_id} (empty)")
        return

    df = df.sort_values(by=["obj_id"], kind="stable")

    cols = ["obj_id", "label", "state", "obj_p", "obj_w", "obj_s", "parts_p", "bg_g_w", "bg_g_s", "bg_p_w", "bg_p_s", "hits", "misses"]
    cols = [c for c in cols if c in df.columns]

    print(f"\n[DEBUG][Memory] frame={frame_id}")
    print_table_auto(df, cols, pinned_cols=["obj_id", "label", "state"], col_space=2)


def distance_edge_total_count(edge) -> int:
    if edge is None:
        return 0
    if hasattr(edge, "total_count"):
        try:
            return int(edge.total_count())
        except Exception:
            return 0
    return int(getattr(edge, "cooccurrence_count", 0))


def summarize_distance_edge_modes(edge, var_floor: float, top_m: int = 3, nd: int = 3) -> str:
    contact_counts = getattr(edge, "contact_counts", None) or {}
    support_like_fn = getattr(edge, "mean_support_like", None)
    gap_mean_fn = getattr(edge, "mean_gap", None)
    center_mean_fn = getattr(edge, "mean_center", None)
    rank_mean_fn = getattr(edge, "mean_rank", None)

    parts = [
        f"gap={fmt(gap_mean_fn() if callable(gap_mean_fn) else None, nd)}",
        f"center={fmt(center_mean_fn() if callable(center_mean_fn) else None, nd)}",
        f"rank={fmt(rank_mean_fn() if callable(rank_mean_fn) else None, nd)}",
    ]
    if contact_counts:
        parts.append(
            "contact="
            + ",".join(f"{k}:{int(contact_counts.get(k, 0))}" for k in ("overlap", "touch", "near", "separate"))
        )
    if callable(support_like_fn):
        parts.append(f"support={fmt(support_like_fn(), nd)}")
    rel_fn = getattr(edge, "reliability", None)
    if callable(rel_fn):
        parts.append(f"rel={fmt(rel_fn(), nd)}")
    return " | ".join(parts)


def print_neighbor_distance_graph(
    config,
    frame_id,
    update_out,
    memory_store,
    object_ids: list[int] | None = None,
):
    """
    Print NeighborDistanceGraph per object (edges and modes).

    Por defecto usa update_out.visible_object_ids si object_ids es None.
    """
    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    upd_dbg = (dbg.get("update", {}) or {})
    if not upd_dbg.get("enabled", True):
        return

    show = bool(upd_dbg.get("show_neighbor_distance_graph", False))
    if not show:
        return

    every = max(1, int(upd_dbg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    if memory_store is None:
        return

    ids = object_ids
    if ids is None:
        ids = getattr(update_out, "visible_object_ids", []) if update_out is not None else []
    ids = [int(x) for x in (ids or [])]
    if not ids:
        return

    only_visible = bool(upd_dbg.get("neighbor_distance_graph_only_visible", True))
    topk_edges = int(upd_dbg.get("neighbor_distance_graph_topk", 8))
    topk_edges = max(1, topk_edges)
    topm_modes = int(upd_dbg.get("neighbor_distance_graph_topmodes", 3))
    topm_modes = max(1, topm_modes)

    visible_set = set(int(x) for x in ids) if only_visible else None

    print(f"\n[DEBUG][NeighborDistGraph] frame={frame_id}")
    for oid in ids:
        obj = memory_store.get(int(oid))
        if obj is None:
            continue

        lbl = getattr(obj, "instance_label", f"ID{int(oid)}")
        dg = getattr(obj, "neighbor_dist", None)
        if dg is None:
            print(f"  {lbl}: neighbor_dist=None")
            continue

        enabled = bool(getattr(dg, "enabled", False))
        var_floor = float(getattr(dg, "var_floor", 0.0))
        scale_min = float(getattr(dg, "scale_min", 0.0))
        edges = getattr(dg, "edges", None) or {}

        print(
            f"  {lbl}: enabled={enabled} n_edges={len(edges)} "
            f"var_floor={fmt(var_floor)} scale_min={fmt(scale_min)}"
        )

        if not enabled or not edges:
            continue

        items = list(edges.items())
        items.sort(key=lambda kv: distance_edge_total_count(kv[1]), reverse=True)

        shown = 0
        for dst_id, edge in items:
            dst_id = int(dst_id)
            if visible_set is not None and dst_id not in visible_set:
                continue

            tc = distance_edge_total_count(edge)
            last_ep = getattr(edge, "last_seen_episode", None)
            last_ts = getattr(edge, "last_seen_ts", None)

            modes_str = summarize_distance_edge_modes(edge, var_floor=var_floor, top_m=topm_modes)

            print(f"    -> {dst_id} total={tc} last_ep={last_ep} last_ts={fmt(last_ts)} {modes_str}")

            shown += 1
            if shown >= topk_edges:
                break


def print_update_summary(config, frame_id, update_out, memory_store, det_id_to_local=None):
    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    upd_dbg = (dbg.get("update", {}) or {})
    if not upd_dbg.get("enabled", True):
        return

    every = max(1, int(upd_dbg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    if update_out is None:
        return

    summ = getattr(update_out, "summary", {}) or {}
    n_m = int(summ.get("n_matches", len(getattr(update_out, "matches", []) or [])))
    n_c = int(summ.get("n_created", len(getattr(update_out, "created", []) or [])))
    n_a = int(summ.get("n_ambiguous", len(getattr(update_out, "ambiguous", []) or [])))
    n_p = int(summ.get("n_provisional", len(getattr(update_out, "provisional", []) or [])))
    n_v = int(summ.get("n_visible", len(getattr(update_out, "visible_object_ids", []) or [])))
    n_i = int(summ.get("n_inactive", len(getattr(update_out, "inactive", []) or [])))
    n_r = int(summ.get("n_removed", len(getattr(update_out, "removed", []) or [])))

    print(f"\n[DEBUG][Update] frame={frame_id} visible={n_v} matches={n_m} created={n_c} ambiguous={n_a} provisional={n_p} inactive={n_i} removed={n_r}")

    if not upd_dbg.get("show_lists", True):
        return

    for m in getattr(update_out, "matches", []) or []:
        det_id = m.get("det_id", None)
        oid = m.get("object_id", None)
        if oid is None:
            continue

        score_final = float(m.get("score_final", 0.0))
        det_l = det_local(det_id, det_id_to_local)

        obj = memory_store.get(int(oid)) if memory_store is not None else None
        lbl = getattr(obj, "instance_label", None) if obj else f"ID{int(oid)}"

        print(f"  MATCH    {det_l}-{lbl}  score_final={score_final:.3f}")

    for c in getattr(update_out, "created", []) or []:
        det_id = c.get("det_id", None)
        oid = c.get("object_id", None)
        if oid is None:
            continue

        det_l = det_local(det_id, det_id_to_local)
        obj = memory_store.get(int(oid)) if memory_store is not None else None
        lbl = getattr(obj, "instance_label", None) if obj else f"ID{int(oid)}"
        origin_mode = str(c.get("origin_mode", "") or "")
        origin_temp = c.get("origin_provisional_temp_id", None)
        origin_parents = [int(x) for x in (c.get("origin_parent_object_ids", []) or []) if x is not None]
        origin_related = [int(x) for x in (c.get("origin_related_known_ids", []) or []) if x is not None]
        origin_support = [int(x) for x in (c.get("origin_support_known_ids", []) or []) if x is not None]
        if origin_mode and origin_mode != "DIRECT_NEW":
            parent_txt = f" parents={origin_parents}" if origin_parents else ""
            related_txt = f" related={origin_related}" if origin_related else ""
            support_txt = f" support={origin_support}" if origin_support and origin_support != origin_related else ""
            temp_txt = f" temp={int(origin_temp)}" if origin_temp is not None else ""
            print(f"  CREATE   {det_l}-{lbl}  origin={origin_mode}{temp_txt}{parent_txt}{related_txt}{support_txt}")
        else:
            print(f"  CREATE   {det_l}-{lbl}")

    for a in getattr(update_out, "ambiguous", []) or []:
        det_id = a.get("det_id", None)
        temp_id = a.get("temp_id", None)
        if temp_id is None:
            continue

        det_l = det_local(det_id, det_id_to_local)
        amb = memory_store.get_ambiguous(int(temp_id)) if memory_store is not None else None
        lbl = amb.display_label(memory_store) if amb is not None else f"T_ID{int(temp_id)}"
        refs = [int(x) for x in (a.get("related_known_ids", a.get("candidate_ids", [])) or []) if x is not None]
        refs_txt = f" refs={refs}" if refs else ""
        print(f"  AMBIG    {det_l}-{lbl}[amb:{int(temp_id)}]{refs_txt}")

    for p in getattr(update_out, "provisional", []) or []:
        det_id = p.get("det_id", None)
        temp_id = p.get("temp_id", None)
        if temp_id is None:
            continue

        det_l = det_local(det_id, det_id_to_local)
        prov = memory_store.get_provisional(int(temp_id)) if memory_store is not None else None
        lbl = prov.display_label(memory_store) if prov is not None else f"T_ID{int(temp_id)}"
        refs = [int(x) for x in (p.get("related_known_ids", p.get("support_known_ids", [])) or []) if x is not None]
        refs_txt = f" refs={refs}" if refs else ""
        print(f"  PROV     {det_l}-{lbl}[prov:{int(temp_id)}]{refs_txt}")

    for oid in getattr(update_out, "inactive", []) or []:
        obj = memory_store.get(int(oid)) if memory_store is not None else None
        lbl = getattr(obj, "instance_label", None) if obj else f"ID{int(oid)}"
        print(f"  INACTIVE {lbl}")

    for oid in getattr(update_out, "removed", []) or []:
        print(f"  REMOVE   ID{int(oid)}")

    show_ng = bool(upd_dbg.get("show_neighbor_graph", True))
    if show_ng and memory_store is not None:
        topk = int(upd_dbg.get("neighbor_graph_topk", 5))
        ids = getattr(update_out, "visible_object_ids", []) or []
        if ids:
            print(f"\n[DEBUG][NeighborsGraph] frame={frame_id}")
            for oid in ids:
                obj = memory_store.get(int(oid))
                if obj is None:
                    continue

                lbl = getattr(obj, "instance_label", f"ID{int(oid)}")
                rows = neighbor_graph_topk_rows(obj, k=topk)
                if not rows:
                    continue

                s = ", ".join([f"{r['dst']}(c={r['c']},w={r['w']},last={r['last']})" for r in rows])
                print(f"  {lbl}: {s}")
