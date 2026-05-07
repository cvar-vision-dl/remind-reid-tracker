from __future__ import annotations

import pandas as pd

from ..debug_format import fmt


def neighbor_graph_topk_rows(obj, k: int = 5):
    """Top-k vecinos del NeighborGraph en formato filas imprimibles."""
    ng = getattr(obj, "neighbors", None)
    if ng is None or not getattr(ng, "enabled", False):
        return []

    items = ng.topk(int(k))
    rows = []
    for it in (items or []):
        if not isinstance(it, dict):
            continue
        rows.append(
            {
                "dst": int(it.get("dst_id", -1)),
                "c": int(it.get("cooc_count", 0)),
                "w": fmt(it.get("weight", 0.0)),
                "last": fmt(it.get("last_seen_ts", 0.0)),
            }
        )
    return rows


def proto_domain(e: dict) -> str:
    """Dominio de evento: OBJ / BG / PARTS."""
    k = str(e.get("kind") or "").lower().strip()
    if k == "bg":
        return "BG"
    if k == "parts":
        return "PARTS"
    return "OBJ"


def proto_extra(e: dict) -> str:
    """Extra compacto para evento: filtra core keys y formatea el resto."""
    if isinstance(e.get("promote_hits", None), (int, float)):
        return f"promote_hits={int(e.get('promote_hits'))}"

    core = {
        "kind",
        "scope",
        "label",
        "object_id",
        "channel",
        "action",
        "s_max",
        "n_before",
        "n_after",
        "merge_pair_sim",
        "evict_strategy",
        "evicted_index",
    }
    out = []
    for k, v in (e or {}).items():
        if k in core or v is None:
            continue
        out.append(f"{k}={v}")
    return ", ".join(out) if out else ""


def bg_scope_fields(e: dict) -> dict:
    """Campos derivados de scope para BG: bank/region/ring."""
    s = e.get("scope", {}) if isinstance(e.get("scope", None), dict) else {}
    bank = str(s.get("bank", "")).upper()
    region = str(s.get("region", "")).upper()
    ring = str(s.get("ring", "")).lower().strip()
    ring_c = "I" if ring == "inner" else ("O" if ring == "outer" else "")
    return {"bank": bank, "kind": region, "ring": ring_c}


def proto_events_to_domain_dataframes(proto_events):
    """Separa proto_events en 3 dataframes: OBJ/BG/PARTS."""
    rows_obj, rows_bg, rows_parts = [], [], []

    for e in (proto_events or []):
        if not isinstance(e, dict):
            continue

        dom = proto_domain(e)
        base = {
            "label": e.get("label") or "?",
            "obj": int(e.get("object_id", -1)),
            "ch": e.get("channel") or "?",
            "act": e.get("action") or "?",
            "s_max": "" if e.get("s_max", None) is None else fmt(e.get("s_max")),
            "n_b": int(e.get("n_before", 0)),
            "n_a": int(e.get("n_after", 0)),
            "merge": "" if e.get("merge_pair_sim", None) is None else fmt(e.get("merge_pair_sim")),
            "evict": "" if e.get("evicted_index", None) is None else int(e.get("evicted_index")),
            "evict_s": "" if not e.get("evict_strategy", None) else str(e.get("evict_strategy")),
            "extra": proto_extra(e),
        }

        if dom == "BG":
            row = dict(base)
            row.update(bg_scope_fields(e))
            rows_bg.append(row)
        elif dom == "PARTS":
            rows_parts.append(dict(base))
        else:
            rows_obj.append(dict(base))

    return {
        "OBJ": pd.DataFrame(rows_obj),
        "BG": pd.DataFrame(rows_bg),
        "PARTS": pd.DataFrame(rows_parts),
    }


def count_bank(bank) -> int:
    """Cuenta protos en bancos BG (bank.prototypes)."""
    if bank is None:
        return 0
    protos = getattr(bank, "prototypes", None)
    return int(len(protos)) if isinstance(protos, list) else 0


def memory_store_to_dataframe(memory_store):
    """Resumen por objeto: estado + nº protos (obj/parts/bg) + hits/misses."""
    rows = []
    if memory_store is None:
        return pd.DataFrame()

    for obj in memory_store.all_objects():
        obj_work = 0
        obj_stable = 0
        for ch in obj.appearance.channel_names():
            obj_work += len(obj.appearance.get_channel_work_embeddings(ch))
            obj_stable += len(obj.appearance.get_channel_stable_embeddings(ch))
        obj_proto = int(obj_work + obj_stable)

        parts_proto = 0
        for ch in obj.parts.channel_names():
            parts_proto += len(obj.parts.get_channel_embeddings(ch))

        bg = getattr(obj, "background", None)

        bg_g_w = count_bank(getattr(bg, "inner_global_work", None)) + count_bank(getattr(bg, "outer_global_work", None))
        bg_g_s = count_bank(getattr(bg, "inner_global_stable", None)) + count_bank(
            getattr(bg, "outer_global_stable", None)
        )
        bg_p_w = count_bank(getattr(bg, "inner_partials_work", None)) + count_bank(
            getattr(bg, "outer_partials_work", None)
        )
        bg_p_s = count_bank(getattr(bg, "inner_partials_stable", None)) + count_bank(
            getattr(bg, "outer_partials_stable", None)
        )

        rows.append(
            {
                "obj_id": int(obj.object_id),
                "label": getattr(obj, "instance_label", None),
                "state": str(getattr(obj, "state", "")),
                "obj_p": int(obj_proto),
                "obj_w": int(obj_work),
                "obj_s": int(obj_stable),
                "parts_p": int(parts_proto),
                "bg_g_w": int(bg_g_w),
                "bg_g_s": int(bg_g_s),
                "bg_p_w": int(bg_p_w),
                "bg_p_s": int(bg_p_s),
                "hits": int(getattr(obj, "hits", 0)),
                "misses": int(getattr(obj, "misses", 0)),
            }
        )

    return pd.DataFrame(rows)
