from __future__ import annotations

import pandas as pd

from ..debug_format import fmt
from ..debug_entity_helpers import det_local, get_track_class_name, get_track_label_short
from ..debug_table_utils import compact_columns, print_table_auto


def print_postcreate_temporal_table(config, frame_id, assoc_out, memory_store, det_id_to_local=None):
    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    assoc_dbg = (dbg.get("association", {}) or {})
    if not assoc_dbg.get("enabled", True):
        return
    if not assoc_dbg.get("show_postcreate_temporal_table", False):
        return

    every = max(1, int(assoc_dbg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    out_dbg = getattr(assoc_out, "debug", None)
    if not isinstance(out_dbg, dict):
        return
    entries = out_dbg.get("postcreate_temporal", None)
    if not isinstance(entries, list) or not entries:
        print(f"\n[DEBUG][PostCreate] frame={frame_id} (empty)")
        return

    def fmt_oid(oid):
        if oid is None:
            return ""
        return get_track_label_short(memory_store, int(oid), n=4) or str(int(oid))

    def fmt_oid_list(oids):
        return ",".join(fmt_oid(int(oid)) for oid in (oids or []))

    def class_name_for_entry(entry):
        candidate_rows = [row for row in (entry.get("candidate_rows", []) or []) if isinstance(row, dict)]
        for row in candidate_rows:
            oid = row.get("object_id", None)
            if oid is None:
                continue
            name = get_track_class_name(memory_store, int(oid))
            if name:
                return str(name).upper()
        return f"CLASS_{int(entry.get('class_id', -1))}"

    rows = []
    cand_rows = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        det_id = int(entry.get("det_id", -1))
        class_name = class_name_for_entry(entry)
        rows.append(
            {
                "det": det_local(det_id, det_id_to_local),
                "cls": class_name,
                "st": str(entry.get("temporal_status", "") or ""),
                "dec": str(entry.get("decision_kind", "") or ""),
                "why": str(entry.get("decision_reason", "") or entry.get("skip_reason", "") or ""),
                "best": fmt_oid(entry.get("best_object_id", None)),
                "best_s": fmt(entry.get("best_score", None)),
                "top": fmt_oid(entry.get("top_supported_object_id", None)),
                "top_s": fmt(entry.get("top_supported_score", None)),
                "supp": fmt_oid_list(entry.get("support_known_ids", [])),
                "block": fmt_oid_list(entry.get("blocked_known_ids", [])),
                "ctx": str(entry.get("context_mode", "") or ""),
                "ctxk": int(entry.get("has_known_context", 0) or 0),
                "vfb": int(entry.get("visual_fallback_ok", 0) or 0),
                "blk_ok": int(entry.get("known_blocked_ok", 0) or 0),
            }
        )
        for row in (entry.get("candidate_rows", []) or []):
            if not isinstance(row, dict):
                continue
            oid = row.get("object_id", None)
            cand_rows.append(
                {
                    "det": det_local(det_id, det_id_to_local),
                    "cls": class_name,
                    "pair": fmt_oid(oid),
                    "temp": fmt(row.get("temp_score", None)),
                    "sim": fmt(row.get("score_sim", None)),
                    "fin": fmt(row.get("score_final", None)),
                    "ctx": int(row.get("support_ctx", 0) or 0),
                    "keep": int(row.get("support_final", 0) or 0),
                    "blk": int(row.get("blocked", 0) or 0),
                    "kp": int(row.get("known_plausible_keep", 0) or 0),
                    "dk": int(row.get("decision_keep", 0) or 0),
                    "min": int(row.get("min_ok", 0) or 0),
                    "gap": int(row.get("gap_ok", 0) or 0),
                    "why": str(row.get("why", "") or ""),
                }
            )

    df = compact_columns(
        pd.DataFrame(rows),
        {
            "cls": 12,
            "why": 34,
            "supp": 16,
            "block": 12,
            "ctx": 18,
        },
    )
    print(f"\n[DEBUG][PostCreate] frame={frame_id}")
    print_table_auto(
        df,
        cols=["det", "cls", "st", "dec", "why", "best", "best_s", "top", "top_s", "supp", "block", "ctx", "ctxk", "vfb", "blk_ok"],
        pinned_cols=["det"],
        col_space=2,
    )

    if cand_rows:
        df_cands = compact_columns(
            pd.DataFrame(cand_rows),
            {
                "cls": 12,
                "pair": 10,
                "why": 24,
            },
        )
        print("\n[POSTCREATE CANDIDATES]")
        print_table_auto(
            df_cands,
            cols=["det", "cls", "pair", "temp", "sim", "fin", "ctx", "keep", "blk", "kp", "dk", "min", "gap", "why"],
            pinned_cols=["det"],
            col_space=2,
        )


def print_known_set_distance_disambiguation_table(config, frame_id, assoc_out, memory_store, det_id_to_local=None):
    dbg = (config.get("debug", {}) or {})
    if not dbg.get("enabled", False):
        return

    assoc_dbg = (dbg.get("association", {}) or {})
    if not assoc_dbg.get("enabled", True):
        return

    every = max(1, int(assoc_dbg.get("every_n_frames", 1)))
    if (int(frame_id) % every) != 0:
        return

    out_dbg = getattr(assoc_out, "debug", None)
    if not isinstance(out_dbg, dict):
        return
    pack = out_dbg.get("known_set_distance_disambiguation", None)
    if not isinstance(pack, dict):
        return
    components = pack.get("components", None)
    if not isinstance(components, list):
        return
    pair_anchors = pack.get("pair_anchors", None)
    if not isinstance(pair_anchors, list):
        pair_anchors = []
    passes = pack.get("passes", None)
    if not isinstance(passes, list):
        passes = []
    resolved_source_by_det_id = pack.get("resolved_source_by_det_id", None)
    if not isinstance(resolved_source_by_det_id, dict):
        resolved_source_by_det_id = {}

    def fmt_det_ids(det_ids):
        return ",".join(str(det_local(int(det_id), det_id_to_local)) for det_id in (det_ids or []))

    def fmt_assign(assignment):
        if not isinstance(assignment, dict) or not assignment:
            return ""
        parts = []
        for det_id, oid in sorted(((int(k), int(v)) for k, v in assignment.items()), key=lambda kv: kv[0]):
            parts.append(f"{det_local(det_id, det_id_to_local)}->{int(oid)}")
        return ",".join(parts)

    def fmt_stable(assignment):
        return fmt_assign(assignment)

    if passes:
        pass_rows = []
        for pass_info in passes:
            if not isinstance(pass_info, dict):
                continue
            pass_rows.append(
                {
                    "pass": int(pass_info.get("pass_index", 0) or 0),
                    "input": fmt_det_ids(pass_info.get("input_det_ids", [])),
                    "resolved": fmt_det_ids(pass_info.get("resolved_det_ids", [])),
                    "remaining": fmt_det_ids(pass_info.get("remaining_det_ids", [])),
                    "comp": len(pass_info.get("components", []) or []),
                    "anc": len(pass_info.get("pair_anchors", []) or []),
                }
            )
        if pass_rows:
            df_passes = pd.DataFrame(pass_rows)
            print(f"\n[DEBUG][KnownSetDistPasses] frame={frame_id}")
            print_table_auto(
                df_passes,
                cols=["pass", "input", "resolved", "remaining", "comp", "anc"],
                pinned_cols=["pass"],
                col_space=2,
            )

    if resolved_source_by_det_id:
        resolved_rows = []
        for det_id, source in sorted(
            ((int(k), str(v)) for k, v in resolved_source_by_det_id.items()),
            key=lambda kv: kv[0],
        ):
            resolved_rows.append(
                {
                    "det": det_local(int(det_id), det_id_to_local),
                    "src": str(source),
                }
            )
        df_resolved = pd.DataFrame(resolved_rows)
        print(f"\n[DEBUG][KnownSetDistResolved] frame={frame_id}")
        print_table_auto(
            df_resolved,
            cols=["det", "src"],
            pinned_cols=["det"],
            col_space=2,
        )

    if not components:
        print(f"\n[DEBUG][KnownSetDist] frame={frame_id} (empty)")
    else:
        rows = []
        for idx, comp in enumerate(components, start=1):
            top = list(comp.get("top_solutions", []) or [])
            top1 = top[0] if len(top) >= 1 and isinstance(top[0], dict) else {}
            top2 = top[1] if len(top) >= 2 and isinstance(top[1], dict) else {}
            rows.append(
                {
                    "comp": int(idx),
                    "pass": int(comp.get("pass_index", 0) or 0),
                    "dets": fmt_det_ids(comp.get("det_ids", [])),
                    "cand": ",".join(str(int(x)) for x in (comp.get("candidate_union", []) or [])),
                    "in": fmt_det_ids(comp.get("pass_input_det_ids", [])),
                    "res": fmt_det_ids(comp.get("pass_resolved_det_ids", [])),
                    "rem": fmt_det_ids(comp.get("pass_remaining_det_ids", [])),
                    "anchors": ",".join(str(int(x)) for x in (comp.get("anchor_ids", []) or [])),
                    "best": fmt(comp.get("best_score", None)),
                    "second": fmt(comp.get("second_score", None)),
                    "gap": fmt(comp.get("gap", None)),
                    "core": fmt(comp.get("core_score", None)),
                    "cgap": fmt(comp.get("core_gap", None)),
                    "evid": fmt(comp.get("evidence", None)),
                    "front": int(comp.get("frontier_size", 0) or 0),
                    "K": int(comp.get("known_assignments", 0) or 0),
                    "A": fmt(comp.get("anchor_term", None)),
                    "H": fmt(comp.get("history_term", None)),
                    "F": fmt(comp.get("frame_term", None)),
                    "Q": fmt(comp.get("anchor_quality_term", None)),
                    "O": fmt(comp.get("order_term", None)),
                    "V": fmt(comp.get("visual_term", None)),
                    "stable": fmt_stable(comp.get("stable_det_assignments", {})),
                    "top1": fmt_assign(top1.get("assignment", {})),
                    "top2": fmt_assign(top2.get("assignment", {})),
                }
            )

        df = pd.DataFrame(rows)
        print(f"\n[DEBUG][KnownSetDist] frame={frame_id}")
        print_table_auto(
            df,
            cols=["comp", "pass", "dets", "cand", "in", "res", "rem", "anchors", "best", "second", "gap", "core", "cgap", "evid", "front", "K", "A", "H", "F", "Q", "O", "V", "stable", "top1", "top2"],
            pinned_cols=["comp", "pass", "dets"],
            col_space=2,
        )

        vote_rows = []
        for idx, comp in enumerate(components, start=1):
            top = [item for item in (comp.get("top_solutions", []) or []) if isinstance(item, dict)]
            for sol_idx, sol in enumerate(top[:2], start=1):
                assign_label = fmt_assign(sol.get("assignment", {}))
                for row in (sol.get("anchor_breakdown", []) or []):
                    if not isinstance(row, dict):
                        continue
                    anchor_id = int(row.get("anchor_id", -1))
                    detail = dict(row.get("order_detail", {}) or {})
                    obs_closer_det = detail.get("obs_closer_det", None)
                    closer_oid = detail.get("closer_oid", None)
                    vote_rows.append(
                        {
                            "comp": int(idx),
                            "sol": int(sol_idx),
                            "assign": assign_label,
                            "anc": get_track_label_short(memory_store, int(anchor_id), n=4) or str(int(anchor_id)),
                            "Q": fmt(row.get("anchor_quality", None)),
                            "O": fmt(row.get("order_fit", None)),
                            "A": fmt(row.get("assignment_match", None)),
                            "ev": fmt(row.get("anchor_evidence", None)),
                            "gap": fmt(row.get("pair_gap", None)),
                            "det+": (str(det_local(int(obs_closer_det), det_id_to_local)) if obs_closer_det is not None else ""),
                            "id+": (get_track_label_short(memory_store, int(closer_oid), n=4) or str(int(closer_oid))) if closer_oid is not None else "",
                            "op": fmt(detail.get("order_prob", None)),
                            "ms": fmt(detail.get("margin_sim", None)),
                            "pw": fmt(detail.get("pair_weight", None)),
                        }
                    )
        if vote_rows:
            df_votes = compact_columns(
                pd.DataFrame(vote_rows),
                {
                    "assign": 16,
                    "anc": 8,
                    "id+": 8,
                },
            )
            print(f"\n[DEBUG][KnownSetVotes] frame={frame_id}")
            print_table_auto(
                df_votes,
                cols=["comp", "sol", "assign", "anc", "Q", "O", "A", "ev", "gap", "det+", "id+", "op", "ms", "pw"],
                pinned_cols=["comp", "sol"],
                col_space=2,
            )

    if not pair_anchors:
        print(f"\n[DEBUG][PairAnchors] frame={frame_id} (empty)")
        return

    hist_rows = []
    frame_rows = []
    pair_src_map = {
        "match": "frm",
        "soft": "soft",
        "selected_only": "sel",
        "history": "hist",
        "history_visible": "hist+v",
    }
    pair_why_map = {
        "selected": "sel",
        "valid_but_not_selected": "valid",
        "below_min_anchor_informativeness": "low_lu",
        "historical_candidate": "hist",
        "historical_visible": "hist+v",
        "no_anchor_candidates": "none",
    }
    for pack in pair_anchors:
        class_id = int(pack.get("class_id", -1))
        class_name = (get_track_class_name(memory_store, next(iter(pack.get('candidate_union', []) or []), None)) or f"CLASS_{class_id}").upper()
        det_ids = [int(x) for x in (pack.get("det_ids", []) or [])]
        cand_ids = [int(x) for x in (pack.get("candidate_union", []) or [])]
        dets_label = ",".join(str(det_local(int(det_id), det_id_to_local)) for det_id in det_ids)
        cand_label = ",".join(get_track_label_short(memory_store, int(oid), n=4) or str(int(oid)) for oid in cand_ids)

        anchor_items = [item for item in (pack.get("anchors", []) or []) if isinstance(item, dict)]
        if not anchor_items:
            hist_rows.append(
                {
                    "class": class_name,
                    "dets": dets_label,
                    "cand": cand_label,
                    "anchor": "-",
                    "src": "",
                    "why": "none",
                    "u": "",
                    "bu": "",
                    "pu": "",
                    "pc": "",
                    "pm": "",
                    "pr": "",
                    "pb": "",
                    "hm": "",
                }
            )
            frame_rows.append(
                {
                    "class": class_name,
                    "dets": dets_label,
                    "cand": cand_label,
                    "anchor": "-",
                    "rk": "",
                    "ok": "",
                    "sel": "",
                    "src": "",
                    "why": "none",
                    "lu": "",
                    "lur": "",
                    "obs": "",
                    "sep": "",
                    "m1": "",
                    "m2": "",
                    "g1": "",
                    "g2": "",
                }
            )
            continue

        for item in anchor_items:
            if not isinstance(item, dict):
                continue
            anchor_id = int(item.get("anchor_id", -1))
            label = get_track_label_short(memory_store, int(anchor_id), n=4) or str(int(anchor_id))
            use = fmt(item.get("usefulness", None))
            local_use = fmt(item.get("local_usefulness", None))
            local_reason = str(item.get("local_reason", "") or "")
            base_use = fmt(item.get("base_usefulness", None))
            pair_use = fmt(item.get("pair_usefulness", None))
            rank = item.get("rank", None)
            valid_frame = "Y" if bool(item.get("frame_valid", False)) else "N"
            selected = "Y" if bool(item.get("selected", False)) else "N"
            source_key = str(item.get("source", "") or "")
            source = pair_src_map.get(source_key, source_key)
            why_key = str(item.get("why", "") or "")
            why = pair_why_map.get(why_key, why_key)
            frame_modes = list(item.get("frame_modes", []) or [])
            frame_distances = list(item.get("frame_distances", item.get("frame_gap", [])) or [])
            gap_mean = fmt(item.get("gap_margin_mean", None))
            pair_cons = fmt(item.get("pair_consistency", None))
            pair_margin = fmt(item.get("pair_margin_mean", None))
            pair_rel = fmt(item.get("pair_reliability", None))
            pair_rob = fmt(item.get("pair_robustness", None))
            hist_rows.append(
                {
                    "class": class_name,
                    "dets": dets_label,
                    "cand": cand_label,
                    "anchor": label,
                    "src": source,
                    "why": why,
                    "u": use,
                    "bu": base_use,
                    "pu": pair_use,
                    "pc": pair_cons,
                    "pm": pair_margin,
                    "pr": pair_rel,
                    "pb": pair_rob,
                    "hm": gap_mean,
                }
            )
            frame_rows.append(
                {
                    "class": class_name,
                    "dets": dets_label,
                    "cand": cand_label,
                    "anchor": label,
                    "rk": rank,
                    "ok": valid_frame,
                    "sel": selected,
                    "src": source,
                    "why": why,
                    "lu": local_use,
                    "lur": local_reason,
                    "obs": item.get("frame_valid_obs", ""),
                    "sep": fmt(item.get("pair_gap", item.get("frame_sep", None))),
                    "m1": str(frame_modes[0]) if len(frame_modes) >= 1 else "",
                    "m2": str(frame_modes[1]) if len(frame_modes) >= 2 else "",
                    "g1": fmt(frame_distances[0]) if len(frame_distances) >= 1 else "",
                    "g2": fmt(frame_distances[1]) if len(frame_distances) >= 2 else "",
                }
            )

    df_hist = compact_columns(
        pd.DataFrame(hist_rows),
        {
            "class": 10,
            "dets": 10,
            "cand": 14,
            "anchor": 8,
            "src": 6,
            "why": 8,
        },
    )
    print(f"\n[DEBUG][PairAnchorsHist] frame={frame_id}")
    print_table_auto(
        df_hist,
        cols=["class", "dets", "cand", "anchor", "src", "why", "u", "bu", "pu", "pc", "pm", "pr", "pb", "hm"],
        col_space=2,
        pinned_cols=["class", "anchor"],
        headers={"class": "cls", "anchor": "anc", "hm": "hgap"},
    )
    df_frame = compact_columns(
        pd.DataFrame(frame_rows),
        {
            "class": 10,
            "dets": 10,
            "cand": 14,
            "anchor": 8,
            "src": 6,
            "why": 8,
            "lur": 10,
        },
    )
    print(f"\n[DEBUG][PairAnchorsFrame] frame={frame_id}")
    print_table_auto(
        df_frame,
        cols=["class", "dets", "cand", "anchor", "rk", "ok", "sel", "src", "why", "lu", "lur", "obs", "sep", "m1", "m2", "g1", "g2"],
        col_space=2,
        pinned_cols=["class", "anchor"],
        headers={
            "class": "cls",
            "anchor": "anc",
            "lu": "gap",
            "obs": "nobs",
            "sep": "dg",
            "g1": "d1",
            "g2": "d2",
        },
    )
