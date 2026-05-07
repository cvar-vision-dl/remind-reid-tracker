from __future__ import annotations

from utils.debug.association.debug_assoc_helpers import (
    candidate_debug_reason,
    candidate_debug_state,
    candidate_decision_keep,
    candidate_known_plausible_keep,
)


class DebugViewBuilder:
    def __init__(self, *, debug_topk: int, sets_provider):
        self.debug_topk = int(debug_topk)
        self.sets_provider = sets_provider

    def ensure_out_debug_schema(self, out) -> dict:
        dbg = getattr(out, "debug", None)
        if not isinstance(dbg, dict):
            dbg = {}
            out.debug = dbg

        dbg.setdefault("schema_version", "1.0")
        canon = dbg.setdefault("canon", {})
        canon.setdefault("frame", {})
        canon.setdefault("reports", {})
        dbg.setdefault("extra", {})
        return dbg

    def topk_pairs(self, candidates: list[dict], key: str, k: int) -> list[tuple[int, float]]:
        scored = []
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            oid = candidate.get("object_id", None)
            if oid is None:
                continue
            scored.append((int(oid), float(candidate.get(key, 0.0))))
        scored.sort(key=lambda item: float(item[1]), reverse=True)
        return [(int(oid), float(score)) for oid, score in scored[: max(0, int(k))]]

    def ranks_by_object_id(self, candidates: list[dict], key: str) -> dict[int, int]:
        scored = []
        for candidate in candidates or []:
            if not isinstance(candidate, dict):
                continue
            oid = candidate.get("object_id", None)
            if oid is None:
                continue
            scored.append((float(candidate.get(key, 0.0)), int(oid)))
        scored.sort(key=lambda item: float(item[0]), reverse=True)

        out = {}
        for rank, (_, oid) in enumerate(scored, start=1):
            out[int(oid)] = int(rank)
        return out

    def build(self, *, out, frame_id: int | None) -> None:
        dbg = self.ensure_out_debug_schema(out)

        canon = dbg.get("canon", {})
        frame = canon.get("frame", {})
        frame["timestamp"] = float(getattr(out, "timestamp", 0.0))
        frame["frame_id"] = None if frame_id is None else int(frame_id)
        frame["summary"] = dict(getattr(out, "frame_summary", {}) or {})
        frame["reliable_anchor_object_ids"] = [int(x) for x in (getattr(out, "reliable_anchor_object_ids", []) or [])]
        frame["decisions"] = {
            "decided_matches": list(getattr(out, "decided_matches", []) or []),
            "to_create": list(getattr(out, "to_create", []) or []),
            "to_ambiguous": list(getattr(out, "to_ambiguous", []) or []),
            "to_provisional_new": list(getattr(out, "to_provisional_new", []) or []),
            "assigned_by_det_id": dict(getattr(out, "assigned_by_det_id", {}) or {}),
        }

        reports_dbg = canon.get("reports", {})
        reports_dbg.clear()

        k = max(0, int(self.debug_topk))
        for det_id, rep in (getattr(out, "reports_by_det_id", {}) or {}).items():
            did = int(det_id)
            candidates = getattr(rep, "candidates", None) or []
            best = getattr(rep, "best", None)

            ranks_sim = self.ranks_by_object_id(candidates, key="score_sim")
            ranks_final = self.ranks_by_object_id(candidates, key="score_final")
            topk_sim = self.topk_pairs(candidates, key="score_sim", k=k)
            topk_final = self.topk_pairs(candidates, key="score_final", k=k)

            best_sim_oid = None
            if isinstance(best, dict) and best.get("object_id", None) is not None:
                best_sim_oid = int(best["object_id"])

            final_oid = getattr(rep, "final_object_id", None)
            override = bool(best_sim_oid is not None and final_oid is not None and int(best_sim_oid) != int(final_oid))

            candidates_sorted = []
            if k > 0 and isinstance(candidates, list) and candidates:
                candidates_sorted = sorted(
                    (candidate for candidate in candidates if isinstance(candidate, dict)),
                    key=lambda candidate: float(candidate.get("score_final", 0.0)),
                    reverse=True,
                )[:k]

            cand_pack = []
            for candidate in candidates_sorted:
                oid = candidate.get("object_id", None)
                if oid is None:
                    continue
                oid = int(oid)
                cand_pack.append(
                    {
                        "object_id": int(oid),
                        "class_id": int(candidate.get("class_id", -1)),
                        "score_sim": float(candidate.get("score_sim", 0.0)),
                        "score_sets": float(candidate.get("score_sets", 0.0)),
                        "bonus_sets": float(candidate.get("bonus_sets", 0.0)),
                        "penalty_sets": float(candidate.get("penalty_sets", 0.0)),
                        "support_sets": float(candidate.get("support_sets", 0.0)),
                        "support_local_sets": float(candidate.get("support_local_sets", 0.0)),
                        "support_global_sets": float(candidate.get("support_global_sets", 0.0)),
                        "quality_sets": float(candidate.get("quality_sets", 0.0)),
                        "score_ctx_local": float(candidate.get("score_ctx_local", 0.0)),
                        "score_ctx_global": float(candidate.get("score_ctx_global", 0.0)),
                        "compat_rel": float(candidate.get("compat_rel", 0.0)),
                        "compat_band": int(candidate.get("compat_band", 0)),
                        "kernel_raw": float(candidate.get("kernel_raw", 0.0)),
                        "kernel_rel": float(candidate.get("kernel_rel", 0.0)),
                        "hyp_rel": float(candidate.get("hyp_rel", 0.0)),
                        "score_final": float(candidate.get("score_final", 0.0)),
                        "decision_keep": int(candidate_decision_keep(candidate)),
                        "known_plausible_keep": int(candidate_known_plausible_keep(candidate)),
                        "decision_state": str(candidate_debug_state(candidate, selected_track_id=final_oid)),
                        "decision_reason": str(candidate_debug_reason(candidate, selected_track_id=final_oid)),
                        "sets_ctx_summary": str(candidate.get("sets_ctx_summary", "") or ""),
                        "sets_class_summary": str(candidate.get("sets_class_summary", "") or ""),
                        "sets_policy_summary": str(candidate.get("sets_policy_summary", "") or ""),
                        "sets_trace": candidate.get("sets_trace", {}) or {},
                        "scores": candidate.get("scores", {}) or {},
                        "rank_sim": int(ranks_sim.get(oid, 0)),
                        "rank_final": int(ranks_final.get(oid, 0)),
                    }
                )

            reports_dbg[did] = {
                "det_id": int(getattr(rep, "det_id", did)),
                "class_id": int(getattr(rep, "class_id", -1)),
                "det_geom": getattr(rep, "det_geom", None),
                "final": {
                    "decision": str(getattr(rep, "final_decision", "")),
                    "object_id": None if final_oid is None else int(final_oid),
                    "score_final": float(getattr(rep, "final_score", 0.0)),
                    "reason": str(getattr(rep, "final_reason", "")),
                },
                "diag_sim": getattr(rep, "match_diag_sim", None),
                "diag_final": getattr(rep, "match_diag_final", None),
                "best_sim_oid": best_sim_oid,
                "topk_sim": list(topk_sim),
                "topk_final": list(topk_final),
                "override": {
                    "is_override": bool(override),
                    "best_sim_oid": best_sim_oid,
                    "final_oid": None if final_oid is None else int(final_oid),
                },
                "candidates": cand_pack,
            }

        extra = dbg.get("extra", {})
        ns_pack = extra.setdefault("ns", {})
        ns_pack["ctx"] = self.sets_provider.build_context(getattr(out, "neighbor_sets_out", None))
        extra.setdefault("hungarian", None)
        extra.setdefault("perf", {})
        extra.setdefault("notes", [])
