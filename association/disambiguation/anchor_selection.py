from __future__ import annotations

import math


class AnchorSelectionMixin:
    def build_frame_pair_anchor_rows(
        self,
        *,
        component_rows: list[dict],
        det_geom_by_id: dict[int, dict],
        anchor_geom_by_oid: dict[int, dict],
    ) -> list[dict]:
        rows = []
        for pack in (component_rows or []):
            if not isinstance(pack, dict):
                continue
            det_ids = [int(x) for x in (pack.get("det_ids", []) or []) if x is not None]
            candidate_union = [int(x) for x in (pack.get("candidate_union", []) or []) if x is not None]
            if not det_ids or not candidate_union:
                continue
            class_id = -1
            for oid in candidate_union:
                obj = None if self.memory_store is None else self.memory_store.get(int(oid))
                if obj is None:
                    continue
                class_id = int(getattr(obj, "class_id", -1))
                if class_id >= 0:
                    break

            component = {
                "det_ids": list(det_ids),
                "candidate_union": list(candidate_union),
            }
            anchor_rows_by_id: dict[int, dict] = {}
            raw_candidates = pack.get("anchor_candidates", None)
            if isinstance(raw_candidates, list) and raw_candidates:
                for item in raw_candidates:
                    if isinstance(item, dict):
                        anchor_id = int(item.get("anchor_id", -1))
                        if anchor_id < 0:
                            continue
                        anchor_desc = self.pair_anchor_discriminator.describe_anchor_for_pair(
                            anchor_id=int(anchor_id),
                            candidate_ids=list(candidate_union),
                            det_ids=list(det_ids),
                            det_geom_by_id=det_geom_by_id,
                            anchor_geom_by_oid=anchor_geom_by_oid,
                        )
                        merged = dict(item)
                        merged["usefulness"] = float(anchor_desc.get("history_usefulness", 0.0) or 0.0)
                        merged["raw_usefulness"] = float(anchor_desc.get("history_usefulness", 0.0) or 0.0)
                        merged["base_usefulness"] = float(anchor_desc.get("base_usefulness", 0.0) or 0.0)
                        merged["pair_usefulness"] = float(anchor_desc.get("pair_usefulness", 0.0) or 0.0)
                        merged["obs_modes"] = list(anchor_desc.get("obs_modes", []) or [])
                        merged["obs_primary"] = list(anchor_desc.get("obs_primary", []) or [])
                        merged["obs_gap"] = list(anchor_desc.get("obs_gap", []) or [])
                        merged["gap_margin_mean"] = anchor_desc.get("gap_margin_mean", None)
                        merged["history_usefulness"] = float(anchor_desc.get("history_usefulness", 0.0) or 0.0)
                        merged["local_usefulness"] = float(item.get("frame_usefulness", item.get("usefulness", item.get("score", 0.0))) or 0.0)
                        merged["local_reason"] = str(item.get("frame_reason", item.get("local_reason", "")) or "")
                        merged["frame_modes"] = list(item.get("frame_modes", []) or [])
                        merged["frame_distances"] = list(item.get("frame_distances", item.get("frame_gap", [])) or [])
                        merged["pair_gap"] = item.get("pair_gap", item.get("frame_sep", None))
                        merged["frame_valid_obs"] = item.get("frame_valid_obs", None)
                        merged["frame_valid"] = bool(item.get("frame_valid", False))
                        merged["pair_consistency"] = anchor_desc.get("pair_consistency", None)
                        merged["pair_margin_mean"] = anchor_desc.get("pair_margin_mean", None)
                        merged["pair_reliability"] = anchor_desc.get("pair_reliability", None)
                        merged["pair_robustness"] = anchor_desc.get("pair_robustness", None)
                        merged["debug_rank_local"] = item.get("rank", None)
                        anchor_rows_by_id[int(anchor_id)] = merged
            else:
                for anchor_id in (pack.get("anchor_ids", []) or []):
                    anchor_id = int(anchor_id)
                    anchor_desc = self.pair_anchor_discriminator.describe_anchor_for_pair(
                        anchor_id=int(anchor_id),
                        candidate_ids=list(candidate_union),
                        det_ids=list(det_ids),
                        det_geom_by_id=det_geom_by_id,
                        anchor_geom_by_oid=anchor_geom_by_oid,
                    )
                    frame_pack = self.component_anchor_frame_usefulness_details(
                        component=component,
                        anchor_id=int(anchor_id),
                        det_geom_by_id=det_geom_by_id,
                        anchor_geom_by_oid=anchor_geom_by_oid,
                    )
                    anchor_rows_by_id[int(anchor_id)] = {
                        "anchor_id": int(anchor_id),
                        "usefulness": float(anchor_desc.get("history_usefulness", 0.0) or 0.0),
                        "raw_usefulness": float(anchor_desc.get("history_usefulness", 0.0) or 0.0),
                        "base_usefulness": float(anchor_desc.get("base_usefulness", 0.0) or 0.0),
                        "pair_usefulness": float(anchor_desc.get("pair_usefulness", 0.0) or 0.0),
                        "history_usefulness": float(anchor_desc.get("history_usefulness", 0.0) or 0.0),
                        "local_usefulness": float(frame_pack.get("score", 0.0) or 0.0),
                        "local_reason": str(frame_pack.get("reason", "sel_only") or "sel_only"),
                        "frame_modes": list(frame_pack.get("obs_modes", []) or []),
                        "frame_distances": list(frame_pack.get("obs_distances", frame_pack.get("obs_gap", [])) or []),
                        "pair_gap": frame_pack.get("pair_gap", frame_pack.get("obs_sep", None)),
                        "frame_valid_obs": int(frame_pack.get("valid_obs", 0) or 0),
                        "frame_valid": bool(float(frame_pack.get("score", 0.0) or 0.0) >= float(self.min_anchor_informativeness)),
                        "obs_modes": list(anchor_desc.get("obs_modes", []) or []),
                        "obs_primary": list(anchor_desc.get("obs_primary", []) or []),
                        "obs_gap": list(anchor_desc.get("obs_gap", []) or []),
                        "gap_margin_mean": anchor_desc.get("gap_margin_mean", None),
                        "pair_consistency": anchor_desc.get("pair_consistency", None),
                        "pair_margin_mean": anchor_desc.get("pair_margin_mean", None),
                        "pair_reliability": anchor_desc.get("pair_reliability", None),
                        "pair_robustness": anchor_desc.get("pair_robustness", None),
                        "selected": True,
                        "valid": True,
                        "source": "selected_only",
                        "why": "selected",
                        "debug_rank_local": None,
                    }

            historical_rows = self.pair_anchor_discriminator.rank_historical_anchors(
                candidate_ids=list(candidate_union),
                det_ids=list(det_ids),
                det_geom_by_id=det_geom_by_id,
                anchor_geom_by_oid=anchor_geom_by_oid,
                excluded_anchor_ids=set(anchor_rows_by_id.keys()),
                limit=int(self.debug_historical_anchor_topk),
            )
            for item in historical_rows:
                anchor_id = int(item.get("anchor_id", -1))
                if anchor_id < 0 or int(anchor_id) in anchor_rows_by_id:
                    continue
                frame_pack = self.component_anchor_frame_usefulness_details(
                    component=component,
                    anchor_id=int(anchor_id),
                    det_geom_by_id=det_geom_by_id,
                    anchor_geom_by_oid=anchor_geom_by_oid,
                )
                merged = dict(item)
                merged["local_usefulness"] = float(frame_pack.get("score", 0.0) or 0.0)
                merged["local_reason"] = str(frame_pack.get("reason", item.get("local_reason", "")) or "")
                merged["frame_modes"] = list(frame_pack.get("obs_modes", []) or [])
                merged["frame_distances"] = list(frame_pack.get("obs_distances", frame_pack.get("obs_gap", [])) or [])
                merged["pair_gap"] = frame_pack.get("pair_gap", frame_pack.get("obs_sep", None))
                merged["frame_valid_obs"] = int(frame_pack.get("valid_obs", 0) or 0)
                merged["frame_valid"] = bool(float(frame_pack.get("score", 0.0) or 0.0) >= float(self.min_anchor_informativeness))
                anchor_rows_by_id[int(anchor_id)] = merged

            anchor_rows = sorted(
                anchor_rows_by_id.values(),
                key=lambda item: (
                    int(bool(item.get("selected", False))),
                    int(self.anchor_debug_source_priority(str(item.get("source", "") or ""))),
                    int(bool(item.get("valid", False))),
                    float(item.get("local_usefulness", item.get("usefulness", 0.0)) or 0.0),
                    -int(item.get("rank", 10**6) or 10**6),
                    -int(item.get("anchor_id", -1) or -1),
                ),
                reverse=True,
            )
            rows.append(
                {
                    "class_id": int(class_id),
                    "det_ids": list(det_ids),
                    "candidate_union": list(candidate_union),
                    "anchor_ids": [int(x) for x in (pack.get("anchor_ids", []) or [])],
                    "anchors": list(anchor_rows),
                    "status": str(pack.get("status", "") or ""),
                }
            )
        return list(rows)

    def anchor_debug_source_priority(self, source: str) -> int:
        src = str(source or "")
        if src in ("selected_only", "match", "soft"):
            return 3
        if src == "history_visible":
            return 2
        if src == "history":
            return 1
        return 0

    def build_matched_geom_by_oid(
        self,
        *,
        decided_matches: list[tuple[int, int, float]],
        det_geom_by_id: dict[int, dict],
    ) -> dict[int, dict]:
        out = {}
        for det_id, oid, _ in (decided_matches or []):
            geom = det_geom_by_id.get(int(det_id), None)
            if isinstance(geom, dict):
                out[int(oid)] = geom
        return out

    def build_soft_anchor_pool(
        self,
        *,
        reports_by_det_id: dict,
        det_geom_by_id: dict[int, dict],
        decided_matches: list[tuple[int, int, float]],
        excluded_det_ids: set[int] | None = None,
    ) -> dict[int, dict]:
        if not self.soft_anchors_enabled:
            return {}

        decided_det_ids = {int(det_id) for det_id, _, _ in (decided_matches or [])}
        excluded = {int(det_id) for det_id in (excluded_det_ids or set())}
        pool: dict[int, dict] = {}
        for det_id, rep in (reports_by_det_id or {}).items():
            det_id = int(det_id)
            if det_id in decided_det_ids:
                continue
            if det_id in excluded:
                continue
            geom = det_geom_by_id.get(int(det_id), None)
            if not isinstance(geom, dict):
                continue
            candidates = getattr(rep, "candidates", None) or []
            supported = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                oid = candidate.get("object_id", None)
                if oid is None:
                    continue
                oid = int(oid)
                if not self.is_real_object_id(int(oid)):
                    continue
                if int(candidate.get("decision_keep", 0) or 0) != 1:
                    continue
                score = self.resolve_candidate_score(rep, int(oid))
                if float(score) < float(self.soft_anchor_min_score):
                    continue
                supported.append((float(score), int(oid)))
            if not supported:
                continue
            supported.sort(key=lambda item: (float(item[0]), int(item[1])), reverse=True)
            best_score, best_oid = supported[0]
            prev = pool.get(int(best_oid), None)
            if prev is None or float(best_score) > float(prev.get("score", 0.0)):
                pool[int(best_oid)] = {
                    "oid": int(best_oid),
                    "det_id": int(det_id),
                    "score": float(best_score),
                    "geom": dict(geom),
                }

        ranked = sorted(
            pool.items(),
            key=lambda item: (float((item[1] or {}).get("score", 0.0)), int(item[0])),
            reverse=True,
        )
        if int(self.soft_anchor_max) > 0:
            ranked = ranked[: int(self.soft_anchor_max)]
        return {int(oid): dict(pack) for oid, pack in ranked}

    def select_anchor_ids(
        self,
        *,
        component: dict,
        decided_matches: list[tuple[int, int, float]],
        det_geom_by_id: dict[int, dict] | None = None,
        anchor_geom_by_oid: dict[int, dict] | None = None,
        soft_anchor_pool: dict[int, dict] | None = None,
    ) -> list[int]:
        ranked = self.rank_anchor_candidates(
            component=component,
            decided_matches=decided_matches,
            det_geom_by_id=det_geom_by_id,
            anchor_geom_by_oid=anchor_geom_by_oid,
            soft_anchor_pool=soft_anchor_pool,
        )
        return [int(item["anchor_id"]) for item in ranked if bool(item.get("selected", False))]

    def rank_anchor_candidates(
        self,
        *,
        component: dict,
        decided_matches: list[tuple[int, int, float]],
        det_geom_by_id: dict[int, dict] | None = None,
        anchor_geom_by_oid: dict[int, dict] | None = None,
        soft_anchor_pool: dict[int, dict] | None = None,
    ) -> list[dict]:
        candidate_union = set(int(x) for x in (component.get("candidate_union", []) or []))
        matched_det_ids = set(int(x) for x in (component.get("det_ids", []) or []))
        scored = []
        for det_id, obj_id, _ in (decided_matches or []):
            if int(det_id) in matched_det_ids:
                continue
            if int(obj_id) in candidate_union:
                continue
            anchor_quality = self.component_anchor_quality_details(
                component=component,
                anchor_id=int(obj_id),
                det_geom_by_id=det_geom_by_id,
                anchor_geom_by_oid=anchor_geom_by_oid,
            )
            scored.append(
                {
                    "anchor_id": int(obj_id),
                    "score": float(anchor_quality.get("combined_usefulness", 0.0) or 0.0),
                    "raw_score": float(anchor_quality.get("combined_usefulness", 0.0) or 0.0),
                    "local_reason": str(anchor_quality.get("frame_reason", "") or ""),
                    "history_usefulness": float(anchor_quality.get("history_usefulness", 0.0) or 0.0),
                    "frame_usefulness": float(anchor_quality.get("frame_usefulness", 0.0) or 0.0),
                    "frame_strength": float(anchor_quality.get("frame_strength", 0.0) or 0.0),
                    "combined_usefulness": float(anchor_quality.get("combined_usefulness", 0.0) or 0.0),
                    "frame_reason": str(anchor_quality.get("frame_reason", "") or ""),
                    "frame_modes": list(anchor_quality.get("frame_modes", []) or []),
                    "frame_distances": list(anchor_quality.get("frame_distances", []) or []),
                    "pair_gap": anchor_quality.get("pair_gap", None),
                    "frame_valid_obs": int(anchor_quality.get("frame_valid_obs", 0) or 0),
                    "frame_valid": bool(float(anchor_quality.get("frame_strength", 0.0) or 0.0) >= float(self.min_anchor_informativeness)),
                    "source": "match",
                    "conf": None,
                }
            )

        for anchor_id, pack in ((soft_anchor_pool or {}) or {}).items():
            if int(anchor_id) in candidate_union:
                continue
            if int((pack or {}).get("det_id", -1)) in matched_det_ids:
                continue
            anchor_quality = self.component_anchor_quality_details(
                component=component,
                anchor_id=int(anchor_id),
                det_geom_by_id=det_geom_by_id,
                anchor_geom_by_oid=anchor_geom_by_oid,
            )
            conf = float((pack or {}).get("score", 0.0) or 0.0)
            conf = max(0.0, min(1.0, conf))
            adj = float((1.0 - self.soft_anchor_conf_weight) + (self.soft_anchor_conf_weight * conf))
            scored.append(
                {
                    "anchor_id": int(anchor_id),
                    "score": float(float(anchor_quality.get("combined_usefulness", 0.0) or 0.0) * adj),
                    "raw_score": float(anchor_quality.get("combined_usefulness", 0.0) or 0.0),
                    "local_reason": str(anchor_quality.get("frame_reason", "") or ""),
                    "history_usefulness": float(anchor_quality.get("history_usefulness", 0.0) or 0.0),
                    "frame_usefulness": float(anchor_quality.get("frame_usefulness", 0.0) or 0.0),
                    "frame_strength": float(anchor_quality.get("frame_strength", 0.0) or 0.0),
                    "combined_usefulness": float(anchor_quality.get("combined_usefulness", 0.0) or 0.0),
                    "frame_reason": str(anchor_quality.get("frame_reason", "") or ""),
                    "frame_modes": list(anchor_quality.get("frame_modes", []) or []),
                    "frame_distances": list(anchor_quality.get("frame_distances", []) or []),
                    "pair_gap": anchor_quality.get("pair_gap", None),
                    "frame_valid_obs": int(anchor_quality.get("frame_valid_obs", 0) or 0),
                    "frame_valid": bool(float(anchor_quality.get("frame_strength", 0.0) or 0.0) >= float(self.min_anchor_informativeness)),
                    "source": "soft",
                    "conf": float(conf),
                }
            )
        scored.sort(key=lambda item: (float(item.get("score", 0.0)), int(item.get("anchor_id", -1))), reverse=True)
        topk = min(int(self.max_anchors), max(2, min(int(self.anchor_pair_topk), int(self.discriminative_anchor_topk))))
        ranked = []
        best_anchor_score = float(scored[0].get("score", 0.0) or 0.0) if scored else 0.0
        selected_score_floor = max(
            float(self.min_anchor_informativeness),
            float(best_anchor_score * float(self.selected_anchor_score_ratio_min)),
        )
        selected_ids = {
            int(item.get("anchor_id", -1))
            for item in scored[: int(topk)]
            if float(item.get("score", 0.0) or 0.0) >= float(selected_score_floor)
        }
        for rank, item in enumerate(scored, start=1):
            anchor_id = int(item.get("anchor_id", -1))
            anchor_desc = self.pair_anchor_discriminator.describe_anchor_for_pair(
                anchor_id=int(anchor_id),
                candidate_ids=list(candidate_union),
                det_ids=[int(x) for x in (component.get("det_ids", []) or [])],
                det_geom_by_id=det_geom_by_id,
                anchor_geom_by_oid=anchor_geom_by_oid,
            )
            base_use = float(anchor_desc.get("base_usefulness", 0.0) or 0.0)
            pair_use = float(anchor_desc.get("pair_usefulness", 0.0) or 0.0)
            valid = bool(float(item.get("score", 0.0) or 0.0) >= float(self.min_anchor_informativeness))
            selected = bool(anchor_id in selected_ids)
            why = "selected" if selected else ("valid_but_not_selected" if valid else "below_min_anchor_informativeness")
            ranked.append(
                {
                    "anchor_id": int(anchor_id),
                    "rank": int(rank),
                    "usefulness": float(item.get("score", 0.0) or 0.0),
                    "raw_usefulness": float(item.get("raw_score", 0.0) or 0.0),
                    "base_usefulness": float(base_use),
                    "pair_usefulness": float(pair_use),
                    "obs_modes": list(anchor_desc.get("obs_modes", []) or []),
                    "obs_primary": list(anchor_desc.get("obs_primary", []) or []),
                    "obs_gap": list(anchor_desc.get("obs_gap", []) or []),
                    "gap_margin_mean": anchor_desc.get("gap_margin_mean", None),
                    "pair_consistency": anchor_desc.get("pair_consistency", None),
                    "pair_margin_mean": anchor_desc.get("pair_margin_mean", None),
                    "pair_reliability": anchor_desc.get("pair_reliability", None),
                    "pair_robustness": anchor_desc.get("pair_robustness", None),
                    "selected": bool(selected),
                    "valid": bool(valid),
                    "source": str(item.get("source", "") or ""),
                    "conf": item.get("conf", None),
                    "local_reason": str(item.get("local_reason", "") or ""),
                    "frame_usefulness": float(item.get("frame_usefulness", item.get("score", 0.0)) or 0.0),
                    "frame_strength": float(item.get("frame_strength", 0.0) or 0.0),
                    "history_usefulness": float(item.get("history_usefulness", anchor_desc.get("history_usefulness", 0.0)) or 0.0),
                    "combined_usefulness": float(item.get("combined_usefulness", item.get("score", 0.0)) or 0.0),
                    "frame_reason": str(item.get("frame_reason", item.get("local_reason", "")) or ""),
                    "frame_modes": list(item.get("frame_modes", []) or []),
                    "frame_distances": list(item.get("frame_distances", item.get("frame_gap", [])) or []),
                    "pair_gap": item.get("pair_gap", item.get("frame_sep", None)),
                    "frame_valid_obs": int(item.get("frame_valid_obs", 0) or 0),
                    "frame_valid": bool(item.get("frame_valid", valid)),
                    "why": str(why),
                }
            )
        return list(ranked)

    def frame_gap_strength(self, gap_value: float | None) -> float:
        if gap_value is None or not math.isfinite(float(gap_value)):
            return 0.0
        if float(gap_value) <= 0.0:
            return 0.0
        return float(
            self.pair_anchor_discriminator.distance_strength(
                float(gap_value),
                ref=float(self.anchor_span_ref),
            )
        )

    def component_anchor_quality_details(
        self,
        *,
        component: dict,
        anchor_id: int,
        det_geom_by_id: dict[int, dict] | None = None,
        anchor_geom_by_oid: dict[int, dict] | None = None,
    ) -> dict[str, float | str | list]:
        candidate_ids = [int(x) for x in (component.get("candidate_union", []) or []) if x is not None]
        det_ids = [int(x) for x in (component.get("det_ids", []) or []) if x is not None]
        anchor_desc = self.pair_anchor_discriminator.describe_anchor_for_pair(
            anchor_id=int(anchor_id),
            candidate_ids=list(candidate_ids),
            det_ids=list(det_ids),
            det_geom_by_id=det_geom_by_id,
            anchor_geom_by_oid=anchor_geom_by_oid,
        )
        frame_pack = self.component_anchor_frame_usefulness_details(
            component=component,
            anchor_id=int(anchor_id),
            det_geom_by_id=det_geom_by_id,
            anchor_geom_by_oid=anchor_geom_by_oid,
        )
        hist_use = float(anchor_desc.get("history_usefulness", 0.0) or 0.0)
        frame_use = float(frame_pack.get("score", 0.0) or 0.0)
        frame_strength = float(self.frame_gap_strength(frame_use))
        total_weight = float(self.anchor_history_score_weight) + float(self.anchor_frame_score_weight)
        if total_weight <= 1e-12:
            combined = float(0.5 * (hist_use + frame_strength))
        else:
            combined = float(
                (
                    (float(self.anchor_history_score_weight) * hist_use)
                    + (float(self.anchor_frame_score_weight) * frame_strength)
                )
                / total_weight
            )
        return {
            "history_usefulness": float(hist_use),
            "frame_usefulness": float(frame_use),
            "frame_strength": float(frame_strength),
            "combined_usefulness": float(combined),
            "frame_reason": str(frame_pack.get("reason", "") or ""),
            "frame_modes": list(frame_pack.get("obs_modes", []) or []),
            "frame_distances": list(frame_pack.get("obs_distances", frame_pack.get("obs_gap", [])) or []),
            "pair_gap": frame_pack.get("pair_gap", frame_pack.get("obs_sep", None)),
            "frame_valid_obs": int(frame_pack.get("valid_obs", 0) or 0),
            "base_usefulness": float(anchor_desc.get("base_usefulness", 0.0) or 0.0),
            "pair_usefulness": float(anchor_desc.get("pair_usefulness", 0.0) or 0.0),
        }

    def build_component_anchor_quality_map(
        self,
        *,
        component: dict,
        anchor_ids: list[int],
        det_geom_by_id: dict[int, dict] | None = None,
        anchor_geom_by_oid: dict[int, dict] | None = None,
    ) -> dict[int, dict]:
        out = {}
        for anchor_id in (anchor_ids or []):
            anchor_id = int(anchor_id)
            out[int(anchor_id)] = dict(
                self.component_anchor_quality_details(
                    component=component,
                    anchor_id=int(anchor_id),
                    det_geom_by_id=det_geom_by_id,
                    anchor_geom_by_oid=anchor_geom_by_oid,
                )
            )
        return out

    def component_anchor_frame_usefulness_details(
        self,
        *,
        component: dict,
        anchor_id: int,
        det_geom_by_id: dict[int, dict] | None = None,
        anchor_geom_by_oid: dict[int, dict] | None = None,
    ) -> dict[str, float | str]:
        det_ids = [int(x) for x in (component.get("det_ids", []) or [])]
        if len(det_ids) < 2:
            return {"score": 0.0, "reason": "1det", "obs_modes": [], "obs_distances": [], "pair_gap": None, "valid_obs": 0}
        det_geom_by_id = dict(det_geom_by_id or {})
        anchor_geom = (anchor_geom_by_oid or {}).get(int(anchor_id), None)
        if not isinstance(anchor_geom, dict):
            return {"score": 0.0, "reason": "no_geom", "obs_modes": [], "obs_distances": [], "pair_gap": None, "valid_obs": 0}

        observed_gap = []
        obs_modes = []
        obs_distances = []
        for det_id in det_ids:
            det_geom = det_geom_by_id.get(int(det_id), None)
            if not isinstance(det_geom, dict):
                continue
            obs = self.relation_observation_cached(
                det_geom,
                anchor_geom,
                scale_min=40.0,
                geom_a_key=("det", int(det_id)),
                geom_b_key=("anchor", int(anchor_id)),
            )
            if not isinstance(obs, dict):
                continue
            gap_value = float(obs.get("mask_gap_n", float("nan")))
            if not math.isfinite(float(gap_value)):
                continue
            obs_modes.append("M")
            obs_distances.append(float(gap_value))
            observed_gap.append(float(gap_value))

        if len(observed_gap) < 2:
            return {
                "score": 0.0,
                "reason": "obs<2",
                "obs_modes": list(obs_modes),
                "obs_distances": list(obs_distances),
                "pair_gap": None,
                "valid_obs": int(len(observed_gap)),
            }

        obs_sep = float(max(observed_gap) - min(observed_gap))
        frame_score = float(obs_sep)
        if float(frame_score) <= 0.0:
            reason = "same_gap" if float(obs_sep) <= 1e-9 else "zero"
            return {
                "score": 0.0,
                "reason": str(reason),
                "obs_modes": list(obs_modes),
                "obs_distances": list(obs_distances),
                "pair_gap": float(obs_sep),
                "valid_obs": int(len(observed_gap)),
            }
        return {
            "score": float(frame_score),
            "reason": "ok",
            "obs_modes": list(obs_modes),
            "obs_distances": list(obs_distances),
            "pair_gap": float(obs_sep),
            "valid_obs": int(len(observed_gap)),
        }
