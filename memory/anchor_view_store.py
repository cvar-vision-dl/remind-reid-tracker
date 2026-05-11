from __future__ import annotations

import math

try:
    from memory.neighbor_distance_graph import compute_relation_observation
except ModuleNotFoundError:  # Allow direct execution from this module directory.
    from neighbor_distance_graph import compute_relation_observation


def pair_anchor_ids(pair_features: dict[tuple[int, int], dict]) -> set[int]:
    out: set[int] = set()
    for a, b in (pair_features or {}).keys():
        out.add(int(a))
        out.add(int(b))
    return out


class AnchorViewSignature:
    def __init__(self, view_id: int, anchor_ids: list[int], pair_features: dict[tuple[int, int], dict]):
        self.view_id = int(view_id)
        self.anchor_ids = tuple(int(x) for x in (anchor_ids or []))
        self.pair_features = {(int(a), int(b)): dict(v) for (a, b), v in (pair_features or {}).items()}
        self.support_count = 1


class AnchorViewStore:
    def __init__(self, config: dict | None = None):
        cfg = (config or {}).get("anchor_views", {}) or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.min_anchors = max(2, int(cfg.get("min_anchors", 3)))
        self.max_anchors = max(self.min_anchors, int(cfg.get("max_anchors", 5)))
        self.max_views = max(1, int(cfg.get("max_views", 12)))
        self.match_min_score = max(0.0, min(1.0, float(cfg.get("match_min_score", 0.72))))
        self.dist_sigma = max(1e-6, float(cfg.get("dist_sigma", 0.35)))
        self.contact_weight = max(0.0, min(1.0, float(cfg.get("contact_weight", 0.35))))
        self.support_weight = max(0.0, min(1.0, float(cfg.get("support_weight", 0.15))))
        self.next_view_id = 0
        self.views: dict[int, AnchorViewSignature] = {}

    def build_signature(self, anchor_geom_by_oid: dict[int, dict]) -> tuple[list[int], dict[tuple[int, int], dict]]:
        ids = sorted(int(x) for x in (anchor_geom_by_oid or {}).keys())
        ids = ids[: self.max_anchors]
        pair_features: dict[tuple[int, int], dict] = {}
        for i, a in enumerate(ids):
            geom_a = anchor_geom_by_oid.get(int(a), None)
            if not isinstance(geom_a, dict):
                continue
            for b in ids[i + 1 :]:
                geom_b = anchor_geom_by_oid.get(int(b), None)
                obs = compute_relation_observation(
                    geom_a,
                    geom_b,
                    scale_min=40.0,
                    contact_margin_px=2.0,
                    near_thresh_n=1.25,
                    exact_gap_max_n=1.75,
                )
                if not isinstance(obs, dict):
                    continue
                pair_features[(int(a), int(b))] = {
                    "center_dist_n": float(obs.get("center_dist_n", 0.0)),
                    "mask_gap_n": float(obs.get("mask_gap_n", 0.0)),
                    "contact_state": str(obs.get("contact_state", "separate")),
                    "support_like": float(obs.get("support_like", 0.0)),
                }
        return ids, pair_features

    def compare_signature(self, pair_features: dict[tuple[int, int], dict], view: AnchorViewSignature) -> float:
        if not pair_features or not isinstance(view, AnchorViewSignature):
            return 0.0
        common = set(pair_features.keys()) & set(view.pair_features.keys())
        if not common:
            return 0.0

        vals = []
        for key in common:
            cur = pair_features.get(key, {}) or {}
            ref = view.pair_features.get(key, {}) or {}
            d_cur = float(cur.get("center_dist_n", 0.0))
            d_ref = float(ref.get("center_dist_n", 0.0))
            g_cur = float(cur.get("mask_gap_n", 0.0))
            g_ref = float(ref.get("mask_gap_n", 0.0))
            s_cur = float(cur.get("support_like", 0.0))
            s_ref = float(ref.get("support_like", 0.0))
            dist_score = math.exp(-0.5 * (((d_cur - d_ref) / self.dist_sigma) ** 2))
            gap_score = math.exp(-0.5 * (((g_cur - g_ref) / self.dist_sigma) ** 2))
            contact_score = 1.0 if str(cur.get("contact_state", "")) == str(ref.get("contact_state", "")) else 0.25
            support_score = 1.0 - min(1.0, abs(s_cur - s_ref))
            vals.append(
                (1.0 - self.contact_weight - self.support_weight) * 0.5 * (dist_score + gap_score)
                + self.contact_weight * contact_score
                + self.support_weight * support_score
            )

        score_pairs = float(sum(vals) / float(len(vals))) if vals else 0.0
        aid_cur = set(int(x) for x in pair_anchor_ids(pair_features))
        aid_ref = set(int(x) for x in view.anchor_ids)
        union = aid_cur | aid_ref
        anchor_j = float(len(aid_cur & aid_ref) / float(len(union))) if union else 0.0
        return float(0.75 * score_pairs + 0.25 * anchor_j)

    def match_view(self, anchor_geom_by_oid: dict[int, dict]) -> tuple[int | None, float]:
        if not self.enabled:
            return None, 0.0
        anchor_ids, pair_features = self.build_signature(anchor_geom_by_oid)
        if len(anchor_ids) < int(self.min_anchors) or not pair_features:
            return None, 0.0
        best_vid = None
        best_score = 0.0
        for vid, view in self.views.items():
            s = float(self.compare_signature(pair_features, view))
            if s > best_score:
                best_score = float(s)
                best_vid = int(vid)
        return best_vid, float(best_score)

    def match_or_create_view(self, anchor_geom_by_oid: dict[int, dict], create: bool = True) -> tuple[int | None, float]:
        if not self.enabled:
            return None, 0.0
        anchor_ids, pair_features = self.build_signature(anchor_geom_by_oid)
        if len(anchor_ids) < int(self.min_anchors) or not pair_features:
            return None, 0.0
        best_vid, best_score = self.match_view(anchor_geom_by_oid)
        if best_vid is not None and float(best_score) >= float(self.match_min_score):
            view = self.views.get(int(best_vid), None)
            if view is not None:
                view.support_count += 1
            return int(best_vid), float(best_score)
        if not create:
            return best_vid, float(best_score)

        vid = int(self.next_view_id)
        self.next_view_id += 1
        self.views[int(vid)] = AnchorViewSignature(view_id=int(vid), anchor_ids=anchor_ids, pair_features=pair_features)
        if len(self.views) > int(self.max_views):
            items = sorted(self.views.items(), key=lambda kv: int(getattr(kv[1], "support_count", 0)), reverse=True)
            self.views = dict(items[: self.max_views])
        return int(vid), 1.0
