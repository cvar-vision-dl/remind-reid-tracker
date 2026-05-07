# utils/visualization.py

from __future__ import annotations

import cv2
import numpy as np

from utils.color_palette import TEMPORARY_TRACK_COLOR, is_temporary_label, label_color_bgr


def bgr_from_int(idx: int) -> tuple:
    """Color determinista por ID (HSV -> BGR)."""
    idx = int(idx)
    h = (idx * 47) % 180
    s = 200
    v = 230
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def darken_bgr(bgr: tuple, factor: float) -> tuple:
    f = float(factor)
    f = max(0.0, min(1.0, f))
    return (int(bgr[0] * f), int(bgr[1] * f), int(bgr[2] * f))


def brighten_bgr(bgr: tuple, add: int = 40) -> tuple:
    a = int(add)
    return (
        int(min(255, bgr[0] + a)),
        int(min(255, bgr[1] + a)),
        int(min(255, bgr[2] + a)),
    )


def overlay_header(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    """Añade una banda superior negra fuera del frame con texto blanco."""
    if frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    box_h = th + baseline + 2 * pad
    out = np.zeros((h + box_h, w, frame_bgr.shape[2]), dtype=frame_bgr.dtype)
    out[box_h:, :, :] = frame_bgr
    cv2.rectangle(out, (0, 0), (w, box_h), (0, 0, 0), -1)
    cv2.putText(
        out,
        text,
        (pad, pad + th),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return out


def overlay_mask_bgr(img_bgr: np.ndarray, mask: np.ndarray, bgr: tuple, alpha: float) -> np.ndarray:
    """Alpha-blend sobre img_bgr donde mask==True (mask 2D bool)."""
    if img_bgr is None:
        return None
    if mask is None:
        return img_bgr

    if mask.dtype != bool:
        mask = mask.astype(bool, copy=False)
    if mask.ndim != 2:
        raise ValueError(f"mask debe ser 2D (H,W), pero es {mask.shape}")
    if not np.any(mask):
        return img_bgr

    a = float(alpha)
    a = max(0.0, min(1.0, a))

    out = img_bgr.copy()
    out_f = out.astype(np.float32)

    color = np.array(bgr, dtype=np.float32)
    out_f[mask] = (1.0 - a) * out_f[mask] + a * color

    return np.clip(out_f, 0, 255).astype(np.uint8)


def patchmask_to_pxmask(patch_mask: np.ndarray, patch_size: int, h: int, w: int) -> np.ndarray:
    ps = int(patch_size)
    pm = patch_mask.astype(bool, copy=False)
    up = np.repeat(np.repeat(pm, ps, axis=0), ps, axis=1)
    return up[:h, :w]


def draw_dino_patch_grid(
    frame_bgr: np.ndarray,
    patch_size: int,
    *,
    enabled: bool = True,
    step_patches: int = 1,
    color=(255, 255, 255),
    thickness: int = 1,
    alpha: float = 0.25,
) -> np.ndarray:
    if frame_bgr is None:
        return None
    if not enabled:
        return frame_bgr

    ps = int(patch_size)
    if ps <= 0:
        return frame_bgr

    step = int(max(1, step_patches))
    step_px = ps * step

    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()

    for x in range(0, w, step_px):
        cv2.line(overlay, (int(x), 0), (int(x), h - 1), color, int(max(1, thickness)), cv2.LINE_AA)
    for y in range(0, h, step_px):
        cv2.line(overlay, (0, int(y)), (w - 1, int(y)), color, int(max(1, thickness)), cv2.LINE_AA)

    a = float(alpha)
    a = max(0.0, min(1.0, a))
    if a <= 0.0:
        return frame_bgr
    if a >= 1.0:
        return overlay

    return cv2.addWeighted(overlay, a, frame_bgr, 1.0 - a, 0.0)


def resolve_bg_rings_from_det_feats(det_feats: dict | None):
    """Resuelve ring_inner/ring_outer (patch masks) desde det_feats['bg']."""
    if not isinstance(det_feats, dict):
        return None, None

    bg = det_feats.get("bg", None)
    if not isinstance(bg, dict):
        return None, None

    rings = bg.get("rings", None)
    if not isinstance(rings, dict):
        rings = bg

    ring_in = rings.get("ring_inner", None)
    ring_out = rings.get("ring_outer", None)
    return ring_in, ring_out


def draw_obj_and_bg_rings_from_det_features(
    frame_bgr: np.ndarray,
    detections: list,
    det_features_by_id: dict,
    patch_size: int,
    *,
    update_output=None,
    memory_store=None,
    color_mode: str = "davis",
    show_bg_rings: bool = True,
    alpha_inner: float = 0.35,
    alpha_outer: float = 0.25,
    outer_dark_factor: float = 0.65,
    show_obj_mask: bool = True,
    alpha_obj: float = 0.55,
    obj_brighten_add: int = 55,
    draw_obj_contour: bool = True,
    contour_thickness: int = 2,
    id_key: str = "detection_id",
) -> np.ndarray:
    if frame_bgr is None:
        return None

    out = frame_bgr.copy()
    h, w = out.shape[:2]
    ps = int(patch_size)

    det2obj = build_det_to_obj_map_from_update_output(update_output) if update_output is not None else {}

    for det in (detections or []):
        det_id = getattr(det, id_key, None)
        if det_id is None:
            continue
        det_id = int(det_id)
        is_temporary_visual = False

        cm = str(color_mode or "").strip().lower()
        if det_id in det2obj:
            item = det2obj[int(det_id)]
            obj_id, _, kind = item[:3]
            meta = item[3] if len(item) >= 4 and isinstance(item[3], dict) else {}
            obj_id = int(obj_id)
            if str(kind) == "ambiguous":
                amb = memory_store.get_ambiguous(int(obj_id)) if memory_store is not None else None
                base_lbl = amb.display_label(memory_store) if amb is not None else "TEMP"
                lbl = with_temp_id_label(base_lbl, int(obj_id), "amb")
            elif str(kind) == "provisional":
                prov = memory_store.get_provisional(int(obj_id)) if memory_store is not None else None
                base_lbl = prov.display_label(memory_store) if prov is not None else "TEMP"
                lbl = with_temp_id_label(base_lbl, int(obj_id), "prov")
            else:
                obj = memory_store.get(int(obj_id)) if memory_store is not None else None
                lbl = getattr(obj, "instance_label", None) if obj is not None else None

            is_temporary_visual = str(kind) in ("ambiguous", "provisional") or is_temporary_label(lbl)
            if is_temporary_visual:
                base_bgr = TEMPORARY_TRACK_COLOR
            elif cm in ("davis", "palette", "label"):
                base_bgr = label_color_bgr(str(lbl) if lbl else f"ID{obj_id}")
            elif cm in ("id", "object_id", "obj"):
                base_bgr = bgr_from_int(int(obj_id))
            else:
                base_bgr = bgr_from_int(det_id)
        else:
            # Fallback: no mapping to a persistent object yet.
            base_bgr = bgr_from_int(det_id)
            meta = {}

        if is_temporary_visual:
            ring_outer_bgr = (56, 56, 56)
            obj_bgr = TEMPORARY_TRACK_COLOR
        else:
            ring_outer_bgr = darken_bgr(base_bgr, outer_dark_factor)
            obj_bgr = brighten_bgr(base_bgr, add=obj_brighten_add)

        if show_bg_rings:
            feats = det_features_by_id.get(det_id, None)
            ring_in, ring_out = resolve_bg_rings_from_det_feats(feats)

            if ring_out is not None:
                outer_px = patchmask_to_pxmask(ring_out, ps, h, w)
                out = overlay_mask_bgr(out, outer_px, ring_outer_bgr, float(alpha_outer))

            if ring_in is not None:
                inner_px = patchmask_to_pxmask(ring_in, ps, h, w)
                out = overlay_mask_bgr(out, inner_px, base_bgr, float(alpha_inner))

        if show_obj_mask:
            m = getattr(det, "mask", None)
            if m is None:
                continue

            if m.dtype != bool:
                m = m.astype(bool, copy=False)

            if m.shape[:2] != (h, w):
                m = cv2.resize((m.astype(np.uint8) * 255), (w, h), interpolation=cv2.INTER_NEAREST) >= 128

            out = overlay_mask_bgr(out, m, obj_bgr, float(alpha_obj))

            if draw_obj_contour:
                cnts, _ = cv2.findContours(
                    (m.astype(np.uint8) * 255),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                if cnts:
                    if is_temporary_visual:
                        cv2.drawContours(
                            out,
                            cnts,
                            -1,
                            (0, 0, 0),
                            int(max(2, contour_thickness + 2)),
                            lineType=cv2.LINE_AA,
                        )
                    contour_color = (255, 255, 255) if str(meta.get("source", "")).startswith("distance_") else obj_bgr
                    cv2.drawContours(
                        out,
                        cnts,
                        -1,
                        contour_color,
                        int(max(1, contour_thickness)),
                        lineType=cv2.LINE_AA,
                    )

    return out


def centroid_from_mask(mask_bool: np.ndarray):
    ys, xs = np.nonzero(mask_bool)
    if ys.size == 0:
        return None
    return int(np.mean(xs)), int(np.mean(ys))


def anchor_point_for_det(det, frame_shape):
    h, w = frame_shape[:2]

    m = getattr(det, "mask", None)
    if m is not None:
        if m.dtype != bool:
            m = m.astype(bool, copy=False)
        if m.shape[:2] == (h, w):
            c = centroid_from_mask(m)
            if c is not None:
                return c

    bbox = getattr(det, "bbox", None)
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        x = int(max(0, min(w - 1, 0.5 * (float(x1) + float(x2)))))
        y = int(max(0, min(h - 1, float(y1))))
        return x, y

    return 10, 20


def build_det_to_obj_map_from_update_output(update_output):
    out = {}
    if update_output is None:
        return out

    matches = getattr(update_output, "matches", None)
    if isinstance(matches, list):
        for it in matches:
            if not isinstance(it, dict):
                continue
            if "det_id" not in it or "object_id" not in it:
                continue
            det_id = int(it["det_id"])
            obj_id = int(it["object_id"])
            score = float(it.get("score_final", 0.0))
            if det_id not in out or score > float(out[det_id][1]):
                out[det_id] = (
                    obj_id,
                    score,
                    "match",
                    {"source": str(it.get("source", "association") or "association")},
                )

    created = getattr(update_output, "created", None)
    if isinstance(created, list):
        for it in created:
            if not isinstance(it, dict):
                continue
            if "det_id" not in it or "object_id" not in it:
                continue
            det_id = int(it["det_id"])
            obj_id = int(it["object_id"])
            if det_id not in out:
                out[det_id] = (obj_id, 0.0, "create")

    ambiguous = getattr(update_output, "ambiguous", None)
    if isinstance(ambiguous, list):
        for it in ambiguous:
            if not isinstance(it, dict):
                continue
            if "det_id" not in it or "temp_id" not in it:
                continue
            det_id = int(it["det_id"])
            temp_id = int(it["temp_id"])
            score = float(it.get("best_score", 0.0))
            candidate_ids = [int(x) for x in (it.get("candidate_ids", []) or [])]
            if det_id not in out:
                out[det_id] = (
                    temp_id,
                    score,
                    "ambiguous",
                    {
                        "candidate_ids": list(candidate_ids),
                        "source": str(it.get("source", "association") or "association"),
                    },
                )

    provisional = getattr(update_output, "provisional", None)
    if isinstance(provisional, list):
        for it in provisional:
            if not isinstance(it, dict):
                continue
            if "det_id" not in it or "temp_id" not in it:
                continue
            det_id = int(it["det_id"])
            temp_id = int(it["temp_id"])
            score = float(it.get("best_score", 0.0))
            if det_id not in out:
                out[det_id] = (
                    temp_id,
                    score,
                    "provisional",
                    {"source": str(it.get("source", "association") or "association")},
                )

    return out


def build_temp_label_from_candidate_ids(class_name: str, candidate_ids: list[int], memory_store) -> str:
    parts: list[str] = []
    for oid in candidate_ids or []:
        obj = memory_store.get(int(oid)) if memory_store is not None else None
        lbl = str(getattr(obj, "instance_label", "") or "")
        if "_" in lbl:
            tail = lbl.rsplit("_", 1)[-1]
            if tail.isdigit():
                parts.append(str(int(tail)))
                continue
        parts.append(str(int(oid)))

    cls = str(class_name or "TEMP").upper()
    body = "|".join(parts) if parts else "?"
    return f"T_{cls}[{body}]"


def with_temp_id_label(base_label: str, temp_id: int, kind: str | None = None) -> str:
    lbl = str(base_label or "").strip()
    if not lbl:
        lbl = "TEMP"
    kind_s = str(kind or "tmp").strip().lower()
    return f"{lbl}[{kind_s}:{int(temp_id)}]"


def match_status_letter(det_id: int, kind: str, assoc_output) -> str:
    if str(kind) == "create":
        return ""

    if assoc_output is None:
        return ""

    reps = getattr(assoc_output, "reports_by_det_id", None)
    if not isinstance(reps, dict):
        return ""

    rep = reps.get(int(det_id), None)
    if rep is None:
        return ""

    diag = getattr(rep, "match_diag_sim", None)
    if not isinstance(diag, dict):
        return ""

    st = str(diag.get("status", "")).upper().strip()
    if st.startswith("STRONG"):
        return "S"
    if st.startswith("AMB"):
        return "A"
    if st.startswith("WEAK"):
        return "W"
    return ""


def decision_status_letter(det_id: int, kind: str, assoc_output) -> str:
    if str(kind) in ("ambiguous", "provisional"):
        return "T"
    if str(kind) == "create":
        return "N"

    if assoc_output is None:
        return ""

    reps = getattr(assoc_output, "reports_by_det_id", None)
    if not isinstance(reps, dict):
        return ""

    rep = reps.get(int(det_id), None)
    if rep is None:
        return ""

    dec = str(getattr(rep, "final_decision", "") or "").upper().strip()
    if dec == "MATCH":
        return "M"
    if dec == "NEW":
        return "N"
    if dec == "UNASSIGNED":
        return "U"
    if dec == "AMBIGUOUS_TRACK":
        return "T"
    return ""


def combined_status_tag(det_id: int, kind: str, assoc_output) -> str:
    q = match_status_letter(det_id=det_id, kind=kind, assoc_output=assoc_output)
    d = decision_status_letter(det_id=det_id, kind=kind, assoc_output=assoc_output)

    parts = []
    if q:
        parts.append(q)
    if d:
        parts.append(d)

    return "|".join(parts)


def draw_track_labels_from_update_output(
    frame_bgr: np.ndarray,
    detections: list,
    update_output,
    memory_store,
    *,
    assoc_output=None,
    show_score: bool = False,
    show_kind: bool = False,
    color=(255, 255, 255),
    color_mode: str = "davis",
    draw_bg_box: bool = True,
    avoid_masks: bool = True,
    allow_on_large_masks: bool = True,
    large_mask_area_frac: float = 0.08,
    draw_link_line: bool = True,
    link_min_dist_px: int = 28,
    link_thickness: int = 1,
    sticky: bool = True,
    sticky_max_dist_px: int = 220,
):
    """Dibuja labels por det usando update_output y tags (S/A/W)|(M/N/U)."""
    if frame_bgr is None:
        return None

    out = frame_bgr.copy()
    h, w = out.shape[:2]

    det2obj = build_det_to_obj_map_from_update_output(update_output)
    if not det2obj:
        return out

    # Evitar tapar máscaras:
    # - siempre intenta evitar tapar máscaras de otros objetos
    # - puede permitir tapar SU propia máscara si es suficientemente grande (para no alejar el label)
    union_integral = None
    mask_integral_by_det_id: dict[int, np.ndarray] = {}
    mask_area_by_det_id: dict[int, int] = {}
    if avoid_masks:
        try:
            union = np.zeros((h, w), dtype=np.uint8)
            for det in (detections or []):
                did = getattr(det, "detection_id", None)
                if did is None:
                    continue
                did = int(did)
                m = getattr(det, "mask", None)
                if m is None:
                    continue
                if m.dtype != bool:
                    m = m.astype(bool, copy=False)
                if m.shape[:2] != (h, w):
                    continue
                mu = m.astype(np.uint8, copy=False)
                union[mu.astype(bool, copy=False)] = 1
                mask_integral_by_det_id[did] = cv2.integral(mu)
                mask_area_by_det_id[did] = int(mu.sum())

            if int(union.sum()) > 0:
                union_integral = cv2.integral(union)  # (h+1, w+1)
        except Exception:
            union_integral = None
            mask_integral_by_det_id = {}
            mask_area_by_det_id = {}

    det_id_to_local = {}
    for i, det in enumerate(detections or []):
        did = getattr(det, "detection_id", None)
        if did is None:
            continue
        det_id_to_local[int(did)] = int(i)

    placed_boxes: list[tuple[int, int, int, int]] = []
    # Cache persistente (por proceso) para estabilizar posiciones de labels entre frames.
    # key: object_id -> (x0,y0,x1,y1)
    last_boxes_by_obj = getattr(draw_track_labels_from_update_output, "_last_boxes_by_obj", None)
    if not isinstance(last_boxes_by_obj, dict):
        last_boxes_by_obj = {}
        setattr(draw_track_labels_from_update_output, "_last_boxes_by_obj", last_boxes_by_obj)

    def rect_sum(ii: np.ndarray | None, rect: tuple[int, int, int, int]) -> int:
        if ii is None:
            return 0
        x0, y0, x1, y1 = rect
        x0 = int(max(0, min(int(w), x0)))
        x1 = int(max(0, min(int(w), x1)))
        y0 = int(max(0, min(int(h), y0)))
        y1 = int(max(0, min(int(h), y1)))
        if x1 <= x0 or y1 <= y0:
            return 0
        s = ii[int(y1), int(x1)] - ii[int(y0), int(x1)] - ii[int(y1), int(x0)] + ii[int(y0), int(x0)]
        return int(s)

    def overlap_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        ix0 = max(ax0, bx0)
        iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1)
        iy1 = min(ay1, by1)
        if ix1 <= ix0 or iy1 <= iy0:
            return 0
        return int((ix1 - ix0) * (iy1 - iy0))

    def pick_box_top_left(
        anchor_xy: tuple[int, int],
        box_wh: tuple[int, int],
        occupied: list[tuple[int, int, int, int]],
        *,
        det_id: int,
        obj_id: int,
    ) -> tuple[int, int]:
        ax, ay = int(anchor_xy[0]), int(anchor_xy[1])
        bw, bh = int(box_wh[0]), int(box_wh[1])
        bw = max(1, min(int(w), bw))
        bh = max(1, min(int(h), bh))

        margin = 6

        # Candidatos en anillos alrededor del anchor (más opciones => menos solapes).
        cand: list[tuple[int, int]] = []
        for s in (1, 2, 3, 4):
            dxs = [0, -bw // 2, bw // 2, -bw, bw]
            dys = [-bh, 0, bh]
            for dy in dys:
                for dx in dxs:
                    x0 = ax + s * dx
                    y0 = ay + s * dy
                    if dy < 0:
                        y0 -= margin
                    elif dy > 0:
                        y0 += margin
                    if dx < 0:
                        x0 -= margin
                    elif dx > 0:
                        x0 += margin
                    cand.append((int(x0), int(y0)))

        # Dedup conservando orden
        seen = set()
        dedup = []
        for x0, y0 in cand:
            k = (int(x0), int(y0))
            if k in seen:
                continue
            seen.add(k)
            dedup.append((int(x0), int(y0)))
        cand = dedup

        self_area = int(mask_area_by_det_id.get(int(det_id), 0))
        is_large = bool(
            allow_on_large_masks
            and float(large_mask_area_frac) > 0.0
            and float(self_area) >= float(large_mask_area_frac) * float(h * w)
        )
        self_ii = mask_integral_by_det_id.get(int(det_id), None)

        def overlaps_any_label(rect: tuple[int, int, int, int]) -> int:
            ov = 0
            for r in occupied:
                ov += overlap_area(rect, r)
            return int(ov)

        def rect_is_ok(rect: tuple[int, int, int, int], *, allow_self: bool) -> bool:
            if overlaps_any_label(rect) != 0:
                return False
            if union_integral is None:
                return True
            all_ov = int(rect_sum(union_integral, rect))
            self_ov = int(rect_sum(self_ii, rect)) if self_ii is not None else 0
            other_ov = int(max(0, all_ov - self_ov))
            if other_ov != 0:
                return False
            if (not allow_self) and self_ov != 0:
                return False
            return True

        # Sticky placement: si el rect anterior sigue siendo válido y no está demasiado lejos, mantenerlo.
        if sticky:
            prev = last_boxes_by_obj.get(int(obj_id), None)
            if isinstance(prev, (tuple, list)) and len(prev) == 4:
                px0, py0, px1, py1 = (int(prev[0]), int(prev[1]), int(prev[2]), int(prev[3]))
                # Ajustar dentro de frame y al tamaño actual de la caja
                px0 = int(max(0, min(int(w) - bw, px0)))
                py0 = int(max(0, min(int(h) - bh, py0)))
                prev_rect = (px0, py0, px0 + bw, py0 + bh)
                # Distancia desde el anchor al rect (punto más cercano)
                qx = int(max(prev_rect[0], min(prev_rect[2], ax)))
                qy = int(max(prev_rect[1], min(prev_rect[3], ay)))
                dx = qx - ax
                dy = qy - ay
                if (dx * dx + dy * dy) <= int(max(0, int(sticky_max_dist_px))) ** 2:
                    if rect_is_ok(prev_rect, allow_self=bool(is_large)):
                        return int(prev_rect[0]), int(prev_rect[1])

        best = None
        best_score = None  # tuple(other_mask_ov, self_mask_ov, dist2)

        # Pass 1: no label overlap, no other-mask overlap, no self-mask overlap.
        for x0, y0 in cand:
            x0 = int(max(0, min(int(w) - bw, int(x0))))
            y0 = int(max(0, min(int(h) - bh, int(y0))))
            rect = (x0, y0, x0 + bw, y0 + bh)

            if overlaps_any_label(rect) != 0:
                continue

            other_mask_ov = 0
            self_mask_ov = 0
            if union_integral is not None:
                all_ov = int(rect_sum(union_integral, rect))
                self_mask_ov = int(rect_sum(self_ii, rect)) if self_ii is not None else 0
                other_mask_ov = int(max(0, all_ov - self_mask_ov))
            if other_mask_ov != 0:
                continue
            if self_mask_ov != 0:
                continue

            dx = x0 - ax
            dy = y0 - ay
            dist2 = dx * dx + dy * dy
            score = (0, 0, int(dist2))
            if best_score is None or score < best_score:
                best_score = score
                best = (x0, y0)

        # Pass 2: permite tapar su propia máscara si es grande (pero nunca otras máscaras), sin solapar labels.
        if best is None and is_large:
            for x0, y0 in cand:
                x0 = int(max(0, min(int(w) - bw, int(x0))))
                y0 = int(max(0, min(int(h) - bh, int(y0))))
                rect = (x0, y0, x0 + bw, y0 + bh)

                if overlaps_any_label(rect) != 0:
                    continue

                other_mask_ov = 0
                self_mask_ov = 0
                if union_integral is not None:
                    all_ov = int(rect_sum(union_integral, rect))
                    self_mask_ov = int(rect_sum(self_ii, rect)) if self_ii is not None else 0
                    other_mask_ov = int(max(0, all_ov - self_mask_ov))
                if other_mask_ov != 0:
                    continue

                dx = x0 - ax
                dy = y0 - ay
                dist2 = dx * dx + dy * dy
                score = (int(other_mask_ov), int(self_mask_ov), int(dist2))
                if best_score is None or score < best_score:
                    best_score = score
                    best = (x0, y0)

        # Pass 3 (último recurso): permitir solapes mínimos, minimizando tapado.
        if best is None:
            for x0, y0 in cand:
                x0 = int(max(0, min(int(w) - bw, int(x0))))
                y0 = int(max(0, min(int(h) - bh, int(y0))))
                rect = (x0, y0, x0 + bw, y0 + bh)

                label_ov = overlaps_any_label(rect)
                other_mask_ov = 0
                self_mask_ov = 0
                if union_integral is not None:
                    all_ov = int(rect_sum(union_integral, rect))
                    self_mask_ov = int(rect_sum(self_ii, rect)) if self_ii is not None else 0
                    other_mask_ov = int(max(0, all_ov - self_mask_ov))

                dx = x0 - ax
                dy = y0 - ay
                dist2 = dx * dx + dy * dy
                score = (int(label_ov), int(other_mask_ov), int(self_mask_ov), int(dist2))
                if best_score is None or score < best_score:
                    best_score = score
                    best = (x0, y0)

        if best is None:
            return (
                int(max(0, min(int(w) - bw, ax))),
                int(max(0, min(int(h) - bh, ay))),
            )

        return best

    # Orden: primero los objetos pequeños (más importante no tapar), los grandes al final.
    det_list = []
    for det in (detections or []):
        did = getattr(det, "detection_id", None)
        if did is None:
            continue
        det_list.append((int(mask_area_by_det_id.get(int(did), 0)), det))
    det_list.sort(key=lambda t: int(t[0]))

    for _, det in det_list:
        det_id = getattr(det, "detection_id", None)
        if det_id is None:
            continue

        det_id = int(det_id)

        if det_id not in det2obj:
            continue

        item = det2obj[det_id]
        obj_id, score, kind = item[:3]
        meta = item[3] if len(item) >= 4 and isinstance(item[3], dict) else {}
        is_distance_match = bool(str(meta.get("source", "")).startswith("distance_"))

        if str(kind) == "ambiguous":
            amb = memory_store.get_ambiguous(int(obj_id)) if memory_store is not None else None
            base_lbl = amb.display_label(memory_store) if amb is not None else "TEMP"
            lbl = with_temp_id_label(base_lbl, int(obj_id), "amb")
        elif str(kind) == "provisional":
            prov = memory_store.get_provisional(int(obj_id)) if memory_store is not None else None
            base_lbl = prov.display_label(memory_store) if prov is not None else "TEMP"
            lbl = with_temp_id_label(base_lbl, int(obj_id), "prov")
        else:
            obj = memory_store.get(int(obj_id)) if memory_store is not None else None
            lbl = getattr(obj, "instance_label", None) if obj is not None else None
            if not lbl:
                lbl = f"ID{obj_id}"

        det_local_idx = det_id_to_local.get(det_id, None)
        det_local_s = str(det_local_idx) if det_local_idx is not None else "?"

        txt = f"{det_local_s}-{lbl}"

        tag = combined_status_tag(
            det_id=det_id,
            kind=str(kind),
            assoc_output=assoc_output,
        )

        if tag:
            txt = f"{txt} [{tag}]"
        if is_distance_match:
            txt = f"{txt} [D]"

        if show_kind:
            txt = f"{txt} [{kind}]"

        if show_score and str(kind) == "match":
            txt = f"{txt} ({float(score):.3f})"

        x, y = anchor_point_for_det(det, out.shape)
        x = int(max(0, min(w - 1, x)))
        y = int(max(0, min(h - 1, y)))

        cm = str(color_mode or "").strip().lower()

        if str(kind) in ("ambiguous", "provisional"):
            col = TEMPORARY_TRACK_COLOR
        elif cm in ("davis", "palette", "label"):
            col = label_color_bgr(str(lbl))
        elif cm in ("id", "object_id", "obj"):
            col = bgr_from_int(int(obj_id))
        else:
            col = tuple(color) if isinstance(color, (tuple, list)) and len(color) == 3 else (255, 255, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fs = 0.6
        th = 1

        (tw, thh), baseline = cv2.getTextSize(txt, font, fs, th)

        pad_x = 3
        pad_y = 3

        box_w = int(tw + 2 * pad_x)
        box_h = int(thh + baseline + 2 * pad_y)

        sticky_obj_id = int(obj_id) if str(kind) not in ("ambiguous", "provisional") else int(-1000000 - int(obj_id))
        x0, y0 = pick_box_top_left((x, y), (box_w, box_h), placed_boxes, det_id=det_id, obj_id=sticky_obj_id)

        placed_boxes.append((int(x0), int(y0), int(x0 + box_w), int(y0 + box_h)))
        last_boxes_by_obj[int(sticky_obj_id)] = placed_boxes[-1]

        if draw_bg_box:
            x1 = int(min(w - 1, x0 + box_w))
            y1 = int(min(h - 1, y0 + box_h))

            if draw_link_line:
                bx0, by0, bx1, by1 = placed_boxes[-1]
                ax, ay = int(x), int(y)
                px = int(max(bx0, min(bx1, ax)))
                py = int(max(by0, min(by1, ay)))
                dx = px - ax
                dy = py - ay
                d2 = dx * dx + dy * dy
                if d2 >= int(max(0, int(link_min_dist_px))) ** 2:
                    cv2.line(out, (ax, ay), (px, py), col, int(max(1, link_thickness)), cv2.LINE_AA)

            cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (0, 0, 0), -1)
            if is_distance_match:
                cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), (255, 255, 255), 3)
                cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), col, 1)
            else:
                cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), col, 2)

        tx = int(x0 + pad_x)
        ty = int(y0 + pad_y + thh)

        cv2.putText(out, txt, (tx, ty), font, fs, (255, 255, 255), 1, cv2.LINE_AA)

    return out
