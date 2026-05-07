from __future__ import annotations

import colorsys
from typing import Dict, Iterable, List, Tuple

ColorBGR = Tuple[int, int, int]

_PALETTE_SIZE = 48
TEMPORARY_TRACK_COLOR: ColorBGR = (255, 255, 255)
_HSV_CANDIDATES = (
    (1.00, 1.00),
    (0.82, 1.00),
    (1.00, 0.90),
    (0.88, 0.94),
    (0.72, 0.92),
    (0.62, 0.86),
    (0.92, 0.78),
)


def _label_hash(label: str) -> int:
    value = 2166136261
    for byte in label.encode("utf-8", errors="ignore"):
        value ^= int(byte)
        value = (value * 16777619) & 0xFFFFFFFF
    return value


def _split_label_instance(label: str) -> Tuple[str, int | None]:
    text = str(label).strip().upper()
    if "_" not in text:
        return text, None

    head, tail = text.rsplit("_", 1)
    if tail.isdigit():
        return head, int(tail)
    return text, None


def _rgb_to_bgr(rgb: Tuple[float, float, float]) -> ColorBGR:
    r, g, b = rgb
    return int(round(b * 255.0)), int(round(g * 255.0)), int(round(r * 255.0))


def _dist2(c1: ColorBGR, c2: ColorBGR) -> int:
    db = int(c1[0]) - int(c2[0])
    dg = int(c1[1]) - int(c2[1])
    dr = int(c1[2]) - int(c2[2])
    return db * db + dg * dg + dr * dr


def is_temporary_label(label: str | None) -> bool:
    if label is None:
        return False
    s = str(label).strip().upper()
    return bool(s.startswith("T_"))


def _candidate_colors() -> List[ColorBGR]:
    out: List[ColorBGR] = []
    seen = set()
    for hue_deg in range(0, 360, 6):
        hue = float(hue_deg) / 360.0
        for sat, val in _HSV_CANDIDATES:
            color = _rgb_to_bgr(colorsys.hsv_to_rgb(hue, sat, val))
            if color in seen:
                continue
            seen.add(color)
            out.append(color)
    return out


def _build_palette() -> Tuple[ColorBGR, ...]:
    candidates = _candidate_colors()
    seeds = [
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (255, 128, 0),
    ]

    selected: List[ColorBGR] = []
    selected_set = set()
    for seed in seeds:
        if seed in candidates and seed not in selected_set:
            selected.append(seed)
            selected_set.add(seed)

    while len(selected) < _PALETTE_SIZE:
        best_color = None
        best_score = -1
        for color in candidates:
            if color in selected_set:
                continue
            score = min(_dist2(color, chosen) for chosen in selected)
            if score > best_score:
                best_score = score
                best_color = color
        if best_color is None:
            break
        selected.append(best_color)
        selected_set.add(best_color)

    return tuple(selected)


DISTINCT_BRIGHT_PALETTE: Tuple[ColorBGR, ...] = _build_palette()
_INSTANCE_COLOR_STRIDE = 17


def build_label_color_map(labels: Iterable[str]) -> Dict[str, ColorBGR]:
    unique_labels: List[str] = []
    seen = set()
    for label in labels:
        label_s = str(label)
        if label_s in seen:
            continue
        seen.add(label_s)
        unique_labels.append(label_s)

    color_map: Dict[str, ColorBGR] = {}
    for label in unique_labels:
        color_map[label] = label_color_bgr(label)
    return color_map


def label_color_bgr(label: str) -> ColorBGR:
    if is_temporary_label(label):
        return TEMPORARY_TRACK_COLOR

    palette_len = len(DISTINCT_BRIGHT_PALETTE)
    base_label, instance_idx = _split_label_instance(label)
    base_idx = _label_hash(base_label) % palette_len

    if instance_idx is None:
        color_idx = (_label_hash(str(label).strip().upper()) + base_idx) % palette_len
    else:
        color_idx = (base_idx + max(0, int(instance_idx) - 1) * _INSTANCE_COLOR_STRIDE) % palette_len

    return DISTINCT_BRIGHT_PALETTE[color_idx]
