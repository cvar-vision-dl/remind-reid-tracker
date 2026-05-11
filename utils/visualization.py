from __future__ import annotations

import cv2
import numpy as np


def overlay_header(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    """Add a black top band outside the frame with white text."""
    if frame_bgr is None:
        return None

    h, w = frame_bgr.shape[:2]
    pad = 6
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 1

    (_, th), baseline = cv2.getTextSize(text, font, scale, thickness)
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
    """Alpha-blend over img_bgr where mask==True (2D bool mask)."""
    if img_bgr is None:
        return None
    if mask is None:
        return img_bgr

    if mask.dtype != bool:
        mask = mask.astype(bool, copy=False)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D (H,W), got {mask.shape}")
    if not np.any(mask):
        return img_bgr

    a = float(alpha)
    a = max(0.0, min(1.0, a))

    out = img_bgr.copy()
    out_f = out.astype(np.float32)

    color = np.array(bgr, dtype=np.float32)
    out_f[mask] = (1.0 - a) * out_f[mask] + a * color

    return np.clip(out_f, 0, 255).astype(np.uint8)
