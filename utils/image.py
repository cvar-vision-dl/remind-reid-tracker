# utils/image.py

import cv2
import numpy as np
import os
from collections import OrderedDict

class ImageHistory:
    """
    Hybrid cache:
      - RAM: LRU (last N frames)
      - DISK: PNG por frame (persistente)
    """

    def __init__(self, disk_dir: str, ram_max_items: int = 300):
        self.disk_dir = str(disk_dir)
        self.ram_max_items = max(1, int(ram_max_items))
        self.cache = OrderedDict()  # frame_id -> bgr

        os.makedirs(self.disk_dir, exist_ok=True)

    def path_for(self, frame_id: int) -> str:
        return os.path.join(self.disk_dir, f"frame_{int(frame_id):06d}.png")

    def has_on_disk(self, frame_id: int) -> bool:
        return os.path.isfile(self.path_for(frame_id))

    def get(self, frame_id: int):
        frame_id = int(frame_id)

        # RAM first
        if frame_id in self.cache:
            self.cache.move_to_end(frame_id)
            return self.cache[frame_id]

        # Disk fallback
        p = self.path_for(frame_id)
        if not os.path.isfile(p):
            return None

        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            return None

        self.put(frame_id, img, save_disk=False)
        return img

    def put(self, frame_id: int, bgr, save_disk: bool = True):
        frame_id = int(frame_id)

        self.cache[frame_id] = bgr
        self.cache.move_to_end(frame_id)

        while len(self.cache) > self.ram_max_items:
            self.cache.popitem(last=False)

        if save_disk:
            p = self.path_for(frame_id)
            cv2.imwrite(p, bgr)


# =============================================================================
# Resize (global)
# =============================================================================

def resize_keep_aspect_by_width(img: np.ndarray, config: dict):
    """
    Downscale keeping aspect ratio by fixing width.
    Never upscales.

    Uses:
      config["system"]["input_width_size"]
    """
    target_w = int(config.get("system", {}).get("input_width_size", img.shape[1]))

    h, w = img.shape[:2]
    if w <= target_w:
        return img, 1.0

    scale = target_w / float(w)
    new_w = int(round(target_w))
    new_h = int(round(h * scale))

    out = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return out, float(scale)


# =============================================================================
# Alignment to patch multiples (CROP ONLY, balanced per side)
# =============================================================================

def align_frame_to_patches_crop(
    frame: np.ndarray,
    patch_multiple: int,
    center: bool = True,
):
    """
    Crop the frame so H and W are multiples of patch_multiple.
    NO padding. Solo crop.

    Si center=True, reparte el recorte de forma equilibrada:
      - En el peor caso (mod = p-1), recorta como mucho:
          floor((p-1)/2) por un lado y ceil((p-1)/2) por el otro.
        Con p=16 => max 7 y 8 px.

    Returns:
      frame_aligned
      meta: dict with crop and sizes
        {
          "mode": "crop",
          "crop": (x0, y0, x1, y1),
          "orig_size": (h, w),
          "aligned_size": (h_eff, w_eff),
          "offset": (x0, y0),
        }
    """
    h, w = frame.shape[:2]
    p = int(patch_multiple)
    if p <= 0:
        raise ValueError("patch_multiple must be > 0")

    h_eff = (h // p) * p
    w_eff = (w // p) * p
    if h_eff <= 0 or w_eff <= 0:
        # too small: return unchanged to keep the pipeline running
        meta = {
            "mode": "crop",
            "crop": (0, 0, w, h),
            "orig_size": (h, w),
            "aligned_size": (h, w),
            "offset": (0, 0),
        }
        return frame, meta

    if h_eff == h and w_eff == w:
        meta = {
            "mode": "crop",
            "crop": (0, 0, w, h),
            "orig_size": (h, w),
            "aligned_size": (h, w),
            "offset": (0, 0),
        }
        return frame, meta

    if center:
        y0 = (h - h_eff) // 2
        x0 = (w - w_eff) // 2
    else:
        y0, x0 = 0, 0

    y1 = y0 + h_eff
    x1 = x0 + w_eff

    out = frame[y0:y1, x0:x1]

    meta = {
        "mode": "crop",
        "crop": (int(x0), int(y0), int(x1), int(y1)),
        "orig_size": (int(h), int(w)),
        "aligned_size": (int(h_eff), int(w_eff)),
        "offset": (int(x0), int(y0)),
    }
    return out, meta
