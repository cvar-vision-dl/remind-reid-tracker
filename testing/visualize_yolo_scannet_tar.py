#!/usr/bin/env python3
"""Interactive YOLO segmentation visualizer for tar-packed ScanNet++ scenes.

Usage examples
~~~~~~~~~~~~~~
# Basic – opens a CV2 window, use trackbars + keyboard to explore:
python visualize_yolo_tar.py \\
    --scene-id f97de2c3e9 \\
    --yolo-model best_seg.pt \\
    --dataset-root ~/data/scannetpp_data

# With custom defaults:
python visualize_yolo_tar.py \\
    --scene-id fb152519ad \\
    --yolo-model runs/seg/weights/best.pt \\
    --yolo-conf 0.3 --yolo-imgsz 1024 \\
    --data-tar-root ~/data/scannetpp_data/data \\
    --annotations-tar-root ~/data/scannetpp_data/annotations

# Save overlay frames to a directory (headless, no window):
python visualize_yolo_tar.py \\
    --scene-id f97de2c3e9 \\
    --yolo-model best_seg.pt \\
    --save-dir ./viz_output --headless

Keyboard controls (in the interactive window)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  →  / d / .         Next frame
  ←  / a / ,         Previous frame
  Page Down           Jump +10 frames
  Page Up             Jump −10 frames
  Home                First frame
  End                 Last frame
  Space               Toggle play / pause (auto-advance)
  g                   Toggle GT mask overlay (if annotations available)
  m                   Cycle overlay mode: mask → contour → mask+contour → off
  s                   Save current overlay frame to disk
  h                   Toggle HUD text
  q / Esc             Quit
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup – import tar helpers from the batch script living next to us.
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
TESTING_DIR = CURRENT_DIR

for p in (str(SRC_DIR), str(TESTING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_tracking_batch_tar import (
    TarFrameSource,
    TarSceneBundle,
    _build_scene_bundle,
    _env_str,
    _normalize_mask_variant,
)

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

def _generate_palette(n: int, seed: int = 42) -> list[tuple[int, int, int]]:
    """Generate *n* visually distinct BGR colours."""
    rng = np.random.RandomState(seed)
    colours: list[tuple[int, int, int]] = []
    for i in range(max(n, 1)):
        hue = int((i * 137.508) % 180)  # golden-angle spacing in [0,180)
        sat = int(rng.randint(180, 256))
        val = int(rng.randint(180, 256))
        hsv = np.uint8([[[hue, sat, val]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        colours.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colours


_PALETTE = _generate_palette(256)


def _colour_for(idx: int) -> tuple[int, int, int]:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# YOLO runner (thin wrapper so we can re-run with different conf)
# ---------------------------------------------------------------------------

class YoloRunner:
    def __init__(
        self,
        model_path: str,
        imgsz: int = 640,
        iou: float = 0.7,
        device: str | None = None,
    ):
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "This script requires the `ultralytics` package.  "
                "Install with:  pip install ultralytics"
            ) from exc

        if not Path(model_path).is_file():
            raise FileNotFoundError(f"YOLO model not found: {model_path}")

        self.model = YOLO(model_path)
        self.names: dict[int, str] = dict(self.model.names or {})
        self.imgsz = imgsz
        self.iou = iou
        self.device = device

    def predict(
        self, frame: np.ndarray, conf: float = 0.25
    ) -> list[dict[str, Any]]:
        """Run YOLO segmentation and return a list of detection dicts.

        Each dict contains:
            cls_id, cls_name, conf, bbox_xyxy, mask  (binary HxW uint8)
        """
        kwargs: dict[str, Any] = {
            "task": "segment",
            "conf": float(conf),
            "iou": self.iou,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model(frame, **kwargs)
        if not results:
            return []

        result = results[0]
        detections: list[dict[str, Any]] = []
        h, w = frame.shape[:2]

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy().astype(float)
        xyxys = boxes.xyxy.cpu().numpy().astype(float)

        has_masks = result.masks is not None and result.masks.data is not None
        if has_masks:
            masks_tensor = result.masks.data.cpu().numpy()

        for i in range(len(cls_ids)):
            det: dict[str, Any] = {
                "cls_id": int(cls_ids[i]),
                "cls_name": self.names.get(int(cls_ids[i]), f"cls_{cls_ids[i]}"),
                "conf": float(confs[i]),
                "bbox_xyxy": xyxys[i].tolist(),
            }
            if has_masks:
                m = masks_tensor[i]
                if m.shape[0] != h or m.shape[1] != w:
                    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                det["mask"] = (m > 0.5).astype(np.uint8)
            else:
                det["mask"] = None
            detections.append(det)
        return detections


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_OVERLAY_MODES = ["mask+contour", "mask", "contour", "off"]


def draw_overlay(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    mode: str = "mask+contour",
    alpha: float = 0.45,
) -> np.ndarray:
    """Draw segmentation overlay on a *copy* of *frame*."""
    vis = frame.copy()
    if mode == "off" or not detections:
        return vis

    overlay = vis.copy()
    for i, det in enumerate(detections):
        colour = _colour_for(i)
        mask = det.get("mask")

        if mask is not None and mode in ("mask", "mask+contour"):
            overlay[mask > 0] = colour

        if mask is not None and mode in ("contour", "mask+contour"):
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, colour, 2)

    if mode in ("mask", "mask+contour"):
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

    # Labels
    for i, det in enumerate(detections):
        colour = _colour_for(i)
        x1, y1, x2, y2 = [int(v) for v in det["bbox_xyxy"]]
        label = f'{det["cls_name"]} {det["conf"]:.2f}'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(
            vis, label, (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return vis


def draw_gt_overlay(
    frame: np.ndarray,
    gt_mask: np.ndarray | None,
    alpha: float = 0.35,
) -> np.ndarray:
    """Draw GT instance mask overlay."""
    if gt_mask is None:
        return frame
    vis = frame.copy()
    overlay = vis.copy()
    instance_ids = np.unique(gt_mask)
    for iid in instance_ids:
        if iid == 0:
            continue
        colour = _colour_for(int(iid) + 128)  # offset from YOLO palette
        overlay[gt_mask == iid] = colour
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    return vis


def draw_hud(
    vis: np.ndarray,
    *,
    frame_idx: int,
    total_frames: int,
    frame_name: str,
    n_dets: int,
    conf: float,
    infer_ms: float,
    mode: str,
    show_gt: bool,
    playing: bool,
) -> np.ndarray:
    """Burn informational text into top-left / bottom-left."""
    lines_top = [
        f"Frame {frame_idx + 1}/{total_frames}  [{frame_name}]",
        f"Detections: {n_dets}   conf>={conf:.2f}   mode={mode}",
        f"Inference: {infer_ms:.1f} ms   GT={'ON' if show_gt else 'OFF'}   "
        f"{'PLAY' if playing else 'PAUSE'}",
    ]
    lines_bot = [
        "arrows=nav  space=play  g=GT  m=mode  s=save  h=HUD  q=quit",
    ]
    y = 22
    for line in lines_top:
        cv2.putText(
            vis, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            vis, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA,
        )
        y += 22

    h = vis.shape[0]
    y = h - 10
    for line in reversed(lines_bot):
        cv2.putText(
            vis, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA,
        )
        cv2.putText(
            vis, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )
        y -= 18

    return vis


# ---------------------------------------------------------------------------
# GT mask reader (reuses tar bundle)
# ---------------------------------------------------------------------------

def read_gt_mask(bundle: TarSceneBundle, frame_id: int) -> np.ndarray | None:
    """Read the GT annotation mask for *frame_id*, or None."""
    rel_path = bundle.annotation_member_rel(int(frame_id))
    try:
        payload = bundle.read_annotations_member_bytes(rel_path)
    except (FileNotFoundError, KeyError):
        return None
    arr = np.frombuffer(payload, dtype=np.uint8)
    if arr.size <= 0:
        return None
    mask = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Interactive YOLO segmentation visualizer for tar-packed "
            "ScanNet++ scenes.  Reads data/annotation tars, runs a YOLO "
            "model per-frame, and shows an OpenCV window with a confidence "
            "trackbar you can drag in real time."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Keyboard controls")[1] if "Keyboard controls" in (__doc__ or "") else "",
    )

    # ---- required ----------------------------------------------------------
    p.add_argument(
        "--scene-id", required=True,
        help="Scene ID to visualise (stem of the .tar file).",
    )
    p.add_argument(
        "--yolo-model", required=True, metavar="FILE",
        help="Path to an Ultralytics YOLO segmentation .pt model.",
    )

    # ---- data paths --------------------------------------------------------
    paths = p.add_argument_group("data paths")
    paths.add_argument(
        "--dataset-root", metavar="DIR",
        help="Top-level dataset directory.  [env: REMIND_SCANNETPP_TAR_ROOT]",
    )
    paths.add_argument(
        "--data-tar-root", metavar="DIR",
        help="Directory with per-scene data .tar files.  "
             "[env: REMIND_SCANNETPP_DATA_TAR_ROOT]",
    )
    paths.add_argument(
        "--annotations-tar-root", metavar="DIR",
        help="Directory with per-scene annotation .tar files.  "
             "[env: REMIND_SCANNETPP_ANNOTATIONS_TAR_ROOT]",
    )
    paths.add_argument(
        "--image-subdir", metavar="PATH",
        help="Image sub-directory inside data tars.  "
             "[env: REMIND_IMAGE_SUBDIR, default: dslr/resized_images]",
    )
    paths.add_argument(
        "--mask-variant", metavar="NAME",
        help="GT mask variant (for overlay comparison).  "
             "[env: REMIND_MASK_VARIANT, default: benchmark]",
    )

    # ---- YOLO knobs --------------------------------------------------------
    yolo = p.add_argument_group("YOLO parameters")
    yolo.add_argument(
        "--yolo-conf", type=float, default=0.25, metavar="F",
        help="Initial confidence threshold (adjustable live via trackbar).  "
             "[default: 0.25]",
    )
    yolo.add_argument(
        "--yolo-iou", type=float, default=0.7, metavar="F",
        help="NMS IoU threshold.  [default: 0.7]",
    )
    yolo.add_argument(
        "--yolo-imgsz", type=int, default=640, metavar="PX",
        help="Inference image size.  [default: 640]",
    )
    yolo.add_argument(
        "--yolo-device", metavar="DEV",
        help="Device for inference (e.g. cuda:0, cpu).  [default: auto]",
    )

    # ---- display -----------------------------------------------------------
    disp = p.add_argument_group("display")
    disp.add_argument(
        "--start-frame", type=int, default=0, metavar="N",
        help="Frame index to start at.  [default: 0]",
    )
    disp.add_argument(
        "--max-width", type=int, default=1280, metavar="PX",
        help="Max display width (frames are scaled to fit).  [default: 1280]",
    )
    disp.add_argument(
        "--no-gt", action="store_true",
        help="Disable GT mask overlay entirely (skip loading annotations).",
    )
    disp.add_argument(
        "--overlay-mode", choices=_OVERLAY_MODES, default="mask+contour",
        help="Initial overlay drawing mode.  [default: mask+contour]",
    )

    # ---- output ------------------------------------------------------------
    out = p.add_argument_group("output")
    out.add_argument(
        "--save-dir", metavar="DIR",
        help="Save every overlay frame to this directory.  "
             "Implies --headless unless --no-headless is also given.",
    )
    out.add_argument(
        "--headless", action="store_true",
        help="Run without an OpenCV window (useful with --save-dir).",
    )
    return p


# ---------------------------------------------------------------------------
# Resolve helpers (same pattern as the batch script)
# ---------------------------------------------------------------------------

def _resolve(cli_val: str | None, env_name: str, default: str) -> str:
    if cli_val is not None and str(cli_val).strip():
        return str(cli_val).strip()
    v = _env_str(env_name, "")
    return v if v else default


# ---------------------------------------------------------------------------
# Main viewer loop
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # ---- resolve paths ----------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir.parent
    project_dir = src_dir.parent.parent

    dataset_root = Path(
        _resolve(args.dataset_root, "REMIND_SCANNETPP_TAR_ROOT",
                 str(project_dir / "data" / "scannetpp_data"))
    ).expanduser().resolve()
    data_tar_root = Path(
        _resolve(args.data_tar_root, "REMIND_SCANNETPP_DATA_TAR_ROOT",
                 str(dataset_root / "data"))
    ).expanduser().resolve()
    annotations_tar_root = Path(
        _resolve(args.annotations_tar_root, "REMIND_SCANNETPP_ANNOTATIONS_TAR_ROOT",
                 str(dataset_root / "annotations"))
    ).expanduser().resolve()
    image_subdir = _resolve(args.image_subdir, "REMIND_IMAGE_SUBDIR",
                            "dslr/resized_images")
    mask_variant = _resolve(args.mask_variant, "REMIND_MASK_VARIANT", "benchmark")
    normalized_mask_variant = _normalize_mask_variant(mask_variant)

    # ---- build scene bundle -----------------------------------------------
    scene_id = str(args.scene_id).strip()
    print(f"[VIZ] Loading scene {scene_id} ...")
    bundle = _build_scene_bundle(
        scene_id=scene_id,
        data_tar_root=data_tar_root,
        annotations_tar_root=annotations_tar_root,
        mask_variant=normalized_mask_variant,
        image_subdir=image_subdir,
    )
    frame_source = TarFrameSource(bundle)
    frame_names = list(bundle.frame_names)
    total_frames = len(frame_names)
    print(f"[VIZ] Scene loaded: {total_frames} frames")

    # ---- load YOLO model --------------------------------------------------
    print(f"[VIZ] Loading YOLO model: {args.yolo_model}")
    runner = YoloRunner(
        model_path=args.yolo_model,
        imgsz=args.yolo_imgsz,
        iou=args.yolo_iou,
        device=args.yolo_device,
    )
    print(f"[VIZ] YOLO ready – {len(runner.names)} classes")

    # ---- optional save dir ------------------------------------------------
    save_dir: Path | None = None
    if args.save_dir:
        save_dir = Path(args.save_dir).expanduser().resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[VIZ] Saving overlays to {save_dir}")

    headless = args.headless or (save_dir is not None and not hasattr(args, "no_headless"))
    # If explicitly --save-dir without --headless, we still show the window
    # unless --headless was given.  The implicit headless only kicks in when
    # there's no display.
    if save_dir and not args.headless:
        headless = False

    # ---- state ------------------------------------------------------------
    frame_idx: int = max(0, min(args.start_frame, total_frames - 1))
    conf_pct: int = int(round(args.yolo_conf * 100))  # trackbar is 0-100
    overlay_mode_idx: int = _OVERLAY_MODES.index(args.overlay_mode)
    show_gt: bool = not args.no_gt
    show_hud: bool = True
    playing: bool = False
    play_delay_ms: int = 80  # ms between frames in play mode
    need_rerun: bool = True  # YOLO needs to be re-run on current frame
    max_display_w: int = args.max_width

    # Detection cache: frame_idx -> (conf_used, detections)
    det_cache: dict[int, tuple[float, list[dict[str, Any]]]] = {}
    gt_cache: dict[int, np.ndarray | None] = {}
    frame_cache: dict[int, np.ndarray] = {}

    WINDOW = "YOLO Segmentation Visualizer"

    # ---- trackbar callback ------------------------------------------------
    def _on_conf_change(val: int) -> None:
        nonlocal conf_pct, need_rerun
        conf_pct = val
        need_rerun = True

    # ---- open window ------------------------------------------------------
    if not headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        cv2.createTrackbar("conf %", WINDOW, conf_pct, 100, _on_conf_change)
        cv2.resizeWindow(WINDOW, max_display_w, 720)

    def _current_conf() -> float:
        return max(0.01, conf_pct / 100.0)

    def _get_frame(idx: int) -> np.ndarray:
        if idx not in frame_cache:
            frame = frame_source.read_bgr(frame_names[idx])
            if frame is None:
                raise RuntimeError(f"Failed to read frame {frame_names[idx]}")
            frame_cache[idx] = frame
            # Keep cache bounded (keep ±20 around current position).
            for old_key in list(frame_cache.keys()):
                if abs(old_key - idx) > 30:
                    del frame_cache[old_key]
        return frame_cache[idx]

    def _get_detections(idx: int, conf: float) -> list[dict[str, Any]]:
        cached = det_cache.get(idx)
        if cached is not None and abs(cached[0] - conf) < 1e-6:
            return cached[1]
        frame = _get_frame(idx)
        dets = runner.predict(frame, conf=conf)
        det_cache[idx] = (conf, dets)
        # Trim cache
        for old_key in list(det_cache.keys()):
            if abs(old_key - idx) > 30:
                del det_cache[old_key]
        return dets

    def _get_gt(idx: int) -> np.ndarray | None:
        if idx not in gt_cache:
            gt_cache[idx] = read_gt_mask(bundle, idx)
            for old_key in list(gt_cache.keys()):
                if abs(old_key - idx) > 30:
                    del gt_cache[old_key]
        return gt_cache[idx]

    def _render(idx: int, conf: float) -> tuple[np.ndarray, list[dict[str, Any]], float]:
        frame = _get_frame(idx)
        t0 = perf_counter()
        dets = _get_detections(idx, conf)
        infer_ms = (perf_counter() - t0) * 1000.0

        mode = _OVERLAY_MODES[overlay_mode_idx]
        vis = draw_overlay(frame, dets, mode=mode)

        if show_gt:
            gt = _get_gt(idx)
            if gt is not None:
                vis = draw_gt_overlay(vis, gt, alpha=0.25)

        if show_hud:
            vis = draw_hud(
                vis,
                frame_idx=idx,
                total_frames=total_frames,
                frame_name=frame_names[idx],
                n_dets=len(dets),
                conf=conf,
                infer_ms=infer_ms,
                mode=mode,
                show_gt=show_gt,
                playing=playing,
            )
        return vis, dets, infer_ms

    def _display(vis: np.ndarray) -> None:
        h, w = vis.shape[:2]
        if w > max_display_w:
            scale = max_display_w / w
            vis = cv2.resize(vis, (max_display_w, int(h * scale)))
        cv2.imshow(WINDOW, vis)

    def _save_frame(vis: np.ndarray, idx: int) -> None:
        if save_dir is None:
            return
        out_path = save_dir / f"frame_{idx:06d}.jpg"
        cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # ---- headless batch mode ----------------------------------------------
    if headless:
        print(f"[VIZ] Headless mode – processing {total_frames} frames ...")
        for idx in range(total_frames):
            conf = _current_conf()
            vis, dets, infer_ms = _render(idx, conf)
            if save_dir:
                _save_frame(vis, idx)
            if (idx + 1) % 20 == 0 or idx == 0 or idx == total_frames - 1:
                print(
                    f"[VIZ] {idx + 1}/{total_frames}  dets={len(dets)}  "
                    f"infer={infer_ms:.1f}ms"
                )
        print(f"[VIZ] Done.  Saved to {save_dir}")
        bundle.close()
        return

    # ---- interactive loop -------------------------------------------------
    print("[VIZ] Interactive mode – drag the 'conf %' trackbar to change threshold.")
    print("[VIZ] Press 'h' for keyboard shortcuts.")

    prev_conf: float = -1.0
    prev_idx: int = -1

    while True:
        conf = _current_conf()

        # Only re-render when something changed.
        if frame_idx != prev_idx or abs(conf - prev_conf) > 1e-6 or need_rerun:
            vis, dets, infer_ms = _render(frame_idx, conf)
            _display(vis)
            if save_dir:
                _save_frame(vis, frame_idx)
            prev_idx = frame_idx
            prev_conf = conf
            need_rerun = False

        wait_ms = play_delay_ms if playing else 30
        key = cv2.waitKey(wait_ms) & 0xFFFF

        if playing:
            frame_idx = (frame_idx + 1) % total_frames
            need_rerun = True

        # ---- handle key presses -------------------------------------------
        if key == 0xFFFF or key == -1:
            # Check if trackbar value changed externally.
            try:
                tb_val = cv2.getTrackbarPos("conf %", WINDOW)
                if tb_val != conf_pct:
                    conf_pct = tb_val
                    need_rerun = True
            except cv2.error:
                pass
            continue

        # Quit
        if key in (ord("q"), ord("Q"), 27):  # q / Esc
            break

        # Navigation
        elif key in (ord("d"), ord("."), 0xFF53, 83):  # → / d / .
            frame_idx = min(frame_idx + 1, total_frames - 1)
            need_rerun = True
        elif key in (ord("a"), ord(","), 0xFF51, 81):  # ← / a / ,
            frame_idx = max(frame_idx - 1, 0)
            need_rerun = True
        elif key in (0xFF56, 86):  # Page Down
            frame_idx = min(frame_idx + 10, total_frames - 1)
            need_rerun = True
        elif key in (0xFF55, 85):  # Page Up
            frame_idx = max(frame_idx - 10, 0)
            need_rerun = True
        elif key in (0xFF50, 0xFF95):  # Home
            frame_idx = 0
            need_rerun = True
        elif key in (0xFF57, 0xFF9C):  # End
            frame_idx = total_frames - 1
            need_rerun = True

        # Play / pause
        elif key == 32:  # Space
            playing = not playing
            need_rerun = True

        # GT toggle
        elif key in (ord("g"), ord("G")):
            if args.no_gt:
                print("[VIZ] GT overlay disabled via --no-gt")
            else:
                show_gt = not show_gt
                need_rerun = True

        # Overlay mode cycle
        elif key in (ord("m"), ord("M")):
            overlay_mode_idx = (overlay_mode_idx + 1) % len(_OVERLAY_MODES)
            need_rerun = True
            print(f"[VIZ] Overlay mode: {_OVERLAY_MODES[overlay_mode_idx]}")

        # Save single frame
        elif key in (ord("s"), ord("S")):
            out_dir = save_dir or Path(".")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{scene_id}_frame_{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"[VIZ] Saved → {out_path}")

        # HUD toggle
        elif key in (ord("h"), ord("H")):
            show_hud = not show_hud
            need_rerun = True

    cv2.destroyAllWindows()
    bundle.close()
    print("[VIZ] Bye.")


if __name__ == "__main__":
    main()