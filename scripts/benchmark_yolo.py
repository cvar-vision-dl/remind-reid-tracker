#!/usr/bin/env python3
"""Benchmark per-frame timing and YOLO VRAM impact across N random scenes.

Usage examples
~~~~~~~~~~~~~~
    # GT mode, 5 random scenes, 50 frames each:
    python benchmark_tar_timing.py --n-scenes 5 --max-frames 50

    # YOLO mode with VRAM profiling:
    python benchmark_tar_timing.py --n-scenes 3 --max-frames 100 \
        --yolo-model /path/to/yolo11n-seg.pt --yolo-device cuda:0

    # Specific scenes instead of random:
    python benchmark_tar_timing.py --scenes abc123 def456 --max-frames 80

All arguments fall back to the same REMIND_* env vars used by
``run_tracking_batch_tar.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Path setup (mirrors run_tracking_batch_tar.py)
# ---------------------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
# Adjust these if you place the script elsewhere:
SRC_DIR = CURRENT_DIR.parent / "src" if (CURRENT_DIR.parent / "src").is_dir() else CURRENT_DIR.parent
TESTING_DIR = SRC_DIR / "testing"
for p in (str(SRC_DIR), str(TESTING_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# -- project imports (lazy – fail fast with a clear message) ----------------
try:
    from run_tracking_batch_tar import (
        TarSceneBundle,
        TarFrameSource,
        TarDavisSegmenter,
        TarYoloSegmenter,
        _build_scene_bundle,
        _discover_tar_scene_ids,
        _normalize_mask_variant,
        _normalize_davis_variant,
        _env_str,
        _resolve,
        _resolve_optional,
        _resolve_int,
        _patched_tar_davis_segmenter,
        _patched_tar_yolo_segmenter,
        _read_exclude_scene_ids,
        _resolve_tar_scene_ids,
    )
except ImportError as exc:
    sys.exit(
        f"Could not import run_tracking_batch_tar – make sure this script "
        f"lives next to it or adjust SRC_DIR.\n  Error: {exc}"
    )

try:
    from config.config_loader import Config
    from pipeline.initialization import initialize_system
    from pipeline.reid_pipeline import ReIDPipeline
    from testing.davis_gt import DavisGroundTruthLoader
    from testing.run_tracking_test import (
        make_process_handle,
        read_process_rss_bytes,
        reset_cuda_peak_memory_stats,
        capture_cuda_memory_stats,
        resolve_aligned_shape,
        build_det_to_object_id,
    )
    import testing.run_tracking_batch as base_batch
except ImportError as exc:
    sys.exit(f"Missing project dependency: {exc}")


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def _gpu_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _gpu_mem_mb() -> dict[str, float]:
    """Return current VRAM usage in MiB (allocated / reserved / peak)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024 ** 2),
        }
    except Exception:
        return {}


def _reset_gpu_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Per-frame result container
# ---------------------------------------------------------------------------

@dataclass
class FrameTiming:
    scene_id: str
    frame_id: int
    frame_name: str
    read_ms: float = 0.0
    yolo_ms: float = 0.0          # isolated YOLO inference (0 in GT mode)
    pipeline_ms: float = 0.0      # full pipeline.process_frame
    gt_ms: float = 0.0
    eval_ms: float = 0.0
    loop_ms: float = 0.0
    # VRAM snapshots (MiB)
    vram_before_pipeline_mb: float = 0.0
    vram_after_pipeline_mb: float = 0.0
    vram_peak_pipeline_mb: float = 0.0

    @property
    def pipeline_minus_yolo_ms(self) -> float:
        return max(0.0, self.pipeline_ms - self.yolo_ms)

    def as_dict(self) -> dict[str, Any]:
        d = {
            "scene_id": self.scene_id,
            "frame_id": self.frame_id,
            "frame_name": self.frame_name,
            "read_ms": round(self.read_ms, 3),
            "yolo_ms": round(self.yolo_ms, 3),
            "pipeline_ms": round(self.pipeline_ms, 3),
            "pipeline_minus_yolo_ms": round(self.pipeline_minus_yolo_ms, 3),
            "gt_ms": round(self.gt_ms, 3),
            "eval_ms": round(self.eval_ms, 3),
            "loop_ms": round(self.loop_ms, 3),
            "vram_before_pipeline_mb": round(self.vram_before_pipeline_mb, 2),
            "vram_after_pipeline_mb": round(self.vram_after_pipeline_mb, 2),
            "vram_peak_pipeline_mb": round(self.vram_peak_pipeline_mb, 2),
        }
        return d


# ---------------------------------------------------------------------------
# Instrumented YOLO segmenter that records its own inference time
# ---------------------------------------------------------------------------

class InstrumentedYoloSegmenter(TarYoloSegmenter):
    """Wraps ``TarYoloSegmenter`` to record per-call inference time."""

    def load_model(self) -> None:
        super().load_model()
        self._last_inference_ms: float = 0.0

    def read_annotation_mask(self, frame_id: int) -> np.ndarray | None:
        t0 = perf_counter()
        mask = super().read_annotation_mask(frame_id)
        self._last_inference_ms = (perf_counter() - t0) * 1000.0
        return mask

    @property
    def last_inference_ms(self) -> float:
        return getattr(self, "_last_inference_ms", 0.0)


# ---------------------------------------------------------------------------
# Single-scene benchmark
# ---------------------------------------------------------------------------

def benchmark_scene(
    *,
    scene_bundle: TarSceneBundle,
    config_path: Path,
    stable_min_frames: int,
    max_frames: int | None,
    force_detector_backend: str,
    yolo_model_path: str | None,
    yolo_conf: float,
    yolo_iou: float,
    yolo_imgsz: int,
    yolo_device: str | None,
) -> tuple[list[FrameTiming], dict[str, Any]]:
    """Run one scene and return per-frame timings + scene-level VRAM stats."""

    use_yolo = bool(yolo_model_path)
    import detection.davis_segmenter as dsm
    import testing.davis_gt as dgt

    config = Config(default_config_path=config_path).to_dict()
    config.setdefault("detector", {})["backend"] = force_detector_backend
    davis_cfg = config.setdefault("davis", {})
    davis_cfg["sequence_name"] = scene_bundle.scene_id
    davis_cfg["variant"] = _normalize_davis_variant(scene_bundle.mask_variant)
    davis_cfg["tar_scene_bundle"] = scene_bundle
    davis_cfg["prefetch_annotations"] = False
    timing_cfg = config.setdefault("timing", {})
    timing_cfg["enabled"] = False
    timing_cfg["table"] = False
    timing_cfg["detail_keys"] = []

    if use_yolo:
        davis_cfg["yolo_model_path"] = str(yolo_model_path)
        davis_cfg["yolo_conf"] = yolo_conf
        davis_cfg["yolo_iou"] = yolo_iou
        davis_cfg["yolo_imgsz"] = yolo_imgsz
        davis_cfg["_yolo_frame_cache"] = {}
        if yolo_device:
            davis_cfg["yolo_device"] = yolo_device

    frame_names = list(scene_bundle.frame_names)
    if max_frames is not None:
        frame_names = frame_names[: max(0, int(max_frames))]

    frame_source = TarFrameSource(scene_bundle)
    timings: list[FrameTiming] = []

    # --- Patch segmenter classes ---
    orig_det = dsm.DavisSegmenter
    orig_gt = dgt.DavisSegmenter
    if use_yolo:
        dsm.DavisSegmenter = InstrumentedYoloSegmenter
        dgt.DavisSegmenter = TarDavisSegmenter
    else:
        dsm.DavisSegmenter = TarDavisSegmenter
        dgt.DavisSegmenter = TarDavisSegmenter

    vram_before_model: dict[str, float] = {}
    vram_after_model: dict[str, float] = {}

    try:
        _reset_gpu_peak()
        vram_before_model = _gpu_mem_mb()

        ctx = initialize_system(config)
        pipeline = ReIDPipeline(ctx)
        gt_loader = DavisGroundTruthLoader(config)
        evaluator = base_batch.TrackingEvaluator(
            stable_min_frames=stable_min_frames, config=config,
        )

        vram_after_model = _gpu_mem_mb()

        yolo_seg: InstrumentedYoloSegmenter | None = None
        if use_yolo:
            det = getattr(ctx, "detector", None) or getattr(ctx, "segmenter", None)
            if isinstance(det, InstrumentedYoloSegmenter):
                yolo_seg = det

        total = len(frame_names)
        for idx, frame_name in enumerate(frame_names):
            loop_t0 = perf_counter()
            frame_id = idx
            ft = FrameTiming(
                scene_id=scene_bundle.scene_id,
                frame_id=frame_id,
                frame_name=frame_name,
            )

            # ---- read ----
            t0 = perf_counter()
            frame = frame_source.read_bgr(frame_name)
            ft.read_ms = (perf_counter() - t0) * 1000.0
            if frame is None:
                continue

            # Feed frame to YOLO cache
            if yolo_seg is not None:
                yolo_seg.set_current_frame(frame_id, frame)
            elif use_yolo:
                davis_cfg.get("_yolo_frame_cache", {})[frame_id] = frame

            # ---- pipeline (includes YOLO inside) ----
            _reset_gpu_peak()
            ft.vram_before_pipeline_mb = _gpu_mem_mb().get("allocated_mb", 0.0)

            t0 = perf_counter()
            p_out, a_out, u_out = pipeline.process_frame(
                frame=frame, frame_id=frame_id, timestamp=float(frame_id),
            )
            ft.pipeline_ms = (perf_counter() - t0) * 1000.0

            gpu_snap = _gpu_mem_mb()
            ft.vram_after_pipeline_mb = gpu_snap.get("allocated_mb", 0.0)
            ft.vram_peak_pipeline_mb = gpu_snap.get("peak_allocated_mb", 0.0)

            # Grab isolated YOLO time from instrumented segmenter
            if yolo_seg is not None:
                ft.yolo_ms = yolo_seg.last_inference_ms

            # Evict frame
            if yolo_seg is not None:
                yolo_seg.evict_frame(frame_id)
            elif use_yolo:
                davis_cfg.get("_yolo_frame_cache", {}).pop(frame_id, None)

            # ---- GT ----
            t0 = perf_counter()
            aligned_shape = resolve_aligned_shape(p_out)
            gt_objects = gt_loader.load_frame(frame_id=frame_id, target_shape=aligned_shape)
            ft.gt_ms = (perf_counter() - t0) * 1000.0

            # ---- eval ----
            det_to_object_id = build_det_to_object_id(u_out)
            t0 = perf_counter()
            evaluator.ingest_frame(
                frame_id=frame_id,
                detections=p_out.detections,
                gt_objects=gt_objects,
                det_to_object_id=det_to_object_id,
                memory_store=ctx.memory,
                association_output=a_out,
                update_output=u_out,
                frame_shape=aligned_shape,
            )
            ft.eval_ms = (perf_counter() - t0) * 1000.0

            ft.loop_ms = (perf_counter() - loop_t0) * 1000.0
            timings.append(ft)

            if (idx + 1) == 1 or (idx + 1) % 20 == 0 or (idx + 1) == total:
                print(
                    f"  [{scene_bundle.scene_id}] {idx+1}/{total}  "
                    f"read={ft.read_ms:.1f}  yolo={ft.yolo_ms:.1f}  "
                    f"pipe={ft.pipeline_ms:.1f}  "
                    f"pipe-yolo={ft.pipeline_minus_yolo_ms:.1f}  "
                    f"gt={ft.gt_ms:.1f}  eval={ft.eval_ms:.1f}  "
                    f"loop={ft.loop_ms:.1f} ms"
                )

        # Scene-level VRAM summary
        vram_summary = {
            "vram_before_model_mb": round(vram_before_model.get("allocated_mb", 0.0), 2),
            "vram_after_model_mb": round(vram_after_model.get("allocated_mb", 0.0), 2),
            "vram_model_delta_mb": round(
                vram_after_model.get("allocated_mb", 0.0)
                - vram_before_model.get("allocated_mb", 0.0), 2
            ),
        }
        if timings:
            vram_summary["vram_peak_during_inference_mb"] = round(
                max(ft.vram_peak_pipeline_mb for ft in timings), 2
            )
            vram_summary["vram_avg_during_pipeline_mb"] = round(
                statistics.mean(ft.vram_after_pipeline_mb for ft in timings), 2
            )

        return timings, vram_summary

    finally:
        dsm.DavisSegmenter = orig_det
        dgt.DavisSegmenter = orig_gt
        scene_bundle.close()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "p95": 0}
    s = sorted(values)
    p95_idx = min(int(len(s) * 0.95), len(s) - 1)
    return {
        "mean": round(statistics.mean(s), 3),
        "median": round(statistics.median(s), 3),
        "std": round(statistics.stdev(s), 3) if len(s) > 1 else 0.0,
        "min": round(s[0], 3),
        "max": round(s[-1], 3),
        "p95": round(s[p95_idx], 3),
    }


def build_aggregate(all_timings: list[FrameTiming]) -> dict[str, Any]:
    """Compute aggregate statistics across all frames."""
    agg: dict[str, Any] = {"n_frames": len(all_timings)}
    for key in ("read_ms", "yolo_ms", "pipeline_ms", "pipeline_minus_yolo_ms",
                "gt_ms", "eval_ms", "loop_ms",
                "vram_before_pipeline_mb", "vram_after_pipeline_mb",
                "vram_peak_pipeline_mb"):
        vals = [
            getattr(ft, key) if hasattr(ft, key) else ft.as_dict().get(key, 0.0)
            for ft in all_timings
        ]
        agg[key] = _stats(vals)
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark per-frame timing (including isolated YOLO) and VRAM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-scenes", type=int, default=3,
                   help="Number of random scenes to benchmark (default: 3)")
    p.add_argument("--scenes", nargs="+", metavar="ID",
                   help="Explicit scene IDs (overrides --n-scenes)")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Max frames per scene (default: all)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for scene selection (default: 42)")
    p.add_argument("--dataset-root", metavar="DIR")
    p.add_argument("--data-tar-root", metavar="DIR")
    p.add_argument("--annotations-tar-root", metavar="DIR")
    p.add_argument("--config-path", metavar="FILE")
    p.add_argument("--image-subdir", metavar="PATH")
    p.add_argument("--mask-variant", metavar="NAME")
    p.add_argument("--stable-min-frames", type=int, default=3)
    p.add_argument("--detector-backend", default=None)
    p.add_argument("--exclude-scenes-file", metavar="FILE")
    # YOLO
    p.add_argument("--yolo-model", metavar="FILE",
                   help="Path to YOLO .pt model (enables YOLO mode)")
    p.add_argument("--yolo-conf", type=float, default=0.25)
    p.add_argument("--yolo-iou", type=float, default=0.7)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--yolo-device", default=None)
    # Output
    p.add_argument("--output-dir", metavar="DIR", default=None,
                   help="Where to write results (default: ./benchmark_results)")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    base_dir = Path(__file__).resolve().parent
    src_dir = base_dir.parent if (base_dir.parent / "config").is_dir() else base_dir
    project_dir = src_dir.parent.parent

    # ---- paths -----------------------------------------------------------
    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else src_dir / "config" / "default_config.yaml"
    )
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
    image_subdir = _resolve(args.image_subdir, "REMIND_IMAGE_SUBDIR", "dslr/resized_images")
    mask_variant = _resolve(args.mask_variant, "REMIND_MASK_VARIANT", "benchmark")
    normalized_mask_variant = _normalize_mask_variant(mask_variant)
    force_detector_backend = _resolve(
        args.detector_backend, "REMIND_BATCH_TAR_DETECTOR_BACKEND", "davis"
    )

    # ---- scene selection --------------------------------------------------
    exclude_file = _resolve_optional(args.exclude_scenes_file,
                                     "REMIND_BATCH_TAR_EXCLUDE_SCENES_FILE")
    exclude_scenes: set[str] = set()
    if exclude_file:
        exclude_scenes = _read_exclude_scene_ids(exclude_file)

    if args.scenes:
        scene_ids = list(args.scenes)
    else:
        all_ids = _resolve_tar_scene_ids(
            data_tar_root=data_tar_root,
            annotations_tar_root=annotations_tar_root,
            exclude_scenes=exclude_scenes or None,
        )
        if not all_ids:
            sys.exit("No scenes found.")
        rng = random.Random(args.seed)
        n = min(args.n_scenes, len(all_ids))
        scene_ids = rng.sample(all_ids, n)

    # ---- YOLO config ------------------------------------------------------
    yolo_model_path = _resolve_optional(args.yolo_model, "REMIND_YOLO_MODEL_PATH")
    yolo_conf = float(_resolve(args.yolo_conf, "REMIND_YOLO_CONF", "0.25"))
    yolo_iou = float(_resolve(args.yolo_iou, "REMIND_YOLO_IOU", "0.7"))
    yolo_imgsz = int(_resolve(args.yolo_imgsz, "REMIND_YOLO_IMGSZ", "640"))
    yolo_device = _resolve_optional(args.yolo_device, "REMIND_YOLO_DEVICE")
    mode_label = "YOLO" if yolo_model_path else "GT"

    # ---- output dir -------------------------------------------------------
    output_dir = Path(args.output_dir or "./benchmark_results").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- banner -----------------------------------------------------------
    print("=" * 72)
    print(f"  BENCHMARK  |  mode={mode_label}  |  scenes={len(scene_ids)}  "
          f"|  max_frames={args.max_frames or 'all'}")
    if yolo_model_path:
        print(f"  YOLO: {yolo_model_path}  conf={yolo_conf}  iou={yolo_iou}  "
              f"imgsz={yolo_imgsz}  device={yolo_device or 'auto'}")
    print(f"  GPU available: {_gpu_available()}")
    print(f"  Output: {output_dir}")
    print("=" * 72)

    # ---- run --------------------------------------------------------------
    all_timings: list[FrameTiming] = []
    scene_summaries: list[dict[str, Any]] = []

    for si, scene_id in enumerate(scene_ids):
        print(f"\n[{si+1}/{len(scene_ids)}] Scene: {scene_id}")
        try:
            bundle = _build_scene_bundle(
                scene_id=scene_id,
                data_tar_root=data_tar_root,
                annotations_tar_root=annotations_tar_root,
                mask_variant=normalized_mask_variant,
                image_subdir=image_subdir,
            )
        except Exception as exc:
            print(f"  SKIP (bundle error): {exc}")
            continue

        try:
            timings, vram_summary = benchmark_scene(
                scene_bundle=bundle,
                config_path=config_path,
                stable_min_frames=args.stable_min_frames,
                max_frames=args.max_frames,
                force_detector_backend=force_detector_backend,
                yolo_model_path=yolo_model_path,
                yolo_conf=yolo_conf,
                yolo_iou=yolo_iou,
                yolo_imgsz=yolo_imgsz,
                yolo_device=yolo_device,
            )
        except Exception as exc:
            print(f"  FAILED: {exc}")
            continue

        all_timings.extend(timings)

        # Per-scene summary
        scene_agg = build_aggregate(timings)
        scene_agg["scene_id"] = scene_id
        scene_agg["n_frames"] = len(timings)
        scene_agg["vram"] = vram_summary
        scene_summaries.append(scene_agg)

        print(f"  Done: {len(timings)} frames | "
              f"avg loop={scene_agg['loop_ms']['mean']:.1f} ms | "
              f"avg yolo={scene_agg['yolo_ms']['mean']:.1f} ms | "
              f"avg pipe-yolo={scene_agg['pipeline_minus_yolo_ms']['mean']:.1f} ms")
        if vram_summary:
            print(f"  VRAM: model_delta={vram_summary.get('vram_model_delta_mb', '?')} MiB  "
                  f"peak_inference={vram_summary.get('vram_peak_during_inference_mb', '?')} MiB")

    if not all_timings:
        sys.exit("No frames were processed.")

    # ---- global aggregate -------------------------------------------------
    global_agg = build_aggregate(all_timings)
    global_agg["mode"] = mode_label
    global_agg["n_scenes"] = len(scene_summaries)

    # ---- print summary table ----------------------------------------------
    print("\n" + "=" * 72)
    print("  AGGREGATE RESULTS")
    print("=" * 72)
    header = f"{'stage':<22} {'mean':>8} {'median':>8} {'p95':>8} {'std':>8} {'min':>8} {'max':>8}"
    print(header)
    print("-" * len(header))
    for key in ("read_ms", "yolo_ms", "pipeline_ms", "pipeline_minus_yolo_ms",
                "gt_ms", "eval_ms", "loop_ms"):
        s = global_agg[key]
        label = key.replace("_ms", "").replace("_", " ")
        print(f"  {label:<20} {s['mean']:>8.2f} {s['median']:>8.2f} "
              f"{s['p95']:>8.2f} {s['std']:>8.2f} {s['min']:>8.2f} {s['max']:>8.2f}")

    vram_keys = ("vram_before_pipeline_mb", "vram_after_pipeline_mb", "vram_peak_pipeline_mb")
    if any(global_agg.get(k, {}).get("max", 0) > 0 for k in vram_keys):
        print()
        for key in vram_keys:
            s = global_agg[key]
            label = key.replace("_mb", "").replace("vram_", "").replace("_", " ")
            print(f"  VRAM {label:<16} mean={s['mean']:.1f}  peak={s['max']:.1f} MiB")

    # ---- pie-chart-style breakdown ----------------------------------------
    loop_mean = global_agg["loop_ms"]["mean"]
    if loop_mean > 0:
        print(f"\n  Time breakdown (% of avg loop = {loop_mean:.1f} ms):")
        for key, label in [
            ("read_ms", "tar read"),
            ("yolo_ms", "YOLO inference"),
            ("pipeline_minus_yolo_ms", "pipeline (excl. YOLO)"),
            ("gt_ms", "GT loading"),
            ("eval_ms", "evaluation"),
        ]:
            pct = global_agg[key]["mean"] / loop_mean * 100
            bar = "█" * int(pct / 2)
            print(f"    {label:<24} {pct:5.1f}%  {bar}")

    # ---- write outputs ----------------------------------------------------
    # 1. Per-frame CSV
    csv_path = output_dir / f"benchmark_per_frame_{mode_label.lower()}.csv"
    rows = [ft.as_dict() for ft in all_timings]
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  Per-frame CSV -> {csv_path}")

    # 2. JSON summary
    json_path = output_dir / f"benchmark_summary_{mode_label.lower()}.json"
    summary_payload = {
        "mode": mode_label,
        "n_scenes": len(scene_summaries),
        "n_frames": len(all_timings),
        "aggregate": global_agg,
        "per_scene": scene_summaries,
    }
    with open(json_path, "w") as f:
        json.dump(summary_payload, f, indent=2, default=str)
    print(f"  Summary JSON -> {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()