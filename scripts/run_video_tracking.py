from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config_loader import Config
from utils.io import list_image_files, parse_frame_id, read_bgr
from utils.visualization import overlay_header, overlay_mask_bgr


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass
class FrameItem:
    frame: np.ndarray
    frame_idx: int
    frame_id: int
    timestamp: float
    name: str


class FrameSource:
    def __init__(
        self,
        source: Path,
        *,
        start_frame: int = 0,
        stride: int = 1,
        input_video_fps: float | None = None,
        frames_timestamp_fps: float = 30.0,
    ):
        self.source = source
        self.start_frame = max(0, int(start_frame))
        self.stride = max(1, int(stride))
        self.frames_timestamp_fps = float(frames_timestamp_fps) if float(frames_timestamp_fps) > 0 else 30.0
        self.requested_input_video_fps = None if input_video_fps is None else float(input_video_fps)
        self.kind = self._resolve_kind(source)
        self.native_fps = self.frames_timestamp_fps
        self.input_sample_fps = self.frames_timestamp_fps
        self.processed_fps = max(0.1, self.input_sample_fps / float(self.stride))
        self.total_frames: int | None = None
        self._image_files: list[str] = []

        if self.kind == "frames":
            self._image_files = list_image_files(str(source))
            self.total_frames = len(self._image_files)
            self.native_fps = self.frames_timestamp_fps
            self.input_sample_fps = self.frames_timestamp_fps
        elif self.kind == "single_image":
            self.total_frames = 1
            self.native_fps = self.frames_timestamp_fps
            self.input_sample_fps = self.frames_timestamp_fps
        else:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video: {source}")
            raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            raw_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.native_fps = raw_fps if raw_fps > 0 else self.frames_timestamp_fps
            self.total_frames = raw_total if raw_total > 0 else None
            if self.requested_input_video_fps is not None and self.requested_input_video_fps > 0:
                self.input_sample_fps = min(float(self.requested_input_video_fps), float(self.native_fps))
            else:
                self.input_sample_fps = float(self.native_fps)
            cap.release()
        self.processed_fps = max(0.1, float(self.input_sample_fps) / float(self.stride))

    @staticmethod
    def _resolve_kind(source: Path) -> str:
        if source.is_dir():
            return "frames"
        suffix = source.suffix.lower()
        if suffix in VIDEO_EXTS:
            return "video"
        if suffix in IMAGE_EXTS:
            return "single_image"
        raise ValueError(f"Unsupported source. Use a video, image, or image directory: {source}")

    def iter_frames(self, max_frames: int | None = None) -> Iterable[FrameItem]:
        yielded = 0
        if self.kind in {"frames", "single_image"}:
            files = self._image_files if self.kind == "frames" else [str(self.source)]
            for view_idx, frame_path in enumerate(files):
                if view_idx < self.start_frame:
                    continue
                if (view_idx - self.start_frame) % self.stride != 0:
                    continue
                if max_frames is not None and yielded >= int(max_frames):
                    break
                frame = read_bgr(frame_path)
                if frame is None:
                    print(f"[WARN] Could not read image: {frame_path}")
                    continue
                parsed_id = parse_frame_id(frame_path)
                frame_id = int(view_idx if parsed_id is None else parsed_id)
                yielded += 1
                yield FrameItem(
                    frame=frame,
                    frame_idx=int(view_idx),
                    frame_id=frame_id,
                    timestamp=float(view_idx) / float(self.frames_timestamp_fps),
                    name=Path(frame_path).name,
                )
            return

        cap = cv2.VideoCapture(str(self.source))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self.source}")
        try:
            if self.start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(self.start_frame))
            raw_idx = self.start_frame
            selected_idx = 0
            sample_step = max(1.0, float(self.native_fps) / max(0.001, float(self.input_sample_fps)))
            next_sample_raw_idx = int(round(float(self.start_frame)))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if raw_idx >= next_sample_raw_idx:
                    keep_by_stride = (selected_idx % self.stride) == 0
                    selected_idx += 1
                    next_sample_raw_idx = int(round(float(self.start_frame) + float(selected_idx) * sample_step))
                    if next_sample_raw_idx <= raw_idx:
                        next_sample_raw_idx = raw_idx + 1
                else:
                    keep_by_stride = False

                if keep_by_stride:
                    if max_frames is not None and yielded >= int(max_frames):
                        break
                    yielded += 1
                    yield FrameItem(
                        frame=frame,
                        frame_idx=int(raw_idx),
                        frame_id=int(raw_idx),
                        timestamp=float(raw_idx) / float(self.native_fps),
                        name=f"frame_{raw_idx:06d}",
                    )
                raw_idx += 1
        finally:
            cap.release()


def _parse_classes(raw: str | None) -> list[int | str] | None:
    if raw is None or not str(raw).strip():
        return None
    out: list[int | str] = []
    for token in str(raw).replace(";", ",").split(","):
        item = token.strip()
        if not item:
            continue
        try:
            out.append(int(item))
        except ValueError:
            out.append(item)
    return out or None


def resolve_scene_source(scene: str, *, test_root: Path, input_kind: str = "auto") -> Path:
    scene_name = str(scene or "").strip()
    if not scene_name:
        raise ValueError("A scene name is required when --source is not provided.")

    root = test_root.expanduser().resolve()
    kinds = [str(input_kind).strip().lower()]
    if kinds[0] == "auto":
        kinds = ["frames", "video"]

    candidates: list[Path] = []
    if "frames" in kinds:
        frames_dir = root / "frames" / scene_name
        if frames_dir.is_dir() and list_image_files(str(frames_dir)):
            candidates.append(frames_dir)

    if "video" in kinds:
        video_dir = root / "videos" / scene_name
        if video_dir.is_dir():
            videos = sorted(
                p for p in video_dir.iterdir()
                if p.is_file() and p.suffix.lower() in VIDEO_EXTS
            )
            candidates.extend(videos)
        direct_videos = sorted(
            p for p in (root / "videos").glob(f"{scene_name}.*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS
        )
        candidates.extend(direct_videos)

    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(
        "Scene not found. Expected one of:\n"
        f"  {root / 'frames' / scene_name}/<images>\n"
        f"  {root / 'videos' / scene_name}/<video-file>\n"
        f"  {root / 'videos'}/{scene_name}.mp4"
    )


def _default_output_dir(source: Path, scene: str | None = None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = str(scene or "").strip() or source.stem
    return (REPO_ROOT / "outputs" / "video_runs" / f"{label}_{stamp}").resolve()


def _configure(args: argparse.Namespace, output_dir: Path) -> dict:
    cfg = Config(args.config, args.override_config).to_dict()

    cfg.setdefault("paths", {})["output_dir"] = str(output_dir)
    cfg.setdefault("detector", {})["backend"] = "yolo"

    yolo_cfg = cfg.setdefault("yolo", {})
    yolo_cfg["model_label"] = "CUSTOM"
    yolo_cfg["models"] = {"CUSTOM": str(args.yolo_model)}
    yolo_cfg["conf_th"] = float(args.yolo_conf)
    yolo_cfg["iou_th"] = float(args.yolo_iou)
    yolo_cfg["max_det"] = int(args.max_det)
    yolo_cfg["classes"] = _parse_classes(args.classes)
    yolo_cfg["mask_erosion_px"] = int(args.mask_erosion_px)
    yolo_cfg["mask_erosion_iters"] = int(args.mask_erosion_iters)

    cfg.setdefault("system", {})["input_width_size"] = int(args.yolo_imgsz)
    cfg.setdefault("runtime", {})["device"] = str(args.device)
    cfg.setdefault("timing", {})["enabled"] = bool(args.verbose_timing)
    cfg.setdefault("timing", {})["table"] = False

    if args.dino_model_label:
        cfg.setdefault("dino", {})["model_label"] = str(args.dino_model_label)

    return cfg


def resolve_yolo_model(model_arg: str, *, models_dir: Path) -> str:
    raw = str(model_arg or "").strip()
    if not raw:
        raise ValueError("YOLO model name cannot be empty.")
    if Path(raw).is_absolute() or "/" in raw or "\\" in raw:
        raise ValueError(
            "Pass only the YOLO model file name, not a path. "
            f"Expected it inside: {models_dir.expanduser().resolve()}"
        )

    model_path = models_dir.expanduser().resolve() / raw
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLO model not found: {model_path}\n"
            f"Put the model file in {models_dir.expanduser().resolve()} and pass its file name."
        )
    return str(model_path)


def _color_for_id(identity: int) -> tuple[int, int, int]:
    value = int(identity) * 37 % 180
    hsv = np.uint8([[[value, 210, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _draw_text_box(img: np.ndarray, text: str, org: tuple[int, int], color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    x, y = int(org[0]), int(org[1])
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    y = max(th + 8, y)
    cv2.rectangle(img, (x, y - th - baseline - 6), (x + tw + 8, y + baseline + 2), color, -1)
    cv2.putText(img, text, (x + 4, y - 4), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def _entries_by_det_id(update_output) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for item in getattr(update_output, "matches", []) or []:
        out[int(item["det_id"])] = {
            "kind": "match",
            "object_id": int(item["object_id"]),
            "label": f"ID {int(item['object_id'])}",
        }
    for item in getattr(update_output, "created", []) or []:
        out[int(item["det_id"])] = {
            "kind": "new",
            "object_id": int(item["object_id"]),
            "label": f"ID {int(item['object_id'])}",
        }
    for item in getattr(update_output, "ambiguous", []) or []:
        label = str(item.get("temp_label", f"T_ID{int(item.get('temp_id', -1))}"))
        out[int(item["det_id"])] = {
            "kind": "ambiguous",
            "object_id": None,
            "label": label,
        }
    for item in getattr(update_output, "provisional", []) or []:
        label = str(item.get("temp_label", f"T_ID{int(item.get('temp_id', -1))}"))
        out[int(item["det_id"])] = {
            "kind": "provisional",
            "object_id": None,
            "label": label,
        }
    return out


def _detection_json(det, entry: dict[str, Any]) -> dict[str, Any]:
    bbox = getattr(det, "bbox", None)
    return {
        "det_id": int(getattr(det, "detection_id", -1)),
        "object_id": entry.get("object_id", None),
        "label": str(entry.get("label", "")),
        "kind": str(entry.get("kind", "detection")),
        "class_id": int(getattr(det, "class_id", -1)),
        "class_name": getattr(det, "class_name", None),
        "confidence": float(getattr(det, "confidence", 0.0) or 0.0),
        "bbox_xyxy": [float(x) for x in bbox] if bbox is not None else None,
    }


def render_frame(frame: np.ndarray, detections: list, update_output, header: str, alpha: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    out = frame.copy()
    entries = _entries_by_det_id(update_output)
    details: list[dict[str, Any]] = []

    for det in detections or []:
        det_id = int(getattr(det, "detection_id", -1))
        entry = entries.get(det_id, {"kind": "detection", "object_id": None, "label": f"DET {det_id}"})
        kind = str(entry.get("kind", "detection"))
        object_id = entry.get("object_id", None)
        if kind in {"ambiguous", "provisional"}:
            color = (255, 255, 255)
        elif object_id is not None:
            color = _color_for_id(int(object_id))
        else:
            color = (180, 180, 180)

        mask = getattr(det, "mask", None)
        if mask is not None:
            out = overlay_mask_bgr(out, mask, color, alpha)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, color, 2)

        bbox = getattr(det, "bbox", None)
        if bbox is not None:
            x1, y1 = [int(round(float(x))) for x in bbox[:2]]
            class_name = getattr(det, "class_name", None) or f"class_{int(getattr(det, 'class_id', -1))}"
            conf = float(getattr(det, "confidence", 0.0) or 0.0)
            text = f"{entry['label']} {class_name} {conf:.2f}"
            _draw_text_box(out, text, (x1, max(0, y1 - 4)), color)

        details.append(_detection_json(det, entry))

    return overlay_header(out, header), details


def _open_writer(path: Path, frame_shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (int(w), int(h)))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {path}")
    return writer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run REMIND tracking on a test scene using YOLO segmentation. "
            "By default scenes are resolved from testData/videos/<scene>/ or testData/frames/<scene>/."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scene", help="Scene name under testData/videos/ or testData/frames/.")
    parser.add_argument("yolo_model", help="YOLO segmentation model file name located inside the yolo/ folder.")
    parser.add_argument("--source", type=Path, help="Direct input video, image, or frame directory. Overrides scene lookup but still uses the scene name for outputs.")
    parser.add_argument("--test-root", type=Path, default=REPO_ROOT / "testData", help="Root containing videos/ and frames/ scene folders.")
    parser.add_argument(
        "--input-kind",
        choices=["auto", "video", "frames"],
        default="auto",
        help="Scene lookup mode when --source is not set. Use video or frames to force one input type.",
    )
    parser.add_argument("--yolo-dir", type=Path, default=REPO_ROOT / "yolo", help="Directory containing YOLO model files.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config" / "default_config.yaml", help="Base REMIND config.")
    parser.add_argument("--override-config", type=Path, default=None, help="Optional YAML config override.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for rendered video and reports.")
    parser.add_argument("--output-video", type=Path, default=None, help="Output .mp4 path. Defaults inside --output-dir.")

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument("--show-viewer", action="store_true", help="Show a live OpenCV viewer while processing.")
    io_group.add_argument("--save-output-video", action="store_true", help="Save the rendered tracking MP4.")
    io_group.add_argument("--save-frames", action="store_true", help="Save rendered PNG frames.")
    io_group.add_argument("--max-frames", type=int, default=None, help="Maximum number of processed frames for both videos and frame folders.")
    io_group.add_argument("--start-frame", type=int, default=0, help="First frame index to process.")
    io_group.add_argument(
        "--input-video-fps",
        type=float,
        default=None,
        help="FPS used to split/sample an input video into frames. No effect for frame folders.",
    )
    io_group.add_argument("--stride", type=int, default=1, help="Process one every N available/sampled frames for both videos and frame folders.")
    io_group.add_argument(
        "--output-fps",
        type=float,
        default=30.0,
        help="FPS of the rendered output video for both videos and frame folders.",
    )
    io_group.add_argument("--display-scale", type=float, default=1.0, help="Scale factor for preview window only.")

    yolo = parser.add_argument_group("YOLO")
    yolo.add_argument("--yolo-conf", type=float, default=0.25, help="YOLO confidence threshold.")
    yolo.add_argument("--yolo-iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    yolo.add_argument("--yolo-imgsz", type=int, default=960, help="YOLO inference image size.")
    yolo.add_argument("--max-det", type=int, default=100, help="Maximum YOLO detections per frame.")
    yolo.add_argument("--classes", default=None, help="Comma-separated YOLO class ids or names to keep.")
    yolo.add_argument("--mask-erosion-px", type=int, default=0, help="Pixels for optional mask erosion.")
    yolo.add_argument("--mask-erosion-iters", type=int, default=1, help="Iterations for optional mask erosion.")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Torch device selection.")
    runtime.add_argument("--dino-model-label", default=None, help="Override dino.model_label from config, e.g. S, B, L.")
    runtime.add_argument("--verbose-timing", action="store_true", help="Print REMIND stage timing per frame.")
    runtime.add_argument("--mask-alpha", type=float, default=0.42, help="Mask overlay opacity.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.source is not None:
        source = args.source.expanduser().resolve()
        scene_name = str(args.scene).strip()
    else:
        scene_name = str(args.scene).strip()
        try:
            source = resolve_scene_source(scene_name, test_root=args.test_root, input_kind=args.input_kind)
        except FileNotFoundError as exc:
            raise SystemExit(f"error: {exc}") from None

    if not source.exists():
        raise SystemExit(f"error: Source not found: {source}")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else _default_output_dir(source, scene=scene_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered_frames_dir = output_dir / "frames"
    if args.save_frames:
        rendered_frames_dir.mkdir(parents=True, exist_ok=True)

    output_video = args.output_video.expanduser().resolve() if args.output_video else output_dir / "tracking.mp4"
    frames_csv = output_dir / "frames.csv"
    detections_jsonl = output_dir / "detections.jsonl"
    summary_json = output_dir / "summary.json"

    frame_source = FrameSource(
        source,
        start_frame=args.start_frame,
        stride=args.stride,
        input_video_fps=args.input_video_fps,
        frames_timestamp_fps=args.output_fps,
    )
    save_fps = max(0.1, float(args.output_fps))
    try:
        args.yolo_model = resolve_yolo_model(args.yolo_model, models_dir=args.yolo_dir)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None

    print("[REMIND-VIDEO] Initializing models...")
    from pipeline.initialization import initialize_system
    from pipeline.reid_pipeline import ReIDPipeline

    config = _configure(args, output_dir)
    ctx = initialize_system(config)
    pipeline = ReIDPipeline(ctx)
    print(f"[REMIND-VIDEO] Scene: {scene_name}")
    print(f"[REMIND-VIDEO] Source: {source}")
    print(f"[REMIND-VIDEO] Output: {output_dir}")
    print(f"[REMIND-VIDEO] YOLO model: {args.yolo_model}")
    print(f"[REMIND-VIDEO] Device: {ctx.device}")
    print(
        f"[REMIND-VIDEO] FPS: native_input={frame_source.native_fps:.3f} "
        f"input_video_sample={frame_source.input_sample_fps:.3f} "
        f"stride={int(args.stride)} processed={frame_source.processed_fps:.3f} "
        f"output={save_fps:.3f}"
    )

    writer: cv2.VideoWriter | None = None
    processed = 0
    t_run = perf_counter()

    with open(frames_csv, "w", newline="", encoding="utf-8") as csv_fh, open(
        detections_jsonl, "w", encoding="utf-8"
    ) as jsonl_fh:
        csv_writer = csv.DictWriter(
            csv_fh,
            fieldnames=[
                "frame_idx",
                "frame_id",
                "timestamp",
                "name",
                "detections",
                "matches",
                "created",
                "ambiguous",
                "provisional",
                "visible",
                "elapsed_seconds",
            ],
        )
        csv_writer.writeheader()

        if args.show_viewer:
            cv2.namedWindow("REMIND tracking", cv2.WINDOW_NORMAL)

        try:
            for item in frame_source.iter_frames(max_frames=args.max_frames):
                t0 = perf_counter()
                p_out, _a_out, u_out = pipeline.process_frame(
                    frame=item.frame,
                    frame_id=int(item.frame_id),
                    timestamp=float(item.timestamp),
                )
                elapsed = perf_counter() - t0
                processed += 1
                fps_now = 1.0 / elapsed if elapsed > 0 else 0.0

                summary = getattr(u_out, "summary", {}) or {}
                header = (
                    f"{item.name} | frame={item.frame_id} | det={len(p_out.detections or [])} | "
                    f"visible={summary.get('n_visible', 0)} | new={summary.get('n_created', 0)} | "
                    f"amb={summary.get('n_ambiguous', 0)} | {fps_now:.2f} FPS"
                )
                rendered, det_details = render_frame(
                    item.frame,
                    p_out.detections or [],
                    u_out,
                    header=header,
                    alpha=float(args.mask_alpha),
                )

                if args.save_output_video:
                    if writer is None:
                        writer = _open_writer(output_video, rendered.shape, save_fps)
                    writer.write(rendered)

                if args.save_frames:
                    cv2.imwrite(str(rendered_frames_dir / f"frame_{item.frame_id:06d}.png"), rendered)

                csv_writer.writerow(
                    {
                        "frame_idx": int(item.frame_idx),
                        "frame_id": int(item.frame_id),
                        "timestamp": float(item.timestamp),
                        "name": item.name,
                        "detections": int(len(p_out.detections or [])),
                        "matches": int(summary.get("n_matches", 0)),
                        "created": int(summary.get("n_created", 0)),
                        "ambiguous": int(summary.get("n_ambiguous", 0)),
                        "provisional": int(summary.get("n_provisional", 0)),
                        "visible": int(summary.get("n_visible", 0)),
                        "elapsed_seconds": float(elapsed),
                    }
                )
                jsonl_fh.write(
                    json.dumps(
                        {
                            "frame_idx": int(item.frame_idx),
                            "frame_id": int(item.frame_id),
                            "timestamp": float(item.timestamp),
                            "name": item.name,
                            "detections": det_details,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

                if args.show_viewer:
                    preview = rendered
                    if float(args.display_scale) != 1.0:
                        scale = max(0.05, float(args.display_scale))
                        preview = cv2.resize(rendered, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow("REMIND tracking", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("[REMIND-VIDEO] Stopped by user.")
                        break
        finally:
            if writer is not None:
                writer.release()
            if args.show_viewer:
                cv2.destroyAllWindows()

    total_seconds = perf_counter() - t_run
    summary_payload = {
        "source": str(source),
        "scene": str(scene_name),
        "output_dir": str(output_dir),
        "output_video": str(output_video) if args.save_output_video else None,
        "frames_csv": str(frames_csv),
        "detections_jsonl": str(detections_jsonl),
        "processed_frames": int(processed),
        "total_seconds": float(total_seconds),
        "avg_fps": float(processed / total_seconds) if total_seconds > 0 else 0.0,
        "yolo_model": str(args.yolo_model),
        "device": str(ctx.device),
        "native_input_fps": float(frame_source.native_fps),
        "input_video_sample_fps": float(frame_source.input_sample_fps),
        "processed_fps": float(frame_source.processed_fps),
        "stride": int(args.stride),
        "output_fps": float(save_fps),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    print("[REMIND-VIDEO] Done.")
    if args.save_output_video:
        print(f"[REMIND-VIDEO] Video: {output_video}")
    print(f"[REMIND-VIDEO] CSV: {frames_csv}")
    print(f"[REMIND-VIDEO] Detections: {detections_jsonl}")
    print(f"[REMIND-VIDEO] Summary: {summary_json}")


if __name__ == "__main__":
    main()
