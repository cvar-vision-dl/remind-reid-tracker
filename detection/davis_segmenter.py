from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import re
from pathlib import Path

import cv2
import numpy as np

from detection.detection import Detection
from utils.time import ExecutionTimer


class DavisSegmenter:
    """
    Segmenter compatible with YoloSegmenter, reading DAVIS masks.

    Public contract:
      - load_model()
      - segment(frame, frame_id, timestamp) -> list[Detection]
      - class_id_to_name
    """

    _LABEL_SUFFIX_RE = re.compile(r"^(?P<class_name>.+)_(?P<instance_id>\d+)$")

    def __init__(self, config: dict, device: str):
        self.config = config or {}
        self.device = device

        self.sys_cfg = self.config.get("system", {}) or {}
        self.dino_cfg = self.config.get("dino", {}) or {}
        self.perception_cfg = self.config.get("perception", {}) or {}
        self.davis_cfg = self.config.get("davis", {}) or {}

        self.class_id_to_name: dict[int, str] = {}
        self.class_name_to_id: dict[str, int] = {}
        self.instance_id_to_class_id: dict[int, int] = {}
        self.instance_id_to_label: dict[int, str] = {}
        self.instance_id_to_original_label: dict[int, str] = {}
        self.instance_id_to_original_class_name: dict[int, str] = {}

        self.meta_path: Path | None = None
        self.annotations_dir: Path | None = None
        self.sequence_name_resolved: str | None = None
        self.prefetch_enabled = bool(self.davis_cfg.get("prefetch_annotations", True))
        self.prefetch_distance = max(1, int(self.davis_cfg.get("prefetch_distance", 1)))
        self._prefetch_executor: ThreadPoolExecutor | None = None
        self._prefetch_future: Future | None = None
        self._prefetch_frame_id: int | None = None
        self.last_timings_seconds: dict[str, float] = {}
        self.last_instance_stats_route: str | None = None

    def load_model(self) -> None:
        meta_path = self.resolve_meta_path()
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f) or {}

        self.meta_path = meta_path
        self.sequence_name_resolved = str(meta.get("sequence", "")).strip() or None
        self.annotations_dir = self.resolve_annotations_dir(meta)
        self.instance_id_to_label = self.build_instance_id_to_label(meta)
        self.instance_id_to_original_label = self.resolve_original_instance_id_to_label(meta_path=meta_path)
        self.instance_id_to_original_class_name = {
            int(instance_id): str(class_name)
            for instance_id, class_name in (
                (
                    int(instance_id),
                    self.extract_class_name(label),
                )
                for instance_id, label in (self.instance_id_to_original_label or {}).items()
            )
            if class_name is not None
        }

        class_names = sorted(
            {
                self.extract_class_name(label)
                for label in self.instance_id_to_label.values()
                if self.extract_class_name(label) is not None
            }
        )

        self.class_id_to_name = {idx: name for idx, name in enumerate(class_names)}
        self.class_name_to_id = {name: idx for idx, name in self.class_id_to_name.items()}

        self.instance_id_to_class_id = {}
        for instance_id, label in self.instance_id_to_label.items():
            class_name = self.extract_class_name(label)
            if class_name is None:
                continue
            class_id = self.class_name_to_id.get(class_name)
            if class_id is not None:
                self.instance_id_to_class_id[int(instance_id)] = int(class_id)

        if self.prefetch_enabled and self._prefetch_executor is None:
            self._prefetch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="davis_mask_prefetch")

    def __del__(self) -> None:
        executor = getattr(self, "_prefetch_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

    def resolve_workspace_root(self) -> Path:
        return Path(__file__).resolve().parents[3]

    def resolve_sequence_name(self) -> str:
        if self.sequence_name_resolved:
            return str(self.sequence_name_resolved)

        return self.preferred_sequence_name()

    def resolve_base_sequence_name(self) -> str:
        seq = self.davis_cfg.get("sequence_name", None)
        if seq is not None:
            seq = str(seq).strip()
            if seq:
                return seq

        input_cfg = self.config.get("input", {}) or {}
        frames_dir = input_cfg.get("frames_dir", None)
        if frames_dir is not None:
            name = Path(str(frames_dir).strip()).name
            if name:
                return str(name)

        raise ValueError(
            "DavisSegmenter needs a sequence. "
            "Use config['davis']['sequence_name'] or config['input']['frames_dir']."
        )

    def resolve_variant(self) -> str:
        raw = str(self.davis_cfg.get("variant", "raw") or "raw").strip().lower()
        return raw if raw in {"raw", "bench"} else "raw"

    def is_bench_sequence_name(self, sequence_name: str) -> bool:
        return str(sequence_name).strip().lower().endswith("_bench")

    def sequence_name_for_variant(self, base_sequence_name: str, variant: str) -> str:
        base = str(base_sequence_name).strip()
        if not base:
            return base

        if variant == "bench":
            return base if self.is_bench_sequence_name(base) else f"{base}_bench"

        if self.is_bench_sequence_name(base):
            return base[: -len("_bench")]
        return base

    def candidate_sequence_names(self) -> list[str]:
        base = self.resolve_base_sequence_name()
        variant = self.resolve_variant()
        seq = str(self.sequence_name_for_variant(base, variant)).strip()
        return [seq] if seq else []

    def preferred_sequence_name(self) -> str:
        candidates = self.candidate_sequence_names()
        if not candidates:
            raise ValueError("No se pudo resolver ninguna secuencia DAVIS.")
        return candidates[0]

    def resolve_path(self, raw_path: str | None) -> Path | None:
        if raw_path is None:
            return None
        raw = str(raw_path).strip()
        if not raw:
            return None

        p = Path(raw)
        if p.is_absolute():
            return p
        return (self.resolve_workspace_root() / p).resolve()

    def resolve_meta_path(self) -> Path:
        explicit = self.resolve_path(self.davis_cfg.get("meta_path", None))
        if explicit is not None:
            if not explicit.exists():
                raise FileNotFoundError(f"Meta DAVIS no encontrado: {explicit}")
            return explicit

        for seq in self.candidate_sequence_names():
            found = self.find_meta_path_for_sequence(seq)
            if found is not None:
                return found

        requested = self.preferred_sequence_name()

        raise FileNotFoundError(
            f"DAVIS meta not found for sequence_name={requested} "
            f"(variant={self.resolve_variant()}) in {(self.resolve_workspace_root() / 'DAVIS_OUT').resolve()}"
        )

    def find_meta_path_for_sequence(self, sequence_name: str) -> Path | None:
        seq = str(sequence_name).strip()
        if not seq:
            return None

        davis_root = (self.resolve_workspace_root() / "DAVIS_OUT").resolve()
        candidates = [
            (davis_root / f"meta{seq}.json").resolve(),
            (davis_root / f"meta{seq.upper()}.json").resolve(),
            (davis_root / f"meta{seq.lower()}.json").resolve(),
        ]
        for p in candidates:
            if p.exists():
                return p

        seq_lower = seq.lower()
        for p in davis_root.glob("meta*.json"):
            stem_tail = p.stem[4:]
            if stem_tail.lower() == seq_lower:
                return p.resolve()
        return None

    def resolve_original_instance_id_to_label(self, *, meta_path: Path) -> dict[int, str]:
        current = self.build_instance_id_to_label(json.loads(meta_path.read_text(encoding="utf-8")) or {})
        if self.resolve_variant() != "bench":
            return dict(current)

        raw_seq = self.sequence_name_for_variant(self.resolve_base_sequence_name(), "raw")
        raw_meta_path = self.find_meta_path_for_sequence(raw_seq)
        if raw_meta_path is None or raw_meta_path.resolve() == meta_path.resolve():
            return dict(current)

        try:
            raw_meta = json.loads(raw_meta_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return dict(current)

        raw_labels = self.build_instance_id_to_label(raw_meta)
        return dict(raw_labels) if raw_labels else dict(current)

    def resolve_annotations_dir(self, meta: dict) -> Path:
        explicit = self.resolve_path(self.davis_cfg.get("annotations_dir", None))
        if explicit is not None:
            if not explicit.is_dir():
                raise FileNotFoundError(f"Directorio de anotaciones DAVIS no encontrado: {explicit}")
            return explicit

        davis_root = self.resolve_path(self.davis_cfg.get("davis_root", None))
        if davis_root is None:
            meta_root = meta.get("davis_root", "DAVIS_OUT")
            davis_root = self.resolve_path(str(meta_root))

        davis_res = str(self.davis_cfg.get("davis_res", meta.get("davis_res", "raw"))).strip() or "raw"
        seq = str(meta.get("sequence", self.resolve_sequence_name())).strip()
        out = (davis_root / "Annotations" / davis_res / seq).resolve()

        if not out.is_dir():
            raise FileNotFoundError(f"Directorio de anotaciones DAVIS no encontrado: {out}")
        return out

    def build_instance_id_to_label(self, meta: dict) -> dict[int, str]:
        raw = meta.get("id_to_label", {}) or {}
        out: dict[int, str] = {}
        for instance_id, label in raw.items():
            try:
                iid = int(instance_id)
            except Exception:
                continue
            out[iid] = str(label)
        return out

    def extract_class_name(self, label: str | None) -> str | None:
        if label is None:
            return None
        s = str(label).strip()
        if not s:
            return None
        m = self._LABEL_SUFFIX_RE.match(s)
        if m is None:
            return s
        return str(m.group("class_name"))

    def resolve_classes(self, classes_spec):
        if classes_spec is None:
            return None

        if isinstance(classes_spec, list) and all(isinstance(x, int) for x in classes_spec):
            valid_ids = set(self.class_id_to_name.keys())
            out = [int(x) for x in classes_spec if int(x) in valid_ids]
            return set(out) if out else None

        if isinstance(classes_spec, list) and all(isinstance(x, str) for x in classes_spec):
            out = []
            for name in classes_spec:
                cid = self.class_name_to_id.get(str(name).strip())
                if cid is not None:
                    out.append(int(cid))
            return set(out) if out else None

        return None

    def erode_mask(self, mask_bool: np.ndarray, erosion_px: int, erosion_iters: int) -> np.ndarray:
        r = int(max(0, erosion_px))
        it = int(max(1, erosion_iters))
        if r <= 0:
            return mask_bool

        k = 2 * r + 1
        kernel = np.ones((k, k), dtype=np.uint8)

        m = mask_bool.astype(np.uint8, copy=False) * 255
        m = cv2.erode(m, kernel, iterations=it)
        return m >= 128

    def mask_center_and_area(self, mask: np.ndarray) -> dict:
        ys, xs = np.nonzero(mask)
        area = float(len(xs))
        if area <= 0:
            return {"center": (None, None), "area": 0.0}

        return {
            "center": (float(xs.mean()), float(ys.mean())),
            "area": area,
        }

    def mask_bbox_xyxy(self, mask: np.ndarray) -> tuple[float, float, float, float] | None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1
        return (float(x1), float(y1), float(x2), float(y2))

    def mask_bbox_center_and_area(
        self,
        mask: np.ndarray,
        *,
        offset_xy: tuple[int, int] = (0, 0),
    ) -> tuple[tuple[float, float, float, float] | None, dict]:
        ys, xs = np.nonzero(mask)
        area = int(len(xs))
        if area <= 0:
            return None, {"center": (None, None), "area": 0.0}

        x0 = int(offset_xy[0])
        y0 = int(offset_xy[1])
        xs_f = xs.astype(np.float64, copy=False) + float(x0)
        ys_f = ys.astype(np.float64, copy=False) + float(y0)

        bbox = (
            float(int(xs.min()) + x0),
            float(int(ys.min()) + y0),
            float(int(xs.max()) + 1 + x0),
            float(int(ys.max()) + 1 + y0),
        )
        geom = {
            "center": (float(xs_f.mean()), float(ys_f.mean())),
            "area": float(area),
        }
        return bbox, geom

    def instance_stats_from_mask(self, mask_labeled: np.ndarray) -> dict[int, dict]:
        self.last_instance_stats_route = None
        arr = np.asarray(mask_labeled)
        if arr.ndim != 2 or arr.size == 0:
            return {}

        fg = arr > 0
        if not np.any(fg):
            return {}

        ys, xs = np.nonzero(fg)
        ids = arr[fg].astype(np.int32, copy=False)
        max_id = int(ids.max(initial=0))
        xs_i = xs.astype(np.int32, copy=False)
        ys_i = ys.astype(np.int32, copy=False)
        self.last_instance_stats_route = "dense"
        if max_id <= 0:
            return {}

        counts = np.bincount(ids, minlength=max_id + 1).astype(np.int64, copy=False)
        valid_ids = np.flatnonzero(counts > 0)
        valid_ids = valid_ids[valid_ids > 0]
        if valid_ids.size <= 0:
            return {}

        sum_x = np.bincount(ids, weights=xs.astype(np.float64, copy=False), minlength=max_id + 1)
        sum_y = np.bincount(ids, weights=ys.astype(np.float64, copy=False), minlength=max_id + 1)

        min_x = np.full(max_id + 1, arr.shape[1], dtype=np.int32)
        min_y = np.full(max_id + 1, arr.shape[0], dtype=np.int32)
        max_x = np.full(max_id + 1, -1, dtype=np.int32)
        max_y = np.full(max_id + 1, -1, dtype=np.int32)
        np.minimum.at(min_x, ids, xs_i)
        np.minimum.at(min_y, ids, ys_i)
        np.maximum.at(max_x, ids, xs_i)
        np.maximum.at(max_y, ids, ys_i)

        out: dict[int, dict] = {}
        for instance_id in valid_ids.tolist():
            count = int(counts[instance_id])
            out[int(instance_id)] = {
                "bbox": (
                    float(int(min_x[instance_id])),
                    float(int(min_y[instance_id])),
                    float(int(max_x[instance_id]) + 1),
                    float(int(max_y[instance_id]) + 1),
                ),
                "geom": {
                    "center": (
                        float(sum_x[instance_id] / float(count)),
                        float(sum_y[instance_id] / float(count)),
                    ),
                    "area": float(count),
                },
            }
        return out

    def frame_filename(self, frame_id: int) -> str:
        return f"frame_{int(frame_id):06d}.png"

    def annotation_mask_path(self, frame_id: int) -> Path:
        if self.annotations_dir is None:
            raise RuntimeError("DavisSegmenter is not initialized. Call load_model() before segment().")
        return self.annotations_dir / self.frame_filename(frame_id)

    def read_annotation_mask_from_path(self, path: Path) -> np.ndarray | None:
        if not path.exists():
            return None

        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            return None
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask

    def schedule_prefetch(self, frame_id: int) -> None:
        if not self.prefetch_enabled or self._prefetch_executor is None:
            return

        target_frame_id = int(frame_id)
        if self._prefetch_frame_id == target_frame_id and self._prefetch_future is not None:
            return

        self._prefetch_frame_id = int(target_frame_id)
        path = self.annotation_mask_path(target_frame_id)
        self._prefetch_future = self._prefetch_executor.submit(self.read_annotation_mask_from_path, path)

    def read_annotation_mask(self, frame_id: int) -> np.ndarray | None:
        if self.annotations_dir is None:
            raise RuntimeError("DavisSegmenter is not initialized. Call load_model() before segment().")

        fid = int(frame_id)
        mask = None
        if (
            self.prefetch_enabled
            and self._prefetch_frame_id == fid
            and self._prefetch_future is not None
        ):
            mask = self._prefetch_future.result()
            self._prefetch_future = None
            self._prefetch_frame_id = None
        else:
            path = self.annotation_mask_path(fid)
            mask = self.read_annotation_mask_from_path(path)

        if self.prefetch_enabled:
            self.schedule_prefetch(fid + self.prefetch_distance)

        return mask

    def resize_mask_to_input_space(self, raw_mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
        target_h, target_w = int(target_shape[0]), int(target_shape[1])

        if raw_mask.shape[:2] == (target_h, target_w):
            return raw_mask

        orig_h, orig_w = raw_mask.shape[:2]
        input_width = int(self.sys_cfg.get("input_width_size", orig_w))
        if orig_w > input_width:
            scale = input_width / float(orig_w)
            resized_w = int(round(input_width))
            resized_h = int(round(orig_h * scale))
            resized_mask = cv2.resize(raw_mask, (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
        else:
            resized_mask = raw_mask

        if resized_mask.shape[:2] == (target_h, target_w):
            return resized_mask

        patch_multiple = int(self.dino_cfg.get("default_patch_size", 16))
        center = bool(self.perception_cfg.get("full_center_crop", True))
        h_rs, w_rs = resized_mask.shape[:2]
        h_eff = (h_rs // patch_multiple) * patch_multiple
        w_eff = (w_rs // patch_multiple) * patch_multiple

        if h_eff > 0 and w_eff > 0:
            if center:
                y0 = (h_rs - h_eff) // 2
                x0 = (w_rs - w_eff) // 2
            else:
                y0, x0 = 0, 0
            aligned = resized_mask[y0:y0 + h_eff, x0:x0 + w_eff]
            if aligned.shape[:2] == (target_h, target_w):
                return aligned

        return cv2.resize(raw_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    def segment(self, frame, frame_id: int, timestamp: float) -> list:
        if not self.class_id_to_name:
            raise RuntimeError("DavisSegmenter is not initialized. Call load_model() before segment().")

        timer = ExecutionTimer()
        self.last_timings_seconds = {}
        try:
            raw_mask = timer.run("read_mask", self.read_annotation_mask, frame_id=frame_id)
            if raw_mask is None:
                return []

            target_mask = timer.run("resize_mask", self.resize_mask_to_input_space, raw_mask, frame.shape[:2])
            allowed_class_ids = timer.run("resolve_classes", self.resolve_classes, self.davis_cfg.get("classes", None))

            erosion_px = int(self.davis_cfg.get("mask_erosion_px", 0))
            erosion_iters = int(self.davis_cfg.get("mask_erosion_iters", 1))

            instance_stats = timer.run("instance_stats", self.instance_stats_from_mask, target_mask)

            detections = []
            next_det_id = 0

            with timer.measure("build_detections"):
                for instance_id in sorted(int(x) for x in instance_stats.keys()):
                    instance_id = int(instance_id)
                    if instance_id <= 0:
                        continue

                    class_id = self.instance_id_to_class_id.get(instance_id, None)
                    if class_id is None:
                        continue
                    if allowed_class_ids is not None and int(class_id) not in allowed_class_ids:
                        continue

                    stats = instance_stats.get(int(instance_id), None)
                    if not isinstance(stats, dict):
                        continue

                    bbox = stats.get("bbox", None)
                    if bbox is None or len(bbox) < 4:
                        continue

                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    mask = np.zeros(target_mask.shape[:2], dtype=bool)
                    mask_roi = (target_mask[y1:y2, x1:x2] == int(instance_id))

                    if erosion_px > 0:
                        mask_roi = self.erode_mask(mask_roi, erosion_px=erosion_px, erosion_iters=erosion_iters)
                        if not np.any(mask_roi):
                            continue
                        mask[y1:y2, x1:x2] = mask_roi
                        bbox, geom = self.mask_bbox_center_and_area(mask_roi, offset_xy=(x1, y1))
                    else:
                        mask[y1:y2, x1:x2] = mask_roi
                        geom = dict(stats.get("geom", {}) or {})

                    if bbox is None:
                        continue

                    detections.append(
                        Detection(
                            detection_id=int(next_det_id),
                            class_id=int(class_id),
                            frame_id=frame_id,
                            timestamp=timestamp,
                            bbox=bbox,
                            mask=mask,
                            confidence=1.0,
                            geom=geom,
                        )
                    )
                    det = detections[-1]
                    det.class_name = self.class_id_to_name.get(int(class_id), None)
                    det.instance_label = self.instance_id_to_label.get(int(instance_id), None)
                    det.original_class_name = self.instance_id_to_original_class_name.get(
                        int(instance_id),
                        det.class_name,
                    )
                    det.original_instance_label = self.instance_id_to_original_label.get(
                        int(instance_id),
                        det.instance_label,
                    )
                    next_det_id += 1

            return detections
        finally:
            self.last_timings_seconds = timer.snapshot_seconds()
            route = str(self.last_instance_stats_route).strip() if self.last_instance_stats_route is not None else ""
            if route:
                self.last_timings_seconds[f"instance_stats_route/{route}"] = 0.0
