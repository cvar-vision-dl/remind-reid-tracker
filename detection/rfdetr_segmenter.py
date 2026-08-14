"""RF-DETR instance-segmentation adapter for REMIND detections."""

from __future__ import annotations

import importlib

import cv2
import numpy as np

from detection.detection import Detection
from utils.time import ExecutionTimer


class RFDETRSegmenter:
    """Adapt optional RF-DETR segmentation predictions to tracker detections."""

    _MODEL_CLASS_NAMES = {
        "nano": "RFDETRSegNano",
        "small": "RFDETRSegSmall",
        "medium": "RFDETRSegMedium",
        "large": "RFDETRSegLarge",
        "xlarge": "RFDETRSegXLarge",
        "2xlarge": "RFDETRSeg2XLarge",
    }

    def __init__(self, config: dict, device: str):
        """Store the runtime configuration without importing the optional backend."""
        self.config = config or {}
        self.device = device
        self.rfdetr_cfg = self.config.get("rfdetr", {}) or {}
        self.model = None
        self.class_id_to_name: dict[int, str] = {}
        self.last_timings_seconds: dict[str, float] = {}

    def load_model(self) -> None:
        """Construct the selected RF-DETR segmentation model and its class map."""
        model_class = self._resolve_model_class()
        kwargs: dict[str, object] = {"device": self.device}
        weights = self.rfdetr_cfg.get("pretrain_weights")
        if weights is not None and str(weights).strip():
            kwargs["pretrain_weights"] = str(weights)

        self.model = model_class(**kwargs)
        self.class_id_to_name = self._build_class_id_to_name(self.model)

    def _resolve_model_class(self):
        """Return the configured RF-DETR segmentation class from the optional package."""
        variant = (
            str(self.rfdetr_cfg.get("model_variant", "medium") or "medium")
            .strip()
            .lower()
        )
        class_name = self._MODEL_CLASS_NAMES.get(variant)
        if class_name is None:
            choices = ", ".join(self._MODEL_CLASS_NAMES)
            raise ValueError(
                f"Unsupported RF-DETR model_variant={variant!r}; choose one of: {choices}."
            )

        try:
            module = importlib.import_module("rfdetr")
        except ModuleNotFoundError as exc:
            if exc.name == "rfdetr":
                raise RuntimeError(
                    "RF-DETR backend requires the optional 'rfdetr' package. "
                    "Install it with `pip install rfdetr`."
                ) from exc
            raise

        try:
            return getattr(module, class_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"Installed rfdetr package does not provide {class_name} for variant {variant!r}."
            ) from exc

    def _build_class_id_to_name(self, model) -> dict[int, str]:
        """Normalize RF-DETR class names into the tracker vocabulary map."""
        names = getattr(model, "class_names", None)
        if isinstance(names, dict):
            return {int(class_id): str(name) for class_id, name in names.items()}
        if isinstance(names, (list, tuple)):
            return {class_id: str(name) for class_id, name in enumerate(names)}
        return {}

    def _resolve_classes(self) -> set[int] | None:
        """Resolve optional ID or name filtering against the loaded class map."""
        classes = self.rfdetr_cfg.get("classes")
        if classes is None:
            return None
        if not isinstance(classes, list):
            raise ValueError(
                "rfdetr.classes must be a list of class IDs or class names."
            )
        name_to_id = {
            name.strip().lower(): class_id
            for class_id, name in self.class_id_to_name.items()
        }
        resolved: set[int] = set()
        for class_spec in classes:
            if isinstance(class_spec, int):
                if class_spec in self.class_id_to_name:
                    resolved.add(class_spec)
                continue
            if isinstance(class_spec, str):
                class_id = name_to_id.get(class_spec.strip().lower())
                if class_id is not None:
                    resolved.add(class_id)
                continue
            raise ValueError("rfdetr.classes must contain class IDs or class names.")
        if classes and not resolved:
            requested = ", ".join(repr(class_spec) for class_spec in classes)
            available = ", ".join(
                f"{class_id}: {name!r}"
                for class_id, name in self.class_id_to_name.items()
            )
            raise ValueError(
                "rfdetr.classes matched no loaded RF-DETR classes. "
                f"Requested: {requested}. Available: {available or 'none'}"
            )
        return resolved

    def _erode_mask(self, mask: np.ndarray) -> np.ndarray:
        """Apply the configured erosion while preserving a boolean two-dimensional mask."""
        erosion_px = max(0, int(self.rfdetr_cfg.get("mask_erosion_px", 0)))
        if erosion_px == 0:
            return mask
        erosion_iters = max(1, int(self.rfdetr_cfg.get("mask_erosion_iters", 1)))
        kernel_size = 2 * erosion_px + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        eroded = cv2.erode(
            mask.astype(np.uint8, copy=False), kernel, iterations=erosion_iters
        )
        return eroded.astype(bool, copy=False)

    def _mask_bbox_and_geometry(
        self, mask: np.ndarray
    ) -> tuple[tuple[float, float, float, float] | None, dict]:
        """Derive tracker geometry and an xyxy box from a final boolean mask."""
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None, {"center": (None, None), "area": 0.0}
        return (
            (
                float(xs.min()),
                float(ys.min()),
                float(xs.max() + 1),
                float(ys.max() + 1),
            ),
            {"center": (float(xs.mean()), float(ys.mean())), "area": float(len(xs))},
        )

    def _validate_result_arrays(
        self, predictions
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return aligned RF-DETR result arrays or raise for a malformed prediction."""
        masks = getattr(predictions, "mask", None)
        if masks is None:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0, 0, 0), dtype=bool),
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float32),
            )
        boxes = np.asarray(getattr(predictions, "xyxy", None))
        masks_array = np.asarray(masks)
        class_ids = np.asarray(getattr(predictions, "class_id", None))
        confidences = np.asarray(getattr(predictions, "confidence", None))
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError(f"RF-DETR xyxy must have shape (N, 4), got {boxes.shape}.")
        if masks_array.ndim != 3:
            raise ValueError(
                f"RF-DETR mask must have shape (N, H, W), got {masks_array.shape}."
            )
        if class_ids.ndim != 1 or confidences.ndim != 1:
            raise ValueError(
                "RF-DETR class_id and confidence must each have shape (N,)."
            )
        counts = (
            boxes.shape[0],
            masks_array.shape[0],
            class_ids.shape[0],
            confidences.shape[0],
        )
        if len(set(counts)) != 1:
            raise ValueError(
                f"RF-DETR result array length mismatch for xyxy/mask/class_id/confidence: {counts}."
            )
        return boxes, masks_array, class_ids, confidences

    def _build_detection(
        self,
        *,
        raw_mask: np.ndarray,
        class_id: int,
        confidence: float,
        detection_id: int,
        frame_shape: tuple[int, int],
        frame_id: int,
        timestamp: float,
    ) -> Detection | None:
        """Convert one RF-DETR mask into a complete tracker detection."""
        if not np.isfinite(confidence):
            raise ValueError(
                f"RF-DETR confidence for class {class_id} is not finite: {confidence!r}."
            )

        frame_height, frame_width = frame_shape
        mask = raw_mask.astype(bool, copy=False)
        if mask.shape != frame_shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        mask = self._erode_mask(mask)
        bbox, geom = self._mask_bbox_and_geometry(mask)
        if bbox is None:
            return None

        detection = Detection(
            detection_id=detection_id,
            class_id=class_id,
            frame_id=frame_id,
            timestamp=timestamp,
            bbox=bbox,
            mask=mask,
            confidence=confidence,
            geom=geom,
        )
        class_name = self.class_id_to_name.get(class_id)
        detection.class_name = class_name
        detection.original_class_name = class_name
        return detection

    def segment(
        self, frame: np.ndarray, frame_id: int, timestamp: float
    ) -> list[Detection]:
        """Predict RGB instances for an aligned BGR frame and return valid tracker detections."""
        if self.model is None:
            raise RuntimeError(
                "RF-DETR is not loaded. Call load_model() before segment()."
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"RF-DETR expects an aligned BGR image with shape (H, W, 3), got {frame.shape}."
            )

        timer = ExecutionTimer()
        self.last_timings_seconds = {}
        try:
            rgb_frame = timer.run("bgr_to_rgb", lambda: frame[..., ::-1].copy())
            predictions = timer.run(
                "predict",
                self.model.predict,
                rgb_frame,
                threshold=float(self.rfdetr_cfg.get("threshold", 0.5)),
            )
            if predictions is None:
                return []

            boxes, masks, class_ids, confidences = timer.run(
                "validate_result", self._validate_result_arrays, predictions
            )
            if masks.shape[0] == 0:
                return []
            allowed_class_ids = timer.run("resolve_classes", self._resolve_classes)
            detections: list[Detection] = []
            with timer.measure("build_detections"):
                for index in range(masks.shape[0]):
                    class_id = int(class_ids[index])
                    if (
                        allowed_class_ids is not None
                        and class_id not in allowed_class_ids
                    ):
                        continue
                    detection = self._build_detection(
                        raw_mask=masks[index],
                        class_id=class_id,
                        confidence=float(confidences[index]),
                        detection_id=len(detections),
                        frame_shape=frame.shape[:2],
                        frame_id=frame_id,
                        timestamp=timestamp,
                    )
                    if detection is None:
                        continue
                    detections.append(detection)
            return detections
        finally:
            self.last_timings_seconds = timer.snapshot_seconds()
