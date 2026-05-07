# detection/yolo_segmenter.py

from __future__ import annotations

from detection.detection import Detection

import numpy as np
import cv2

from ultralytics import YOLO


class YoloSegmenter:
    """
    Wrapper YOLO
    """

    def __init__(self, config: dict, device: str):
        self.config = config
        self.device = device

        self.model = None
        self.class_id_to_name = None

        self.sys_cfg = self.config.get("system", {}) or {}
        self.yolo_cfg = self.config.get("yolo", {}) or {}

    def load_model(self) -> None:
        ycfg = self.yolo_cfg
        weights = self.resolve_weights_path(ycfg)
        model = YOLO(weights)

        try:
            model.to(self.device)
        except Exception:
            pass

        self.model = model
        self.class_id_to_name = self.build_id_to_name_map(model)

    def resolve_weights_path(self, config_yolo: dict) -> str:
        model_label = config_yolo.get("model_label", None)
        models_map = config_yolo.get("models", None)
        key = str(model_label).strip().upper()

        return str(models_map[key])

    def build_id_to_name_map(self, model) -> dict:
        names = getattr(model, "names", None)
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(n) for i, n in enumerate(names)}
        return {}

    def resolve_classes(self, classes_spec):
        """
        classes_spec puede ser:
          - None
          - list[int]
          - list[str]  (nombres exactos de model.names)
        Devuelve list[int] o None.
        """
        if classes_spec is None:
            return None

        if self.model is None:
            raise RuntimeError("YOLO no cargado. Llama a load_model() antes de resolve_classes().")

        id_to_name = self.class_id_to_name or self.build_id_to_name_map(self.model)
        name_to_id = {v.lower(): k for k, v in id_to_name.items()}

        if isinstance(classes_spec, list) and all(isinstance(x, int) for x in classes_spec):
            valid_ids = set(id_to_name.keys())
            out = [int(x) for x in classes_spec if int(x) in valid_ids]
            return out if out else None

        if isinstance(classes_spec, list) and all(isinstance(x, str) for x in classes_spec):
            out = []
            for s in classes_spec:
                sid = name_to_id.get(str(s).lower())
                if sid is not None:
                    out.append(int(sid))
            return out if out else None

        return None


    # -------------------------
    # Mask post-processing
    # -------------------------

    def erode_mask(self, mask_bool: np.ndarray, erosion_px: int, erosion_iters: int) -> np.ndarray:
        """
        Erosiona una máscara booleana (H,W) en píxeles.
        """
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
        """
        Centroide y área a partir de máscara booleana.
        """
        ys, xs = np.nonzero(mask)
        area = float(len(xs))

        if area <= 0:
            return {"center": (None, None), "area": 0.0}

        cx = float(xs.mean())
        cy = float(ys.mean())

        return {
            "center": (cx, cy),
            "area": area,
        }

    # -------------------------
    # Inference
    # -------------------------

    def segment(self, frame, frame_id: int, timestamp: float) -> list:
        """
        Ejecuta YOLO-seg sobre un frame
        """
        if self.model is None:
            raise RuntimeError("YOLO no cargado. Llama a load_model() antes de segment().")

        sys_cfg = self.sys_cfg
        ycfg = self.yolo_cfg

        conf = float(ycfg.get("conf_th", 0.25))
        iou = float(ycfg.get("iou_th", 0.7))
        imgsz = int(sys_cfg.get("input_width_size", 1056))
        max_det = int(ycfg.get("max_det", 100))
        classes_spec = ycfg.get("classes", None)

        erosion_px = int(ycfg.get("mask_erosion_px", 0))
        erosion_iters = int(ycfg.get("mask_erosion_iters", 1))

        classes_ids = self.resolve_classes(classes_spec)

        results = self.model.predict(
            frame,
            verbose=False,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            classes=classes_ids,
        )

        if results is None or len(results) == 0:
            return []

        r0 = results[0]

        if getattr(r0, "masks", None) is None:
            return []

        h, w = frame.shape[:2]

        masks = r0.masks.data
        boxes_xyxy = r0.boxes.xyxy
        classes_idx = r0.boxes.cls
        scores = r0.boxes.conf

        masks_np = masks.detach().cpu().numpy()
        boxes_np = boxes_xyxy.detach().cpu().numpy()
        cls_np = classes_idx.detach().cpu().numpy().astype(int)
        conf_np = scores.detach().cpu().numpy()

        detections = []

        for k in range(masks_np.shape[0]):
            m = masks_np[k]
            mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) >= 0.5

            if erosion_px > 0:
                mask = self.erode_mask(mask, erosion_px=erosion_px, erosion_iters=erosion_iters)

            geom = self.mask_center_and_area(mask)

            bbox = tuple(float(x) for x in boxes_np[k].tolist())
            class_id = int(cls_np[k])
            score = float(conf_np[k])

            det = Detection(
                detection_id=int(k),
                class_id=class_id,
                frame_id=frame_id,
                timestamp=timestamp,
                bbox=bbox,
                mask=mask,
                confidence=score,
                geom=geom,
            )
            class_name = self.class_id_to_name.get(int(class_id), None)
            det.class_name = class_name
            det.original_class_name = class_name
            detections.append(det)

        return detections

