from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from detection.davis_segmenter import DavisSegmenter


@dataclass
class GroundTruthObject:
    instance_id: int
    label: str
    class_name: str | None
    mask: np.ndarray
    area: int
    bbox_xyxy: tuple[int, int, int, int] | None


class DavisGroundTruthLoader:
    """
    Load DAVIS GT with preserved instance IDs.

    Reuses the same path resolution and spatial transform
    used by DavisSegmenter to ensure GT and detections live in the same
    evaluation space.
    """

    def __init__(self, config: dict):
        self.config = config or {}
        self.segmenter = DavisSegmenter(config=self.config, device="cpu")
        self.segmenter.load_model()
        self.ignored_class_names = self.resolve_ignored_class_names()

    @property
    def sequence_name(self) -> str:
        return self.segmenter.resolve_sequence_name()

    @property
    def annotations_dir(self):
        return self.segmenter.annotations_dir

    def resolve_ignored_class_names(self) -> set[str]:
        det_cfg = self.config.get("detector", {}) or {}
        classes_spec = det_cfg.get("ignored_classes", None)
        if not isinstance(classes_spec, list):
            return set()
        out: set[str] = set()
        for item in classes_spec:
            if not isinstance(item, str):
                continue
            name = str(item).strip().lower()
            if name:
                out.add(name)
        return out

    def load_frame(self, frame_id: int, target_shape: tuple[int, int]) -> dict[int, GroundTruthObject]:
        raw_mask = self.segmenter.read_annotation_mask(frame_id=int(frame_id))
        if raw_mask is None:
            return {}

        mask = self.segmenter.resize_mask_to_input_space(raw_mask, target_shape)
        out: dict[int, GroundTruthObject] = {}
        instance_stats = self.segmenter.instance_stats_from_mask(mask)

        for iid in sorted(int(x) for x in instance_stats.keys()):
            if iid <= 0:
                continue

            stats = instance_stats.get(int(iid), None)
            if not isinstance(stats, dict):
                continue
            bbox_raw = stats.get("bbox", None)
            geom_raw = stats.get("geom", None)
            if bbox_raw is None or len(bbox_raw) < 4 or not isinstance(geom_raw, dict):
                continue

            x1 = int(bbox_raw[0])
            y1 = int(bbox_raw[1])
            x2 = int(bbox_raw[2])
            y2 = int(bbox_raw[3])
            if x2 <= x1 or y2 <= y1:
                continue

            area = int(geom_raw.get("area", 0.0) or 0.0)
            if area <= 0:
                continue

            label = str(self.segmenter.instance_id_to_label.get(iid, f"instance_{iid}"))
            class_name = self.segmenter.extract_class_name(label)
            if class_name is not None and str(class_name).strip().lower() in self.ignored_class_names:
                continue
            obj_mask = (mask[y1:y2, x1:x2] == int(iid))

            out[iid] = GroundTruthObject(
                instance_id=iid,
                label=label,
                class_name=class_name,
                mask=obj_mask.astype(bool, copy=False),
                area=area,
                bbox_xyxy=(x1, y1, x2, y2),
            )

        return out
