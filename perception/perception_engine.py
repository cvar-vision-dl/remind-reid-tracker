# perception/perception_engine.py

from __future__ import annotations

import cv2
import numpy as np

from utils.image import align_frame_to_patches_crop, resize_keep_aspect_by_width
from utils.math import l2_normalize_rows
from utils.time import ExecutionTimer
from utils.config import bg_partials_enabled

from features.background_features import BackgroundFeatureExtractor
from features.object_features import ObjectFeatureExtractor
from features.part_features import PartFeatureExtractor


class FramePerceptionContext:
    def __init__(self, frame_id: int, timestamp: float):
        self.frame_id = int(frame_id)
        self.timestamp = float(timestamp)


class FramePerceptionOutput:
    def __init__(self, frame_id: int, timestamp: float):
        self.frame_id = int(frame_id)
        self.timestamp = float(timestamp)

        self.detections = []
        self.frame_features = {}
        self.det_features_by_id = {}

        self.transforms = {}
        self.debug = {}
        self.timings_seconds = {}

        self.summary = {
            "n_detections": 0,
            "has_fmap": False,
            "has_attn_mean": False,
            "has_attn_heads": False,
            "parts_enabled": False,
            "bg_local_enabled": False,
        }


class PerceptionEngine:
    """
    FULL Perception engine.

    Pipeline:
      - resize
      - align to patch multiple
      - YOLO on aligned
      - DINO once on aligned (fmap, y attn solo si es necesaria)
      - per-detection packs: obj / bg / parts
    """

    def __init__(self, config: dict, yolo, dino):
        self.config = config
        self.yolo = yolo
        self.dino = dino
        (
            self.ignored_detection_class_ids,
            self.ignored_detection_class_names,
        ) = self.resolve_ignored_detection_classes()

        obj_cfg = self.config.get("object_features", {}) or {}
        self.object_features = ObjectFeatureExtractor(obj_cfg)

        parts_cfg = self.config.get("part_descriptors", {}) or {}
        self.parts_enabled = bool(parts_cfg.get("enabled", False))
        self.part_features = PartFeatureExtractor(parts_cfg) if self.parts_enabled else None

        bg_local_cfg = self.config.get("bg_local", {}) or {}
        self.bg_partials_enabled = bool(bg_partials_enabled(self.config))
        if not self.bg_partials_enabled:
            bg_local_cfg = dict(bg_local_cfg)
            proto_cfg = dict(bg_local_cfg.get("prototypes", {}) or {})
            proto_cfg["enabled"] = False
            bg_local_cfg["prototypes"] = proto_cfg
        self.bg_features = BackgroundFeatureExtractor(bg_local_cfg)
        self.bg_local_enabled = bool(bg_local_cfg.get("enabled", False))

    def resolve_ignored_detection_classes(self) -> tuple[set[int], set[str]]:
        det_cfg = self.config.get("detector", {}) or {}
        classes_spec = det_cfg.get("ignored_classes", None)
        return (
            self.normalize_detection_class_ids(classes_spec),
            self.normalize_detection_class_names(classes_spec),
        )

    def normalize_detection_class_ids(self, classes_spec) -> set[int]:
        if classes_spec is None:
            return set()
        if not isinstance(classes_spec, list):
            return set()

        class_id_to_name = getattr(self.yolo, "class_id_to_name", None)
        if not isinstance(class_id_to_name, dict):
            class_id_to_name = {}
        name_to_id = {
            str(name).strip().lower(): int(class_id)
            for class_id, name in class_id_to_name.items()
            if name is not None
        }

        out: set[int] = set()
        for item in classes_spec:
            if not isinstance(item, str):
                continue
            key = str(item).strip().lower()
            if not key:
                continue
            class_id = name_to_id.get(key, None)
            if class_id is not None:
                out.add(int(class_id))

        return out

    def normalize_detection_class_names(self, classes_spec) -> set[str]:
        if classes_spec is None:
            return set()
        if not isinstance(classes_spec, list):
            return set()

        out: set[str] = set()
        for item in classes_spec:
            if not isinstance(item, str):
                continue
            key = str(item).strip().lower()
            if key:
                out.add(str(key))
        return out

    def filter_ignored_detections(self, detections: list) -> tuple[list, int]:
        ignored_class_ids = set(int(x) for x in (self.ignored_detection_class_ids or set()))
        ignored_class_names = {
            str(x).strip().lower()
            for x in (self.ignored_detection_class_names or set())
            if str(x).strip()
        }
        if not ignored_class_ids and not ignored_class_names:
            return list(detections or []), 0

        kept = []
        ignored = 0
        for det in (detections or []):
            class_id = getattr(det, "class_id", None)
            class_name = getattr(det, "class_name", None)
            original_class_name = getattr(det, "original_class_name", None)

            ignored_by_id = bool(class_id is not None and int(class_id) in ignored_class_ids)
            ignored_by_name = bool(
                (
                    class_name is not None
                    and str(class_name).strip().lower() in ignored_class_names
                )
                or (
                    original_class_name is not None
                    and str(original_class_name).strip().lower() in ignored_class_names
                )
            )
            if ignored_by_id or ignored_by_name:
                ignored += 1
                continue
            kept.append(det)
        return kept, int(ignored)

    def process_frame(self, frame, frame_context: FramePerceptionContext) -> FramePerceptionOutput:
        out = FramePerceptionOutput(frame_id=frame_context.frame_id, timestamp=frame_context.timestamp)
        timer = ExecutionTimer()

        frame_proc, scale_in = resize_keep_aspect_by_width(frame, self.config)

        patch = int(getattr(self.dino, "patch_size", 16))
        center = bool((self.config.get("perception", {}) or {}).get("full_center_crop", True))

        frame_aligned, meta = align_frame_to_patches_crop(
            frame_proc,
            patch_multiple=patch,
            center=center,
        )
        hA, wA = frame_aligned.shape[:2]

        out.transforms = {
            "scale_in": scale_in,
            "align_meta": meta,
            "aligned_shape": (hA, wA),
            "patch_size": patch,
            "center_crop": center,
        }
        out.debug = {"frame_aligned_bgr": frame_aligned}

        with timer.measure("detector"):
            detections = timer.run(
                "detector/segment",
                self.yolo.segment,
                frame=frame_aligned,
                frame_id=frame_context.frame_id,
                timestamp=frame_context.timestamp,
            )
            timer.extend(
                getattr(self.yolo, "last_timings_seconds", {}),
                prefix="detector/segment/",
            )
            filtered_detections, ignored_count = timer.run(
                "detector/ignored_filter",
                self.filter_ignored_detections,
                detections or [],
            )
        out.detections = filtered_detections
        out.summary["n_detections_raw"] = int(len(detections or []))
        out.summary["n_detections_ignored"] = int(ignored_count)
        out.summary["n_detections"] = int(len(out.detections))

        frame_rgb = cv2.cvtColor(frame_aligned, cv2.COLOR_BGR2RGB)

        attn_required = False
        head_ids = None
        if self.parts_enabled and self.part_features is not None:
            attn_required = bool(getattr(self.part_features, "enable_attention", False))
            if attn_required:
                head_ids = getattr(self.part_features, "attn_head_ids", None)

        if attn_required:
            if not hasattr(self.dino, "extract_patches_and_attn"):
                raise RuntimeError("Parts attention enabled pero dino.extract_patches_and_attn no existe.")
            fmap_full, attn_mean, attn_heads = timer.run(
                "dino",
                self.dino.extract_patches_and_attn,
                frame_rgb,
                head_ids=head_ids,
            )
        else:
            if not hasattr(self.dino, "extract_patches"):
                raise RuntimeError("PerceptionEngine espera dino.extract_patches(frame_rgb).")
            fmap_full = timer.run(
                "dino",
                self.dino.extract_patches,
                frame_rgb,
            )
            attn_mean, attn_heads = None, None

        attn_for_parts = attn_heads if attn_heads is not None else attn_mean

        if attn_required:
            if head_ids is not None and attn_heads is None:
                raise RuntimeError("Parts attention pide head_ids pero attn_heads=None.")
            if attn_for_parts is None:
                raise RuntimeError("Parts attention enabled pero no hay atención (attn_for_parts=None).")

        out.frame_features = {
            "space": "full",
            "full_space": meta,
            "frame_shape": (hA, wA),
            "fmap_full": fmap_full,
            "attn_mean": attn_mean,
            "attn_heads": attn_heads,
        }

        out.summary["has_fmap"] = fmap_full is not None
        out.summary["has_attn_mean"] = attn_mean is not None
        out.summary["has_attn_heads"] = attn_heads is not None
        out.summary["parts_enabled"] = bool(self.parts_enabled)
        out.summary["bg_local_enabled"] = bool(self.bg_local_enabled)

        frame_feature_cache = None
        if fmap_full is not None and fmap_full.ndim == 3:
            _hp, _wp, dim = fmap_full.shape
            flat_feats = fmap_full.reshape(-1, dim).astype(np.float32, copy=False)
            frame_feature_cache = {"flat_feats": flat_feats}

            bg_proto_cfg = (self.bg_features.config.get("prototypes", {}) or {}) if self.bg_local_enabled else {}
            need_flat_feats_n = (
                bool(self.parts_enabled)
                or bool(self.object_features.enable_patch_descs and self.object_features.patch_l2_normalize)
                or bool(self.bg_partials_enabled and bg_proto_cfg.get("enabled", False))
            )
            if need_flat_feats_n:
                frame_feature_cache["flat_feats_n"] = l2_normalize_rows(flat_feats)

        det_features_by_id = {}

        for det in out.detections:
            det_id = getattr(det, "detection_id", None)
            if det_id is None:
                continue
            det_id = int(det_id)

            obj_mask = getattr(det, "mask", None)
            if obj_mask is not None:
                obj_mask = obj_mask.astype(bool, copy=False)
                if obj_mask.shape[:2] != (hA, wA):
                    raise ValueError(f"Mask shape {obj_mask.shape} != aligned frame {(hA, wA)}")

            geom = getattr(det, "geom", None)
            if not isinstance(geom, dict):
                raise RuntimeError(f"Detection {det_id} no tiene geom (dict) en det.geom")

            c = geom.get("center", None)
            a = geom.get("area", None)
            if c is None or a is None or not isinstance(c, (tuple, list)) or len(c) != 2:
                raise RuntimeError(f"Detection {det_id} geom inválida: {geom}")

            obj_patch_cache = None
            if obj_mask is not None and fmap_full is not None and fmap_full.ndim == 3:
                hp_f, wp_f, _ = fmap_full.shape
                cov = self.dino.mask_px_to_patch_coverage(
                    obj_mask.astype(np.uint8, copy=False),
                    hp_f,
                    wp_f,
                )
                obj_patch_cache = {
                    "cov": cov,
                    "patch_mask": self.dino.patch_mask_from_coverage(cov),
                }

            obj_out = None
            if obj_mask is not None:
                obj_out = timer.run(
                    "obj_features",
                    self.object_features.extract,
                    dino=self.dino,
                    fmap=fmap_full,
                    obj_mask_px=obj_mask,
                    patch_cache=obj_patch_cache,
                    frame_cache=frame_feature_cache,
                )

            desc_global = obj_out.get("desc_global", None) if isinstance(obj_out, dict) else None
            desc_global_trimmed = obj_out.get("desc_global_trimmed", None) if isinstance(obj_out, dict) else None
            patch_descs = obj_out.get("patch_descs", None) if isinstance(obj_out, dict) else None
            effective_obj_patches = obj_out.get("effective_patches", None) if isinstance(obj_out, dict) else None

            n_obj_patches = None
            if patch_descs is not None:
                n_obj_patches = int(len(patch_descs))

            obj_pack = {
                "global": {"desc": desc_global},
                "global_trimmed": {"desc": desc_global_trimmed},
            }

            bg_pack = None
            if self.bg_local_enabled and obj_mask is not None:
                bg_pack = timer.run(
                    "bg_features",
                    self.bg_features.extract,
                    dino=self.dino,
                    fmap=fmap_full,
                    obj_mask_px=obj_mask,
                    patch_cache=obj_patch_cache,
                    frame_cache=frame_feature_cache,
                )
                timer.extend(getattr(self.bg_features, "last_timings_seconds", {}), prefix="bg_features/")

            parts_out = None
            if self.parts_enabled and self.part_features is not None and obj_mask is not None:
                parts_out = timer.run(
                    "parts_features",
                    self.part_features.extract,
                    dino=self.dino,
                    fmap=fmap_full,
                    obj_mask_px=obj_mask,
                    attn=attn_for_parts if attn_required else None,
                    obj_patch_descs=patch_descs,
                    patch_cache=obj_patch_cache,
                    frame_cache=frame_feature_cache,
                )
                timer.extend(getattr(self.part_features, "last_timings_seconds", {}), prefix="parts_features/")

            n_parts_valid = 0
            parts_support_max = 0.0
            if isinstance(parts_out, dict):
                for pack in parts_out.values():
                    if not isinstance(pack, dict):
                        continue
                    part_descs = pack.get("part_descs", None) or []
                    part_stats = pack.get("part_stats", None) or []
                    n_parts_valid = max(int(n_parts_valid), int(len(part_descs)))
                    support_sum = 0.0
                    for stat in part_stats:
                        if isinstance(stat, dict):
                            support_sum += float(stat.get("support", 0.0) or 0.0)
                    parts_support_max = max(float(parts_support_max), float(support_sum))

            bg_quality = (bg_pack.get("quality", {}) or {}) if isinstance(bg_pack, dict) else {}
            n_bg_inner_patches = int(bg_quality.get("inner_patch_count", 0) or 0)
            n_bg_outer_patches = int(bg_quality.get("outer_patch_count", 0) or 0)
            bg_mask_quality = float(bg_quality.get("mask_quality", 1.0) or 1.0)

            det_features_by_id[det_id] = {
                "mask": obj_mask,
                "meta": {
                    "n_obj_patches": n_obj_patches,
                    "effective_obj_patches": None if effective_obj_patches is None else float(effective_obj_patches),
                    "n_parts_valid": int(n_parts_valid),
                    "parts_support": float(parts_support_max),
                    "n_bg_inner_patches": int(n_bg_inner_patches),
                    "n_bg_outer_patches": int(n_bg_outer_patches),
                    "bg_mask_quality": float(bg_mask_quality),
                },
                "obj": obj_pack,
                "bg": bg_pack,
                "parts": parts_out,
            }

        out.det_features_by_id = det_features_by_id
        out.timings_seconds = timer.snapshot_seconds()
        return out
