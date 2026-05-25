#!/usr/bin/env python3
"""Build a YOLO segmentation dataset from ScanNet++ benchmark masks.

The script consumes the prepared ScanNet++ annotation layout produced by the
project exporters:

    scannetpp_data/
        data/<scene_id>/dslr/resized_images/
        annotations/<scene_id>/meta_benchmark_instance.json
        annotations/<scene_id>/annotations/benchmark_instance/*.png

Benchmark instances whose base class is "object" are ignored by default.

Scene data directories may also be stored as tar archives (.tar, .tar.gz,
.tar.bz2, .tar.xz, .tgz) alongside or instead of extracted folders.  When a
scene directory is missing but a matching archive exists, the archive is
transparently extracted to a temporary directory, processed, and cleaned up.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import tarfile
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_LABELS_CSV = PROJECT_ROOT / "APP2" / "Src" / "testing" / "scene_labels.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "APP2" / "yolo_datasets" / "scannetpp_benchmark_no_object"
DEFAULT_IMAGE_SUBDIR = "dslr/resized_images"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG")
LABEL_SUFFIX_RE = re.compile(r"^(?P<class_name>.+)_(?P<instance_id>\d+)$")
NATURAL_RE = re.compile(r"(\d+)")
TAR_EXTENSIONS = (".tar", ".tar.gz", ".tar.bz2", ".tar.xz", ".tgz")


@dataclass(frozen=True)
class SceneSource:
    scene_id: str
    frames_dir: Path
    annotations_dir: Path
    meta_path: Path
    mode: str
    meta: dict


@dataclass(frozen=True)
class Sample:
    scene_id: str
    image_path: Path
    mask_path: Path
    output_stem: str
    split: str


@dataclass(frozen=True)
class SceneJob:
    source: SceneSource
    pixel_to_class: dict[int, int]
    scene_classes: list[str]
    samples: list[Sample]
    discovery_stats: dict[str, int]


def natural_key(path: Path | str) -> tuple:
    text = str(path.name if isinstance(path, Path) else path)
    return tuple(int(part) if part.isdigit() else part.lower() for part in NATURAL_RE.split(text))


def strip_instance_id(label: str | None) -> str:
    text = str(label or "").strip()
    match = LABEL_SUFFIX_RE.match(text)
    return match.group("class_name") if match else text


def normalize_mask_variant(mask_variant: str) -> str:
    variant = str(mask_variant or "").strip().lower() or "benchmark_instance"
    return "benchmark_instance" if variant in {"bench", "benchmark"} else variant


def sanitize_for_filename(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = re.sub(r"_+", "_", text).strip("._-")
    return text or "sample"


def flatten_class_list(values: Iterable[str] | None) -> set[str]:
    out: set[str] = set()
    for value in values or []:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.add(item)
    return out


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def should_print_progress(done: int, total: int, every: int) -> bool:
    if done <= 1 or done >= total:
        return True
    return every > 0 and done % every == 0


def default_dataset_root() -> Path | None:
    candidates = [
        os.environ.get("APP2_SCANNETPP_DATASET_ROOT", ""),
        "/media/pablo/LINUX/Qsync/2026_tracker_reid/datasets/scannetpp_data",
        "/media/pablo/LINUX/scannetpp_data",
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.exists():
            return path.resolve()
    return None


def resolve_data_root(dataset_root: Path | None, data_root: Path | None) -> Path | None:
    if data_root is not None:
        return data_root.expanduser().resolve()
    if dataset_root is None:
        return None
    candidate = dataset_root / "data"
    return candidate.resolve() if candidate.exists() else dataset_root.resolve()


def resolve_annotations_root(dataset_root: Path | None, annotations_root: Path | None) -> Path | None:
    if annotations_root is not None:
        return annotations_root.expanduser().resolve()
    if dataset_root is None:
        return None
    candidate = dataset_root / "annotations"
    return candidate.resolve() if candidate.exists() else dataset_root.resolve()


def read_scene_ids_from_csv(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = csv.DictReader(f)
        scene_ids = [str(row.get("scene_id", "")).strip() for row in rows]
    return [sid for sid in scene_ids if sid]


def discover_scene_ids(data_root: Path | None, annotations_root: Path | None) -> list[str]:
    roots = [root for root in (annotations_root, data_root) if root is not None and root.is_dir()]
    scene_ids: set[str] = set()
    for root in roots:
        for child in root.iterdir():
            if child.is_dir() and re.fullmatch(r"[0-9a-fA-F]{10}", child.name):
                scene_ids.add(child.name)
            # Also discover scenes available as tar archives
            elif child.is_file():
                name = child.name
                for ext in TAR_EXTENSIONS:
                    if name.endswith(ext):
                        stem = name[: -len(ext)]
                        if re.fullmatch(r"[0-9a-fA-F]{10}", stem):
                            scene_ids.add(stem)
                        break
    return sorted(scene_ids)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


# ---------------------------------------------------------------------------
# Tar archive support
# ---------------------------------------------------------------------------

def find_scene_tar(root: Path, scene_id: str) -> Path | None:
    """Find a tar archive for *scene_id* under *root*.

    Looks for ``<root>/<scene_id>.tar``, ``.tar.gz``, etc. as well as
    inside ``<root>/data/`` and ``<root>/annotations/``.
    """
    if root is None:
        return None
    search_dirs = [root]
    for subdir in ("data", "annotations", "2Dmasks"):
        candidate = root / subdir
        if candidate.is_dir():
            search_dirs.append(candidate)

    for search_dir in search_dirs:
        for ext in TAR_EXTENSIONS:
            tar_path = search_dir / f"{scene_id}{ext}"
            if tar_path.is_file():
                return tar_path
    return None


@contextmanager
def maybe_extract_tar(
    scene_id: str,
    data_root: Path | None,
    annotations_root: Path | None,
    tmp_base: Path | None = None,
) -> Generator[tuple[Path | None, Path | None], None, None]:
    """Context manager that extracts tar archives for a scene when needed.

    If the scene directories already exist on disk the original roots are
    returned unchanged.  When a matching ``.tar*`` file is found (in
    *data_root* or *annotations_root*), it is extracted into a temporary
    directory that is cleaned up on exit.

    The caller receives ``(effective_data_root, effective_annotations_root)``
    which can be used in place of the originals.
    """
    tmpdir: tempfile.TemporaryDirectory | None = None
    eff_data_root = data_root
    eff_ann_root = annotations_root

    # Check whether we need to extract anything
    need_data_tar = False
    need_ann_tar = False
    data_tar: Path | None = None
    ann_tar: Path | None = None

    # Data root: check if scene dir already exists
    if data_root is not None:
        scene_exists = any(
            (data_root / prefix / scene_id).is_dir()
            for prefix in ("", "data")
            if (data_root / prefix).is_dir() or prefix == ""
        ) or (data_root / scene_id).is_dir()
        if not scene_exists:
            data_tar = find_scene_tar(data_root, scene_id)
            if data_tar is not None:
                need_data_tar = True

    # Annotations root: check if scene dir already exists
    if annotations_root is not None and annotations_root != data_root:
        scene_exists = any(
            (annotations_root / prefix / scene_id).is_dir()
            for prefix in ("", "annotations", "2Dmasks")
            if (annotations_root / prefix).is_dir() or prefix == ""
        ) or (annotations_root / scene_id).is_dir()
        if not scene_exists:
            ann_tar = find_scene_tar(annotations_root, scene_id)
            if ann_tar is not None:
                need_ann_tar = True
    elif annotations_root is not None and annotations_root == data_root:
        # Same root – if we already found a data tar it covers annotations too
        if need_data_tar:
            need_ann_tar = False
            ann_tar = None

    if not need_data_tar and not need_ann_tar:
        # Nothing to extract – pass through the originals
        yield eff_data_root, eff_ann_root
        return

    # Create a single temporary directory for all extractions
    tmpdir = tempfile.TemporaryDirectory(prefix=f"scannetpp_{scene_id}_", dir=tmp_base)
    tmp_path = Path(tmpdir.name)
    try:
        if need_data_tar and data_tar is not None:
            print(f"  [TAR] Extracting {data_tar.name} → {tmp_path}")
            with tarfile.open(data_tar, "r:*") as tf:
                tf.extractall(tmp_path, filter="data")
            # The archive may contain the scene_id folder directly, or the
            # files at the root.  Normalise so that <tmp>/<scene_id>/... exists.
            _normalise_extracted(tmp_path, scene_id)
            eff_data_root = tmp_path

        if need_ann_tar and ann_tar is not None and ann_tar != data_tar:
            print(f"  [TAR] Extracting {ann_tar.name} → {tmp_path}")
            with tarfile.open(ann_tar, "r:*") as tf:
                tf.extractall(tmp_path, filter="data")
            _normalise_extracted(tmp_path, scene_id)
            eff_ann_root = tmp_path
        elif need_ann_tar and ann_tar == data_tar:
            # Already extracted above
            eff_ann_root = tmp_path

        # If data and annotations share the same root, keep them consistent
        if data_root == annotations_root:
            if need_data_tar:
                eff_ann_root = eff_data_root

        yield eff_data_root, eff_ann_root
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()


def _normalise_extracted(tmp_path: Path, scene_id: str) -> None:
    """Ensure ``<tmp_path>/<scene_id>/`` exists after extraction.

    Some tars store files as ``<scene_id>/...`` (already fine).  Others store
    them flat or under an unexpected top-level folder.  This helper renames
    when needed.
    """
    target = tmp_path / scene_id
    if target.is_dir():
        return  # already correct

    # Check if there's a single top-level directory that can be renamed
    children = [c for c in tmp_path.iterdir() if c.name != scene_id]
    if len(children) == 1 and children[0].is_dir():
        children[0].rename(target)
        return

    # Files are loose at the root – wrap them in a scene_id folder
    target.mkdir(exist_ok=True)
    for child in tmp_path.iterdir():
        if child == target:
            continue
        child.rename(target / child.name)


# ---------------------------------------------------------------------------


def resolve_disk_scene(
    *,
    scene_id: str,
    data_root: Path | None,
    annotations_root: Path | None,
    image_subdir: str,
    mask_variant: str,
) -> SceneSource | None:
    if data_root is None or annotations_root is None:
        return None

    variant = normalize_mask_variant(mask_variant)
    frames_candidates = [
        data_root / scene_id / image_subdir,
        data_root / "data" / scene_id / image_subdir,
    ]
    ann_scene_candidates = [
        annotations_root / scene_id,
        annotations_root / "annotations" / scene_id,
        annotations_root / "2Dmasks" / scene_id,
        data_root / scene_id,
        data_root / "data" / scene_id,
    ]

    for frames_dir in frames_candidates:
        for ann_scene in ann_scene_candidates:
            meta_path = ann_scene / f"meta_{variant}.json"
            annotations_dir = ann_scene / "annotations" / variant
            if frames_dir.is_dir() and meta_path.is_file() and annotations_dir.is_dir():
                return SceneSource(
                    scene_id=scene_id,
                    frames_dir=frames_dir.resolve(),
                    annotations_dir=annotations_dir.resolve(),
                    meta_path=meta_path.resolve(),
                    mode="scannetpp_data",
                    meta=load_json(meta_path),
                )
    return None


def resolve_scene_source(
    *,
    scene_id: str,
    data_root: Path | None,
    annotations_root: Path | None,
    image_subdir: str,
    mask_variant: str,
) -> SceneSource | None:
    return resolve_disk_scene(
        scene_id=scene_id,
        data_root=data_root,
        annotations_root=annotations_root,
        image_subdir=image_subdir,
        mask_variant=mask_variant,
    )


def sorted_id_to_label(meta: dict) -> list[tuple[int, str]]:
    raw = meta.get("id_to_label", {}) or {}
    out: list[tuple[int, str]] = []
    for key, label in raw.items():
        try:
            instance_id = int(key)
        except Exception:
            continue
        if instance_id <= 0:
            continue
        out.append((instance_id, str(label)))
    return sorted(out, key=lambda item: item[0])


def register_scene_classes(
    source: SceneSource,
    class_to_idx: dict[str, int],
    exclude_classes: set[str],
) -> tuple[dict[int, int], list[str]]:
    pixel_to_class: dict[int, int] = {}
    scene_classes: list[str] = []
    seen_scene_classes: set[str] = set()

    for instance_id, label in sorted_id_to_label(source.meta):
        base = strip_instance_id(label)
        if not base or base in exclude_classes:
            continue
        if base not in class_to_idx:
            class_to_idx[base] = len(class_to_idx)
        pixel_to_class[instance_id] = class_to_idx[base]
        if base not in seen_scene_classes:
            scene_classes.append(base)
            seen_scene_classes.add(base)

    return pixel_to_class, scene_classes


def find_image(frames_dir: Path, name_or_stem: str) -> Path | None:
    raw = str(name_or_stem or "").strip()
    if not raw:
        return None

    direct = frames_dir / raw
    if direct.is_file():
        return direct

    stem = Path(raw).stem
    suffix = Path(raw).suffix
    if suffix:
        candidates = sorted(frames_dir.glob(f"{stem}.*"), key=natural_key)
    else:
        candidates = [frames_dir / f"{stem}{ext}" for ext in IMAGE_EXTS]
        candidates.extend(sorted(frames_dir.glob(f"{stem}.*"), key=natural_key))

    for candidate in candidates:
        if candidate.is_file() and candidate.suffix in IMAGE_EXTS:
            return candidate
    return None


def discover_scene_samples(
    *,
    source: SceneSource,
    max_frames_per_scene: int,
    frame_sampling: str,
    val_ratio: float,
    seed: int,
    scene_index: int,
) -> tuple[list[Sample], dict[str, int]]:
    mask_files = sorted(source.annotations_dir.glob("*.png"), key=natural_key)
    frame_names = source.meta.get("frame_names", None)
    if not isinstance(frame_names, list):
        frame_names = []

    pairs: list[tuple[Path, Path]] = []
    missing_images = 0
    for idx, mask_path in enumerate(mask_files):
        image_path = None
        if idx < len(frame_names):
            image_path = find_image(source.frames_dir, str(frame_names[idx]))
        if image_path is None:
            image_path = find_image(source.frames_dir, mask_path.stem)
        if image_path is None:
            missing_images += 1
            continue
        pairs.append((image_path, mask_path))

    if max_frames_per_scene and max_frames_per_scene > 0 and len(pairs) > max_frames_per_scene:
        if frame_sampling == "first":
            pairs = pairs[:max_frames_per_scene]
        elif frame_sampling == "random":
            rng = random.Random(seed + scene_index)
            pairs = sorted(rng.sample(pairs, max_frames_per_scene), key=lambda item: natural_key(item[1]))
        else:
            step = (len(pairs) - 1) / float(max_frames_per_scene - 1) if max_frames_per_scene > 1 else 0.0
            indices = [round(i * step) for i in range(max_frames_per_scene)]
            pairs = [pairs[int(i)] for i in indices]

    rng = random.Random(seed + 1009 * (scene_index + 1))
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    if len(indices) <= 1 or val_ratio <= 0:
        val_indices: set[int] = set()
    else:
        val_count = int(round(len(indices) * val_ratio))
        val_count = max(1, min(len(indices) - 1, val_count))
        val_indices = set(indices[:val_count])

    samples: list[Sample] = []
    safe_scene = sanitize_for_filename(source.scene_id)
    for idx, (image_path, mask_path) in enumerate(pairs):
        split = "val" if idx in val_indices else "train"
        output_stem = f"{safe_scene}__{sanitize_for_filename(mask_path.stem)}"
        samples.append(
            Sample(
                scene_id=source.scene_id,
                image_path=image_path,
                mask_path=mask_path,
                output_stem=output_stem,
                split=split,
            )
        )

    return samples, {"masks": len(mask_files), "missing_images": missing_images, "paired": len(pairs)}


def extract_polygons(binary: np.ndarray, epsilon_frac: float, min_area: int) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon_frac * perimeter, True)
        if len(approx) < 3:
            continue
        poly = approx.reshape(-1, 2)
        if poly.shape[0] >= 3:
            polygons.append(poly)
    return polygons


def mask_to_yolo_lines(
    mask: np.ndarray,
    pixel_to_class: dict[int, int],
    idx_to_class: list[str],
    epsilon_frac: float,
    min_area: int,
) -> tuple[list[str], dict[str, int]]:
    if mask.ndim == 3:
        mask = mask[..., 0]
    h, w = mask.shape[:2]
    lines: list[str] = []
    class_counts: dict[str, int] = {}

    for raw_obj_id in np.unique(mask):
        obj_id = int(raw_obj_id)
        if obj_id == 0:
            continue
        class_id = pixel_to_class.get(obj_id)
        if class_id is None:
            continue

        binary = mask == obj_id
        polygons = extract_polygons(binary, epsilon_frac, min_area)
        if not polygons:
            continue

        class_name = idx_to_class[class_id]
        for poly in polygons:
            coords: list[str] = []
            for x, y in poly:
                coords.append(f"{float(x) / float(w):.6f}")
                coords.append(f"{float(y) / float(h):.6f}")
            lines.append(f"{class_id} " + " ".join(coords))
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

    return lines, class_counts


def copy_or_symlink(src: Path, dst: Path, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def load_existing_class_names(output_dir: Path) -> list[str]:
    manifest_path = output_dir / "dataset_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
            names = manifest.get("class_names", [])
            if isinstance(names, list):
                return [str(name) for name in names]
        except Exception:
            pass

    yaml_path = output_dir / "dataset.yaml"
    if not yaml_path.is_file():
        return []

    names: list[tuple[int, str]] = []
    in_names = False
    for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names and line and not line.startswith(" "):
            break
        if in_names:
            match = re.match(r"\s*(\d+)\s*:\s*(.+?)\s*$", line)
            if match:
                value = match.group(2).strip().strip("\"'")
                names.append((int(match.group(1)), value))
    return [name for _, name in sorted(names)]


def existing_dataset_has_val(output_dir: Path) -> bool:
    yaml_path = output_dir / "dataset.yaml"
    if yaml_path.is_file():
        for raw_line in yaml_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line.startswith("val:"):
                return True
    return (output_dir / "images" / "val").exists() or (output_dir / "labels" / "val").exists()


def load_existing_sample_manifest(output_dir: Path) -> list[dict[str, str]]:
    path = output_dir / "sample_manifest.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def dedupe_sample_manifest_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_label: dict[str, dict[str, str]] = {}
    ordered_keys: list[str] = []
    for row in rows:
        key = str(row.get("yolo_label", "")).strip()
        if not key:
            key = "|".join(
                [
                    str(row.get("scene_id", "")),
                    str(row.get("split", "")),
                    str(row.get("frame_id", "")),
                    str(row.get("yolo_image", "")),
                ]
            )
        if key not in by_label:
            ordered_keys.append(key)
        by_label[key] = row
    return [by_label[key] for key in ordered_keys]


def write_dataset_yaml(output_dir: Path, class_names: list[str], include_val: bool) -> None:
    lines = [
        f"path: {json.dumps(str(output_dir.resolve()))}",
        "train: images/train",
    ]
    if include_val:
        lines.append("val: images/val")
    lines.extend([f"nc: {len(class_names)}", "names:"])
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {json.dumps(name)}")
    (output_dir / "dataset.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_output_dir(output_dir: Path, append: bool, overwrite: bool, include_val: bool) -> None:
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and not append:
        existing_files = any(output_dir.rglob("*"))
        if existing_files:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}. "
                "Use --overwrite to recreate it or --append to add scenes."
            )
    splits = ("train", "val") if include_val else ("train",)
    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def init_manifest(output_dir: Path, args: argparse.Namespace, class_names: list[str], exclude_classes: set[str]) -> dict:
    manifest_path = output_dir / "dataset_manifest.json"
    manifest: dict = {}
    if args.append and manifest_path.is_file():
        try:
            manifest = load_json(manifest_path)
        except Exception:
            manifest = {}

    if not manifest:
        manifest = {
            "source": "ScanNet++",
            "mask_variant": normalize_mask_variant(args.mask_variant),
            "excluded_classes": sorted(exclude_classes),
            "max_frames_per_scene": int(args.max_frames_per_scene),
            "image_subdir": str(args.image_subdir),
            "class_names": list(class_names),
            "scenes": {},
            "totals": {},
        }

    manifest["source"] = "ScanNet++"
    manifest["mask_variant"] = normalize_mask_variant(args.mask_variant)
    manifest["excluded_classes"] = sorted(exclude_classes)
    manifest["max_frames_per_scene"] = int(args.max_frames_per_scene)
    manifest["image_subdir"] = str(args.image_subdir)
    manifest["class_names"] = list(class_names)
    manifest.setdefault("scenes", {})
    totals = manifest.setdefault("totals", {})
    for key in (
        "resolved_scenes",
        "missing_scenes",
        "written_images",
        "skipped_empty",
        "skipped_unreadable_masks",
        "missing_images",
    ):
        totals.setdefault(key, 0)
    return manifest


def parse_args() -> argparse.Namespace:
    default_root = default_dataset_root()
    parser = argparse.ArgumentParser(
        description="Generate a YOLO segmentation dataset from ScanNet++ benchmark_instance masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, default=default_root, help="ScanNet++ root containing data/ and annotations/.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override root containing scene data dirs.")
    parser.add_argument("--annotations-root", type=Path, default=None, help="Override root containing annotation scene dirs.")
    parser.add_argument("--scene-ids", nargs="*", default=None, help="Scene ids to process. If omitted, use --scene-labels-csv or discover dirs.")
    parser.add_argument("--scene-ids-file", type=Path, default=None, help="Text file with one scene id per line.")
    parser.add_argument("--scene-labels-csv", type=Path, default=DEFAULT_SCENE_LABELS_CSV, help="CSV with a scene_id column.")
    parser.add_argument("--num-scenes", type=int, default=None, help="Limit number of scenes after discovery.")
    parser.add_argument("--image-subdir", default=DEFAULT_IMAGE_SUBDIR, help="Image directory inside each scene data dir.")
    parser.add_argument("--mask-variant", default="benchmark_instance", help="Mask variant: benchmark, bench, or benchmark_instance.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output YOLO dataset directory.")
    parser.add_argument("--max-frames-per-scene", dest="max_frames_per_scene", type=int, default=0, help="Max frames per scene (0 = all).")
    parser.add_argument("--frame-sampling", choices=["uniform", "random", "first"], default="uniform", help="How to choose frames when max_frames_per_scene is set.")
    parser.add_argument("--val-ratio", type=float, default=0.0, help="Validation fraction per scene. Use 0 to write everything to train.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epsilon", type=float, default=0.001, help="Contour simplification as a fraction of perimeter.")
    parser.add_argument("--min-area", type=int, default=50, help="Minimum contour area in pixels to keep.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N processed frames. Use 1 for every frame.")
    parser.add_argument("--exclude-class", action="append", default=["object"], help="Class to exclude. Can be repeated or comma-separated.")
    parser.add_argument("--include-empty", action="store_true", help="Keep images whose masks produce no YOLO labels.")
    parser.add_argument("--symlink", action="store_true", help="Symlink images instead of copying them.")
    parser.add_argument("--append", action="store_true", help="Append to an existing YOLO dataset and class map.")
    parser.add_argument("--overwrite", action="store_true", help="Remove the output directory before generating.")
    parser.add_argument("--tmp-dir", type=Path, default=None, help="Base directory for temporary tar extractions (default: system tmpdir).")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.overwrite and args.append:
        print("ERROR: --overwrite and --append are mutually exclusive.", file=sys.stderr)
        return 2
    if args.val_ratio < 0 or args.val_ratio >= 1:
        print("ERROR: --val-ratio must be in [0, 1).", file=sys.stderr)
        return 2
    if args.max_frames_per_scene < 0:
        print("ERROR: --max-frames-per-scene must be >= 0.", file=sys.stderr)
        return 2
    if args.min_area < 0:
        print("ERROR: --min-area must be >= 0.", file=sys.stderr)
        return 2
    if args.epsilon < 0:
        print("ERROR: --epsilon must be >= 0.", file=sys.stderr)
        return 2
    if args.progress_every < 0:
        print("ERROR: --progress-every must be >= 0.", file=sys.stderr)
        return 2

    dataset_root = args.dataset_root.expanduser().resolve() if args.dataset_root else None
    data_root = resolve_data_root(dataset_root, args.data_root)
    annotations_root = resolve_annotations_root(dataset_root, args.annotations_root)
    output_dir = args.output.expanduser().resolve()
    exclude_classes = flatten_class_list(args.exclude_class)
    epsilon = float(args.epsilon)
    tmp_base = args.tmp_dir.expanduser().resolve() if args.tmp_dir else None

    if args.scene_ids:
        scene_ids = [str(scene).strip() for scene in args.scene_ids if str(scene).strip()]
    elif args.scene_ids_file is not None:
        scene_ids = [
            line.strip()
            for line in args.scene_ids_file.expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    else:
        scene_ids = read_scene_ids_from_csv(args.scene_labels_csv.expanduser())
        if not scene_ids:
            scene_ids = discover_scene_ids(data_root, annotations_root)

    if args.num_scenes is not None:
        scene_ids = scene_ids[: args.num_scenes]
    if not scene_ids:
        print("ERROR: No scenes to process. Pass --scene-ids or check the dataset roots.", file=sys.stderr)
        return 2

    include_val = args.val_ratio > 0 or (args.append and existing_dataset_has_val(output_dir))
    prepare_output_dir(output_dir, append=args.append, overwrite=args.overwrite, include_val=include_val)

    class_names = load_existing_class_names(output_dir) if args.append else []
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    idx_to_class = list(class_names)
    manifest = init_manifest(output_dir, args, idx_to_class, exclude_classes)
    manifest["epsilon"] = float(epsilon)
    manifest["min_area"] = int(args.min_area)
    sample_manifest_rows: list[dict[str, str]] = load_existing_sample_manifest(output_dir) if args.append else []

    # ------------------------------------------------------------------
    # Phase 1: discover & register classes (needs tar extraction per scene)
    # Phase 2: process frames (needs tar extraction per scene again)
    #
    # To avoid extracting each tar twice we do both phases together,
    # one scene at a time, inside the tar context manager.
    # ------------------------------------------------------------------

    total_frames = 0
    processed_frames = 0
    start_time = time.monotonic()

    # We process scene by scene so the tar is only inflated while needed.
    for scene_index, scene_id in enumerate(scene_ids):
        with maybe_extract_tar(scene_id, data_root, annotations_root, tmp_base) as (eff_data, eff_ann):
            source = resolve_scene_source(
                scene_id=scene_id,
                data_root=eff_data,
                annotations_root=eff_ann,
                image_subdir=args.image_subdir,
                mask_variant=args.mask_variant,
            )
            if source is None:
                manifest["totals"]["missing_scenes"] += 1
                print(f"[WARN] Scene not found or not prepared: {scene_id}")
                continue

            pixel_to_class, scene_classes = register_scene_classes(source, class_to_idx, exclude_classes)
            # Rebuild idx_to_class after potential new classes
            idx_to_class = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]

            samples, discovery_stats = discover_scene_samples(
                source=source,
                max_frames_per_scene=args.max_frames_per_scene,
                frame_sampling=args.frame_sampling,
                val_ratio=args.val_ratio,
                seed=args.seed,
                scene_index=scene_index,
            )

            scene_total = len(samples)
            total_frames += scene_total
            scene_number = scene_index + 1

            scene_record = {
                "mode": source.mode,
                "frames_dir": str(source.frames_dir),
                "annotations_dir": str(source.annotations_dir),
                "meta_path": str(source.meta_path),
                "classes": scene_classes,
                "class_polygon_counts": {},
                "samples": {"train": 0, "val": 0},
                "discovery": discovery_stats,
            }
            manifest["totals"]["resolved_scenes"] += 1
            manifest["totals"]["missing_images"] += discovery_stats["missing_images"]

            for sample in samples:
                processed_frames += 1
                mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_UNCHANGED)
                if mask is None:
                    manifest["totals"]["skipped_unreadable_masks"] += 1
                    if should_print_progress(processed_frames, total_frames, args.progress_every):
                        elapsed = time.monotonic() - start_time
                        rate = processed_frames / elapsed if elapsed > 0 else 0.0
                        eta = (total_frames - processed_frames) / rate if rate > 0 else 0.0
                        print(
                            f"[PROGRESS] {processed_frames}/{total_frames} frames | "
                            f"scene {scene_number}/{len(scene_ids)} {sample.scene_id} | "
                            f"elapsed {format_duration(elapsed)} | eta {format_duration(eta)}"
                        )
                    continue

                lines, class_counts = mask_to_yolo_lines(
                    mask=mask,
                    pixel_to_class=pixel_to_class,
                    idx_to_class=idx_to_class,
                    epsilon_frac=epsilon,
                    min_area=args.min_area,
                )
                if not lines and not args.include_empty:
                    manifest["totals"]["skipped_empty"] += 1
                    if should_print_progress(processed_frames, total_frames, args.progress_every):
                        elapsed = time.monotonic() - start_time
                        rate = processed_frames / elapsed if elapsed > 0 else 0.0
                        eta = (total_frames - processed_frames) / rate if rate > 0 else 0.0
                        print(
                            f"[PROGRESS] {processed_frames}/{total_frames} frames | "
                            f"scene {scene_number}/{len(scene_ids)} {sample.scene_id} | "
                            f"elapsed {format_duration(elapsed)} | eta {format_duration(eta)}"
                        )
                    continue

                label_path = output_dir / "labels" / sample.split / f"{sample.output_stem}.txt"
                label_path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")

                # IMPORTANT: copy (not symlink) when source is from a tar,
                # because the temp directory will be deleted after this scene.
                from_tar = str(sample.image_path).startswith(str(tmp_base or tempfile.gettempdir()))
                use_symlink = args.symlink and not from_tar
                image_out = output_dir / "images" / sample.split / f"{sample.output_stem}{sample.image_path.suffix}"
                copy_or_symlink(sample.image_path, image_out, symlink=use_symlink)

                sample_manifest_rows.append(
                    {
                        "scene_id": sample.scene_id,
                        "split": sample.split,
                        "frame_id": sample.mask_path.stem,
                        "yolo_image": str(image_out.relative_to(output_dir)),
                        "yolo_label": str(label_path.relative_to(output_dir)),
                        "source_image": str(sample.image_path),
                        "source_mask": str(sample.mask_path),
                    }
                )

                scene_record["samples"][sample.split] += 1
                manifest["totals"]["written_images"] += 1
                for class_name, count in class_counts.items():
                    current = scene_record["class_polygon_counts"].get(class_name, 0)
                    scene_record["class_polygon_counts"][class_name] = current + count

                if should_print_progress(processed_frames, total_frames, args.progress_every):
                    elapsed = time.monotonic() - start_time
                    rate = processed_frames / elapsed if elapsed > 0 else 0.0
                    eta = (total_frames - processed_frames) / rate if rate > 0 else 0.0
                    print(
                        f"[PROGRESS] {processed_frames}/{total_frames} frames | "
                        f"scene {scene_number}/{len(scene_ids)} {sample.scene_id} | "
                        f"elapsed {format_duration(elapsed)} | eta {format_duration(eta)}"
                    )

            manifest["scenes"][source.scene_id] = scene_record
            print(
                f"[OK] {source.scene_id}: {scene_record['samples']['train']} train, "
                f"{scene_record['samples']['val']} val, classes={scene_classes}"
            )
        # ← tar temp dir is cleaned up here

    manifest["class_names"] = idx_to_class
    write_dataset_yaml(output_dir, idx_to_class, include_val=include_val)
    sample_manifest_rows = dedupe_sample_manifest_rows(sample_manifest_rows)
    manifest["totals"]["resolved_scenes"] = len(manifest["scenes"])
    manifest["totals"]["written_images"] = len(sample_manifest_rows)
    manifest["totals"]["missing_images"] = sum(
        int((record.get("discovery", {}) or {}).get("missing_images", 0) or 0)
        for record in manifest["scenes"].values()
        if isinstance(record, dict)
    )
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    scene_classes_path = output_dir / "scene_classes.json"
    scene_classes_payload = {
        scene_id: {
            "classes": record["classes"],
            "class_polygon_counts": record["class_polygon_counts"],
        }
        for scene_id, record in manifest["scenes"].items()
    }
    scene_classes_path.write_text(
        json.dumps(scene_classes_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    sample_manifest_path = output_dir / "sample_manifest.csv"
    with sample_manifest_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["scene_id", "split", "frame_id", "yolo_image", "yolo_label", "source_image", "source_mask"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sample_manifest_rows)

    print(f"\nDone: {output_dir}")
    print(f"Classes ({len(idx_to_class)}): {idx_to_class}")
    print(f"dataset.yaml: {output_dir / 'dataset.yaml'}")
    print(f"scene classes: {scene_classes_path}")
    print(f"sample manifest: {sample_manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())