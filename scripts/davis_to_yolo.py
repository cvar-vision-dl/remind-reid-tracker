#!/usr/bin/env python3
"""
Convert a DAVIS-style video segmentation dataset to YOLO segmentation format.

Reads the metadata JSON (e.g. metaCUSTOMVIDEO.json) to extract the
id_to_label mapping.  Instance suffixes are stripped so that e.g.
  potted_plant_1 (pixel 1)  and  potted_plant_2 (pixel 8)
both map to YOLO class "potted_plant".

Expected input:
    <davis_root>/
        Annotations/raw/FRAMES/*.png   (index-encoded masks)
        metaCUSTOMVIDEO.json

    <images_dir>/*.jpg|png              (frames, filenames match masks)

Output:
    <output_dir>/
        images/train/  images/val/
        labels/train/  labels/val/
        dataset.yaml
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml


# ── Class-map helpers ─────────────────────────────────────────────────

def strip_instance_id(label: str) -> str:
    """Remove trailing _<digits> instance suffix from a label.

    'potted_plant_1' → 'potted_plant'
    'bowl_2'         → 'bowl'
    """
    parts = label.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return label


def build_class_map_from_json(meta_path: Path):
    """Parse the metadata JSON and return (pixel_id → class_idx, [class_names]).

    Instances sharing the same base class name get the same YOLO class index.
    """
    with open(meta_path) as f:
        data = json.load(f)

    id_to_label = data.get("id_to_label", {})
    if not id_to_label:
        sys.exit("ERROR: 'id_to_label' not found or empty in JSON.")

    # Collect unique base class names (preserve first-seen order)
    seen = {}
    for pixel_id in sorted(id_to_label, key=lambda k: int(k)):
        base = strip_instance_id(id_to_label[pixel_id])
        if base not in seen:
            seen[base] = len(seen)

    class_names = list(seen.keys())

    # Map each pixel value → YOLO class index
    pixel_to_class = {}
    for pixel_id_str, label in id_to_label.items():
        base = strip_instance_id(label)
        pixel_to_class[int(pixel_id_str)] = seen[base]

    return pixel_to_class, class_names


# ── Contour / polygon helpers ─────────────────────────────────────────

def extract_polygons(mask: np.ndarray, obj_id: int, epsilon_frac: float,
                     min_area: int):
    """Extract simplified polygon contours for a given pixel value."""
    binary = (mask == obj_id).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon_frac * peri, True)
        if len(approx) < 3:
            continue
        polygons.append(approx.squeeze())
    return polygons


def mask_to_yolo_lines(mask: np.ndarray, pixel_to_class: dict,
                       h: int, w: int, epsilon_frac: float,
                       min_area: int) -> list[str]:
    """Convert an index-encoded mask to YOLO segmentation label lines."""
    obj_ids = sorted(set(mask.flat) - {0})
    lines = []
    for obj_id in obj_ids:
        class_id = pixel_to_class.get(obj_id)
        if class_id is None:
            continue
        for poly in extract_polygons(mask, obj_id, epsilon_frac, min_area):
            coords = []
            for x, y in poly:
                coords.append(f"{x / w:.6f}")
                coords.append(f"{y / h:.6f}")
            lines.append(f"{class_id} " + " ".join(coords))
    return lines


# ── Discovery ─────────────────────────────────────────────────────────

def find_image(images_dir: Path, stem: str,
               exts=(".jpg", ".jpeg", ".png", ".bmp")):
    for ext in exts:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def discover_samples(masks_dir: Path, images_dir: Path):
    mask_files = sorted(masks_dir.glob("*.png"))
    if not mask_files:
        sys.exit(f"ERROR: No .png masks found in {masks_dir}")

    samples, missing = [], 0
    for mask_path in mask_files:
        img_path = find_image(images_dir, mask_path.stem)
        if img_path is None:
            missing += 1
            if missing <= 5:
                print(f"WARN: No image for mask '{mask_path.name}', skipping.")
            continue
        samples.append((img_path, mask_path))

    if missing > 5:
        print(f"WARN: ... and {missing - 5} more masks without matching images.")
    return samples


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert DAVIS-style masks + images to YOLO segmentation format.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "davis_root",
        type=Path,
        help="DAVIS root dir (contains Annotations/ and the metadata JSON).",
    )
    parser.add_argument(
        "images_dir",
        type=Path,
        help="Directory with the source image frames.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("yolo_dataset"),
        help="Output directory for the YOLO dataset.",
    )
    parser.add_argument(
        "-j", "--json",
        type=str,
        default=None,
        help="Metadata JSON filename inside davis_root "
             "(auto-detected if a single .json exists).",
    )
    parser.add_argument(
        "--masks-subdir",
        type=str,
        default="Annotations/raw/FRAMES",
        help="Relative path from davis_root to the mask PNGs.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of samples for validation (0-1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/val split.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="Polygon simplification factor (fraction of contour perimeter).",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=50,
        help="Minimum contour area in pixels to keep.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink images instead of copying (saves disk space).",
    )
    args = parser.parse_args()

    davis_root = args.davis_root
    if not davis_root.is_dir():
        sys.exit(f"ERROR: DAVIS root not found: {davis_root}")

    # ── Locate metadata JSON ──────────────────────────────────────────
    if args.json:
        meta_path = davis_root / args.json
    else:
        jsons = list(davis_root.glob("*.json"))
        if len(jsons) == 1:
            meta_path = jsons[0]
        elif len(jsons) == 0:
            sys.exit("ERROR: No .json found in davis_root. Use --json to specify.")
        else:
            sys.exit(f"ERROR: Multiple .json files found: {[j.name for j in jsons]}. "
                     f"Use --json to specify which one.")

    print(f"Metadata  : {meta_path}")
    pixel_to_class, class_names = build_class_map_from_json(meta_path)
    print(f"Classes ({len(class_names)}): {class_names}")

    # ── Locate masks ──────────────────────────────────────────────────
    masks_dir = davis_root / args.masks_subdir
    if not masks_dir.is_dir():
        sys.exit(f"ERROR: Masks dir not found: {masks_dir}")

    if not args.images_dir.is_dir():
        sys.exit(f"ERROR: Images dir not found: {args.images_dir}")

    # ── Discover samples ──────────────────────────────────────────────
    print(f"Masks dir : {masks_dir}")
    print(f"Images dir: {args.images_dir}")
    samples = discover_samples(masks_dir, args.images_dir)
    if not samples:
        sys.exit("ERROR: No valid image/mask pairs found.")
    print(f"Found {len(samples)} image/mask pairs.")

    # ── Train / Val split ─────────────────────────────────────────────
    random.seed(args.seed)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    val_count = max(1, int(len(samples) * args.val_ratio))
    val_set = set(indices[:val_count])

    print(f"Split: {len(samples) - val_count} train / {val_count} val  "
          f"({100 * (1 - args.val_ratio):.0f}% / {100 * args.val_ratio:.0f}%)")

    # ── Create output dirs ────────────────────────────────────────────
    out = args.output
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # ── Convert ───────────────────────────────────────────────────────
    converted, skipped = 0, 0
    for i, (img_path, mask_path) in enumerate(samples):
        split = "val" if i in val_set else "train"
        fname = mask_path.stem

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"WARN: Could not read mask {mask_path}, skipping.")
            skipped += 1
            continue
        h, w = mask.shape[:2]

        lines = mask_to_yolo_lines(mask, pixel_to_class, h, w,
                                   args.epsilon, args.min_area)
        if not lines:
            skipped += 1
            continue

        # Write label
        (out / "labels" / split / f"{fname}.txt").write_text(
            "\n".join(lines) + "\n"
        )

        # Copy / symlink image
        img_out = out / "images" / split / f"{fname}{img_path.suffix}"
        if args.symlink:
            if not img_out.exists():
                img_out.symlink_to(img_path.resolve())
        else:
            shutil.copy2(img_path, img_out)

        converted += 1

    # ── Write dataset.yaml ────────────────────────────────────────────
    yaml_data = {
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(class_names),
        "names": class_names,
    }
    yaml_path = out / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

    print(f"\nDone! Output: {out.resolve()}")
    print(f"  Converted: {converted}  |  Skipped (empty/unreadable): {skipped}")
    print(f"  dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    main()