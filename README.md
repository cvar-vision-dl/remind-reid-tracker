

# REMIND — Re-identification with Memory and INstance Descriptors

> A multi-object re-identification tracker that maintains consistent identities across long sequences using semantic part descriptors, relational context, and adaptive memory.

---

## Overview

REMIND addresses a core challenge in visual tracking: re-identifying objects that disappear and reappear, look similar to one another, or are observed from changing viewpoints. Rather than relying on position or motion cues, REMIND builds appearance-based identity models per object using DINOv3 patch features, decomposed into:

- **Global and part-level descriptors** — K-means and attention-guided semantic parts extracted per detection
- **Relational context (neighbor sets)** — structural scene layout encoded as co-occurrence graphs of neighboring objects
- **Known-set distance disambiguation** — geometry-aware resolution of visually ambiguous groups
- **Adaptive memory** — per-object appearance, part, and background models that update over time

The association pipeline runs a per-frame sequence of visual evidence building, context activation, global Hungarian assignment, and post-assignment guards, producing explicit uncertainty signals (ambiguous, provisional) alongside confident identity decisions.

Evaluation outputs span case, object, frame, class, scene, and batch levels, with full internal telemetry for diagnostic analysis.

https://github.com/user-attachments/assets/1cbf5700-19ba-4303-a871-4fdfb1ce9a80

### Tracking behavior demonstration

The video demonstrates REMIND operating under challenging long-term tracking conditions, including:

- illumination and exposure changes
- viewpoint and pose variation
- temporary disappearances and later re-identification
- disambiguation between visually similar objects

Consistent colors and identity labels across frames indicate successful identity preservation over time, even after occlusions or reappearances.

Detections rendered in **white** correspond to internally flagged ambiguous states, where the model intentionally avoids committing to a potentially incorrect identity assignment until sufficient visual evidence becomes available.

---

## Installation

```bash
git clone https://github.com/cvar-vision-dl/remind-reid-tracker
cd remind-reid-tracker

# Create environment (Python 3.10+ recommended)
conda create -n remind python=3.10 -y
conda activate remind

# Install dependencies
pip install torch torchvision transformers scikit-learn scipy numpy opencv-python tqdm psutil ultralytics
```

Models are loaded automatically at runtime:
- **DINOv3** — fetched from HuggingFace on first use (configurable via `dino.model_label` in `config/default_config.yaml`)
- **YOLO** — place segmentation weights under `yolo_models/` and update the paths in `config/default_config.yaml` under `yolo.models` (only needed with `--detector-backend yolo`)

---

## Quick Usage

All scripts are run from the repository root:

### User video or frame scene

Place your inputs under `testData/videos/` or `testData/frames/`, one folder per scene. Put local YOLO weights under `yolo/`:

```text
testData/
  videos/
    my_scene/
      video.mp4
  frames/
    my_frame_scene/
      frame_000001.jpg
      frame_000002.jpg
yolo/
  custom-seg.pt
```

Run REMIND by scene name using a YOLO segmentation model:

```bash
python main.py my_scene custom-seg.pt \
  --input-video-fps 5 \
  --output-fps 25 \
  --save-output-video
```

The second argument is the YOLO model file name, and the file must exist under `yolo/`.

For video input, frame selection is controlled in this order:

- `--input-video-fps`: split/sample the source video into frames at this FPS; it has no effect for frame folders
- `--stride`: process one every N available/sampled frames; common to videos and frame folders
- `--max-frames`: stop after this many processed frames; common to videos and frame folders
- `--output-fps`: FPS of the rendered `tracking.mp4`; common to videos and frame folders

For example, `--input-video-fps 10 --stride 2 --max-frames 100 --output-fps 25` samples the input video at 10 FPS, processes every second sampled frame, stops after 100 processed frames, and writes the rendered output video at 25 FPS. For frame folders, `--input-video-fps` is ignored because the frames are already extracted.

For a frame scene and live preview:

```bash
python main.py my_frame_scene custom-seg.pt \
  --input-kind frames \
  --show-viewer \
  --save-output-video \
  --output-fps 30 \
  --max-frames 300
```

Scene lookup defaults to `--input-kind auto`, which prefers `testData/frames/<scene>/` when it exists and otherwise uses `testData/videos/<scene>/` or `testData/videos/<scene>.mp4`. Pass `--input-kind video` or `--input-kind frames` to force one layout.

You can still bypass the scene layout with `--source /path/to/video.mp4`.

The script writes `frames.csv`, `detections.jsonl`, and `summary.json` under `outputs/video_runs/`. With `--save-output-video`, it also writes the rendered `tracking.mp4`.

### Single sequence

```bash
python testing/run_tracking_test.py \
  --detector-backend davis \
  --frames-dir /path/to/FRAMES/ \
  --davis-meta-path /path/to/metaCUSTOMVIDEO.json \
  --davis-annotations-dir /path/to/Annotations/raw/FRAMES \
  --sequence-name FRAMES \
  --output-dir /path/to/outputs/
```

### Batch evaluation

```bash
python testing/run_tracking_batch.py \
  --images-root /path/to/scannetpp_small_test/ \
  --masks-root /path/to/scannetpp_small_test/ \
  --mask-variant raw \
  --masks-subdir annotations \
  --image-subdir dslr/resized_images \
  --detector-backend davis \
  --max-scenes 1 \
  --output-dir /path/to/outputs/
```

Outputs include `per_case.csv`, `per_object.csv`, `per_scene.csv`, `summary_global.csv`, and internal module telemetry — ready for offline analysis or direct inclusion in research tables.

### Config overrides

The pipeline reads `config/default_config.yaml` by default. Any parameter can be overridden by passing a second YAML file:

```python
Config("config/default_config.yaml", "my_override.yaml")
```

Detector backends: `"davis"` (ground-truth masks from DAVIS / ScanNet++) or `"yolo"` (YOLO instance segmentation).

---

*This project is part of ongoing research. Documentation will be expanded.*
