

# REMIND — RE-Identification with Memory for INDoor Navigation

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

### Citing
```
@misc{diazpereda2026remind,
      title={REMIND: RE-Identification with Memory for INDoor Navigation}, 
      author={Pablo Diaz-Pereda and Alejandro Rodriguez-Ramos and David Perez-Saura and Pascual Campoy},
      year={2026},
      eprint={2607.09267},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.09267}
}
```

### Authors' Links
- Pablo Diaz-Pereda [[linkedin](linkedin.com/in/pablo-diaz-192114388)]
- Alejandro Rodriguez-Ramos [[alejandrorodriguezramos.me](https://alejandrorodriguezramos.me)]
- David Perez-Saura [[Google Scholar](https://scholar.google.com/citations?user=Afdcjx4AAAAJ&hl=es)]
- Pascual Campoy [[Google Scholar](https://scholar.google.com/citations?user=apPMLQ4AAAAJ&hl=en)]

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
- **YOLO** — place segmentation weights under `yolo/` and pass the model file name to `main.py`

---

## Running On Your Own Data

This mode is for trying REMIND on your own videos or frame folders. It does not require ground-truth annotations. You only need input images/video and a YOLO segmentation model.

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

Results are saved under `outputs/video_runs/`. Use `--save-output-video` to save the rendered video with masks and IDs.

---

## Evaluation With Ground Truth

The scripts under `testing/` are intended for quantitative evaluation and internal experiment reproduction. They require ground-truth annotations, such as DAVIS-style instance masks and metadata, or prepared ScanNet++-style folders/tars depending on the script.

These annotation files and datasets are not included in this repository. To run the evaluation scripts, you must provide your own data in the expected layout, including:

- image frames
- annotation masks in DAVIS-compatible format
- metadata files mapping frames, object IDs, classes, and sequence names
- dataset roots matching the arguments passed to each script

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

Use these evaluation scripts when you want metrics against GT. Use `main.py` when you only want to run REMIND on a video or frame sequence and visually inspect the tracking output.

---

### Config overrides

The pipeline reads `config/default_config.yaml` by default. Any parameter can be overridden by passing a second YAML file:

```python
Config("config/default_config.yaml", "my_override.yaml")
```

Detector backends: `"davis"` (ground-truth masks from DAVIS / ScanNet++) or `"yolo"` (YOLO instance segmentation).

---

