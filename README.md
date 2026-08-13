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

<p align="center">
  <img width="650" alt="star-history-2026812" src="https://github.com/user-attachments/assets/8f224e95-4bf7-4dbb-b4a0-0b5d6fb1bf94" />
</p>

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
- Pablo Diaz-Pereda [[linkedin](https://www.linkedin.com/in/pablo-diaz-192114388/)]
- Alejandro Rodriguez-Ramos [[alejandrorodriguezramos.me](https://alejandrorodriguezramos.me)]
- David Perez-Saura [[Google Scholar](https://scholar.google.com/citations?user=Afdcjx4AAAAJ&hl=es)]
- Pascual Campoy [[Google Scholar](https://scholar.google.com/citations?user=apPMLQ4AAAAJ&hl=en)]

---

## Installation

```bash
git clone https://github.com/cvar-vision-dl/remind-reid-tracker
cd remind-reid-tracker

# Create environment (Python 3.10+ recommended)
conda env create -f environment.yml
conda activate remind
```

Models are loaded automatically at runtime:
- **DINOv3** — fetched from HuggingFace on first use (configurable via `dino.model_label` in `config/default_config.yaml`)
- **YOLO** — place segmentation weights under `yolo/` and pass the model file name to `main.py`
- **RF-DETR** — included in `environment.yml`; select a supported variant with `--detector-backend rfdetr --rfdetr-model medium`, or provide a local checkpoint with `--rfdetr-weights /path/to/checkpoint`.

RF-DETR model and checkpoint provenance is your responsibility. Record the exact package version, model variant, checkpoint source, and applicable license for every run. This repository does not redistribute RF-DETR checkpoints. The standard `rfdetr` package and Roboflow's Apache-designated models are Apache-2.0 licensed.

---

## Running On Your Own Data

This mode is for trying REMIND on your own videos or frame folders. It does not require ground-truth annotations. You need input images/video plus either a YOLO segmentation model or the optional RF-DETR backend.

Place your inputs under `evalData/videos/` or `evalData/frames/`, one folder per scene. Put local YOLO weights under `yolo/`:

```text
evalData/
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

To run the same frame/video path with RF-DETR, create the shipped Conda environment and omit the legacy YOLO positional model argument:

```bash
python main.py my_scene \
  --detector-backend rfdetr \
  --rfdetr-model medium \
  --rfdetr-threshold 0.5 \
  --save-output-video
```

Use `--rfdetr-weights /path/to/checkpoint` only for a local checkpoint whose provenance and license you have verified. The detector configuration is exposed as `rfdetr.model_variant`, `rfdetr.pretrain_weights`, `rfdetr.threshold`, `rfdetr.classes`, and mask-erosion settings in the shipped YAML configs. The existing `python main.py SCENE YOLO_MODEL` command remains the default YOLO path.

For a bounded RF-DETR CLI smoke run on one frame, select the Nano model and limit processing explicitly:

```bash
python main.py rfdetr_cli_smoke \
  --test-root tests/fixtures/video_tracking \
  --input-kind frames \
  --detector-backend rfdetr \
  --rfdetr-model nano \
  --rfdetr-threshold 1.0 \
  --input-width 64 \
  --device cpu \
  --max-frames 1
```

The regression suite runs this workflow against the committed `rfdetr_cli_smoke` frame scene with the installed `rfdetr` package. Its test-only DINO substitute avoids downloading a second, unrelated Hugging Face model; a direct CLI invocation still loads the configured DINO model normally.

Both frame/video detector paths use a normalized maximum input width of 960 pixels by default; smaller inputs are not upscaled. This single resize is passed to both the selected segmenter and DINO feature extraction through `system.input_width_size`. Use `--input-width WIDTH` to change it for either backend. Existing YOLO commands may retain `--yolo-imgsz WIDTH` as a YOLO-only compatibility override; RF-DETR rejects that option and the positional YOLO model argument so contradictory commands fail immediately. For RF-DETR, a non-empty `--classes` filter must match at least one loaded vocabulary entry; a partial match keeps its matched classes, while an all-unmatched filter reports the requested and available classes instead of silently dropping every detection.

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

Scene lookup defaults to `--input-kind auto`, which prefers `evalData/frames/<scene>/` when it exists and otherwise uses `evalData/videos/<scene>/` or `evalData/videos/<scene>.mp4`. Pass `--input-kind video` or `--input-kind frames` to force one layout.

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
python evaluation/run_tracking_test.py \
  --detector-backend davis \
  --frames-dir /path/to/FRAMES/ \
  --davis-meta-path /path/to/metaCUSTOMVIDEO.json \
  --davis-annotations-dir /path/to/Annotations/raw/FRAMES \
  --sequence-name FRAMES \
  --output-dir /path/to/outputs/
```

### Batch evaluation

```bash
python evaluation/run_tracking_batch.py \
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

Evaluation configs also support `"davis"` for ground-truth masks from DAVIS / ScanNet++. The frame/video CLI exposes `"yolo"` (default) and `"rfdetr"`; use the evaluation scripts rather than `main.py` for DAVIS-backed metrics.

---
