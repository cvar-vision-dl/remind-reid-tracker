"""Real-package RF-DETR inference smoke coverage."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

RFDETR_CLI_FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "video_tracking"


def _offline_dino_load(self) -> None:
    """Avoid an unrelated Hugging Face DINO download in the RF-DETR CLI smoke test."""
    self.patch_size = 16


def _offline_dino_extract(self, image_rgb: np.ndarray) -> np.ndarray:
    """Return a deterministic patch map for the 64-pixel smoke-test scene."""
    height, width = image_rgb.shape[:2]
    return np.zeros(
        (height // self.patch_size, width // self.patch_size, 1), dtype=np.float32
    )


def test_rfdetr_seg_nano_constructs_and_predicts_deterministic_frame() -> None:
    """Exercise the public RF-DETR package without replacing its model or prediction boundary."""
    from detection.rfdetr_segmenter import RFDETRSegmenter
    from rfdetr import RFDETRSegNano

    height, width = 64, 96
    x_coords = np.broadcast_to(np.arange(width, dtype=np.uint8), (height, width))
    y_coords = np.broadcast_to(
        np.arange(height, dtype=np.uint8)[:, None], (height, width)
    )
    bgr_frame = np.stack(
        (
            x_coords,
            y_coords,
            (x_coords.astype(np.uint16) + y_coords.astype(np.uint16)).astype(np.uint8),
        ),
        axis=-1,
    )
    segmenter = RFDETRSegmenter(
        config={
            "rfdetr": {
                "model_variant": "nano",
                "pretrain_weights": None,
                "threshold": 1.0,
                "classes": None,
                "mask_erosion_px": 0,
                "mask_erosion_iters": 1,
            }
        },
        device="cpu",
    )

    segmenter.load_model()
    detections = segmenter.segment(bgr_frame, frame_id=12, timestamp=0.5)

    assert isinstance(segmenter.model, RFDETRSegNano)
    assert segmenter.class_id_to_name
    assert isinstance(detections, list)
    assert "predict" in segmenter.last_timings_seconds
    for detection in detections:
        assert detection.mask.shape == (height, width)
        assert detection.mask.dtype == bool


def test_cli_rfdetr_nano_processes_one_committed_scene_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Run the CLI entry point through one real RF-DETR prediction on the committed scene.

    DINO is replaced only because it unconditionally downloads a separate external
    model before the frame loop. RF-DETR construction and prediction use the
    installed package and both call-through spies retain their original behavior.
    """
    from features.dino_extractor import DinoExtractor
    from rfdetr import RFDETRSegNano

    constructor_calls: list[dict[str, object]] = []
    predict_calls: list[tuple[tuple[int, ...], float]] = []
    original_init = RFDETRSegNano.__init__
    original_predict = RFDETRSegNano.predict

    def record_init(model, *args: object, **kwargs: object) -> None:
        constructor_calls.append(dict(kwargs))
        original_init(model, *args, **kwargs)

    def record_predict(model, image: np.ndarray, threshold: float):
        predict_calls.append((tuple(image.shape), float(threshold)))
        return original_predict(model, image, threshold=threshold)

    monkeypatch.setattr(DinoExtractor, "load_model", _offline_dino_load)
    monkeypatch.setattr(DinoExtractor, "extract_patches", _offline_dino_extract)
    monkeypatch.setattr(RFDETRSegNano, "__init__", record_init)
    monkeypatch.setattr(RFDETRSegNano, "predict", record_predict)

    output_dir = tmp_path / "rfdetr-cli-output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "rfdetr_cli_smoke",
            "--test-root",
            str(RFDETR_CLI_FIXTURE_ROOT),
            "--input-kind",
            "frames",
            "--detector-backend",
            "rfdetr",
            "--rfdetr-model",
            "nano",
            "--rfdetr-threshold",
            "1.0",
            "--device",
            "cpu",
            "--input-width",
            "64",
            "--max-frames",
            "1",
            "--output-dir",
            str(output_dir),
        ],
    )
    runpy.run_path(str(REPOSITORY_ROOT / "main.py"), run_name="__main__")

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert constructor_calls == [{"device": "cpu"}]
    assert predict_calls == [((64, 64, 3), 1.0)]
    assert summary["processed_frames"] == 1
    assert summary["detector_backend"] == "rfdetr"
    assert summary["rfdetr_model"] == "nano"
