"""Integration contracts shared by the RF-DETR backend and tracker pipeline."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from detection.detection import Detection
from perception.perception_engine import FramePerceptionContext, PerceptionEngine


class StubDetector:
    """Tracker-compatible detector returning one aligned-mask detection."""

    class_id_to_name = {4: "chair"}

    def __init__(self) -> None:
        self.received_frame: np.ndarray | None = None

    def segment(self, frame: np.ndarray, frame_id: int, timestamp: float) -> list[Detection]:
        self.received_frame = frame.copy()
        return [
            Detection(
                detection_id=11,
                class_id=4,
                frame_id=frame_id,
                timestamp=timestamp,
                bbox=(0.0, 0.0, 2.0, 2.0),
                mask=np.array(
                    [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                    dtype=bool,
                ),
                confidence=0.9,
                geom={"center": (0.5, 0.5), "area": 4.0},
            )
        ]


class StubDino:
    """Minimal deterministic DINO boundary for the perception hand-over test."""

    patch_size = 2

    def __init__(self) -> None:
        self.received_rgb: np.ndarray | None = None

    def extract_patches(self, image_rgb: np.ndarray) -> np.ndarray:
        self.received_rgb = image_rgb.copy()
        return np.zeros((2, 2, 3), dtype=np.float32)

    def mask_px_to_patch_coverage(
        self, mask: np.ndarray, height_patches: int, width_patches: int
    ) -> np.ndarray:
        assert mask.shape == (4, 4)
        assert (height_patches, width_patches) == (2, 2)
        return np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    def patch_mask_from_coverage(self, coverage: np.ndarray) -> np.ndarray:
        return coverage > 0


def test_build_segmenter_dispatches_rfdetr_backend(monkeypatch) -> None:
    """Prevent the completed adapter from being unreachable through the core factory."""
    created: list[object] = []

    class FakeRFDETRSegmenter:
        def __init__(self, config: dict, device: str) -> None:
            self.config = config
            self.device = device
            self.loaded = False
            created.append(self)

        def load_model(self) -> None:
            self.loaded = True

    monkeypatch.setitem(
        sys.modules,
        "detection.rfdetr_segmenter",
        types.SimpleNamespace(RFDETRSegmenter=FakeRFDETRSegmenter),
    )
    from pipeline.initialization import build_segmenter

    config = {"detector": {"backend": "rfdetr"}}
    segmenter = build_segmenter(config, device="cpu")

    assert segmenter is created[0]
    assert segmenter.config is config
    assert segmenter.device == "cpu"
    assert segmenter.loaded is True


def test_perception_hands_aligned_masks_to_feature_store() -> None:
    """Prevent a backend-compatible detection from losing its mask or ID after perception."""
    detector = StubDetector()
    dino = StubDino()
    engine = PerceptionEngine(
        config={
            "system": {"input_width_size": 4},
            "perception": {"full_center_crop": True},
            "object_features": {},
            "part_descriptors": {"enabled": False},
            "bg_local": {"enabled": False},
        },
        detector=detector,
        dino=dino,
    )
    bgr_frame = np.array(
        [
            [[3, 2, 1], [6, 5, 4], [9, 8, 7], [12, 11, 10]],
            [[15, 14, 13], [18, 17, 16], [21, 20, 19], [24, 23, 22]],
            [[27, 26, 25], [30, 29, 28], [33, 32, 31], [36, 35, 34]],
            [[39, 38, 37], [42, 41, 40], [45, 44, 43], [48, 47, 46]],
        ],
        dtype=np.uint8,
    )

    output = engine.process_frame(bgr_frame, FramePerceptionContext(frame_id=3, timestamp=0.1))

    assert detector.received_frame is not None
    np.testing.assert_array_equal(detector.received_frame, bgr_frame)
    np.testing.assert_array_equal(dino.received_rgb, bgr_frame[..., ::-1])
    assert [det.detection_id for det in output.detections] == [11]
    assert output.summary["n_detections"] == 1
    assert set(output.det_features_by_id) == {11}
    np.testing.assert_array_equal(output.det_features_by_id[11]["mask"], output.detections[0].mask)
    assert output.det_features_by_id[11]["meta"]["effective_obj_patches"] == 1.0


def test_cli_rfdetr_selection_needs_no_yolo_model_and_preserves_yolo_config(tmp_path: Path) -> None:
    """Prevent RF-DETR mode from requiring or overwriting YOLO-only configuration."""
    from scripts.run_video_tracking import _configure, build_parser

    config_path = tmp_path / "base.yaml"
    config_path.write_text(
        """
detector:
  backend: yolo
yolo:
  model_label: KEEP
  models:
    KEEP: preserved.pt
  conf_th: 0.91
rfdetr:
  pretrain_weights: null
""".lstrip(),
        encoding="utf-8",
    )
    weights_path = tmp_path / "custom.pth"
    args = build_parser().parse_args(
        [
            "scene-name",
            "--detector-backend",
            "rfdetr",
            "--rfdetr-model",
            "small",
            "--rfdetr-weights",
            str(weights_path),
            "--rfdetr-threshold",
            "0.65",
            "--config",
            str(config_path),
            "--classes",
            "chair,table",
            "--mask-erosion-px",
            "2",
        ]
    )

    configured = _configure(args, output_dir=tmp_path / "outputs")

    assert args.yolo_model is None
    assert configured["detector"]["backend"] == "rfdetr"
    assert configured["rfdetr"] == {
        "pretrain_weights": str(weights_path),
        "model_variant": "small",
        "threshold": pytest.approx(0.65),
        "classes": ["chair", "table"],
        "mask_erosion_px": 2,
        "mask_erosion_iters": 1,
    }
    assert configured["yolo"] == {
        "model_label": "KEEP",
        "models": {"KEEP": "preserved.pt"},
        "conf_th": 0.91,
    }


@pytest.mark.parametrize(
    ("argv", "error_match"),
    [
        pytest.param(
            ["scene-name", "yolo-seg.pt", "--detector-backend", "rfdetr"],
            "yolo_model.*only valid",
            id="positional-model",
        ),
        pytest.param(
            ["scene-name", "--detector-backend", "rfdetr", "--yolo-imgsz", "736"],
            "--yolo-imgsz.*only valid",
            id="yolo-image-size",
        ),
    ],
)
def test_cli_rfdetr_rejects_yolo_only_arguments_before_source_resolution(
    argv: list[str], error_match: str
) -> None:
    """Prevent RF-DETR runs from silently accepting YOLO-only arguments."""
    from scripts.run_video_tracking import main

    with pytest.raises(SystemExit, match=error_match):
        main(argv)


@pytest.mark.parametrize(
    ("argv", "expected_width"),
    [
        pytest.param(
            ["scene-name", "--detector-backend", "rfdetr"],
            960,
            id="rfdetr-default",
        ),
        pytest.param(
            ["scene-name", "--detector-backend", "rfdetr", "--input-width", "736"],
            736,
            id="rfdetr-explicit-input-width",
        ),
        pytest.param(["scene-name", "yolo-seg.pt"], 960, id="yolo-default"),
        pytest.param(
            ["scene-name", "yolo-seg.pt", "--input-width", "736"],
            736,
            id="yolo-explicit-input-width",
        ),
        pytest.param(
            ["scene-name", "yolo-seg.pt", "--yolo-imgsz", "736"],
            736,
            id="yolo-legacy-image-size",
        ),
    ],
)
def test_cli_configure_normalizes_input_width_and_preserves_yolo_alias(
    tmp_path: Path,
    argv: list[str],
    expected_width: int,
) -> None:
    """Prevent detector choice from changing resize width or breaking legacy YOLO commands."""
    from scripts.run_video_tracking import _configure, build_parser

    config_path = tmp_path / "base.yaml"
    config_path.write_text("system:\n  input_width_size: 1280\n", encoding="utf-8")
    args = build_parser().parse_args([*argv, "--config", str(config_path)])

    configured = _configure(args, output_dir=tmp_path / "outputs")

    assert configured["system"]["input_width_size"] == expected_width
