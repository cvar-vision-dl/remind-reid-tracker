"""Black-box contract tests for the optional RF-DETR segmenter backend."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


class FakeDetections:
    """Minimal stand-in for ``supervision.Detections`` at the adapter boundary."""

    def __init__(
        self,
        *,
        xyxy: np.ndarray,
        mask: np.ndarray | None,
        class_id: np.ndarray,
        confidence: np.ndarray,
    ) -> None:
        self.xyxy = xyxy
        self.mask = mask
        self.class_id = class_id
        self.confidence = confidence


class FakeRFDETRModel:
    """Records model calls while returning a configured fake inference result."""

    detections: FakeDetections | None = None
    class_names = ["background", "chair", "table"]
    received_image: np.ndarray | None = None
    received_threshold: float | None = None

    def __init__(self, **_kwargs: object) -> None:
        pass

    def predict(self, image: np.ndarray, threshold: float) -> FakeDetections | None:
        type(self).received_image = image.copy()
        type(self).received_threshold = threshold
        return type(self).detections


@pytest.fixture(autouse=True)
def fake_rfdetr_module(monkeypatch: pytest.MonkeyPatch) -> type[FakeRFDETRModel]:
    """Install a deterministic optional-dependency substitute for each scenario."""
    FakeRFDETRModel.detections = None
    FakeRFDETRModel.class_names = ["background", "chair", "table"]
    FakeRFDETRModel.received_image = None
    FakeRFDETRModel.received_threshold = None
    monkeypatch.setitem(
        sys.modules,
        "rfdetr",
        types.SimpleNamespace(
            RFDETRSegNano=FakeRFDETRModel,
            RFDETRSegSmall=FakeRFDETRModel,
            RFDETRSegMedium=FakeRFDETRModel,
            RFDETRSegLarge=FakeRFDETRModel,
            RFDETRSegXLarge=FakeRFDETRModel,
            RFDETRSeg2XLarge=FakeRFDETRModel,
        ),
    )
    return FakeRFDETRModel


def _segmenter_config(**rfdetr_overrides: object) -> dict:
    rfdetr = {
        "model_variant": "medium",
        "pretrain_weights": None,
        "threshold": 0.42,
        "classes": None,
        "mask_erosion_px": 0,
        "mask_erosion_iters": 1,
    }
    rfdetr.update(rfdetr_overrides)
    return {"rfdetr": rfdetr}


def _loaded_segmenter(config: dict):
    from detection.rfdetr_segmenter import RFDETRSegmenter

    segmenter = RFDETRSegmenter(config=config, device="cpu")
    segmenter.load_model()
    return segmenter


def test_segment_one_and_multiple_detections_maps_tracker_contract(
    fake_rfdetr_module: type[FakeRFDETRModel],
) -> None:
    """Prevent output-array ordering or metadata loss across the adapter boundary."""
    fake_rfdetr_module.detections = FakeDetections(
        xyxy=np.array([[0, 0, 2, 2], [1, 0, 4, 2]], dtype=np.float32),
        mask=np.array(
            [
                [[1, 0, 0, 0], [1, 1, 0, 0]],
                [[0, 1, 1, 0], [0, 0, 1, 1]],
            ],
            dtype=np.uint8,
        ),
        class_id=np.array([1, 2], dtype=np.int64),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
    )
    bgr_frame = np.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]],
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [11, 12, 13]],
        ],
        dtype=np.uint8,
    )

    segmenter = _loaded_segmenter(_segmenter_config())
    assert segmenter.class_id_to_name == {0: "background", 1: "chair", 2: "table"}

    detections = segmenter.segment(
        bgr_frame, frame_id=7, timestamp=1.25
    )

    assert fake_rfdetr_module.received_threshold == pytest.approx(0.42)
    np.testing.assert_array_equal(fake_rfdetr_module.received_image, bgr_frame[..., ::-1])
    assert [det.detection_id for det in detections] == [0, 1]
    assert [det.class_id for det in detections] == [1, 2]
    assert [det.class_name for det in detections] == ["chair", "table"]
    assert [det.original_class_name for det in detections] == ["chair", "table"]
    assert [det.frame_id for det in detections] == [7, 7]
    assert [det.timestamp for det in detections] == [1.25, 1.25]
    assert [det.confidence for det in detections] == pytest.approx([0.9, 0.8])
    assert detections[0].bbox == pytest.approx((0.0, 0.0, 2.0, 2.0))
    np.testing.assert_array_equal(detections[1].mask, np.array([[0, 1, 1, 0], [0, 0, 1, 1]], dtype=bool))
    assert detections[0].geom == {"center": (1 / 3, 2 / 3), "area": 3.0}
    assert detections[1].geom == {"center": (2.0, 0.5), "area": 4.0}


def test_segment_resizes_mask_with_nearest_neighbor_and_recomputes_geometry(
    fake_rfdetr_module: type[FakeRFDETRModel],
) -> None:
    """Prevent bilinear mask resizing from inventing fractional object pixels."""
    fake_rfdetr_module.detections = FakeDetections(
        xyxy=np.array([[0, 0, 4, 4]], dtype=np.float32),
        mask=np.array([[[1, 0], [0, 0]]], dtype=np.uint8),
        class_id=np.array([1]),
        confidence=np.array([0.6]),
    )

    detections = _loaded_segmenter(_segmenter_config()).segment(
        np.zeros((4, 4, 3), dtype=np.uint8), frame_id=0, timestamp=0.0
    )

    assert len(detections) == 1
    np.testing.assert_array_equal(
        detections[0].mask,
        np.array(
            [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=bool,
        ),
    )
    assert detections[0].geom == {"center": (0.5, 0.5), "area": 4.0}


@pytest.mark.parametrize(
    ("detections", "case_id"),
    [
        pytest.param(None, "none_result", id="none-result"),
        pytest.param(
            FakeDetections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                mask=np.empty((0, 2, 2), dtype=np.uint8),
                class_id=np.empty((0,), dtype=np.int64),
                confidence=np.empty((0,), dtype=np.float32),
            ),
            "empty_result",
            id="empty-result",
        ),
        pytest.param(
            FakeDetections(
                xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32),
                mask=None,
                class_id=np.array([1]),
                confidence=np.array([0.5]),
            ),
            "missing_masks",
            id="missing-masks",
        ),
    ],
)
def test_segment_no_usable_masks_returns_empty_list(
    fake_rfdetr_module: type[FakeRFDETRModel],
    detections: FakeDetections | None,
    case_id: str,
) -> None:
    """Prevent no-mask outputs from leaking incomplete detections downstream."""
    del case_id
    fake_rfdetr_module.detections = detections

    result = _loaded_segmenter(_segmenter_config()).segment(
        np.zeros((2, 2, 3), dtype=np.uint8), frame_id=0, timestamp=0.0
    )

    assert result == []


def test_segment_skips_mask_eroded_to_empty(fake_rfdetr_module: type[FakeRFDETRModel]) -> None:
    """Prevent zero-area detections after enabled mask erosion."""
    fake_rfdetr_module.detections = FakeDetections(
        xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32),
        mask=np.array(
            [[[0, 0, 0], [0, 1, 0], [0, 0, 0]]],
            dtype=np.uint8,
        ),
        class_id=np.array([1]),
        confidence=np.array([0.5]),
    )

    detections = _loaded_segmenter(
        _segmenter_config(mask_erosion_px=1, mask_erosion_iters=1)
    ).segment(np.zeros((3, 3, 3), dtype=np.uint8), frame_id=0, timestamp=0.0)

    assert detections == []


def test_segment_rejects_misaligned_result_arrays(fake_rfdetr_module: type[FakeRFDETRModel]) -> None:
    """Prevent zip-style truncation when RF-DETR output fields disagree in length."""
    fake_rfdetr_module.detections = FakeDetections(
        xyxy=np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32),
        mask=np.ones((1, 2, 2), dtype=np.uint8),
        class_id=np.array([1, 2]),
        confidence=np.array([0.5, 0.7]),
    )

    with pytest.raises(ValueError, match="mask|length|shape|count"):
        _loaded_segmenter(_segmenter_config()).segment(
            np.zeros((2, 2, 3), dtype=np.uint8), frame_id=0, timestamp=0.0
        )


def test_segment_rejects_class_filter_without_a_loaded_class(
    fake_rfdetr_module: type[FakeRFDETRModel],
) -> None:
    """Prevent an invalid RF-DETR filter from silently discarding every detection."""
    fake_rfdetr_module.detections = FakeDetections(
        xyxy=np.array([[0, 0, 2, 2]], dtype=np.float32),
        mask=np.array([[[1, 1], [1, 1]]], dtype=np.uint8),
        class_id=np.array([1]),
        confidence=np.array([0.9]),
    )
    segmenter = _loaded_segmenter(_segmenter_config(classes=["not-in-vocabulary"]))

    with pytest.raises(ValueError, match="matched no loaded RF-DETR classes"):
        segmenter.segment(np.zeros((2, 2, 3), dtype=np.uint8), frame_id=0, timestamp=0.0)


def test_load_model_reports_missing_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent selecting RF-DETR from breaking startup with an opaque import traceback."""
    from detection import rfdetr_segmenter

    def missing_rfdetr_module(name: str):
        assert name == "rfdetr"
        raise ModuleNotFoundError("No module named 'rfdetr'", name="rfdetr")

    monkeypatch.setattr(rfdetr_segmenter.importlib, "import_module", missing_rfdetr_module)

    with pytest.raises((ImportError, RuntimeError), match="rfdetr.*install|install.*rfdetr"):
        rfdetr_segmenter.RFDETRSegmenter(config=_segmenter_config(), device="cpu").load_model()
