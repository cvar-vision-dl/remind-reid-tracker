"""RF-DETR prediction coverage for ScanNet++ tar evaluation."""

from __future__ import annotations

import io
import json
import sys
import tarfile
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from detection import davis_segmenter as davis_segmenter_module  # noqa: E402
from detection.rfdetr_segmenter import RFDETRSegmenter  # noqa: E402
from evaluation import davis_gt as davis_gt_module  # noqa: E402
from evaluation import run_tracking_batch_tar as tar_runner  # noqa: E402
from features.dino_extractor import DinoExtractor  # noqa: E402


class FakeRFDETRSegNano:
    """Deterministic external RF-DETR boundary with one table prediction."""

    class_names = {7: "pred_table"}
    received_image: np.ndarray | None = None
    received_threshold: float | None = None
    init_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        self.__class__.init_kwargs = dict(kwargs)

    def predict(self, image: np.ndarray, threshold: float):
        """Record RGB inference input and return a full-frame-sized instance mask."""
        self.__class__.received_image = image.copy()
        self.__class__.received_threshold = float(threshold)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask[2:7, 3:8] = 1
        return types.SimpleNamespace(
            xyxy=np.array([[3.0, 2.0, 8.0, 7.0]], dtype=np.float32),
            mask=mask[None, ...],
            class_id=np.array([7], dtype=np.int64),
            confidence=np.array([0.8], dtype=np.float32),
        )


def _add_tar_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    """Add in-memory fixture data to a tar archive."""
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _write_tar_scene(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    """Create one data tar and one independent annotation tar for a scene."""
    scene_id = "scene-1"
    data_root = tmp_path / "data"
    annotations_root = tmp_path / "annotations"
    data_root.mkdir()
    annotations_root.mkdir()

    bgr_frame = np.full((16, 16, 3), (11, 37, 83), dtype=np.uint8)
    ok, encoded_frame = cv2.imencode(".png", bgr_frame)
    assert ok

    annotation_mask = np.zeros((16, 16), dtype=np.uint8)
    annotation_mask[2:7, 3:8] = 1
    ok, encoded_mask = cv2.imencode(".png", annotation_mask)
    assert ok

    with tarfile.open(data_root / f"{scene_id}.tar", "w") as archive:
        _add_tar_member(
            archive,
            f"{scene_id}/dslr/resized_images/frame_000000.png",
            encoded_frame.tobytes(),
        )
    with tarfile.open(annotations_root / f"{scene_id}.tar", "w") as archive:
        _add_tar_member(
            archive,
            f"{scene_id}/meta_benchmark_instance.json",
            json.dumps(
                {
                    "scene_id": scene_id,
                    "frame_names": ["frame_000000.png"],
                    "id_to_label": {"1": "gt_chair_1"},
                }
            ).encode("utf-8"),
        )
        _add_tar_member(
            archive,
            f"{scene_id}/annotations/benchmark_instance/frame_000000.png",
            encoded_mask.tobytes(),
        )
    return data_root, annotations_root, bgr_frame


def _write_cpu_test_config(tmp_path: Path) -> Path:
    """Copy the project config with deterministic CPU-only feature settings."""
    project_config = (
        Path(__file__).resolve().parents[1] / "config" / "default_config.yaml"
    )
    config = yaml.safe_load(project_config.read_text(encoding="utf-8"))
    config["system"]["input_width_size"] = 16
    config["runtime"]["device"] = "cpu"
    config["detector"]["ignored_classes"] = []
    config["object_features"] = {}
    config["part_descriptors"] = {"enabled": False}
    config["bg_local"] = {"enabled": False}
    config["paths"]["output_dir"] = str(tmp_path / "pipeline-output")

    config_path = tmp_path / "tar-rfdetr-test.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _offline_dino_load(self: DinoExtractor) -> None:
    """Avoid the unrelated remote DINO model download during this tar test."""
    self.patch_size = 16


def _offline_dino_extract(self: DinoExtractor, image_rgb: np.ndarray) -> np.ndarray:
    """Return a deterministic one-patch DINO feature map."""
    assert image_rgb.shape == (16, 16, 3)
    return np.ones((1, 1, 1), dtype=np.float32)


def test_tar_rfdetr_prediction_keeps_davis_ground_truth_loader_independent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Prevent RF-DETR tar predictions from replacing DAVIS GT or leaking patches."""
    data_root, annotations_root, bgr_frame = _write_tar_scene(tmp_path)
    config_path = _write_cpu_test_config(tmp_path)
    bundle = tar_runner._build_scene_bundle(
        scene_id="scene-1",
        data_tar_root=data_root,
        annotations_tar_root=annotations_root,
        mask_variant="benchmark",
        image_subdir="dslr/resized_images",
    )
    FakeRFDETRSegNano.received_image = None
    FakeRFDETRSegNano.received_threshold = None
    FakeRFDETRSegNano.init_kwargs = None
    monkeypatch.setitem(
        sys.modules,
        "rfdetr",
        types.SimpleNamespace(RFDETRSegNano=FakeRFDETRSegNano),
    )
    monkeypatch.setattr(DinoExtractor, "load_model", _offline_dino_load)
    monkeypatch.setattr(DinoExtractor, "extract_patches", _offline_dino_extract)

    contexts: list[object] = []
    gt_loaders: list[object] = []
    perception_outputs: list[object] = []
    original_initialize_system = tar_runner.initialize_system
    original_gt_loader = tar_runner.DavisGroundTruthLoader
    original_process_frame = tar_runner.ReIDPipeline.process_frame

    def capture_context(config: dict):
        """Preserve factory behavior while retaining the constructed detector."""
        context = original_initialize_system(config)
        contexts.append(context)
        return context

    def capture_gt_loader(config: dict):
        """Preserve DAVIS loading while retaining its tar-specific segmenter."""
        loader = original_gt_loader(config)
        gt_loaders.append(loader)
        return loader

    def capture_process_frame(*args: object, **kwargs: object):
        """Preserve pipeline processing while retaining detector output for its oracle."""
        outputs = original_process_frame(*args, **kwargs)
        perception_outputs.append(outputs[0])
        return outputs

    monkeypatch.setattr(tar_runner, "initialize_system", capture_context)
    monkeypatch.setattr(tar_runner, "DavisGroundTruthLoader", capture_gt_loader)
    monkeypatch.setattr(tar_runner.ReIDPipeline, "process_frame", capture_process_frame)
    original_detector_davis = davis_segmenter_module.DavisSegmenter
    original_gt_davis = davis_gt_module.DavisSegmenter

    results, _ = tar_runner._evaluate_scene_tar(
        project_dir=tmp_path,
        config_path=config_path,
        scene_bundle=bundle,
        stable_min_frames=1,
        max_frames=1,
        force_detector_backend="rfdetr",
        rfdetr_model="nano",
        rfdetr_weights="custom-rfdetr.pth",
        rfdetr_threshold=0.73,
    )

    assert FakeRFDETRSegNano.init_kwargs == {
        "device": "cpu",
        "pretrain_weights": "custom-rfdetr.pth",
    }
    assert FakeRFDETRSegNano.received_threshold == pytest.approx(0.73)
    np.testing.assert_array_equal(
        FakeRFDETRSegNano.received_image, bgr_frame[..., ::-1]
    )
    assert len(contexts) == 1
    assert isinstance(contexts[0].detector, RFDETRSegmenter)
    assert contexts[0].detector.class_id_to_name == {7: "pred_table"}
    assert len(gt_loaders) == 1
    assert isinstance(gt_loaders[0].segmenter, tar_runner.TarDavisSegmenter)
    gt_objects = gt_loaders[0].load_frame(frame_id=0, target_shape=(16, 16))
    assert gt_objects[1].label == "gt_chair_1"
    assert gt_objects[1].class_name == "gt_chair"
    assert gt_objects[1].area == 25
    assert gt_objects[1].bbox_xyxy == (3, 2, 8, 7)
    assert results["per_frame"][0]["n_objects"] == 1
    assert results["per_object"][0]["gt_label"] == "gt_chair_1"
    assert results["per_object"][0]["gt_class_name"] == "gt_chair"
    assert len(perception_outputs) == 1
    assert len(perception_outputs[0].detections) == 1
    assert perception_outputs[0].detections[0].class_id == 7
    assert perception_outputs[0].detections[0].class_name == "pred_table"
    assert results["timing_summary"]["detector_mode"] == "rfdetr"
    assert results["timing_summary"]["detector_backend"] == "rfdetr"
    expected_provenance = {
        "detector_mode": "rfdetr",
        "detector_backend": "rfdetr",
        "rfdetr_model_variant": "nano",
        "rfdetr_pretrain_weights": "custom-rfdetr.pth",
        "rfdetr_threshold": pytest.approx(0.73),
    }
    assert results["detector_provenance"] == expected_provenance
    for key, value in expected_provenance.items():
        assert results["timing_summary"][key] == value
        assert results["summary"][key] == value
    assert davis_segmenter_module.DavisSegmenter is original_detector_davis
    assert davis_gt_module.DavisSegmenter is original_gt_davis


def test_tar_parser_exposes_rfdetr_overrides() -> None:
    """Prevent tar CLI RF-DETR selection from lacking the adapter's public knobs."""
    args = tar_runner._build_parser().parse_args(
        [
            "--detector-backend",
            "rfdetr",
            "--rfdetr-model",
            "nano",
            "--rfdetr-weights",
            "custom-rfdetr.pth",
            "--rfdetr-threshold",
            "0.73",
        ]
    )

    assert args.detector_backend == "rfdetr"
    assert args.rfdetr_model == "nano"
    assert args.rfdetr_weights == "custom-rfdetr.pth"
    assert args.rfdetr_threshold == pytest.approx(0.73)
    defaults = tar_runner._build_parser().parse_args([])
    assert defaults.rfdetr_model is None
    assert defaults.rfdetr_weights is None
    assert defaults.rfdetr_threshold is None


def test_tar_davis_provenance_keeps_legacy_gt_mode_label() -> None:
    """Prevent RF-DETR provenance from renaming existing DAVIS tar results."""
    provenance = tar_runner._build_detector_provenance(
        detector_mode="davis",
        config={"detector": {"backend": "davis"}},
        yolo_model_path=None,
    )

    assert provenance == {"detector_mode": "gt", "detector_backend": "davis"}
