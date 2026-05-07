# pipeline/initialization.py

from __future__ import annotations

from pathlib import Path
import random
import numpy as np

import torch

from memory.memory_store import MemoryStore
from features.dino_extractor import DinoExtractor


class RuntimeContext:
    """
    Contexto global del sistema.
    """

    def __init__(
        self,
        config: dict,
        device: str,
        memory: MemoryStore,
        yolo,
        dino: DinoExtractor,
        output_dir: Path,
    ):
        self.config = config
        self.device = device
        self.memory = memory
        self.detector = yolo
        self.yolo = yolo
        self.dino = dino
        self.output_dir = output_dir


def set_seeds(seed: int | None) -> None:
    if seed is None:
        return

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def select_device(config: dict) -> str:
    runtime_cfg = config.get("runtime", {}) or {}
    requested = runtime_cfg.get("device", "auto")

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if torch is not None and torch.cuda.is_available():
        return "cuda"

    return "cpu"


def prepare_output_dir(config: dict) -> Path:
    paths_cfg = config.get("paths", {})
    out_cfg = paths_cfg.get("output_dir", "outputs")

    project_root = Path(__file__).resolve().parents[2]
    out = (project_root / out_cfg).resolve()

    out.mkdir(parents=True, exist_ok=True)
    (out / "debug").mkdir(exist_ok=True)
    (out / "metrics").mkdir(exist_ok=True)

    return out


def build_segmenter(config: dict, device: str):
    det_cfg = config.get("detector", {}) or {}
    backend = str(det_cfg.get("backend", "yolo")).strip().lower()

    if backend == "yolo":
        from detection.yolo_segmenter import YoloSegmenter

        seg = YoloSegmenter(config=config, device=device)
        seg.load_model()
        return seg

    if backend == "davis":
        from detection.davis_segmenter import DavisSegmenter

        seg = DavisSegmenter(config=config, device=device)
        seg.load_model()
        return seg

    raise ValueError(f"Backend de detector no soportado: {backend}")


def initialize_system(config: dict) -> RuntimeContext:
    device = select_device(config)

    seed = (config.get("runtime", {}) or {}).get("seed", None)
    set_seeds(seed)

    output_dir = prepare_output_dir(config)

    # Detector/segmentador
    yolo = build_segmenter(config=config, device=device)

    # Memory
    mem_cfg = config.get("memory", {}) or {}
    memory = MemoryStore(
        config=config,
        start_object_id=int(mem_cfg.get("start_object_id", 0)),
    )

    # DINO
    dino_cfg = config.get("dino", {}) or {}
    dino = DinoExtractor(config=dino_cfg, device=device)
    dino.load_model()

    return RuntimeContext(
        config=config,
        device=device,
        memory=memory,
        yolo=yolo,
        dino=dino,
        output_dir=output_dir,
    )
