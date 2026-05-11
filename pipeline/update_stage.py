# pipeline/update_stage.py

from __future__ import annotations

from update.memory_manager import MemoryManager


class UpdateStage:
    """
    Memory update stage (lifecycle + models).
    """

    def __init__(self, config: dict, memory_store, class_id_to_name=None):
        self.config = config
        self.memory_store = memory_store
        self.class_id_to_name = class_id_to_name if isinstance(class_id_to_name, dict) else None

        self.manager = MemoryManager(
            config=config,
            memory_store=memory_store,
            class_id_to_name=self.class_id_to_name,
        )

    def process_frame(
        self,
        detections: list,
        features_by_det: dict,
        association_output,
        frame_id: int,
        timestamp: float,
    ):
        return self.manager.apply_frame(
            detections=detections,
            det_features_by_id=features_by_det,
            association_output=association_output,
            timestamp=timestamp,
            frame_id=frame_id,
        )
