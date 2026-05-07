# pipeline/association_stage.py

from __future__ import annotations

from association.engine.data_association import DataAssociationEngine


class FrameAssociationContext:
    def __init__(self, frame_id: int, timestamp: float):
        self.frame_id = int(frame_id)
        self.timestamp = float(timestamp)


class AssociationStage:
    """
    Etapa de asociación por frame.

    Calcula evidencias, aplica contexto y deja preparada la decisión
    consumida por update:
      - decided_matches
      - to_create
      - to_ambiguous
    """

    def __init__(self, config, memory_store, output_dir=None, class_id_to_name=None):
        self.config = config
        self.memory_store = memory_store
        self.engine = DataAssociationEngine(
            config,
            memory_store,
            output_dir=output_dir,
            class_id_to_name=class_id_to_name,
        )

    def process_frame(self, detections, features_by_det, frame_id: int, timestamp: float):
        ctx = FrameAssociationContext(frame_id=frame_id, timestamp=timestamp)

        return self.engine.process_frame(
            detections=detections,
            det_features_by_id=features_by_det,
            frame_context=ctx,
        )
