# pipeline/perception_stage_full.py

from __future__ import annotations

from perception.perception_engine import PerceptionEngine, FramePerceptionContext


class PerceptionStageFull:
    """
    Wrapper fino: delega en PerceptionEngine.
    """

    def __init__(self, config, yolo, dino):
        self.engine = PerceptionEngine(config=config, yolo=yolo, dino=dino)

    def process_frame(self, frame, frame_id, timestamp):
        ctx = FramePerceptionContext(frame_id=frame_id, timestamp=timestamp)
        return self.engine.process_frame(frame=frame, frame_context=ctx)
