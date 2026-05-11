# pipeline/reid_pipeline.py

from __future__ import annotations

from pipeline.perception_stage_full import PerceptionStageFull
from pipeline.association_stage import AssociationStage
from pipeline.update_stage import UpdateStage
from utils.time import ExecutionTimer, format_timing_tree_line, format_timing_tree_table


class ReIDPipeline:
    """
    Pipeline completo por frame (FULL).

    Nuevo contrato:
      - perception -> FramePerceptionOutput
      - association -> FrameAssociationOutput
      - update -> FrameUpdateOutput
    """

    def __init__(self, runtime_ctx):
        self.ctx = runtime_ctx
        timing_cfg = (runtime_ctx.config.get("timing", {}) or {})
        self.stage_timing_enabled = bool(timing_cfg.get("enabled", True))
        self.stage_timing_precision = int(timing_cfg.get("precision", 2))
        self.stage_timing_table = bool(timing_cfg.get("table", True))
        detail_keys_cfg = timing_cfg.get(
            "detail_keys",
            [
                "detector",
                "detector/segment",
                "detector/ignored_filter",
                "dino",
                "obj_features",
                "bg_features",
                "bg_features/bg_rings",
                "bg_features/bg_inner_global",
                "bg_features/bg_outer_global",
                "bg_features/bg_proto_inner",
                "bg_features/bg_proto_outer",
                "parts_features",
                "parts_features/parts_kmeans",
                "parts_features/parts_attention",
            ],
        )
        if isinstance(detail_keys_cfg, list):
            self.perception_detail_keys = [str(x) for x in detail_keys_cfg]
        else:
            self.perception_detail_keys = ["detector", "dino"]

        assoc_detail_keys_cfg = timing_cfg.get(
            "association_detail_keys",
            [
                "sim_candidates",
                "reliable_anchor_ids",
                "neighbor_sets",
                "ambiguity_sim",
                "hungarian",
                "hungarian/partition",
                "hungarian/prepare",
                "hungarian/prepare/context",
                "hungarian/prepare/shortlist",
                "hungarian/prepare/soft_gate",
                "hungarian/assign_classes",
                "hungarian/assign_classes/score_tables",
                "hungarian/assign_classes/locks",
                "hungarian/assign_classes/solve",
                "hungarian/assign_classes/solve/score_tables",
                "hungarian/assign_classes/solve/cost_matrix",
                "hungarian/assign_classes/solve/resolve",
                "post_assignment",
                "post_assignment/identity_stability",
                "post_assignment/ambiguous_candidates",
                "post_assignment/temporal_reconcile",
                "post_assignment/final_pack",
                "post_assignment/geom_pack",
                "post_assignment/finalize",
                "ambiguity_final",
            ],
        )
        if isinstance(assoc_detail_keys_cfg, list):
            self.association_detail_keys = [str(x) for x in assoc_detail_keys_cfg]
        else:
            self.association_detail_keys = []

        update_detail_keys_cfg = timing_cfg.get(
            "update_detail_keys",
            [
                "matches",
                "creates",
                "ambiguous_tracks",
                "provisional_tracks",
                "neighbor_graphs",
                "neighbor_graphs/dist_observations",
                "neighbor_graphs/graph_updates",
                "misses",
                "finalize",
            ],
        )
        if isinstance(update_detail_keys_cfg, list):
            self.update_detail_keys = [str(x) for x in update_detail_keys_cfg]
        else:
            self.update_detail_keys = []
        self.last_stage_times_seconds: dict[str, float] = {}
        self.last_stage_details_seconds: dict[str, dict[str, float]] = {}
        self.last_stage_timing_line: str = ""
        self.last_stage_timing_table: str = ""

        self.perception_stage = PerceptionStageFull(
            config=runtime_ctx.config,
            yolo=runtime_ctx.yolo,
            dino=runtime_ctx.dino,
        )

        class_id_to_name = getattr(runtime_ctx.yolo, "class_id_to_name", None)

        self.association_stage = AssociationStage(
            runtime_ctx.config,
            runtime_ctx.memory,
            output_dir=runtime_ctx.output_dir,
            class_id_to_name=class_id_to_name,
        )

        self.update_stage = UpdateStage(
            config=runtime_ctx.config,
            memory_store=runtime_ctx.memory,
            class_id_to_name=class_id_to_name,
        )

    def process_frame(self, frame, frame_id: int, timestamp: float):
        timer = ExecutionTimer()

        p_out = timer.run(
            "perception",
            self.perception_stage.process_frame,
            frame=frame,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        a_out = timer.run(
            "association",
            self.association_stage.process_frame,
            detections=p_out.detections,
            features_by_det=p_out.det_features_by_id,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        u_out = timer.run(
            "update",
            self.update_stage.process_frame,
            detections=p_out.detections,
            features_by_det=p_out.det_features_by_id,
            association_output=a_out,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        self.last_stage_times_seconds = timer.snapshot_seconds()
        perception_timing = getattr(p_out, "timings_seconds", {}) or {}
        assoc_timing = getattr(a_out, "timings_seconds", {}) or {}
        update_timing = getattr(u_out, "timings_seconds", {}) or {}

        timing_aliases = {"yolo": "detector", "detector": "detector"}

        def collect_stage_details(
            timings: dict[str, float],
            preferred_keys: list[str] | None,
            *,
            aliases: dict[str, str] | None = None,
        ) -> dict[str, float]:
            out: dict[str, float] = {}
            seen: set[str] = set()
            alias_map = aliases or {}

            for key in (preferred_keys or []):
                resolved = alias_map.get(str(key), str(key))
                if resolved in timings and resolved not in seen:
                    out[str(resolved)] = float(timings[resolved])
                    seen.add(str(resolved))

            for key, seconds in (timings or {}).items():
                name = str(key)
                if name in seen:
                    continue
                out[name] = float(seconds)
                seen.add(name)

            return out

        per_details = collect_stage_details(
            perception_timing,
            self.perception_detail_keys,
            aliases=timing_aliases,
        )
        assoc_details = collect_stage_details(
            assoc_timing,
            self.association_detail_keys,
        )
        update_details = collect_stage_details(
            update_timing,
            self.update_detail_keys,
        )

        self.last_stage_details_seconds = {
            "perception": per_details,
            "association": assoc_details,
            "update": update_details,
        }

        stage_order = ["perception", "association", "update"]
        detail_order_by_stage = {
            "perception": [timing_aliases.get(str(k), str(k)) for k in self.perception_detail_keys],
            "association": list(self.association_detail_keys or []),
            "update": list(self.update_detail_keys or []),
        }

        self.last_stage_timing_table = format_timing_tree_table(
            self.last_stage_times_seconds,
            stage_order=stage_order,
            details_by_stage=self.last_stage_details_seconds,
            detail_order_by_stage=detail_order_by_stage,
            precision=self.stage_timing_precision,
            total_seconds=timer.total_seconds(),
            title=f"[TIME][frame={frame_id}]",
        )
        self.last_stage_timing_line = format_timing_tree_line(
            self.last_stage_times_seconds,
            stage_order=stage_order,
            details_by_stage=self.last_stage_details_seconds,
            detail_order_by_stage=detail_order_by_stage,
            precision=self.stage_timing_precision,
            total_seconds=timer.total_seconds(),
        )
        if self.stage_timing_enabled:
            if self.stage_timing_table and self.last_stage_timing_table:
                print(self.last_stage_timing_table)
            else:
                print(f"[TIME][frame={frame_id}] {self.last_stage_timing_line}")

        return p_out, a_out, u_out
