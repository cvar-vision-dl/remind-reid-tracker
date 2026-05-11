from __future__ import annotations

from association.context.neighbor_sets_influence import NeighborSetsInfluence
from association.context.sets_provider import SetsContextProvider
from association.flow import AssociationFrameRequest, FrameAssociationFlow
from association.engine.assignment import HungarianAssigner
from association.engine.assignment_executor import AssignmentExecutor
from association.engine.observability_runtime import DataAssociationObservabilityRuntime
from association.engine.assignment_result_applier import AssignmentResultApplier
from association.engine.candidate_generation import CandidateGenerator
from association.engine.debug_view_builder import DebugViewBuilder
from association.models import AssignmentResult
from association.policy.confirmation_policy import ReliableVisualAnchorPolicy
from association.policy.outcome_policy import AssociationOutcomePolicy
from association.reports import FrameAssociationOutput, SimilarityReport
from association.score_aggregator import ScoreAggregator
from association.scores.base_scores import SimilarityCombiner
from utils.time import ExecutionTimer


class DataAssociationEngine:
    """
    Association por frame:
      - computes candidates (score_sim)
      - decides MATCH/NEW with Hungarian
      - exposes decisions and geometry for Update
    """

    def __init__(self, config, memory_store, output_dir=None, class_id_to_name=None):
        self.config = config
        self.memory_store = memory_store
        self.output_dir = output_dir
        self.class_id_to_name = self._normalize_class_id_to_name(class_id_to_name)
        self.scores = ScoreAggregator(config, memory_store=memory_store)
        self.combiner = SimilarityCombiner(config)
        assoc = (config.get("association", {}) or {})
        match = (assoc.get("matching", {}) or {})
        self.match_thr = float(match.get("match_thr", 0.0))
        self.clear_margin = float(match.get("clear_margin", 0.07))
        self.candidate_generator = CandidateGenerator(
            scores=self.scores,
            combiner=self.combiner,
            memory_store=self.memory_store,
        )
        self.neighbor_sets_influence = NeighborSetsInfluence(config=config, memory_store=memory_store)
        self.sets_provider = SetsContextProvider(
            scores=self.scores,
            neighbor_sets_influence=self.neighbor_sets_influence,
            memory_store=self.memory_store,
        )

        self.assigner = HungarianAssigner(config=config, memory_store=memory_store)
        self.outcome_policy = AssociationOutcomePolicy(
            config=config,
            combiner=self.combiner,
            neighbor_sets_influence=self.neighbor_sets_influence,
        )

        confirmation = (assoc.get("confirmation", {}) or {})
        self.confirm_thr_strong = float(confirmation.get("confirm_thr_strong", self.match_thr))
        self.confirm_clear_margin = float(confirmation.get("confirm_clear_margin", self.clear_margin))
        self.reliable_anchor_policy = ReliableVisualAnchorPolicy(
            candidate_generator=self.candidate_generator,
            confirm_thr_strong=self.confirm_thr_strong,
            confirm_clear_margin=self.confirm_clear_margin,
        )

        upd = (config.get("update", {}) or {})
        self.min_match_score = float(upd.get("min_match_score", 0.0))
        self.assignment_executor = AssignmentExecutor(
            assigner=self.assigner,
            memory_store=self.memory_store,
            match_thr=self.match_thr,
            min_match_score=self.min_match_score,
            neighbor_sets_influence=self.neighbor_sets_influence,
        )
        self.assignment_result_applier = AssignmentResultApplier(
            outcome_policy=self.outcome_policy,
            config=config,
            memory_store=memory_store,
        )

        dbg = (config.get("debug", {}) or {})
        dbg_assoc = (dbg.get("association", {}) or {})
        self.debug_enabled = bool(dbg_assoc.get("enabled", False))
        self.debug_topk = int(dbg_assoc.get("candidates_topk", 5))
        self.debug_view_builder = DebugViewBuilder(
            debug_topk=self.debug_topk,
            sets_provider=self.sets_provider,
        )
        self.observability = DataAssociationObservabilityRuntime(
            debug_view_builder=self.debug_view_builder,
            debug_enabled=self.debug_enabled,
        )
        self.flow = FrameAssociationFlow(self)

    def _normalize_class_id_to_name(self, class_id_to_name) -> dict[int, str] | None:
        if not isinstance(class_id_to_name, dict):
            return None
        normalized: dict[int, str] = {}
        for raw_class_id, raw_class_name in class_id_to_name.items():
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError):
                continue
            if raw_class_name is None:
                continue
            class_name = str(raw_class_name).strip()
            if not class_name:
                continue
            normalized[int(class_id)] = class_name
        return normalized or None

    def process_frame(
        self,
        detections: list,
        det_features_by_id: dict,
        frame_context=None,
    ) -> FrameAssociationOutput:
        request = AssociationFrameRequest(
            detections=list(detections or []),
            det_features_by_id=dict(det_features_by_id or {}),
            frame_context=frame_context,
            timestamp=self.resolve_frame_timestamp(detections, frame_context),
        )
        return self.flow.run(request)

    def reset_runtime_caches(self) -> None:
        if hasattr(self.outcome_policy, "reset_runtime_caches"):
            self.outcome_policy.reset_runtime_caches()

    def init_frame_summary(self, out: FrameAssociationOutput, detections: list) -> None:
        out.frame_summary["n_detections"] = int(len(detections or []))
        out.frame_summary["n_tracked_total"] = int(len(self.memory_store.all_objects()))

    def finalize_empty_frame(self, *, out: FrameAssociationOutput, timer: ExecutionTimer, frame_context) -> FrameAssociationOutput:
        out.reliable_anchor_object_ids = []
        out.reliable_anchor_det_by_object_id = {}
        out.frame_summary["n_reliable_anchors"] = 0
        out.frame_summary["n_strong"] = 0
        out.frame_summary["n_ambiguous"] = 0
        out.frame_summary["n_weak"] = 0
        out.frame_summary["n_matches"] = 0
        out.frame_summary["n_created"] = 0
        out.frame_summary["n_reports"] = 0
        out.assigned_by_det_id = {}
        out.geom_by_object_id = {}

        self.observability.build_debug_view_if_enabled(
            out=out,
            frame_context=frame_context,
        )

        out.timings_seconds = timer.snapshot_seconds()
        self.observability.finish_frame(frame_context=frame_context)
        return out

    def compute_similarity_reports(
        self,
        *,
        out: FrameAssociationOutput,
        detections: list,
        det_features_by_id: dict,
        frame_context,
        timer: ExecutionTimer,
    ) -> None:
        with timer.measure("sim_candidates"):
            for det in detections or []:
                det_id = getattr(det, "detection_id", None)
                if det_id is None:
                    continue
                det_id = int(det_id)
                det_feats = det_features_by_id.get(det_id, None)
                out.reports_by_det_id[det_id] = self.process_one_detection(det_id, det, det_feats, frame_context)

    def resolve_runtime_flags(self) -> dict:
        dbg_cfg = (self.config.get("debug", {}) or {})
        assoc_dbg = (dbg_cfg.get("association", {}) or {})
        want_debug = bool(dbg_cfg.get("enabled", False)) and bool(assoc_dbg.get("enabled", True))
        want_sets_table = bool(assoc_dbg.get("show_neighbor_sets_table", False))

        match_cfg = ((self.config.get("association", {}) or {}).get("matching", {}) or {})
        ns_influence_cfg = (match_cfg.get("neighbor_sets_influence", {}) or {})
        want_sets_for_matching = bool(ns_influence_cfg.get("enabled", False))

        return {
            "want_debug": bool(want_debug),
            "want_sets_table": bool(want_sets_table),
            "want_sets": bool(want_sets_for_matching) or bool(want_debug and want_sets_table),
        }

    def compute_neighbor_sets_if_needed(
        self,
        *,
        out: FrameAssociationOutput,
        detections: list,
        timestamp: float,
        runtime: dict,
        timer: ExecutionTimer,
    ) -> None:
        self.sets_provider.compute_if_needed(
            out=out,
            detections=detections,
            timestamp=timestamp,
            runtime=runtime,
            timer=timer,
        )

    def annotate_similarity_ambiguity(self, *, out: FrameAssociationOutput, timer: ExecutionTimer) -> None:
        with timer.measure("ambiguity_sim"):
            counts = self.outcome_policy.annotate_similarity_ambiguity(out.reports_by_det_id)
        out.frame_summary["n_strong"] = int(counts.get("strong", 0))
        out.frame_summary["n_ambiguous"] = int(counts.get("ambiguous", 0))
        out.frame_summary["n_weak"] = int(counts.get("weak", 0))

    def run_assignment(
        self,
        *,
        out: FrameAssociationOutput,
        detections: list,
        det_features_by_id: dict,
        timer: ExecutionTimer,
    ) -> AssignmentResult:
        return self.assignment_executor.run(
            out=out,
            detections=detections,
            det_features_by_id=det_features_by_id,
            timer=timer,
        )

    def apply_assignment_results(
        self,
        *,
        out: FrameAssociationOutput,
        detections: list,
        decided_matches: list[tuple[int, int, float]],
        to_create: list[tuple[int, int]],
        timer: ExecutionTimer,
    ) -> None:
        timer.run(
            "post_assignment",
            self.assignment_result_applier.apply,
            out=out,
            detections=detections,
            decided_matches=decided_matches,
            to_create=to_create,
            timer=timer,
        )

    def annotate_final_ambiguity(self, *, out: FrameAssociationOutput, timer: ExecutionTimer) -> None:
        with timer.measure("ambiguity_final"):
            self.outcome_policy.annotate_final_ambiguity(out.reports_by_det_id)

    def annotate_reports_final_decisions(
        self,
        reports_by_det_id: dict,
        assigned_by_det_id: dict[int, int],
        created_det_ids: set[int],
        ambiguous_by_det_id: dict[int, dict],
        provisional_by_det_id: dict[int, dict],
        score_final_by_det_id: dict[int, float],
    ) -> None:
        self.outcome_policy.annotate_reports_final_decisions(
            reports_by_det_id=reports_by_det_id,
            assigned_by_det_id=assigned_by_det_id,
            created_det_ids=created_det_ids,
            ambiguous_by_det_id=ambiguous_by_det_id,
            provisional_by_det_id=provisional_by_det_id,
            score_final_by_det_id=score_final_by_det_id,
        )

    def build_ambiguous_track_candidates(
        self,
        reports_by_det_id: dict,
        decided_matches: list[tuple[int, int, float]],
        to_create: list[dict],
        assigned_by_det_id: dict[int, int],
    ) -> list[dict]:
        return self.outcome_policy.build_ambiguous_track_candidates(
            reports_by_det_id=reports_by_det_id,
            decided_matches=decided_matches,
            to_create=to_create,
            assigned_by_det_id=assigned_by_det_id,
        )

    def build_provisional_new_candidates(
        self,
        *,
        reports_by_det_id: dict,
        to_create: list[dict],
        assigned_by_det_id: dict[int, int] | None = None,
        excluded_det_ids: set[int] | None = None,
    ) -> list[dict]:
        decisions = self.build_postcreate_temporal_decisions(
            reports_by_det_id=reports_by_det_id,
            to_create=to_create,
            assigned_by_det_id=assigned_by_det_id,
            excluded_det_ids=excluded_det_ids,
        )
        return [dict(item) for item in ((decisions or {}).get("provisional_entries", []) or [])]

    def build_postcreate_temporal_decisions(
        self,
        *,
        reports_by_det_id: dict,
        to_create: list[dict],
        assigned_by_det_id: dict[int, int] | None = None,
        excluded_det_ids: set[int] | None = None,
    ) -> dict:
        return self.outcome_policy.build_postcreate_temporal_decisions(
            reports_by_det_id=reports_by_det_id,
            to_create=to_create,
            assigned_by_det_id=assigned_by_det_id,
            excluded_det_ids=excluded_det_ids,
        )

    def amb_track_score(self, candidate: dict) -> float:
        return self.outcome_policy.amb_track_score(candidate)

    def candidate_has_ambiguous_support(self, candidate: dict) -> bool:
        return self.outcome_policy.candidate_has_ambiguous_support(candidate)

    def amb_track_gap_allows(self, top_score: float, cand_score: float) -> bool:
        return self.outcome_policy.amb_track_gap_allows(top_score, cand_score)

    def build_geom_by_object_id(self, decided_matches: list, detections: list) -> dict:
        return self.assignment_result_applier.build_geom_by_object_id(decided_matches, detections)

    def compute_match_diag(self, rep: SimilarityReport, key: str = "score_sim") -> dict | None:
        return self.outcome_policy.compute_match_diag(rep, key=key)

    def compute_confidence(
        self,
        rep: SimilarityReport,
        s1: float,
        s2: float,
        gap: float,
        scores_sorted: list[float],
    ) -> dict:
        return self.outcome_policy.compute_confidence(
            rep=rep,
            s1=s1,
            s2=s2,
            gap=gap,
            scores_sorted=scores_sorted,
        )

    def compute_margin_factor(self, s1: float, gap: float, eps: float) -> tuple[float, float]:
        return self.outcome_policy.compute_margin_factor(s1=s1, gap=gap, eps=eps)

    def sigmoid(self, x: float) -> float:
        return self.outcome_policy.sigmoid(x)

    def compute_coverage_factor(self, rep: SimilarityReport) -> float:
        return self.outcome_policy.compute_coverage_factor(rep)

    def softmax_top2_p1(self, s1: float, s2: float, T: float) -> float:
        return self.outcome_policy.softmax_top2_p1(s1, s2, T)

    def process_one_detection(
        self,
        det_id: int,
        detection,
        det_feats: dict | None,
        frame_context=None,
    ) -> SimilarityReport:
        return self.candidate_generator.process_one_detection(
            det_id=det_id,
            detection=detection,
            det_feats=det_feats,
            frame_context=frame_context,
        )

    def build_similarity_candidate(self, *, det_feats: dict, tracked_object) -> dict:
        return self.candidate_generator.build_similarity_candidate(
            det_feats=det_feats,
            tracked_object=tracked_object,
        )

    def build_reliable_anchor_pairs(self, frame_out: FrameAssociationOutput) -> dict[int, int]:
        return self.reliable_anchor_policy.build_reliable_anchor_pairs(frame_out)

    def build_reliable_anchor_ids(self, frame_out: FrameAssociationOutput) -> set[int]:
        return set(int(x) for x in self.build_reliable_anchor_pairs(frame_out).keys())

    def pick_best(self, candidates: list[dict], key: str) -> dict | None:
        return self.candidate_generator.pick_best(candidates, key=key)

    def pick_second_best(self, candidates: list[dict], best: dict, key: str) -> dict | None:
        return self.candidate_generator.pick_second_best(candidates, best, key=key)

    def resolve_timestamp(self, detection, frame_context) -> float:
        return self.candidate_generator.resolve_timestamp(detection, frame_context)

    def resolve_frame_timestamp(self, detections: list, frame_context) -> float:
        if frame_context is not None and hasattr(frame_context, "timestamp"):
            return float(frame_context.timestamp)
        dets = detections or []
        if dets and hasattr(dets[0], "timestamp"):
            return float(dets[0].timestamp)
        return 0.0
