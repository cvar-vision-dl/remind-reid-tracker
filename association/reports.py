# association/reports.py

from __future__ import annotations


class SimilarityReport:
    """Evidencias para una detección: candidatos, mejor candidato y diagnósticos sim/final."""

    def __init__(self, det_id, class_id: int, timestamp: float):
        self.det_id = int(det_id)
        self.class_id = int(class_id)
        self.timestamp = float(timestamp)

        self.candidates = []
        self.best = None

        self.det_geom = None

        self.match_diag_sim = None
        self.match_diag_final = None

        self.final_object_id = None
        self.final_decision = "UNASSIGNED"
        self.final_score = 0.0
        self.final_reason = "NO_ASSIGNMENT"
        self.ambiguous_candidate_ids = []
        self.ambiguous_candidate_scores = {}
        self.provisional_support_ids = []
        self.provisional_support_scores = {}
        self.provisional_blocked_known_ids = []
        self.provisional_blocked_known_scores = {}
        self.provisional_related_known_ids = []
        self.provisional_related_known_scores = {}


class FrameAssociationOutput:
    """Salida de asociación por frame: reports + neighbor-sets + decisiones."""

    def __init__(self, timestamp: float):
        self.timestamp = float(timestamp)
        self.reports_by_det_id = {}

        self.reliable_anchor_object_ids = []
        self.reliable_anchor_det_by_object_id = {}

        self.neighbor_sets_out = None

        self.geom_by_object_id = {}

        self.decided_matches = []
        self.to_create = []
        self.to_ambiguous = []
        self.to_provisional_new = []
        self.assigned_by_det_id = {}

        self.timings_seconds = {}

        self.frame_summary = {
            "n_detections": 0,
            "n_tracked_total": 0,
            "n_reports": 0,
            "n_reliable_anchors": 0,
            "n_strong": 0,
            "n_ambiguous": 0,
            "n_weak": 0,
            "n_matches": 0,
            "n_created": 0,
            "n_ambiguous_tracks": 0,
            "n_provisional_new": 0,
        }
