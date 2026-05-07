from __future__ import annotations

from copy import deepcopy


PIPELINE_SCHEMA_V1 = {
    "schema_version": "1.0",
    "module": "association",
    "orientation": "top_down",
    "unit_of_analysis": "frame_class",
    "phases": [
        {"id": "prepare", "label": "Prepare", "order": 10},
        {"id": "visual", "label": "Visual Evidence", "order": 20},
        {"id": "context", "label": "Context Activation", "order": 30},
        {"id": "candidate_shaping", "label": "Candidate Shaping", "order": 40},
        {"id": "resolution", "label": "Global Resolution", "order": 50},
        {"id": "post_assignment", "label": "Post Assignment", "order": 60},
        {"id": "outcome", "label": "Final Outcome", "order": 70},
    ],
    "nodes": [
        {"id": "prepare.class_partition", "phase": "prepare", "type": "decision", "scope": "class", "order": 10},
        {"id": "visual.build_candidates", "phase": "visual", "type": "score", "scope": "candidate", "order": 10},
        {"id": "prepare.reliable_visual_anchors", "phase": "visual", "type": "gate", "scope": "class", "order": 20},
        {"id": "context.neighbor_sets_hypotheses", "phase": "context", "type": "score", "scope": "class", "order": 10},
        {"id": "context.sets_activation", "phase": "context", "type": "gate", "scope": "class", "order": 20},
        {"id": "visual.report_diagnosis", "phase": "context", "type": "decision", "scope": "detection", "order": 30},
        {"id": "prepare.valid_detections", "phase": "candidate_shaping", "type": "gate", "scope": "detection", "order": 10},
        {"id": "shape.allow_for_report", "phase": "candidate_shaping", "type": "gate", "scope": "detection", "order": 20},
        {"id": "shape.context_veto", "phase": "candidate_shaping", "type": "gate", "scope": "candidate", "order": 30},
        {"id": "shape.final_score_tables", "phase": "candidate_shaping", "type": "score", "scope": "candidate", "order": 40},
        {"id": "resolve.locks", "phase": "resolution", "type": "resolver", "scope": "global", "order": 10},
        {"id": "resolve.hungarian", "phase": "resolution", "type": "resolver", "scope": "global", "order": 20},
        {"id": "post.assignment_ambiguity", "phase": "post_assignment", "type": "decision", "scope": "global", "order": 10},
        {"id": "post.identity_stability", "phase": "post_assignment", "type": "gate", "scope": "detection", "order": 20},
        {"id": "post.create_competition", "phase": "post_assignment", "type": "decision", "scope": "global", "order": 30},
        {
            "id": "post.ambiguous_track_candidates",
            "phase": "post_assignment",
            "type": "decision",
            "scope": "detection",
            "order": 35,
        },
        {
            "id": "post.known_set_distance_disambiguation",
            "phase": "post_assignment",
            "type": "resolver",
            "scope": "global",
            "order": 40,
        },
        {
            "id": "post.provisional_reconciliation",
            "phase": "post_assignment",
            "type": "decision",
            "scope": "detection",
            "order": 50,
        },
        {
            "id": "post.final_decision_pack",
            "phase": "post_assignment",
            "type": "decision",
            "scope": "detection",
            "order": 60,
        },
        {"id": "outcome.final_ambiguity", "phase": "outcome", "type": "decision", "scope": "detection", "order": 10},
        {"id": "outcome.finalize", "phase": "outcome", "type": "outcome", "scope": "detection", "order": 20},
    ],
    "edges": [
        {"from": "prepare.class_partition", "to": "visual.build_candidates"},
        {"from": "visual.build_candidates", "to": "prepare.reliable_visual_anchors"},
        {"from": "prepare.reliable_visual_anchors", "to": "context.neighbor_sets_hypotheses"},
        {"from": "context.neighbor_sets_hypotheses", "to": "context.sets_activation"},
        {"from": "context.sets_activation", "to": "visual.report_diagnosis"},
        {"from": "visual.report_diagnosis", "to": "prepare.valid_detections"},
        {"from": "prepare.valid_detections", "to": "shape.allow_for_report"},
        {"from": "shape.allow_for_report", "to": "shape.context_veto"},
        {"from": "shape.context_veto", "to": "shape.final_score_tables"},
        {"from": "shape.final_score_tables", "to": "resolve.locks"},
        {"from": "resolve.locks", "to": "resolve.hungarian"},
        {"from": "resolve.hungarian", "to": "post.assignment_ambiguity"},
        {"from": "post.assignment_ambiguity", "to": "post.identity_stability"},
        {"from": "post.identity_stability", "to": "post.create_competition"},
        {"from": "post.create_competition", "to": "post.ambiguous_track_candidates"},
        {"from": "post.ambiguous_track_candidates", "to": "post.known_set_distance_disambiguation"},
        {"from": "post.known_set_distance_disambiguation", "to": "post.provisional_reconciliation"},
        {"from": "post.provisional_reconciliation", "to": "post.final_decision_pack"},
        {"from": "post.final_decision_pack", "to": "outcome.final_ambiguity"},
        {"from": "outcome.final_ambiguity", "to": "outcome.finalize"},
    ],
}


def build_node_order_map(schema: dict | None = None) -> dict[str, tuple[int, int, str]]:
    active_schema = PIPELINE_SCHEMA_V1 if schema is None else schema
    phase_order_by_id = {
        str(phase.get("id")): int(phase.get("order", 0) or 0)
        for phase in (active_schema.get("phases", []) or [])
        if isinstance(phase, dict)
    }
    return {
        str(node.get("id")): (
            int(phase_order_by_id.get(str(node.get("phase")), 0)),
            int(node.get("order", 0) or 0),
            str(node.get("id")),
        )
        for node in (active_schema.get("nodes", []) or [])
        if isinstance(node, dict) and node.get("id") is not None
    }


def build_pipeline_schema() -> dict:
    return deepcopy(PIPELINE_SCHEMA_V1)
