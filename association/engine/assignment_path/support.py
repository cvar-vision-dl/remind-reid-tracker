from __future__ import annotations

from association.engine.assignment_path.contracts import AssignmentPartition, PreparedAssignmentInputs


class AssignmentPathSupport:
    """Infraestructura ligera para preparar y recorrer el camino de asignacion."""

    def partition_detections(self, detections: list) -> AssignmentPartition:
        by_class: dict[int, list[int]] = {}
        detections_by_id: dict[int, object] = {}

        for detection in (detections or []):
            det_id = getattr(detection, "detection_id", None)
            if det_id is None:
                continue
            det_id = int(det_id)
            detections_by_id[int(det_id)] = detection
            class_id = int(getattr(detection, "class_id", -1))
            by_class.setdefault(class_id, []).append(det_id)

        return AssignmentPartition(
            detections_by_id=dict(detections_by_id),
            det_ids_by_class={int(cid): [int(did) for did in det_ids] for cid, det_ids in by_class.items()},
        )

    def prepare_assignment_inputs(
        self,
        *,
        assigner,
        by_class: dict[int, list[int]],
        snapshot_ids: set[int],
        association_output,
        use_neighbor_sets: bool,
        neighbor_sets_influence,
        ns_ctx_override: dict | None,
        match_thr: float,
        timer=None,
        timer_prefix: str = "",
    ) -> PreparedAssignmentInputs:
        snap = {int(x) for x in (snapshot_ids or set())}
        step = (lambda suffix: f"{timer_prefix}{suffix}" if timer_prefix else str(suffix))
        if timer is not None:
            context = timer.run(
                step("context"),
                assigner.resolve_assignment_context,
                association_output=association_output,
                use_neighbor_sets=use_neighbor_sets,
                neighbor_sets_influence=neighbor_sets_influence,
                ns_ctx_override=ns_ctx_override,
            )
        else:
            context = assigner.resolve_assignment_context(
                association_output=association_output,
                use_neighbor_sets=use_neighbor_sets,
                neighbor_sets_influence=neighbor_sets_influence,
                ns_ctx_override=ns_ctx_override,
            )
        prepared = PreparedAssignmentInputs()
        prepared.snapshot_ids = set(snap)
        prepared.context = context
        return prepared
