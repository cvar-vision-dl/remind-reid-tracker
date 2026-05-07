# Mapa técnico de trazabilidad de `association`

## Objetivo

Traducir el diseño de
`APP2/Src/association_trace/ASSOCIATION_TRACE_VISUALIZATION.md` a un mapa
tecnico directo contra el codigo real del pipeline.

Este documento sirve como blueprint tecnico de la instrumentacion actual. Para
cada nodo visible fija:

- `node_id`
- tipo y scope
- donde se emite la traza
- que parte del pipeline produce el dato de negocio
- datos minimos emitidos
- checks visibles minimos

## Criterio

La tabla no redefine el pipeline. Solo mapea el pipeline real actual.

Cuando un nodo tenga dos capas distintas, se diferencian asi:

- `trace file` / `trace symbol`: donde se emite el `node_run` JSON
- `hook file` / `hook symbol`: donde el pipeline dispara ese nodo de traza
- `support file` / `support symbol`: donde se calcula o prepara la logica de
  negocio que alimenta ese nodo

## Tabla tecnica

### 1. `prepare.class_partition`

- type: `decision`
- scope: `class`
- trace file: `APP2/Src/association/engine/trace_runtime.py`
- trace symbol: `DataAssociationTraceRuntime.trace_class_partition`
- hook file: `APP2/Src/association/engine/observability_runtime.py`
- hook symbol: `DataAssociationObservabilityRuntime.trace_after_similarity_reports`
- support file: `APP2/Src/association/engine/trace_runtime.py`
- support symbol: `DataAssociationTraceRuntime.build_trace_class_specs`
- historical support file: `APP2/Src/association/engine/assignment_path/support.py`
- historical support symbol: `AssignmentPathSupport.partition_detections`
- emite:
  - `class_id`
  - `class_name`
  - `detection_count`
  - `snapshot_object_count`
  - `det_ids`
  - `snapshot_object_ids`
- checks minimos:
  - la clase activa tiene detecciones trazables

### 2. `prepare.reliable_visual_anchors`

- type: `gate`
- scope: `class`
- trace file: `APP2/Src/association/engine/trace_runtime.py`
- trace symbol: `DataAssociationTraceRuntime.trace_reliable_visual_anchors`
- flow file: `APP2/Src/association/flow/frame_association_flow.py`
- flow symbol: `FrameAssociationFlow.select_reliable_visual_anchors`
- hook file: `APP2/Src/association/engine/observability_runtime.py`
- hook symbol: `DataAssociationObservabilityRuntime.trace_after_reliable_anchor_selection`
- support file: `APP2/Src/association/engine/data_association.py`
- support symbol: `DataAssociationEngine.build_reliable_anchor_pairs`
- policy file: `APP2/Src/association/policy/confirmation_policy.py`
- policy symbol: `ReliableVisualAnchorPolicy.build_reliable_anchor_pairs`
- emite:
  - `anchor_object_ids`
  - `anchor_count`
  - `class_id`
  - `class_name`
- checks minimos:
  - existencia de anchors visuales fiables

### 3. `prepare.valid_detections`

- type: `gate`
- scope: `detection`
- trace file: `APP2/Src/association/engine/assignment.py`
- trace symbol: `HungarianAssigner.trace_valid_detections`
- support file: `APP2/Src/association/engine/assignment.py`
- support symbol: `HungarianAssigner.split_class_detection_inputs`
- emite:
  - `detection_count`
  - `valid_det_ids`
  - `valid_count`
  - `missing_feature_det_ids`
  - `missing_feature_count`
- checks minimos:
  - la deteccion tiene features para entrar en matching
  - la deteccion entra en el subconjunto procesable de su clase

### 4. `visual.build_candidates`

- type: `score`
- scope: `candidate`
- trace file: `APP2/Src/association/engine/trace_runtime.py`
- trace symbol: `DataAssociationTraceRuntime.trace_visual_build_candidates`
- hook file: `APP2/Src/association/engine/observability_runtime.py`
- hook symbol: `DataAssociationObservabilityRuntime.trace_after_similarity_reports`
- support file: `APP2/Src/association/engine/data_association.py`
- support symbol: `DataAssociationEngine.compute_similarity_reports`
- implementation file: `APP2/Src/association/engine/candidate_generation.py`
- implementation symbol: `CandidateGenerator.process_one_detection`
- emite:
  - por `det_id`:
    - `candidate_count`
    - `best_object_id`
    - `best_score_sim`
    - `second_object_id`
    - `second_score_sim`
  - por `(det_id, object_id)`:
    - `rank`
    - `score_sim`
    - `score_sim_base`
    - `score_known`
- checks minimos:
  - existencia de candidatos de similitud por deteccion

### 5. `visual.report_diagnosis`

- type: `decision`
- scope: `detection`
- trace file: `APP2/Src/association/engine/trace_runtime.py`
- trace symbol: `DataAssociationTraceRuntime.trace_visual_report_diagnosis`
- hook file: `APP2/Src/association/engine/observability_runtime.py`
- hook symbol: `DataAssociationObservabilityRuntime.trace_after_similarity_diagnosis`
- support file: `APP2/Src/association/engine/data_association.py`
- support symbol: `DataAssociationEngine.annotate_similarity_ambiguity`
- policy file: `APP2/Src/association/policy/outcome_policy.py`
- policy symbol: `AssociationOutcomePolicy.annotate_similarity_ambiguity`
- emite:
  - `match_diag_sim.status`
  - `best_score`
  - `second_score`
  - `gap`
  - `confidence`
- checks minimos:
  - reglas reales que llevan a `STRONG`
  - reglas reales que llevan a `AMBIGUOUS`
  - reglas reales que llevan a `WEAK`

### 6. `context.neighbor_sets_hypotheses`

- type: `score`
- scope: `class`
- trace file: `APP2/Src/association/engine/trace_runtime.py`
- trace symbol: `DataAssociationTraceRuntime.trace_neighbor_sets_hypotheses`
- flow file: `APP2/Src/association/flow/frame_association_flow.py`
- flow symbol: `FrameAssociationFlow.activate_context_layers`
- hook file: `APP2/Src/association/engine/observability_runtime.py`
- hook symbol: `DataAssociationObservabilityRuntime.trace_after_context_activation`
- support file: `APP2/Src/association/context/sets_provider.py`
- support symbol: `SetsContextProvider.compute_if_needed`
- support file: `APP2/Src/association/scores/sets/neighbor_sets_score.py`
- support symbol: `NeighborSetsScore.compute`
- support file: `APP2/Src/association/scores/sets/sets_summary.py`
- support symbol: `SetsSummaryBuilder.build_result`
- emite:
  - `n_hypotheses`
  - `retained_hypotheses`
  - `beam_width`
  - `topk_sets_limit`
  - `context_k`
  - `best_score`
  - `second_score`
  - `gap_best`
  - `coverage_eff_best`
  - `density_best`
  - `mean_maturity_best`
  - `shortlist_size`
  - `anchor_count`
  - `prior_count`
  - `shortlist_object_ids`
  - `anchor_object_ids`
  - `prior_object_ids`
  - filas globales `row_type = hypothesis` con hipótesis relevantes para la clase
  - filas globales `row_type = object_support` con lectura por objeto de la clase:
    - `prior`
    - `support_sum`
    - `maturity_score`
    - `shortlist_hit`
    - `supported_hit`
    - `soft_supported_hit`
    - `coverage_ok`
    - `compat_rel`
    - `kernel_raw`
    - `kernel_hit_count`
    - `kernel_hit_ratio`
    - `kernel_rel`
    - `hyp_rel`
- checks minimos:
  - existe salida computada de `neighbor_sets`
  - existen hipotesis globales utilizables

Nota de lectura:

- `n_hypotheses` / `retained_hypotheses` representan la salida retenida por el
  engine, no un recuento exhaustivo del espacio posible;
- la lectura por objeto sí puede incluir soporte adicional derivado del kernel
  contextual sobre todos los objetos de la clase.

### 7. `context.sets_activation`

- type: `gate`
- scope: `class`
- trace file: `APP2/Src/association/engine/trace_runtime.py`
- trace symbol: `DataAssociationTraceRuntime.trace_sets_activation`
- flow file: `APP2/Src/association/flow/frame_association_flow.py`
- flow symbol: `FrameAssociationFlow.activate_context_layers`
- hook file: `APP2/Src/association/engine/observability_runtime.py`
- hook symbol: `DataAssociationObservabilityRuntime.trace_after_context_activation`
- support file: `APP2/Src/association/context/sets_provider.py`
- support symbol: `SetsContextProvider.compute_if_needed`
- support file: `APP2/Src/association/context/sets_provider.py`
- support symbol: `SetsContextProvider.build_context`
- support file: `APP2/Src/association/context/sets_context_builder.py`
- support symbol: `SetsContextBuilder.build_context`
- emite:
  - `enabled`
  - `global_ok`
  - `reason`
  - `quality`
  - `best`
  - `coverage_eff`
  - `maturity`
  - `density`
  - `k_best`
  - `n_hypotheses`
  - `shortlist_size`
  - `anchor_count`
  - `prior_count`
  - thresholds reales:
    - `min_size_threshold`
    - `best_score_threshold`
    - `coverage_eff_threshold`
    - `quality_threshold`
  - `quality_terms`
- checks minimos:
  - `sets_context_built`
  - `n_hypotheses > 0`
  - `k_best >= min_size`
  - `best >= min_best_score`
  - `coverage_eff >= min_coverage_eff`
  - `quality >= min_quality`
  - gate agregado `global_ok`

Nota de lectura:

- este nodo no rankea objetos ni genera hipótesis nuevas;
- solo decide si el contexto ya construido entra como `active`, `degraded` o
  `inactive`.

### 8. `shape.allow_for_report`

- type: `gate`
- scope: `detection`
- trace file: `APP2/Src/association/engine/assignment.py`
- trace symbol: `HungarianAssigner.trace_allow_for_report`
- support file: `APP2/Src/association/context/neighbor_sets_influence.py`
- support symbol: `NeighborSetsInfluence.allow_for_report`
- emite:
  - `stage` (`pre_locks` o `post_locks`)
  - `report_status`
  - `allowed`
  - `allowed_by_used_object_overlap`
  - `reason`
  - resumen de clase `class_context_available`, `allowed_count`, `blocked_count`, `used_object_ids_count`
- checks minimos:
  - `report_status` habilita o no el uso de contexto
  - override por solape entre candidatos del reporte y `used_object_ids`

### 9. `shape.context_veto`

- type: `gate`
- scope: `candidate`
- trace file: `APP2/Src/association/engine/assignment.py`
- trace symbol: `HungarianAssigner.trace_context_veto`
- support file: `APP2/Src/association/engine/assignment.py`
- support symbol: `HungarianAssigner.candidate_context_veto_reason`
- policy file: `APP2/Src/association/policy/candidate_score_policy.py`
- policy symbol: `CandidateScorePolicy.candidate_context_veto_reason`
- emite:
  - por `(det_id, object_id)`:
    - `known_plausible_keep`
    - `known_plausible_reason`
    - `ctx_keep`
    - `decision_keep`
    - `veto_reason`
    - `gate_reason`
    - `score_sim`
    - `score_sets`
    - `bonus_sets`
    - `support_sets`
    - `support_local_sets`
    - `support_global_sets`
    - `quality_sets`
    - `compat_rel`
    - `compat_band`
    - `kernel_raw`
    - `kernel_hit_count`
    - `kernel_hit_ratio`
    - `kernel_rel`
    - `hyp_rel`
    - `shortlist_hit`
    - `supported_hit`
    - `soft_supported_hit`
    - si existe veto local maduro:
      - `local_ctx_has_supported_alternative`
      - `local_ctx_episode_count`
      - `local_ctx_kernel_source`
      - `local_ctx_kernel_size`
      - `local_ctx_frame_kernel_size`
      - `local_ctx_expected_count`
      - `local_ctx_hit_count`
      - `local_ctx_hit_ratio`
      - `local_ctx_maturity`
    - `score_assign`
    - `score_final`
- checks minimos:
  - el candidato sobrevive o cae tras el veto contextual

### 12. `shape.final_score_tables`

- type: `score`
- scope: `candidate`
- trace file: `APP2/Src/association/engine/assignment.py`
- trace symbol: `HungarianAssigner.trace_final_score_tables`
- support file: `APP2/Src/association/engine/assignment.py`
- support symbol: `HungarianAssigner.build_class_score_tables`
- facade file: `APP2/Src/association/engine/candidate_shaping/score_path.py`
- facade symbol: `CandidateScoreShaper.build_class_score_tables`
- emite:
  - por `det_id`:
    - `candidate_count`
    - `best_object_id`
    - `best_score_final`
  - por `(det_id, object_id)`:
    - `rank`
    - `score_sim`
    - `score_assign`
    - `score_final`
    - `score_sets`
    - `bonus_sets`
    - `score_ctx_local`
    - `score_ctx_global`
    - `gate_reason`
    - `known_plausible_keep`
- checks minimos:
  - existencia de filas finales utilizables para resolucion

### 13. `resolve.locks`

- type: `resolver`
- scope: `global`
- trace file: `APP2/Src/association/engine/assignment.py`
- trace symbol: `HungarianAssigner.trace_locks`
- support file: `APP2/Src/association/engine/assignment.py`
- support symbol: `HungarianAssigner.lock_passes`
- resolver file: `APP2/Src/association/resolver/lock_resolver.py`
- resolver symbol: `LockResolver`
- emite:
  - `candidate_det_count`
  - `candidate_object_count`
  - `locked_count`
  - `locked_det_ids`
  - `locked_object_ids`
  - por `det_id` bloqueado: `lock_source` (`object`, `detection`, `unknown`)
- checks minimos:
  - lock por mejor deteccion
  - lock por mejor objeto
  - thresholds y gaps efectivos del lock

### 14. `resolve.hungarian`

- type: `resolver`
- scope: `global`
- trace file: `APP2/Src/association/engine/assignment.py`
- trace symbol: `HungarianAssigner.trace_hungarian_result`
- resolver file: `APP2/Src/association/resolver/hungarian_resolver.py`
- resolver symbol: `HungarianResolver.resolve`
- emite:
  - `participant_det_ids`
  - `participant_object_ids`
  - `object_column_count`
  - `dummy_column_count`
  - `total_column_count`
  - `n_matches`
  - `n_creates`
  - `create_det_ids`
  - por `det_id`:
    - `assigned_kind`
    - `assigned_column`
    - `assigned_object_id`
    - `selected_score_assign`
    - `selected_score_sim`
    - `selected_score_final`
    - `dummy_score`
    - `passes_match_thr`
    - `passes_min_match_score`
    - `final_action`
    - `reason`
  - por `(det_id, object_id)`:
    - `rank`
    - `selected`
    - `score_assign`
    - `score_sim`
    - `score_final`
  - filas globales `match` y `create`
- checks minimos:
  - asignacion a objeto real o a dummy
  - umbral `match_thr` sobre `score_final`
  - umbral `min_match_score` sobre `score_sim`

### 15. `post.identity_stability`

- type: `gate`
- scope: `detection`
- trace file: `APP2/Src/association/engine/assignment_result_applier.py`
- trace symbol: `AssignmentResultApplier.trace_identity_stability`
- support file: `APP2/Src/association/engine/assignment_result_applier.py`
- support symbol: `AssignmentResultApplier.apply_identity_stability_policy`
- emite:
  - `initial_match_count`
  - `final_match_count`
  - `create_count`
  - `kept_count`
  - `remapped_count`
  - `diverted_count`
- checks minimos:
  - el match inicial se conserva, se remapea o se desvía a create

### 16. `post.assignment_ambiguity`

- type: `decision`
- scope: `global`
- trace file: `APP2/Src/association/engine/assignment_result_applier.py`
- trace symbol: `AssignmentResultApplier.trace_assignment_ambiguity`
- support file: `APP2/Src/association/engine/assignment_result_applier.py`
- support symbols:
  - `AssignmentResultApplier.build_identity_stability_ambiguous_entries`
  - `AssignmentResultApplier.analyze_identity_component_assignments`
  - `AssignmentResultApplier.build_identity_components`
- emite:
  - `component_det_ids`
  - `component_object_ids`
  - `best_assignment`
  - `second_assignment`
  - `assignment_gap`
  - `ambiguity_reason`
- checks minimos:
  - gap entre asignaciones completas
  - desacuerdo real entre asignaciones para alguna deteccion

### 17. `post.known_set_distance_disambiguation`

- type: `resolver`
- scope: `global`
- trace file: `APP2/Src/association/engine/assignment_result_applier.py`
- trace symbol: `AssignmentResultApplier.trace_known_set_distance_disambiguation`
- support file: `APP2/Src/association/disambiguation/known_set_distance_disambiguator.py`
- support symbols:
  - `KnownSetDistanceDisambiguator.resolve`
  - `KnownSetDistanceDisambiguator.debug_pack`
  - `KnownSetDistanceDisambiguator.policy_core_thresholds`
- emite:
  - resumen de clase:
    - `component_count`
    - `pair_anchor_count`
    - `pass_count`
    - `resolved_component_count`
    - `partial_component_count`
  - filas globales `row_type = pass_summary`:
    - `pass_index`
    - `input_det_ids`
    - `resolved_det_ids`
    - `remaining_det_ids`
    - `component_count`
    - `pair_anchor_count`
  - filas globales `row_type = component`:
    - `det_ids`
    - `candidate_union`
    - `status`
    - `reason`
    - `best_score`
    - `core_score`
    - `core_gap`
    - `gap`
    - `pass_index`
    - `pass_input_det_ids`
    - `pass_resolved_det_ids`
    - `pass_remaining_det_ids`
  - filas globales `row_type = pair_anchor`:
    - `det_pair`
    - `anchor_pair`
    - `score`
    - `reason`
    - `pass_index`
- checks minimos:
  - evidencia minima
  - score minimo de asignacion
  - `core_score` minimo segun modo
  - `core_gap` minimo segun modo
  - gap final minimo para resolver

### 18. `post.create_competition`

- type: `decision`
- scope: `global`
- trace file: `APP2/Src/association/engine/assignment_result_applier.py`
- trace symbol: `AssignmentResultApplier.trace_create_competition`
- support file: `APP2/Src/association/engine/assignment_result_applier.py`
- support symbol: `AssignmentResultApplier.build_committed_new_competition_entries`
- emite:
  - `create_entry_count`
  - `competition_count`
  - `selected_competition_count`
  - filas globales de competicion parent/create
- checks minimos:
  - gap permitido entre parent y create
  - competicion realmente seleccionada o no

### 19. `post.ambiguous_track_candidates`

- type: `decision`
- scope: `detection`
- trace file: `APP2/Src/association/engine/assignment_result_applier.py`
- trace symbol: `AssignmentResultApplier.trace_ambiguous_track_candidates`
- support file: `APP2/Src/association/engine/assignment_result_applier.py`
- support symbol: `AssignmentResultApplier.build_post_assignment_ambiguous_entries`
- support policy file: `APP2/Src/association/policy/outcome_policy.py`
- support policy symbol: `AssociationOutcomePolicy.build_ambiguous_track_candidates`
- emite:
  - `candidate_count`
  - `policy_count`
  - `identity_stability_count`
  - `committed_new_count`
  - por `det_id`:
    - `from_policy`
    - `from_identity_stability`
    - `from_committed_new`
    - `selected_source`
    - `candidate_ids`
    - `candidate_scores`
    - `best_score`
    - `score_gap`
    - `reason`
    - `committed_new_object_id`
    - `committed_new_parent_ids`
- checks minimos:
  - la deteccion tiene alguna fuente de ambiguedad
  - la deteccion conserva varias alternativas plausibles

### 20. `post.provisional_reconciliation`

- type: `decision`
- scope: `detection`
- trace file: `APP2/Src/association/engine/assignment_result_applier.py`
- trace symbol: `AssignmentResultApplier.trace_provisional_reconciliation`
- support file: `APP2/Src/association/engine/assignment_result_applier.py`
- support symbol: `AssignmentResultApplier.reconcile_known_ambiguity_and_postcreate`
- support policy file: `APP2/Src/association/policy/outcome_policy.py`
- support policy symbols:
  - `AssociationOutcomePolicy.build_postcreate_temporal_decisions`
  - `AssociationOutcomePolicy.provisional_context_mode`
  - `AssociationOutcomePolicy.provisional_parent_alignment_ok`
- emite:
  - `decision_kind`
  - `final_kind`
  - `reason`
  - `focus_source`
  - `context_mode`
  - `support_mode`
  - `relation`
  - `has_known_context`
  - `visual_fallback_ok`
  - `known_blocked_ok`
  - `status_not_allowed`
  - `provisional_parent_status_ok`
  - `provisional_parent_ok`
  - `support_known_ids`
  - `support_known_scores`
  - `blocked_known_ids`
  - `blocked_known_scores`
  - `related_known_ids`
  - `related_known_scores`
  - `candidate_rows`
- checks minimos:
  - contexto conocido o fallback visual suficiente
  - estado temporal permitido o excepción real
  - reglas reales de provisional parent
  - reglas reales de provisional new
  - alineacion con parent si aplica

### 21. `post.final_decision_pack`

- type: `decision`
- scope: `detection`
- trace file: `APP2/Src/association/engine/assignment_result_applier.py`
- trace symbol: `AssignmentResultApplier.trace_final_decision_pack`
- support file: `APP2/Src/association/engine/assignment_result_applier.py`
- support symbol: `AssignmentResultApplier.build_final_decision_pack`
- support file: `APP2/Src/association/engine/post_assignment/support.py`
- support symbol: `PostAssignmentSupport.build_final_decision_pack`
- emite:
  - resumen de clase:
    - `input_match_count`
    - `input_create_count`
    - `input_ambiguous_count`
    - `input_provisional_count`
    - `final_match_count`
    - `final_create_count`
    - `final_ambiguous_count`
    - `final_provisional_count`
  - por `det_id`:
    - `input_match`
    - `input_create`
    - `input_ambiguous`
    - `input_provisional`
    - `blocked_match`
    - `blocked_create`
    - `final_bucket`
    - `final_object_id`
    - `final_score`
    - `reason`
- checks minimos:
  - la deteccion conserva bucket final
  - un match no es desplazado por una rama de mayor prioridad

### 22. `outcome.final_ambiguity`

- type: `decision`
- scope: `detection`
- trace file: `APP2/Src/association/engine/trace_runtime.py`
- trace symbol: `DataAssociationTraceRuntime.trace_final_ambiguity`
- hook file: `APP2/Src/association/engine/observability_runtime.py`
- hook symbol: `DataAssociationObservabilityRuntime.trace_after_final_ambiguity`
- support file: `APP2/Src/association/engine/data_association.py`
- support symbol: `DataAssociationEngine.annotate_final_ambiguity`
- policy file: `APP2/Src/association/policy/outcome_policy.py`
- policy symbol: `AssociationOutcomePolicy.annotate_final_ambiguity`
- emite:
  - `status`
  - `reason`
  - `s1`
  - `s2`
  - `gap`
  - `n_close`
  - `confidence`
  - `final_decision`
  - `final_reason`
- checks minimos:
  - compuerta final `strong`
  - fallback final `ambiguous`

### 23. `outcome.finalize`

- type: `outcome`
- scope: `detection`
- trace file: `APP2/Src/association/engine/trace_runtime.py`
- trace symbol: `DataAssociationTraceRuntime.trace_final_outcomes`
- support file: `APP2/Src/association/engine/data_association.py`
- support symbol: `DataAssociationEngine.annotate_reports_final_decisions`
- policy file: `APP2/Src/association/policy/outcome_policy.py`
- policy symbol: `AssociationOutcomePolicy.annotate_reports_final_decisions`
- flow file: `APP2/Src/association/flow/frame_association_flow.py`
- flow symbol: `FrameAssociationFlow.finalize_outcomes`
- hook file: `APP2/Src/association/engine/observability_runtime.py`
- hook symbol: `DataAssociationObservabilityRuntime.trace_finalize_outcomes`
- emite:
  - `final_decision`
  - `final_reason`
  - `final_object_id`
  - `final_score`
  - `match_source`
  - `ambiguous_candidate_ids`
  - `ambiguous_candidate_scores`
  - `provisional_support_ids`
  - `provisional_support_scores`
  - `provisional_blocked_known_ids`
  - `provisional_blocked_known_scores`
  - `provisional_related_known_ids`
  - `provisional_related_known_scores`
- checks minimos:
  - no anade gates nuevos; consolida el resultado real del pipeline

## Observaciones de implementacion

### 1. Fuente principal de verdad

La fuente principal debe ser la ejecucion real del pipeline, no una
reconstruccion posterior desde logs de texto.

### 2. Instrumentacion actual

La instrumentacion actual ya:

- puede activarse o desactivarse por flag;
- emite trazas estructuradas JSON;
- usa ids de nodo estables;
- separa `trace_*` del resto de la logica de negocio.

### 3. Relacion con la UI

La UI futura no debe inferir reglas ni recalcular caminos.

Debe limitarse a:

1. cargar `pipeline_schema`;
2. cargar `execution_trace`;
3. pintar nodos, conexiones, checks y caminos por deteccion.
