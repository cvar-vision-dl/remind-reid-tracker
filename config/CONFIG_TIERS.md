# Clasificacion comportamental de parametros de `default_config.yaml`

## Alcance

Este documento clasifica la configuracion completa pensando en una pregunta concreta:

- que parametros conviene estudiar variando sus valores porque cambian de verdad el comportamiento del pipeline;
- que parametros sirven para afinar una politica ya fijada;
- y que parametros son ya heuristica fina o puro soporte operativo.

Los tres tiers son:

- `Esenciales`: merecen estudio temprano porque activan modulos, cambian la politica principal o mueven fronteras de decision de primer orden.
- `Secundarios`: influyen de forma clara, pero tiene sentido afinarlos despues de fijar la arquitectura y la politica principal.
- `Internos`: heuristicas locales, epsilons, caps, gammas y detalles finos; solo conviene tocarlos con una hipotesis muy concreta.

Ademas, se separan unos `Parametros fuera del tiering comportamental`: sirven para ejecutar, depurar, visualizar o reproducir, pero no son buenos candidatos para estudios de comportamiento del pipeline.

La referencia es `REMIND/config/default_config.yaml` y el codigo actual del repositorio.

## Regla de uso

Orden recomendado de estudio:

1. fijar backend, backbone, resolucion y modulos activos;
2. fijar mezcla principal de similitud, matching, ambiguedad y update;
3. afinar memoria, thresholds y reglas relacionales;
4. tocar heuristicas internas solo cuando ya existe un fallo concreto que perseguir.

## Parametros fuera del tiering comportamental

### Ejecucion, reproducibilidad e IO

- `runtime.device`
- `runtime.seed`
- `paths.output_dir`
- `input.frames_dir`
- `memory.start_object_id`

### DAVIS como fuente de datos

- `davis.sequence_name`
- `davis.davis_root`
- `davis.meta_path`
- `davis.annotations_dir`
- `davis.prefetch_annotations`
- `davis.prefetch_distance`

### Timing, paralelismo operativo y depuracion

- `timing.*`
- `update.neighbor_graphs.dist_observations_parallel.*`
- `update.neighbor_graphs.graph_updates_parallel.*`
- `debug.*`
- `association.ambiguous_tracks.known_set_distance_disambiguation.debug_*`

## Parametros esenciales

### Resolucion, detector y backbone

- `system.input_width_size`
- `detector.backend`
- `detector.ignored_classes`
- `yolo.model_label`
- `yolo.models.*`
- `yolo.classes`
- `yolo.conf_th`
- `yolo.iou_th`
- `yolo.max_det`
- `davis.variant`
- `davis.davis_res`
- `davis.classes`
- `dino.model_label`
- `dino.models.*`

### Modulos de percepcion activos

- `perception.full_center_crop`
- `object_features.global.enabled`
- `object_features.global_trimmed.enabled`
- `object_features.patch.enabled`
- `part_descriptors.enabled`
- `part_descriptors.kmeans.enabled`
- `part_descriptors.attention.enabled`
- `bg_local.enabled`
- `bg_local.prototypes.enabled`
- `association.similarity.background_partials.enabled`

### Memoria activa

- `memory.appearance.enabled`
- `memory.parts.enabled`
- `memory.background.enabled`
- `memory.neighbors.enabled`
- `memory.neighbors_distance.enabled`
- `memory.anchor_views.enabled`
- `memory.cross_view_identity.enabled`

### Politica principal de asociacion

- `association.similarity.quality.enabled`
- `association.confirmation.*`
- `association.matching.match_thr`
- `association.matching.clear_margin`
- `association.matching.proto_source_mode`
- `association.matching.object_mode`
- `association.matching.background_mode`
- `association.matching.parts_mode`
- `association.matching.renormalize_missing`
- `association.matching.weights.*`
- `association.matching.hungarian.enable_dummies`
- `association.matching.hungarian.use_confidence_dummy`
- `association.matching.hungarian.gate_by_match_thr`
- `association.matching.hungarian.gate_by_min_match_score`
- `association.matching.hungarian.locks.enabled`
- `association.matching.hungarian.identity_stability.enabled`
- `association.matching.hungarian.committed_new_competition.enabled`
- `association.matching.neighbor_sets_influence.enabled`
- `association.matching.neighbor_sets_context_veto.enabled`
- `association.ambiguity.*`
- `association.ambiguous_tracks.enabled`
- `association.ambiguous_tracks.score_mode`
- `association.ambiguous_tracks.supported_only`
- `association.ambiguous_tracks.require_context`
- `association.ambiguous_tracks.visual_fallback.enabled`
- `association.ambiguous_tracks.known_set_distance_disambiguation.enabled`
- `association.provisional_new.enabled`
- `association.provisional_new.require_context`
- `association.provisional_new.visual_fallback.enabled`

### Neighbor sets como modulo de contexto

- `association.scores.neighbor_sets.topk_sets`
- `association.scores.neighbor_sets.beam_width`
- `association.scores.neighbor_sets.per_class_det_k`
- `association.scores.neighbor_sets.per_class_pool_k`
- `association.scores.neighbor_sets.max_class_options`
- `association.scores.neighbor_sets.allow_partial_coverage`
- `association.scores.neighbor_sets.use_mutual`
- `association.scores.neighbor_sets.kernel_max`
- `association.scores.neighbor_sets.weights.*`
- `association.scores.neighbor_sets.context.enabled`
- `association.scores.neighbor_sets.options.*`

### Lifecycle y update principal

- `update.min_match_score`
- `update.confirm_hits`
- `update.remove_enabled`
- `update.max_misses`
- `update.max_misses_tentative`
- `update.max_misses_confirmed`
- `update.inactive_ttl`
- `update.ambiguous_tracks.enabled`
- `update.provisional_new.enabled`
- `update.robust_updates.enabled`
- `update.appearance_memory.enabled`
- `update.background_memory.enabled`
- `update.parts_memory.enabled`

## Parametros secundarios

### Postproceso de detector y DINO

- `yolo.mask_erosion_px`
- `yolo.mask_erosion_iters`
- `davis.mask_erosion_px`
- `davis.mask_erosion_iters`
- `dino.default_patch_size`
- `dino.normalize_embeddings`
- `dino.patch_selection`
- `dino.patch_threshold`
- `dino.patch_coverage_mode`

### Construccion de descriptores de objeto, partes y fondo

- `object_features.global.weighted`
- `object_features.global_trimmed.weighted`
- `object_features.global_trimmed.keep_frac`
- `object_features.global_trimmed.min_patches`
- `object_features.patch.max_patches`
- `object_features.patch.return_coverage`
- `object_features.patch.l2_normalize`

- `part_descriptors.kmeans.k`
- `part_descriptors.kmeans.weighted`
- `part_descriptors.kmeans.use_trimmed_mean`
- `part_descriptors.kmeans.trimmed_keep_frac`
- `part_descriptors.kmeans.trimmed_min_patches`
- `part_descriptors.kmeans.min_cluster_patches`
- `part_descriptors.kmeans.min_support`
- `part_descriptors.kmeans.merge.*`

- `part_descriptors.attention.head_ids`
- `part_descriptors.attention.weighted`
- `part_descriptors.attention.use_trimmed_mean`
- `part_descriptors.attention.trimmed_keep_frac`
- `part_descriptors.attention.trimmed_min_patches`
- `part_descriptors.attention.max_seeds`
- `part_descriptors.attention.region_frac`
- `part_descriptors.attention.min_region_patches`
- `part_descriptors.attention.max_region_frac`
- `part_descriptors.attention.seed_score`

- `bg_local.inner_radius_patches`
- `bg_local.outer_radius_patches`
- `bg_local.ring_mode`
- `bg_local.obj_patch_min_coverage`
- `bg_local.combine_weights.*`
- `bg_local.sanitize.enabled`
- `bg_local.sanitize.mode`
- `bg_local.sanitize.fill_holes`
- `bg_local.sanitize.keep_largest_component`
- `bg_local.adaptive.*`
- `bg_local.prototypes.k_mode`
- `bg_local.prototypes.patches_per_cluster`
- `bg_local.prototypes.k_min`
- `bg_local.prototypes.k_max`
- `bg_local.prototypes.top_n`

### Capacidades y umbrales de memoria

- `memory.appearance.*`
- `memory.parts.*`
- `memory.background.*`
- `memory.neighbors.*`
- `memory.neighbors_distance.*`
- `memory.anchor_views.*`
- `memory.cross_view_identity.*`

### Similaridad base, calidad y Hungarian

- `association.similarity.parts.topk`
- `association.similarity.background_partials.topk`
- `association.similarity.quality.object.*`
- `association.similarity.quality.parts.*`
- `association.similarity.quality.background.*`

- `association.matching.hungarian.dummy_score`
- `association.matching.hungarian.conf_alpha`
- `association.matching.hungarian.dummy_score_cap`
- `association.matching.hungarian.locks.object_enabled`
- `association.matching.hungarian.locks.det_enabled`
- `association.matching.hungarian.locks.thr`
- `association.matching.hungarian.locks.gap_abs_min`
- `association.matching.hungarian.locks.gap_rel_thr`
- `association.matching.hungarian.identity_stability.alt_margin`
- `association.matching.hungarian.identity_stability.keep_margin`
- `association.matching.hungarian.identity_stability.component_max_size`
- `association.matching.hungarian.identity_stability.assignment_gap_max`
- `association.matching.hungarian.identity_stability.fragile_gate_reasons`
- `association.matching.hungarian.committed_new_competition.*`

### Contexto relacional y veto

- `association.matching.neighbor_sets_influence.*`
- `association.matching.neighbor_sets_context_veto.*`

### Ambiguedad, provisionales y confianza

- `association.ambiguous_tracks.*`
- `association.provisional_new.*`
- `association.confidence.*`

- `association.ambiguous_tracks.known_set_distance_disambiguation.max_passes`
- `association.ambiguous_tracks.known_set_distance_disambiguation.max_group_size`
- `association.ambiguous_tracks.known_set_distance_disambiguation.max_candidate_union`
- `association.ambiguous_tracks.known_set_distance_disambiguation.max_anchors`
- `association.ambiguous_tracks.known_set_distance_disambiguation.discriminative_anchor_topk`
- `association.ambiguous_tracks.known_set_distance_disambiguation.anchor_pair_topk`
- `association.ambiguous_tracks.known_set_distance_disambiguation.selected_anchor_score_ratio_min`
- `association.ambiguous_tracks.known_set_distance_disambiguation.anchor_weight`
- `association.ambiguous_tracks.known_set_distance_disambiguation.visual_weight`
- `association.ambiguous_tracks.known_set_distance_disambiguation.anchor_history_score_weight`
- `association.ambiguous_tracks.known_set_distance_disambiguation.anchor_frame_score_weight`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_total_evidence`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_assignment_score`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_gap`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_core_assignment_score`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_core_gap`
- `association.ambiguous_tracks.known_set_distance_disambiguation.soft_anchors_enabled`
- `association.ambiguous_tracks.known_set_distance_disambiguation.soft_anchor_min_score`
- `association.ambiguous_tracks.known_set_distance_disambiguation.soft_anchor_max`

### Neighbor sets: politica de scoring util de estudiar

- `association.scores.neighbor_sets.max_set_size`
- `association.scores.neighbor_sets.min_set_score`
- `association.scores.neighbor_sets.min_edge_p`
- `association.scores.neighbor_sets.coverage_gamma`
- `association.scores.neighbor_sets.coverage_size_tau`
- `association.scores.neighbor_sets.coverage_size_boost`
- `association.scores.neighbor_sets.coverage_explained_beta`
- `association.scores.neighbor_sets.size.*`
- `association.scores.neighbor_sets.class_terms.ambig_beta`
- `association.scores.neighbor_sets.class_terms.plausible_rel`
- `association.scores.neighbor_sets.class_terms.fill_gamma`
- `association.scores.neighbor_sets.class_terms.maturity_enabled`
- `association.scores.neighbor_sets.class_terms.maturity_softmin_p`
- `association.scores.neighbor_sets.class_terms.maturity_gamma`
- `association.scores.neighbor_sets.core.*`
- `association.scores.neighbor_sets.context.k`
- `association.scores.neighbor_sets.context.min_p`
- `association.scores.neighbor_sets.candidate_pair_matrix.enabled`
- `association.scores.neighbor_sets.density_factor.*`

### Update temporal y politicas de memoria

- `update.ambiguous_tracks.*`
- `update.provisional_new.*`
- `update.temporary_tracks.max_observation_history`
- `update.robust_updates.safe_alpha_scale`
- `update.appearance_memory.*`
- `update.background_memory.*`
- `update.parts_memory.*`

## Parametros internos

### Heuristicas finas de descriptores

- `part_descriptors.kmeans.iters`
- `part_descriptors.kmeans.n_init`
- `part_descriptors.kmeans.seed`
- `part_descriptors.kmeans.return_masks`
- `part_descriptors.attention.return_masks`
- `bg_local.sanitize.close_px`
- `bg_local.sanitize.open_px`
- `bg_local.prototypes.c_sqrt`
- `bg_local.prototypes.min_pts_per_cluster`
- `bg_local.prototypes.merge_sim_thr`
- `bg_local.prototypes.min_mass_frac`
- `bg_local.prototypes.cohesion_power`
- `bg_local.prototypes.proto_mode`

### Neighbor sets: heuristica interna de scoring y busqueda

- `association.scores.neighbor_sets.density_gate_min_edge_p`
- `association.scores.neighbor_sets.density_edge_cov_gamma`
- `association.scores.neighbor_sets.connectivity_require_min_degree`
- `association.scores.neighbor_sets.connectivity_node_gamma`
- `association.scores.neighbor_sets.connectivity_edge_gamma`
- `association.scores.neighbor_sets.exclusivity.*`
- `association.scores.neighbor_sets.class_terms.stability_eps`
- `association.scores.neighbor_sets.class_terms.info_*`
- `association.scores.neighbor_sets.class_terms.excl_*`
- `association.scores.neighbor_sets.conf_lambda`
- `association.scores.neighbor_sets.context.gamma`
- `association.scores.neighbor_sets.context.maturity_tau`
- `association.scores.neighbor_sets.candidate_pair_matrix.max_objects`
- `association.scores.neighbor_sets.beam_state_mode`

### Known-set distance disambiguation fina

- `association.ambiguous_tracks.known_set_distance_disambiguation.discriminative_anchor_min_sep`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_edge_reliability`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_anchor_informativeness`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_singleton_core_assignment_score`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_singleton_core_gap`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_full_sibling_core_assignment_score`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_full_sibling_core_gap`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_partial_same_class_core_assignment_score`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_partial_same_class_core_gap`
- `association.ambiguous_tracks.known_set_distance_disambiguation.gap_sigma`
- `association.ambiguous_tracks.known_set_distance_disambiguation.center_sigma`
- `association.ambiguous_tracks.known_set_distance_disambiguation.rank_sigma`
- `association.ambiguous_tracks.known_set_distance_disambiguation.anchor_span_ref`
- `association.ambiguous_tracks.known_set_distance_disambiguation.order_margin_ref`
- `association.ambiguous_tracks.known_set_distance_disambiguation.distance_score_power`
- `association.ambiguous_tracks.known_set_distance_disambiguation.distance_score_scale`
- `association.ambiguous_tracks.known_set_distance_disambiguation.support_penalty`
- `association.ambiguous_tracks.known_set_distance_disambiguation.rank_weight`
- `association.ambiguous_tracks.known_set_distance_disambiguation.max_component_pair_distance`
- `association.ambiguous_tracks.known_set_distance_disambiguation.min_component_pair_score`
- `association.ambiguous_tracks.known_set_distance_disambiguation.obs_gap_quality_weight`
- `association.ambiguous_tracks.known_set_distance_disambiguation.obs_truncation_penalty`
- `association.ambiguous_tracks.known_set_distance_disambiguation.soft_anchor_conf_weight`
- `association.ambiguous_tracks.known_set_distance_disambiguation.anchor_pair_order_weight`
- `association.ambiguous_tracks.known_set_distance_disambiguation.anchor_pair_margin_weight`
- `association.ambiguous_tracks.known_set_distance_disambiguation.anchor_pair_min_consistency`
- `association.ambiguous_tracks.known_set_distance_disambiguation.anchor_pair_min_margin`
- `association.ambiguous_tracks.known_set_distance_disambiguation.consistency_floor`
- `association.ambiguous_tracks.known_set_distance_disambiguation.consistency_power`

## Lista minima para una primera ronda de estudio

Si se quiere una lista reducida para experimentacion manual, la recomendacion inicial es:

- `detector.backend`
- `detector.ignored_classes`
- `yolo.model_label` o `davis.variant`
- `dino.model_label`
- `system.input_width_size`
- `part_descriptors.enabled`
- `part_descriptors.kmeans.enabled`
- `part_descriptors.attention.enabled`
- `bg_local.enabled`
- `bg_local.prototypes.enabled`
- `memory.neighbors.enabled`
- `memory.neighbors_distance.enabled`
- `memory.cross_view_identity.enabled`
- `association.similarity.quality.enabled`
- `association.confirmation.*`
- `association.matching.match_thr`
- `association.matching.clear_margin`
- `association.matching.proto_source_mode`
- `association.matching.weights.*`
- `association.matching.hungarian.enable_dummies`
- `association.matching.hungarian.locks.enabled`
- `association.matching.hungarian.identity_stability.enabled`
- `association.matching.neighbor_sets_influence.enabled`
- `association.matching.neighbor_sets_context_veto.enabled`
- `association.scores.neighbor_sets.topk_sets`
- `association.scores.neighbor_sets.beam_width`
- `association.scores.neighbor_sets.weights.*`
- `association.ambiguous_tracks.enabled`
- `association.provisional_new.enabled`
- `update.min_match_score`
- `update.ambiguous_tracks.enabled`
- `update.provisional_new.enabled`
- `update.robust_updates.enabled`
- `update.appearance_memory.enabled`
- `update.background_memory.enabled`
- `update.parts_memory.enabled`
