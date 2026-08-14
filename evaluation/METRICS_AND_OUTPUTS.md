# Testing Metrics And Outputs

Este documento resume que metricas y que informacion guarda el pipeline de evaluacion en `REMIND/testing`, que significa cada bloque y como interpretar los CSV resultantes.

La idea es distinguir dos cosas:

- Metricas primitivas o decisiones internas que conviene guardar inline porque luego no se pueden reconstruir bien.
- Agregaciones o analisis secundarios que se pueden calcular offline a partir de los CSV.

## 1. Principios de interpretacion

- La unidad basica de evaluacion es el `caso`, es decir, una observacion GT visible en un frame concreto.
- `n_cases` suele equivaler al numero de observaciones GT visibles evaluadas. Incluye tambien casos sin deteccion si el GT estaba visible.
- En el batch, todo queda ligado a `run_id`, `scene_id` y `scene_name`.
- Las metricas `strict` y `permissive` pertenecen al analisis base de asignaciones GT<->pred y no son exactamente lo mismo que las metricas `collapsed_*`. Se calculan sobre todos los GT visibles: si un caso visible no tiene asignacion firme correcta, penaliza.
- `tracking_iou` se contabiliza como la IoU del caso solo cuando la decision colapsada final es correcta. Si no, vale `0.0`.
- `distance_correct` significa que el modulo de distancia participo en la resolucion y que el resultado colapsado final del caso fue correcto. No mide solo "si el modulo de distancia propuso algo razonable".
- `context_change_correct` significa que hubo intervencion del contexto o neighbor sets y que el resultado final colapsado fue correcto.
- `HOTA` aqui se usa como aproximacion derivada de `DetA` y `AssA`:
  `HOTA = sqrt(DetA * AssA)`.
  No pretende sustituir una implementacion oficial completa de benchmark MOT.
- La asociacion deteccion-GT usada para evaluar observaciones visibles es 1 a 1 por maximizacion de IoU y, cuando hay clase en ambos lados, mantiene consistencia de clase. Solo se consideran candidatos con `IoU > 0`.
- En el evaluador generico de tracking puro tambien se guardan variantes adicionales con sufijo `_iou40`. Estas no sustituyen a las metricas oficiales actuales: reutilizan la misma asociacion GT-pred, pero solo cuentan como match/caso correcto aquellos con `IoU >= 0.40`.
- `MT/PT/ML` se calcula por objeto GT usando `tracking_recall_object`:
  `MT >= 0.80`, `ML <= 0.20`, el resto `PT`.

## 2. Ficheros generados

### 2.1. Salidas por escena

En cada escena del batch se guardan normalmente estos ficheros:

- `scene_summary.csv`
- `per_class.csv`
- `per_object.csv`
- `per_case.csv`
- `per_case_modules.csv`
- `per_frame.csv`
- `per_pred_track.csv`
- `per_event.csv`
- `report.txt`

### 2.2. Salidas globales del batch

En la carpeta del batch se reconstruyen:

- `summary_global.csv`
- `per_scene.csv`
- `per_class.csv`
- `per_object.csv`
- `per_case.csv`
- `per_case_modules.csv`
- `per_frame.csv`
- `per_pred_track.csv`
- `per_event.csv`
- `manifest.csv`
- `run_config.csv`
- `report.txt`

## 3. Identificadores y campos comunes

- `run_id`: identificador estable del batch.
- `scene_id`: identificador de escena. Es el campo mas importante para filtrar y agrupar.
- `scene_name`: nombre legible de escena. En este flujo suele coincidir con `scene_id`.
- `frame_id`: frame evaluado.
- `gt_instance_id`: id numerico del objeto GT.
- `gt_label`: etiqueta textual del GT.
- `gt_class_name`: clase del objeto GT.
- `pred_object_id`: id interno del track o identidad predicha.
- `pred_instance_label`: etiqueta textual del track predicho, si existe.

## 4. Bloques de metricas principales

## 4.1. Collapsed metrics

Estas metricas resumen si el sistema acierta al decidir si un caso corresponde a un objeto existente, a uno nuevo o a una hipotesis incierta.

- `n_cases`: numero total de casos evaluados.
- `n_existing_gt`: numero de casos cuyo GT era un objeto ya existente.
- `n_new_gt`: numero de casos cuyo GT aparecia como nuevo.
- `n_ambiguous_cases`: numero de casos resueltos como ambiguos.
- `accuracy_global_collapsed`: porcentaje de casos correctos tras colapsar la salida final a `existing` o `new`.
- `accuracy_existing_vs_new_collapsed`: porcentaje de casos en los que se acierta al distinguir entre objeto existente y objeto nuevo, aunque no siempre se acierte la identidad exacta.
- `accuracy_parent_collapsed`: en casos GT existentes, porcentaje en que el parent o identidad de referencia es correcto.
- `set_accuracy_ambiguous`: en casos ambiguos, porcentaje en que el GT correcto estaba dentro del conjunto ambiguo propuesto.
- `new_detection_accuracy_collapsed`: en GT realmente nuevos, porcentaje en que el sistema los marca finalmente como `new`.

## 4.2. Collapsed identity metrics

Estas metricas miran identidad y continuidad temporal una vez colapsadas las decisiones.

- `n_gt_observations`: numero de observaciones GT visibles.
- `n_matched_gt_observations`: observaciones GT visibles que tuvieron una deteccion con `IoU > 0`.
- `n_pred_observations`: numero de observaciones predichas visibles en los frames evaluados.
- `n_unique_gt_ids`: numero de objetos GT unicos.
- `n_unique_existing_pred_ids`: numero de tracks usados como `existing`.
- `n_unique_new_pred_ids`: numero de tracks usados como `new`.
- `n_unique_pred_ids`: numero total de tracks unicos usados.
- `idtp`: identity true positives.
- `idfp`: identity false positives.
- `idfn`: identity false negatives.
- `idp`: precision de identidad interna sobre la representacion colapsada, `idtp / (idtp + idfp)`.
- `idr`: recall de identidad interna sobre la representacion colapsada, `idtp / (idtp + idfn)`.
- `idf1`: F1 de identidad interna sobre la representacion colapsada, `2 * idtp / (2 * idtp + idfp + idfn)`.
- `idsw`: numero interno de identity switches en la secuencia colapsada.
- `frag`: numero interno de fragmentaciones en la secuencia colapsada.
- `tracking_recall`: porcentaje de casos visibles en los que la salida colapsada final fue correcta.
- `mean_tracking_iou`: media de `tracking_iou` sobre todos los casos visibles.
- `deta`: aproximacion de Detection Accuracy, `n_matched_gt_observations / n_visible_gt_observations`.
- `assa`: aproximacion de Association Accuracy, `n_tracking_correct / n_matched_gt_observations`.
- `hota`: `sqrt(deta * assa)`.

## 4.3. Uncertainty metrics

Estas metricas describen cuanto decide el sistema de forma firme y cuanto deja en estados inciertos.

- `n_firm`: casos con decision firme, normalmente `MATCH` o `NEW`.
- `n_ambiguous`: casos `AMBIGUOUS_TRACK`.
- `n_provisional_parent`: casos `PROVISIONAL_PARENT`.
- `n_provisional_new`: casos `PROVISIONAL_NEW`.
- `coverage_firm`: proporcion de casos con decision firme.
- `firm_accuracy`: exactitud dentro de los casos firmes.
- `firm_error_rate_over_all_cases`: tasa de errores firmes sobre el total de casos.
- `ambiguity_rate`: tasa de casos ambiguos.
- `provisional_parent_rate`: tasa de casos `PROVISIONAL_PARENT`.
- `provisional_new_rate`: tasa de casos `PROVISIONAL_NEW`.
- `uncertain_rate`: tasa total de decisiones no firmes.
- `parent_hit_rate_provisional`: en `PROVISIONAL_PARENT`, proporcion en que el parent correcto estaba dentro del conjunto provisional propuesto.
- `new_detection_accuracy_uncertain`: en GT nuevos, proporcion en que el sistema al menos detecta novedad de forma incierta o firme.
- `avg_ambiguous_candidates`: media del tamano de conjuntos ambiguos.
- `max_ambiguous_candidates`: maximo tamano de conjunto ambiguo.
- `avg_provisional_parent_candidates`: media de candidatos parent provisionales.
- `max_provisional_parent_candidates`: maximo de candidatos parent provisionales.
- `hypothesis_recall_uncertain`: capacidad de las salidas inciertas para contener la respuesta correcta. Incluye acierto por conjunto ambiguo, acierto dentro del conjunto de parents provisional y deteccion correcta de novedad en `PROVISIONAL_NEW`.

## 4.4. Summary o metricas auxiliares globales

Estas metricas ayudan a entender estabilidad temporal, inflacion de tracks y fallos tipicos.

- `n_frames`: numero total de frames procesados en la escena o agregado.
- `n_objects`: numero total de objetos GT de la escena o agregado.
- `n_assignments`: numero de asignaciones GT-pred directas del evaluador base.
- `n_unique_real_pred_tracks`: numero de tracks reales usados por el sistema.
- `pred_track_surplus_vs_gt`: diferencia entre tracks reales y objetos GT.
- `pred_track_inflation_factor`: `n_unique_real_pred_tracks / n_objects`.
- `n_existing_gt_reopened_as_new_rows`: numero de filas donde un GT existente reaparece como `new`.
- `n_existing_gt_reopened_as_new_ids`: numero de objetos GT afectados por reapertura como nuevo.
- `reopen_rate_existing`: proporcion de casos GT existentes reabiertos como `new`.
- `gt_with_reopen_rate`: proporcion de objetos GT que sufrieron al menos una reapertura.
- `global_frame_accuracy_strict`: exactitud frame a frame usando solo el track de referencia global.
- `global_frame_accuracy_permissive`: exactitud frame a frame aceptando cualquier track canonico del mismo GT.
- `global_object_accuracy_strict`: proporcion de objetos GT perfectos en modo estricto.
- `global_object_accuracy_permissive`: proporcion de objetos GT perfectos en modo permisivo.
- `objects_fragmented`: numero de GT que acabaron repartidos en varios tracks propios.
- `objects_with_foreign_id_use`: numero de GT que en algun momento usaron un track canonico de otro objeto.
- `id_changes_total`: numero total de cambios de id.
- `objects_recovered_reference`: GT que recuperan mas tarde su track de referencia.
- `objects_recovered_own_identity`: GT que recuperan al menos un track propio aunque no sea el de referencia.
- `stable_foreign_segments_total`: segmentos estables usando ids ajenos.
- `stable_own_new_segments_total`: segmentos estables usando un id propio nuevo, distinto del de referencia.
- `swap_events_total`: numero de swaps detectados.
- `theft_with_new_id_total`: robos de id donde la victima pasa a un id nuevo.
- `theft_with_displacement_total`: robos de id donde la victima se desplaza a otro id ya existente.
- `distance_used_count`: numero de casos donde el modulo de distancia intervino.
- `distance_resolved_count`: numero de casos resueltos por distancia.
- `distance_correct_count`: numero de casos resueltos por distancia y correctos al final.
- `distance_usage_rate`: tasa de uso de distancia.
- `distance_resolution_rate`: proporcion de usos de distancia que terminaron en resolucion efectiva.
- `distance_disambiguation_accuracy`: exactitud de los casos resueltos por distancia.
- `distance_unresolved_rate`: fraccion de usos de distancia que no terminaron resolviendo.
- `neighbor_sets_available_count`: casos donde habia neighbor sets disponibles.
- `neighbor_sets_available_rate`: tasa de disponibilidad de neighbor sets.
- `context_intervened_count`: casos donde el contexto o los neighbor sets cambiaron la mejor opcion.
- `context_correct_count`: intervenciones de contexto que acabaron correctas.
- `context_rescue_count`: casos donde el contexto rescato la opcion final.
- `context_veto_case_count`: casos con al menos un candidato vetado por contexto.
- `context_intervention_rate`: tasa de intervencion del contexto.
- `context_intervention_accuracy`: exactitud de las intervenciones del contexto.
- `context_rescue_rate`: tasa de rescates de contexto.
- `context_veto_rate`: tasa de vetos de contexto.
- `context_net_gain`: medida neta de ganancia del contexto, premiando intervenciones correctas y penalizando incorrectas.
- `total_runtime_seconds`: tiempo total de ejecucion.
- `avg_runtime_seconds`: tiempo medio por escena o por unidad agregada.
- `total_loop_ms`: tiempo total de bucle.
- `avg_loop_ms`: tiempo medio por frame.

## 5. CSV detallados

## 5.1. `per_case.csv`

Es el CSV mas importante para analisis fino. Cada fila representa un caso GT visible en un frame.

### Identificacion basica

- `run_id`, `scene_id`, `scene_name`
- `frame_id`
- `gt_instance_id`
- `gt_label`
- `gt_class_name`
- `det_id`: id de la deteccion asociada. Vale `-1` si el GT estaba visible pero no hubo deteccion asignada.
- `iou`: IoU entre la deteccion y el GT.

### Estado real y decision del sistema

- `real_state`: `new` o `existing`.
- `final_decision`: decision original del pipeline. Valores habituales:
  `MATCH`, `NEW`, `AMBIGUOUS_TRACK`, `PROVISIONAL_PARENT`, `PROVISIONAL_NEW`, `NO_DETECTION`.
- `final_reason`: motivo textual resumido.
- `firm_kind`: version simplificada de la decision firme.
- `firm_pred_object_id`: id firme si la decision fue `MATCH`.
- `ambiguous_candidate_ids`: ids candidatos en una salida ambigua.
- `provisional_parent_ids`: parent ids candidatos en provisional.
- `provisional_temp_id`: id temporal provisional.
- `created_object_id`: id del nuevo objeto creado.
- `created_origin_provisional_temp_id`: id temporal del que provino el objeto creado.

### Decision colapsada para evaluar

- `collapsed_kind`: salida colapsada a `existing`, `new` u otras variantes internas.
- `collapsed_pred_object_id`: id colapsado final.
- `collapsed_pred_class_name`: clase asociada al id colapsado.
- `collapsed_existing_vs_new_correct`: si acierta solo el tipo `existing/new`.
- `collapsed_parent_correct`: si acierta el parent correcto en GT existentes.
- `collapsed_global_correct`: si el caso queda correctamente resuelto en sentido global.
- `firm_global_correct`: exactitud considerando solo salidas firmes.
- `ambiguous_set_hit`: si el GT correcto estaba dentro del conjunto ambiguo.
- `provisional_parent_hit`: si el parent correcto estaba dentro del conjunto de parents provisional.
- `novelty_detected_uncertain`: si en un GT nuevo se detecto novedad aunque fuese de forma incierta.

### Dificultad y contexto visual

- `gt_area_px`: area del GT en pixeles.
- `frame_gt_count`: numero total de GT visibles en ese frame.
- `frame_same_class_count`: numero de GT visibles de la misma clase.
- `frame_area_px`: area total del frame.
- `gt_area_frac`: area relativa del GT sobre el frame.
- `n_gt_visible_in_frame`: alias practico de `frame_gt_count`.
- `n_gt_same_class_in_frame`: alias practico de `frame_same_class_count`.
- `n_total_distractors`: `frame_gt_count - 1`.
- `n_same_class_distractors`: `frame_same_class_count - 1`.
- `gt_is_new_this_frame`: si el GT aparece por primera vez en ese frame.
- `gt_age_frames`: edad del GT en frames desde su primera aparicion.

### Tracking por caso

- `is_tracked_visible_case`: si hay una salida colapsada visible usable para tracking.
- `tracking_iou`: IoU del caso si el resultado colapsado es correcto; si no, `0.0`.

### Telemetria de modulos internos

- `best_sim_object_id`: mejor candidato segun similitud pura.
- `best_sim_score`: score de similitud pura del mejor candidato.
- `best_final_candidate_object_id`: mejor candidato final tras aplicar politicas y contexto.
- `best_final_candidate_score`: score final del mejor candidato.
- `best_sim_margin`: diferencia entre el mejor y el segundo mejor score de similitud.
- `best_final_margin`: diferencia entre el mejor y el segundo mejor score final.
- `match_source`: origen de la asignacion final, por ejemplo asociacion normal o distancia.
- `distance_used`: si el modulo de distancia participo.
- `distance_resolved`: si la decision final vino de una resolucion por distancia.
- `distance_correct`: si esa resolucion por distancia acabo siendo correcta.
- `neighbor_sets_available`: si habia neighbor sets disponibles.
- `context_intervened`: si el contexto cambio la opcion preferida respecto a la similitud pura.
- `context_change_correct`: si ese cambio contextual fue correcto.
- `context_rescue_applied`: si el contexto rescato un candidato final.
- `context_veto_candidate_count`: cuantos candidatos fueron vetados por contexto.
- `selected_candidate_score_sets`: score asociado a sets del candidato final.
- `selected_candidate_quality_sets`: calidad del set del candidato final.

## 5.2. `per_case_modules.csv`

Es un subconjunto de `per_case.csv` centrado en diagnostico interno de modulos.

- Repite identificadores y decision final.
- Resume solo las variables mas utiles para distancia, contexto, neighbor sets y ranking de candidatos.
- Sirve para estudiar modulos sin cargar todo el detalle del caso.

## 5.3. `per_object.csv`

Cada fila representa un objeto GT agregado a lo largo del tiempo.

### Identidad y cobertura temporal

- `gt_instance_id`, `gt_label`, `gt_class_name`
- `n_frames`: numero de frames visibles del GT.
- `assigned_n_frames`: frames con asignacion base.
- `visible_n_frames`: frames en que el GT estuvo visible segun `per_case`.
- `first_frame`, `last_frame`
- `reference_pred_id`: track de referencia global asignado al GT.
- `reference_pred_label`

### Exactitud y estabilidad

- `strict_accuracy`: exactitud del GT usando solo el track de referencia, medida sobre todos sus frames visibles.
- `permissive_accuracy`: exactitud aceptando cualquier track propio canonico, medida sobre todos sus frames visibles.
- `perfect_strict`: si el GT fue perfecto en estricto.
- `perfect_permissive`: si el GT fue perfecto en permisivo.
- `tracking_recall_object`: recall temporal del objeto.
- `mean_tracking_iou_object`: media de `tracking_iou` para ese objeto.
- `mt_pt_ml_label`: clasificacion `MT`, `PT` o `ML`.

### Multiplicidad de ids

- `n_unique_pred_ids`: cuantos ids distintos uso ese GT.
- `n_own_pred_ids`: ids propios usados por el GT.
- `n_foreign_pred_ids`: ids ajenos usados por el GT.
- `id_changes`: cambios de id por segmentos de trayectoria.
- `idsw_object`: switches de id a nivel objeto usando la secuencia colapsada.
- `frag_object`: fragmentacion del objeto por interrupciones o cambios de track.

### Recuperacion y segmentos

- `stable_foreign_segments`
- `stable_own_new_segments`
- `first_failure_frame`
- `recovered_reference`
- `recovered_own_identity`
- `post_failure_strict_accuracy`
- `segments`: lista estructurada con segmentos consecutivos.
- `pred_ids_timeline`: ids usados por frame.
- `frames_timeline`: frames del timeline.

### Dificultad media del objeto

- `mean_visible_gt_in_frame`
- `mean_total_distractors`
- `mean_same_class_distractors`
- `mean_gt_area_frac`

### Incertidumbre y modulos para ese objeto

- `n_ambiguous_cases`
- `n_provisional_parent_cases`
- `n_provisional_new_cases`
- `distance_used_count`
- `distance_correct_count`
- `context_intervened_count`
- `context_correct_count`

## 5.4. `per_frame.csv`

Cada fila resume un frame.

- `frame_id`
- `n_objects`: objetos GT visibles en ese frame.
- `strict_accuracy`: exactitud estricta sobre todos los GT visibles en ese frame.
- `permissive_accuracy`: exactitud permisiva sobre todos los GT visibles en ese frame.
- `strict_correct`
- `permissive_correct`
- `visible_n_objects`: GT visibles en el frame segun `per_case`. En la implementacion actual coincide con `n_objects`.
- `n_classes_visible`: numero de clases visibles.
- `tracking_recall_frame`: proporcion de casos correctos en ese frame.
- `mean_tracking_iou_frame`
- `n_new_gt`
- `n_existing_gt`
- `n_firm`
- `n_ambiguous`
- `n_provisional_parent`
- `n_provisional_new`
- `n_distance_used`
- `n_context_interventions`
- `read_ms`, `pipeline_ms`, `gt_ms`, `eval_ms`, `post_ms`, `loop_ms`: tiempos por frame cuando vienen del batch.

## 5.5. `per_class.csv`

Cada fila resume una clase GT dentro de una escena.

Es decir, la unidad correcta aqui es `scene_id + class_name`, no una clase global agregada sobre todo el batch.

Esto permite:

- agregar despues offline a nivel global por clase;
- recuperar comparativas por escena;
- cruzar clase con tipologia manual de escena sin perder informacion.

- `class_name`
- `n_gt_objects`
- `n_real_pred_tracks`
- `pred_track_surplus_vs_gt`
- `pred_track_inflation_factor`
- `weighted_strict_accuracy`
- `weighted_permissive_accuracy`
- `mean_pred_ids_per_gt`
- `mean_id_changes_per_gt`
- `tracking_recall`
- `mean_tracking_iou`
- `deta`
- `assa`
- `hota`
- `accuracy_existing_vs_new_collapsed`
- `accuracy_parent_collapsed`
- `new_detection_accuracy_collapsed`
- `uncertain_rate`
- `hypothesis_recall_uncertain`
- `reopen_rate_existing`
- `distance_usage_rate`
- `distance_disambiguation_accuracy`
- `context_intervention_rate`
- `context_intervention_accuracy`
- `gt_objects_with_foreign_id_use`
- `existing_gt_reopened_as_new_rows`
- `existing_gt_reopened_as_new_ids`

## 5.6. `per_pred_track.csv`

Cada fila representa un track predicho real.

- `pred_object_id`
- `pred_instance_label`
- `pred_class_name`
- `canonical_gt`: GT canonico asociado a ese track.
- `majority_gt`: GT mayoritario por uso.
- `reference_gt`: GT del que ese track es referencia global, si aplica.
- `gt_users`: lista de GT que usaron ese track.
- `n_gt_users`
- `first_frame`
- `last_frame`
- `n_frames_present`
- `is_pure_track`: track usado por un unico GT.
- `is_fragment_track`: track compartido por varios GT en la historia.
- `is_foreign_track`: track cuyo owner mayoritario no coincide con el GT de referencia.

## 5.7. `per_event.csv`

Registro de eventos interpretables.

- `event_type`: tipo de evento. Valores actuales:
  `swap`, `theft_with_new_id`, `theft_with_displacement`, `reopen_existing_as_new`.
- `frame_id`
- `gt_a`, `gt_b`: GT implicados.
- `pred_id_main`, `pred_id_aux`
- `class_name`
- `detail`: resumen textual del evento.

## 5.8. `scene_summary.csv` y `per_scene.csv`

Cada fila resume una escena completa.

### Trazabilidad y configuracion

- `run_id`, `scene_id`, `scene_name`
- `status`
- `started_at`, `finished_at`
- `output_dir`
- `detector_backend`
- `stable_min_frames`
- `max_frames`
- `mask_variant`
- `image_subdir`

### Tamano y dificultad

- `n_frames`
- `n_gt_objects`
- `n_cases`
- `n_visible_gt_observations`
- `n_matched_gt_observations`
- `mean_gt_per_frame`
- `total_distractors_sum`
- `same_class_distractors_sum`
- `mean_total_distractors`
- `mean_same_class_distractors`
- `gt_area_frac_sum`
- `gt_area_frac_count`
- `mean_gt_area_frac`
- `n_new_gt`
- `n_existing_gt`
- `new_object_rate`

### Identidad, tracking y estabilidad

- `idtp`, `idfp`, `idfn`
- `idf1`, `idp`, `idr`
- `idsw`, `frag`
- `tracking_recall`
- `tracking_iou_sum`
- `mean_tracking_iou`
- `deta`, `assa`, `hota`
- `n_mt_objects`, `n_pt_objects`, `n_ml_objects`
- `mt`, `pt`, `ml`

### Exactitud de decision e incertidumbre

- `accuracy_global_collapsed`
- `accuracy_existing_vs_new_collapsed`
- `accuracy_parent_collapsed`
- `new_detection_accuracy_collapsed`
- `coverage_firm`
- `firm_accuracy`
- `uncertain_rate`
- `hypothesis_recall_uncertain`

### Inflacion de tracks y reaperturas

- `n_unique_real_pred_tracks`
- `pred_track_inflation_factor`
- `reopen_rate_existing`
- `n_existing_gt_reopened_as_new_rows`
- `n_existing_gt_reopened_as_new_ids`

### Modulos internos

- `distance_used_count`
- `distance_resolved_count`
- `distance_correct_count`
- `distance_usage_rate`
- `distance_resolution_rate`
- `distance_disambiguation_accuracy`
- `distance_unresolved_rate`
- `neighbor_sets_available_count`
- `neighbor_sets_available_rate`
- `context_intervened_count`
- `context_correct_count`
- `context_rescue_count`
- `context_veto_case_count`
- `context_intervention_rate`
- `context_intervention_accuracy`
- `context_rescue_rate`
- `context_veto_rate`
- `context_net_gain`

### Eventos y tiempos

- `objects_fragmented`
- `objects_with_foreign_id_use`
- `swap_events_total`
- `theft_with_new_id_total`
- `theft_with_displacement_total`
- `total_runtime_seconds`
- `avg_runtime_seconds`
- `total_loop_ms`
- `avg_loop_ms`

## 5.9. `summary_global.csv`

Es el agregado del batch. Los campos son practicamente los mismos que en `per_scene.csv`, pero agregados sobre todas las escenas completadas.

Importante:

- `summary_global.csv` si es global;
- `per_class.csv`, `per_object.csv`, `per_case.csv` y `per_frame.csv` deben mantenerse como tablas internas a escena y luego agregarse offline si hace falta.

Campos adicionales importantes:

- `n_scenes_completed`: numero de escenas completadas.
- `run_id`: id del batch agregado.

## 5.10. `manifest.csv`

Sirve para reanudar y auditar el batch.

- `run_id`
- `scene_id`
- `scene_name`
- `status`: `completed`, `pending` o `failed`.
- `started_at`
- `finished_at`
- `n_frames_expected`
- `n_frames_processed`
- `output_dir`
- `error_message`

## 5.11. `run_config.csv`

Configuracion efectiva del batch.

- `run_id`
- `batch_name`
- `batch_dir`
- `created_at`
- `detector_backend`
- `stable_min_frames`
- `max_frames`
- `max_scenes`
- `mask_variant`
- `image_subdir`
- `masks_root_base`
- `images_root_base`
- `target_scene_count`
- `target_scene_ids`

## 5.12. Telemetria de memoria recomendada

Esta parte no pertenece a la optimizacion de parametros, sino a la informacion que conviene guardar durante la evaluacion.

### Comportamiento actual de la memoria

La logica actual de actualizacion de prototipos sigue este patron general en objeto, partes y fondo:

- si hay hueco, se inserta;
- si no hay hueco, se intenta `merge`;
- si no procede `merge`, se hace `evict + insert`.

Por tanto:

- el numero bruto final de descriptores guardados en una escena no debe interpretarse por si solo como una metrica principal;
- depende del numero de frames, del tiempo visible de cada objeto y de si la capacidad se llega a saturar o no;
- solo es interpretable junto con capacidad, ocupacion y eventos de memoria.

### Memoria persistente de descriptores

Lo importante aqui es medir el estado persistente que realmente usa el pipeline entre frames.

Conviene guardar por frame y resumir por escena:

- numero total de prototipos activos en memoria persistente;
- desglose por dominio:
  - objeto `work`
  - objeto `stable`
  - partes `work`
  - partes `stable`
  - fondo global `work`
  - fondo global `stable`
  - fondo partials `work`
  - fondo partials `stable`
- ratio de ocupacion respecto a la capacidad configurada;
- media, maximo y valor final por escena.

### Tamano real de las memorias de descriptores

No basta con contar prototipos. Conviene estimar tambien bytes reales de embeddings en RAM para las memorias de descriptores.

Guardar por frame y resumir por escena:

- bytes persistentes de memoria de apariencia;
- bytes persistentes de memoria de partes;
- bytes persistentes de memoria de fondo;
- bytes persistentes totales de memorias de descriptores;
- media, maximo y valor final por escena;
- bytes persistentes por track activo.

Presentacion recomendada:

- guardar el valor base en bytes;
- exponer tambien columnas derivadas en `KB`, `MB` o `GB` solo para lectura humana y tablas finales.

Nota metodologica:

- esta metrica debe contar solo estado necesario del pipeline;
- no debe incluir imagenes, overlays ni artefactos de debug;
- tampoco debe mezclar caches o estructuras de depuracion que no sean necesarias para la ejecucion normal.

### Saturacion

Guardar por escena:

- primer frame en el que cada bloque llega a capacidad, si ocurre;
- fraccion de frames con memoria saturada por bloque;
- numero de bloques o canales que llegan a saturarse al menos una vez.

Lectura:

- esto es mas informativo que el conteo bruto final para comparar los tres regimenes:
  - se llena pronto;
  - se llena mas tarde;
  - rara vez se llena.

### Eventos de mantenimiento de memoria

Derivar de `proto_events` y guardar por frame y por escena:

- `INSERT`
- `MERGE_INSERT`
- `EVICT_INSERT`
- actualizaciones por duplicado (`DUP_*`, `STABLE_*`)
- promociones a `stable` (`PROMOTE_*`)
- skips relevantes (`SKIP_LOW_QUALITY`, `SKIP_NOT_NOVEL`)

Metricas recomendadas:

- conteos absolutos;
- tasas normalizadas por frame;
- tasas normalizadas por objeto visible o por track activo.

Lectura:

- si una memoria pequena fuerza demasiados `EVICT_INSERT`, eso es una senal clara;
- si una memoria grande reduce evicciones pero apenas cambia las metricas finales, probablemente no compense.

### Convergencia solo cuando la memoria no se satura

Si una configuracion alta rara vez llega al limite, entonces si tiene sentido analizar si la memoria converge.

Metricas recomendadas:

- pendiente del numero de prototipos en el ultimo tramo de la escena;
- diferencia entre media del ultimo tercio y del tercio anterior;
- varianza del numero de prototipos en el tramo final.

Lectura:

- si la curva se estabiliza, la memoria converge a un tamano efectivo;
- si sigue creciendo casi hasta el final, la escena no permite hablar de convergencia real.

### Memoria temporal por frame

Esto es distinto del estado persistente de descriptores.

Guardar por frame:

- RSS del proceso antes y despues del frame;
- pico de memoria CPU del proceso durante el frame, si se puede medir con fiabilidad;
- pico de memoria GPU del frame, si se usa `torch.cuda`;
- delta temporal del frame, separada del estado persistente.

Lectura:

- esto captura el coste transitorio de procesar un frame, que no debe mezclarse con el tamano de las memorias persistentes.

### Memoria acumulada a nivel de escena

Guardar por escena:

- maximo de memoria CPU del proceso durante toda la escena;
- maximo de memoria GPU durante toda la escena;
- maximo de bytes persistentes del estado del pipeline;
- media de bytes persistentes del estado del pipeline.

Lectura:

- separa claramente:
  - memoria temporal por frame;
  - memoria persistente del pipeline a lo largo de la escena.

### Espacio en disco estrictamente necesario

Solo si el pipeline persiste algo necesario para funcionar o para reproducibilidad minima.

Guardar:

- tamano total de artefactos minimos de salida:
  - `tracking_eval.json`
  - CSV esenciales
  - `run_config.csv`

No incluir:

- imagenes
- videos
- overlays
- dumps de debug no necesarios

Nota:

- si el pipeline no guarda descriptores a disco durante la ejecucion, el espacio en disco no es una metrica central del comportamiento de memoria del tracker;
- en ese caso se puede reportar solo como coste de artefactos finales del experimento.

### Donde conviene guardar cada cosa

En `per_frame.csv`:

- ocupacion persistente por dominio;
- bytes persistentes por dominio y total;
- RSS y picos temporales del frame;
- conteos de eventos de memoria del frame.

En `per_scene.csv`:

- medias, maximos y valores finales de ocupacion;
- medias, maximos y valores finales de bytes persistentes;
- primer frame de saturacion y fraccion de frames saturados;
- agregados de `INSERT`, `MERGE_INSERT`, `EVICT_INSERT`, `PROMOTE_*` y `DUP_*`;
- maximo CPU y GPU de toda la escena.

En `summary_global.csv`:

- medias y sumas agregadas a nivel batch de las metricas anteriores, para comparar configuraciones completas.

### Diccionario de columnas exactas

#### Columnas base en `per_frame.csv`

Ocupacion persistente:

- `mem_active_track_count`: numero de tracks activos en memoria en ese frame.
- `mem_obj_work_count`: prototipos de apariencia en banco `work`.
- `mem_obj_stable_count`: prototipos de apariencia en banco `stable`.
- `mem_obj_count`: total de prototipos de apariencia.
- `mem_parts_work_count`: prototipos de partes en banco `work`.
- `mem_parts_stable_count`: prototipos de partes en banco `stable`.
- `mem_parts_count`: total de prototipos de partes.
- `mem_bg_global_work_count`: prototipos de fondo global en bancos `work`.
- `mem_bg_global_stable_count`: prototipos de fondo global en bancos `stable`.
- `mem_bg_global_count`: total de prototipos de fondo global.
- `mem_bg_partials_work_count`: prototipos de fondo partials en bancos `work`.
- `mem_bg_partials_stable_count`: prototipos de fondo partials en bancos `stable`.
- `mem_bg_partials_count`: total de prototipos de fondo partials.
- `mem_descriptor_count`: total global de prototipos persistentes de descriptor memory.

Capacidades:

- `mem_obj_work_capacity`
- `mem_obj_stable_capacity`
- `mem_obj_capacity`
- `mem_parts_work_capacity`
- `mem_parts_stable_capacity`
- `mem_parts_capacity`
- `mem_bg_global_work_capacity`
- `mem_bg_global_stable_capacity`
- `mem_bg_global_capacity`
- `mem_bg_partials_work_capacity`
- `mem_bg_partials_stable_capacity`
- `mem_bg_partials_capacity`
- `mem_descriptor_capacity`

Ratios y saturacion:

- `mem_obj_fill_ratio`
- `mem_parts_fill_ratio`
- `mem_bg_global_fill_ratio`
- `mem_bg_partials_fill_ratio`
- `mem_descriptor_fill_ratio`
- `mem_obj_saturated`
- `mem_parts_saturated`
- `mem_bg_global_saturated`
- `mem_bg_partials_saturated`
- `mem_descriptor_saturated`

Tamano de memorias de descriptores en bytes:

- `mem_obj_bytes`: bytes persistentes estimados de memoria de apariencia.
- `mem_parts_bytes`: bytes persistentes estimados de memoria de partes.
- `mem_bg_global_bytes`: bytes persistentes estimados de memoria de fondo global.
- `mem_bg_partials_bytes`: bytes persistentes estimados de memoria de fondo partials.
- `mem_descriptor_bytes`: bytes persistentes estimados totales de descriptor memory.

Eventos de memoria por frame:

- `mem_evt_total_count`
- `mem_evt_obj_count`
- `mem_evt_parts_count`
- `mem_evt_bg_count`
- `mem_evt_insert_count`
- `mem_evt_merge_insert_count`
- `mem_evt_evict_insert_count`
- `mem_evt_dup_count`
- `mem_evt_stable_count`
- `mem_evt_promote_count`
- `mem_evt_skip_count`

Memoria temporal CPU por frame:

- `mem_process_rss_before_bytes`
- `mem_process_rss_after_read_bytes`
- `mem_process_rss_after_pipeline_bytes`
- `mem_process_rss_after_eval_bytes`
- `mem_process_rss_peak_approx_bytes`
- `mem_process_rss_delta_bytes`

Memoria temporal GPU por frame:

- `mem_gpu_allocated_after_pipeline_bytes`
- `mem_gpu_reserved_after_pipeline_bytes`
- `mem_gpu_allocated_after_eval_bytes`
- `mem_gpu_reserved_after_eval_bytes`
- `mem_gpu_peak_allocated_bytes`
- `mem_gpu_peak_reserved_bytes`

Notas:

- las columnas GPU pueden quedar vacias si no hay CUDA disponible;
- los tamanos se guardan en bytes y las conversiones a `KB/MB/GB` se hacen offline para visualizacion.

#### Columnas derivadas en `per_scene.csv` y `summary_global.csv`

Para campos numericos de memoria persistente y temporal se usan estos sufijos:

- `_mean`: media a lo largo de los frames.
- `_max`: maximo observado.
- `_final`: valor del ultimo frame de la escena.

Ejemplos:

- `mem_descriptor_bytes_mean`
- `mem_descriptor_bytes_max`
- `mem_descriptor_bytes_final`
- `mem_process_rss_after_eval_bytes_mean`
- `mem_process_rss_after_eval_bytes_max`
- `mem_process_rss_after_eval_bytes_final`

Para saturacion se usan estos sufijos:

- `_first_frame`: primer frame donde ese bloque estuvo saturado.
- `_frame_fraction`: fraccion de frames de la escena en que estuvo saturado.

Ejemplos:

- `mem_obj_saturated_first_frame`
- `mem_obj_saturated_frame_fraction`
- `mem_descriptor_saturated_first_frame`
- `mem_descriptor_saturated_frame_fraction`

Para eventos de memoria se usan estos sufijos:

- `_total`: numero total de eventos de ese tipo en la escena o batch.
- `_rate_per_frame`: tasa media por frame.

Ejemplos:

- `mem_evt_insert_total`
- `mem_evt_insert_rate_per_frame`
- `mem_evt_evict_insert_total`
- `mem_evt_evict_insert_rate_per_frame`

En `summary_global.csv` se reutiliza la misma convencion, agregando sobre todos los `per_frame` del batch.

## 6. Sobre strict, permissive y collapsed

Conviene no mezclar estos conceptos:

- `strict`: exige seguir exactamente el track de referencia global del GT.
- `permissive`: acepta cualquier track que siga siendo canonico del mismo GT.
- `collapsed_*`: evalua la salida final del sistema tras colapsar ambiguedades y provisionales a una interpretacion comparable.

Ademas:

- `strict` y `permissive` usan como denominador todos los GT visibles, no solo los que tuvieron asignacion base.
- `IDP`, `IDR`, `IDF1`, `IDSW` y `Frag` en estos CSV son metricas internas consistentes del evaluador sobre la representacion colapsada, no una exportacion literal de un toolkit oficial externo.

En general:

- Para papers y tablas principales, suele tener mas sentido usar `collapsed_*`, `IDF1`, `tracking_recall`, `DetA`, `AssA`, `HOTA`, `IDSW`, `Frag`.
- Para diagnostico interno, `strict` y `permissive` ayudan mucho a entender fragmentacion y reaperturas.

## 7. Que merece la pena guardar inline y que no

Lo que si tiene sentido guardar inline:

- decisiones finales por caso
- ids y trazabilidad por escena
- señales internas de distancia y contexto
- metrica temporal por objeto y por track
- dificultad local por caso
- eventos raros explicativos
- tiempos
- telemetria de memoria persistente y temporal del pipeline

Lo que normalmente no hace falta guardar inline porque se puede recomputar offline:

- comparativas macro vs micro
- analisis de sesgo por clase
- tipologias manuales de escena
- rankings y tablas para el TFM o paper
- graficos y resumenes derivados

## 8. Recomendacion practica

Si hay que elegir solo unos pocos CSV para casi todo el analisis posterior, los mas utiles son:

- `per_case.csv`: diagnostico fino y casi cualquier agregado offline.
- `per_object.csv`: estabilidad temporal y fragmentacion por GT.
- `per_class.csv`: comparativas por clase.
- `per_scene.csv`: resumen por escena y tablas globales.
- `per_case_modules.csv`: estudio interno de distancia, contexto y neighbor sets.

Si el estudio incluye sensibilidad de memorias o coste de recursos, conviene anadir tambien:

- `per_frame.csv`: para ocupacion, saturacion y picos temporales de memoria.
