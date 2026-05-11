# Performance Optimizations Log

Este documento registra las optimizaciones de rendimiento aplicadas en el pipeline, con foco en:

- qué módulo/bloque se tocó
- qué se cambió exactamente
- qué timings debería afectar
- riesgo esperado sobre comportamiento
- notas para depurar regresiones futuras

La idea es que, si dentro de unos días aparece una diferencia de resultados o una regresión de tiempo, podamos revisar rápido qué cambios son candidatos.

## Convenciones

- `Sin impacto esperado`: refactor/caché/estructura que no debería cambiar resultados.
- `Riesgo bajo`: puede cambiar algún valor fino o desempate, pero solo en casos acotados.
- `No documentado aquí`: cambios experimentales ya revertidos o no activos.

## Estado actual de cuellos de botella

Según los últimos timings, los bloques caros principales siguen siendo:

- `perception/detector/segment`
- `perception/bg_features`
- `perception/parts_features/parts_kmeans`
- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick`
- `association/post_assignment/temporal_reconcile`
- `update/neighbor_graphs/dist_observations`

## Cambios aplicados

### 0. `detection/davis_segmenter.py` - extracción de instancias GT en una sola pasada

Archivo:

- `REMIND/detection/davis_segmenter.py`

Bloque objetivo:

- `perception/detector/segment` cuando el backend es DAVIS/GT

Cambio:

- Se sustituyó el patrón de:
  - `np.unique(...)` por instancia
  - máscara completa por instancia
  - escaneos separados para `bbox`, `center` y `area`
- por una pasada global sobre la máscara etiquetada para obtener stats por instancia:
  - `bbox`
  - `center`
  - `area`
- Después, para cada instancia, la máscara booleana solo se reconstruye dentro de su ROI.
- Si hay erosión, el recalculo geométrico se limita también al ROI.

Impacto esperado:

- bajar `segment` en secuencias con muchas instancias GT

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`
- prueba sintética de sanidad para `bbox`/`center`/`area`

Notas:

- Este cambio solo aplica al backend DAVIS; no toca el camino de otros segmentadores.

### 1. `utils/math.py`

Archivo:

- `REMIND/utils/math.py`

Bloque objetivo:

- `bg_features/bg_proto_inner`
- `bg_features/bg_proto_outer`
- `parts_features/parts_kmeans`

Cambio:

- Optimización de `kmeans_np` con:
  - precálculo de normas
  - cálculo de distancias más barato
  - recomputación de centros más ligera

Impacto esperado:

- bajar coste de k-means intra-frame en partes y prototipos de fondo

Riesgo:

- `Sin impacto esperado`

Notas:

- Si alguna vez aparecen diferencias numéricas raras en prototipos, revisar primero esta función.

### 2. `features/part_features.py`

Archivo:

- `REMIND/features/part_features.py`

Bloque objetivo:

- `parts_features/parts_kmeans`

Cambio:

- Se dejó de reconstruir máscaras completas y hacer pooling sobre todo el `fmap` para cada cluster.
- El pooling se hace directamente sobre los patches ya seleccionados del objeto.

Impacto esperado:

- reducir coste del backend de partes por k-means

Riesgo:

- `Sin impacto esperado`

Notas:

- Si cambia el número de partes válidas o sus soportes sin razón clara, revisar aquí y en `kmeans_np`.

### 3. `memory/neighbor_distance_graph.py` - ROI para gap exacto

Archivo:

- `REMIND/memory/neighbor_distance_graph.py`

Bloque objetivo:

- `update/neighbor_graphs/dist_observations`
- cualquier uso de `compute_relation_observation(...)`

Cambio:

- La distancia máscara-a-máscara exacta dejó de operar sobre toda la imagen.
- Ahora se limita al ROI unión de ambas máscaras.

Impacto esperado:

- reducción clara de coste por par al calcular observaciones geométricas

Riesgo:

- `Sin impacto esperado`

Notas:

- Este cambio solo abarata el mismo cálculo exacto; no altera la lógica geométrica.

### 4. `memory/neighbor_distance_graph.py` - shortcut por bbox para pares muy lejanos

Archivo:

- `REMIND/memory/neighbor_distance_graph.py`

Bloque objetivo:

- `update/neighbor_graphs/dist_observations`

Cambio:

- Si dos objetos ya están claramente separados por bbox y `bbox_gap_n > exact_gap_max_n`, se evita calcular el gap exacto entre máscaras.
- En esos casos se usa directamente `bbox_gap`.
- Se mantiene `gap_quality=1.0` en este camino para no penalizar artificialmente pares lejanos válidos.

Impacto esperado:

- bajar bastante `dist_observations` en escenas con muchos pares lejanos

Riesgo:

- `Riesgo bajo`

Notas:

- Este es el cambio con más riesgo semántico de los aplicados hasta ahora.
- Puede cambiar `mask_gap_n` y algún score geométrico fino en pares claramente lejanos.
- Si hay una regresión geométrica rara, este es uno de los primeros sitios a revisar.

### 5. `association/scores/sets/sets_options.py` - caché de opciones por clase

Archivo:

- `REMIND/association/scores/sets/sets_options.py`

Bloque objetivo:

- `association/neighbor_sets/build_class_options`
- parte de `run_beam_search/expand_beam_for_class/class_options`

Cambio:

- Las `class_options` base se generan una vez por `(class_id, kernel, vocab_size)`.
- Para cada estado del beam, se filtran por `used_obj_ids` en vez de regenerarlas.
- El debug caro de `class_options` solo se construye si la tabla correspondiente está activa.

Impacto esperado:

- reducir recomputación de opciones y trabajo de debug innecesario

Riesgo:

- `Sin impacto esperado`

Notas:

- Si el tiempo de `build_class_options` vuelve a subir, comprobar si el debug de neighbor sets está activo.

### 6. `association/scores/sets/sets_search.py` + `sets_scoring.py` - score incremental

Archivos:

- `REMIND/association/scores/sets/sets_search.py`
- `REMIND/association/scores/sets/sets_scoring.py`

Bloque objetivo:

- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states`

Cambio:

- El estado del beam acumula incrementalmente:
  - `explained_n`
  - producto de `class_info`
  - sumas/pesos de `class_support`
  - sumas de `class_stability`
  - `class_logC_sum`
  - acumulados de exclusividad
- `score_state_quick` reutiliza esos agregados en lugar de reconstruirlos desde `per_class_sel`.

Impacto esperado:

- bajar coste repetido por estado en el beam

Riesgo:

- `Sin impacto esperado`

Notas:

- Si algún score de set deja de cuadrar con logs de referencia, revisar los acumulados del estado.

### 7. `association/scores/sets/sets_search.py` - simplificación de estado del beam

Archivo:

- `REMIND/association/scores/sets/sets_search.py`

Bloque objetivo:

- `association/neighbor_sets/run_beam_search/expand_beam_for_class/transition`
- `association/neighbor_sets/collect_hypotheses`

Cambio:

- Se simplificó el estado:
  - `explained_det` pasa a `explained_det_sorted`
  - se elimina `obj_ids` duplicado y se usa `obj_ids_sorted`
  - se elimina `per_class_sel` completo y se guardan solo `last_class_id` y `last_class_k`
- Menos copias de `set`/`dict` por transición.

Impacto esperado:

- bajar `transition`
- bajar algo de `collect_hypotheses`

Riesgo:

- `Sin impacto esperado`

Notas:

- Si aparece un bug de selección diversa por `k`, revisar `last_class_id`/`last_class_k`.

### 8. `association/scores/sets/sets_graph_utils.py` - caché por par y MST denso

Archivo:

- `REMIND/association/scores/sets/sets_graph_utils.py`

Bloque objetivo:

- `association/neighbor_sets/.../density_score_cached`

Cambio:

- Caché por frame del grafo de cada objeto.
- Caché por frame de la métrica de cada par de objetos para densidad.
- Sustitución del máximo spanning tree por una versión densa tipo Prim, evitando construir/ordenar todas las aristas cada vez.

Impacto esperado:

- gran bajada de `density_score_cached`
- mejora clara de `score_state_quick`

Riesgo:

- `Sin impacto esperado`

Validación:

- Se comparó la implementación nueva con la anterior en casos aleatorios y devolvió exactamente los mismos valores de `density_score`.

Notas:

- Si `density_score_cached` vuelve a dominar, revisar si el beam está explorando sets con poca reutilización entre sí.

### 9. `utils/time.py` + `pipeline/reid_pipeline.py` + `main.py` - trazabilidad de timings

Archivos:

- `REMIND/utils/time.py`
- `REMIND/pipeline/reid_pipeline.py`
- `REMIND/main.py`
- `REMIND/association/context/sets_provider.py`
- `REMIND/association/scores/sets/neighbor_sets_score.py`
- `REMIND/association/scores/sets/sets_options.py`
- `REMIND/association/scores/sets/sets_search.py`
- `REMIND/association/scores/sets/sets_scoring.py`

Bloque objetivo:

- Diagnóstico de rendimiento, no optimización directa

Cambio:

- La tabla de timings pasó a usar árbol real por prefijos.
- Se añadió `other` para hacer visibles tiempos del padre no cubiertos por hijos directos.
- Se propagaron timings internos de `neighbor_sets`.
- Se añadió `[TIME_FRAME]` para medir latencia real del frame completo fuera del pipeline.

Impacto esperado:

- mejor capacidad de localizar cuellos reales

Riesgo:

- `Sin impacto esperado` sobre resultados del pipeline

Notas:

- No confundir `[TIME] total` con `[TIME_FRAME] total`.

### 10. `perception/perception_engine.py` + `features/object_features.py` + `features/part_features.py` + `features/background_features.py` - reutilización de caches patch-space

Archivos:

- `REMIND/perception/perception_engine.py`
- `REMIND/features/object_features.py`
- `REMIND/features/part_features.py`
- `REMIND/features/background_features.py`

Bloque objetivo:

- `perception/obj_features`
- `perception/parts_features/parts_kmeans`
- `perception/bg_features/bg_proto_inner`
- `perception/bg_features/bg_proto_outer`

Cambio:

- Se añadió un cache por frame con:
  - `flat_feats`
  - `flat_feats_n` cuando compensa
- Se añadió un cache por detección con:
  - `cov`
  - `patch_mask`
- `obj_features`, `parts` y `bg` reutilizan esos datos en vez de:
  - volver a aplanar el `fmap`
  - volver a normalizar filas del `fmap`
  - volver a proyectar la máscara del objeto a patch-space

Impacto esperado:

- reducir trabajo duplicado entre módulos de features sobre la misma detección

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`
- prueba sintética de sanidad para los tres extractores usando los caches nuevos

Notas:

- El fondo sigue sanitizando su máscara como antes; solo reutiliza el `fmap` ya aplanado/normalizado.
- La proyección raw `mask -> coverage` compartida se usa en objeto y partes, que ya operaban sobre esa misma semántica.

### 11. `features/background_features.py` - caché local de dilataciones en `bg_rings`

Archivo:

- `REMIND/features/background_features.py`

Bloque objetivo:

- `perception/bg_features/bg_rings`

Cambio:

- Dentro de `build_local_rings_patch_masks(...)` se memoizan las dilataciones del `obj_patch` por radio.
- Así, cuando la adaptación de radios o la exclusión de borde vuelven a pedir el mismo radio, se reutiliza el resultado en vez de relanzar `cv2.dilate`.

Impacto esperado:

- bajar `bg_rings`, especialmente cuando hay radios adaptativos o varios radios coinciden entre sí

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`

Notas:

- No cambia la definición de `ring_inner` ni `ring_outer`; solo evita recomputar máscaras idénticas.

### 12. `memory/neighbor_distance_graph.py` - `distanceTransform` exacta limitada al ROI

Archivo:

- `REMIND/memory/neighbor_distance_graph.py`

Bloque objetivo:

- `update/neighbor_graphs/dist_observations`

Cambio:

- En el cálculo exacto de gap máscara-a-máscara, la `distanceTransform` ya no se lanza sobre toda la imagen.
- Ahora se calcula directamente sobre el ROI unión de ambas cajas, que ya contiene toda la máscara del objeto de referencia.

Impacto esperado:

- bajar el coste de los pares cercanos donde todavía se usa gap exacto

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`
- prueba sintética de sanidad de `_signed_mask_gap_px`

Notas:

- Este cambio es complementario al recorte del ROI; antes el ROI solo se aplicaba a la consulta final, pero la transformada seguía siendo full-frame.

### 13. `update/update_general.py` - recorrido por pares únicos en `dist_observations`

Archivo:

- `REMIND/update/update_general.py`

Bloque objetivo:

- `update/neighbor_graphs/dist_observations`

Cambio:

- `build_dist_observations(...)` dejó de recorrer `ids x ids` con descarte por `seen_pairs`.
- Ahora primero prepara solo los objetos elegibles para observación relacional:
  - con episodio permitido
  - con `neighbor_dist` habilitado
  - con geometría válida
- Después recorre solo pares únicos `i < j`, reutilizando por objeto:
  - parámetros de `neighbor_dist`
  - `mask_runtime`

Impacto esperado:

- bajar sobrecarga Python en el montaje de observaciones por par
- reducir lookups repetidos de memoria/configuración dentro del doble bucle

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`

Notas:

- La observación calculada por pareja sigue siendo la misma que antes.
- Se conserva el mismo criterio implícito de parametrización por el primer objeto que aparece en `ids`.

### 14. `association/policy/outcome_policy.py` - menos trabajo redundante en decisiones postcreate

Archivo:

- `REMIND/association/policy/outcome_policy.py`

Bloque objetivo:

- `association/post_assignment/temporal_reconcile`

Cambio:

- En `build_postcreate_temporal_decisions(...)` se evita trabajo redundante por detección:
  - retorno temprano si `to_create` está vacío
  - para `raw_cands` se usa solo el mejor candidato en vez de ordenar toda la lista cuando no hace falta
  - el `focus_score_map` reutiliza `score_map` o `raw_score_map` ya calculados
  - si el foco ya viene de `supported`, no se vuelve a ordenar con el mismo criterio

Impacto esperado:

- bajar parte del coste Python dentro de `temporal_reconcile`
- reducir recomputación de scores y ordenaciones locales en postcreate

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`

Notas:

- No cambia qué candidatos entran en `focus`, ni la forma de puntuarles.
- El orden efectivo de `supported` se conserva, porque ya estaba ordenado con el mismo `score_map`.

### 15. `features/background_features.py` - reutilización de agregados exactos en prototipos

Archivo:

- `REMIND/features/background_features.py`

Bloque objetivo:

- `perception/bg_features/bg_proto_inner`
- `perception/bg_features/bg_proto_outer`

Cambio:

- `compute_cluster_stats(...)` ahora agrupa una sola vez los índices por `label` y guarda agregados exactos por cluster:
  - `count`
  - suma de features normalizadas
  - suma ponderada cuando hay `weights`
- `build_merged_clusters(...)` reutiliza esos agregados para recomputar:
  - masa
  - centro merged
  - cohesión
- Se mantiene la concatenación de `idxs` para que la selección final de prototipos siga funcionando igual.

Impacto esperado:

- bajar trabajo Python/Numpy repetido tras `kmeans`
- reducir coste de fusión de clusters en `bg_proto_*`

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`

Notas:

- No cambia ni el `kmeans`, ni el criterio de merge por similitud, ni la elección final de prototipos.
- La cohesión merged sigue calculándose como media no ponderada de similitudes, igual que antes.

### 16. `config/default_config.yaml` + cableado de runtime - flag explícito para desactivar `bg_partials`

Archivos:

- `REMIND/config/default_config.yaml`
- `REMIND/utils/config.py`
- `REMIND/perception/perception_engine.py`
- `REMIND/association/similarity_computer.py`
- `REMIND/association/scores/base_scores.py`
- `REMIND/update/descriptors/update_background.py`

Bloque objetivo:

- control experimental y operativo de `perception/bg_proto_*`
- control funcional de `association/.../bg_partials`

Cambio:

- Se añadió `association.similarity.background_partials.enabled`.
- Cuando está a `false`, la rama `bg_partials` queda desactivada por completo:
  - no se generan prototipos observados en percepción
  - no se actualizan bancos `inner_partials` / `outer_partials`
  - no se usa el término `bg_partials` en scoring
- El término `bg_global` sigue activo y sin cambios.

Impacto esperado:

- ninguno mientras el flag siga en `true`
- si se pone en `false`, ahorro claro de tiempo en `bg_proto_*` y desaparición funcional de `bg_partials`

Riesgo:

- `Sin impacto esperado` con el valor por defecto `true`
- `Riesgo alto` si se usa en `false`, porque cambia las señales activas del sistema

Validación:

- `py_compile`

Notas:

- Esta entrada no es una optimización activa por sí sola; deja preparada una palanca explícita para pruebas controladas.
- El helper central exige tanto `association.similarity.background_partials.enabled` como `bg_local.prototypes.enabled`.

### 17. `assignment_result_applier.py` + `known_set_distance_disambiguator.py` - menos copias en `temporal_reconcile`

Archivos:

- `REMIND/association/engine/assignment_result_applier.py`
- `REMIND/association/disambiguation/known_set_distance_disambiguator.py`

Bloque objetivo:

- `association/post_assignment/temporal_reconcile`

Cambio:

- Se eliminaron copias redundantes de listas y `dict`s entre pases de reconciliación:
  - los `ambiguous_entries` ya no se clonan varias veces antes de resolver/mezclar
  - `remaining_ambiguous_entries` se reutiliza tal como lo devuelve el desambiguador
  - las salidas de `postcreate_temporal` se propagan sin recopiado inmediato
- `assigned_by_det_id` se actualiza incrementalmente con `resolved_matches` en vez de reconstruirse entero tras cada merge.
- Se añadió retorno temprano cuando no hay ni ambiguos ni `create_entries`.

Impacto esperado:

- bajar sobrecarga Python y churn de memoria dentro de `temporal_reconcile`
- reducir trabajo fijo por pase incluso cuando la decisión final no cambia

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`

Notas:

- No cambia la lógica de resolución ni los thresholds.
- Las copias necesarias siguen existiendo en los puntos de salida donde el estado se materializa fuera del bloque.

### 18. `default_config.yaml` + `assignment_result_applier.py` - recorte conservador de exploración en `temporal_reconcile`

Archivos:

- `REMIND/config/default_config.yaml`
- `REMIND/association/engine/assignment_result_applier.py`

Bloque objetivo:

- `association/post_assignment/temporal_reconcile`

Cambio:

- Se redujo el espacio de exploración del reconciliador temporal con un paquete conservador de config:
  - `known_set_distance_disambiguation.max_passes: 3 -> 2`
  - `known_set_distance_disambiguation.max_candidate_union: 6 -> 5`
  - `known_set_distance_disambiguation.soft_anchor_max: 6 -> 4`
- `max_passes` deja de estar hardcodeado y pasa a leerse desde config.

Impacto esperado:

- bajar el coste de `temporal_reconcile` en escenas cargadas
- limitar casos donde el solver entra en combinatoria alta por demasiados candidatos o demasiadas anchors blandas

Riesgo:

- `Bajo`, con posible cambio leve en casos frontera

Validación:

- `py_compile`

Notas:

- `max_group_size` se mantiene en `4` para no recortar todavía casos de 4 ambiguos bien acotados.
- `max_anchors`, `discriminative_anchor_topk` y `anchor_pair_topk` se dejan intactos porque hoy la selección efectiva ya está dominada por los top-k discriminativos.

### 19. `background_features.py` + `perception_engine.py` - reutilización segura de `patch_cache` en `bg_rings`

Archivos:

- `REMIND/features/background_features.py`
- `REMIND/perception/perception_engine.py`

Bloque objetivo:

- `perception/bg_features/bg_rings`

Cambio:

- `bg_features` recibe ahora la misma `patch_cache` ya construida para `object_features` y `part_features`.
- Dentro de `build_local_rings_patch_masks(...)`, si la máscara saneada coincide exactamente con la máscara original, se reutilizan:
  - `cov`
  - `patch_mask`
- Si `sanitize` modifica la máscara, `background_features` mantiene el camino anterior y recalcula cobertura desde la máscara saneada.

Impacto esperado:

- evitar una reproyección `mask_px -> patch coverage` redundante en casos donde el saneado no altera la máscara
- bajar parte del coste fijo de `bg_rings` sin tocar la geometría efectiva de los anillos

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`

Notas:

- La reutilización es deliberadamente conservadora: no se comparte la cobertura si la máscara saneada difiere de la original.
- El coste de `sanitize_object_mask(...)` y de las dilataciones adaptativas sigue existiendo; esta optimización solo elimina una parte redundante del bloque.

### 20. `tracking_metrics.py` + `davis_gt.py` - IoU exacta con recorte por bbox en evaluación offline

Archivos:

- `REMIND/testing/tracking_metrics.py`
- `REMIND/testing/davis_gt.py`

Bloque objetivo:

- `testing/run_tracking_test.py` -> `eval`

Cambio:

- El matching detección-GT ya no calcula IoU sobre la máscara completa del frame para cada par.
- Ahora:
  - GT precalcula `area` y `bbox_xyxy`
  - cada detección reutiliza `bbox`/`area` cuando existen
  - solo se intenta IoU si las `bbox` se solapan
  - la intersección exacta se calcula sobre el recorte de la zona común
  - la unión se obtiene con `area_det + area_gt - inter`
- El resultado de IoU sigue siendo exacto; cambia solo la forma de calcularlo.

Impacto esperado:

- recorte muy fuerte del coste de `eval` en secuencias largas
- eliminación del barrido full-frame por cada par `det x gt`

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`

Notas:

- La optimización es especialmente rentable cuando las máscaras ocupan una fracción pequeña del frame.
- Si una detección no trae `bbox`/`area` válidos, se cae de forma segura al cálculo desde máscara.

### 21. `davis_gt.py` - GT por instancia con stats vectorizadas y máscara recortada

Archivos:

- `REMIND/testing/davis_gt.py`
- `REMIND/testing/tracking_metrics.py`

Bloque objetivo:

- `testing/run_tracking_test.py` -> `gt`

Cambio:

- La carga GT deja de recorrer `np.unique(mask)` + construir una máscara booleana de tamaño frame completo para cada instancia.
- Ahora reutiliza `DavisSegmenter.instance_stats_from_mask(...)`, que extrae `bbox` y `area` de todas las instancias en una sola pasada vectorizada.
- Cada `GroundTruthObject` guarda su máscara recortada al `bbox`, no una copia booleana de toda la imagen.
- El cálculo de IoU en evaluación se ajusta para usar esa máscara recortada manteniendo exactamente la misma IoU final.

Impacto esperado:

- bajar claramente el coste de `gt`
- reducir asignaciones de memoria y comparaciones full-frame por instancia

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`

Notas:

- La semántica de `bbox_xyxy` y `area` no cambia; solo cambia la representación interna de `mask`.
- Esta optimización es especialmente útil en frames grandes con muchas instancias pequeñas.

### 22. `part_features.py` - postproceso exacto de clusters sin máscaras booleanas repetidas

Archivos:

- `REMIND/features/part_features.py`

Bloque objetivo:

- `perception/parts_features/parts_kmeans`

Cambio:

- `extract_kmeans_parts(...)` deja de recorrer cada cluster con `sel = labels_obj == ci` y slices repetidos sobre todos los patches.
- Ahora agrupa una sola vez los índices por label usando `argsort + bincount`.
- Cuando `use_trimmed_mean` está desactivado, el descriptor del cluster se calcula de forma exacta por agregación directa:
  - media ponderada sobre `x`
  - normalización final
  - coherencia sobre `x_n`
- Si `use_trimmed_mean` está activado, se conserva el camino anterior.
- También se elimina un cast redundante a `float64` en la llamada a `kmeans_np(...)`, ya que la propia función ya hace la conversión necesaria.

Impacto esperado:

- bajar parte del coste fijo de `parts_kmeans` fuera del solver k-means
- reducir boolean masks y materializaciones intermedias por cluster

Riesgo:

- `Sin impacto esperado`

Validación:

- `py_compile`

Notas:

- El algoritmo de `kmeans` no cambia.
- La ruta rápida solo se activa cuando `trimmed_mean` está desactivado, que es la configuración actual.

### 23. `default_config.yaml` - recorte conservador de coste en `parts_kmeans`

Archivos:

- `REMIND/config/default_config.yaml`

Bloque objetivo:

- `perception/parts_features/parts_kmeans`

Cambio:

- Se redujo la carga del `kmeans` de partes con este ajuste de config:
  - `part_descriptors.kmeans.n_init: 3 -> 2`
  - `part_descriptors.kmeans.k: 6 -> 4`
  - `part_descriptors.kmeans.iters: 10 -> 5`

Impacto esperado:

- bajar de forma clara el coste de `parts_kmeans`
- reducir reinicios, número de clusters y número máximo de iteraciones por objeto

Riesgo:

- `Bajo`, con posible pérdida leve de detalle en la señal de partes

Validación:

- no requiere validación de sintaxis Python; cambio de config

Notas:

- Este cambio ya no es una optimización puramente neutra: modifica la granularidad y robustez del clustering de partes.
- Se asume que `parts` tiene peso práctico bajo frente a `object` y `bg`, así que el ahorro potencial compensa el riesgo.

### 24. `association/scores/sets/neighbor_sets_score.py` + `association/scores/sets/sets_graph_utils.py` - matriz de pares candidata por frame para `density_score`

Archivos:

- `REMIND/association/scores/sets/neighbor_sets_score.py`
- `REMIND/association/scores/sets/sets_graph_utils.py`
- `REMIND/config/default_config.yaml`

Bloque objetivo:

- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick`
- en particular el cálculo repetido de `density_score_cached`

Cambio:

- Se añadió una matriz densa por frame para los objetos candidatos presentes en los pools de `neighbor_sets`.
- Antes de entrar en el beam:
  - se recopilan los `object_id` candidatos;
  - se construye una matriz simétrica de pesos de par;
  - se construye una matriz simétrica de presencia de arista válida.
- Cuando un set a puntuar está contenido en ese vocabulario candidato, `density_score(...)` usa una ruta vectorizada basada en submatrices en vez de reconstruir pares y listas Python estado a estado.
- Se añadió config:
  - `association.scores.neighbor_sets.candidate_pair_matrix.enabled`
  - `association.scores.neighbor_sets.candidate_pair_matrix.max_objects`

Impacto esperado:

- reducir trabajo Python repetido al puntuar estados distintos sobre el mismo conjunto candidato;
- abaratar `density_score` cuando el beam explora muchos sets únicos dentro del mismo frame.

Riesgo:

- `Sin impacto esperado`

Validación:

- `python3 -m py_compile` sobre:
  - `REMIND/association/scores/sets/sets_graph_utils.py`
  - `REMIND/association/scores/sets/neighbor_sets_score.py`
- comparación sintética entre dos implementaciones de `max_spanning_tree_mean_dense`

Notas:

- La construcción de la matriz está capada por `max_objects` para evitar costes de memoria o arranque excesivos en frames grandes.
- Si no hay mejora o empeora en secuencias con baja reutilización entre estados, revisar primero este bloque.

### 25. `detection/davis_segmenter.py` - prefetch de anotaciones DAVIS

Archivos:

- `REMIND/detection/davis_segmenter.py`
- `REMIND/config/default_config.yaml`

Bloque objetivo:

- `perception/detector/segment` cuando el backend es DAVIS/GT

Cambio:

- Se añadió prefetch opcional de la máscara de anotación del siguiente frame usando un `ThreadPoolExecutor` de un worker.
- Si el pipeline avanza frame a frame, la lectura/decodificación del PNG siguiente puede solaparse con el procesamiento del frame actual.
- Se añadieron flags:
  - `davis.prefetch_annotations`
  - `davis.prefetch_distance`

Impacto esperado:

- bajar latencia visible de `segment` en secuencias DAVIS con acceso secuencial a frames;
- reducir tiempo bloqueado en lectura/decodificación del PNG de anotación.

Riesgo:

- `Sin impacto esperado`

Validación:

- `python3 -m py_compile` sobre `REMIND/detection/davis_segmenter.py`

Notas:

- Si los `frame_id` no avanzan de forma secuencial, el código cae a lectura síncrona normal.
- No cambia el formato de máscara ni la lógica de detecciones; solo intenta adelantar I/O del siguiente frame.

### 26. `detection/davis_segmenter.py` + `perception/perception_engine.py` - subbloques de timing en `detector/segment`

Archivos:

- `REMIND/detection/davis_segmenter.py`
- `REMIND/perception/perception_engine.py`

Bloque objetivo:

- Diagnóstico fino de `perception/detector/segment`

Cambio:

- `DavisSegmenter.segment(...)` ahora publica timings internos de:
  - `read_mask`
  - `resize_mask`
  - `resolve_classes`
  - `instance_stats`
  - `build_detections`
- `PerceptionEngine` propaga esos subtiempos bajo el prefijo `detector/segment/`.

Impacto esperado:

- mejor capacidad de localizar qué parte real domina dentro de `segment`;
- no optimiza directamente, pero evita iterar a ciegas.

Riesgo:

- `Sin impacto esperado`

Validación:

- `python3 -m py_compile` sobre:
  - `REMIND/detection/davis_segmenter.py`
  - `REMIND/perception/perception_engine.py`

Notas:

- Esta entrada es de observabilidad; cualquier optimización posterior de `segment` debería apoyarse en estos subbloques.

### 27. `detection/davis_segmenter.py` - ruta densa para `instance_stats_from_mask`

Archivos:

- `REMIND/detection/davis_segmenter.py`

Bloque objetivo:

- `perception/detector/segment/instance_stats`

Cambio:

- `instance_stats_from_mask(...)` ahora usa siempre una ruta directa basada en `np.bincount` indexado por `instance_id`.
- Se evita el coste de `np.unique(..., return_inverse=True)` en el formato real del proyecto, donde los IDs vienen numerados de `1..N`.

Impacto esperado:

- bajar de forma apreciable el subbloque `instance_stats` en máscaras etiquetadas DAVIS;
- reducir coste total de `segment` cuando ese subbloque domina.

Riesgo:

- `Sin impacto esperado`

Validación:

- `python3 -m py_compile` sobre `REMIND/detection/davis_segmenter.py`
- comparación sintética contra la implementación anterior para `uint8`, `uint16` y `uint32`

Notas:

- Esta simplificación asume el contrato real del dataset del proyecto: IDs de instancia numerados de `1..N`.
- En una prueba sintética local dio una reducción clara frente a la variante con `np.unique(..., return_inverse=True)`.

### 28. `features/background_features.py` - subbloques de timing en `bg_rings`

Archivos:

- `REMIND/features/background_features.py`

Bloque objetivo:

- Diagnóstico fino de `perception/bg_features/bg_rings`

Cambio:

- `build_local_rings_patch_masks(...)` ahora desglosa subtiempos de:
  - `bg_rings/sanitize_mask`
  - `bg_rings/mask_to_patch_coverage`
  - `bg_rings/adaptive_rings`
- Estos subbloques cuelgan del timing global de `bg_rings`.

Impacto esperado:

- mejor capacidad de localizar si el coste real viene de morfología/saneado, de la proyección a patch-space o de las dilataciones adaptativas;
- no optimiza directamente, pero evita iterar a ciegas.

Riesgo:

- `Sin impacto esperado`

Validación:

- `python3 -m py_compile` sobre `REMIND/features/background_features.py`

Notas:

- Esta entrada es de observabilidad; cualquier optimización posterior de `bg_rings` debería apoyarse en estos subbloques.

### 29. `features/background_features.py` + config - modo alternativo `convex_hull` para `sanitize`

Archivos:

- `REMIND/features/background_features.py`
- `REMIND/config/default_config.yaml`
- `REMIND/config/CONFIG_TIERS.md`

Bloque objetivo:

- `perception/bg_features/bg_rings/sanitize_mask`

Cambio:

- Se añadió `bg_local.sanitize.mode` con dos variantes:
  - `morphology`: camino anterior basado en fill holes + close/open
  - `convex_hull`: saneado por envolvente convexa del objeto
- El objetivo es poder comparar una versión más simple y previsiblemente más barata para construir anillos de fondo sin depender de agujeros internos.

Impacto esperado:

- reducir el coste de `sanitize_mask` cuando la envolvente convexa sea suficiente para el uso de fondo local;
- facilitar comparación A/B sin perder el modo anterior.

Riesgo:

- `Riesgo bajo`

Validación:

- `python3 -m py_compile` sobre `REMIND/features/background_features.py`

Notas:

- `convex_hull` puede sobreexpandir el objeto frente a la máscara saneada por morfología, así que el anillo puede empezar algo más lejos en zonas cóncavas.
- El modo por defecto se mantiene en `morphology`.

### 30. `features/dino_extractor.py` + config - `patch_coverage` con `INTER_AREA`

Archivos:

- `REMIND/features/dino_extractor.py`
- `REMIND/config/default_config.yaml`
- `REMIND/config/CONFIG_TIERS.md`

Bloque objetivo:

- `perception/bg_features/bg_rings/mask_to_patch_coverage`
- y cualquier otro uso de `dino.mask_px_to_patch_coverage(...)`

Cambio:

- Se añadió `dino.patch_coverage_mode` con dos caminos:
  - `resize_area`: reduce la máscara a patch-space con `cv2.resize(..., INTER_AREA)`
  - `reshape_mean`: mantiene la media exacta por bloques del método anterior
- El modo por defecto pasa a `resize_area` para probar una ruta más rápida en C/OpenCV sin tocar la interfaz de los extractores.

Impacto esperado:

- reducir el coste de proyección de máscaras a patch-space, especialmente en `bg_rings`, `object_features` y `part_features`.

Riesgo:

- `Riesgo bajo`

Validación:

- `python3 -m py_compile` sobre `REMIND/features/dino_extractor.py`
- benchmark local comparando `resize_area` frente a `reshape_mean` con diferencias numéricas solo de redondeo flotante.

Notas:

- `resize_area` no debería cambiar la semántica práctica del coverage cuando la reducción es por bloques regulares, pero puede introducir diferencias mínimas de coma flotante frente al camino exacto.
- Si hubiera cualquier duda de comportamiento, basta con volver a `dino.patch_coverage_mode: "reshape_mean"`.

### 31. `sets_search.py` + `outcome_policy.py` - menos sobrecoste Python en `neighbor_sets` y `temporal_reconcile`

Archivos:

- `REMIND/association/scores/sets/sets_search.py`
- `REMIND/association/scores/sets/sets_graph_utils.py`
- `REMIND/association/policy/outcome_policy.py`

Bloque objetivo:

- `association/neighbor_sets/run_beam_search/expand_beam_for_class/transition`
- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states`
- parte Python de `association/post_assignment/temporal_reconcile`

Cambio:

- En `neighbor_sets`:
  - la unión ordenada de IDs en transiciones del beam deja de usar `set(...) + sorted(...)` sobre todo el acumulado y pasa a una mezcla lineal sobre secuencias ya ordenadas;
  - se añade una caché por frame para `score_state_quick` indexada por la firma efectiva del estado, evitando recalcular el mismo score cuando varias rutas del beam colapsan al mismo estado útil.
  - el estado del beam deja de copiar `pairs` completos en cada transición y solo materializa esa traza al construir las hipótesis finales.
- En `postcreate_temporal`:
  - se reutilizan los scores temporales ya calculados para candidatos `ambiguity` y `raw` dentro de la misma detección;
  - las ordenaciones, comparaciones y tablas debug reutilizan esos valores en vez de relanzar `temporal_candidate_score(...)` varias veces.
  - la desambiguación known-set-distance deja de construir `anchor_breakdown` y detalle fino de orden para todas las asignaciones enumeradas; ahora puntúa primero en modo ligero y solo recompone ese detalle para las top soluciones que realmente se exponen en debug.

Impacto esperado:

- bajar sobrecoste fijo Python en la expansión y puntuación del beam de `neighbor_sets`;
- bajar trabajo repetido por detección dentro de `temporal_reconcile`, especialmente con debug activo.

Riesgo:

- `Sin impacto esperado`

Validación:

- `python3 -m py_compile` sobre:
  - `REMIND/association/scores/sets/sets_search.py`
  - `REMIND/association/scores/sets/sets_graph_utils.py`
  - `REMIND/association/policy/outcome_policy.py`

Notas:

- La caché de `score_state_quick` usa solo los campos que realmente afectan al score rápido; no depende de detalles debug ni de la identidad del `dict` de estado.
- En `postcreate_temporal` no cambia ningún threshold ni el criterio de soporte/contexto; solo se reutilizan resultados ya equivalentes.

### 32. `neighbor_sets_score.py` + `sets_options.py` + `sets_search.py` + `sets_scoring.py` - bitmasks por frame en el beam de `neighbor_sets`

Archivos:

- `REMIND/association/scores/sets/neighbor_sets_score.py`
- `REMIND/association/scores/sets/sets_options.py`
- `REMIND/association/scores/sets/sets_search.py`
- `REMIND/association/scores/sets/sets_scoring.py`

Bloque objetivo:

- `association/neighbor_sets/run_beam_search/expand_beam_for_class/class_options`
- `association/neighbor_sets/run_beam_search/expand_beam_for_class/transition`
- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states`
- `association/neighbor_sets/collect_hypotheses`

Cambio:

- Se añadió una representación por frame basada en bitmasks para IDs de objetos y detecciones candidatas.
- Las `class_options` precalculan `object_mask` y `det_mask`.
- El beam reutiliza esos masks para:
  - comprobar solapes de objetos ya usados;
  - propagar uniones de objetos/detecciones;
  - construir claves de caché y de deduplicación más baratas.
- La materialización de tuplas ordenadas de IDs se retrasa a los puntos donde realmente hace falta para scoring final o salida/debug.

Impacto esperado:

- bajar sobrecoste Python en la expansión del beam;
- bajar parte del coste fijo en `score_beam_states`;
- reducir churn de `set`/tuplas/listas en frames con muchos estados intermedios.

Riesgo:

- `Bajo`

Validación:

- `python3 -m py_compile` sobre:
  - `REMIND/association/scores/sets/neighbor_sets_score.py`
  - `REMIND/association/scores/sets/sets_options.py`
  - `REMIND/association/scores/sets/sets_search.py`
  - `REMIND/association/scores/sets/sets_scoring.py`
- revisión manual de que:
  - no cambian thresholds, pesos ni fórmulas de score;
  - el cambio afecta a representación interna, caché y filtrado, no a la semántica objetivo.

Notas:

- El riesgo no es cero porque cambia la representación interna del estado del beam y el momento exacto en que algunas secuencias de IDs se materializan.
- En pruebas posteriores, este camino mostró degradación de resultados al menos en un vídeo propio, con caída aproximada de `-6%` en score, así que no debe considerarse equivalente por defecto.
- El cambio queda mantenido detrás de config con dos modos:
  - `association.scores.neighbor_sets.beam_state_mode: "classic"`
  - `association.scores.neighbor_sets.beam_state_mode: "bitmask_full"`
- El valor por defecto recomendado es `"classic"` hasta completar validación A/B sobre secuencias reales.

### 33. `neighbor_distance_graph.py` + `update_general.py` + `known_set_distance_disambiguator.py` - `bbox` de máscara perezoso (lazy) en observaciones relacionales

Archivos:

- `REMIND/memory/neighbor_distance_graph.py`
- `REMIND/update/update_general.py`
- `REMIND/association/disambiguation/known_set_distance_disambiguator.py`

Bloque objetivo:

- `update/neighbor_graphs/dist_observations`
- cálculo de observaciones relacionales en desambiguación temporal (`known_set_distance`)

Cambio:

- `prepare_relation_mask_runtime(...)` ahora soporta `compute_bbox=False`.
- En rutas de alto volumen (`dist_observations` y registro geométrico para desambiguación) se prepara runtime de máscara sin calcular `bbox` upfront.
- La `bbox` se calcula de forma perezosa solo si realmente se necesita el camino de gap exacto máscara-a-máscara.
- Se mantiene el mismo `touches_border` y la misma geometría final cuando se ejecuta el cálculo exacto.

Impacto esperado:

- reducir coste fijo por objeto cuando hay muchos pares y gran parte no requiere gap exacto;
- bajar tiempo de `dist_observations` en escenas con muchas máscaras/objetos visibles.

Riesgo:

- `Sin impacto esperado`

Validación:

- `python3 -m py_compile` sobre los tres archivos modificados.

Notas:

- El resultado geométrico no cambia: solo cambia el momento en que se materializa la `bbox` de máscara.

### 34. `outcome_policy.py` + `data_association.py` + `frame_association_flow.py` - cachés runtime por frame para `temporal_reconcile`

Archivos:

- `REMIND/association/policy/outcome_policy.py`
- `REMIND/association/engine/data_association.py`
- `REMIND/association/flow/frame_association_flow.py`

Bloque objetivo:

- `association/post_assignment/temporal_reconcile`
- especialmente recomputación de:
  - `iter_candidates(...)` por scope;
  - `compute_comparable_score_map(...)`;
  - `combine_comparable_pack(...)` sobre los mismos candidatos.

Cambio:

- Se añadieron cachés runtime en `AssociationOutcomePolicy` para:
  - candidatos por `(report, scope)`;
  - score comparable por candidato y firma de términos permitidos;
  - mapas comparables por firma de candidatos.
- `compute_comparable_score_map(...)` evita la segunda recomputación completa cuando el conjunto de términos comparables ya es idéntico en todos los candidatos.
- Se añadió reseteo explícito de cachés al inicio de cada frame (`FrameAssociationFlow.prepare_frame -> engine.reset_runtime_caches()`).

Impacto esperado:

- bajar sobrecoste Python repetido en `temporal_reconcile`;
- reducir llamadas duplicadas a `combine_comparable_pack` dentro de la misma detección/frame.

Riesgo:

- `Sin impacto esperado`

Validación:

- `python3 -m py_compile` sobre los archivos modificados.

Notas:

- No cambian thresholds, ni criterios de decisión, ni orden de prioridad entre políticas.
- El cambio es de memoización intra-frame y evita trabajo redundante equivalente.

### 35. `neighbor_distance_graph.py` - reutilización de `distanceTransform` por objeto (lazy) en gap exacto

Archivos:

- `REMIND/memory/neighbor_distance_graph.py`
- `REMIND/update/update_general.py`
- `REMIND/association/disambiguation/known_set_distance_disambiguator.py`

Bloque objetivo:

- `update/neighbor_graphs/dist_observations`
- observación relacional exacta en desambiguación de conocidos

Cambio probado:

- Se añadió caché lazy de `distanceTransform` por máscara runtime (`runtime["dt"]`) para reutilizar DT entre pares del mismo objeto en un frame.
- Además se probó inyectar `bbox` conocida en runtime para evitar escaneos extra.

Resultado real observado:

- Regresión fuerte en rendimiento (frame 1):
  - `update/neighbor_graphs/dist_observations`: `174.84 ms -> 299.37 ms` (`+124.53 ms`, ~`+71%`).
  - `total`: `1109.42 ms -> 1235.46 ms` (`+126.04 ms`).

Estado:

- `REVERTIDO`.

Riesgo:

- `No aceptable` (coste claramente peor).

Validación:

- Se revirtió el experimento en:
  - `REMIND/memory/neighbor_distance_graph.py`
  - `REMIND/update/update_general.py`
  - `REMIND/association/disambiguation/known_set_distance_disambiguator.py`
- `python3 -m py_compile` sobre los tres archivos tras revertir.

Notas:

- No repetir este enfoque tal cual; el coste extra de indexado/DT full-frame supera el ahorro esperado en este pipeline.

### 36. `neighbor_distance_graph.py` - cast bool de máscara diferido (lazy) para gap exacto

Archivos:

- `REMIND/memory/neighbor_distance_graph.py`

Bloque objetivo:

- `update/neighbor_graphs/dist_observations`
- observación relacional exacta cuando el par cae en camino de máscara

Cambio:

- `prepare_relation_mask_runtime(...)` deja de convertir siempre la máscara completa a `bool`.
- El runtime guarda la máscara original y solo materializa `mask_bool` cuando `_signed_mask_gap_px(...)` realmente entra al cálculo exacto.
- Se mantiene caché de `bbox` y `touches_border` en runtime, sin alterar la lógica de decisión.

Impacto esperado:

- reducir coste fijo por objeto en frames con muchos pares que terminan en camino `bbox` (lejanos);
- bajar trabajo inútil de cast completo cuando no se usa `gap` exacto.

Riesgo:

- `Sin impacto esperado`

Validación:

- `python3 -m py_compile REMIND/memory/neighbor_distance_graph.py`

Notas:

- Optimización semánticamente neutra: cambia solo cuándo se hace el cast, no el resultado.

### 37. `update_general.py` + config - paralelización por pares en `dist_observations` con gate por carga

Archivos:

- `REMIND/update/update_general.py`
- `REMIND/config/default_config.yaml`
- `REMIND/config/CONFIG_TIERS.md`

Bloque objetivo:

- `update/neighbor_graphs/dist_observations`

Cambio:

- Se paraleliza el cálculo de observaciones relacionales por par con `ThreadPoolExecutor`.
- Se mantiene exactamente el mismo `compute_relation_observation(...)`; solo cambia el scheduling.
- La paralelización se activa solo a partir de un umbral de pares (`min_pairs`) para evitar overhead en frames pequeños.
- Se añadieron parámetros:
  - `update.neighbor_graphs.dist_observations_parallel.enabled`
  - `update.neighbor_graphs.dist_observations_parallel.min_pairs`
  - `update.neighbor_graphs.dist_observations_parallel.workers`
  - `update.neighbor_graphs.dist_observations_parallel.max_auto_workers`

Impacto esperado:

- bajar `dist_observations` en frames con muchos pares y camino de gap exacto pesado.

Riesgo:

- `Sin impacto esperado` en salida semántica (mismo cálculo por par).
- Riesgo operativo bajo: posible no-mejora o empeoramiento por overhead en algunas máquinas.

Validación:

- `python3 -m py_compile REMIND/update/update_general.py`

Notas:

- Si empeora, desactivar rápido con:
  - `update.neighbor_graphs.dist_observations_parallel.enabled: false`
  - o subir `min_pairs`.

### 38. `update_general.py` + config - paralelización por objeto en `graph_updates`

Archivos:

- `REMIND/update/update_general.py`
- `REMIND/config/default_config.yaml`
- `REMIND/config/CONFIG_TIERS.md`

Bloque objetivo:

- `update/neighbor_graphs/graph_updates`

Cambio:

- Se paraleliza la aplicación de updates de grafo por objeto (`NeighborUpdater.update(...)`) con `ThreadPoolExecutor`.
- Cada objeto actualiza su propio `neighbors`/`neighbor_dist`; no se cambia la lógica interna por objeto.
- Se añade gate por número de objetos para evitar overhead en escenas pequeñas.
- Parámetros añadidos:
  - `update.neighbor_graphs.graph_updates_parallel.enabled`
  - `update.neighbor_graphs.graph_updates_parallel.min_objects`
  - `update.neighbor_graphs.graph_updates_parallel.workers`
  - `update.neighbor_graphs.graph_updates_parallel.max_auto_workers`

Impacto esperado:

- recortar el nuevo cuello tras acelerar `dist_observations`, especialmente con muchos objetos visibles.

Riesgo:

- `Sin impacto esperado` en semántica de resultado.
- Riesgo operativo bajo: posible no-mejora por overhead en hardware con pocos cores.

Validación:

- `python3 -m py_compile REMIND/update/update_general.py`

Notas:

- Si empeora, rollback rápido vía config:
  - `update.neighbor_graphs.graph_updates_parallel.enabled: false`
  - o subir `min_objects`.

### 39. `neighbor_sets` - modo híbrido `bitmask_used` para acelerar transición/cache sin cambiar semántica

Archivos:

- `REMIND/association/scores/sets/neighbor_sets_score.py`
- `REMIND/association/scores/sets/sets_search.py`
- `REMIND/association/scores/sets/sets_scoring.py`
- `REMIND/association/scores/sets/sets_graph_utils.py`
- `REMIND/config/default_config.yaml`

Bloque objetivo:

- `association/neighbor_sets/run_beam_search/expand_beam_for_class/transition`
- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick`

Cambio:

- Se añade modo de beam `bitmask_used`:
  - usa bitmask para chequeo de solape de objetos y claves de caché;
  - mantiene representación clásica de uniones/detecciones para no tocar deduplicación/salida final.
- `transition_state(...)` conserva `obj_mask` incremental también en modo no-full, reduciendo sobrecoste de checks/reconstrucciones.
- `state_score_cache_key(...)` prioriza `obj_mask` cuando está disponible.
- `score_state_quick(...)` reutiliza cachés por máscara para `density` y `maturity` (evita parte de materialización/re-hash por tuplas).
- El valor por defecto pasa a `beam_state_mode: "bitmask_used"`.

Impacto esperado:

- reducir coste Python en transición del beam;
- recortar parte del coste de `score_state_quick` por claves más compactas.

Riesgo:

- `Bajo` (misma lógica de scoring/thresholds; cambia representación interna intermedia).

Validación:

- `python3 -m py_compile` sobre los archivos modificados.

Notas:

- Rollback inmediato por config:
  - `association.scores.neighbor_sets.beam_state_mode: "classic"`

### 40. `neighbor_sets` - fast-path por máscara en transición + scoring (sin cambiar score final)

Archivos:

- `REMIND/association/scores/sets/sets_search.py`
- `REMIND/association/scores/sets/sets_scoring.py`
- `REMIND/association/scores/sets/sets_options.py`
- `REMIND/association/scores/sets/sets_graph_utils.py`

Bloque objetivo:

- `association/neighbor_sets/run_beam_search/expand_beam_for_class/transition`
- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick`

Cambio:

- En `bitmask_used`, el estado del beam conserva `obj_mask` y `explained_det_mask` incrementales y difiere materialización de tuplas ordenadas (`obj_ids_sorted`, `explained_det_sorted`) hasta cuando se necesiten.
- `state_object_ids_sorted(...)` y `state_explained_det_sorted(...)` aceptan máscara también en `bitmask_used` (no solo en `bitmask_full`).
- `build_class_options(...)` precomputa un pequeño pack de transición (`_transition_pack`) y el `last_pair` (`_pair`) para reducir trabajo repetido en `transition_state(...)`.
- `score_state_quick(...)` usa `obj_mask.bit_count()` para `k` cuando hay máscara, evitando reconstrucción de IDs en la ruta caliente.
- `density_score_cached_by_mask(...)` incorpora ruta directa por índices de matriz candidata a partir de máscara, evitando conversión máscara->IDs->índices en cada estado.
- En `bitmask_used` también se genera `det_mask` por opción para conservar exactitud de detecciones explicadas sin reconstrucciones frecuentes.

Impacto esperado:

- reducción fuerte de overhead Python en transición del beam;
- recorte adicional de `score_state_quick/density_score_cached` en escenas con muchos estados/máscaras únicas.

Riesgo:

- `Bajo-Medio` (misma fórmula de score y mismos thresholds; cambia representación interna y timing de materialización).
- Si se detecta diferencia funcional, rollback por config:
  - `association.scores.neighbor_sets.beam_state_mode: "classic"`

Validación:

- `python3 -m py_compile REMIND/association/scores/sets/neighbor_sets_score.py REMIND/association/scores/sets/sets_search.py REMIND/association/scores/sets/sets_scoring.py REMIND/association/scores/sets/sets_options.py REMIND/association/scores/sets/sets_graph_utils.py`

### 41. `neighbor_sets` - fast-path adicional en `density`/`maturity` por máscara (sin cambios de criterio)

Archivos:

- `REMIND/association/scores/sets/sets_graph_utils.py`

Bloque objetivo:

- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick/density_score_cached`
- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick/maturity_pack_cached`

Cambio:

- En densidad por máscara se añade ruta especializada para sets pequeños (`n<=10`) que evita submatrices temporales (`np.ix_`) y calcula MST/coberturas directamente por índices candidatos.
- Se añade tabla por bit de madurez de objeto (`_frame_object_maturity_by_bit_index`) y cálculo de `maturity_pack` directamente desde máscara, evitando `mask -> object_ids -> múltiples lookups` en la ruta caliente.
- Se mantiene fallback al camino previo cuando no hay mapeo por bit disponible, preservando comportamiento.

Impacto esperado:

- recorte adicional en `density_score_cached` cuando hay muchas máscaras únicas;
- recorte de `maturity_pack_cached` por menos materialización y menos accesos de diccionario.

Riesgo:

- `Bajo` (misma formulación de densidad/madurez, solo cambia el camino de cómputo y cache).

Validación:

- `python3 -m py_compile REMIND/association/scores/sets/sets_graph_utils.py REMIND/association/scores/sets/sets_search.py REMIND/association/scores/sets/sets_scoring.py REMIND/association/scores/sets/sets_options.py REMIND/association/scores/sets/neighbor_sets_score.py`

### 42. `dino_extractor` + `neighbor_distance_graph` + `assignment_result_applier` + `sets_scoring` - recortes de overhead Python en rutas calientes

Archivos:

- `REMIND/features/dino_extractor.py`
- `REMIND/memory/neighbor_distance_graph.py`
- `REMIND/association/engine/assignment_result_applier.py`
- `REMIND/association/scores/sets/sets_scoring.py`

Bloques objetivo:

- `perception/bg_features/bg_inner_global`
- `perception/bg_features/bg_outer_global`
- `update/neighbor_graphs/dist_observations`
- `association/post_assignment/temporal_reconcile`
- `association/neighbor_sets/run_beam_search/expand_beam_for_class/score_beam_states/score_state_quick`

Cambio:

- `dino_extractor`: en pooling por máscara se evitan copias innecesarias (`astype(..., copy=False)`), y se evita recalcular `sum()` de pesos cuando no hace falta.
- `neighbor_distance_graph`: `compute_relation_observation(...)` calcula métricas de bbox en una sola pasada (gap/proyecciones/intersección/centro dentro), evitando parseos y llamadas repetidas por par.
- `assignment_result_applier`: en `temporal_reconcile`, la agregación de debug por pass (`components`/`pair_anchors`) se hace en una sola pasada, sin reconstrucción posterior.
- `sets_scoring`: `score_state_quick(...)` usa agregados de clase inline (sin crear diccionario intermedio por estado), manteniendo las mismas fórmulas.

Impacto esperado:

- bajar coste en rutas con muchas iteraciones por frame:
  - pooling de fondo por detección;
  - observaciones de distancia por pares;
  - scoring rápido de estados del beam;
  - consolidación de debug en reconciliación temporal.

Riesgo:

- `Sin impacto esperado` (no cambia criterios de decisión; solo elimina trabajo duplicado/copias intermedias).

Validación:

- `python3 -m py_compile REMIND/features/dino_extractor.py REMIND/memory/neighbor_distance_graph.py REMIND/association/engine/assignment_result_applier.py REMIND/association/scores/sets/sets_scoring.py`

### 43. `utils/math.py` - `kmeans_np` más eficiente para `parts_kmeans` (sin cambiar API)

Archivo:

- `REMIND/utils/math.py`

Bloque objetivo:

- `perception/parts_features/parts_kmeans`
- también afecta a k-means usados en fondo/prototipos cuando estén activos

Cambio:

- Se optimiza `kmeans_np(...)` manteniendo la misma firma y salidas:
  - camino `float32` cuando la entrada ya es `float32` (evita promoción innecesaria);
  - asignación de labels por centro con buffers reutilizados (`best_dist`, `best_lab`) para reducir temporales;
  - actualización de centros por clusters ordenados (evita `np.add.at` en la ruta caliente);
  - uso de operaciones in-place (`np.minimum(..., out=...)`) para reducir allocaciones.

Impacto esperado:

- recorte significativo de tiempo en `parts_kmeans` cuando hay muchos objetos/patches por frame.

Riesgo:

- `Riesgo bajo`: no cambia criterios de selección ni contrato de la función, pero al usar `float32` en ese camino puede haber diferencias numéricas finas respecto al cálculo previo en `float64`.

Validación:

- `python3 -m py_compile REMIND/utils/math.py REMIND/features/part_features.py`
- prueba sintética local de sanidad de shapes/inercia para varios `(N,D,K)`.

### 44. `candidate_generation.py` - eliminar doble combinación de similitud en `sim_candidates`

Archivo:

- `REMIND/association/engine/candidate_generation.py`

Bloque objetivo:

- `association/sim_candidates`

Cambio:

- En `process_one_detection(...)`, los candidatos se crean con `scores` base y se difiere la combinación de similitud (`combine_pack`) hasta la pasada consistente (`apply_consistent_similarity_policy`).
- Se mantiene intacto el comportamiento externo de `build_similarity_candidate(...)` para otros flujos (por ejemplo `update/memory_manager.py`), que sigue calculando el pack inmediato por defecto.
- Se factoriza la escritura de campos de score/calidad en `apply_similarity_pack(...)` para reutilizar exactamente la misma asignación tanto en la ruta inmediata como en la consistente.

Impacto esperado:

- reducir de forma notable coste Python en `sim_candidates` cuando hay muchos objetos por clase (se elimina una combinación completa por candidato en ese flujo).

Riesgo:

- `Bajo` (el report final sigue usando el mismo `combine_consistent_pack` que ya se aplicaba antes; solo se elimina trabajo intermedio que se sobrescribía).

Validación:

- `python3 -m py_compile REMIND/association/engine/candidate_generation.py`

### 45. `candidate_score_policy.py` - precálculo de top-k frame-local para evitar resortes repetidos en tablas de asignación

Archivo:

- `REMIND/association/policy/candidate_score_policy.py`

Bloque objetivo:

- `association/hungarian/assign_classes/score_tables`
- `association/hungarian/assign_classes/solve/score_tables`

Cambio:

- Se añade `build_frame_local_plausible_source_topk_by_det(...)` para calcular una vez por frame (y umbral) los `top-k` plausibles por detección.
- `build_score_tables(...)` reutiliza ese precálculo para construir los `frame_local_ctx_kernel_ids` de cada detección sin volver a recorrer/ordenar todos los candidatos de todas las demás detecciones en cada iteración.
- Se conserva el mismo criterio de filtrado y el mismo orden lógico (iteración de `reports` + ranking por `(score_sim, object_id)` descendente).

Impacto esperado:

- bajar overhead en construcción de `score_tables`, especialmente en escenas con muchas detecciones/candidatos.

Riesgo:

- `Bajo` (misma regla de selección, mismo `top-k`, mismo umbral; cambia solo la estrategia de cálculo para evitar repetición).

Validación:

- `python3 -m py_compile REMIND/association/policy/candidate_score_policy.py`

## Cambios explícitamente revertidos

- El experimento de `benchmark_mode` en `main.py` y config fue revertido.
- No forma parte del estado actual.
- El experimento de precalentado por frame de métricas de pares candidatas en `neighbor_sets` fue revertido.
- No mostró mejora apreciable frente al coste adicional de arranque del frame.

## Cómo revisar una posible regresión futura

### Si falla algo geométrico

Revisar primero:

- `REMIND/memory/neighbor_distance_graph.py`

Especialmente:

- shortcut por `exact_gap_max_n`
- definición de `gap_quality`
- cambio de cálculo exacto por ROI

### Si fallan sets o prioridades de `neighbor_sets`

Revisar primero:

- `REMIND/association/scores/sets/sets_graph_utils.py`
- `REMIND/association/scores/sets/sets_search.py`
- `REMIND/association/scores/sets/sets_scoring.py`
- `REMIND/association/scores/sets/sets_options.py`

### Si sube el tiempo pero no falla nada

Mirar primero:

- si el debug visual o de asociación está activado
- si el número de objetos/pares por frame ha subido mucho
- si `density_score_cached`, `dist_observations` o `temporal_reconcile` han cambiado de peso relativo

## Próximos candidatos de optimización

Pendientes con mejor pinta:

- `REMIND/detection/davis_segmenter.py`
- `REMIND/features/background_features.py`
  - especialmente prototipos y construcción de anillos
- `REMIND/features/part_features.py`
  - reducir más coste de `parts_kmeans`
- `REMIND/association/engine/assignment_result_applier.py`
  - revisar `temporal_reconcile`
- `REMIND/features/object_features.py`
  - posible reutilización de cobertura/máscara para evitar trabajo duplicado

## Regla para futuras entradas

Cada vez que se añada una optimización nueva, documentar:

1. archivo(s)
2. bloque de timing objetivo
3. descripción corta
4. riesgo sobre comportamiento
5. validación realizada
