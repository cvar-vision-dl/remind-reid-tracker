# Trazabilidad visual de `association`

## Objetivo

Definir una capa de trazabilidad visual fiel al código real del módulo de
`association`, sin alterar comportamiento y sin introducir lógica nueva.

La idea es construir dos piezas desacopladas:

1. Un `pipeline schema` fijo, que represente la estructura real del pipeline.
2. Un `execution trace` dinámico, que represente lo que ocurrió en una
   ejecución concreta.

La visualización final debe permitir responder con claridad:

- qué camino siguió cada detección;
- qué reglas se evaluaron;
- con qué valores;
- qué candidatos cayeron o sobrevivieron;
- por qué la decisión final fue la que fue.

## Estado actual

La instrumentación JSON ya existe y el contrato `v1` que debe consumir el
visor queda fijado en:

- `APP2/Src/association_trace/ASSOCIATION_TRACE_CONTRACT.md`

## Principios

### 1. Fidelidad al código

La visualización no debe reinterpretar el pipeline ni simplificarlo de forma
artificial.

- si en código hay 3 fases reales, en la visualización habrá 3 fases;
- si una regla importante existe en código, debe existir en el layout;
- si una rama no existe en código, no debe aparecer.

### 2. Unidad de análisis

La unidad principal será `frame_id + class_id`.

Esto permite representar correctamente decisiones conjuntas entre varias
detecciones de una misma clase, por ejemplo en `locks`, `Hungarian` o
ambigüedad estructural postasignación.

Dentro de esa unidad debe poder enfocarse el camino de una detección concreta.

### 3. Profundidad útil

La traza debe ser profunda, pero orientada a decisión.

Sí deben aparecer:

- activación o desactivación de contexto;
- reglas que descartan o rescatan candidatos;
- scores y thresholds que alteran ranking útil;
- decisiones globales;
- guardas postasignación;
- `final_decision` y `final_reason`.

No hace falta incluir en una primera versión:

- todas las variables temporales internas;
- todos los subdetalles de cálculo que no cambian ramas;
- todo el detalle fino geométrico o de hipótesis si no altera decisión.

### 4. Instrumentación opcional

La traza visual debe ser opcional.

- en `main` normal puede ir activada;
- en testing o ejecuciones finales del tracker puede ir desactivada;
- si está desactivada, el coste extra debe ser mínimo.

La implementación futura debería usar una capa de instrumentación enchufable,
idealmente con un `collector` no-op cuando la flag esté apagada.

## Modelo general

### `pipeline schema`

Representa el pipeline "en vacío". Debe ser estable y cambiar solo cuando
cambie el pipeline real.

Debe definir:

- fases;
- regiones;
- nodos;
- conexiones;
- tipo de nodo;
- scope del nodo;
- orden visual.

### `execution trace`

Representa lo ocurrido en una ejecución concreta de `frame_id + class_id`.

Debe guardar:

- nodos visitados;
- checks evaluados;
- valores observados;
- ramas seguidas;
- candidatos vivos o descartados;
- decisiones finales por detección.

## Layout conceptual

La orientación prevista es `top_down`.

- arriba: `Inputs`;
- abajo: `Final Outcome`.

El layout no debe ser un grafo automático arbitrario, sino un layout semántico
derivado del pipeline real.

- el eje vertical representa progreso lógico del pipeline;
- el eje horizontal representa regiones o familias paralelas;
- las fases deben verse como bandas diferenciadas;
- nodos hermanos de una misma fase deben quedar en la misma región;
- nodos globales deben aparecer centrados y destacados.

Las detecciones se visualizarán como recorridos de distinto color sobre un
grafo compartido por clase.

El color principal identifica detección, no resultado.

El resultado de cada regla se representará con checks y estado local del nodo:

- `PASS`
- `FAIL`
- `SOFT`
- `N/A`

## Cobertura mínima obligatoria

No puede faltar ninguna regla que:

- active o desactive `sets` o `distance`;
- elimine o rescate candidatos;
- cambie el ranking útil previo a resolución;
- altere `locks` o `Hungarian`;
- reinterprete una asignación en postasignación;
- explique `final_decision` o `final_reason`.

Regla práctica:

- si una condición puede cambiar rama, descarte, asignación u outcome, debe
  existir como nodo visible o como `check` visible.

## Nodos `v0.1`

Lista mínima inicial de nodos para la primera versión:

- `prepare.class_partition`
- `prepare.reliable_visual_anchors`
- `prepare.valid_detections`
- `visual.build_candidates`
- `visual.report_diagnosis`
- `context.neighbor_sets_hypotheses`
- `context.sets_activation`
- `shape.allow_for_report`
- `shape.context_veto`
- `shape.final_score_tables`
- `resolve.locks`
- `resolve.hungarian`
- `post.identity_stability`
- `post.assignment_ambiguity`
- `post.known_set_distance_disambiguation`
- `post.create_competition`
- `post.ambiguous_track_candidates`
- `post.provisional_reconciliation`
- `post.final_decision_pack`
- `outcome.final_ambiguity`
- `outcome.finalize`

## Campos comunes de un `node_run`

Todos los nodos deben poder emitir el mismo bloque base:

- `node_id`
- `entered`
- `skipped_reason`
- `scope_key`
- `participants`
- `checks`
- `values`
- `decision`

Semántica:

- `entered`: indica si el nodo se ejecutó realmente;
- `skipped_reason`: explica por qué no aplicó;
- `scope_key`: identifica el contexto del nodo, al menos `frame_id` y
  `class_id`;
- `participants`: detecciones y objetos implicados;
- `checks`: reglas que pueden cambiar el camino;
- `values`: resultados o contexto producido por el nodo;
- `decision`: rama o salida tomada.

## Campos comunes de un `check`

Todo check visible debe poder representarse así:

- `id`
- `label`
- `lhs`
- `op`
- `rhs`
- `passed`
- `reason`
- `effect`

Ejemplo:

```json
{
  "id": "sets.quality",
  "label": "sets_quality",
  "lhs": 0.41,
  "op": ">=",
  "rhs": 0.60,
  "passed": false,
  "reason": "LOW_KERNEL_MARGIN",
  "effect": "disable_sets_context"
}
```

## Qué va a `checks`

Va a `checks` todo lo que funcione como puerta o condición de bifurcación.

Ejemplos:

- activación de `sets` o `distance`;
- pertenencia a shortlist;
- veto contextual;
- thresholds de calidad;
- márgenes de ambigüedad;
- reglas de estabilidad;
- criterios de resolvabilidad;
- transición a `PROVISIONAL_*` o `AMBIGUOUS`.

## Qué va a `values`

Va a `values` todo lo que explique el resultado del nodo sin ser una puerta en
sí misma.

Ejemplos:

- listas de detecciones y objetos;
- shortlist resultante;
- `prior_by_oid`;
- scores por candidato;
- asignaciones de Hungarian;
- componentes ambiguos;
- anchors candidatos;
- `final_decision` y `final_reason`.

## Tipos de nodo y scope

Tipos previstos:

- `gate`
- `score`
- `resolver`
- `decision`
- `outcome`

Scopes previstos:

- `class`
- `detection`
- `candidate`
- `global`

## Blueprint `v0.1` por nodo

### `prepare.class_partition`

- type: `decision`
- scope: `class`
- values:
  - `det_ids`
  - `snapshot_object_ids`
- decision:
  - conjunto real de trabajo por clase

### `prepare.reliable_visual_anchors`

- type: `gate`
- scope: `class`
- checks:
  - existencia de anchors visuales fiables
- values:
  - `anchor_object_ids`
  - `anchor_det_by_object_id`

### `prepare.valid_detections`

- type: `gate`
- scope: `detection`
- checks:
  - detección usable
- values:
  - `valid_det_ids`
  - `invalid_det_ids`
  - `invalid_reason`

### `visual.build_candidates`

- type: `score`
- scope: `candidate`
- values por `det_id`:
  - `candidate_object_ids`
  - `score_sim`
  - `best_candidate_id`
  - `second_candidate_id`

### `visual.report_diagnosis`

- type: `decision`
- scope: `detection`
- checks:
  - reglas reales de `STRONG / AMBIGUOUS / WEAK`
- values:
  - `match_diag_sim.status`
  - `best_score`
  - `second_score`
  - `gap`

### `context.neighbor_sets_hypotheses`

- type: `score`
- scope: `class`
- checks:
  - `neighbor_sets_output_available`
  - `neighbor_sets_found_hypotheses`
- values:
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
  - `shortlist_object_ids`
  - `anchor_object_ids`
  - `prior_object_ids`
  - filas `hypothesis`
  - filas `object_support`

Lectura recomendada:

- este nodo enseña hipótesis retenidas por el pipeline, no un total exhaustivo;
- la tabla por objeto puede incluir apoyo adicional derivado del kernel
  contextual, aunque el objeto no destaque en el top visible de hipótesis.

### `context.sets_activation`

- type: `gate`
- scope: `class`
- checks:
  - `sets_context_built`
  - `n_hypotheses > 0`
  - `k_best >= min_size`
  - `best >= min_best_score`
  - `coverage_eff >= min_coverage_eff`
  - `quality >= min_quality`
  - gate agregado `global_ok`
- values:
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
  - `shortlist_object_ids`
  - `anchor_object_ids`
  - `prior_object_ids`

Lectura recomendada:

- este nodo no construye hipótesis nuevas ni rankea objetos;
- solo decide si el contexto ya construido entra con peso real en la
  asociación posterior.
  - `k_best`
  - `n_hypotheses`
  - `shortlist_object_ids`
  - `anchor_object_ids`
  - `prior_object_ids`
  - thresholds reales
  - `quality_terms`

### `shape.allow_for_report`

- type: `gate`
- scope: `detection`
- checks:
  - `report_status in allowed_status`
  - override por solape con `used_object_ids`
- values:
  - `stage` (`pre_locks` o `post_locks`)
  - `report_status`
  - `allowed`
  - `allowed_by_used_object_overlap`
  - `reason`
  - resumen de clase:
    - `class_context_available`
    - `allowed_count`
    - `blocked_count`
    - `used_object_ids_count`

### `shape.context_veto`

- type: `gate`
- scope: `candidate`
- checks:
  - plausibilidad conocida
  - gates reales de entrada
  - rescate contextual
  - veto contextual y filtros duros
- values por `(det_id, object_id)`:
  - `known_plausible_keep`
  - `known_plausible_reason`
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
  - `kernel_rel`
  - `hyp_rel`
  - si el veto es local:
    - `local_ctx_episode_count`
    - `local_ctx_kernel_source`
    - `local_ctx_kernel_size`
    - `local_ctx_expected_count`
    - `local_ctx_hit_count`
    - `local_ctx_hit_ratio`
    - `local_ctx_maturity`

Lectura recomendada:

- este nodo no representa solo un veto binario;
- aquí conviven plausibilidad conocida, gates, rescates y contradicción
  contextual antes de construir las tablas operativas.

### `shape.final_score_tables`

- type: `score`
- scope: `candidate`
- values por `det_id`:
  - `candidate_count`
  - `best_object_id`
  - `best_score_final`
- values por `(det_id, object_id)`:
  - `score_sim`
  - `score_assign`
  - `score_final`
  - `score_sets`
  - `bonus_sets`
  - `score_ctx_local`
  - `score_ctx_global`
  - `gate_reason`
  - `rank`

Lectura recomendada:

- `score_assign` es la señal que optimiza Hungarian;
- `score_final` se usa después para aceptar o rechazar la asignación elegida.

### `resolve.locks`

- type: `resolver`
- scope: `global`
- checks:
  - `locks.object`: lock por mejor deteccion para cada objeto
  - `locks.detection`: lock por mejor objeto para cada deteccion
  - thresholds reales de lock (`locks_thr`, gap absoluto y gap relativo)
- values:
  - `candidate_det_count`
  - `candidate_object_count`
  - `locked_count`
  - `locked_det_ids`
  - `locked_object_ids`
- values por `det_id` bloqueada:
  - `locked`
  - `locked_object_id`
  - `score_final`
  - `lock_source` (`object`, `detection`, `unknown`)
  - `lock_modes`

### `resolve.hungarian`

- type: `resolver`
- scope: `global`
- checks por `det_id`:
  - `assigned_to_real_object`
  - `selected_score_final_reaches_match_thr`
  - `selected_score_sim_reaches_min_match_score`
- values:
  - `participant_det_ids`
  - `participant_object_ids`
  - `object_column_count`
  - `dummy_column_count`
  - `total_column_count`
  - `n_matches`
  - `n_creates`
  - `create_det_ids`
- values por `det_id`:
  - `assigned_kind`
  - `assigned_object_id`
  - `selected_score_assign`
  - `selected_score_sim`
  - `selected_score_final`
  - `dummy_score`
  - `final_action`
  - `reason`
- values por `(det_id, object_id)`:
  - `rank`
  - `selected`
  - `score_assign`
  - `score_sim`
  - `score_final`

Lectura recomendada:

- la optimización compite contra objetos y dummies;
- una asignación a objeto real no se convierte automáticamente en match si no
  supera los umbrales duros posteriores.

### `post.identity_stability`

- type: `gate`
- scope: `detection`
- checks:
  - regla real de estabilidad
- values:
  - `proposed_match`
  - `accepted`
  - `reason`

### `post.assignment_ambiguity`

- type: `decision`
- scope: `global`
- checks:
  - reglas de ambigüedad estructural
- values:
  - `component_det_ids`
  - `component_object_ids`
  - `best_assignment`
  - `second_assignment`
  - `ambiguity_reason`

### `post.known_set_distance_disambiguation`

- type: `resolver`
- scope: `global`
- checks:
  - criterios reales de resolvabilidad
- values:
  - resumen por pasada con `input_det_ids`, `resolved_det_ids` y
    `remaining_det_ids`
  - componentes evaluados con `best_score`, `core_score`, `core_gap`, `gap`
  - `pair_anchors` con su `pass_index`

### `post.create_competition`

- type: `decision`
- scope: `global`
- checks:
  - reglas reales de competición entre `create`
- values:
  - `create_candidates`
  - `winners`
  - `losers`
  - `decision_reason`

### `post.ambiguous_track_candidates`

- type: `decision`
- scope: `detection`
- checks:
  - la detección tiene al menos una fuente de ambigüedad
  - la detección conserva varias alternativas comparables
- values:
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

Lectura recomendada:

- este nodo no resuelve nada todavía;
- solo materializa la bolsa ambigua real que pasará a known-set-distance y a
  la reconciliación temporal.

### `post.provisional_reconciliation`

- type: `decision`
- scope: `detection`
- checks:
  - contexto suficiente o `visual_fallback`
  - estado temporal permitido o excepción real
  - reglas reales de `provisional_parent`
- values:
  - `focus_source`
  - `context_mode`
  - `support_mode`
  - `relation`
  - `support_known_ids`
  - `support_known_scores`
  - `blocked_known_ids`
  - `blocked_known_scores`
  - `related_known_ids`
  - `related_known_scores`
  - `candidate_rows`

### `post.final_decision_pack`

- type: `decision`
- scope: `detection`
- checks:
  - `detection_has_final_bucket`
  - `match_survives_after_ambiguous_or_provisional`
- values:
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

Lectura recomendada:

- este nodo no cambia la evidencia del caso;
- solo arbitra la precedencia final entre ramas del post-assignment para que
  cada detección llegue al outcome con un único bucket semántico.

### `outcome.final_ambiguity`

- type: `decision`
- scope: `detection`
- values:
  - `status`
  - `reason`
  - `s1`
  - `s2`
  - `gap`
  - `n_close`
  - `confidence`
  - `final_decision`
  - `final_reason`

Lectura recomendada:

- este nodo recalcula la claridad final usando `score_final`;
- no cambia la semántica del caso, pero deja visible si el resultado final
  quedó fuerte, ambiguo o débil antes del outcome legible.

### `outcome.finalize`

- type: `outcome`
- scope: `detection`
- values:
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

Lectura recomendada:

- este nodo no abre ramas nuevas;
- solo anota de forma legible la decisión ya fijada por el pipeline y el
  soporte contextual o ambiguo que conviene conservar.

## Relación con el código actual

Este documento no redefine el pipeline. Solo propone cómo exponer visualmente
la estructura real descrita en:

- `APP2/Src/association/ASSOCIATION_DECISION_PATH.md`
- `APP2/Src/association/policy/POLICY_TAXONOMY.md`
- `APP2/Src/association_trace/ASSOCIATION_TRACE_NODE_MAP.md`
- `APP2/Src/association_trace/ASSOCIATION_TRACE_RUNTIME.md`

La evolución futura deseable es:

1. fijar un `schema` declarativo;
2. introducir una capa de instrumentación opcional;
3. emitir `execution traces` estructurados;
4. construir un visor offline que renderice `schema + trace`.

## Siguiente paso

Antes de tocar UI, conviene hacer:

1. una tabla final `node_id -> archivo -> símbolo -> datos a emitir`;
2. una propuesta de `collector` opcional;
3. un formato JSON estable para `pipeline_schema` y `execution_trace`.
