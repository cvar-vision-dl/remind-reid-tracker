# Camino de decisión de `association`

## Estado actual

Este documento es la referencia canónica de la organización actual del módulo
de asociación.

Los documentos detallados de `sets` y `distance` siguen siendo útiles para la
lógica local de cada bloque, pero la lectura global del pipeline debe hacerse
desde aquí y desde `REMIND/association/policy/POLICY_TAXONOMY.md`.

## Secuencia del frame

La asociación por frame se recorre en este orden:

1. `prepare_frame`
2. `build_visual_evidence`
3. `select_reliable_visual_anchors`
4. `activate_context_layers`
5. `diagnose_reports`
6. `resolve_global_assignment`
7. `apply_post_assignment_guards`
8. `finalize_outcomes`

Archivo principal:

- `REMIND/association/flow/frame_association_flow.py`

Fachada de entrada:

- `REMIND/association/engine/data_association.py`

## Capas del módulo

### 1. Evidencia visual

Responsabilidad:

- construir candidatos por detección
- calcular `score_sim`
- dejar trazas base de similitud

Archivos:

- `REMIND/association/engine/candidate_generation.py`
- `REMIND/association/scores/base_scores.py`
- `REMIND/association/similarity_computer.py`

### 2. Activación de contexto

Responsabilidad:

- decidir si `sets` o `distance` aportan contexto usable
- preparar contexto compacto para la decisión

Archivos:

- `REMIND/association/context/sets_provider.py`
- `REMIND/association/context/neighbor_sets_influence.py`

Nota actual:

- la infraestructura de `distance` existe, pero el camino principal de asociación
  la mantiene desacoplada de la decisión final en el estado actual del engine.

### 3. Candidate shaping

Responsabilidad:

- shortlist
- prior
- rescue
- soft-gate
- veto contextual
- construcción de tablas `score_sim / score_assign / score_final`

Fachada semántica actual:

- `REMIND/association/engine/candidate_shaping/score_path.py`

Implementación real hoy:

- `REMIND/association/policy/candidate_score_policy.py`
- `REMIND/association/policy/known_plausible_keep_policy.py`
- `REMIND/association/policy/sets_rule_policy.py`

Nota actual:

- no existe hoy una capa separada `candidate_context.py`;
- la parte contextual del shaping vive integrada en la construcción de filas y
  en las policies que controlan bonus, rescate, veto y plausibilidad conocida.
- `distance` queda reservado para memoria relacional y desambiguación
  post-asignación.

### 4. Resolución global

Responsabilidad:

- preparar el problema por clase
- resolver locks evidentes
- ejecutar Hungarian en el resto

Archivos:

- `REMIND/association/engine/assignment.py`
- `REMIND/association/engine/assignment_path/support.py`
- `REMIND/association/resolver/hungarian_resolver.py`
- `REMIND/association/resolver/lock_resolver.py`

### 5. Guardas postasignación

Responsabilidad:

- aplicar `identity_stability`
- desambiguar con `known_set_distance_disambiguation`
- decidir `committed_new_competition`
- reconciliar decisiones temporales

Archivos:

- `REMIND/association/engine/assignment_result_applier.py`
- `REMIND/association/engine/post_assignment/support.py`

### 6. Outcomes

Responsabilidad:

- diagnosticar ambigüedad inicial y final
- traducir a `MATCH / NEW / AMBIGUOUS / PROVISIONAL`
- dejar salida lista para `update`

Policy canónica:

- `REMIND/association/policy/outcome_policy.py`

## Lectura recomendada

Si alguien necesita entender el módulo de arriba abajo, el orden recomendado es:

1. `REMIND/association/ASSOCIATION_DECISION_PATH.md`
2. `REMIND/association/policy/POLICY_TAXONOMY.md`
3. `REMIND/association/flow/frame_association_flow.py`
4. `REMIND/association/engine/assignment.py`
5. `REMIND/association/engine/assignment_result_applier.py`

Los documentos `REMIND/association/policy/SETS_RULES.md` y `REMIND/association/policy/DISTANCE_RULES.md` deben leerse como inventario detallado por bloque, no como mapa principal del flujo.
