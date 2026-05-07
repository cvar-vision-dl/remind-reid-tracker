# Runtime e instrumentación de trazabilidad de `association`

## Objetivo

Definir cómo debería funcionar en runtime la futura capa de trazabilidad visual
de `association`, sin cambiar comportamiento y manteniendo el coste bajo cuando
esté desactivada.

Este documento complementa:

- `APP2/Src/association_trace/ASSOCIATION_TRACE_VISUALIZATION.md`
- `APP2/Src/association_trace/ASSOCIATION_TRACE_NODE_MAP.md`
- `APP2/Src/association_trace/ASSOCIATION_TRACE_CONTRACT.md`

## Requisitos

La instrumentación debe cumplir estas reglas:

- ser opcional;
- no alterar scores, ramas ni decisiones;
- poder apagarse en testing final;
- poder activarse en `main` normal;
- escribir artefactos estructurados, no logs de texto;
- tener ids estables de nodo;
- desacoplar captura y renderizado.

## Arquitectura propuesta

### 1. `pipeline schema`

Artefacto fijo que describe la arquitectura visual del pipeline.

- se mantiene a mano;
- cambia solo cuando cambia el pipeline real;
- no depende de una ejecución concreta.

### 2. `execution trace`

Artefacto dinámico que representa una ejecución concreta de:

- `frame_id`
- `class_id`

Debe contener solo datos observados en ejecución, nunca cálculos nuevos.

### 3. `trace collector`

La captura en runtime debería pasar por un collector con interfaz estable.

Modos previstos:

- `NoOpAssociationTraceCollector`
- `JsonAssociationTraceCollector`

El pipeline hablaría siempre con la misma interfaz. Si la traza está
desactivada, se usaría el collector no-op.

## Interfaz conceptual del collector

No es una implementación todavía, solo el contrato que conviene perseguir.

### Métodos de ciclo de vida

- `start_frame(frame_id, timestamp, det_ids)`
- `start_class(frame_id, class_id, class_name, det_ids, snapshot_object_ids)`
- `finish_class(frame_id, class_id)`
- `finish_frame(frame_id)`
- `flush()`

### Métodos por nodo

- `enter_node(node_id, scope_key, participants=None)`
- `skip_node(node_id, scope_key, reason, participants=None)`
- `add_check(node_id, scope_key, check)`
- `add_value(node_id, scope_key, key, value)`
- `set_values(node_id, scope_key, values)`
- `set_decision(node_id, scope_key, decision)`
- `add_candidate_row(node_id, scope_key, row)`
- `add_detection_row(node_id, scope_key, row)`
- `add_global_row(node_id, scope_key, row)`
- `leave_node(node_id, scope_key)`

## Collector no-op

Cuando la instrumentación esté desactivada, el pipeline debería hablar con un
collector sin coste relevante.

Objetivos del no-op:

- evitar llenar el código de `if trace_enabled`;
- centralizar el apagado;
- mantener el pipeline limpio.

Comportamiento:

- todos los métodos aceptan llamadas;
- no guardan nada;
- devuelven inmediatamente.

## Collector JSON

Cuando la instrumentación esté activada, el collector debe construir artefactos
JSON estructurados y estables.

Responsabilidades:

- agrupar eventos por `frame_id + class_id`;
- serializar `node_runs`;
- escribir archivos por ejecución;
- no recalcular información que el pipeline no haya emitido.

## Propuesta de configuración

La activación debería vivir en `debug`, no en la lógica base del tracker.

Propuesta de árbol de config:

```yaml
debug:
  association_trace:
    enabled: true
    mode: "full"
    write_json: true
    output_group: "association_trace"
    write_schema_once: true
    write_per_frame_class_trace: true
```

### Significado de flags

- `enabled`
  activa o desactiva toda la instrumentación.

- `mode`
  nivel de profundidad.

- `write_json`
  si se escriben artefactos JSON al disco.

- `output_group`
  carpeta base dentro de `outputs`.

- `write_schema_once`
  si se escribe el `pipeline_schema` una vez por ejecución.

- `write_per_frame_class_trace`
  si se escribe un JSON independiente por `frame_id + class_id`.

## Modos propuestos

### `off`

- sin collector real;
- usa collector no-op.

### `summary`

- nodos visitados;
- decisiones finales;
- checks principales;
- sin detalle completo por candidato.

### `full`

- nodos visitados;
- checks completos;
- valores importantes;
- detalle por candidato cuando afecte al camino;
- detalle global de resolución y postasignación.

Recomendación:

- `main`: `full`
- testing final: `off`

## Artefactos propuestos

### Carpeta base por ejecución

La salida debería guardarse en una carpeta fechada, igual que ya se hace con
otros artefactos del proyecto.

Propuesta:

- `APP2/outputs/association_trace/run_YYYYMMDD_HHMMSS/`

Contenido:

- `pipeline_schema.json`
- `manifest.json`
- `frames/frame_000042_class_001.json`
- `frames/frame_000042_class_002.json`

## `pipeline_schema.json`

Debe contener:

- `schema_version`
- `module`
- `orientation`
- `unit_of_analysis`
- `phases`
- `nodes`
- `edges`

Debe ser el mismo para todas las trazas de una misma ejecución, salvo que el
pipeline cambie.

## `manifest.json`

Debe permitir navegar sin escanear todos los archivos.

Campos propuestos:

- `run_id`
- `created_at`
- `module`
- `schema_version`
- `trace_version`
- `frame_ids`
- `class_entries`

Ejemplo conceptual:

```json
{
  "run_id": "run_20260319_203000",
  "module": "association",
  "schema_version": "1.0",
  "trace_version": "1.0",
  "frame_ids": [42, 43],
  "class_entries": [
    {
      "frame_id": 42,
      "class_id": 1,
      "class_name": "person",
      "path": "frames/frame_000042_class_001.json"
    }
  ]
}
```

## `execution_trace` por `frame_id + class_id`

El contrato estable de estos JSON está fijado en:

- `APP2/Src/association_trace/ASSOCIATION_TRACE_CONTRACT.md`

Campos base propuestos:

- `trace_version`
- `schema_version`
- `module`
- `run_id`
- `frame_id`
- `class_id`
- `class_name`
- `timestamp`
- `det_ids`
- `snapshot_object_ids`
- `node_runs`

### `node_runs`

Cada `node_run` debería incluir:

- `node_id`
- `entered`
- `skipped_reason`
- `scope_key`
- `participants`
- `checks`
- `values`
- `decision`
- `candidate_rows`
- `detection_rows`
- `global_rows`

No todos los campos tendrán contenido en todos los nodos, pero la estructura
debería ser estable.

## Serialización de valores

Para que el JSON sea estable y fácil de renderizar:

- ids siempre como enteros serializados como números;
- scores y thresholds como `float`;
- sets convertidos a listas ordenadas;
- diccionarios por id con claves serializables;
- no guardar objetos Python crudos;
- no guardar tensores ni arrays completos si no son esenciales.

Regla práctica:

- serializar lo que explica una decisión;
- no serializar estructuras pesadas que no añaden valor visual.

## Dónde se engancha hoy el collector

La instrumentacion actual ya esta conectada en los limites naturales del flujo.

En `flow`:

- `FrameAssociationFlow.select_reliable_visual_anchors()`
- `FrameAssociationFlow.activate_context_layers()`
- `FrameAssociationFlow.finalize_outcomes()`

En `observability`:

- `DataAssociationObservabilityRuntime.start_frame()`
- `DataAssociationObservabilityRuntime.trace_after_similarity_reports()`
- `DataAssociationObservabilityRuntime.trace_after_reliable_anchor_selection()`
- `DataAssociationObservabilityRuntime.trace_after_context_activation()`
- `DataAssociationObservabilityRuntime.trace_after_similarity_diagnosis()`
- `DataAssociationObservabilityRuntime.trace_after_final_ambiguity()`
- `DataAssociationObservabilityRuntime.trace_finalize_outcomes()`
- `DataAssociationObservabilityRuntime.finish_frame()`

En `trace_runtime` (emision efectiva de `node_runs`):

- `DataAssociationTraceRuntime.trace_class_partition()`
- `DataAssociationTraceRuntime.trace_reliable_visual_anchors()`
- `DataAssociationTraceRuntime.trace_visual_build_candidates()`
- `DataAssociationTraceRuntime.trace_visual_report_diagnosis()`
- `DataAssociationTraceRuntime.trace_neighbor_sets_hypotheses()`
- `DataAssociationTraceRuntime.trace_sets_activation()`
- `DataAssociationTraceRuntime.trace_final_ambiguity()`
- `DataAssociationTraceRuntime.trace_final_outcomes()`

En `assignment`:

- `HungarianAssigner.trace_valid_detections()`
- `HungarianAssigner.trace_allow_for_report()`
- `HungarianAssigner.trace_context_veto()`
- `HungarianAssigner.trace_final_score_tables()`
- `HungarianAssigner.trace_locks()`
- `HungarianAssigner.trace_hungarian_result()`
- `HungarianAssigner.trace_skip_node_for_class()`

En `post_assignment`:

- `AssignmentResultApplier.trace_identity_stability()`
- `AssignmentResultApplier.trace_assignment_ambiguity()`
- `AssignmentResultApplier.trace_known_set_distance_disambiguation()`
- `AssignmentResultApplier.trace_create_competition()`
- `AssignmentResultApplier.trace_provisional_reconciliation()`
- `AssignmentResultApplier.trace_skip_node_for_class()`

## Criterio de intrusión mínima

La instrumentación actual y futura debería seguir estas reglas:

- añadir infraestructura, no cambiar decisiones;
- centralizar el encendido/apagado de traza en una unica capa de observabilidad;
- reutilizar wrappers y fases ya existentes;
- capturar donde ya hay límites naturales de etapa;
- evitar meter lógica visual en `policy`;
- evitar duplicar cálculos solo para la UI.

## Estrategia de implementación seguida

Orden ya seguido en esta base:

1. fijar el `pipeline_schema.json` estático;
2. implementar el collector no-op;
3. implementar el collector JSON;
4. instrumentar el camino principal de `association`;
5. estabilizar el contrato `v1`;
6. construir después el visor offline.

## Siguiente paso

Con este documento, el siguiente paso natural ya sería diseñar la interfaz
concreta de clases y helpers:

- `AssociationTraceCollector`
- `NoOpAssociationTraceCollector`
- `JsonAssociationTraceCollector`
- helpers de serialización
- factory según config
