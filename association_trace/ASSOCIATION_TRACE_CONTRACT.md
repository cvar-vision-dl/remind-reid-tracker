# Contrato estable de `association_trace` (`v1`)

## Objetivo

Fijar el formato JSON que consume el visor offline de `association`.

Desde este punto, la traza debe tratarse como una interfaz estable entre:

- el pipeline de `association`, que emite artefactos;
- el visor, que consume esos artefactos.

No significa que el formato no pueda evolucionar. Significa que, mientras no
se cambie de versión, el visor debe poder confiar en este contrato.

## Alcance

Este contrato cubre:

- `pipeline_schema.json`
- `manifest.json`
- `frames/frame_<frame_id>_class_<class_id>.json`

No cubre:

- detalles internos del pipeline no serializados;
- logs de texto;
- artefactos de debug ajenos a `association_trace`.

## Versiones

- `schema_version: "1.0"`
- `trace_version: "1.0"`

Mientras la versión siga en `1.0`:

- no se deben eliminar campos obligatorios;
- no se deben renombrar campos ya publicados;
- no se debe mover información entre bloques sin razón fuerte;
- sí se pueden añadir campos nuevos si no rompen compatibilidad.

Si hubiera un cambio incompatible, deberá subirse la versión.

## Artefactos y estabilidad

### `pipeline_schema.json`

Representa la arquitectura fija del pipeline real.

Debe contener:

- `schema_version`
- `module`
- `orientation`
- `unit_of_analysis`
- `phases`
- `nodes`
- `edges`

Reglas:

- los `node_id` son ids estables;
- el schema cambia solo cuando cambia el pipeline real;
- el visor debe usar el schema como fuente de verdad para layout y orden.

### `manifest.json`

Es el índice navegable de una ejecución.

Debe contener:

- `run_id`
- `created_at`
- `module`
- `schema_version`
- `trace_version`
- `frame_ids`
- `class_entries`

Reglas:

- `frame_ids` debe ir ordenado;
- `class_entries` debe ir ordenado por `frame_id`, `class_id`, `path`;
- el visor puede usarlo para descubrir traces sin escanear toda la carpeta.

### `execution_trace` por `frame_id + class_id`

Cada archivo `frame_<...>_class_<...>.json` debe contener:

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

Reglas:

- `det_ids` y `snapshot_object_ids` deben ir ordenados;
- `node_runs` debe escribirse ordenado según el `pipeline_schema`, no según el
  orden accidental de instrumentación;
- el visor no debe inferir fases o nodos que no estén en el schema.

## Contrato de `node_run`

Todo `node_run` debe tener estos campos:

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

### Semántica

- `node_id`
  Id estable del nodo. Debe existir en `pipeline_schema.json`.

- `entered`
  `true` si el nodo se ejecutó realmente.
  `false` si la rama no llegó a ejecutarse.

- `skipped_reason`
  Cadena vacía si `entered=true`.
  Razón estable si `entered=false`.

- `scope_key`
  Al menos:
  - `frame_id`
  - `class_id`

- `participants`
  Debe incluir:
  - `det_ids`
  - `object_ids`

  Regla:
  incluso en nodos `skipped`, se deben rellenar cuando exista contexto
  razonable de clase.

- `checks`
  Lista de condiciones visibles que cambian camino.

- `values`
  Estado o resultados del nodo.

- `decision`
  Resultado semántico del nodo.

- `candidate_rows`
  Filas por candidato cuando el nodo opera a nivel `(det_id, object_id)`.

- `detection_rows`
  Filas por detección cuando el nodo opera por `det_id`.

- `global_rows`
  Filas globales o por componente cuando el nodo opera a nivel conjunto.

## Contrato de `check`

Todo `check` visible debe poder representarse con:

- `id`
- `label`
- `lhs`
- `op`
- `rhs`
- `passed`
- `reason`
- `effect`

Reglas:

- `id` debe ser estable dentro del nodo;
- `label` es legible por humanos;
- `lhs`, `op`, `rhs` deben reflejar lo evaluado por el pipeline;
- `passed` resume el resultado lógico;
- `reason` explica el resultado;
- `effect` explica el efecto sobre el camino.

## Campos obligatorios y opcionales

Campos obligatorios:

- todos los campos base del trace
- todos los campos base de `node_run`
- todos los campos base de `check`

Campos opcionales:

- cualquier valor adicional dentro de `values`
- cualquier detalle adicional dentro de filas
- campos nuevos añadidos sin romper estructura existente

## Política de evolución

Cambios permitidos dentro de `1.x`:

- añadir nuevos nodos al schema cuando el pipeline crezca;
- añadir nuevos `checks`;
- añadir nuevos campos en `values` o filas;
- añadir nuevas razones de `skipped` si son estables y semánticas.

Cambios que requieren nueva versión:

- renombrar `node_id`;
- eliminar campos obligatorios;
- mover un dato obligatorio a otro bloque;
- cambiar el significado de `entered`, `skipped_reason`, `checks`, `values` o
  `decision`.

## Recomendación para el visor

El visor debería asumir:

- el orden visual viene del `pipeline_schema`;
- `node_runs` ya vendrán serializados en ese orden, pero el schema sigue siendo
  la fuente de verdad;
- los nodos `skipped` forman parte del camino y deben poder renderizarse;
- la ausencia de un nodo solo debería interpretarse como error o versión
  incompatible, no como “rama no recorrida”.
