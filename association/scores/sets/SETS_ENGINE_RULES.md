# Reglas internas del motor `sets`

## Alcance

Este documento describe las heurísticas internas del motor de hipótesis de `sets`. No cubre las reglas de policy de asociación posteriores, que están documentadas en `APP2/Src/association/policy/SETS_RULES.md`.

El motor se implementa en:

- `APP2/Src/association/scores/sets/neighbor_sets_score.py`
- `APP2/Src/association/scores/sets/sets_options.py`
- `APP2/Src/association/scores/sets/sets_search.py`
- `APP2/Src/association/scores/sets/sets_scoring.py`
- `APP2/Src/association/scores/sets/sets_graph_utils.py`
- `APP2/Src/association/scores/sets/sets_summary.py`

## Fases del algoritmo

El motor opera en cinco fases:

1. Selección de detecciones por clase
2. Construcción de pools y opciones por clase
3. Búsqueda beam sobre clases
4. Scoring de estados e hipótesis
5. Resumen final a `core/debug`

## Reglas de entrada y reducción inicial

### Selección top de detecciones por clase

Archivo: `APP2/Src/association/scores/sets/sets_options.py`

Regla:

- por clase solo se conservan las `per_class_det_k` detecciones con mayor `confidence`;
- el desempate se resuelve por `detection_id`.

Finalidad:

- reducir el espacio de búsqueda antes de construir combinatorias.

### Límite de clases participantes

Archivo: `APP2/Src/association/scores/sets/neighbor_sets_score.py`

Regla:

- las clases se ordenan por tamaño de pool creciente;
- si `max_set_size > 0`, solo participan las primeras `max_set_size`.

Finalidad:

- evitar explosión combinatoria en frames con muchas clases.

## Reglas de construcción de pools

### Pool base por clase

Archivo: `APP2/Src/association/scores/sets/sets_options.py`

Regla:

- el pool inicial de una clase son los objetos conocidos de esa clase en memoria.

### Priorización por anchors

Regla:

- si existen anchors, los candidatos del pool se ordenan por el máximo `p_conditional` observado desde cualquier anchor hacia ese objeto;
- solo se conservan candidatos con `p_conditional >= min_edge_p`.

Finalidad:

- priorizar objetos compatibles con el contexto anclado del frame.

### Relleno por madurez

Regla:

- si la priorización por anchors no llena `per_class_pool_k`, el resto del pool se completa por `object_maturity_score`.

Finalidad:

- no dejar una clase sin candidatos por falta de enlaces fuertes recientes.

### Inclusión preferente de anchors de la misma clase

Regla:

- si algún anchor pertenece a la clase, se inserta al principio del pool final antes de truncar.

Finalidad:

- estabilizar la continuidad de IDs ya anclados de forma fiable.

## Reglas de construcción de opciones por clase

### Opción vacía bajo cobertura parcial

Archivo: `APP2/Src/association/scores/sets/sets_options.py`

Regla:

- si `allow_partial_coverage = true`, cada clase puede emitir una opción con `k = 0`.

Finalidad:

- permitir hipótesis que no explican todas las clases presentes.

### Ranking local por soporte a kernel

Regla:

- los objetos candidatos se rankean por soporte al `kernel`;
- si no hay `kernel`, se rankean por madurez.

Finalidad:

- producir variantes de objetos plausibles con coste bajo.

### Definición de `target_k`

Regla:

- `target_k` se deriva del número de objetos con soporte por encima de `class_plausible_rel * top_support`.

Finalidad:

- estimar cuántos objetos de la clase son plausibles antes de generar variantes.

### Variantes de objetos capadas

Regla:

- no se materializan todas las combinaciones `C(L, k)`;
- se usa una familia pequeña de variantes:
- top-1: los `k` mejores por soporte;
- variantes adicionales: `top-(k-1) + sustitución del último`.

Control:

- `options_obj_Lmax`
- `options_obj_variants`

Finalidad:

- mantener diversidad útil sin explosión combinatoria.

### Capado de combinaciones de detecciones

Regla:

- por cada `k`, solo se generan hasta `min(options_det_combo_max, options_det_variants)` combinaciones de detecciones.

Finalidad:

- limitar el coste en clases con muchas detecciones.

### Exclusividad de clase

Regla:

- las opciones de una misma clase y mismo `k` reciben una medida de exclusividad basada en la separación entre `opt_rank` top-1 y top-2.

Finalidad:

- premiar clases donde la selección interna es especialmente discriminativa.

## Reglas de búsqueda beam

### Estado inicial

Archivo: `APP2/Src/association/scores/sets/sets_search.py`

Regla:

- el beam arranca con un único estado vacío.

### Expansión por clase

Regla:

- la búsqueda avanza clase a clase, no detección a detección.

Finalidad:

- explotar la estructura del problema por clase y reducir branching.

### Restricción 1-a-1 de objetos

Regla:

- un estado no puede reutilizar un `object_id` ya usado en otra clase previa del mismo beam state.

Finalidad:

- mantener consistencia global entre hipótesis.

### Kernel dinámico

Regla:

- antes de expandir una clase, se construye un `kernel` con anchors y objetos ya seleccionados en el estado parcial.

Finalidad:

- adaptar el ranking local de candidatos al contexto parcial ya fijado.

### Selección diversa del beam

Regla:

- tras puntuar los estados expandidos, se conserva primero el mejor estado por cada `k_c` de la clase actual;
- el resto de huecos del beam se rellenan por score global.

Finalidad:

- evitar que todo el beam colapse a un único tamaño por clase.

## Reglas de scoring interno

### Cobertura efectiva

Archivo: `APP2/Src/association/scores/sets/sets_scoring.py`

Regla:

- la cobertura observada se transforma con:
- potencia `coverage_gamma`;
- boost por tamaño del frame;
- damping por número de detecciones explicadas.

Finalidad:

- distinguir entre explicar poco en frames pequeños y explicar poco en frames grandes.

### Utilidad de tamaño

Regla:

- los sets con `k <= size_k_min` no reciben utilidad de tamaño;
- a partir de ahí la utilidad crece con una exponencial saturante.

### Densidad y conectividad

Archivo: `APP2/Src/association/scores/sets/sets_graph_utils.py`

Reglas:

- la coherencia del set se mide con la media del árbol de expansión máximo;
- opcionalmente se anulan enlaces por debajo de `min_edge_p`;
- se calcula además:
- `edge_cov`
- `node_cov`
- `min_deg`

Efecto:

- el score final puede quedar anulado o atenuado por conectividad insuficiente.

### Información de clase

Regla:

- la ambigüedad combinatoria de clase se mide mediante `log_n_choose_k`;
- esta cantidad se transforma a `class_info` con `class_ambig_beta`.

Finalidad:

- penalizar selecciones poco informativas en clases con muchas combinaciones equivalentes.

### Reponderación por tamaño del set

Regla:

- los pesos de `class_info` y `exclusivity` se escalan con `k`;
- en sets pequeños estas señales pesan más;
- en sets grandes pesan menos.

Finalidad:

- evitar sobrevalorar rareza/exclusividad cuando el set ya es grande y discriminativo por sí mismo.

### Madurez

Reglas:

- `object_maturity_score` depende del número de episodios del grafo de vecinos;
- `maturity_coherence` usa una media generalizada blanda sobre las madureces del set;
- si `maturity_enabled`, el score final se multiplica por `maturity_rel`.

Finalidad:

- penalizar sets apoyados en memoria todavía inmadura.

### Cobertura contextual

Regla:

- si `context_enabled` y `k >= 2`, se mide cuánto de los vecinos esperados top-k de cada objeto aparece también en el set.

Finalidad:

- estimar coherencia de contexto más allá de los enlaces internos mínimos.

### Combinación final de score

Regla:

- `score_sets` combina, con normalización por pesos activos:
- cobertura efectiva
- utilidad de tamaño
- densidad
- información de clase
- soporte de clase
- estabilidad de clase
- exclusividad

La exclusividad solo entra si:

- está habilitada;
- existe evidencia válida de exclusividad;
- `k >= excl_k_min`;
- la madurez media supera `excl_maturity_min`.

## Reglas de aceptación de hipótesis

### Filtro por tamaño mínimo en frames no triviales

Archivo: `APP2/Src/association/scores/sets/neighbor_sets_score.py`

Regla:

- si `total_dets > 1`, no se aceptan hipótesis con `k < 2`.

### Filtro por score mínimo

Regla:

- una hipótesis solo se conserva si `score_sets >= min_set_score`.

### Eliminación de duplicados por contenido

Regla:

- dos estados con el mismo conjunto de objetos y mismas detecciones explicadas se consideran equivalentes.

Finalidad:

- evitar multiplicidad artificial en la salida final del beam.

### Top-k final de hipótesis

Regla:

- las hipótesis se ordenan por `score_sets` y se conservan las primeras `topk_sets`.

## Reglas de resumen final

Archivo: `APP2/Src/association/scores/sets/sets_summary.py`

Reglas:

- `shortlist` se construye a partir de todas las hipótesis cuyo score cae dentro de `best_score * (1 - shortlist_rel)`;
- `prior_by_oid` toma el mejor score observado por objeto entre las hipótesis top;
- `class_prior_by_cid` conserva el mejor prior por clase dentro de la shortlist;
- `selective_classes` marca clases con prior dominante suficiente y gap relativo suficiente.

## Edge cases formalizados

- opción vacía de clase solo cuando se permite cobertura parcial;
- relleno por madurez cuando los anchors no generan candidatos suficientes;
- reducción fuerte del espacio de objetos por variantes top-k en lugar de combinatoria completa;
- diversidad del beam por `k_c` para evitar colapso estructural;
- exclusividad desactivada en sets pequeños o inmaduros;
- hipótesis de un solo objeto descartadas en frames con más de una detección;
- normalización del score solo con términos activos para no castigar ausencia de señales opcionales.
