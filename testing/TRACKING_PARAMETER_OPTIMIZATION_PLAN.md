# Plan de Sensibilidad de Parametros del Pipeline

## Objetivo

Definir un protocolo de analisis de sensibilidad de parametros para el pipeline de `APP2/Src` orientado a medir que cambios mueven de verdad las metricas sobre una tanda representativa del dataset, con coste computacional controlado y con decisiones reproducibles.

Este documento es de planificacion. No define todavia codigo de automatizacion. La idea es cerrar aqui:

- metricas objetivo y metricas de control;
- tanda de escenas para el analisis principal;
- parametros que entran en el estudio;
- valores `baseline`, `low` y `high` de cada parametro;
- reglas de descarte temprano;
- criterio de cierre del estudio.

## Referencias del repositorio

- Configuracion base: `APP2/Src/config/default_config.yaml`
- Tiers de parametros: `APP2/Src/config/CONFIG_TIERS.md`
- Runner de evaluacion: `APP2/Src/testing/run_tracking_test.py`
- Metricas de tracking: `APP2/Src/testing/tracking_metrics.py`

## Principios de trabajo

1. No optimizar todo a la vez.
2. No asumir independencia fuerte entre bloques de parametros.
3. No optimizar contra una sola metrica sin guardarrailes.
4. No usar una tanda demasiado pequena para decidir configuraciones finales.
5. No gastar presupuesto grande en configuraciones claramente peores que baseline.
6. Mantener siempre una separacion entre:
   - escenas usadas para analisis principal;
   - escenas usadas para validacion final.

## Preguntas que este plan debe cerrar

### 1. Metricas

Hay que decidir:

- metrica principal;
- metricas de control obligatorias;
- si se usa una metrica principal con restricciones o una lectura puramente descriptiva contra baseline.

### 2. Tanda de escenas

Hay que decidir:

- cuantas escenas usar en el analisis principal;
- como seleccionar escenas representativas;
- cuantas escenas reservar como holdout final.

### 3. Parametros del estudio

Hay que decidir:

- que parametros entran en el analisis;
- para cada parametro:
  - tipo (`bool`, `categorico`, `int`, `float`);
  - valor `baseline`;
  - valor `low`;
  - valor `high`;
  - dependencias con otros parametros.

## Propuesta inicial de metricas

### Metrica principal candidata

- `collapsed_identity_metrics.idf1`

Motivo:

- resume bastante bien la calidad de identidad colapsada;
- es una metrica fuerte para ranking global de configuraciones;
- castiga fragmentacion y asignaciones inconsistentes mejor que accuracy simple por frame.

### Metricas de control candidatas

- `summary.pred_track_inflation_factor`
- `summary.reopen_rate_existing`
- `collapsed_metrics.accuracy_existing_vs_new_collapsed`
- `collapsed_metrics.accuracy_parent_collapsed`
- `collapsed_metrics.new_detection_accuracy_collapsed`

Metricas opcionales, utiles si la rama incierta es importante:

- `collapsed_metrics.set_accuracy_ambiguous`
- `uncertainty_metrics.hypothesis_recall_uncertain`

### Decision adoptada

Se usara este esquema:

- tomar `collapsed_identity_metrics.idf1` como referencia principal;
- leer el resto como guardarrailes obligatorios frente a baseline;
- evitar una score compuesta mientras no haga falta.

### Recomendacion actual

Empezar con:

- objetivo principal: maximizar `idf1`;
- restricciones:
  - no empeorar `pred_track_inflation_factor` por encima de un margen definido respecto a baseline;
  - no empeorar `reopen_rate_existing` por encima de un margen definido respecto a baseline;
  - no degradar `accuracy_existing_vs_new_collapsed` por debajo de un umbral minimo.

## Estrategia final

El objetivo ya no es hacer una optimizacion extensa ni una grid search grande.

La estrategia final acordada es:

- usar el baseline actual como configuracion de referencia;
- hacer solo un analisis de sensibilidad local;
- variar un unico parametro cada vez;
- dejar todos los demas parametros fijos en baseline;
- usar solo un valor por debajo y un valor por encima del baseline para cada parametro;
- no evaluar combinaciones entre parametros en esta fase.

Lectura metodologica correcta:

- esto no es una busqueda global de hiperparametros;
- esto es un estudio de sensibilidad local alrededor de una configuracion ya funcional;
- el objetivo es medir que parametros mueven de verdad las metricas y en que direccion.

## Seleccion de escenas

La unidad de evaluacion sigue siendo la escena completa.

Principios:

- no mezclar frames de escenas distintas;
- usar siempre la misma tanda de escenas para comparar sensibilidades;
- mantener separadas escenas de analisis y escenas de validacion final si se quiere reportar generalizacion.

En esta fase, lo importante no es construir un sistema de tuning complejo, sino garantizar comparabilidad entre configuraciones.

## Espacio de sensibilidad: formato

Para cada parametro que se quiera evaluar, conviene dejar esta ficha:

- `nombre`
- `tipo`
- `baseline`
- `low`
- `high`
- `dependencias`
- `comentario`

### Ejemplo de ficha

- `association.matching.match_thr`
- tipo: `float`
- baseline: `0.75`
- low: `0.65`
- high: `0.85`
- dependencias: ninguna
- comentario: umbral principal de matching

## Reglas para definir low y high

### Regla general

Por defecto, los valores `low` y `high` deben quedar cerca del baseline actual del `default_config.yaml`.

Motivo:

- mantiene el analisis en una zona del espacio ya funcional;
- reduce el riesgo de sacar conclusiones a partir de configuraciones absurdas;
- permite interpretar el resultado como sensibilidad local real.

Aplicacion practica:

- para `float`, usar un valor por debajo y otro por encima razonablemente proximos al baseline;
- no hace falta simetria perfecta, pero si una vecindad interpretable;
- para esta fase no se introducen mallas finas ni barridos densos.

## Parametros que entran en el analisis

### 1. Umbrales nucleares de matching

- `association.matching.match_thr`
- `association.matching.clear_margin`
- `update.min_match_score`

Objetivo:

- mover las fronteras principales entre match y no-match;
- ajustar el acoplamiento entre policy de asociacion y policy de update.

Decision final:

- estos tres parametros entran si o si;
- se evaluan con sensibilidad local 1 a 1;
- no se combinan entre si en esta fase.

Baselines actuales:

- `association.matching.match_thr = 0.75`
- `association.matching.clear_margin = 0.08`
- `update.min_match_score = 0.60`

Valores finales acordados:

- `association.matching.match_thr`
  - tipo: `float`
  - baseline: `0.75`
  - low: `0.65`
  - high: `0.85`
  - comentario: umbral principal para aceptar el score final de asignacion

- `association.matching.clear_margin`
  - tipo: `float`
  - baseline: `0.08`
  - low: `0.05`
  - high: `0.12`
  - comentario: margen estructural usado tambien como referencia por varias heuristicas de ambiguedad y fallback visual

- `update.min_match_score`
  - tipo: `float`
  - baseline: `0.60`
  - low: `0.55`
  - high: `0.65`
  - comentario: umbral principal sobre el score de similitud para permitir un match y evitar ir a create

Restricciones suaves recomendadas:

- mantener `update.min_match_score <= association.matching.match_thr`
- mantener `association.matching.clear_margin < association.matching.match_thr`
- evitar configuraciones extremas donde `match_thr` sube mucho pero `min_match_score` queda demasiado bajo, o al reves

Motivo de estas restricciones:

- `match_thr` y `min_match_score` se usan juntos en la aceptacion final del matching hungarian;
- `clear_margin` no solo afecta al margen de claridad, sino que alimenta defaults de ambiguedad y ciertos fallbacks visuales;
- `min_match_score` tambien alimenta defaults de `association.provisional_new.*`, asi que moverlo demasiado cambia mas politica de la que parece.

#### Ablaciones estructurales opcionales

- `association.matching.hungarian.enable_dummies`
- `association.matching.hungarian.locks.enabled`
- `association.matching.neighbor_sets_influence.enabled`
- `association.matching.neighbor_sets_context_veto.enabled`
- `association.ambiguous_tracks.enabled`
- `association.provisional_new.enabled`
- `update.robust_updates.enabled`

Objetivo:

- verificar de forma puntual si algun modulo estructural esta perjudicando claramente;
- contrastar decisiones de diseno solo cuando haya una hipotesis concreta;
- no consumir presupuesto temprano en desactivar piezas que forman parte del pipeline acordado.

Decision de trabajo actual:

- estos `bool` no forman parte del espacio principal de optimizacion temprana;
- por defecto se mantienen en los valores fijados en el `default_config.yaml`;
- solo entran en juego si:
  - el usuario quiere evaluar explicitamente una variante del pipeline;
  - aparece una hipotesis fuerte de que un modulo concreto esta haciendo dano;
  - o se quiere hacer una validacion final tipo ablation para entender contribuciones.

Momento recomendado:

- como analisis posterior, separado del estudio de sensibilidad principal.

Baselines actuales:

- `association.matching.hungarian.enable_dummies = true`
- `association.matching.hungarian.locks.enabled = true`
- `association.matching.neighbor_sets_influence.enabled = true`
- `association.matching.neighbor_sets_context_veto.enabled = true`
- `association.ambiguous_tracks.enabled = true`
- `association.provisional_new.enabled = true`
- `update.robust_updates.enabled = false`

Si se activan estas ablaciones mas adelante:

- probar cambios unitarios respecto a baseline;
- evitar mallas exhaustivas de `2^n`;
- no mezclar demasiadas ablaciones estructurales con tuning fino de umbrales en la misma tanda;
- tratar el resultado como comparativa entre variantes de pipeline, no como hiperparametrizacion normal.

### 2. Mezcla principal de scores

- `association.matching.weights.object`
- `association.matching.weights.background_global`
- `association.matching.weights.parts`

Objetivo:

- medir cuanto influyen los pesos gruesos de evidencia visual sin entrar en heuristicas finas.

Restriccion importante:

- no optimizar los pesos como variables libres e independientes sin restriccion;
- tratarlos como una mezcla normalizada;
- la suma efectiva debe quedar fijada en `1.0`.

Recomendacion practica:

- no tocar `background_partials`;
- trabajar solo con la mezcla activa normalizada;
- parametrizarla con `p_object` y `p_bg_vs_parts`.

Decision final:

- `background_partials` no se optimiza;
- la mezcla evaluada en este estudio es:
  - `object`
  - `background_global`
  - `parts`
- estos tres pesos deben normalizarse para que sumen `1.0`.

Parametrizacion recomendada:

- no optimizar directamente los `3` pesos como variables libres;
- usar `2` variables auxiliares:
  - `p_object`
  - `p_bg_vs_parts`

Construccion de pesos:

- `w_object = p_object`
- `resto = 1 - w_object`
- `w_background_global = resto * p_bg_vs_parts`
- `w_parts = resto * (1 - p_bg_vs_parts)`

Ventajas:

- suma exactamente `1.0`;
- no genera pesos negativos;
- reduce redundancia en la parametrizacion del estudio;
- deja una interpretacion clara:
  - primero se decide cuanto manda `object`;
  - despues se reparte el resto entre `background_global` y `parts`.

Centro inicial recomendado para los pesos

Tomar como centro no los pesos brutos del `yaml`, sino la mezcla activa normalizada.

Valores actuales en `default_config.yaml`:

- `object = 0.60`
- `background_global = 0.25`
- `parts = 0.10`
- `background_partials = 0.05`, pero esta rama queda fuera del estudio y se fija a `0.0`

Por tanto, la mezcla activa normalizada de referencia para este estudio pasa a ser aproximadamente:

- `w_object = 0.632`
- `w_background_global = 0.263`
- `w_parts = 0.105`

Interpretacion en la parametrizacion auxiliar:

- `p_object` debe centrarse alrededor de `0.632`;
- `p_bg_vs_parts` no es un peso global sobre el total;
- `p_bg_vs_parts` solo reparte el resto que queda despues de fijar `object`;
- por tanto, debe centrarse alrededor de:
  - `0.263 / (0.263 + 0.105) ≈ 0.715`

Lectura correcta:

- primero, `object` se lleva aproximadamente el `63.2%` del total;
- despues, del `36.8%` restante:
  - `background_global` se lleva aproximadamente el `71.5%`;
  - `parts` se lleva aproximadamente el `28.5%`.

No debe interpretarse como:

- `background_global = 71.5% del total`
- ni `parts = 28.5% del total`

Valores finales acordados:

- `p_object`
  - tipo: `float`
  - baseline: `0.632`
  - low: `0.55`
  - high: `0.70`
  - comentario: controla cuanto peso total recibe la rama de `object`

- `p_bg_vs_parts`
  - tipo: `float`
  - baseline: `0.715`
  - low: `0.60`
  - high: `0.80`
  - comentario: reparte el peso restante entre `background_global` y `parts`

Interpretacion del rango propuesto:

- `p_object` permite explorar mezclas donde `object` siga siendo dominante, pero sin obligarlo a absorber casi todo el peso;
- `p_bg_vs_parts` permite moverse alrededor del sesgo actual hacia `background_global`, dejando aun espacio suficiente para que `parts` gane importancia si el dataset lo agradece.

Restricciones y notas recomendadas:

- mantener siempre `association.matching.weights.background_partials = 0.0`;
- derivar siempre los pesos finales a partir de `p_object` y `p_bg_vs_parts`, sin optimizar los `3` pesos directamente;
- no evaluar combinaciones entre `p_object` y `p_bg_vs_parts` en esta fase.

Motivo de esta prudencia:

- los pesos del `yaml` son nominales, pero el score final usa pesos efectivos modulados por calidad;
- ademas, con `renormalize_missing = true`, la mezcla se renormaliza sobre los terminos realmente presentes;
- por tanto, un cambio grande en pesos nominales no se traduce siempre en un cambio igual de grande en el comportamiento efectivo, y no conviene abrir de entrada un rango excesivo.

## 3. Tamano de memoria de descriptores

Este bloque si merece entrar en el estudio, pero de forma muy controlada.

Motivo:

- cambia la capacidad de representacion historica de cada identidad;
- puede afectar tanto a robustez de matching como a inflacion, reaperturas y coste;
- es una hipotesis bastante interpretable para el TFM.

Decision metodologica:

- incluir solo memoria de apariencia de objeto y memoria de partes;
- incluir tambien memoria de fondo, pero solo como capacidad acoplada;
- no abrir la memoria de fondo como varios knobs independientes, porque ahi el "tamano" real esta repartido en varios bancos distintos y seria mas dificil interpretar el efecto.

Regla especifica para memorias:

- aqui no hace falta buscar simetria alrededor del baseline;
- interesa forzar tres regimens claros:
  - memoria que se llena pronto;
  - memoria que se llena, pero tarda mas;
  - memoria que rara vez llega a llenarse.

### 3.1 Memoria de apariencia de objeto

La memoria de apariencia tiene dos capacidades separadas:

- `memory.appearance.max_prototypes_per_channel`
- `memory.appearance.max_stable_prototypes_per_channel`

Para este estudio no conviene moverlas por separado.

Decision final:

- tratarlas como un unico knob acoplado;
- cuando se cambie una, cambiar la otra al mismo valor.

Baselines actuales:

- `memory.appearance.max_prototypes_per_channel = 20`
- `memory.appearance.max_stable_prototypes_per_channel = 20`

Valores finales acordados:

- `appearance_memory_capacity`
  - tipo: `int` acoplado
  - baseline: `20 / 20`
  - low: `10 / 10`
  - high: `60 / 60`
  - comentario: capacidad por canal para memoria `work` y `stable` de apariencia

Lectura esperada:

- si bajar memoria apenas cambia resultados, el baseline puede estar sobredimensionado;
- si subir memoria mejora identidades dificiles o reduce reaperturas, hay senal de que la memoria actual se queda corta;
- si subir memoria empeora o apenas ayuda pero encarece bastante, conviene mantener una memoria mas compacta.

### 3.2 Memoria de partes

La memoria de partes expone un tamaño mas limpio y facil de interpretar:

- `memory.parts.max_prototypes_per_channel`

Baseline actual:

- `memory.parts.max_prototypes_per_channel = 80`

Valores finales acordados:

- `parts_memory_capacity`
  - tipo: `int`
  - baseline: `80`
  - low: `40`
  - high: `240`
  - comentario: capacidad por canal de la memoria de descriptores de partes

Lectura esperada:

- si la informacion de partes realmente ayuda en escenas complejas, una memoria algo mayor deberia notarse sobre todo en `idf1` y en reaperturas;
- si apenas cambia nada, probablemente no merece abrir mas este bloque en esta fase.

### 3.3 Memoria de fondo

La memoria de fondo esta repartida en varios bancos:

- global `inner` y `outer`
- partials `inner` y `outer`
- versiones `work` y `stable` de esos bancos

Por eso no conviene estudiar cada capacidad por separado en esta fase.

Decision final:

- tratar toda la memoria de fondo como un unico knob acoplado;
- mantener las proporciones actuales entre bancos;
- mover a la vez capacidades `work` y `stable` para cada banco.

Baselines actuales:

- `memory.background.max_inner = 20`
- `memory.background.max_outer = 30`
- `memory.background.max_inner_partials = 60`
- `memory.background.max_outer_partials = 80`
- `memory.background.max_inner_global_stable = 20`
- `memory.background.max_outer_global_stable = 30`
- `memory.background.max_inner_partials_stable = 60`
- `memory.background.max_outer_partials_stable = 80`

Valores finales acordados:

- `background_memory_capacity`
  - tipo: `int` acoplado
  - baseline: `20/30/60/80` con los bancos `stable` espejados
  - low: `10/15/30/40` con los bancos `stable` espejados
  - high: `60/90/180/240` con los bancos `stable` espejados
  - comentario: escala conjunta para toda la capacidad de memoria de fondo, manteniendo las proporciones entre bancos

Lectura esperada:

- si el fondo esta ayudando de verdad a estabilizar identidades, una memoria algo mayor podria reducir errores de matching en escenas con contexto repetitivo;
- si no hay efecto claro, no merece abrir este bloque en subparametros mas finos;
- si el coste sube demasiado para mejoras pequenas, la capacidad actual probablemente ya sea suficiente.

## Parametros que quedan fuera

En esta fase se dejan fuera:

- subparametros finos de `neighbor_sets`;
- politicas temporales detalladas de `ambiguous_tracks` y `provisional_new`;
- thresholds de memorias de update (`appearance_memory`, `background_memory`, `parts_memory`);
- capacidades detalladas e independientes de memoria de fondo (`memory.background.*`);
- parametros internos de `known_set_distance_disambiguation`;
- `bool` estructurales del pipeline.

Estos bloques solo deberian entrar si el analisis de sensibilidad deja una hipotesis de fallo muy concreta.

## Diseno experimental final

Se evaluan exactamente estas configuraciones:

1. baseline
2. `match_thr` low
3. `match_thr` high
4. `clear_margin` low
5. `clear_margin` high
6. `min_match_score` low
7. `min_match_score` high
8. `p_object` low
9. `p_object` high
10. `p_bg_vs_parts` low
11. `p_bg_vs_parts` high
12. `appearance_memory_capacity` low
13. `appearance_memory_capacity` high
14. `parts_memory_capacity` low
15. `parts_memory_capacity` high
16. `background_memory_capacity` low
17. `background_memory_capacity` high

Total:

- `17` configuraciones

Regla clave:

- cada configuracion modifica solo un parametro respecto a baseline.

No se evaluan:

- combinaciones de dos o mas parametros;
- mallas `3 x 3`;
- grid search global;
- fases de recombinacion.

## Analisis esperado

Para cada configuracion se comparara contra baseline al menos en:

- `collapsed_identity_metrics.idf1`
- `summary.pred_track_inflation_factor`
- `summary.reopen_rate_existing`
- `collapsed_metrics.accuracy_existing_vs_new_collapsed`

Metricas de apoyo recomendadas:

- `collapsed_metrics.accuracy_parent_collapsed`
- `collapsed_metrics.new_detection_accuracy_collapsed`

La lectura buscada no es "encontrar el mejor punto global", sino:

- si el parametro tiene sensibilidad apreciable;
- si la mejora o empeora al subirlo;
- si la mejora o empeora al bajarlo;
- si el baseline ya estaba en una zona razonablemente estable.

Si entran parametros de tamano de memoria, tambien conviene vigilar:

- tiempo medio por escena;
- si aparece algun crecimiento de coste poco justificable para mejoras pequenas.

La definicion detallada de telemetria de memoria e informacion a guardar no pertenece a este plan de sensibilidad, sino a la documentacion de metricas y salidas:

- ver `APP2/Src/testing/METRICS_AND_OUTPUTS.md`

En este plan solo interesa recordar que, para las variantes de tamano de memoria, conviene analizar despues:

- ocupacion y saturacion de memoria;
- eventos `merge` y `evict`;
- bytes persistentes de memorias de descriptores;
- coste temporal y de RAM/GPU asociado.

## Reglas de descarte temprano

Para no gastar presupuesto completo en configuraciones malas:

- si una configuracion va claramente peor que baseline en las primeras escenas, se poda;
- si dispara mucho `pred_track_inflation_factor`, se poda;
- si dispara mucho `reopen_rate_existing`, se poda;
- si empeora fuertemente `accuracy_existing_vs_new_collapsed`, se poda.

Pero:

- no sacar conclusiones fuertes a partir de una sola escena rara;
- no interpretar un unico cambio pequeno como prueba de mejor punto global.

### Pendiente de definir

Hay que fijar margenes concretos de descarte, por ejemplo:

- caida maxima tolerable de `idf1`;
- subida maxima tolerable de inflation;
- subida maxima tolerable de reopen.

## Criterio de cierre

El estudio se considera cerrado cuando:

- se hayan evaluado las `17` configuraciones previstas;
- quede claro que parametros muestran sensibilidad real;
- quede claro que parametros apenas mueven las metricas en la vecindad del baseline.

## Artefactos a guardar

- configuracion evaluada;
- escenas usadas;
- metricas agregadas;
- metricas por escena;
- telemetria de memoria por frame y por escena;
- ranking final de configuraciones;
- motivo de poda si aplica;
- version del baseline usado.

## Decisiones cerradas

- no usar solo `1-5` escenas para tomar la decision final;
- no hacer grid search grande;
- no hacer combinaciones entre parametros en esta fase;
- usar solo sensibilidad local 1 a 1;
- empezar por los parametros mas nucleares y la mezcla principal de score;
- tratar los `bool` estructurales como ablaciones, no como tuning normal.

## Pendientes inmediatos

1. Cerrar la tanda exacta de escenas sobre la que se ejecutaran las `17` configuraciones.
2. Ejecutar baseline y variantes univariantes.
3. Volcar resultados en una tabla comparativa simple respecto a baseline.
4. Decidir si hace falta o no una segunda tanda pequena sobre algun bloque concreto.

## Version del plan

- Estado: `alineado con analisis de sensibilidad local`
- Siguiente paso recomendado: ejecutar las `17` configuraciones definidas
