# Reglas de `sets`

> Estado de referencia: este documento describe el comportamiento real actual
> del bloque `sets`. Para el mapa global del módulo conviene leer además
> `REMIND/association/ASSOCIATION_DECISION_PATH.md` y
> `REMIND/association/policy/POLICY_TAXONOMY.md`.

## Alcance

`sets` introduce contexto relacional de co-ocurrencia en la asociación. No
reemplaza la similitud visual. Su papel real hoy es:

- proponer hipótesis globales de objetos compatibles con el frame;
- resumir esas hipótesis como shortlist, prior y soporte por objeto;
- decidir si ese contexto global es lo bastante sólido para entrar en juego;
- aportar bonus, rescate o veto sobre candidatos concretos;
- conservar una noción de "objeto conocido todavía plausible" para ambigüedad.

## Flujo real hoy

### 1. Generación de hipótesis globales

Archivos:

- `REMIND/association/scores/sets/neighbor_sets_score.py`
- `REMIND/association/scores/sets/sets_search.py`
- `REMIND/association/scores/sets/sets_summary.py`

Responsabilidad:

- explorar combinaciones plausibles de objetos conocidos para explicar el
  frame;
- puntuar esas hipótesis;
- construir `core/debug` con shortlist, priors y metadatos globales.

Punto importante:

- la búsqueda no es exhaustiva;
- primero queda acotada por `beam_width`;
- después la salida retenida se recorta a `topk_sets`.

Consecuencia:

- `n_hypotheses` significa "hipótesis retenidas por el pipeline", no "todas
  las hipótesis plausibles del frame".

### 2. Construcción de contexto usable

Archivos:

- `REMIND/association/context/neighbor_sets_influence.py`
- `REMIND/association/context/sets_context_builder.py`

Responsabilidad:

- traducir la salida global de `neighbor_sets` a contexto utilizable por clase;
- medir calidad global del contexto;
- construir un `class_pack` para cada clase con shortlist, soporte y selectividad.

Punto importante:

- esta fase no se limita a los objetos que salieron arriba en las hipótesis
  retenidas;
- vuelve a recorrer todos los objetos conocidos de cada clase;
- para eso construye un `kernel` con anchors y objetos de las primeras
  hipótesis contextuales (`context_k`);
- después mide afinidad de cada objeto con ese kernel.

Consecuencia:

- un objeto puede recibir soporte contextual aunque no estuviera bien
  representado en el top principal de hipótesis;
- eso mitiga el sesgo del top-k, pero no recupera de forma exhaustiva las
  hipótesis descartadas.

### 3. Activación y scoring contextual

Archivos:

- `REMIND/association/policy/sets_rule_policy.py`

Responsabilidad:

- decidir si una detección concreta puede usar contexto (`allow_for_report`);
- calcular bonus neto por candidato (`bonus_for_candidate`);
- exponer explicación detallada por candidato (`explain_candidate`).

La contribución de `sets` combina:

- soporte local al kernel;
- soporte global dentro de la clase;
- contradicción contextual;
- calidad global del contexto.

### 4. Integración en la fila candidata

Archivos:

- `REMIND/association/policy/candidate_score_policy.py`
- `REMIND/association/policy/known_plausible_keep_policy.py`

Responsabilidad:

- decidir si el candidato sobrevive como plausible conocido;
- aplicar rescate por `SETS_RESCUE` cuando la similitud visual queda un poco
  por debajo del umbral pero el bonus contextual lo compensa;
- aplicar veto contextual fuerte;
- construir `score_sim`, `score_assign` y `score_final`.

## Activación global del contexto

El contexto de `sets` solo entra de verdad si:

- existe salida válida de `neighbor_sets`;
- hay al menos una base mínima de hipótesis;
- la mejor hipótesis tiene tamaño suficiente;
- la cobertura efectiva supera el mínimo;
- el mejor score supera el mínimo;
- la calidad global agregada supera el mínimo.

La calidad global se resume combinando:

- fuerza del mejor contexto;
- cobertura;
- madurez;
- densidad;
- tamaño del mejor grupo;
- capacidad real de poda por clase.

El resultado práctico del bloque es uno de estos estados:

- `active`: el contexto puede influir con normalidad;
- `degraded`: existe contexto, pero se usa con más cautela;
- `inactive`: el frame no deja un contexto relacional utilizable.

## Soporte por objeto dentro de la clase

El `class_pack` que construye `sets_context_builder.py` deja varias lecturas
por objeto:

- `prior_by_oid`: qué apoyo recibe el objeto desde las hipótesis retenidas;
- `support_sum_by_oid`: cuánto apoyo acumulado recoge en el conjunto retenido;
- `kernel_raw_by_oid`: afinidad absoluta con el kernel contextual;
- `kernel_rel_by_oid`: afinidad relativa dentro de la clase;
- `hyp_rel_by_oid`: apoyo relativo heredado de hipótesis/prior;
- `coverage_ok_by_oid`: si el objeto tiene cobertura mínima suficiente;
- `supported`: banda fuerte de objetos respaldados;
- `soft_supported`: banda suave de objetos todavía plausibles.

Lectura correcta:

- `supported` y `soft_supported` no significan matching final;
- significan "objeto contextual y relacionalmente compatible con este frame".

## Qué significa realmente el top-k

Hay tres recortes distintos que conviene no mezclar:

1. `beam_width`
   La búsqueda ya explora solo una parte del espacio de estados.
2. `topk_sets`
   De las hipótesis encontradas se conservan solo las primeras.
3. `context_k`
   El kernel contextual no usa todas las hipótesis retenidas, sino solo las
   primeras que sirven para formar vecindad.

Consecuencia práctica:

- si dos objetos muy parecidos compiten y uno queda ligeramente por debajo del
  top visible, puede desaparecer del `prior` directo;
- más tarde puede reaparecer parcialmente por afinidad con el kernel;
- aun así el sesgo del top-k no desaparece por completo.

## Uso por detección y por candidato

### `allow_for_report`

Archivo:

- `REMIND/association/policy/sets_rule_policy.py`

Finalidad:

- permitir o bloquear el uso de contexto según el diagnóstico visual de la
  detección.

Lectura:

- una clase puede tener contexto activo y, aun así, una detección concreta no
  usarlo si su estado visual no es apto.

### `bonus_for_candidate`

Archivo:

- `REMIND/association/policy/sets_rule_policy.py`

Finalidad:

- convertir el contexto ya activado en una contribución escalar por candidato.

Componentes principales:

- soporte local;
- soporte global;
- contradicción;
- calidad contextual;
- términos explicativos como `compat_rel`, `kernel_rel` e `hyp_rel`.

### `SETS_RESCUE`

Archivo:

- `REMIND/association/policy/candidate_score_policy.py`

Finalidad:

- rescatar candidatos cuya similitud visual no alcanza `match_thr`, pero que
  sí quedan por encima al sumar bonus contextual positivo.

Lectura:

- no recupera hipótesis omitidas;
- rescata candidatos concretos en el gating del matching.

### `candidate_context_veto_reason`

Archivo:

- `REMIND/association/policy/candidate_score_policy.py`

Finalidad:

- expulsar candidatos conocidos cuando el contexto de `sets` los contradice de
  forma suficiente.

Ramas principales:

- `OUTSIDE_CTX`: el objeto queda fuera de shortlist y de soporte suave, y la
  clase muestra suficiente potencia de poda como para confiar en esa exclusión;
- `LOCAL_CTX_CONTRADICTION`: el objeto tiene contexto local maduro, pero el
  kernel visible del frame contradice sus vecinos esperados.

### `known_plausible_keep`

Archivo:

- `REMIND/association/policy/known_plausible_keep_policy.py`

Finalidad:

- separar dos conceptos:
  - seguir vivo para el matching actual;
  - seguir siendo un conocido todavía plausible para ambigüedad temporal.

Lectura:

- un candidato puede caer operativamente del matching y seguir siendo relevante
  para explicar una ambigüedad conocida;
- solo los vetos fuertes lo expulsan también de esa plausibilidad conocida.

## Edge cases que siguen vigentes

- frames sin anchors: el contexto puede seguir existiendo, pero suele degradarse;
- clases con muchos objetos parecidos: el top-k puede sesgar `prior` y
  `shortlist`;
- objetos ausentes del top visible: pueden recuperar algo de soporte vía
  kernel, no vía hipótesis exhaustivas;
- detecciones débiles: pueden no usar contexto aunque la clase sí lo tenga;
- candidatos visualmente flojos: pueden entrar por `SETS_RESCUE` si el contexto
  los respalda;
- candidatos conocidos pero contextual y localmente incompatibles: pueden
  salir por veto fuerte aunque visualmente no sean los peores.
