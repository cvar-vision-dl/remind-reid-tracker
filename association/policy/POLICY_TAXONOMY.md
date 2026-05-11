# Taxonomía de `policy`

## Objetivo

Este documento fija una lectura por capas de las reglas de asociación.

La idea es que el árbol de `policy/` pueda leerse como un conjunto de puertas:

1. selección de anchors visuales fiables
2. shaping contextual de candidatos
3. shaping de score por fila
4. traducción y reinterpretación de outcomes

## Capa 1. Reliable Visual Anchors

- `REMIND/association/policy/confirmation_policy.py`

## Capa 2. Candidate Context

- estado actual:
  - no hay un wrapper `REMIND/association/policy/path/candidate_context.py`
  - esta capa vive repartida entre:
    - `REMIND/association/policy/candidate_score_policy.py`
    - `REMIND/association/policy/known_plausible_keep_policy.py`
    - `REMIND/association/policy/sets_rule_policy.py`

Responsabilidad:

- shortlist
- prior
- rescue
- soft-gate
- veto contextual fuerte
- plausibilidad conocida para ambigüedad temporal

## Capa 3. Candidate Rows

- implementación principal: `REMIND/association/policy/candidate_score_policy.py`
- reglas base que alteran la fila:
  - `REMIND/association/policy/sets_rule_policy.py`
  - `REMIND/association/policy/known_plausible_keep_policy.py`

Responsabilidad:

- construir tablas `score_sim / score_assign / score_final`
- aplicar veto contextual y plausibilidad conocida
- resolver rescate por `SETS_RESCUE`
- las ayudas de `distance` no forman parte de este camino de matching

## Capa 4. Outcomes

- implementación: `REMIND/association/policy/outcome_policy.py`

Responsabilidad:

- diagnóstico de ambigüedad
- reinterpretación temporal
- decisión final de `MATCH / NEW / AMBIGUOUS / PROVISIONAL`

## Compatibilidad

La taxonomía sigue siendo semántica: algunas capas viven repartidas entre varias
policies reales del módulo. Este documento prioriza la implementación activa
frente a una taxonomía idealizada para no inducir a error.
