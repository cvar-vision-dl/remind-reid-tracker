# Reglas activas de `distance`

## Alcance

El bloque de `distance` no participa en el `matching` principal ni en la
construcción de `score_final`.

Su uso activo queda limitado a dos piezas:

1. memoria relacional entre objetos visibles;
2. desambiguación post-asignación de conjuntos conocidos.

## Pipeline vigente

### 1. Update de memoria relacional

Archivos:

- `REMIND/update/update_general.py`
- `REMIND/memory/neighbor_distance_graph.py`
- `REMIND/memory/tracked_object.py`

Responsabilidad:

- construir observaciones geométricas entre objetos visibles del frame;
- actualizar `NeighborDistanceGraph` por objeto;
- conservar historial relacional reutilizable entre frames.

### 2. Desambiguación de conocidos

Archivos:

- `REMIND/association/engine/assignment_result_applier.py`
- `REMIND/association/disambiguation/known_set_distance_disambiguator.py`
- `REMIND/association/disambiguation/pair_anchor_discriminator.py`

Responsabilidad:

- resolver componentes ambiguos de IDs conocidos;
- comparar hipótesis con memoria relacional ya acumulada;
- elegir entre candidatos conocidos cuando el conjunto correcto ya está acotado.

## Alcance en matching

- `distance` no altera `score_assign`;
- `distance` no altera `score_final`;
- la configuración válida de `distance` vive hoy en `memory.*` y en
  `association.ambiguous_tracks.known_set_distance_disambiguation.*`.
