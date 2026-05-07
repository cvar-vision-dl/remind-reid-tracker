# Reglas activas de `distance`

## Alcance actual

El bloque de `distance` ya no participa en el `matching` principal ni en la
construcción de `score_final`.

Su uso activo hoy queda limitado a dos piezas:

1. memoria relacional entre objetos visibles;
2. desambiguación post-asignación de conjuntos conocidos.

## Pipeline vigente

### 1. Update de memoria relacional

Archivos:

- `APP2/Src/update/update_general.py`
- `APP2/Src/memory/neighbor_distance_graph.py`
- `APP2/Src/memory/tracked_object.py`

Responsabilidad:

- construir observaciones geométricas entre objetos visibles del frame;
- actualizar `NeighborDistanceGraph` por objeto;
- conservar historial relacional reutilizable entre frames.

### 2. Desambiguación de conocidos

Archivos:

- `APP2/Src/association/engine/assignment_result_applier.py`
- `APP2/Src/association/disambiguation/known_set_distance_disambiguator.py`
- `APP2/Src/association/disambiguation/pair_anchor_discriminator.py`

Responsabilidad:

- resolver componentes ambiguos de IDs conocidos;
- comparar hipótesis con memoria relacional ya acumulada;
- elegir entre candidatos conocidos cuando el conjunto correcto ya está acotado.

## Alcance en matching

- `distance` no altera `score_assign`;
- `distance` no altera `score_final`;
- la configuración válida de `distance` vive hoy en `memory.*` y en
  `association.ambiguous_tracks.known_set_distance_disambiguation.*`.
