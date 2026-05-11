# update/update_neighbors.py

from __future__ import annotations

from memory.neighbor_graph import NeighborEdge, NeighborGraph
from utils.math import jaccard


class NeighborUpdater:
    """
    Update del grafo de vecinos por episodios (debounce + decay + trim).

    Anti-contamination policy:
      - observe_frame (dist_graph) is always allowed when geometry exists.
      - accept_episode (consolidate stable_context/edges) only if allow_episode=True.
    """

    def __init__(self, config: dict | None = None):
        ncfg = ((config or {}).get("memory", {}) or {}).get("neighbors", {}) or {}

        self.enabled = bool(ncfg.get("enabled", True))

        self.jaccard_thr = float(ncfg.get("jaccard_thr", 0.8))
        self.jaccard_thr = max(0.0, min(1.0, self.jaccard_thr))

        self.debounce_frames = max(1, int(ncfg.get("debounce_frames", 2)))

        self.force_episode_every_frames = max(0, int(ncfg.get("force_episode_every_frames", 0)))

        self.decay_per_episode = float(ncfg.get("decay_per_episode", 1.0))
        self.decay_per_episode = max(1e-12, min(1.0, self.decay_per_episode))

        self.max_neighbors = max(0, int(ncfg.get("max_neighbors", 50)))

        self.trim_strategy = str(ncfg.get("trim_strategy", "weight")).strip().lower()
        if self.trim_strategy not in ("weight", "recent"):
            self.trim_strategy = "weight"

    def update(
        self,
        graph: NeighborGraph,
        dist_graph,
        self_id: int,
        visible_object_ids: list[int] | None,
        timestamp: float,
        frame_id: int | None = None,
        geom_by_object_id: dict | None = None,
        dist_obs_by_other_id: dict | None = None,
        allow_episode: bool = True,
        view_id: int | None = None,
    ) -> None:
        if not self.enabled or not graph.enabled:
            return

        graph.trim_strategy = self.trim_strategy

        sid = int(self_id)
        cur = set(int(x) for x in (visible_object_ids or []) if int(x) != sid)

        if dist_graph is not None:
            if isinstance(dist_obs_by_other_id, dict) and dist_obs_by_other_id:
                for other_id, obs in dist_obs_by_other_id.items():
                    if int(other_id) == sid:
                        continue
                    dist_graph.add_pending_observation(int(other_id), obs)
            elif geom_by_object_id:
                dist_graph.observe_frame(
                    self_id=sid,
                    visible_object_ids=list(cur | {sid}),
                    geom_by_object_id=geom_by_object_id,
                )

        if not allow_episode:
            return

        if self.should_force_episode(graph, frame_id):
            self.accept_episode(graph, dist_graph, cur, timestamp, frame_id, view_id=view_id)
            return

        if graph.stable_context is None:
            self.accept_episode(graph, dist_graph, cur, timestamp, frame_id, view_id=view_id)
            return

        if jaccard(cur, graph.stable_context) >= self.jaccard_thr:
            graph.pending_context = None
            graph.pending_hits = 0
            return

        if graph.pending_context is None or cur != graph.pending_context:
            graph.pending_context = set(cur)
            graph.pending_hits = 1
        else:
            graph.pending_hits += 1

        if graph.pending_hits >= self.debounce_frames:
            self.accept_episode(graph, dist_graph, set(cur), timestamp, frame_id, view_id=view_id)
            graph.pending_context = None
            graph.pending_hits = 0

    def should_force_episode(self, graph: NeighborGraph, frame_id: int | None) -> bool:
        if self.force_episode_every_frames <= 0 or frame_id is None:
            return False
        if graph.last_episode_frame_id is None:
            return False
        return (int(frame_id) - int(graph.last_episode_frame_id)) >= int(self.force_episode_every_frames)

    def accept_episode(
        self,
        graph: NeighborGraph,
        dist_graph,
        new_context: set[int],
        timestamp: float,
        frame_id: int | None,
        view_id: int | None = None,
    ) -> None:
        graph.episode_count += 1
        graph.episode_idx += 1
        ep = int(graph.episode_idx)

        graph.stable_context = set(new_context)
        if frame_id is not None:
            graph.last_episode_frame_id = int(frame_id)

        self.apply_decay(graph)

        for oid in graph.stable_context:
            oid = int(oid)
            e = graph.edges.get(oid)
            if e is None:
                graph.edges[oid] = NeighborEdge(dst_id=oid, timestamp=timestamp, episode_idx=ep)
            elif int(e.last_seen_episode) != ep:
                e.bump(timestamp=timestamp, episode_idx=ep)

        self.trim(graph)

        if dist_graph is not None:
            dist_graph.accept_episode(
                stable_context=set(graph.stable_context),
                timestamp=timestamp,
                episode_idx=ep,
                view_id=view_id,
            )

    def apply_decay(self, graph: NeighborGraph) -> None:
        if self.decay_per_episode >= 1.0 or not graph.edges:
            return
        d = float(self.decay_per_episode)
        for e in graph.edges.values():
            e.weight = float(e.weight) * d

    def trim(self, graph: NeighborGraph) -> None:
        if self.max_neighbors <= 0:
            return
        if len(graph.edges) <= self.max_neighbors:
            return

        items = list(graph.edges.items())
        if self.trim_strategy == "recent":
            items.sort(key=lambda kv: float(kv[1].last_seen_ts), reverse=True)
        else:
            items.sort(key=lambda kv: (float(kv[1].weight), float(kv[1].last_seen_ts)), reverse=True)

        graph.edges = dict(items[: self.max_neighbors])
