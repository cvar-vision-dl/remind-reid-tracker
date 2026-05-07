from __future__ import annotations


class SetsContextProvider:
    def __init__(self, *, scores, neighbor_sets_influence, memory_store):
        self.scores = scores
        self.neighbor_sets_influence = neighbor_sets_influence
        self.memory_store = memory_store

    def compute_if_needed(
        self,
        *,
        out,
        detections: list,
        timestamp: float,
        runtime: dict,
        timer,
    ) -> None:
        if not bool(runtime.get("want_sets", False)):
            return

        anchors = [int(x) for x in (out.reliable_anchor_object_ids or [])]
        vocab_size = int(len(self.memory_store.all_objects()))
        out.neighbor_sets_out = timer.run(
            "neighbor_sets",
            self.scores.compute_neighbor_sets,
            detections=detections,
            anchor_object_ids=anchors,
            timestamp=timestamp,
            vocab_size=vocab_size,
        )
        timer.extend(getattr(getattr(self.scores, "neigh_sets", None), "last_timings_seconds", {}), prefix="neighbor_sets/")
        if isinstance(out.neighbor_sets_out, dict):
            dbg_pack = out.neighbor_sets_out.get("debug", None)
            if isinstance(dbg_pack, dict):
                meta = dbg_pack.get("meta", None)
                if not isinstance(meta, dict):
                    meta = {}
                    dbg_pack["meta"] = meta
                meta.setdefault("anchors_raw", list(anchors))
                meta.setdefault("anchors_filtered", list(anchors))

    def build_context(self, neighbor_sets_out) -> dict:
        cache_owner = neighbor_sets_out.get("debug", None) if isinstance(neighbor_sets_out, dict) else None
        if not isinstance(cache_owner, dict):
            cache_owner = neighbor_sets_out if isinstance(neighbor_sets_out, dict) else None

        if isinstance(cache_owner, dict):
            cached = cache_owner.get("_cached_sets_context", None)
            if isinstance(cached, dict):
                return cached

        if getattr(self.neighbor_sets_influence, "enabled", False):
            ctx = self.neighbor_sets_influence.build_context(neighbor_sets_out)
            if isinstance(ctx, dict):
                if isinstance(cache_owner, dict):
                    cache_owner["_cached_sets_context"] = dict(ctx)
                return ctx

        disabled = {"enabled": False}
        if isinstance(cache_owner, dict):
            cache_owner["_cached_sets_context"] = dict(disabled)
        return disabled
