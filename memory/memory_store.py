# memory/memory_store.py

from memory.ambiguous_track import AmbiguousTrack
from memory.anchor_view_store import AnchorViewStore
from memory.provisional_new_track import ProvisionalNewTrack
from memory.tracked_object import TrackedObject


class MemoryStore:
    """
    Container for all persistent objects in memory.

    Responsabilidad:
      - assign unique object_id values
      - store objects
      - indexar por class_id
      - asignar instance_label
    """

    def __init__(self, config: dict, start_object_id: int = 0):
        """
        Initialize memory and internal counters.
        """
        self.config = config

        self.objects = {}          # object_id -> TrackedObject
        self.objects_by_class = {} # class_id -> set(object_id)
        self.next_object_id = int(start_object_id)
        self.ambiguous_tracks = {}  # temp_id -> AmbiguousTrack
        self.next_ambiguous_id = 0
        self.provisional_new_tracks = {}  # temp_id -> ProvisionalNewTrack
        self.next_provisional_id = 0

        self.class_instance_counters = {}
        mem_cfg = (config.get("memory", {}) or {}) if isinstance(config, dict) else {}
        self.anchor_views = AnchorViewStore(config=mem_cfg)

    def alloc_instance_label(self, tracked_object: TrackedObject) -> str:
        """
        Generate a unique instance_label per class (for example CHAIR_1).
        Incrementa el contador interno asociado al class_id.
        """
        cid = int(tracked_object.class_id)

        base = tracked_object.class_name
        if not base:
            base = f"CLASS{cid}"

        base = str(base).upper()

        next_i = int(self.class_instance_counters.get(cid, 0)) + 1
        self.class_instance_counters[cid] = next_i

        return f"{base}_{next_i}"

    def ensure_instance_label(self, tracked_object: TrackedObject) -> None:
        """
        Ensure the object has a valid instance_label.
        Si ya existe, sincroniza el contador interno.
        """
        if tracked_object is None:
            return

        if tracked_object.instance_label:
            cid = int(tracked_object.class_id)
            label = str(tracked_object.instance_label)

            if "_" in label:
                tail = label.rsplit("_", 1)[-1]
                if tail.isdigit():
                    suffix = int(tail)
                    cur = int(self.class_instance_counters.get(cid, 0))
                    if suffix > cur:
                        self.class_instance_counters[cid] = suffix
            return

        tracked_object.instance_label = self.alloc_instance_label(tracked_object)

    # ------------------------------------------------------------------
    # Creation / deletion
    # ------------------------------------------------------------------

    def create_tracked_object(
        self,
        class_id: int,
        timestamp: float,
        class_name=None
    ):
        """
        Create a new TrackedObject with a unique object_id,
        assign instance_label, and register it in memory.
        """
        object_id = self.next_object_id
        self.next_object_id += 1

        tracked_object = TrackedObject(
            object_id=object_id,
            class_id=class_id,
            timestamp=timestamp,
            config=self.config,
            class_name=class_name,
        )

        self.ensure_instance_label(tracked_object)
        self.add(tracked_object)
        return tracked_object

    def create_tracked_object_with_id(
        self,
        *,
        object_id: int,
        class_id: int,
        timestamp: float,
        class_name=None,
    ):
        """
        Create or retrieve a TrackedObject with a prefixed object_id.
        Used to materialize identities already committed upstream.
        """
        object_id = int(object_id)
        existing = self.objects.get(int(object_id), None)
        if existing is not None:
            return existing

        if int(object_id) >= int(self.next_object_id):
            self.next_object_id = int(object_id) + 1

        tracked_object = TrackedObject(
            object_id=object_id,
            class_id=class_id,
            timestamp=timestamp,
            config=self.config,
            class_name=class_name,
        )

        self.ensure_instance_label(tracked_object)
        self.add(tracked_object)
        return tracked_object

    def add(self, tracked_object: TrackedObject):
        """
        Insert a TrackedObject into memory and update
        indexes by class_id.
        """
        self.ensure_instance_label(tracked_object)

        self.objects[tracked_object.object_id] = tracked_object

        cid = tracked_object.class_id
        if cid not in self.objects_by_class:
            self.objects_by_class[cid] = set()

        self.objects_by_class[cid].add(tracked_object.object_id)

    def remove(self, object_id: int):
        """
        Remove an object from memory and update indexes.
        No falla si el ID no existe.
        """
        obj = self.objects.pop(object_id, None)
        if obj is None:
            return

        self.objects_by_class[obj.class_id].discard(object_id)

    def create_ambiguous_track(
        self,
        class_id: int,
        timestamp: float,
        candidate_ids: list[int],
        candidate_scores: dict[int, float] | None = None,
        class_name=None,
        ttl: int = 5,
    ) -> AmbiguousTrack:
        temp_id = int(self.next_ambiguous_id)
        self.next_ambiguous_id += 1
        track = AmbiguousTrack(
            temp_id=temp_id,
            class_id=int(class_id),
            class_name=class_name,
            candidate_ids=list(candidate_ids or []),
            candidate_scores=candidate_scores or {},
            timestamp=float(timestamp),
            ttl=int(ttl),
            config=self.config,
        )
        self.ambiguous_tracks[int(temp_id)] = track
        return track

    @staticmethod
    def _track_related_ids(track) -> list[int]:
        if track is None:
            return []
        related = getattr(track, "related_known_ids", None)
        if isinstance(related, list) and related:
            return [int(x) for x in related]
        support = getattr(track, "support_known_ids", None)
        if isinstance(support, list) and support:
            return [int(x) for x in support]
        candidates = getattr(track, "current_candidate_ids", None)
        if isinstance(candidates, list) and candidates:
            return [int(x) for x in candidates]
        candidates = getattr(track, "candidate_ids", None)
        if isinstance(candidates, list) and candidates:
            return [int(x) for x in candidates]
        return []

    def get_ambiguous(self, temp_id: int):
        return self.ambiguous_tracks.get(int(temp_id), None)

    def remove_ambiguous(self, temp_id: int) -> None:
        self.ambiguous_tracks.pop(int(temp_id), None)

    def all_ambiguous_tracks(self):
        return list(self.ambiguous_tracks.values())

    def create_provisional_new_track(
        self,
        class_id: int,
        timestamp: float,
        support_known_ids: list[int] | None = None,
        support_known_scores: dict[int, float] | None = None,
        class_name=None,
        ttl: int = 5,
        context_mode: str = "none",
        reason: str = "UNCERTAIN_NEW",
    ) -> ProvisionalNewTrack:
        temp_id = int(self.next_provisional_id)
        self.next_provisional_id += 1
        track = ProvisionalNewTrack(
            temp_id=temp_id,
            class_id=int(class_id),
            class_name=class_name,
            support_known_ids=support_known_ids or [],
            support_known_scores=support_known_scores or {},
            context_mode=str(context_mode or "none"),
            timestamp=float(timestamp),
            ttl=int(ttl),
            reason=str(reason or "UNCERTAIN_NEW"),
            config=self.config,
        )
        self.provisional_new_tracks[int(temp_id)] = track
        return track

    def get_provisional(self, temp_id: int):
        return self.provisional_new_tracks.get(int(temp_id), None)

    def remove_provisional(self, temp_id: int) -> None:
        self.provisional_new_tracks.pop(int(temp_id), None)

    def all_provisional_new_tracks(self):
        return list(self.provisional_new_tracks.values())

    def find_best_provisional_match(
        self,
        class_id: int,
        support_known_ids: list[int] | None = None,
        min_overlap: float = 0.5,
    ):
        cand_set = set(int(x) for x in (support_known_ids or []))
        if not cand_set:
            return None

        best = None
        best_overlap = float(min_overlap)
        for track in self.provisional_new_tracks.values():
            if int(getattr(track, "class_id", -1)) != int(class_id):
                continue
            tr_set = set(int(x) for x in (self._track_related_ids(track) or []))
            if not tr_set:
                continue
            inter = int(len(cand_set & tr_set))
            if inter <= 0:
                continue
            overlap = float((2.0 * inter) / float(len(cand_set) + len(tr_set)))
            if overlap >= best_overlap:
                best_overlap = float(overlap)
                best = track
        return best

    def find_best_provisional_origin(
        self,
        class_id: int,
        support_known_ids: list[int] | None = None,
        min_overlap: float = 0.5,
    ):
        best = self.find_best_provisional_match(
            class_id=int(class_id),
            support_known_ids=support_known_ids,
            min_overlap=float(min_overlap),
        )
        if best is not None:
            return best

        if support_known_ids:
            return None

        same_class = [
            track
            for track in self.provisional_new_tracks.values()
            if int(getattr(track, "class_id", -1)) == int(class_id)
            and not str(getattr(track, "reason", "") or "").upper().startswith("UNCERTAIN_PARENT")
        ]
        if len(same_class) == 1:
            return same_class[0]
        return None

    def find_best_ambiguous_match(
        self,
        class_id: int,
        candidate_ids: list[int],
        min_overlap: float = 0.5,
        exclude_temp_ids: set[int] | None = None,
    ):
        cand_set = set(int(x) for x in (candidate_ids or []))
        if not cand_set:
            return None

        best = None
        best_overlap = float(min_overlap)
        excluded = set(int(x) for x in (exclude_temp_ids or set()))
        for track in self.ambiguous_tracks.values():
            temp_id = int(getattr(track, "temp_id", -1))
            if int(temp_id) in excluded:
                continue
            if int(getattr(track, "class_id", -1)) != int(class_id):
                continue
            tr_ids = self._track_related_ids(track)
            tr_set = set(int(x) for x in (tr_ids or []))
            if not tr_set:
                continue
            inter = int(len(cand_set & tr_set))
            if inter <= 0:
                continue
            overlap = float((2.0 * inter) / float(len(cand_set) + len(tr_set)))
            if overlap >= best_overlap:
                best_overlap = float(overlap)
                best = track
        return best

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def get(self, object_id: int):
        """
        Return the TrackedObject associated with object_id,
        or None if it does not exist.
        """
        return self.objects.get(object_id, None)

    def get_by_class(self, class_id: int):
        """
        Return the list of active objects belonging
        to a specific class.
        """
        ids = self.objects_by_class.get(class_id, [])
        return [self.objects[i] for i in ids]

    def all_objects(self):
        """
        Return all currently stored objects.
        """
        return list(self.objects.values())

