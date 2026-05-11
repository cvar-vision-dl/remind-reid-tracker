from __future__ import annotations


class LockResolver:
    """Resolve obvious locks before Hungarian without contextual policy access."""

    def __init__(
        self,
        *,
        locks_enabled: bool,
        locks_object_enabled: bool,
        locks_det_enabled: bool,
        locks_thr: float,
        locks_gap_abs_min: float,
        locks_gap_rel_thr: float,
    ):
        self.locks_enabled = bool(locks_enabled)
        self.locks_object_enabled = bool(locks_object_enabled)
        self.locks_det_enabled = bool(locks_det_enabled)
        self.locks_thr = float(locks_thr)
        self.locks_gap_abs_min = float(locks_gap_abs_min)
        self.locks_gap_rel_thr = float(locks_gap_rel_thr)

    def lock_passes(self, s1: float, s2: float) -> bool:
        if not self.locks_enabled:
            return False
        if s1 < self.locks_thr:
            return False

        gap_abs = float(s1 - s2)
        if gap_abs < self.locks_gap_abs_min:
            return False

        gap_rel = float(gap_abs / max(1e-12, float(s1)))
        return bool(gap_rel >= self.locks_gap_rel_thr)

    def compute_object_locks(
        self,
        det_ids: list[int],
        table_sim: dict[int, dict[int, float]],
        table_final: dict[int, dict[int, float]],
        obj_ids: set[int],
        used_det_ids: set[int],
        used_obj_ids: set[int],
    ) -> list[tuple[int, int, float]]:
        if not (self.locks_enabled and self.locks_object_enabled):
            return []

        proposals: list[tuple[float, int, int]] = []

        for oid in obj_ids:
            if int(oid) in used_obj_ids:
                continue

            scored: list[tuple[float, int]] = []
            for did in det_ids:
                did = int(did)
                if did in used_det_ids:
                    continue
                s = (table_sim.get(did, {}) or {}).get(int(oid), None)
                if s is None:
                    continue
                scored.append((float(s), int(did)))

            if not scored:
                continue

            scored.sort(key=lambda x: float(x[0]), reverse=True)
            s1, did1 = float(scored[0][0]), int(scored[0][1])
            s2 = float(scored[1][0]) if len(scored) > 1 else 0.0

            if self.lock_passes(s1=s1, s2=s2):
                proposals.append((float(s1), int(did1), int(oid)))

        proposals.sort(key=lambda x: float(x[0]), reverse=True)

        locked: list[tuple[int, int, float]] = []
        for _, did, oid in proposals:
            if int(did) in used_det_ids:
                continue
            if int(oid) in used_obj_ids:
                continue

            s_final = float((table_final.get(int(did), {}) or {}).get(int(oid), 0.0))
            locked.append((int(did), int(oid), float(s_final)))
            used_det_ids.add(int(did))
            used_obj_ids.add(int(oid))

        return locked

    def compute_det_locks(
        self,
        det_ids: list[int],
        table_sim: dict[int, dict[int, float]],
        table_final: dict[int, dict[int, float]],
        used_det_ids: set[int],
        used_obj_ids: set[int],
    ) -> list[tuple[int, int, float]]:
        if not (self.locks_enabled and self.locks_det_enabled):
            return []

        proposals: list[tuple[float, int, int]] = []

        for did in det_ids:
            did = int(did)
            if did in used_det_ids:
                continue

            row = table_sim.get(did, None)
            if not row:
                continue

            scored = [(float(s), int(oid)) for oid, s in row.items() if int(oid) not in used_obj_ids]
            if not scored:
                continue

            scored.sort(key=lambda x: float(x[0]), reverse=True)
            s1, oid1 = float(scored[0][0]), int(scored[0][1])
            s2 = float(scored[1][0]) if len(scored) > 1 else 0.0

            if self.lock_passes(s1=s1, s2=s2):
                proposals.append((float(s1), int(did), int(oid1)))

        proposals.sort(key=lambda x: float(x[0]), reverse=True)

        locked: list[tuple[int, int, float]] = []
        for _, did, oid in proposals:
            if int(did) in used_det_ids:
                continue
            if int(oid) in used_obj_ids:
                continue

            s_final = float((table_final.get(int(did), {}) or {}).get(int(oid), 0.0))
            locked.append((int(did), int(oid), float(s_final)))
            used_det_ids.add(int(did))
            used_obj_ids.add(int(oid))

        return locked
