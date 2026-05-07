from __future__ import annotations


class PostAssignmentSupport:
    """Utilidades compartidas para normalizar y materializar resultados post-Hungarian."""

    def normalize_matches(
        self,
        matches: list[tuple[int, int, float]] | None,
    ) -> list[tuple[int, int, float]]:
        by_det_id: dict[int, tuple[int, int, float]] = {}
        order: list[int] = []
        for det_id, obj_id, score_final in (matches or []):
            det_id = int(det_id)
            if det_id not in by_det_id:
                order.append(int(det_id))
            by_det_id[int(det_id)] = (int(det_id), int(obj_id), float(score_final))
        return [by_det_id[int(det_id)] for det_id in order if int(det_id) in by_det_id]

    def build_assignment_state(
        self,
        matches: list[tuple[int, int, float]] | None,
    ) -> dict:
        normalized_matches = self.normalize_matches(matches)
        return {
            "decided_matches": list(normalized_matches),
            "assigned_by_det_id": {
                int(det_id): int(obj_id) for det_id, obj_id, _ in (normalized_matches or [])
            },
        }

    def merge_matches(
        self,
        *,
        base_matches: list[tuple[int, int, float]] | None,
        override_matches: list[tuple[int, int, float]] | None,
    ) -> list[tuple[int, int, float]]:
        merged = self.normalize_matches(base_matches)
        if not override_matches:
            return merged
        override_det_ids = {int(det_id) for det_id, _, _ in (override_matches or [])}
        merged = [item for item in merged if int(item[0]) not in override_det_ids]
        merged.extend(
            (int(det_id), int(obj_id), float(score_final))
            for det_id, obj_id, score_final in (override_matches or [])
        )
        return self.normalize_matches(merged)

    def normalize_create_entries(
        self,
        to_create: list[tuple[int, int]] | list[dict] | None,
    ) -> list[dict]:
        by_det_id: dict[int, dict] = {}
        order: list[int] = []
        for item in (to_create or []):
            if isinstance(item, dict):
                det_id = int(item.get("det_id", -1))
                class_id = int(item.get("class_id", -1))
                base = dict(item)
            else:
                det_id = int(item[0])
                class_id = int(item[1])
                base = {"det_id": int(det_id), "class_id": int(class_id)}
            if det_id < 0 or class_id < 0:
                continue
            if det_id not in by_det_id:
                order.append(int(det_id))
            base["det_id"] = int(det_id)
            base["class_id"] = int(class_id)
            by_det_id[int(det_id)] = dict(base)
        return [dict(by_det_id[int(det_id)]) for det_id in order if int(det_id) in by_det_id]

    def drop_create_entries_for_det_ids(
        self,
        create_entries: list[dict] | None,
        det_ids,
    ) -> list[dict]:
        blocked = {int(x) for x in (det_ids or [])}
        return [
            dict(item)
            for item in (create_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) not in blocked
        ]

    def merge_ambiguous_entries(
        self,
        *,
        base_entries: list[dict] | None,
        extra_entries: list[dict] | None,
    ) -> list[dict]:
        merged_by_det_id: dict[int, dict] = {}
        order: list[int] = []
        for item in list(base_entries or []) + list(extra_entries or []):
            if not isinstance(item, dict):
                continue
            det_id = int(item.get("det_id", -1))
            if det_id < 0:
                continue
            if det_id not in merged_by_det_id:
                order.append(int(det_id))
            merged_by_det_id[int(det_id)] = dict(item)
        return [dict(merged_by_det_id[int(det_id)]) for det_id in order if int(det_id) in merged_by_det_id]

    def same_ambiguous_det_ids(
        self,
        prev_entries: list[dict] | None,
        next_entries: list[dict] | None,
    ) -> bool:
        def signature(entries: list[dict] | None) -> set[tuple]:
            out = set()
            for item in (entries or []):
                if not isinstance(item, dict):
                    continue
                det_id = int(item.get("det_id", -1))
                if det_id < 0:
                    continue
                candidate_ids = tuple(
                    sorted(
                        int(x)
                        for x in (item.get("candidate_ids", []) or [])
                        if x is not None
                    )
                )
                out.add(
                    (
                        int(det_id),
                        int(item.get("class_id", -1)),
                        candidate_ids,
                        str(item.get("reason", "") or ""),
                    )
                )
            return out

        return bool(signature(prev_entries) == signature(next_entries))

    def build_final_decision_pack(
        self,
        *,
        decided_matches: list[tuple[int, int, float]],
        create_entries: list[dict],
        ambiguous_entries: list[dict],
        provisional_entries: list[dict],
    ) -> dict:
        ambiguous_by_det_id = {
            int(item["det_id"]): dict(item)
            for item in (ambiguous_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        provisional_by_det_id = {
            int(item["det_id"]): dict(item)
            for item in (provisional_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }
        create_by_det_id = {
            int(item["det_id"]): dict(item)
            for item in (create_entries or [])
            if isinstance(item, dict) and int(item.get("det_id", -1)) >= 0
        }

        final_matches: list[tuple[int, int, float]] = []
        assigned_by_det_id: dict[int, int] = {}
        score_final_by_det_id: dict[int, float] = {}
        blocked_match_det_ids = set(ambiguous_by_det_id) | set(provisional_by_det_id)
        for det_id, obj_id, score_final in self.normalize_matches(decided_matches):
            if int(det_id) in blocked_match_det_ids:
                continue
            final_matches.append((int(det_id), int(obj_id), float(score_final)))
            assigned_by_det_id[int(det_id)] = int(obj_id)
            score_final_by_det_id[int(det_id)] = float(score_final)

        blocked_create_det_ids = set(assigned_by_det_id) | set(ambiguous_by_det_id) | set(provisional_by_det_id)
        final_create_entries = [
            dict(item)
            for det_id, item in sorted(create_by_det_id.items())
            if int(det_id) not in blocked_create_det_ids
        ]

        final_ambiguous_entries = [dict(item) for _, item in sorted(ambiguous_by_det_id.items())]
        final_provisional_entries = [dict(item) for _, item in sorted(provisional_by_det_id.items())]

        return {
            "matches": list(final_matches),
            "create_entries": list(final_create_entries),
            "ambiguous_entries": list(final_ambiguous_entries),
            "provisional_entries": list(final_provisional_entries),
            "assigned_by_det_id": dict(assigned_by_det_id),
            "score_final_by_det_id": dict(score_final_by_det_id),
        }
