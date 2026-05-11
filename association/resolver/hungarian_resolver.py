from __future__ import annotations

class HungarianResolver:
    """Build the cost matrix and resolve MATCH vs NEW on a prepared table."""

    def __init__(
        self,
        *,
        enable_dummies: bool,
    ):
        self.enable_dummies = bool(enable_dummies)

    def build_cost_matrix(
        self,
        *,
        remaining_det_ids: list[int],
        cand_obj_list: list[int],
        table_assign_rem: dict,
        reports: dict,
        report_status_fn,
        resolve_dummy_score_fn,
    ) -> list[list[float]]:
        n = len(remaining_det_ids)
        m = len(cand_obj_list)
        n_dummies = n if self.enable_dummies else 0
        big = 1e6
        cost = [[big for _ in range(m + n_dummies)] for _ in range(n)]

        for i, did in enumerate(remaining_det_ids):
            row_assign = table_assign_rem.get(int(did), {}) or {}

            for j, oid in enumerate(cand_obj_list):
                score_assign = row_assign.get(int(oid), None)
                if score_assign is None:
                    continue
                # Hungarian optimizes the stable identity signal. Set bonuses
                # still open candidates through gating, but do not steal
                # identities during global assignment.
                cost[i][j] = -float(score_assign)

            if n_dummies > 0:
                dummy_cost = -float(resolve_dummy_score_fn(reports.get(int(did), None)))
                for dj in range(n_dummies):
                    cost[i][m + dj] = float(dummy_cost)

        return cost

    def resolve_assignment(
        self,
        *,
        class_id: int,
        remaining_det_ids: list[int],
        cand_obj_list: list[int],
        row_ind,
        col_ind,
        table_sim_rem: dict,
        table_final_rem: dict,
        reports: dict,
        match_thr: float,
        min_match_score: float,
        report_status_fn,
    ) -> tuple[list[tuple[int, int, float]], list[tuple[int, int]]]:
        matches: list[tuple[int, int, float]] = []
        to_create: list[tuple[int, int]] = []
        assigned_det = set()
        m = len(cand_obj_list)

        for i, j in zip(row_ind, col_ind):
            did = int(remaining_det_ids[int(i)])
            assigned_det.add(int(did))

            if int(j) >= m:
                to_create.append((int(did), int(class_id)))
                continue

            oid = int(cand_obj_list[int(j)])
            s_sim = float((table_sim_rem.get(int(did), {}) or {}).get(int(oid), 0.0))
            s_final = float((table_final_rem.get(int(did), {}) or {}).get(int(oid), 0.0))

            accept_hard = bool(s_final >= float(match_thr) and s_sim >= float(min_match_score))
            if accept_hard:
                matches.append((int(did), int(oid), float(s_final)))
            else:
                to_create.append((int(did), int(class_id)))

        for did in remaining_det_ids:
            if int(did) not in assigned_det:
                to_create.append((int(did), int(class_id)))

        return matches, to_create
