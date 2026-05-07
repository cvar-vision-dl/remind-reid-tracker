from __future__ import annotations


class ReliableVisualAnchorPolicy:
    def __init__(self, *, candidate_generator, confirm_thr_strong: float, confirm_clear_margin: float):
        self.candidate_generator = candidate_generator
        self.confirm_thr_strong = float(confirm_thr_strong)
        self.confirm_clear_margin = float(confirm_clear_margin)

    def build_reliable_anchor_pairs(self, frame_out) -> dict[int, int]:
        anchor_det_by_oid: dict[int, tuple[int, float]] = {}

        for rep in frame_out.reports_by_det_id.values():
            cands = rep.candidates
            if not cands:
                continue

            best = self.candidate_generator.pick_best(cands, key="score_sim")
            if best is None:
                continue

            second = self.candidate_generator.pick_second_best(cands, best, key="score_sim")
            best_s = float(best.get("score_sim", 0.0))
            second_s = float(second.get("score_sim", 0.0)) if second is not None else 0.0

            if best_s >= self.confirm_thr_strong and (best_s - second_s) >= self.confirm_clear_margin:
                oid = int(best["object_id"])
                det_id = int(getattr(rep, "det_id", -1))
                prev = anchor_det_by_oid.get(int(oid), None)
                if prev is None or float(best_s) > float(prev[1]):
                    anchor_det_by_oid[int(oid)] = (int(det_id), float(best_s))

        return {int(oid): int(det_id) for oid, (det_id, _) in anchor_det_by_oid.items()}

    def build_reliable_anchor_ids(self, frame_out) -> set[int]:
        return set(int(x) for x in self.build_reliable_anchor_pairs(frame_out).keys())
