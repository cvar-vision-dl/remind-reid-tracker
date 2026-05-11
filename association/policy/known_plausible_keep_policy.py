from __future__ import annotations


class KnownPlausibleKeepPolicy:
    """
    Decide whether a known candidate remains plausible for temporal ambiguity
    reasoning.

    This policy formalizes `known_plausible_keep` semantics:
      - starts from default "keep"
      - drops only under a strong contextual veto

    Public contract:
      - the exposed field is named `known_plausible_keep`;
      - it does not express final matching eligibility.

    Does not decide Hungarian eligibility or the final decision; that remains
    the responsibility of `decision_keep`.
    """

    def __init__(self, *, context_veto_reason_fn):
        self.context_veto_reason_fn = context_veto_reason_fn

    def evaluate(
        self,
        *,
        det_class_id: int,
        object_id: int,
        candidate: dict,
        ns_ctx: dict | None,
        neighbor_sets_influence,
    ) -> dict:
        veto_reason = str(
            self.context_veto_reason_fn(
                det_class_id=int(det_class_id),
                object_id=int(object_id),
                candidate=candidate,
                ns_ctx=ns_ctx,
                neighbor_sets_influence=neighbor_sets_influence,
            )
            or ""
        )

        keep = int(not bool(veto_reason))
        return {
            "keep": int(keep),
            "reason": "KNOWN_OK" if keep else str(veto_reason),
            "veto_reason": str(veto_reason),
        }
