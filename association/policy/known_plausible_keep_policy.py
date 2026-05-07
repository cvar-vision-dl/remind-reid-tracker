from __future__ import annotations


class KnownPlausibleKeepPolicy:
    """
    Decide si un candidato conocido sigue siendo plausible para razonamiento
    de ambiguedad temporal.

    Esta policy formaliza la semantica de `known_plausible_keep`:
      - parte de "keep" por defecto
      - solo cae si existe un veto contextual fuerte

    Contrato publico:
      - el campo expuesto se llama `known_plausible_keep`;
      - no expresa elegibilidad para matching final.

    No decide elegibilidad para Hungarian ni para la decision final; eso sigue
    siendo responsabilidad de `decision_keep`.
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
