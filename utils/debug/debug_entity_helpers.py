from __future__ import annotations


def get_track_label(memory_store, track_id):
    """Return the memory instance_label for a track id, or None."""
    if track_id is None:
        return None
    obj = memory_store.get(int(track_id)) if memory_store is not None else None
    return getattr(obj, "instance_label", None) if obj else None


def get_track_class_name(memory_store, track_id):
    """Return the memory class_name for a track id, or None."""
    if track_id is None:
        return None
    obj = memory_store.get(int(track_id)) if memory_store is not None else None
    return getattr(obj, "class_name", None) if obj else None


def det_local(det_id, det_id_to_local=None):
    """Printable detection ID: use the frame-local index when available."""
    if det_id is None:
        return "?"
    did = int(det_id)
    if isinstance(det_id_to_local, dict):
        local_id = det_id_to_local.get(int(did), None)
        if local_id is not None:
            return int(local_id)
    return int(did)


def short_class_code(name: str | None, n: int = 3) -> str | None:
    """Recorta class_name a n letras (upper)."""
    if not name:
        return None
    s = str(name).strip().upper()
    if not s:
        return None
    return s[: max(1, int(n))]


def short_instance_label(lbl: str | None, n: int = 3) -> str | None:
    """Convert 'LAPTOP_1' -> 'LAP-1'. If there is no numeric suffix, shorten the class."""
    if not lbl:
        return None

    s = str(lbl).strip()
    if not s:
        return None

    parts = s.split("_")
    if len(parts) >= 2:
        tail = parts[-1].strip()
        head = "_".join(parts[:-1]).strip()
        if tail.isdigit():
            code = short_class_code(head, n=n) or short_class_code(s, n=n)
            return f"{code}-{int(tail)}" if code else f"ID{int(tail)}"

    code = short_class_code(s, n=n)
    return code if code else None


def get_track_label_short(memory_store, track_id, n: int = 3) -> str | None:
    """Devuelve instance_label acortado (p.ej. 'LAPTOP_1' -> 'LAP-1')."""
    lbl = get_track_label(memory_store, track_id)
    return short_instance_label(lbl, n=n) if lbl else None


def pair_label(det_id, track_id, memory_store, det_id_to_local=None):
    """'detId-shortLabel' label for tables."""
    d = det_local(det_id, det_id_to_local)
    lbl = get_track_label_short(memory_store, track_id, n=3)
    return f"{d}-{lbl}" if lbl else f"{d}-?"


def assignment_tokens(
    det_oid_pairs: list[tuple[int, int]],
    memory_store,
    det_id_to_local=None,
) -> str:
    """Formatea pares (det_id, obj_id) como 'detId-LAB-1, ...'."""
    toks = []
    for did, oid in det_oid_pairs or []:
        d = det_local(int(did), det_id_to_local)
        lbl = get_track_label_short(memory_store, int(oid), n=3) or f"ID{int(oid)}"
        toks.append(f"{d}-{lbl}")
    return ",".join(toks)
