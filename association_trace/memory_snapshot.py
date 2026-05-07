from __future__ import annotations


def _display_label(obj) -> str:
    label = str(getattr(obj, "instance_label", "") or "").strip()
    if label:
        return label.lower()
    class_name = str(getattr(obj, "class_name", "") or "").strip().lower()
    object_id = int(getattr(obj, "object_id", -1))
    if class_name:
        return f"{class_name}_{object_id}"
    return f"id_{object_id}"


def _count_object_descriptors(obj) -> tuple[int, dict]:
    appearance = getattr(obj, "appearance", None)
    channels = getattr(appearance, "channels", {}) or {}
    breakdown = {}
    total = 0
    for name, channel in channels.items():
        work = len(getattr(channel, "work_protos", []) or [])
        stable = len(getattr(channel, "stable_protos", []) or [])
        count = int(work + stable)
        breakdown[str(name)] = count
        total += count
    return int(total), breakdown


def _count_parts_descriptors(obj) -> tuple[int, dict]:
    parts = getattr(obj, "parts", None)
    channels = getattr(parts, "channels", {}) or {}
    breakdown = {}
    total = 0
    for name, channel in channels.items():
        work = len(getattr(channel, "work_protos", []) or [])
        stable = len(getattr(channel, "stable_protos", []) or [])
        count = int(work + stable)
        breakdown[str(name)] = count
        total += count
    return int(total), breakdown


def _count_background_descriptors(obj) -> tuple[int, dict]:
    background = getattr(obj, "background", None)
    banks = {
        "inner_global_work": getattr(background, "inner_global_work", None),
        "outer_global_work": getattr(background, "outer_global_work", None),
        "inner_global_stable": getattr(background, "inner_global_stable", None),
        "outer_global_stable": getattr(background, "outer_global_stable", None),
        "inner_partials_work": getattr(background, "inner_partials_work", None),
        "outer_partials_work": getattr(background, "outer_partials_work", None),
        "inner_partials_stable": getattr(background, "inner_partials_stable", None),
        "outer_partials_stable": getattr(background, "outer_partials_stable", None),
    }
    breakdown = {}
    total = 0
    for name, bank in banks.items():
        count = len(getattr(bank, "prototypes", []) or []) if bank is not None else 0
        breakdown[str(name)] = int(count)
        total += int(count)
    return int(total), breakdown


def build_association_memory_snapshot(*, memory_store, frame_id: int, timestamp: float | None = None) -> dict:
    rows = []
    objects = []
    if memory_store is not None and hasattr(memory_store, "all_objects"):
        objects = list(memory_store.all_objects() or [])

    for obj in objects:
        obj_desc_count, obj_desc_breakdown = _count_object_descriptors(obj)
        parts_desc_count, parts_desc_breakdown = _count_parts_descriptors(obj)
        bg_desc_count, bg_desc_breakdown = _count_background_descriptors(obj)
        rows.append(
            {
                "label": _display_label(obj),
                "label_raw": str(getattr(obj, "instance_label", "") or ""),
                "object_id": int(getattr(obj, "object_id", -1)),
                "class_id": int(getattr(obj, "class_id", -1)),
                "class_name": str(getattr(obj, "class_name", "") or ""),
                "state": str(getattr(obj, "state", "") or ""),
                "hits": int(getattr(obj, "hits", 0) or 0),
                "last_seen": float(getattr(obj, "last_seen", 0.0) or 0.0),
                "first_seen": float(getattr(obj, "first_seen", 0.0) or 0.0),
                "obj_desc_count": int(obj_desc_count),
                "bg_desc_count": int(bg_desc_count),
                "parts_desc_count": int(parts_desc_count),
                "obj_desc_breakdown": obj_desc_breakdown,
                "bg_desc_breakdown": bg_desc_breakdown,
                "parts_desc_breakdown": parts_desc_breakdown,
            }
        )

    rows.sort(key=lambda row: int(row.get("object_id", -1)))
    return {
        "frame_id": int(frame_id),
        "timestamp": None if timestamp is None else float(timestamp),
        "n_objects": int(len(rows)),
        "object_rows": rows,
    }
