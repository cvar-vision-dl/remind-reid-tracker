from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from time import perf_counter


class ExecutionTimer:
    """
    Cronómetro acumulativo por bloques nombrados.
    """

    def __init__(self):
        self._seconds_by_name: OrderedDict[str, float] = OrderedDict()

    @contextmanager
    def measure(self, name: str):
        key = str(name)
        t0 = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - t0
            self._seconds_by_name[key] = self._seconds_by_name.get(key, 0.0) + elapsed

    def run(self, name: str, fn, *args, **kwargs):
        with self.measure(name):
            return fn(*args, **kwargs)

    def add(self, name: str, seconds: float) -> None:
        key = str(name)
        self._seconds_by_name[key] = self._seconds_by_name.get(key, 0.0) + float(seconds)

    def extend(self, timings_seconds: dict[str, float] | None, prefix: str = "") -> None:
        for name, seconds in (timings_seconds or {}).items():
            key = f"{prefix}{name}" if prefix else str(name)
            self.add(key, float(seconds))

    def snapshot_seconds(self) -> dict[str, float]:
        return dict(self._seconds_by_name)

    def total_seconds(self) -> float:
        return float(sum(self._seconds_by_name.values()))

    def format_ms(self, precision: int = 2, include_total: bool = True) -> str:
        p = max(0, int(precision))
        parts = [
            f"{name}={seconds * 1000.0:.{p}f} ms"
            for name, seconds in self._seconds_by_name.items()
        ]
        if include_total:
            parts.append(f"total={self.total_seconds() * 1000.0:.{p}f} ms")
        return " | ".join(parts)


def format_timing_table(
    timings_seconds: dict[str, float] | None,
    *,
    order: list[str] | None = None,
    precision: int = 2,
    total_seconds: float | None = None,
    title: str | None = None,
) -> str:
    """
    Tabla de timings en ms.

    - `order` fija el orden preferente y luego añade el resto.
    - `total_seconds` permite forzar total sin sumar filas solapadas.
    """
    t = timings_seconds or {}
    p = max(0, int(precision))

    names = []
    seen = set()

    for n in (order or []):
        key = str(n)
        if key in t and key not in seen:
            names.append(key)
            seen.add(key)

    for n in t.keys():
        key = str(n)
        if key not in seen:
            names.append(key)
            seen.add(key)

    rows: list[tuple[str, str]] = []
    for name in names:
        sec = t.get(name, None)
        if sec is None:
            continue
        rows.append((name, f"{float(sec) * 1000.0:.{p}f}"))

    if total_seconds is not None:
        rows.append(("total", f"{float(total_seconds) * 1000.0:.{p}f}"))

    if not rows:
        return ""

    hdr_name = "block"
    hdr_ms = "ms"
    w_name = max(len(hdr_name), max(len(name) for name, _ in rows))
    w_ms = max(len(hdr_ms), max(len(ms) for _, ms in rows))

    lines = []
    if title:
        lines.append(str(title))
    lines.append(f"{hdr_name:<{w_name}}  {hdr_ms:>{w_ms}}")
    lines.append(f"{'-' * w_name}  {'-' * w_ms}")

    for name, ms in rows:
        lines.append(f"{name:<{w_name}}  {ms:>{w_ms}}")

    return "\n".join(lines)


def _ordered_timing_names(timings_seconds: dict[str, float] | None, order: list[str] | None = None) -> list[str]:
    t = timings_seconds or {}
    names: list[str] = []
    seen: set[str] = set()
    for n in (order or []):
        key = str(n)
        if key in t and key not in seen:
            names.append(key)
            seen.add(key)
    for n in t.keys():
        key = str(n)
        if key not in seen:
            names.append(key)
            seen.add(key)
    return names


def _build_timing_tree(
    timings_seconds: dict[str, float] | None,
    *,
    order: list[str] | None = None,
):
    t = timings_seconds or {}
    order_pos = {str(name): idx for idx, name in enumerate(order or [])}
    next_insert = 0

    root = {
        "full_name": "",
        "label": "",
        "seconds": None,
        "children": OrderedDict(),
        "insert_idx": -1,
    }

    for full_name, seconds in t.items():
        parts = [part for part in str(full_name).split("/") if part]
        if not parts:
            continue

        cur = root
        prefix_parts: list[str] = []
        for part in parts:
            prefix_parts.append(str(part))
            child = cur["children"].get(str(part), None)
            if child is None:
                child = {
                    "full_name": "/".join(prefix_parts),
                    "label": str(part),
                    "seconds": None,
                    "children": OrderedDict(),
                    "insert_idx": int(next_insert),
                }
                next_insert += 1
                cur["children"][str(part)] = child
            cur = child
        cur["seconds"] = float(seconds)

    rank_cache: dict[str, tuple[float, int]] = {}

    def subtree_order_key(node) -> tuple[float, int]:
        full_name = str(node.get("full_name", "") or "")
        cached = rank_cache.get(full_name, None)
        if cached is not None:
            return cached

        rank = float(order_pos.get(full_name, float("inf")))
        insert_idx = int(node.get("insert_idx", 0))
        for child in node.get("children", {}).values():
            child_rank, child_insert = subtree_order_key(child)
            if child_rank < rank:
                rank = float(child_rank)
            if child_rank == rank:
                insert_idx = min(insert_idx, int(child_insert))

        out = (float(rank), int(insert_idx))
        rank_cache[full_name] = out
        return out

    return root, subtree_order_key


def _append_timing_tree_rows(
    rows: list[tuple[str, str]],
    node,
    *,
    depth: int,
    precision: int,
    indent: str,
    subtree_order_key,
) -> None:
    seconds = node.get("seconds", None)
    if seconds is not None:
        label = f"{indent * depth}{str(node.get('label', ''))}"
        rows.append((label, f"{float(seconds) * 1000.0:.{precision}f}"))
        child_depth = depth + 1
    else:
        child_depth = depth

    children = list(node.get("children", {}).values())
    children.sort(key=subtree_order_key)

    child_seconds_sum = 0.0
    for child in children:
        child_seconds = child.get("seconds", None)
        if child_seconds is not None:
            child_seconds_sum += float(child_seconds)

    for child in children:
        _append_timing_tree_rows(
            rows,
            child,
            depth=child_depth,
            precision=precision,
            indent=indent,
            subtree_order_key=subtree_order_key,
        )

    if seconds is not None and children:
        other_seconds = float(seconds) - float(child_seconds_sum)
        if other_seconds > 5e-7:
            rows.append((f"{indent * child_depth}other", f"{other_seconds * 1000.0:.{precision}f}"))


def format_timing_tree_table(
    stage_seconds: dict[str, float] | None,
    *,
    details_by_stage: dict[str, dict[str, float]] | None = None,
    stage_order: list[str] | None = None,
    detail_order_by_stage: dict[str, list[str]] | None = None,
    precision: int = 2,
    total_seconds: float | None = None,
    title: str | None = None,
    indent: str = "  ",
) -> str:
    """
    Tabla jerárquica de timings en ms.

    - Mantiene `total_seconds` separado para evitar doble conteo.
    - Muestra sub-bloques indentados bajo su etapa.
    - Soporta claves jerárquicas con `/` para anidar varios niveles.
    """
    stages = stage_seconds or {}
    details_by_stage = details_by_stage or {}
    detail_order_by_stage = detail_order_by_stage or {}
    p = max(0, int(precision))

    stage_names = _ordered_timing_names(stages, stage_order)

    rows: list[tuple[str, str]] = []
    for st in stage_names:
        sec = stages.get(st, None)
        if sec is None:
            continue
        rows.append((str(st), f"{float(sec) * 1000.0:.{p}f}"))

        sub = details_by_stage.get(st, None)
        if not isinstance(sub, dict) or not sub:
            continue

        tree_root, subtree_order_key = _build_timing_tree(
            sub,
            order=detail_order_by_stage.get(st, []) or [],
        )
        children = list(tree_root.get("children", {}).values())
        children.sort(key=subtree_order_key)
        child_seconds_sum = 0.0
        for child in children:
            child_seconds = child.get("seconds", None)
            if child_seconds is not None:
                child_seconds_sum += float(child_seconds)
        for child in children:
            _append_timing_tree_rows(
                rows,
                child,
                depth=1,
                precision=p,
                indent=indent,
                subtree_order_key=subtree_order_key,
            )
        other_seconds = float(sec) - float(child_seconds_sum)
        if children and other_seconds > 5e-7:
            rows.append((f"{indent}other", f"{other_seconds * 1000.0:.{p}f}"))

    if total_seconds is not None:
        rows.append(("total", f"{float(total_seconds) * 1000.0:.{p}f}"))

    if not rows:
        return ""

    hdr_name = "block"
    hdr_ms = "ms"
    w_name = max(len(hdr_name), max(len(name) for name, _ in rows))
    w_ms = max(len(hdr_ms), max(len(ms) for _, ms in rows))

    lines = []
    if title:
        lines.append(str(title))
    lines.append(f"{hdr_name:<{w_name}}  {hdr_ms:>{w_ms}}")
    lines.append(f"{'-' * w_name}  {'-' * w_ms}")

    for name, ms in rows:
        lines.append(f"{name:<{w_name}}  {ms:>{w_ms}}")

    return "\n".join(lines)


def format_timing_tree_line(
    stage_seconds: dict[str, float] | None,
    *,
    details_by_stage: dict[str, dict[str, float]] | None = None,
    stage_order: list[str] | None = None,
    detail_order_by_stage: dict[str, list[str]] | None = None,
    precision: int = 2,
    total_seconds: float | None = None,
) -> str:
    """
    Línea jerárquica (compacta) en ms.
    """
    stages = stage_seconds or {}
    details_by_stage = details_by_stage or {}
    detail_order_by_stage = detail_order_by_stage or {}
    p = max(0, int(precision))

    stage_names = _ordered_timing_names(stages, stage_order)

    parts: list[str] = []
    for st in stage_names:
        sec = stages.get(st, None)
        if sec is None:
            continue
        base = f"{st}={float(sec) * 1000.0:.{p}f} ms"

        sub = details_by_stage.get(st, None)
        if isinstance(sub, dict) and sub:
            tree_root, subtree_order_key = _build_timing_tree(
                sub,
                order=detail_order_by_stage.get(st, []) or [],
            )

            def render_line_entries(node) -> list[str]:
                out: list[str] = []
                seconds = node.get("seconds", None)
                if seconds is not None:
                    out.append(f"{str(node.get('full_name', ''))}={float(seconds) * 1000.0:.{p}f}")
                children = list(node.get("children", {}).values())
                children.sort(key=subtree_order_key)
                child_seconds_sum = 0.0
                for child in children:
                    child_seconds = child.get("seconds", None)
                    if child_seconds is not None:
                        child_seconds_sum += float(child_seconds)
                for child in children:
                    out.extend(render_line_entries(child))
                if seconds is not None and children:
                    other_seconds = float(seconds) - float(child_seconds_sum)
                    if other_seconds > 5e-7:
                        full_name = str(node.get("full_name", "") or "")
                        other_name = f"{full_name}/other" if full_name else "other"
                        out.append(f"{other_name}={other_seconds * 1000.0:.{p}f}")
                return out

            sub_parts: list[str] = []
            children = list(tree_root.get("children", {}).values())
            children.sort(key=subtree_order_key)
            child_seconds_sum = 0.0
            for child in children:
                child_seconds = child.get("seconds", None)
                if child_seconds is not None:
                    child_seconds_sum += float(child_seconds)
            for child in children:
                sub_parts.extend(render_line_entries(child))
            other_seconds = float(sec) - float(child_seconds_sum)
            if children and other_seconds > 5e-7:
                sub_parts.append(f"{st}/other={other_seconds * 1000.0:.{p}f}")
            if sub_parts:
                base = f"{base} ({', '.join(sub_parts)} ms)"

        parts.append(base)

    if total_seconds is not None:
        parts.append(f"total={float(total_seconds) * 1000.0:.{p}f} ms")

    return " | ".join(parts)
