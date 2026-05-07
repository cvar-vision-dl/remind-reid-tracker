from __future__ import annotations

import shutil
import sys

import pandas as pd


def clip_text(v, max_chars: int) -> str:
    if v is None:
        return ""
    s = str(v)
    m = max(4, int(max_chars))
    if len(s) <= m:
        return s
    return s[: m - 3] + "..."


def compact_columns(df: pd.DataFrame, limits: dict[str, int]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col, max_chars in (limits or {}).items():
        if col in out.columns:
            out[col] = out[col].map(lambda x: clip_text(x, max_chars))
    return out


def clamp_dataframe_rows(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Recorta df a max_rows si aplica."""
    mr = int(max_rows)
    if mr > 0 and df is not None and not df.empty and len(df) > mr:
        return df.head(mr)
    return df


def sort_dataframe_if_possible(df: pd.DataFrame, sort_by: list) -> pd.DataFrame:
    """Ordena df por columnas existentes."""
    if df is None or df.empty:
        return df
    cols = [c for c in (sort_by or []) if c in df.columns]
    if not cols:
        return df
    return df.sort_values(by=cols, kind="stable")


def print_dataframe(df: pd.DataFrame, cols: list[str], *, index: bool = False, col_space: int = 2) -> None:
    """Imprime DataFrame con columnas existentes."""
    if df is None:
        return
    cols = [c for c in (cols or []) if c in df.columns]
    if not cols:
        print(df.to_string(index=index))
        return
    print(df[cols].to_string(index=index, col_space=col_space))


def _wrap_tokens(text: str, width: int, *, sep: str = ",") -> list[str]:
    """
    Wrap greedy por tokens separados por `sep` (p.ej. "A,B,C") para evitar que el terminal
    parta una fila a mitad.
    """
    s = str(text or "")
    w = max(8, int(width))
    if len(s) <= w:
        return [s]

    toks = [t for t in s.split(sep) if t != ""]
    if not toks:
        return [s[i : i + w] for i in range(0, len(s), w)]

    lines: list[str] = []
    cur = ""
    for t in toks:
        tok = str(t)
        candidate = tok if not cur else f"{cur}{sep}{tok}"
        if len(candidate) <= w:
            cur = candidate
            continue

        if cur:
            lines.append(cur)
            cur = tok
        else:
            lines.extend([tok[i : i + w] for i in range(0, len(tok), w)])
            cur = ""

        if cur and len(cur) > w:
            lines.extend([cur[i : i + w] for i in range(0, len(cur), w)])
            cur = ""

    if cur:
        lines.append(cur)
    return lines if lines else [s[:w]]


def render_table_wrap_column(
    df: pd.DataFrame,
    cols: list[str],
    *,
    wrap_col: str,
    table_width: int = 0,
    col_space: int = 1,
    headers: dict[str, str] | None = None,
    wrap_sep: str = ",",
) -> str:
    """
    Renderiza una tabla estilo `to_string`, pero envolviendo `wrap_col` en múltiples líneas
    para que el terminal no rompa filas (sin perder información).
    """
    if df is None or df.empty:
        return ""

    cols = [c for c in (cols or []) if c in df.columns]
    if not cols:
        return df.to_string(index=False)
    if wrap_col not in cols:
        return df[cols].to_string(index=False, col_space=max(0, int(col_space)))

    headers = headers or {}
    sp = " " * max(1, int(col_space))

    if int(table_width) <= 0:
        table_width = int(shutil.get_terminal_size((140, 20)).columns)
    table_width = max(60, int(table_width))

    # Materializa strings por celda
    rows: list[dict[str, str]] = []
    for _, r in df[cols].iterrows():
        pack = {}
        for c in cols:
            v = r.get(c, "")
            pack[str(c)] = "" if v is None else str(v)
        rows.append(pack)

    labels = {c: str(headers.get(c, c)) for c in cols}
    fixed_cols = [c for c in cols if c != wrap_col]

    fixed_w: dict[str, int] = {}
    for c in fixed_cols:
        w = len(labels[c])
        for rr in rows:
            w = max(w, len(rr.get(c, "")))
        fixed_w[c] = int(w)

    base = 0
    if fixed_cols:
        base = sum(fixed_w[c] for c in fixed_cols) + (len(fixed_cols)) * len(sp) + len(sp)
    wrap_w = max(10, int(table_width) - int(base))
    wrap_w = max(len(labels[wrap_col]), int(wrap_w))

    # Header + rule
    hdr_parts = [f"{labels[c]:<{fixed_w[c]}}" for c in fixed_cols] + [f"{labels[wrap_col]:<{wrap_w}}"]
    rule_parts = [("-" * fixed_w[c]) for c in fixed_cols] + [("-" * wrap_w)]

    lines = [sp.join(hdr_parts), sp.join(rule_parts)]

    for rr in rows:
        wrapped = _wrap_tokens(rr.get(wrap_col, ""), wrap_w, sep=wrap_sep)
        for j, chunk in enumerate(wrapped):
            if j == 0:
                parts = [f"{rr.get(c, ''):<{fixed_w[c]}}" for c in fixed_cols]
            else:
                parts = [(" " * fixed_w[c]) for c in fixed_cols]
            parts.append(f"{chunk:<{wrap_w}}")
            lines.append(sp.join(parts))

    return "\n".join(lines)


def _table_width_for_render(table_width: int = 0) -> int:
    if int(table_width) <= 0:
        table_width = int(shutil.get_terminal_size((140, 20)).columns)
    return max(60, int(table_width))


def _materialize_table_rows(df: pd.DataFrame, cols: list[str]) -> tuple[list[dict[str, str]], dict[str, str]]:
    rows: list[dict[str, str]] = []
    for _, r in df[cols].iterrows():
        pack = {}
        for c in cols:
            v = r.get(c, "")
            pack[str(c)] = "" if v is None else str(v)
        rows.append(pack)
    headers = {str(c): str(c) for c in cols}
    return rows, headers


def _column_widths(rows: list[dict[str, str]], cols: list[str], headers: dict[str, str] | None = None) -> dict[str, int]:
    hdrs = headers or {}
    out: dict[str, int] = {}
    for c in cols:
        w = len(str(hdrs.get(c, c)))
        for rr in rows:
            w = max(w, len(rr.get(c, "")))
        out[str(c)] = int(w)
    return out


def _estimate_chunk_width(cols: list[str], widths: dict[str, int], col_space: int) -> int:
    if not cols:
        return 0
    sp = max(1, int(col_space))
    return int(sum(int(widths.get(c, len(str(c)))) for c in cols) + max(0, len(cols) - 1) * sp)


def split_columns_for_width(
    df: pd.DataFrame,
    cols: list[str],
    *,
    pinned_cols: list[str] | None = None,
    table_width: int = 0,
    col_space: int = 1,
    headers: dict[str, str] | None = None,
) -> list[list[str]]:
    cols = [c for c in (cols or []) if c in df.columns]
    if not cols:
        return []

    pinned = [c for c in (pinned_cols or []) if c in cols]
    pinned = list(dict.fromkeys(pinned))
    others = [c for c in cols if c not in pinned]

    rows, default_headers = _materialize_table_rows(df, cols)
    hdrs = dict(default_headers)
    hdrs.update(headers or {})
    widths = _column_widths(rows, cols, headers=hdrs)

    max_width = _table_width_for_render(table_width)
    pinned_width = _estimate_chunk_width(pinned, widths, col_space)

    if not others:
        return [list(pinned)] if pinned else [list(cols)]

    chunks: list[list[str]] = []
    cur: list[str] = []
    for c in others:
        trial = list(pinned) + list(cur) + [str(c)]
        trial_width = _estimate_chunk_width(trial, widths, col_space)
        if cur and trial_width > int(max_width):
            chunks.append(list(pinned) + list(cur))
            cur = [str(c)]
            continue
        if not cur and pinned and pinned_width > 0 and trial_width > int(max_width):
            chunks.append(list(pinned) + [str(c)])
            cur = []
            continue
        cur.append(str(c))

    if cur:
        chunks.append(list(pinned) + list(cur))

    if not chunks:
        chunks.append(list(cols))
    return chunks


def render_table_auto(
    df: pd.DataFrame,
    cols: list[str],
    *,
    pinned_cols: list[str] | None = None,
    wrap_col: str | None = None,
    table_width: int = 0,
    col_space: int = 1,
    headers: dict[str, str] | None = None,
    wrap_sep: str = ",",
) -> str:
    if df is None or df.empty:
        return ""

    cols = [c for c in (cols or []) if c in df.columns]
    if not cols:
        return df.to_string(index=False)

    chunks = split_columns_for_width(
        df,
        cols,
        pinned_cols=pinned_cols,
        table_width=table_width,
        col_space=col_space,
        headers=headers,
    )

    rendered: list[str] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        if total > 1:
            rendered.append(f"[cols {idx}/{total}]")
        if wrap_col is not None and wrap_col in chunk:
            rendered.append(
                render_table_wrap_column(
                    df,
                    chunk,
                    wrap_col=wrap_col,
                    table_width=table_width,
                    col_space=col_space,
                    headers=headers,
                    wrap_sep=wrap_sep,
                )
            )
        else:
            rendered.append(df[chunk].to_string(index=False, col_space=max(0, int(col_space))))
    return "\n\n".join(x for x in rendered if str(x))


def _tee_streams():
    out = sys.stdout
    console = getattr(out, "_orig_stdout", None)
    file_obj = getattr(out, "_f", None)
    if console is not None and file_obj is not None:
        return console, file_obj
    return None, None


def _file_table_string(
    df: pd.DataFrame,
    cols: list[str],
    *,
    col_space: int = 1,
    headers: dict[str, str] | None = None,
) -> str:
    if df is None or df.empty:
        return ""
    cols = [c for c in (cols or []) if c in df.columns]
    if not cols:
        return df.to_string(index=False)
    out = df[cols].copy()
    if headers:
        rename_map = {c: str(headers.get(c, c)) for c in cols}
        out = out.rename(columns=rename_map)
    return out.to_string(index=False, col_space=max(0, int(col_space)))


def print_dual_render(console_text: str, *, file_text: str | None = None) -> None:
    console_text = str(console_text or "")
    file_text = console_text if file_text is None else str(file_text or "")

    console, file_obj = _tee_streams()
    if console is None or file_obj is None:
        print(console_text)
        return

    console.write(console_text)
    if not console_text.endswith("\n"):
        console.write("\n")
    console.flush()

    file_obj.write(file_text)
    if not file_text.endswith("\n"):
        file_obj.write("\n")
    file_obj.flush()


def print_table_auto(
    df: pd.DataFrame,
    cols: list[str],
    *,
    pinned_cols: list[str] | None = None,
    wrap_col: str | None = None,
    table_width: int = 0,
    col_space: int = 1,
    headers: dict[str, str] | None = None,
    wrap_sep: str = ",",
) -> None:
    console_text = render_table_auto(
        df,
        cols,
        pinned_cols=pinned_cols,
        wrap_col=wrap_col,
        table_width=table_width,
        col_space=col_space,
        headers=headers,
        wrap_sep=wrap_sep,
    )
    file_text = _file_table_string(
        df,
        cols,
        col_space=col_space,
        headers=headers,
    )
    print_dual_render(console_text, file_text=file_text)


def print_table_wrap_column(
    df: pd.DataFrame,
    cols: list[str],
    *,
    wrap_col: str,
    table_width: int = 0,
    col_space: int = 1,
    headers: dict[str, str] | None = None,
    wrap_sep: str = ",",
) -> None:
    console_text = render_table_wrap_column(
        df,
        cols,
        wrap_col=wrap_col,
        table_width=table_width,
        col_space=col_space,
        headers=headers,
        wrap_sep=wrap_sep,
    )
    file_text = _file_table_string(
        df,
        cols,
        col_space=col_space,
    )
    print_dual_render(console_text, file_text=file_text)
