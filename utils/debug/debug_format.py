# utils/debug_format.py

def fmt(x, nd=2):
    """Format numeric values to nd decimals; None -> 'NA'. Keep non-castable values."""
    if x is None:
        return "NA"
    try:
        return round(float(x), nd)
    except Exception:
        return x


def fmt_ws(v, nd=2, sep="|"):
    """Formatea valores work/stable como 'work|stable' usando fmt()."""
    if v is None:
        return "NA"
    if isinstance(v, dict):
        return f"{fmt(v.get('work'), nd)}{sep}{fmt(v.get('stable'), nd)}"
    return fmt(v, nd)


def safe_float(x, default: float = 0.0) -> float:
    """Robustly cast to float; for {work, stable} dicts use max(work, stable)."""
    if x is None:
        return None if default is None else float(default)
    if isinstance(x, dict):
        c = x.get("combined", None)
        if c is not None:
            try:
                return float(c)
            except Exception:
                pass
        w = x.get("work", None)
        s = x.get("stable", None)
        vals = []
        if w is not None:
            try:
                vals.append(float(w))
            except Exception:
                pass
        if s is not None:
            try:
                vals.append(float(s))
            except Exception:
                pass
        if vals:
            return float(max(vals))
        return None if default is None else float(default)
    try:
        return float(x)
    except Exception:
        return None if default is None else float(default)
