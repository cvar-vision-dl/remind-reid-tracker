# utils/config_utils.py

def cfg_get(cfg, key, default=None):
    """
    Safe getter for nested dict configs using dot notation.

    Example:
        v = cfg_get(config, "memory.background.max_near", 20)
    """
    if cfg is None:
        return default

    cur = cfg
    for part in str(key).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def cfg_bool(cfg, key, default=False) -> bool:
    value = cfg_get(cfg, key, default)
    return bool(default if value is None else value)


def cfg_int(cfg, key, default=0, *, min_value=None, max_value=None) -> int:
    value = cfg_get(cfg, key, default)
    try:
        out = int(value)
    except Exception:
        out = int(default)
    if min_value is not None:
        out = max(int(min_value), int(out))
    if max_value is not None:
        out = min(int(max_value), int(out))
    return int(out)


def cfg_float(cfg, key, default=0.0, *, min_value=None, max_value=None) -> float:
    value = cfg_get(cfg, key, default)
    try:
        out = float(value)
    except Exception:
        out = float(default)
    if min_value is not None:
        out = max(float(min_value), float(out))
    if max_value is not None:
        out = min(float(max_value), float(out))
    return float(out)


def cfg_str(cfg, key, default="") -> str:
    value = cfg_get(cfg, key, default)
    if value is None:
        return str(default)
    return str(value)


def bg_partials_enabled(cfg) -> bool:
    """
    Central flag to fully enable/disable the `bg_partials` branch.

    Requires:
    - `association.similarity.background_partials.enabled` if defined
    - `bg_local.prototypes.enabled` so observed prototypes exist
    """
    branch_enabled = bool(cfg_get(cfg, "association.similarity.background_partials.enabled", True))
    proto_enabled = bool(cfg_get(cfg, "bg_local.prototypes.enabled", True))
    return bool(branch_enabled and proto_enabled)
