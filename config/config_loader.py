from __future__ import annotations

from pathlib import Path
import yaml


class Config:
    """
    Central system configuration container.
    """

    def __init__(
        self,
        default_config_path: str | Path,
        override_config_path: str | Path | None = None,
    ):
        base = self.load_yaml(default_config_path)

        if override_config_path is not None:
            override = self.load_yaml(override_config_path)
            base = self.merge_dicts(base, override)

        self._data = base

    def load_yaml(self, path: str | Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {path}") from e

        if data is None:
            return {}

        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a dict: {path}")

        return data

    def merge_dicts(self, base: dict, override: dict) -> dict:
        out = dict(base)
        for k, v in override.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = self.merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    def get(self, key, default=None):
        return self._data.get(key, default)

    def to_dict(self) -> dict:
        return dict(self._data)
