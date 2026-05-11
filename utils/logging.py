# utils/logging.py

from __future__ import annotations

import os
import sys
from datetime import datetime


def default_run_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class TeeStdout:
    """
    Duplicate everything printed to console into a file without changing prints.
    Respeta exactamente el formato (tablas, saltos, etc.).

    Uso:
      tee = TeeStdout(path)
      tee.install()
      ...
      tee.close()
    """

    def __init__(self, file_path: str, mode: str = "w", encoding: str = "utf-8"):
        self.file_path = str(file_path)
        self.mode = str(mode)
        self.encoding = str(encoding)

        self._orig_stdout = None
        self._f = None

    def install(self) -> None:
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self._orig_stdout = sys.stdout
        self._f = open(self.file_path, self.mode, encoding=self.encoding, buffering=1)
        sys.stdout = self

    def write(self, s: str) -> int:
        n = 0
        if self._orig_stdout is not None:
            n = self._orig_stdout.write(s)
            self._orig_stdout.flush()
        if self._f is not None:
            self._f.write(s)
            self._f.flush()
        return n

    def flush(self) -> None:
        if self._orig_stdout is not None:
            self._orig_stdout.flush()
        if self._f is not None:
            self._f.flush()

    def close(self) -> None:
        try:
            sys.stdout = self._orig_stdout if self._orig_stdout is not None else sys.__stdout__
        except Exception:
            pass
        if self._f is not None:
            try:
                self._f.close()
            except Exception:
                pass
        self._f = None
        self._orig_stdout = None


def default_run_log_path(output_dir: str, prefix: str = "console") -> str:
    ts = default_run_timestamp()
    return os.path.join(str(output_dir), "logs", f"{prefix}_{ts}.txt")


def default_run_artifact_dir(output_dir: str, group: str, prefix: str = "run", timestamp: str | None = None) -> str:
    ts = str(timestamp or default_run_timestamp())
    return os.path.join(str(output_dir), str(group), f"{prefix}_{ts}")
