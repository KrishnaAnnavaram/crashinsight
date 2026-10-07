"""Runtime settings. Every value comes from an environment variable with a safe default."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    output_dir: Path
    seed: int
    crash_year: int
    n_splits: int

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.environ.get("CRASHINSIGHT_DATA_DIR", "").strip() or "data")
        output_dir = Path(os.environ.get("CRASHINSIGHT_OUTPUT_DIR", "").strip() or "outputs")
        n_splits = _int("CRASHINSIGHT_N_SPLITS", 5)
        if n_splits < 2:
            raise ValueError("CRASHINSIGHT_N_SPLITS must be 2 or more")
        return cls(
            data_dir=data_dir,
            output_dir=output_dir,
            seed=_int("CRASHINSIGHT_SEED", 42),
            crash_year=_int("CRASHINSIGHT_CRASH_YEAR", 2015),
            n_splits=n_splits,
        )
