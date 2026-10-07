"""Crash-grouped, outcome-stratified splits.

Vehicles in the same crash share the crash conditions and often the outcome. All vehicles of one
crash (`CASENUM`) therefore go to the same side of every split.
"""
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_folds(
    y: pd.Series, groups: pd.Series, n_splits: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from splitter.split(np.zeros(len(y)), np.asarray(y), np.asarray(groups))


def grouped_holdout(
    frame: pd.DataFrame, y: pd.Series, test_fraction: float = 0.2, seed: int = 42, group: str = "CASENUM"
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_index, test_index). The test part is one fold of a stratified group k-fold."""
    if not 0.05 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.05 and 0.5")
    n_splits = max(2, int(round(1 / test_fraction)))
    train_idx, test_idx = next(grouped_folds(y, frame[group], n_splits=n_splits, seed=seed))
    return train_idx, test_idx


def assert_disjoint_groups(groups: pd.Series, train_idx: np.ndarray, test_idx: np.ndarray) -> None:
    overlap = set(np.asarray(groups)[train_idx]) & set(np.asarray(groups)[test_idx])
    if overlap:
        raise AssertionError(f"{len(overlap)} crashes are on both sides of the split")
