"""Metrics for an imbalanced severity target.

Accuracy alone hides the rare classes. This module reports macro-F1, balanced accuracy, recall for
each class, PR-AUC for the serious-or-fatal (KA) group, the Brier score and a calibration table.
Probability columns are selected by class label, never by a fixed column position.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

SERIOUS_LABELS = ("K", "A")


def ka_score(proba: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """Probability of a serious or fatal injury, from the columns that belong to KA labels."""
    classes = [str(c) for c in classes]
    if set(classes) <= {"0", "1"}:
        return proba[:, classes.index("1")]
    cols = [classes.index(c) for c in SERIOUS_LABELS if c in classes]
    if not cols:
        raise ValueError(f"no KA class in {classes}")
    return proba[:, cols].sum(axis=1)


def ka_truth(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).astype(str)
    if set(np.unique(y)) <= {"0", "1"}:
        return (y == "1").astype(int)
    return np.isin(y, SERIOUS_LABELS).astype(int)


def calibration_table(truth: np.ndarray, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(score, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int(mask.sum()),
                         "mean_score": float(score[mask].mean()), "observed_rate": float(truth[mask].mean())})
    return pd.DataFrame(rows)


def expected_calibration_error(truth: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    table = calibration_table(truth, score, bins)
    if table.empty:
        return float("nan")
    return float(np.sum(table["n"] * (table["mean_score"] - table["observed_rate"]).abs()) / table["n"].sum())


def classification_metrics(y_true, proba: np.ndarray, classes: np.ndarray) -> dict:
    y = np.asarray(y_true).astype(str)
    labels = np.asarray([str(c) for c in classes])
    pred = labels[np.argmax(proba, axis=1)]
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "recall_by_class": {
            str(c): float(r) for c, r in zip(labels, recall_score(y, pred, labels=labels, average=None, zero_division=0))
        },
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "labels": labels.tolist(),
    }
    truth = ka_truth(y)
    score = ka_score(proba, labels)
    if 0 < truth.sum() < len(truth):
        out["ka_pr_auc"] = float(average_precision_score(truth, score))
        out["ka_roc_auc"] = float(roc_auc_score(truth, score))
        out["ka_prevalence"] = float(truth.mean())
    out["ka_brier"] = float(brier_score_loss(truth, np.clip(score, 0, 1)))
    out["ka_ece"] = expected_calibration_error(truth, np.clip(score, 0, 1))
    if len(labels) > 2 and len(np.unique(y)) == len(labels):
        order = np.argsort(labels)  # roc_auc_score needs sorted labels with matching columns
        out["macro_roc_auc_ovr"] = float(
            roc_auc_score(y, proba[:, order], multi_class="ovr", labels=labels[order], average="macro"))
    return out


def grouped_bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    proba: np.ndarray,
    groups,
    n_boot: int = 200,
    seed: int = 42,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile CI for a metric. It resamples whole crashes, not single vehicles."""
    y = np.asarray(y_true)
    g = np.asarray(groups)
    uniq, inverse = np.unique(g, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        try:
            values.append(metric(y[idx], proba[idx]))
        except ValueError:
            continue
    if not values:
        return float("nan"), float("nan")
    alpha = (1 - level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return float(low), float(high)


def macro_f1_metric(classes: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
    labels = np.asarray([str(c) for c in classes])

    def metric(y, proba):
        pred = labels[np.argmax(proba, axis=1)]
        return float(f1_score(np.asarray(y).astype(str), pred, labels=labels, average="macro", zero_division=0))

    return metric


def ka_pr_auc_metric(classes: np.ndarray) -> Callable[[np.ndarray, np.ndarray], float]:
    labels = np.asarray([str(c) for c in classes])

    def metric(y, proba):
        truth = ka_truth(y)
        if truth.min() == truth.max():
            raise ValueError("one class only")
        return float(average_precision_score(truth, ka_score(proba, labels)))

    return metric
