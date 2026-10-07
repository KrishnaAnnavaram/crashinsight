"""Explanations on held-out rows.

The core method is permutation importance on the raw features. It shuffles one raw column (for
example `body_group`) and measures the loss of KA PR-AUC, so one-hot columns of one variable count
together. SHAP values are an optional extra (`pip install -e ".[explain]"`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score

from .evaluate import ka_score, ka_truth


def _ka_ap(estimator, X, y) -> float:
    proba = estimator.predict_proba(X)
    return float(average_precision_score(ka_truth(np.asarray(y)), ka_score(proba, np.asarray(estimator.classes_))))


def permutation_table(pipeline, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, seed: int = 42) -> pd.DataFrame:
    result = permutation_importance(pipeline, X, y, scoring=_ka_ap, n_repeats=n_repeats, random_state=seed)
    table = pd.DataFrame({
        "feature": X.columns,
        "ka_pr_auc_drop": result.importances_mean,
        "std": result.importances_std,
    })
    return table.sort_values("ka_pr_auc_drop", ascending=False).reset_index(drop=True)


def shap_values(pipeline, X: pd.DataFrame, max_rows: int = 500):
    """Return SHAP values for the model step. Needs the optional `shap` package."""
    try:
        import shap  # noqa: PLC0415 - optional heavy dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError('SHAP needs the optional extra: pip install -e ".[explain]"') from exc
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    sample = X.sample(min(max_rows, len(X)), random_state=0)
    matrix = prep.transform(sample)
    explainer = shap.Explainer(model.predict_proba, matrix)
    return explainer(matrix), prep.get_feature_names_out()
