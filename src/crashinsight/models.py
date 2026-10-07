"""Model pipelines. Preprocessing lives inside the pipeline, so it is fit on training rows only.

- Nominal codes are one-hot encoded. They are never scaled and never used as numbers.
- Numeric features get a median imputer with a missing indicator, then a scaler.
- Binary flags get 0 for missing plus a missing indicator.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .dataset import KABCO_ORDER, FeatureSpec

MODEL_NAMES = ("majority", "logreg", "forest", "boosting", "ordinal")


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    binary = SimpleImputer(strategy="constant", fill_value=0, add_indicator=True)
    nominal = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("numeric", numeric, spec.numeric), ("binary", binary, spec.binary), ("nominal", nominal, spec.nominal)],
        sparse_threshold=0,
        verbose_feature_names_out=True,
    )


class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    """Frank and Hall ordinal model: one binary model for each threshold P(y > k).

    It keeps the order O < C < B < A < K. `predict_proba` returns columns in `classes_` order.
    """

    def __init__(self, base=None, order=tuple(KABCO_ORDER)):
        self.base = base
        self.order = order

    def fit(self, X, y, sample_weight=None):
        order = list(self.order)
        y = np.asarray(y)
        unknown = set(np.unique(y)) - set(order)
        if unknown:
            raise ValueError(f"labels {sorted(unknown)} are not in the order {order}")
        rank = np.array([order.index(v) for v in y])
        self.classes_ = np.array(order)
        base = self.base if self.base is not None else LogisticRegression(max_iter=2000, class_weight="balanced")
        self.models_ = []
        for k in range(len(order) - 1):
            target = (rank > k).astype(int)
            if target.min() == target.max():
                self.models_.append(float(target[0]))
                continue
            model = clone(base)
            model.fit(X, target, sample_weight=sample_weight) if sample_weight is not None else model.fit(X, target)
            self.models_.append(model)
        return self

    def predict_proba(self, X):
        n = X.shape[0]
        greater = np.column_stack([
            np.full(n, m) if isinstance(m, float) else m.predict_proba(X)[:, 1] for m in self.models_
        ])
        # Make the threshold probabilities non-increasing, then take differences.
        greater = np.minimum.accumulate(greater, axis=1)
        upper = np.column_stack([np.ones(n), greater])
        lower = np.column_stack([greater, np.zeros(n)])
        proba = np.clip(upper - lower, 0.0, 1.0)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_estimator(name: str, seed: int = 42):
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight="balanced")
    if name == "forest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=seed
        )
    if name == "boosting":
        # A fixed number of iterations. Early stopping would use a random row split that ignores crashes.
        return HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
            class_weight="balanced", random_state=seed,
        )
    if name == "ordinal":
        return OrdinalClassifier()
    raise ValueError(f"unknown model {name!r}. Use one of {', '.join(MODEL_NAMES)}")


def build_pipeline(spec: FeatureSpec, model: str, seed: int = 42) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(spec)), ("model", build_estimator(model, seed))])
