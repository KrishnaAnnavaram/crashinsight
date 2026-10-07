"""Seeded training: crash-grouped cross-validation, a grouped hold-out test and a model card."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import __version__
from .dataset import FeatureSpec, target_values
from .evaluate import classification_metrics, grouped_bootstrap_ci, ka_pr_auc_metric, macro_f1_metric
from .models import build_pipeline
from .split import assert_disjoint_groups, grouped_folds, grouped_holdout

SUMMARY_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "ka_pr_auc", "ka_roc_auc", "ka_brier", "ka_ece")


@dataclass
class CVResult:
    model: str
    target: str
    n_splits: int
    folds: list[dict]
    oof: dict
    ci: dict = field(default_factory=dict)

    def mean(self, key: str) -> float:
        values = [f[key] for f in self.folds if key in f]
        return float(np.mean(values)) if values else float("nan")


@dataclass
class TrainResult:
    pipeline: object
    spec: FeatureSpec
    target: str
    model: str
    seed: int
    cv: CVResult
    test: dict
    baseline_test: dict
    n_train: int
    n_test: int
    test_ci: dict = field(default_factory=dict)


def _check(model: str, target: str) -> None:
    if model == "ordinal" and target != "kabco":
        raise ValueError("the ordinal model needs the 'kabco' target")


def cross_validate(
    frame: pd.DataFrame, spec: FeatureSpec, target: str, model: str, n_splits: int = 5, seed: int = 42,
    n_boot: int = 200,
) -> CVResult:
    _check(model, target)
    y = target_values(frame, target)
    X = frame[spec.all]
    folds = []
    oof = None
    classes = None
    for k, (tr, te) in enumerate(grouped_folds(y, frame["CASENUM"], n_splits=n_splits, seed=seed)):
        assert_disjoint_groups(frame["CASENUM"], tr, te)
        pipe = build_pipeline(spec, model, seed)
        pipe.fit(X.iloc[tr], y.iloc[tr])
        proba = pipe.predict_proba(X.iloc[te])
        if classes is None:
            classes = np.asarray(pipe.classes_)
            oof = np.zeros((len(frame), len(classes)))
        aligned = np.zeros((len(te), len(classes)))
        for j, c in enumerate(pipe.classes_):
            aligned[:, list(classes).index(c)] = proba[:, j]
        oof[te] = aligned
        metrics = classification_metrics(y.iloc[te], aligned, classes)
        metrics["fold"] = k
        folds.append(metrics)
    oof_metrics = classification_metrics(y, oof, classes)
    ci = {}
    if n_boot:
        ci["macro_f1"] = grouped_bootstrap_ci(macro_f1_metric(classes), y.to_numpy(), oof, frame["CASENUM"], n_boot, seed)
        ci["ka_pr_auc"] = grouped_bootstrap_ci(ka_pr_auc_metric(classes), y.to_numpy(), oof, frame["CASENUM"], n_boot, seed)
    return CVResult(model=model, target=target, n_splits=n_splits, folds=folds, oof=oof_metrics, ci=ci)


def train(
    frame: pd.DataFrame, spec: FeatureSpec, target: str = "ka", model: str = "boosting", seed: int = 42,
    n_splits: int = 5, test_fraction: float = 0.2, n_boot: int = 200,
) -> TrainResult:
    """Hold out whole crashes for the test, cross-validate on the rest, then fit the final model."""
    _check(model, target)
    y = target_values(frame, target)
    tr, te = grouped_holdout(frame, y, test_fraction=test_fraction, seed=seed)
    assert_disjoint_groups(frame["CASENUM"], tr, te)
    train_frame = frame.iloc[tr].reset_index(drop=True)
    test_frame = frame.iloc[te].reset_index(drop=True)
    cv = cross_validate(train_frame, spec, target, model, n_splits=n_splits, seed=seed, n_boot=n_boot)
    pipe = build_pipeline(spec, model, seed).fit(train_frame[spec.all], y.iloc[tr])
    y_test = target_values(test_frame, target)
    proba = pipe.predict_proba(test_frame[spec.all])
    test = classification_metrics(y_test, proba, pipe.classes_)
    test_ci = {}
    if n_boot:
        test_ci["ka_pr_auc"] = grouped_bootstrap_ci(
            ka_pr_auc_metric(pipe.classes_), y_test.to_numpy(), proba, test_frame["CASENUM"], n_boot, seed)
    base = build_pipeline(spec, "majority", seed).fit(train_frame[spec.all], y.iloc[tr])
    baseline_test = classification_metrics(y_test, base.predict_proba(test_frame[spec.all]), base.classes_)
    return TrainResult(
        pipeline=pipe, spec=spec, target=target, model=model, seed=seed, cv=cv, test=test,
        baseline_test=baseline_test, n_train=len(tr), n_test=len(te), test_ci=test_ci,
    )


def _summary(metrics: dict) -> dict:
    return {k: round(metrics[k], 4) for k in SUMMARY_KEYS if k in metrics}


def model_card(result: TrainResult, codebook_version: str, data_label: str) -> str:
    cv = result.cv
    lines = [
        f"# Model card: crashinsight {result.model} ({result.target})",
        "",
        f"- Package version: {__version__}. Date: {date.today().isoformat()}. Seed: {result.seed}.",
        f"- Data: {data_label}. Codebook: {codebook_version}.",
        f"- Target: {'KA (serious or fatal injury) vs. not KA' if result.target == 'ka' else 'KABCO (O, C, B, A, K)'}.",
        f"- Features ({len(result.spec.all)}): {', '.join(result.spec.all)}.",
        f"- Split: {result.n_train} training rows and {result.n_test} test rows. Whole crashes (CASENUM) stay on one side.",
        "",
        "## Intended use",
        "",
        "Research on the factors that relate to injury severity in police-reported crashes.",
        "It is not a tool for decisions about a person, an insurance claim or a medical triage.",
        "",
        "## Metrics",
        "",
        "| Metric | CV mean | Hold-out test | Majority baseline (test) |",
        "|---|---|---|---|",
    ]
    for key in SUMMARY_KEYS:
        if key in result.test:
            base = result.baseline_test.get(key, float("nan"))
            lines.append(f"| {key} | {cv.mean(key):.3f} | {result.test[key]:.3f} | {base:.3f} |")
    lines += [
        "",
        "Recall for each class (test): "
        + ", ".join(f"{k} = {v:.3f}" for k, v in result.test["recall_by_class"].items()) + ".",
        "",
        "## Limits",
        "",
        "- The metrics are unweighted. They describe this sample, not the national crash population.",
        "- Police-reported severity (KABCO) has known misclassification, mostly between C, B and A.",
        "- The model shows association, not cause.",
    ]
    return "\n".join(lines) + "\n"


def save(result: TrainResult, out_dir: str | Path, codebook_version: str, data_label: str) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": out_dir / "model.joblib",
        "metrics": out_dir / "metrics.json",
        "card": out_dir / "model_card.md",
    }
    joblib.dump({"pipeline": result.pipeline, "spec": asdict(result.spec), "target": result.target,
                 "model": result.model, "codebook": codebook_version}, paths["model"])
    payload = {
        "model": result.model, "target": result.target, "seed": result.seed,
        "cv_mean": {k: result.cv.mean(k) for k in SUMMARY_KEYS},
        "cv_out_of_fold": _summary(result.cv.oof), "cv_ci": result.cv.ci,
        "test": result.test, "test_ci": result.test_ci, "baseline_test": _summary(result.baseline_test),
        "n_train": result.n_train, "n_test": result.n_test,
    }
    paths["metrics"].write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    paths["card"].write_text(model_card(result, codebook_version, data_label), encoding="utf-8")
    return paths


def load(path: str | Path) -> dict:
    bundle = joblib.load(path)
    bundle["spec"] = FeatureSpec(**bundle["spec"])
    return bundle
