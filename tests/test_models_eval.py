"""Problems 4 (nominal codes as numbers), 5 (binary metrics on a multi-class target) and 8 (no baseline)."""
import numpy as np
import pandas as pd
import pytest

from crashinsight.dataset import target_values
from crashinsight.evaluate import (
    calibration_table,
    classification_metrics,
    grouped_bootstrap_ci,
    ka_pr_auc_metric,
    ka_score,
)
from crashinsight.explain import permutation_table
from crashinsight.models import OrdinalClassifier, build_pipeline, build_preprocessor
from crashinsight.split import assert_disjoint_groups, grouped_folds, grouped_holdout
from crashinsight.train import cross_validate, load, save, train


def test_nominal_features_are_one_hot_not_scaled(built):
    prep = build_preprocessor(built.spec).fit(built.frame[built.spec.all])
    names = list(prep.get_feature_names_out())
    assert any(n.startswith("nominal__body_group_") for n in names)
    assert "nominal__body_group" not in names
    scaled = [n for n in names if n.startswith("numeric__")]
    assert not any("body_group" in n or "region" in n for n in scaled)


def test_preprocessing_is_fit_on_training_rows_only(built):
    train_rows = built.frame[built.frame["region"] != "West"]
    prep = build_preprocessor(built.spec).fit(train_rows[built.spec.all])
    names = prep.get_feature_names_out()
    assert not any(n.endswith("region_West") for n in names)
    # Unseen categories at predict time do not fail.
    prep.transform(built.frame[built.spec.all])


def test_folds_never_split_a_crash(built):
    y = target_values(built.frame, "ka")
    for tr, te in grouped_folds(y, built.frame["CASENUM"], n_splits=4, seed=0):
        assert_disjoint_groups(built.frame["CASENUM"], tr, te)
    tr, te = grouped_holdout(built.frame, y, test_fraction=0.2, seed=0)
    assert_disjoint_groups(built.frame["CASENUM"], tr, te)
    assert 0.1 < len(te) / len(built.frame) < 0.3


def test_disjoint_check_detects_overlap():
    groups = pd.Series([1, 1, 2, 2])
    with pytest.raises(AssertionError):
        assert_disjoint_groups(groups, np.array([0, 2]), np.array([1, 3]))


def test_ka_score_uses_labels_not_column_positions():
    classes = np.array(["A", "B", "C", "K", "O"])
    proba = np.array([[0.1, 0.2, 0.3, 0.05, 0.35]])
    assert ka_score(proba, classes)[0] == pytest.approx(0.15)
    assert ka_score(np.array([[0.7, 0.3]]), np.array([0, 1]))[0] == pytest.approx(0.3)


def test_multiclass_metrics_run_and_are_consistent():
    rng = np.random.default_rng(0)
    classes = np.array(["O", "C", "B", "A", "K"])
    y = rng.choice(classes, 300, p=[0.6, 0.15, 0.12, 0.1, 0.03])
    proba = rng.dirichlet(np.ones(5), 300)
    m = classification_metrics(y, proba, classes)
    assert set(m["recall_by_class"]) == set(classes)
    assert 0 <= m["macro_f1"] <= 1 and 0 <= m["ka_pr_auc"] <= 1
    assert "macro_roc_auc_ovr" in m
    assert np.asarray(m["confusion_matrix"]).sum() == 300


def test_calibration_table_bins():
    table = calibration_table(np.array([0, 0, 1, 1]), np.array([0.05, 0.15, 0.85, 0.95]), bins=10)
    assert table["n"].sum() == 4


def test_bootstrap_ci_resamples_crashes():
    y = np.array([0, 1] * 50)
    proba = np.column_stack([1 - np.linspace(0, 1, 100), np.linspace(0, 1, 100)])
    low, high = grouped_bootstrap_ci(ka_pr_auc_metric(np.array([0, 1])), y, proba, np.arange(100) // 2, n_boot=50)
    assert 0 <= low <= high <= 1


def test_models_beat_the_majority_baseline_on_rare_class(built):
    cv_base = cross_validate(built.frame, built.spec, "ka", "majority", n_splits=3, seed=0, n_boot=0)
    cv_lr = cross_validate(built.frame, built.spec, "ka", "logreg", n_splits=3, seed=0, n_boot=0)
    assert cv_base.oof["recall_by_class"]["1"] == 0.0
    assert cv_lr.oof["recall_by_class"]["1"] > 0.2
    assert cv_lr.oof["ka_pr_auc"] > cv_base.oof["ka_pr_auc"]
    assert cv_lr.oof["macro_f1"] > cv_base.oof["macro_f1"]


def test_ordinal_model_keeps_order_and_valid_probabilities(built):
    pipe = build_pipeline(built.spec, "ordinal", seed=0)
    y = target_values(built.frame, "kabco")
    pipe.fit(built.frame[built.spec.all], y)
    proba = pipe.predict_proba(built.frame[built.spec.all].head(50))
    assert list(pipe.classes_) == ["O", "C", "B", "A", "K"]
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()
    with pytest.raises(ValueError):
        OrdinalClassifier().fit(np.zeros((2, 1)), ["O", "X"])


def test_ordinal_model_needs_kabco_target(built):
    with pytest.raises(ValueError, match="kabco"):
        train(built.frame, built.spec, target="ka", model="ordinal")


def test_unknown_model_name(built):
    with pytest.raises(ValueError):
        build_pipeline(built.spec, "svm")


def test_train_save_load_roundtrip(built, tmp_path):
    result = train(built.frame, built.spec, target="ka", model="logreg", seed=0, n_splits=3, n_boot=20)
    assert result.n_train + result.n_test == len(built.frame)
    assert result.baseline_test["ka_pr_auc"] < result.test["ka_pr_auc"]
    paths = save(result, tmp_path, "ges2015-v1", "synthetic")
    card = paths["card"].read_text(encoding="utf-8")
    assert "Intended use" in card and "not a tool for decisions" in card
    bundle = load(paths["model"])
    proba = bundle["pipeline"].predict_proba(built.frame[bundle["spec"].all].head(5))
    assert proba.shape == (5, 2)


def test_training_is_reproducible(built):
    a = train(built.frame, built.spec, target="ka", model="boosting", seed=3, n_splits=3, n_boot=0)
    b = train(built.frame, built.spec, target="ka", model="boosting", seed=3, n_splits=3, n_boot=0)
    assert a.test["ka_pr_auc"] == b.test["ka_pr_auc"]


def test_permutation_importance_by_raw_feature(built):
    y = target_values(built.frame, "ka")
    pipe = build_pipeline(built.spec, "logreg").fit(built.frame[built.spec.all], y)
    table = permutation_table(pipe, built.frame[built.spec.all].head(400), y.head(400), n_repeats=2)
    assert set(table["feature"]) == set(built.spec.all)
    assert table.iloc[0]["ka_pr_auc_drop"] >= table.iloc[-1]["ka_pr_auc_drop"]
