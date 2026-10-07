"""Problem 6 (survey weights ignored) and problem 7 (mis-specified hypothesis test)."""
import numpy as np
import pandas as pd
import pytest

from crashinsight.analysis import NEWER_MAX_AGE, age_effect
from crashinsight.survey import domain_mean, linearized_variance, weighted_mean, weighted_rate_table


def test_weighted_mean_matches_hand_calculation():
    y = np.array([1, 0, 0, 1.0])
    w = np.array([1, 1, 1, 3.0])
    est = weighted_mean(y, w, strata=[1, 1, 1, 1], psu=[1, 2, 3, 4])
    assert est.value == pytest.approx(4 / 6)
    assert est.se > 0
    assert est.weighted_n == 6


def test_weights_change_the_estimate():
    y = np.array([1, 0, 0, 0.0])
    unweighted = weighted_mean(y, np.ones(4), [1] * 4, [1, 2, 3, 4]).value
    weighted = weighted_mean(y, np.array([10, 1, 1, 1.0]), [1] * 4, [1, 2, 3, 4]).value
    assert unweighted == pytest.approx(0.25)
    assert weighted == pytest.approx(10 / 13)


def test_linearized_variance_known_value():
    # One stratum, two PSUs with totals 1 and 3: variance = 2/1 * ((1-2)^2 + (3-2)^2) = 4.
    assert linearized_variance(np.array([1.0, 3.0]), np.array([1, 1]), np.array([1, 2])) == pytest.approx(4.0)
    # A stratum with one PSU adds nothing.
    assert linearized_variance(np.array([5.0]), np.array([1]), np.array([1])) == 0.0


def test_clustering_increases_the_standard_error():
    rng = np.random.default_rng(0)
    psu = np.repeat(np.arange(20), 25)
    cluster_effect = rng.normal(0, 1, 20)[psu]
    y = (cluster_effect + rng.normal(0, 0.3, len(psu)) > 0).astype(float)
    w = np.ones(len(y))
    clustered = weighted_mean(y, w, np.ones(len(y)), psu).se
    independent = weighted_mean(y, w, np.ones(len(y)), np.arange(len(y))).se
    assert clustered > 1.5 * independent


def test_domain_mean_keeps_all_psus():
    y = pd.Series([1, 0, 1, 0.0])
    est = domain_mean(y, pd.Series([True, True, False, False]), np.ones(4), [1, 1, 1, 1], [1, 2, 3, 4])
    assert est.value == pytest.approx(0.5)
    assert est.n == 2


def test_rate_table_has_ci_and_weighted_counts(built):
    table = weighted_rate_table(built.frame, "body_group")
    assert {"rate", "se", "ci_low", "ci_high", "weighted_n", "unweighted_rate", "small_cell"} <= set(table.columns)
    assert (table["ci_low"] <= table["rate"]).all() and (table["rate"] <= table["ci_high"]).all()
    assert table["n"].sum() == len(built.frame)


def test_age_effect_is_adjusted_and_two_sided(built):
    result = age_effect(built.frame, bootstrap=8, seed=1)
    assert result.newer_max_age == NEWER_MAX_AGE == 10
    assert {"body_group", "speed_limit", "rollover", "alcohol"} <= set(result.controls)
    assert result.ci_low <= result.odds_ratio <= result.ci_high
    assert result.bootstrap_reps > 0
    assert "odds ratio = 1" in result.summary()


def test_age_effect_recovers_a_planted_effect():
    rng = np.random.default_rng(1)
    n = 4000
    age = rng.integers(0, 25, n).astype(float)
    newer = age <= 10
    p = np.where(newer, 0.05, 0.15)
    frame = pd.DataFrame({
        "vehicle_age": age, "ka": (rng.random(n) < p).astype(int), "WEIGHT": np.ones(n),
        "PSUSTRAT": rng.integers(1, 4, n), "PSU": rng.integers(1, 30, n),
        "body_group": rng.choice(["Passenger car", "Pickup truck"], n),
    })
    result = age_effect(frame, bootstrap=10, seed=0)
    expected = (0.05 / 0.95) / (0.15 / 0.85)
    assert result.odds_ratio == pytest.approx(expected, rel=0.25)
    assert result.significant


def test_age_effect_needs_both_groups(built):
    frame = built.frame.assign(vehicle_age=1.0)
    with pytest.raises(ValueError):
        age_effect(frame, bootstrap=2)
