"""The vehicle-age question, answered with a weighted regression that has controls.

Question: do vehicles of 10 model years or less at the crash have a different chance of a serious or
fatal injury (KA) than older vehicles, after control for body type, speed limit, collision manner,
rollover, alcohol, speeding, region, time of day and light condition?

H0: the adjusted odds ratio (newer vs. older) is 1. H1 (two-sided): it is not 1.
The confidence interval comes from a PSU bootstrap within each variance stratum.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .survey import Estimate, domain_mean

NEWER_MAX_AGE = 10
CONTROL_NOMINAL = ["body_group", "collision_manner", "region", "time_of_day", "light"]
CONTROL_NUMERIC = ["speed_limit"]
CONTROL_BINARY = ["rollover", "alcohol", "speeding"]


@dataclass
class AgeEffect:
    newer_max_age: int
    n: int
    rate_newer: Estimate
    rate_older: Estimate
    odds_ratio: float
    ci_low: float
    ci_high: float
    bootstrap_reps: int
    controls: list[str] = field(default_factory=list)

    @property
    def significant(self) -> bool:
        return not (self.ci_low <= 1.0 <= self.ci_high)

    def summary(self) -> str:
        direction = "lower" if self.odds_ratio < 1 else "higher"
        verdict = "rejects" if self.significant else "does not reject"
        return (
            f"Adjusted odds ratio of KA, newer (<= {self.newer_max_age} years) vs. older: "
            f"{self.odds_ratio:.3f} (95% bootstrap CI {self.ci_low:.3f} to {self.ci_high:.3f}, "
            f"{self.bootstrap_reps} PSU resamples). Newer vehicles have {direction} adjusted odds. "
            f"The interval {verdict} H0 (odds ratio = 1)."
        )


def _design(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    nominal = [c for c in CONTROL_NOMINAL if c in frame.columns]
    numeric = [c for c in CONTROL_NUMERIC if c in frame.columns]
    binary = [c for c in CONTROL_BINARY if c in frame.columns]
    data = frame[frame["vehicle_age"].notna()].copy()
    data["newer"] = (data["vehicle_age"] <= NEWER_MAX_AGE).astype(int)
    return data, nominal, numeric, binary


def _fit_log_odds_ratio(data: pd.DataFrame, nominal, numeric, binary) -> float:
    prep = ColumnTransformer([
        ("newer", "passthrough", ["newer"]),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)),
                          ("sc", StandardScaler())]), numeric),
        ("bin", SimpleImputer(strategy="constant", fill_value=0, add_indicator=True), binary),
        ("nom", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("oh", OneHotEncoder(drop="first", handle_unknown="ignore"))]), nominal),
    ], sparse_threshold=0)
    model = Pipeline([("prep", prep), ("lr", LogisticRegression(C=1e4, max_iter=5000))])
    model.fit(data[["newer"] + numeric + binary + nominal], data["ka"], lr__sample_weight=data["WEIGHT"].to_numpy())
    # The "newer" column is the first column of the design matrix.
    return float(model.named_steps["lr"].coef_[0][0])


def _psu_resample(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    parts = []
    for _, stratum in data.groupby("PSUSTRAT", sort=False):
        psus = stratum["PSU"].unique()
        picks = rng.choice(psus, size=len(psus), replace=True)
        for k, psu in enumerate(picks):
            part = stratum[stratum["PSU"] == psu]
            parts.append(part.assign(PSU=f"{psu}_{k}"))
    return pd.concat(parts, ignore_index=True)


def age_effect(frame: pd.DataFrame, bootstrap: int = 50, seed: int = 42) -> AgeEffect:
    data, nominal, numeric, binary = _design(frame)
    if data["newer"].nunique() < 2:
        raise ValueError("Both newer and older vehicles are necessary for the age analysis")
    design = (data["WEIGHT"], data["PSUSTRAT"], data["PSU"])
    rate_newer = domain_mean(data["ka"], data["newer"] == 1, *design)
    rate_older = domain_mean(data["ka"], data["newer"] == 0, *design)
    beta = _fit_log_odds_ratio(data, nominal, numeric, binary)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        sample = _psu_resample(data, rng)
        if sample["ka"].nunique() < 2 or sample["newer"].nunique() < 2:
            continue
        draws.append(_fit_log_odds_ratio(sample, nominal, numeric, binary))
    if draws:
        low, high = np.percentile(draws, [2.5, 97.5])
    else:
        low = high = float("nan")
    return AgeEffect(
        newer_max_age=NEWER_MAX_AGE,
        n=len(data),
        rate_newer=rate_newer,
        rate_older=rate_older,
        odds_ratio=float(np.exp(beta)),
        ci_low=float(np.exp(low)),
        ci_high=float(np.exp(high)),
        bootstrap_reps=len(draws),
        controls=nominal + numeric + binary,
    )
