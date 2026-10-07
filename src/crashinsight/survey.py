"""Survey-weighted estimates for the GES complex sample design.

GES is a stratified, clustered probability sample. A national estimate must use the case weight
(`WEIGHT`). Its standard error must use the variance strata (`PSUSTRAT`) and the primary sampling
units (`PSU`). This module uses the Taylor linearization (ultimate cluster) estimator with PSUs drawn
with replacement within each stratum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class Estimate:
    value: float
    se: float
    n: int
    weighted_n: float

    def ci(self, level: float = 0.95) -> tuple[float, float]:
        z = stats.norm.ppf(0.5 + level / 2)
        return self.value - z * self.se, self.value + z * self.se


def linearized_variance(z: np.ndarray, strata: np.ndarray, psu: np.ndarray) -> float:
    """Variance of a total of linearized values `z` under a stratified with-replacement PSU design."""
    frame = pd.DataFrame({"z": z, "h": strata, "c": psu})
    totals = frame.groupby(["h", "c"], sort=False)["z"].sum().reset_index()
    var = 0.0
    for _, group in totals.groupby("h", sort=False):
        n_h = len(group)
        if n_h < 2:
            # A stratum with one PSU adds no variance information. Its contribution is zero.
            continue
        dev = group["z"].to_numpy() - group["z"].mean()
        var += n_h / (n_h - 1) * float(np.sum(dev**2))
    return var


def weighted_mean(
    y: pd.Series | np.ndarray,
    weight: pd.Series | np.ndarray,
    strata: pd.Series | np.ndarray,
    psu: pd.Series | np.ndarray,
) -> Estimate:
    """Ratio estimate sum(w*y)/sum(w) with a linearized standard error. NaN rows are excluded."""
    y = np.asarray(y, dtype=float)
    w = np.asarray(weight, dtype=float)
    h = np.asarray(strata)
    c = np.asarray(psu)
    keep = ~np.isnan(y)
    y, w, h, c = y[keep], w[keep], h[keep], c[keep]
    if len(y) == 0 or w.sum() <= 0:
        return Estimate(float("nan"), float("nan"), 0, 0.0)
    total_w = w.sum()
    mean = float(np.sum(w * y) / total_w)
    z = w * (y - mean) / total_w
    se = float(np.sqrt(linearized_variance(z, h, c)))
    return Estimate(mean, se, int(len(y)), float(total_w))


def weighted_rate_table(
    frame: pd.DataFrame,
    by: str,
    outcome: str = "ka",
    weight: str = "WEIGHT",
    strata: str = "PSUSTRAT",
    psu: str = "PSU",
    min_n: int = 30,
) -> pd.DataFrame:
    """Weighted outcome rate for each level of `by`, with SE, 95% CI, raw n and weighted n."""
    rows = []
    for level in frame[by].drop_duplicates().sort_values(na_position="last"):
        # Domain estimation: rows outside the domain keep their design position with z = 0.
        indicator = (frame[by] == level) if not pd.isna(level) else frame[by].isna()
        y = frame[outcome].where(indicator, np.nan)
        est = domain_mean(frame[outcome], indicator, frame[weight], frame[strata], frame[psu])
        low, high = est.ci()
        rows.append({
            by: "Missing" if pd.isna(level) else level,
            "n": est.n,
            "weighted_n": int(round(est.weighted_n)),
            "rate": est.value,
            "se": est.se,
            "ci_low": max(0.0, low),
            "ci_high": min(1.0, high),
            "unweighted_rate": float(np.nanmean(y)),
            "small_cell": est.n < min_n,
        })
    return pd.DataFrame(rows)


def domain_mean(y, indicator, weight, strata, psu) -> Estimate:
    """Weighted mean of `y` in a domain (subpopulation). Rows outside the domain keep z = 0."""
    y = np.asarray(y, dtype=float)
    d = np.asarray(indicator, dtype=bool) & ~np.isnan(y)
    w = np.asarray(weight, dtype=float)
    total_w = float(np.sum(w[d]))
    if total_w <= 0:
        return Estimate(float("nan"), float("nan"), 0, 0.0)
    mean = float(np.sum(w[d] * y[d]) / total_w)
    z = np.where(d, w * (np.nan_to_num(y) - mean) / total_w, 0.0)
    se = float(np.sqrt(linearized_variance(z, np.asarray(strata), np.asarray(psu))))
    return Estimate(mean, se, int(d.sum()), total_w)
