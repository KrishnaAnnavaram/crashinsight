"""Build one analysis table: decoded features, the severity target and the survey design columns.

Rules:
- No outlier removal. Codes are categories, not measurements, and serious injuries are the rare
  outcome that matters.
- Sentinel codes become NaN or their own category (see codebook.py).
- Post-crash and outcome variables (damage, towing, injury counts, imputed severity) never become features.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .codebook import Codebook, load_codebook
from .schema import DRIVER_PER_TYP

# KABCO scale. Index order is the severity order.
KABCO_ORDER = ["O", "C", "B", "A", "K"]
MAX_VSEV_TO_KABCO = {0: "O", 1: "C", 2: "B", 3: "A", 4: "K"}
SERIOUS = {"K", "A"}

# Columns that describe the outcome or what happened after the crash. They must never be features.
LEAKAGE_COLUMNS = frozenset({
    "MAX_VSEV", "MXVSEV_IM", "NUM_INJV", "NUMINJ_IM", "DEFORMED", "TOWED", "FIRE_EXP",
})

NOMINAL_SOURCES = {
    "region": "REGION", "body_group": "BODY_TYP", "collision_manner": "MAN_COLL",
    "surface": "VSURCOND", "trafficway": "VTRAFWAY", "alignment": "VALIGN", "grade": "VPROFILE",
    "weight_class": "GVWR", "light": "LGT_COND", "weather": "WEATHER", "driver_sex": "SEX",
}
BINARY_SOURCES = {
    "hit_run": "HIT_RUN", "rollover": "ROLLOVER", "alcohol": "VEH_ALCH", "speeding": "SPEEDREL",
}
NUMERIC_SOURCES = {"travel_speed": "TRAV_SP", "speed_limit": "VSPD_LIM", "occupants": "NUMOCCS", "driver_age": "AGE"}


@dataclass
class FeatureSpec:
    numeric: list[str] = field(default_factory=list)
    binary: list[str] = field(default_factory=list)
    nominal: list[str] = field(default_factory=list)

    @property
    def all(self) -> list[str]:
        return self.numeric + self.binary + self.nominal


@dataclass
class BuildResult:
    frame: pd.DataFrame
    spec: FeatureSpec
    dropped_unknown_target: int
    dropped_not_applicable_target: int


def time_of_day(hour: pd.Series) -> pd.Series:
    bins = [-1, 5, 9, 15, 19, 23]
    names = ["Night 00-05", "Morning 06-09", "Midday 10-15", "Evening 16-19", "Late 20-23"]
    out = pd.cut(hour, bins=bins, labels=names).astype(object)
    return out.where(hour.notna(), np.nan)


def kabco_from_max_vsev(codes: pd.Series) -> pd.Series:
    raw = pd.to_numeric(codes, errors="coerce")
    return raw.map(MAX_VSEV_TO_KABCO)


def join_context(vehicle: pd.DataFrame, accident: pd.DataFrame | None, person: pd.DataFrame | None) -> pd.DataFrame:
    """Add crash-level (ACCIDENT) and driver-level (PERSON) columns to each vehicle row."""
    out = vehicle.copy()
    if accident is not None:
        cols = ["CASENUM", "LGT_COND", "WEATHER"]
        out = out.merge(accident[cols], on="CASENUM", how="left", validate="many_to_one")
    if person is not None:
        drivers = person[person["PER_TYP"] == DRIVER_PER_TYP][["CASENUM", "VEH_NO", "AGE", "SEX"]]
        drivers = drivers.drop_duplicates(["CASENUM", "VEH_NO"])
        out = out.merge(drivers, on=["CASENUM", "VEH_NO"], how="left", validate="one_to_one")
    return out


def build_dataset(
    vehicle: pd.DataFrame,
    accident: pd.DataFrame | None = None,
    person: pd.DataFrame | None = None,
    codebook: Codebook | None = None,
    crash_year: int = 2015,
) -> BuildResult:
    codebook = codebook or load_codebook()
    raw = join_context(vehicle, accident, person)
    out = pd.DataFrame(index=raw.index)
    for col in ("CASENUM", "VEH_NO", "PSU", "PSUSTRAT"):
        out[col] = raw[col].astype(np.int64)
    out["WEIGHT"] = raw["WEIGHT"].astype(float)

    spec = FeatureSpec()
    for name, source in NUMERIC_SOURCES.items():
        if source in raw.columns:
            out[name] = codebook[source].decode(raw[source])
            spec.numeric.append(name)
    model_year = codebook["MOD_YEAR"].decode(raw["MOD_YEAR"])
    # Vehicle age at the time of the crash, not at the time of the analysis.
    out["vehicle_age"] = (crash_year - model_year).clip(lower=0)
    spec.numeric.append("vehicle_age")
    for name, source in BINARY_SOURCES.items():
        if source in raw.columns:
            out[name] = codebook[source].decode(raw[source])
            spec.binary.append(name)
    for name, source in NOMINAL_SOURCES.items():
        if source in raw.columns:
            out[name] = codebook[source].decode(raw[source])
            spec.nominal.append(name)
    out["time_of_day"] = time_of_day(codebook["HOUR"].decode(raw["HOUR"]))
    spec.nominal.append("time_of_day")

    severity = kabco_from_max_vsev(raw["MAX_VSEV"])
    sev_var = codebook["MAX_VSEV"]
    max_vsev = pd.to_numeric(raw["MAX_VSEV"], errors="coerce")
    n_unknown = int(max_vsev.isin(list(sev_var.unknown)).sum())
    n_na = int(max_vsev.isin(list(sev_var.not_applicable)).sum())
    out["severity"] = severity
    out["ka"] = severity.isin(SERIOUS).astype(float).where(severity.notna())
    out = out[out["severity"].notna()].reset_index(drop=True)
    out["ka"] = out["ka"].astype(int)
    return BuildResult(frame=out, spec=spec, dropped_unknown_target=n_unknown, dropped_not_applicable_target=n_na)


def target_values(frame: pd.DataFrame, target: str) -> pd.Series:
    if target == "ka":
        return frame["ka"].astype(int)
    if target == "kabco":
        return frame["severity"].astype(str)
    raise ValueError(f"target must be 'ka' or 'kabco', got {target!r}")
