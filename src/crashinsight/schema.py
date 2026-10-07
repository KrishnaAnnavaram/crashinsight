"""Load the GES CSV files and validate their schema before any analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY_COLUMNS = ["CASENUM", "VEH_NO"]
DESIGN_COLUMNS = ["WEIGHT", "PSU", "PSUSTRAT"]
VEHICLE_REQUIRED = KEY_COLUMNS + DESIGN_COLUMNS + [
    "MAX_VSEV", "REGION", "BODY_TYP", "MOD_YEAR", "TRAV_SP", "VSPD_LIM", "MAN_COLL", "HIT_RUN",
    "ROLLOVER", "VEH_ALCH", "SPEEDREL", "VSURCOND", "NUMOCCS", "HOUR",
]
VEHICLE_OPTIONAL = ["VTRAFWAY", "VALIGN", "VPROFILE", "FIRE_EXP", "GVWR", "STRATUM"]
ACCIDENT_REQUIRED = ["CASENUM", "LGT_COND", "WEATHER"]
PERSON_REQUIRED = ["CASENUM", "VEH_NO", "PER_TYP", "AGE", "SEX"]
DRIVER_PER_TYP = 1


class SchemaError(ValueError):
    """The input file does not have the columns or the values that crashinsight needs."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def _check_columns(frame: pd.DataFrame, required: list[str], name: str) -> list[str]:
    missing = [c for c in required if c not in frame.columns]
    return [f"{name}: missing column {c}" for c in missing]


def _check_integer(frame: pd.DataFrame, columns: list[str], name: str) -> list[str]:
    problems = []
    for col in columns:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        if values.isna().any():
            problems.append(f"{name}: column {col} has {int(values.isna().sum())} empty or non-numeric values")
        elif not (values == values.round()).all():
            problems.append(f"{name}: column {col} must contain integer codes")
    return problems


def validate_vehicle(frame: pd.DataFrame) -> pd.DataFrame:
    problems = _check_columns(frame, VEHICLE_REQUIRED, "VEHICLE")
    if problems:
        raise SchemaError(problems)
    code_cols = [c for c in VEHICLE_REQUIRED + VEHICLE_OPTIONAL if c in frame.columns and c != "WEIGHT"]
    problems += _check_integer(frame, code_cols, "VEHICLE")
    weight = pd.to_numeric(frame["WEIGHT"], errors="coerce")
    if weight.isna().any() or (weight <= 0).any():
        problems.append("VEHICLE: WEIGHT must be a positive number in every row")
    if frame.duplicated(KEY_COLUMNS).any():
        problems.append("VEHICLE: (CASENUM, VEH_NO) must be unique")
    if problems:
        raise SchemaError(problems)
    return frame


def validate_accident(frame: pd.DataFrame) -> pd.DataFrame:
    problems = _check_columns(frame, ACCIDENT_REQUIRED, "ACCIDENT")
    if not problems:
        problems += _check_integer(frame, ACCIDENT_REQUIRED, "ACCIDENT")
        if frame.duplicated(["CASENUM"]).any():
            problems.append("ACCIDENT: CASENUM must be unique")
    if problems:
        raise SchemaError(problems)
    return frame


def validate_person(frame: pd.DataFrame) -> pd.DataFrame:
    problems = _check_columns(frame, PERSON_REQUIRED, "PERSON")
    if not problems:
        problems += _check_integer(frame, PERSON_REQUIRED, "PERSON")
    if problems:
        raise SchemaError(problems)
    return frame


def _read(path: Path) -> pd.DataFrame:
    # NHTSA files are Latin-1 encoded. Codes stay as numbers. The codebook decodes them later.
    return pd.read_csv(path, encoding="latin-1", low_memory=False)


def find_file(data_dir: Path, stem: str) -> Path | None:
    for candidate in (f"{stem}.CSV", f"{stem}.csv", f"{stem.lower()}.csv"):
        path = data_dir / candidate
        if path.exists():
            return path
    return None


def load_tables(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load VEHICLE (required), ACCIDENT and PERSON (optional) from one folder and validate them."""
    data_dir = Path(data_dir)
    vehicle_path = find_file(data_dir, "VEHICLE")
    if vehicle_path is None:
        raise FileNotFoundError(f"No VEHICLE.CSV in {data_dir}. Read data/README.md, or run `crashinsight synth`.")
    tables = {"vehicle": validate_vehicle(_read(vehicle_path))}
    accident_path = find_file(data_dir, "ACCIDENT")
    if accident_path is not None:
        tables["accident"] = validate_accident(_read(accident_path))
    person_path = find_file(data_dir, "PERSON")
    if person_path is not None:
        tables["person"] = validate_person(_read(person_path))
    return tables
