"""Problems 2 (IQR filter removed K and A), 3 (sentinels) and 9 (portability), plus the joins."""
import numpy as np
import pandas as pd
import pytest

from crashinsight.config import Settings
from crashinsight.dataset import LEAKAGE_COLUMNS, build_dataset, kabco_from_max_vsev, target_values
from crashinsight.schema import SchemaError, load_tables, validate_vehicle
from crashinsight.synthetic import write


def test_serious_and_fatal_rows_are_kept(tables, built):
    raw = tables["vehicle"]
    assert (built.frame["severity"] == "K").sum() == (raw["MAX_VSEV"] == 4).sum()
    assert (built.frame["severity"] == "A").sum() == (raw["MAX_VSEV"] == 3).sum()


def test_unknown_and_not_applicable_targets_are_removed_and_counted(tables, built):
    raw = tables["vehicle"]["MAX_VSEV"]
    assert built.dropped_unknown_target == int(raw.isin([5, 9]).sum())
    assert built.dropped_not_applicable_target == int(raw.isin([6, 8]).sum())
    assert len(built.frame) == int(raw.isin([0, 1, 2, 3, 4]).sum())


def test_kabco_mapping():
    assert kabco_from_max_vsev(pd.Series([0, 1, 2, 3, 4, 9])).tolist()[:5] == ["O", "C", "B", "A", "K"]
    assert pd.isna(kabco_from_max_vsev(pd.Series([9]))[0])


def test_ka_target_matches_severity(built):
    ka = target_values(built.frame, "ka")
    assert ((ka == 1) == built.frame["severity"].isin(["K", "A"])).all()
    with pytest.raises(ValueError):
        target_values(built.frame, "fatal")


def test_no_sentinel_values_reach_numeric_features(built):
    frame = built.frame
    assert frame["travel_speed"].dropna().max() <= 152
    assert frame["vehicle_age"].dropna().max() < 200
    assert frame["speed_limit"].dropna().max() <= 95
    assert frame["driver_age"].dropna().max() <= 120


def test_vehicle_age_uses_the_crash_year(tables):
    vehicle = tables["vehicle"].copy()
    vehicle["MOD_YEAR"] = 2005
    result = build_dataset(vehicle, crash_year=2015)
    assert set(result.frame["vehicle_age"].unique()) == {10.0}


def test_leakage_columns_are_never_features(built):
    sources = {"MAX_VSEV", "NUM_INJV", "DEFORMED", "TOWED", "MXVSEV_IM"}
    assert sources <= LEAKAGE_COLUMNS
    assert not set(built.spec.all) & {c.lower() for c in LEAKAGE_COLUMNS}
    assert "severity" not in built.spec.all and "ka" not in built.spec.all


def test_driver_and_crash_context_are_joined(built):
    assert {"driver_age", "driver_sex", "light", "weather"} <= set(built.spec.all)
    assert built.frame["light"].notna().all()


def test_schema_rejects_missing_column_and_bad_weight(tables):
    with pytest.raises(SchemaError, match="missing column MAX_VSEV"):
        validate_vehicle(tables["vehicle"].drop(columns=["MAX_VSEV"]))
    bad = tables["vehicle"].copy()
    bad.loc[0, "WEIGHT"] = 0
    with pytest.raises(SchemaError, match="WEIGHT"):
        validate_vehicle(bad)
    dup = pd.concat([tables["vehicle"].head(2), tables["vehicle"].head(1)])
    with pytest.raises(SchemaError, match="unique"):
        validate_vehicle(dup)


def test_load_tables_from_folder(tmp_path):
    write(tmp_path, n_crashes=50, seed=1)
    loaded = load_tables(tmp_path)
    assert set(loaded) == {"vehicle", "accident", "person"}
    with pytest.raises(FileNotFoundError):
        load_tables(tmp_path / "missing")


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("CRASHINSIGHT_SEED", "7")
    monkeypatch.setenv("CRASHINSIGHT_CRASH_YEAR", "2014")
    monkeypatch.setenv("CRASHINSIGHT_N_SPLITS", "3")
    s = Settings.from_env()
    assert (s.seed, s.crash_year, s.n_splits) == (7, 2014, 3)
    monkeypatch.setenv("CRASHINSIGHT_N_SPLITS", "1")
    with pytest.raises(ValueError):
        Settings.from_env()
    monkeypatch.setenv("CRASHINSIGHT_N_SPLITS", "abc")
    with pytest.raises(ValueError):
        Settings.from_env()


def test_synthetic_generator_is_seeded():
    from crashinsight.synthetic import generate
    a = generate(40, seed=5)["vehicle"]
    b = generate(40, seed=5)["vehicle"]
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(generate(40, seed=6)["vehicle"])
