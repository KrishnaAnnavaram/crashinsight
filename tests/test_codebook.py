"""Problem 1 (invented labels) and problem 3 (sentinels used as numbers)."""
import json

import numpy as np
import pandas as pd
import pytest

from crashinsight.codebook import NOT_APPLICABLE_LABEL, UNKNOWN_LABEL, load_codebook


def test_bundled_codebook_loads_with_source(codebook):
    assert codebook.version == "ges2015-v1"
    assert "DOT HS 812 320" in codebook.source
    assert {"MAX_VSEV", "BODY_TYP", "TRAV_SP", "MOD_YEAR", "WEATHER"} <= set(codebook.variables)


def test_body_type_is_decoded_at_group_level_without_invented_labels(codebook):
    body = codebook["BODY_TYP"]
    decoded = body.decode(pd.Series([4, 14, 31, 49, 66, 80, 99]))
    assert decoded.tolist() == [
        "Passenger car", "Utility vehicle", "Pickup truck", "Other light truck", "Medium or heavy truck",
        "Motorcycle", UNKNOWN_LABEL,
    ]
    labels = {text for _, _, text in body.ranges}
    assert not {"Aircraft", "Spacecraft"} & labels


def test_max_vsev_labels_follow_kabco(codebook):
    sev = codebook["MAX_VSEV"]
    assert sev.label_of(4).startswith("K")
    assert sev.label_of(3).startswith("A")
    assert sev.unknown == {5, 9}
    assert sev.not_applicable == {6, 8}


@pytest.mark.parametrize("var, raw, expected", [
    ("TRAV_SP", [45, 998, 999, 997], [45.0, np.nan, np.nan, 152.0]),
    ("MOD_YEAR", [2010, 9998, 9999], [2010.0, np.nan, np.nan]),
    ("VSPD_LIM", [0, 35, 98, 99], [np.nan, 35.0, np.nan, np.nan]),
    ("HOUR", [0, 23, 99], [0.0, 23.0, np.nan]),
])
def test_numeric_sentinels_become_nan(codebook, var, raw, expected):
    out = codebook[var].decode(pd.Series(raw)).to_numpy()
    np.testing.assert_array_equal(np.isnan(out), np.isnan(expected))
    np.testing.assert_allclose(out[~np.isnan(out)], np.asarray(expected)[~np.isnan(expected)])


def test_binary_decode_keeps_unknown_missing(codebook):
    alcohol = codebook["VEH_ALCH"].decode(pd.Series([1, 2, 8, 9])).tolist()
    assert alcohol[:2] == [1.0, 0.0]
    assert np.isnan(alcohol[2]) and np.isnan(alcohol[3])
    rollover = codebook["ROLLOVER"].decode(pd.Series([0, 1, 2, 9])).tolist()
    assert rollover == [0.0, 1.0, 1.0, 1.0]


def test_nominal_sentinels_get_their_own_category(codebook):
    out = codebook["VSURCOND"].decode(pd.Series([1, 0, 99])).tolist()
    assert out == ["Dry", NOT_APPLICABLE_LABEL, UNKNOWN_LABEL]


def test_every_synthetic_code_has_a_label(tables, codebook):
    for frame in tables.values():
        assert codebook.undecoded_codes(frame) == {}


def test_unknown_codes_are_reported(codebook):
    frame = pd.DataFrame({"MAN_COLL": [1, 2, 3, 4], "TRAV_SP": [30, 400, 999, 10]})
    assert codebook.undecoded_codes(frame) == {"MAN_COLL": [3, 4], "TRAV_SP": [400]}


def test_bad_codebook_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"variables": {"X": {"kind": "numeric"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="valid range"):
        load_codebook(path)
