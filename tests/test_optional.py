"""Tests for the optional SHAP extra. They skip when `shap` is not installed (for example in CI)."""
import pytest

from crashinsight.dataset import target_values
from crashinsight.explain import shap_values
from crashinsight.models import build_pipeline


def test_shap_values_for_logreg(built):
    pytest.importorskip("shap")
    y = target_values(built.frame, "ka")
    pipe = build_pipeline(built.spec, "logreg").fit(built.frame[built.spec.all], y)
    values, names = shap_values(pipe, built.frame[built.spec.all], max_rows=40)
    assert values.values.shape[0] == 40
    assert len(names) == values.values.shape[1]
