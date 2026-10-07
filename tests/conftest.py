import os

import pytest

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

from crashinsight.codebook import load_codebook  # noqa: E402
from crashinsight.dataset import build_dataset  # noqa: E402
from crashinsight.synthetic import generate  # noqa: E402


@pytest.fixture(scope="session")
def tables():
    return generate(n_crashes=800, seed=3)


@pytest.fixture(scope="session")
def codebook():
    return load_codebook()


@pytest.fixture(scope="session")
def built(tables, codebook):
    return build_dataset(tables["vehicle"], tables["accident"], tables["person"], codebook)
