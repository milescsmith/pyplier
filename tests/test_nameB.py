import importlib.resources

import pandas as pd
import pytest

from pyplier.nameB import nameB
from pyplier.stubs import PLIERResults

from .create_plierRes import plierRes


@pytest.fixture
def prenameB_plierRes(plierRes: PLIERResults):
    prenameB_file = importlib.resources.files("tests").joinpath(
        "data/nameB/plierRes_b_pre-nameB.csv"
    )
    with importlib.resources.as_file(prenameB_file) as nf:
        prior_names = pd.read_csv(nf, index_col=0)

    plierRes["B"] = prior_names
    return plierRes


@pytest.fixture
def names():
    names_file = importlib.resources.files("tests").joinpath(
        "data/common/plierRes_b.csv"
    )
    with importlib.resources.as_file(names_file) as nf:
        prior_names = list(pd.read_csv(nf, index_col=0).index)
    return prior_names


def test_nameB(plierRes, names):
    new_names = nameB(plierRes)

    assert new_names == names
