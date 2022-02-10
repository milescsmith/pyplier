from sys import version_info

if version_info[1] == 8:
    import importlib_resources as ir
elif version_info[1] >= 9:
    import importlib.resources as ir

import pandas as pd
import pytest

from pyplier.nameB import nameB
from pyplier.stubs import PLIERResults

from .create_plierRes import plierRes


@pytest.fixture
def prenameB_plierRes(plierRes: PLIERResults):
    prenameB_file = ir.files("tests").joinpath("data/nameB/plierRes_b_pre-nameB.csv.gz")
    with ir.as_file(prenameB_file) as nf:
        prior_names = pd.read_csv(nf, index_col=0)

    plierRes.B = prior_names
    return plierRes


@pytest.fixture
def names():
    names_file = ir.files("tests").joinpath("data", "common", "plierRes_b.csv.gz")
    with ir.as_file(names_file) as nf:
        prior_names = list(pd.read_csv(nf, index_col=0).index)
    return prior_names


def test_nameB(plierRes, names):
    new_names = nameB(plierRes)

    assert new_names == names
