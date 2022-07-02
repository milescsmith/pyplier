from sys import version_info

if version_info[1] == 8:
    import importlib_resources as ir
elif version_info[1] >= 9:
    import importlib.resources as ir

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings

from pyplier.AUC import mannwhitneyu_conf_int


# cheat here a bit to speed things up.
@given(
    st.lists(
        st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
        min_size=20,
    ),
    st.lists(
        st.floats(min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False),
        min_size=20,
    ),
)
@settings(deadline=None)
def testconfint(x, y):
    assume(len(x) > 20)
    assume(len(y) > 20)
    mannwhitneyu_conf_int(x, y)


@pytest.fixture
def test_pos():
    posval_file = ir.files("tests").joinpath(
        "data", "mannwhitney_conf_int", "posval.csv.gz"
    )
    with ir.as_file(posval_file) as lf:
        posval_sr = pd.read_csv(lf, index_col=0).squeeze("columns")
    return posval_sr


@pytest.fixture
def test_neg():
    negval_file = ir.files("tests").joinpath(
        "data", "mannwhitney_conf_int", "negval.csv.gz"
    )
    with ir.as_file(negval_file) as lf:
        negval_sr = pd.read_csv(lf, index_col=0).squeeze("columns")
    return negval_sr


def test_mannwhitneyu_conf_int_greater(test_pos, test_neg):
    low, high = mannwhitneyu_conf_int(test_pos, test_neg, alternative="greater")
    assert (low, high) == (pytest.approx(0.007149831510470957), np.inf)


def test_mannwhitneyu_conf_int_two_sided(test_pos, test_neg):
    low, high = mannwhitneyu_conf_int(test_pos, test_neg, alternative="two_sided")
    assert (low, high) == (
        pytest.approx(0.0023365916996431397),
        pytest.approx(0.05108464103143653),
    )


def test_mannwhitneyu_conf_int_less(test_pos, test_neg):
    low, high = mannwhitneyu_conf_int(test_pos, test_neg, alternative="less")
    assert (low, high) == (-np.inf, pytest.approx(0.044610369079090474))
