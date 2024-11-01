import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from pyplier.auc import mannwhitneyu_conf_int


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
def mw_conf_int_dir(data_dir):
    return data_dir.joinpath("mannwhitney_conf_int")


@pytest.fixture
def test_pos(mw_conf_int_dir):
    return pd.read_csv(mw_conf_int_dir.joinpath("posval.csv.gz"), index_col=0).squeeze("columns")


@pytest.fixture
def test_neg(mw_conf_int_dir):
    return pd.read_csv(mw_conf_int_dir.joinpath("negval.csv.gz"), index_col=0).squeeze("columns")


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
