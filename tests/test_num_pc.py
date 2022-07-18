import importlib.resources as ir

import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings

from pyplier.num_pc import num_pc


@pytest.fixture
def test_100_mat() -> np.ndarray:
    mat_file = ir.files("tests").joinpath("data/num_pc/test_pc_mat_100.csv.gz")
    with ir.as_file(mat_file) as m100:
        mat_100 = np.loadtxt(m100, delimiter=",")

    return mat_100


@pytest.fixture
def test_1000_mat() -> np.ndarray:
    mat_file = ir.files("tests").joinpath("data/num_pc/test_pc_mat_1000.csv.gz")
    with ir.as_file(mat_file) as m1000:
        mat_1000 = np.loadtxt(m1000, delimiter=",")

    return mat_1000


def test_num_pc(test_100_mat, test_1000_mat):
    res_100 = num_pc(data=test_100_mat, method="elbow")

    res_1000 = num_pc(data=test_1000_mat, method="elbow")

    assert res_100 == pytest.approx(9.0)
    assert res_1000 == pytest.approx(10.0)


@settings(deadline=None)
@given(
    npst.arrays(
        float,
        npst.array_shapes(min_dims=2, max_dims=2, min_side=3),
        elements=st.floats(2, 100),
    )
)
def test_hypothesis_num_pc(x):
    num_pc(data=x, method="elbow")
