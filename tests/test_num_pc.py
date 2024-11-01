import hypothesis.extra.numpy as npst
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from pyplier.num_pc import num_pc


@pytest.fixture
def num_pc_dir(data_dir):
    return data_dir.joinpath("num_pc")


@pytest.fixture
def test_100_mat(num_pc_dir) -> np.ndarray:
    return np.loadtxt(num_pc_dir.joinpath("test_pc_mat_100.csv.gz"), delimiter=",")


@pytest.fixture
def test_1000_mat(num_pc_dir) -> np.ndarray:
    return np.loadtxt(num_pc_dir.joinpath("test_pc_mat_1000.csv.gz"), delimiter=",")


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
