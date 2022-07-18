import importlib.resources as ir

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import array_shapes, arrays

from pyplier.colSumNorm import colSumNorm


@pytest.fixture
def test_arr():
    arr_file = ir.files("tests").joinpath("data", "colSumNorm", "test_arr.csv.gz")
    with ir.as_file(arr_file) as ar:
        arr = np.loadtxt(ar, delimiter=",")
    return arr


@pytest.fixture
def expected_res():
    res_file = ir.files("tests").joinpath("data", "colSumNorm", "res_mat.csv.gz")
    with ir.as_file(res_file) as rf:
        res = np.loadtxt(rf, delimiter=",")
    return res


@pytest.fixture
def expected_res_ss():
    res_ss_file = ir.files("tests").joinpath("data", "colSumNorm", "res_ss.csv.gz")
    with ir.as_file(res_ss_file) as rsf:
        res_ss = np.loadtxt(rsf, delimiter=",")
    return res_ss


@pytest.fixture
def expected_res_full(expected_res, expected_res_ss):
    return {"mat": expected_res, "ss": expected_res_ss}


def test_colSumNorm(test_arr, expected_res):
    res = colSumNorm(mat=test_arr, return_all=False)
    np.testing.assert_equal(res, expected_res)


def test_colSumNorm_full(test_arr, expected_res_full):
    res = colSumNorm(mat=test_arr, return_all=True)
    np.testing.assert_equal(res["mat"], expected_res_full["mat"])
    np.testing.assert_equal(res["ss"], expected_res_full["ss"])


@given(arrays(np.float32, array_shapes(min_dims=2, min_side=2, max_dims=2)))
@settings(deadline=None)
def test_colSumNorm_hypothesis(arr):
    colSumNorm(mat=arr, return_all=True)
