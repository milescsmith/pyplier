import importlib.resources

import pandas as pd
import pytest

from pyplier.pinv_ridge import pinv_ridge


@pytest.fixture
def input_data():
    input_file = importlib.resources.files("tests").joinpath(
        "data", "pinv_ridge", "input_matrix.csv"
    )
    with importlib.resources.as_file(input_file) as inpf:
        input_data = pd.read_csv(inpf, index_col=0)

    return input_data


@pytest.fixture
def expected_results():
    expected_file = importlib.resources.files("tests").joinpath(
        "data", "pinv_ridge", "expected_results.csv"
    )
    with importlib.resources.as_file(expected_file) as ef:
        expected_results = pd.read_csv(ef, index_col=0)

    return expected_results


def test_nameB(input_data, expected_results):
    pinv_ridge_results = pinv_ridge(input_data)

    pd.testing.assert_frame_equal(pinv_ridge_results, expected_results)
