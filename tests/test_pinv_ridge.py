import pandas as pd
import pytest
from pyplier.regression import pinv_ridge


@pytest.fixture
def pinv_dir(data_dir):
    return data_dir.joinpath("pinv_ridge")


@pytest.fixture
def input_data(pinv_dir):
    return pd.read_csv(pinv_dir.joinpath("input_matrix.csv.gz"), index_col=0)


@pytest.fixture
def expected_results(pinv_dir):
    return pd.read_csv(pinv_dir.joinpath("expected_results.csv.gz"), index_col=0)


def test_nameB(input_data, expected_results):
    pinv_ridge_results = pinv_ridge(input_data)

    pd.testing.assert_frame_equal(pinv_ridge_results, expected_results)
