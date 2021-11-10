import importlib.resources

import pandas as pd
import pytest
from joblib import load

from pyplier.AUC import AUC


@pytest.fixture
def test_labels():
    labels_file = importlib.resources.files("tests").joinpath("data/AUC/labels.csv")
    with importlib.resources.as_file(labels_file) as lf:
        labels_df = pd.read_csv(lf, index_col=0)
    return labels_df


@pytest.fixture
def test_values():
    values_file = importlib.resources.files("tests").joinpath("data/AUC/values.csv")
    with importlib.resources.as_file(values_file) as vf:
        values_df = pd.read_csv(vf, index_col=0)
    return values_df


@pytest.fixture
def expected_AUC():
    expected_file = importlib.resources.files("tests").joinpath(
        "data/AUC/auc_test_result.pkl"
    )
    with importlib.resources.as_file(expected_file) as ef:
        expected_res = load(ef)
    return expected_res


# @pytest.mark.auc
def test_AUC(test_labels, test_values, expected_AUC):
    __tracebackhide__ = True
    test_res = AUC(test_labels.iloc[:, 0], test_values.iloc[:, 0])

    assert test_res == expected_AUC
