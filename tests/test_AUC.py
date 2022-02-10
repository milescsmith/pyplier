from sys import version_info

if version_info[1] == 8:
    import importlib_resources as ir
elif version_info[1] >= 9:
    import importlib.resources as ir

import pandas as pd
import pytest
from joblib import load

from pyplier.AUC import AUC


@pytest.fixture
def test_labels():
    labels_file = ir.files("tests").joinpath("data", "AUC", "labels.csv.gz")
    with ir.as_file(labels_file) as lf:
        labels_df = pd.read_csv(lf, index_col=0)
    return labels_df


@pytest.fixture
def test_values():
    values_file = ir.files("tests").joinpath("data", "AUC", "values.csv.gz")
    with ir.as_file(values_file) as vf:
        values_df = pd.read_csv(vf, index_col=0)
    return values_df


@pytest.fixture
def expected_AUC():
    expected_file = ir.files("tests").joinpath("data", "AUC", "auc_test_result.pkl")
    with ir.as_file(expected_file) as ef:
        expected_res = load(ef)
    return expected_res


# @pytest.mark.auc
def test_AUC(test_labels, test_values, expected_AUC):
    __tracebackhide__ = True
    test_res = AUC(test_labels.iloc[:, 0], test_values.iloc[:, 0])

    assert test_res == expected_AUC
