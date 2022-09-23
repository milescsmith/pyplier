import gzip
import importlib.resources as ir
from json import load

import pandas as pd
import pytest

from pyplier.auc import AUC


@pytest.fixture
def test_labels():
    labels_file = ir.files("tests").joinpath("data", "AUC", "labels.csv.gz")
    with ir.as_file(labels_file) as lf:
        labels_df = pd.read_csv(lf, index_col=0).squeeze("columns")
    return labels_df


@pytest.fixture
def test_values():
    values_file = ir.files("tests").joinpath("data", "AUC", "values.csv.gz")
    with ir.as_file(values_file) as vf:
        values_df = pd.read_csv(vf, index_col=0).squeeze("columns")
    return values_df


@pytest.fixture
def expected_AUC():
    expected_file = ir.files("tests").joinpath("data", "AUC", "auc_test_result.json.gz")
    with ir.as_file(expected_file) as ef:
        expected_res = load(gzip.open(ef))
    return expected_res


# @pytest.mark.auc
def test_AUC(test_labels, test_values, expected_AUC):
    __tracebackhide__ = True
    test_res = AUC(test_labels, test_values)

    assert test_res == expected_AUC
