import gzip
import importlib.resources as ir
from json import load

import pandas as pd
import pytest
from pyplier.auc import auc


@pytest.fixture
def auc_dir(data_dir):
    return data_dir.joinpath("auc")


@pytest.fixture
def test_labels(auc_dir):
    labels_file = auc_dir.joinpath("labels.csv.gz")
    labels_df = pd.read_csv(labels_file, index_col=0).squeeze("columns")
    return labels_df


@pytest.fixture
def test_values(auc_dir):
    values_file = auc_dir.joinpath("values.csv.gz")
    values_df = pd.read_csv(values_file, index_col=0).squeeze("columns")
    return values_df


@pytest.fixture
def expected_auc(auc_dir):
    expected_file = auc_dir.joinpath("auc_test_result.json.gz")
    with ir.as_file(expected_file) as ef:
        expected_res = load(gzip.open(ef))
    return expected_res


# @pytest.mark.auc
def test_auc(test_labels, test_values, expected_auc):
    __tracebackhide__ = True
    test_res = auc(test_labels, test_values)

    assert test_res == expected_auc  # noqa: S101
