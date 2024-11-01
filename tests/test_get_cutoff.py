import pandas as pd
import pytest

# from pyplier.plier_res import PLIERResults
from pyplier.utils import get_cutoff


@pytest.fixture
def get_auc_dir(data_dir):
    return data_dir.joinpath("get_auc")


@pytest.fixture
def summary_df(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_summary.csv.gz"), index_col=0)


@pytest.fixture
def uauc_df(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_uauc.csv.gz"), index_col=0)


@pytest.fixture
def upval_df(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_up.csv.gz"), index_col=0)


@pytest.fixture
def auc_results(uauc_df, upval_df, summary_df):
    return {"Uauc": uauc_df, "Upval": upval_df, "summary": summary_df}


def test_getCutoff(auc_results):
    __tracebackhide__ = True

    cutoff = get_cutoff(auc_results, 0.01)

    assert cutoff == pytest.approx(0.0090271002031886)  # noqa: S101
