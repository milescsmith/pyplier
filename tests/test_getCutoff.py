import importlib.resources as ir

import pandas as pd
import pytest

from pyplier.getCutoff import getCutoff
from pyplier.PLIERRes import PLIERResults


@pytest.fixture
def AUC_results():
    summary_file = ir.files("tests").joinpath("data", "getAUC", "getAUC_summary.csv.gz")
    uauc_file = ir.files("tests").joinpath("data", "getAUC", "getAUC_uauc.csv.gz")
    upval_file = ir.files("tests").joinpath("data", "getAUC", "getAUC_up.csv.gz")

    with ir.as_file(summary_file) as sf, ir.as_file(uauc_file) as uaf, ir.as_file(
        upval_file
    ) as upf:
        summary_df = pd.read_csv(sf, index_col=0)
        uauc_df = pd.read_csv(uaf, index_col=0)
        upval_df = pd.read_csv(upf, index_col=0)
    return {"Uauc": uauc_df, "Upval": upval_df, "summary": summary_df}


def test_getAUC(AUC_results):
    __tracebackhide__ = True

    cutoff = getCutoff(AUC_results, 0.01)

    assert cutoff == pytest.approx(0.0090271002031886)
