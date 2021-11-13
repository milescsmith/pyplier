import importlib.resources

import pandas as pd
import pytest

from pyplier.getCutoff import getCutoff


@pytest.fixture
def AUC_results():
    summary_file = importlib.resources.files("tests").joinpath(
        "data/getAUC/aucresults_summary.csv"
    )
    uauc_file = importlib.resources.files("tests").joinpath(
        "data/getAUC/aucresults_uauc.csv"
    )
    upval_file = importlib.resources.files("tests").joinpath(
        "data/getAUC/aucresults_upval.csv"
    )

    with importlib.resources.as_file(summary_file) as sf, importlib.resources.as_file(
        uauc_file
    ) as uaf, importlib.resources.as_file(upval_file) as upf:
        summary_df = pd.read_csv(sf, index_col=0)
        uauc_df = pd.read_csv(uaf, index_col=0)
        upval_df = pd.read_csv(upf, index_col=0)
    return {"Uauc": uauc_df, "Upval": upval_df, "summary": summary_df}


def test_getAUC(AUC_results):
    __tracebackhide__ = True

    cutoff = getCutoff(AUC_results, 0.01)

    assert cutoff == 0.0012647771071858
