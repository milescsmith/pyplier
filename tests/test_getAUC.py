import importlib.resources as ir

import pandas as pd
import pytest

from pyplier.getAUC import getAUC
from pyplier.PLIERRes import PLIERResults


@pytest.fixture
def test_plierRes():

    B_file = ir.files("tests").joinpath("data", "getAUC", "getAUC_b.csv.gz")
    C_file = ir.files("tests").joinpath("data", "getAUC", "getAUC_c.csv.gz")
    residual_file = ir.files("tests").joinpath(
        "data", "getAUC", "getAUC_residual.csv.gz"
    )
    U_file = ir.files("tests").joinpath("data", "getAUC", "getAUC_u.csv.gz")
    Z_file = ir.files("tests").joinpath("data", "getAUC", "getAUC_z.csv.gz")
    L1 = 18.43058
    L2 = 36.86117
    L3 = 0.0005530844

    with ir.as_file(B_file) as bf, ir.as_file(C_file) as cf, ir.as_file(
        residual_file
    ) as resf, ir.as_file(U_file) as uf, ir.as_file(Z_file) as zf:
        b_df = pd.read_csv(bf, index_col=0)
        c_df = pd.read_csv(cf, index_col=0)
        res_df = pd.read_csv(resf, index_col=0)
        u_df = pd.read_csv(uf, index_col=0)
        z_df = pd.read_csv(zf, index_col=0)

    plierRes = PLIERResults(
        residual=res_df,
        B=b_df,
        C=c_df,
        U=u_df,
        Z=z_df,
        L1=L1,
        L2=L2,
        L3=L3,
        heldOutGenes=list(),
    )
    return plierRes


@pytest.fixture
def test_data():
    data_file = ir.files("tests").joinpath("data", "getAUC", "getAUC_data.csv.gz")
    with ir.as_file(data_file) as df:
        data_df = pd.read_csv(df, index_col=0)
    return data_df


@pytest.fixture
def test_priorMat():
    priorMat_file = ir.files("tests").joinpath(
        "data", "getAUC", "getAUC_priormat.csv.gz"
    )
    with ir.as_file(priorMat_file) as pmf:
        priorMat_df = pd.read_csv(pmf, index_col=0)
    return priorMat_df


@pytest.fixture
def expected_AUC():
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


# @pytest.mark.auc
def test_getAUC(test_plierRes, test_data, test_priorMat, expected_AUC):
    __tracebackhide__ = True
    test_res = getAUC(plierRes=test_plierRes, data=test_data, priorMat=test_priorMat)

    pd.testing.assert_frame_equal(test_res["Uauc"], expected_AUC["Uauc"])
    pd.testing.assert_frame_equal(test_res["Upval"], expected_AUC["Upval"])
    pd.testing.assert_frame_equal(test_res["summary"], expected_AUC["summary"])
