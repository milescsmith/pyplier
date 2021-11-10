import importlib.resources

import pandas as pd
import pytest

from pyplier.getAUC import getAUC


@pytest.fixture
def test_plierRes():
    heldOutGenes_file = importlib.resources.files("tests").joinpath(
        "data/common/heldOutGenes.csv"
    )
    B_file = importlib.resources.files("tests").joinpath("data/common/plierRes_b.csv")
    C_file = importlib.resources.files("tests").joinpath("data/common/plierRes_c.csv")
    residual_file = importlib.resources.files("tests").joinpath(
        "data/common/plierRes_residual.csv"
    )
    U_file = importlib.resources.files("tests").joinpath("data/common/plierRes_u.csv")
    Z_file = importlib.resources.files("tests").joinpath("data/common/plierRes_z.csv")
    L1 = 18.5633
    L2 = 37.12661
    L3 = 0.0005530844

    with importlib.resources.as_file(
        heldOutGenes_file
    ) as hogf, importlib.resources.as_file(B_file) as bf, importlib.resources.as_file(
        C_file
    ) as cf, importlib.resources.as_file(
        residual_file
    ) as resf, importlib.resources.as_file(
        U_file
    ) as uf, importlib.resources.as_file(
        Z_file
    ) as zf:
        b_df = pd.read_csv(bf, index_col=0)
        c_df = pd.read_csv(cf, index_col=0)
        res_df = pd.read_csv(resf, index_col=0)
        u_df = pd.read_csv(uf, index_col=0)
        z_df = pd.read_csv(zf, index_col=0)
        hog_df = pd.read_csv(hogf, index_col=0)

    plierRes = {
        "residual": res_df,
        "B": b_df,
        "C": c_df,
        "U": u_df,
        "Z": z_df,
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "heldOutGenes": {k: g["value"].tolist() for k, g in hog_df.groupby("name")},
    }
    return plierRes


@pytest.fixture
def test_data():
    data_file = importlib.resources.files("tests").joinpath("data/common/data.csv")
    with importlib.resources.as_file(data_file) as df:
        data_df = pd.read_csv(df, index_col=0)
    return data_df


@pytest.fixture
def test_priorMat():
    priorMat_file = importlib.resources.files("tests").joinpath(
        "data/common/priorMat.csv"
    )
    with importlib.resources.as_file(priorMat_file) as pmf:
        priorMat_df = pd.read_csv(pmf, index_col=0)
    return priorMat_df


@pytest.fixture
def expected_AUC():
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


# @pytest.mark.auc
def test_getAUC(test_plierRes, test_data, test_priorMat, expected_AUC):
    __tracebackhide__ = True
    test_res = getAUC(plierRes=test_plierRes, data=test_data, priorMat=test_priorMat)

    pd.testing.assert_frame_equal(test_res["Uauc"], expected_AUC["Uauc"])
    pd.testing.assert_frame_equal(test_res["Upval"], expected_AUC["Upval"])
    pd.testing.assert_frame_equal(test_res["summary"], expected_AUC["summary"])
