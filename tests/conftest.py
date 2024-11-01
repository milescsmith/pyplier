import importlib.resources as ir

import pandas as pd
import numpy as np
import pytest
from pyplier.plier_res import PLIERResults


@pytest.fixture
def data_dir():
    return ir.files("tests").joinpath("data")


@pytest.fixture
def common_dir(data_dir):
    return data_dir.joinpath("common")


@pytest.fixture
def l1():
    return 18.43058


@pytest.fixture
def l2():
    return 36.86117


@pytest.fixture
def l3():
    return 0.0005530844


@pytest.fixture
def held_out_genes_df(common_dir):
    return pd.read_csv(common_dir.joinpath("plier_res_heldoutgenes.csv.gz"), index_col=0)


@pytest.fixture
def b_df(common_dir):
    return pd.read_csv(common_dir.joinpath("plier_res_b.csv.gz"), index_col=0).astype(np.float64)


@pytest.fixture
def c_df(common_dir):
    return pd.read_csv(common_dir.joinpath("plier_res_c.csv.gz"), index_col=0).astype(np.int64)


@pytest.fixture
def residual_df(common_dir):
    return pd.read_csv(common_dir.joinpath("plier_res_residual.csv.gz"), index_col=0).astype(np.float64)


@pytest.fixture
def u_df(common_dir):
    return pd.read_csv(common_dir.joinpath("plier_res_u.csv.gz"), index_col=0).astype(np.float64)


@pytest.fixture
def z_df(common_dir):
    return pd.read_csv(common_dir.joinpath("plier_res_z.csv.gz"), index_col=0).astype(np.float64)


@pytest.fixture
def uauc_df(common_dir):
    return pd.read_csv(common_dir.joinpath("plier_res_uauc.csv.gz"), index_col=0).astype(np.float64)


@pytest.fixture
def upval_df(common_dir):
    return pd.read_csv(common_dir.joinpath("plier_res_up.csv.gz"), index_col=0).astype(np.float64)


@pytest.fixture
def summary_df(common_dir):
    return pd.read_csv(
        common_dir.joinpath("plier_res_summary.csv.gz"),
        index_col=0,
        dtype={"LV index": str, "AUC": float, "p-value": float, "FDR": float},
    )


@pytest.fixture
def test_plier_res(
    held_out_genes_df,
    b_df,
    c_df,
    residual_df,
    u_df,
    z_df,
    l1,
    l2,
    l3,
    uauc_df,
    upval_df,
    summary_df,
) -> PLIERResults:
    with_prior = {
        "LV1": 1,
        "LV2": 2,
        "LV3": 3,
        "LV4": 4,
        "LV5": 5,
        "LV6": 6,
        "LV7": 7,
        "LV8": 8,
        "LV9": 9,
        "LV10": 10,
        "LV11": 11,
        "LV14": 14,
        "LV15": 15,
        "LV18": 18,
        "LV20": 20,
        "LV23": 23,
        "LV24": 24,
        "LV26": 26,
        "LV27": 27,
        "LV29": 29,
        "LV30": 30,
    }

    return PLIERResults(
        residual=residual_df,
        b=b_df,
        z=z_df,
        u=u_df,
        c=c_df,
        l1=l1,
        l2=l2,
        l3=l3,
        held_out_genes={k: g["value"].to_list() for k, g in held_out_genes_df.groupby("name")},
        with_prior=with_prior,
        uauc=uauc_df,
        up=upval_df,
        summary=summary_df.astype({"LV index": np.float64})
        .astype({"LV index": np.int64})
        .astype({"LV index": str, "AUC": float, "p-value": float, "FDR": float}),
    )
