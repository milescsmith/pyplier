import pandas as pd
import pytest

from pyplier.auc import get_auc
from pyplier.plier_res import PLIERResults


@pytest.fixture
def get_auc_dir(data_dir):
    return data_dir.joinpath("get_auc")


@pytest.fixture
def b_df(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_b.csv.gz"), index_col=0)


@pytest.fixture
def c_df(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_c.csv.gz"), index_col=0)


@pytest.fixture
def residual_df(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_residual.csv.gz"), index_col=0)


@pytest.fixture
def u_df(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_u.csv.gz"), index_col=0)


@pytest.fixture
def z_df(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_z.csv.gz"), index_col=0)


@pytest.fixture
def held_out_genes():
    return []


@pytest.fixture
def test_plier_res(b_df, c_df, residual_df, u_df, z_df, l1, l2, l3, held_out_genes):
    return PLIERResults(
        residual=residual_df,
        b=b_df,
        c=c_df,
        u=u_df,
        z=z_df,
        l1=l1,
        l2=l2,
        l3=l3,
        held_out_genes=held_out_genes,
    )


@pytest.fixture
def test_data(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_data.csv.gz"), index_col=0)


@pytest.fixture
def test_prior_mat(get_auc_dir):
    return pd.read_csv(get_auc_dir.joinpath("get_auc_priormat.csv.gz"), index_col=0)


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
def expected_auc(summary_df, uauc_df, upval_df):
    return {"uauc": uauc_df, "upval": upval_df, "summary": summary_df}


# @pytest.mark.auc
def test_get_auc(test_plier_res, test_data, test_prior_mat, expected_auc):
    __tracebackhide__ = True
    test_res = get_auc(plierRes=test_plier_res, data=test_data, priorMat=test_prior_mat)

    pd.testing.assert_frame_equal(test_res["uauc"], expected_auc["uauc"])
    pd.testing.assert_frame_equal(test_res["upval"], expected_auc["upval"])
    pd.testing.assert_frame_equal(test_res["summary"], expected_auc["summary"])
