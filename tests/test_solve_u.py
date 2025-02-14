import numpy as np
import pandas as pd
import pytest
from pytest import approx

from pyplier.solve_u import solve_u


@pytest.fixture
def solve_u_dir(data_dir):
    return data_dir.joinpath("solve_u")


@pytest.fixture
def test_z(solve_u_dir) -> pd.DataFrame:
    return pd.read_csv(solve_u_dir.joinpath("z.csv.gz"), index_col="gene")


@pytest.fixture
def test_chat(solve_u_dir) -> pd.DataFrame:
    return pd.read_csv(solve_u_dir.joinpath("chat.csv.gz"), index_col="pathway")


@pytest.fixture
def test_prior_mat(solve_u_dir) -> pd.DataFrame:
    prior_mat = pd.read_csv(solve_u_dir.joinpath("prior_mat.csv.gz"), index_col="gene")
    prior_mat.columns.name = "pathway"
    return prior_mat


@pytest.fixture
def test_penalty_factor(solve_u_dir) -> np.ndarray:
    return np.loadtxt(solve_u_dir.joinpath("penalty_factor.csv.gz"))


@pytest.fixture
def expected_u_complete(solve_u_dir) -> pd.DataFrame:
    return pd.read_csv(solve_u_dir.joinpath("u_complete.csv.gz"), index_col="pathway").astype(np.float64)


@pytest.fixture
def expected_u_fast(solve_u_dir) -> pd.DataFrame:
    return pd.read_csv(solve_u_dir.joinpath("u_fast.csv.gz"), index_col="pathway").astype(np.float64)


@pytest.mark.parametrize(
    "pathway_selection, expected_u",
    [("complete", "expected_u_complete"), ("fast", "expected_u_fast")],
)
def test_solve_u(
    test_z,
    test_chat,
    test_prior_mat,
    test_penalty_factor,
    pathway_selection,
    expected_u,
    request,
):
    u_result = solve_u(
        z=test_z,
        chat=test_chat,
        prior_mat=test_prior_mat,
        penalty_factor=test_penalty_factor,
        pathway_selection=pathway_selection,
        glm_alpha=0.9,
        max_path=10,
        target_frac=0.7,
        l3=None,
    )
    expected_u = request.getfixturevalue(expected_u)
    pd.testing.assert_frame_equal(u_result["u"], expected_u)
    assert u_result["l3"] == approx(5.144486e-05)
