import importlib.resources as ir

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from pyplier.solve_u import solveU


@pytest.fixture
def test_Z() -> pd.DataFrame:
    Z_file = ir.files("tests").joinpath("data", "solveU", "Z.csv.gz")
    with ir.as_file(Z_file) as Zf:
        Z = pd.read_csv(Zf, index_col="gene")
    return Z


@pytest.fixture
def test_Chat() -> pd.DataFrame:
    Chat_file = ir.files("tests").joinpath("data", "solveU", "Chat.csv.gz")
    with ir.as_file(Chat_file) as Chf:
        Chat = pd.read_csv(Chf, index_col="pathway")
    return Chat


@pytest.fixture
def test_priorMat() -> pd.DataFrame:
    pM_file = ir.files("tests").joinpath("data", "solveU", "priorMat.csv.gz")
    with ir.as_file(pM_file) as pm:
        priorMat = pd.read_csv(pm, index_col="gene")
    priorMat.columns.name = "pathway"
    return priorMat


@pytest.fixture
def test_penalty_factor() -> np.ndarray:
    penalty_factor = ir.files("tests").joinpath(
        "data", "solveU", "penalty_factor.csv.gz"
    )
    return np.loadtxt(penalty_factor)


@pytest.fixture
def expected_U_complete() -> pd.DataFrame:
    U_file = ir.files("tests").joinpath("data", "solveU", "U_complete.csv.gz")
    with ir.as_file(U_file) as uf:
        U = pd.read_csv(uf, index_col="pathway")
    U = U.astype(np.float64)
    return U


@pytest.fixture
def expected_U_fast() -> pd.DataFrame:
    U_file = ir.files("tests").joinpath("data", "solveU", "U_fast.csv.gz")
    with ir.as_file(U_file) as uf:
        U = pd.read_csv(uf, index_col="pathway")
    U = U.astype(np.float64)
    return U


def test_solveU(
    test_Z,
    test_Chat,
    test_priorMat,
    test_penalty_factor,
    expected_U_complete,
    expected_U_fast,
):
    U_complete = solveU(
        z=test_Z,
        chat=test_Chat,
        priorMat=test_priorMat,
        penalty_factor=test_penalty_factor,
        pathway_selection="complete",
        glm_alpha=0.9,
        max_path=10,
        target_frac=0.7,
        L3=None,
    )

    U_fast = solveU(
        z=test_Z,
        chat=test_Chat,
        priorMat=test_priorMat,
        penalty_factor=test_penalty_factor,
        pathway_selection="fast",
        glm_alpha=0.9,
        max_path=10,
        target_frac=0.7,
        L3=None,
    )

    pd.testing.assert_frame_equal(U_complete["U"], expected_U_complete)
    pd.testing.assert_frame_equal(U_fast["U"], expected_U_fast)
    assert U_complete["L3"] == approx(5.144486e-05)
    assert U_fast["L3"] == approx(5.144486e-05)
