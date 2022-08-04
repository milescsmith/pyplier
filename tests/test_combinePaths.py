import importlib.resources as ir

import pandas as pd
import pytest

from pyplier.combinePaths import combinePaths


@pytest.fixture
def blood():
    blood_file = ir.files("tests").joinpath(
        "data", "combinePaths", "bloodCellMarkersIRISDMAP.csv.gz"
    )
    with ir.as_file(blood_file) as rf:
        blood_df = pd.read_csv(rf, index_col=0)

    return blood_df


@pytest.fixture
def canonical():
    canon_file = ir.files("tests").joinpath(
        "data", "combinePaths", "canonicalPathways.csv.gz"
    )
    with ir.as_file(canon_file) as cf:
        canon_df = pd.read_csv(cf, index_col=0)

    return canon_df


@pytest.fixture
def expected_allPaths():
    allPaths_file = ir.files("tests").joinpath(
        "data", "combinePaths", "allPaths.csv.gz"
    )
    with ir.as_file(allPaths_file) as cf:
        allPaths_df = pd.read_csv(cf, index_col=0)

    return allPaths_df


def test_combinePaths(
    blood: pd.DataFrame, canonical: pd.DataFrame, expected_allPaths: pd.DataFrame
):
    allPaths = combinePaths(blood, canonical)
    pd.testing.assert_frame_equal(allPaths, expected_allPaths)
