import importlib.resources

import pandas as pd
import pytest

from pyplier.pathwayFromGMT import pathwayFromGMT


@pytest.fixture
def expected_df():
    reactome_file = importlib.resources.files("tests").joinpath(
        "data/pathwayFromGMT/reactome_wide.csv"
    )
    with importlib.resources.as_file(reactome_file) as rf:
        reactome_wide_df = pd.read_csv(rf, index_col=0)

    return reactome_wide_df


def test_pathwayFromGMT(expected_df):
    reactome_gmt = importlib.resources.files("tests").joinpath(
        "data/pathwayFromGMT/c2.cp.reactome.v7.4.symbols.gmt"
    )

    reactome_pathways = pathwayFromGMT(reactome_gmt)

    pd.testing.assert_frame_equal(reactome_pathways, expected_df)
