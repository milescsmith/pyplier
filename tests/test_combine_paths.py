import pandas as pd
import pytest
from pyplier.pathways import combine_paths


@pytest.fixture
def combine_paths_dir(data_dir):
    return data_dir.joinpath("combine_paths")


@pytest.fixture
def blood_df(combine_paths_dir):
    return pd.read_csv(combine_paths_dir.joinpath("blood_cell_markers_IRISDMAP.csv.gz"), index_col=0)


@pytest.fixture
def canonical_df(combine_paths_dir):
    return pd.read_csv(combine_paths_dir.joinpath("canonical_pathways.csv.gz"), index_col=0)


@pytest.fixture
def expected_all_paths_df(combine_paths_dir):
    return pd.read_csv(combine_paths_dir.joinpath("all_paths.csv.gz"), index_col=0)


def test_combine_paths(blood_df: pd.DataFrame, canonical_df: pd.DataFrame, expected_all_paths_df: pd.DataFrame):
    all_paths = combine_paths(blood_df, canonical_df)
    pd.testing.assert_frame_equal(all_paths, expected_all_paths_df)
