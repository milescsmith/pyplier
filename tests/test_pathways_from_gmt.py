import pandas as pd
import pytest
from pyplier.pathways import pathway_from_gmt
from pyplier.utils import fix_dataframe_dtypes


@pytest.fixture
def pathway_dir(data_dir):
    return data_dir.joinpath("pathway_from_gmt")


@pytest.fixture
def expected_df(pathway_dir):
    df = pd.read_csv(pathway_dir.joinpath("reactome_wide.csv.gz"), index_col=0)
    df = fix_dataframe_dtypes(df)
    return df


@pytest.fixture
def reactome_gene_matrix(pathway_dir):
    return pathway_dir.joinpath("c2.cp.reactome.v7.4.symbols.gmt")


def test_pathway_from_gmt(reactome_gene_matrix, expected_df):
    reactome_pathways = pathway_from_gmt(reactome_gene_matrix)

    pd.testing.assert_frame_equal(reactome_pathways, expected_df)
