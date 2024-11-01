# from sys import breakpoint
import pandas as pd
import pytest
from pyplier.name_b import name_b
from pyplier.plier_res import PLIERResults


@pytest.fixture
def pre_name_b_plier_res(data_dir, test_plier_res):
    prior_names = pd.read_csv(data_dir.joinpath("nameB", "plier_res_b_pre-name_b.csv.gz"), index_col=0)
    test_plier_res.b = prior_names
    return test_plier_res


@pytest.fixture
def prior_names(common_dir):
    return pd.read_csv(common_dir.joinpath("plier_res_b.csv.gz"), index_col=0).index.to_list()


def test_name_b(test_plier_res, prior_names: list[str]) -> None:
    new_names = name_b(test_plier_res)
    assert new_names == prior_names
