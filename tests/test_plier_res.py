import importlib.resources as ir
from typing import Any
import numpy as np
import pytest
from deepdiff import DeepDiff
from dill import load
from pyplier.plier_res import PLIERResults


@pytest.fixture
def plier_res_dir(data_dir):
    return data_dir.joinpath("plier_res")


@pytest.fixture
def pickled_dict(plier_res_dir) -> dict[str, Any]:
    dict_file = plier_res_dir.joinpath("plier_res_dict.pkl")
    with dict_file.open("rb") as df:
        pdt = load(df)
    return pdt


@pytest.fixture
def plier_res_json(plier_res_dir):
    test_file = plier_res_dir.joinpath("plier_res.json.gz")
    tf_obj = PLIERResults().read_json(test_file)
    return tf_obj


@pytest.fixture
def plier_res_h5(plier_res_dir):
    test_file = plier_res_dir.joinpath("plier_res.h5")
    tf_obj = PLIERResults().read_hdf5(test_file)
    # tf_obj.summary = tf_obj.summary.astype({"LV index": np.float64}).astype({"LV index": np.int64}).astype({"LV index": str, "AUC": float, "p-value": float, "FDR": float})
    return tf_obj


def test_plier_res_repr(test_plier_res: PLIERResults) -> None:
    assert test_plier_res.__repr__() == (
        "b: 30 rows x 36 columns\n"
        "z: 5892 rows x 30 columns\n"
        "u: 606 rows x 30 columns\n"
        "c: 5892 rows x 606 columns\n"
        "uauc: 606 rows x 30 columns\n"
        "up: 606 rows x 30 columns\n"
        "summary: 64 rows x 4 columns\n"
        "residual: 5892 rows x 36 columns\n"
        "held_out_genes: 603\n"
        "with_prior: 21\n"
        "l1 is set to 18.4306\n"
        "l2 is set to 36.8612\n"
        "l3 is set to 0.0006"
    )


def test_plier_res_to_dict(test_plier_res: PLIERResults, pickled_dict: dict[str, Any]):
    dict_diff = DeepDiff(
        t1=test_plier_res.to_dict(),
        t2=pickled_dict,
        ignore_order=False,
        max_diffs=1,
        cache_size=5000,
    )
    assert dict_diff == {}


def test_plier_res_from_dict(test_plier_res: PLIERResults, pickled_dict: dict[str, Any]):
    assert test_plier_res == PLIERResults().from_dict(pickled_dict)


@pytest.mark.parametrize("plier_res", ["plier_res_json", "plier_res_h5"])
def test_plier_res_from_disk(test_plier_res, plier_res, request):
    plier_res = request.getfixturevalue(plier_res)
    assert test_plier_res == plier_res
