import importlib.resources as ir
from typing import Any

import pandas as pd
import pytest
from deepdiff import DeepDiff
from dill import load

from pyplier.PLIERRes import PLIERResults


@pytest.fixture
def test_plierRes() -> PLIERResults:
    heldOutGenes_file = ir.files("tests").joinpath(
        "data", "common", "plierRes_heldoutgenes.csv.gz"
    )
    B_file = ir.files("tests").joinpath("data", "common", "plierRes_b.csv.gz")
    C_file = ir.files("tests").joinpath("data", "common", "plierRes_c.csv.gz")
    residual_file = ir.files("tests").joinpath(
        "data", "common", "plierRes_residual.csv.gz"
    )
    U_file = ir.files("tests").joinpath("data", "common", "plierRes_u.csv.gz")
    Z_file = ir.files("tests").joinpath("data", "common", "plierRes_z.csv.gz")

    Uauc_file = ir.files("tests").joinpath("data", "common", "plierRes_uauc.csv.gz")
    Upval_file = ir.files("tests").joinpath("data", "common", "plierRes_up.csv.gz")
    summary_file = ir.files("tests").joinpath(
        "data", "common", "plierRes_summary.csv.gz"
    )

    L1 = 18.43058
    L2 = 36.86117
    L3 = 0.0004307425
    withPrior = {
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

    with ir.as_file(heldOutGenes_file) as hogf, ir.as_file(B_file) as bf, ir.as_file(
        C_file
    ) as cf, ir.as_file(residual_file) as resf, ir.as_file(U_file) as uf, ir.as_file(
        Z_file
    ) as zf, ir.as_file(
        Uauc_file
    ) as uacf, ir.as_file(
        Upval_file
    ) as upf, ir.as_file(
        summary_file
    ) as sf:

        plierRes = PLIERResults(
            residual=pd.read_csv(resf, index_col=0),
            B=pd.read_csv(bf, index_col=0),
            Z=pd.read_csv(zf, index_col=0),
            U=pd.read_csv(uf, index_col=0),
            C=pd.read_csv(cf, index_col=0),
            L1=L1,
            L2=L2,
            L3=L3,
            heldOutGenes={
                k: g["value"].tolist()
                for k, g in pd.read_csv(hogf, index_col=0).groupby("name")
            },
            withPrior=withPrior,
            Uauc=pd.read_csv(uacf, index_col=0),
            Up=pd.read_csv(upf, index_col=0),
            summary=pd.read_csv(sf, index_col=0),
        )
        return plierRes


def test_plierRes_repr(test_plierRes: PLIERResults) -> None:
    assert test_plierRes.__repr__() == (
        "B : 30 rows x 36 columns\n"
        "Z : 5892 rows x 30 columns\n"
        "U : 606 rows x 30 columns\n"
        "C : 5892 rows x 606 columns\n"
        "heldOutGenes: 603\n"
        "withPrior: 21\n"
        "Uauc: 606 rows x 30 columns\n"
        "Up: 606 rows x 30 columns\n"
        "summary: 64 rows x 4 columns\n"
        "residual: 5892 rows x 36 columns\n"
        "L1 is set to 18.4306\n"
        "L2 is set to 36.8612\n"
        "L3 is set to 0.0004"
    )


@pytest.fixture
def pickled_dict() -> dict[str, Any]:
    dict_file = ir.files("tests").joinpath("data/plierRes/plierRes_dict.pkl")
    with ir.as_file(dict_file) as df:
        pdt = load(open(df, "rb"))
    return pdt


def test_plierRes_to_dict(test_plierRes: PLIERResults, pickled_dict: dict[str, Any]):
    dict_diff = DeepDiff(
        t1=test_plierRes.to_dict(),
        t2=pickled_dict,
        ignore_order=False,
        max_diffs=1,
        cache_size=5000,
    )
    assert dict_diff == {}


def test_plierRes_from_dict(test_plierRes: PLIERResults, pickled_dict: dict[str, Any]):
    assert test_plierRes == PLIERResults().from_dict(pickled_dict)


def test_plierRes_to_disk():
    pass


def test_plierRes_from_disk(test_plierRes: PLIERResults):
    test_file = ir.files("tests").joinpath("data/plierRes/plierRes.json.gz")
    with ir.as_file(test_file) as tf:
        tf_obj = PLIERResults().from_disk(tf)
    assert test_plierRes == tf_obj
