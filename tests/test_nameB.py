import importlib.resources as ir

import pandas as pd
import pytest

from pyplier.name_b import nameB
from pyplier.plier_res import PLIERResults


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
            b=pd.read_csv(bf, index_col=0),
            z=pd.read_csv(zf, index_col=0),
            u=pd.read_csv(uf, index_col=0),
            c=pd.read_csv(cf, index_col=0),
            l1=L1,
            l2=L2,
            l3=L3,
            held_out_genes={
                k: g["value"].tolist()
                for k, g in pd.read_csv(hogf, index_col=0).groupby("name")
            },
            with_prior=withPrior,
            uauc=pd.read_csv(uacf, index_col=0),
            up=pd.read_csv(upf, index_col=0),
            summary=pd.read_csv(sf, index_col=0),
        )
    return plierRes


@pytest.fixture
def prenameB_plierRes(test_plierRes: PLIERResults):
    prenameB_file = ir.files("tests").joinpath("data/nameB/plierRes_b_pre-nameB.csv.gz")
    with ir.as_file(prenameB_file) as nf:
        prior_names = pd.read_csv(nf, index_col=0)

    test_plierRes.b = prior_names
    return test_plierRes


@pytest.fixture
def names():
    names_file = ir.files("tests").joinpath("data", "common", "plierRes_b.csv.gz")
    with ir.as_file(names_file) as nf:
        prior_names = list(pd.read_csv(nf, index_col=0).index)
    return prior_names


def test_nameB(test_plierRes: PLIERResults, names: list[str]) -> None:
    new_names = nameB(test_plierRes)

    assert new_names == names
