from sys import version_info

if version_info[1] == 8:
    import importlib_resources as ir
elif version_info[1] >= 9:
    import importlib.resources as ir

import pandas as pd
import pytest

from pyplier.stubs import PLIERResults


@pytest.fixture
def plierRes() -> PLIERResults:
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
            residual= pd.read_csv(resf, index_col=0),
            B = pd.read_csv(bf, index_col=0),
            Z = pd.read_csv(zf, index_col=0),
            U = pd.read_csv(uf, index_col=0),
            C = pd.read_csv(cf, index_col=0),
            L1 = L1,
            L2 = L2,
            L3 = L3,
            heldOutGenes = {
                k: g["value"].tolist()
                for k, g in pd.read_csv(hogf, index_col=0).groupby("name")
            },
            withPrior = withPrior,
            Uauc = pd.read_csv(uacf, index_col=0),
            Up = pd.read_csv(upf, index_col=0),
            summary = pd.read_csv(sf, index_col=0),
        )
    return plierRes