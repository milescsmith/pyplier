# TODO: we should be able to handle other file types (plain text, JSON, grp, whatelse?)
# TODO: should also be able to just download pathways
from pathlib import Path

import pandas as pd


def pathwayFromGMT(gmt_file: Path) -> pd.DataFrame:
    with gmt_file.open("r") as gf:
        gmt = gf.readlines()

    gmt_long = (
        pd.DataFrame.from_dict(
            data=[
                {"pathway": y[0], "url": y[1], "genes": y[2].split("\t")}
                for y in [x.split("\t", maxsplit=2) for x in gmt]
            ]
        )
        .drop(columns=["url"])
        .explode(column="genes")
        .assign(genes=lambda x: x["genes"].str.strip(), count=1)
    )
    gmt_wide = pd.pivot_table(
        gmt_long, values="count", index="genes", columns="pathway", fill_value=0
    ).rename_axis(columns=None)

    return gmt_wide
