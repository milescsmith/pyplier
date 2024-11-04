# TODO: we should be able to handle other file types (plain text, JSON, grp, whatelse?)
# TODO: should also be able to just download pathways
# TODO: would like a `loadPathways` function here, that could take multiple files maybe and
# automatically merge the pathway sets

from pathlib import Path

import pandas as pd

from pyplier.utils import fix_dataframe_dtypes


def combine_paths(*args) -> pd.DataFrame:
    return pd.concat(args, axis=1, join="outer").fillna(0).astype(int)


def pathway_from_gmt(gmt_file: Path | str) -> pd.DataFrame:
    gmt_file = Path(gmt_file) if isinstance(gmt_file, str) else gmt_file
    if not gmt_file.exists():
        msg = f"{gmt_file} was not found"
        raise FileNotFoundError(msg)

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
    df = pd.pivot_table(
        gmt_long,
        values="count",
        index="genes",
        columns="pathway",
        fill_value=0,
    ).rename_axis(columns=None)

    return fix_dataframe_dtypes(df)
