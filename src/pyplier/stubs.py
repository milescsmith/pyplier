from typing import TypedDict

from pandas import DataFrame

PLIERResults = TypedDict(
    "PlierResults",
    {
        "residual": DataFrame,
        "B": DataFrame,
        "Z": DataFrame,
        "U": DataFrame,
        "C": DataFrame,
        "L1": float,
        "L2": float,
        "L3": float,
        "heldOutGenes": dict[str, list[str]],
        "withPrior": list[str],
        "Uauc": DataFrame,
        "Up": DataFrame,
        "summary": DataFrame,
    },
)
