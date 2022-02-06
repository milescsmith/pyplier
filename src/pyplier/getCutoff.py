import pandas as pd

from .stubs import PLIERResults


def getCutoff(plierRes: PLIERResults, fdr_cutoff: float = 0.01) -> float:
    return max(plierRes["summary"][plierRes["summary"]["FDR"] <= fdr_cutoff]["p-value"])
