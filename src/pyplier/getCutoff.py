import pandas as pd


def getCutoff(plierRes: dict[str, pd.DataFrame], fdr_cutoff: float = 0.01) -> float:
    return max(plierRes["summary"][plierRes["summary"]["FDR"] <= fdr_cutoff]["p-value"])
