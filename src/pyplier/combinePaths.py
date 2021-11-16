import pandas as pd


def combinePaths(*args):
    return pd.concat(args, axis=1, join="outer").astype(float).fillna(0)
