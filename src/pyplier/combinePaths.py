import pandas as pd


def combinePaths(*args):
    return pd.concat(args, axis=1, join="outer").fillna(0).astype(int)
