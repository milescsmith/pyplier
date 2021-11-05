import pandas as pd


def commonRows(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Index:
    return pd.Index.intersection(df1.index, df2.index)
