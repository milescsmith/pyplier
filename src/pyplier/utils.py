from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

# from numba import jit


# @jit(nopython=True)
def crossprod(
    mat1: npt.NDArray | pd.DataFrame,
    mat2: npt.NDArray | pd.DataFrame | None = None,
) -> npt.NDArray | pd.DataFrame:
    return mat1.transpose() @ mat1 if mat2 is None else mat1.transpose() @ mat2


# @jit(nopython=True)
def tcrossprod(
    mat1: npt.NDArray | pd.DataFrame,
    mat2: npt.NDArray | pd.DataFrame | None = None,
) -> npt.NDArray | pd.DataFrame:
    return mat1 @ mat1.transpose() if mat2 is None else mat1 @ mat2.transpose()


def zscore(arr: pd.Series | npt.NDArray) -> pd.Series | npt.NDArray:
    if np.std(arr, ddof=1) == 0.0:
        return np.zeros(len(arr))
    else:
        return (arr - np.mean(arr)) / np.std(arr, ddof=1)


def setdiff(list1: list[Any], list2: list[Any]) -> list[Any]:
    return [_ for _ in set(list1) if _ not in list2]


def copyMat(df: pd.DataFrame, zero: bool = False) -> pd.DataFrame:
    return pd.DataFrame(np.zeros(shape=df.shape), index=df.index, columns=df.columns) if zero else df.copy(deep=True)


def get_cutoff(aucRes: dict[str, pd.DataFrame], fdr_cutoff: float = 0.01) -> float:
    return np.amax(aucRes["summary"][aucRes["summary"]["FDR"] <= fdr_cutoff]["p-value"])


def fix_dataframe_dtypes(df: pd.DataFrame, codec: str = "UTF-8") -> pd.DataFrame:
    df = df.apply(lambda x: x.str.decode(codec) if x.dtype == "object" else x)
    df = df.apply(lambda x: x.astype(np.float64) if (np.mod(x, 1) != 0).any() else x.astype(np.int64), axis=0)
    if (df.dtypes == "float64").any():
        df = df.astype(np.float64)
    return df
