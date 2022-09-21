from typing import Any, Optional, Union

import numpy as np
import pandas as pd

# from numba import jit


# @jit(nopython=True)
def crossprod(
    mat1: Union[np.array, pd.DataFrame],
    mat2: Optional[Union[np.array, pd.DataFrame]] = None,
) -> Union[np.array, pd.DataFrame]:

    if mat2 is None:
        return mat1.transpose() @ mat1
    else:
        return mat1.transpose() @ mat2


# @jit(nopython=True)
def tcrossprod(
    mat1: Union[np.array, pd.DataFrame],
    mat2: Optional[Union[np.array, pd.DataFrame]] = None,
) -> Union[np.array, pd.DataFrame]:

    if mat2 is None:
        return mat1 @ mat1.transpose()
    else:
        return mat1 @ mat2.transpose()


def zscore(arr: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    if np.std(arr, ddof=1) == 0.0:
        return np.zeros(len(arr))
    else:
        return (arr - np.mean(arr)) / np.std(arr, ddof=1)


def setdiff(list1: list[Any], list2: list[Any]) -> list[Any]:
    return [_ for _ in set(list1) if _ not in list2]


def copyMat(df: pd.DataFrame, zero: bool = False) -> pd.DataFrame:
    if zero:
        dfnew = pd.DataFrame(
            np.zeros(shape=df.shape), index=df.index, columns=df.columns
        )
    else:
        dfnew = df.copy(deep=True)

    return dfnew


def getCutoff(aucRes: dict[str, pd.DataFrame], fdr_cutoff: float = 0.01) -> float:
    return np.amax(aucRes["summary"][aucRes["summary"]["FDR"] <= fdr_cutoff]["p-value"])
