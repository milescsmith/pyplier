from functools import singledispatch
from typing import Any, List, Optional, Union

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


@singledispatch
def rowNorm(arr):
    return arr


@rowNorm.register
def _(arr: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda x: ((x - np.mean(x)) / np.std(x, ddof=1)), 1, arr)


@rowNorm.register
def _(arr: pd.DataFrame) -> pd.DataFrame:
    return arr.apply(lambda x: ((x - np.mean(x)) / np.std(x, ddof=1)), axis=1)


def setdiff(list1: List[Any], list2: List[Any]) -> List[Any]:
    return [_ for _ in set(list1) if _ not in list2]
