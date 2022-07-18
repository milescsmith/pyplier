from functools import singledispatch
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


@singledispatch
def rowNorm(arr):
    return arr


@rowNorm.register
def _(arr: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda x: ((x - np.mean(x)) / np.std(x, ddof=1)), 1, arr)


@rowNorm.register
def _(arr: pd.DataFrame) -> pd.DataFrame:
    return arr.apply(lambda x: ((x - np.mean(x)) / np.std(x, ddof=1)), axis=1)


@singledispatch
def colNorm(arr):
    return arr


@colNorm.register
def _(arr: np.ndarray) -> np.ndarray:
    return np.apply_along_axis(lambda x: ((x - np.mean(x)) / np.std(x, ddof=1)), 0, arr)


@colNorm.register
def _(arr: pd.DataFrame) -> pd.DataFrame:
    return arr.apply(lambda x: ((x - np.mean(x)) / np.std(x, ddof=1)), axis=0)


def setdiff(list1: list[Any], list2: list[Any]) -> list[Any]:
    return [_ for _ in set(list1) if _ not in list2]
