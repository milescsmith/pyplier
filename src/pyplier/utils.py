from typing import Optional, Union

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
        return mat1.tranpose() @ mat2


# @jit(nopython=True)
def tcrossprod(
    mat1: Union[np.array, pd.DataFrame],
    mat2: Optional[Union[np.array, pd.DataFrame]] = None,
) -> Union[np.array, pd.DataFrame]:

    if mat2 is None:
        return mat1 @ mat1.transpose()
    else:
        return mat1 @ mat2.tranpose()