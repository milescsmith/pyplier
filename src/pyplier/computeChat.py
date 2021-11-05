import numpy as np
import pandas as pd
from numba import jit
from scipy.linalg import LinAlgError, svd

from .utils import crossprod


def computeChat(gsMat, reg: int = 5):
    a = crossprod(gsMat)
    b = pinv_ridge(a, reg)

    Chat = b @ gsMat.transpose()

    return Chat


@jit(nopython=True)
def pinv_ridge(df: pd.DataFrame, alfa: int = 0) -> pd.DataFrame:
    """
    A variation of calculating the Moore-Penrose inverse with ridge adjustment
    Note that df MUST be symmetric

    Params
    ------
    df: `class::pd.DataFrame`
    alfa: `int` ridge penality adjustment

    """

    if df.shape[0] != df.shape[1]:
        raise LinAlgError("Non-symmetric matrix")

    u, d, v = svd(df)

    if alfa > 0:
        di = (np.power(d, 2) + alfa ** 2) / d
    else:
        di = d
    out = v.transpose() @ np.multiply(1 / di, u).transpose()

    out_df = pd.DataFrame(out, columns=df.columns, index=df.index)

    return out_df
