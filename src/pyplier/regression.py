import numpy as np
import pandas as pd

# from numba import jit
from scipy.linalg import LinAlgError, svd

from pyplier.utils import crossprod


def computeChat(gsMat, reg: int = 5):
    a = crossprod(gsMat)
    b = pinv_ridge(a, reg)

    return b @ gsMat.transpose()


# @jit(nopython=True)
def pinv_ridge(df: pd.DataFrame, alfa: int = 0) -> pd.DataFrame:
    """
    A variation of calculating the Moore-Penrose inverse with ridge adjustment
    Note that df MUST be symmetric

    Params
    ------
    df: :class:`~pd.DataFrame`

    alfa: `int`
        ridge penality adjustment

    """

    if df.shape[0] != df.shape[1]:
        msg = "Non-symmetric matrix"
        raise LinAlgError(msg)

    u, d, v = svd(df)  # note: compared to the R version of svd, the v matrix is returned transposed

    if len(d) == 0:
        return pd.DataFrame(np.zeros(tuple(reversed(df.shape))), columns=df.columns, index=df.index)

    di = (np.power(d, 2) + alfa**2) / d if alfa > 0 else d
    out = v.transpose() @ np.multiply(1 / di, u).transpose()

    return pd.DataFrame(out, columns=df.columns, index=df.index)
