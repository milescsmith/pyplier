import numpy as np
import pandas as pd


def pinv_ridge(m: pd.DataFrame, alpha: int = 0) -> pd.DataFrame:
    u, d, v = np.linalg.svd(m) # note: compared to the R version of svd, the v matrix is returned transposed
    if len(d) == 0:
        return np.zeros(tuple(reversed(m.shape)))
    else:
        if alpha > 0:
            ss = (d ** 2) + alpha ** 2
            d = ss / d
        out = v.transpose() @ (1 / d * u.transpose())
        out = pd.DataFrame(out)
        if isinstance(m, pd.DataFrame):
            out.index = m.index
            out.columns = m.columns
        return out
