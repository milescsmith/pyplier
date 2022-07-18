from typing import Union

import numpy as np


def colSumNorm(
    mat: np.ndarray, return_all: bool = False
) -> Union[dict[str, np.ndarray], np.ndarray]:
    ss = np.sqrt(np.sum(np.power(mat, 2), axis=1))
    ss = np.where(ss < 1e-16, 1, ss)
    if return_all:
        return {"mat": (mat.transpose() / ss).transpose(), "ss": ss}
    else:
        return (mat.transpose() / ss).transpose()
