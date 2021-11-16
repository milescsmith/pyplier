import numpy as np


def colSumNorm(mat: np.array, return_all: bool = False):
    ss = np.sqrt(np.sum(np.power(mat, 2), axis=0))
    ss = np.where(ss < 1e-16, 1, ss)
    if return_all:
        return (mat.transpose() / ss).transpose()
    else:
        return {"mat": (mat.transpose() / ss).transpose(), "ss": ss}
