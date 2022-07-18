import numpy as np
import pandas as pd
from icontract import ensure
from numpy.linalg import pinv


def VarianceExplained(
    Y: pd.DataFrame,
    Z: pd.DataFrame,
    B: pd.DataFrame,
    k: int = None,
    option: str = "regression",
):
    if k is None:
        k = min(Y.shape[1], Y.shape[0], Z.shape[1], B.shape[0])

    if option == "regression":
        np.subtract((Y.transpose().values @ Z.values), np.mean(B.values, axis=1))
    elif option == "matrix_regression":
        res = [calc_res(B.iloc[i, :].values, Y.values) for i in range(k)]
    elif option == "simple":
        # so the R version attempts to apply some function named 'normF'
        # columnwise to Z and rowize to B, but there is no function
        # named 'normF' except in the library {fugible}.  That library, however
        # is not listed as a dependency. So, I'm
        # guessing this should just be the Frobenius normal
        Z_norm = np.linalg.norm(Z, axis=1)
        B_norm = np.linalg.norm(B, axis=0)
        Z = np.divide(Z, np.reshape(Z_norm, (len(Z_norm), 1)))
        B = np.divide(B, np.reshape(B_norm, (1, len(B_norm))))
        res = (np.diag(Z.T @ Y @ B.T)[:k]) ** 2
    elif option == "project":
        res = [calc_project(B.iloc[: i + 1].values, Y) for i in range(k)]
        res = np.append(res[0], ([x - y for x, y in zip(res[1:k], res[0 : (k - 1)])]))
    return res


@ensure(lambda a: a.ndim == 1, "a is not 1d")
@ensure(lambda b: b.ndim == 1, "b is not 1d")
def calc_res(a: np.ndarray, b: np.ndarray) -> float:
    Zreg = np.multiply(b @ a, pinv(a)[0])
    res = np.sum(
        (
            (
                np.reshape(Zreg, (len(Zreg), 1)) @ np.reshape(a, (1, len(a)))
                - np.mean(b.flatten())
            )
            ** 2
        )
    )
    return res


@ensure(lambda a: a.ndim == 1, "a must be 1-d")
@ensure(lambda b: b.ndim == 1, "b must be 1-d")
def matmul1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(np.reshape(a, (len(a), 1)), np.reshape(b, (1, len(b))))


def calc_project(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape[0] == 1:
        Xk = matmul1d(
            (b @ a.T @ pinv(a @ a.T)[0]).values,
            a.flatten(),
        )
    else:
        Xk = np.matmul(
            (b @ a.T @ pinv(a @ a.T)),
            a,
        )
    return np.diag(Xk.T @ Xk).sum()
