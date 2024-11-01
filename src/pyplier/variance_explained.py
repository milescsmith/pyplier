import numpy as np
import numpy.typing as npt
import pandas as pd
from icontract import ensure
from numpy.linalg import pinv


def VarianceExplained(
    Y: pd.DataFrame,
    z: pd.DataFrame,
    b: pd.DataFrame,
    k: int | None = None,
    option: str = "regression",
):
    if k is None:
        k = min(Y.shape[1], Y.shape[0], z.shape[1], b.shape[0])

    if option == "matrix_regression":
        res = [calc_res(b.iloc[i, :].values, Y.values) for i in range(k)]
    elif option == "project":
        res = [calc_project(b.iloc[: i + 1].values, Y) for i in range(k)]
        res = np.append(
            res[0],
            [x - y for x, y in zip(res[1:k], res[: k - 1], strict=False)],
        )
    elif option == "regression":
        np.subtract((Y.transpose().values @ z.values), np.mean(b.values, axis=1))
    elif option == "simple":
        # so the R version attempts to apply some function named 'normF'
        # columnwise to Z and rowize to B, but there is no function
        # named 'normF' except in the library {fugible}.  That library, however
        # is not listed as a dependency. So, I'm
        # guessing this should just be the Frobenius normal
        z_norm = np.linalg.norm(z, axis=1)
        b_norm = np.linalg.norm(b, axis=0)
        z = np.divide(z, np.reshape(z_norm, (len(z_norm), 1)))
        b = np.divide(b, np.reshape(b_norm, (1, len(b_norm))))
        res = (np.diag(z.T @ Y @ b.T)[:k]) ** 2
    return res


@ensure(lambda a: a.ndim == 1, "a is not 1d")
@ensure(lambda b: b.ndim == 1, "b is not 1d")
def calc_res(a: npt.NDArray, b: npt.NDArray) -> float:
    zreg = np.multiply(b @ a, pinv(a)[0])
    return np.sum((np.reshape(zreg, (len(zreg), 1)) @ np.reshape(a, (1, len(a))) - np.mean(b.flatten())) ** 2)


@ensure(lambda a: a.ndim == 1, "a must be 1-d")
@ensure(lambda b: b.ndim == 1, "b must be 1-d")
def matmul1d(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    return np.matmul(np.reshape(a, (len(a), 1)), np.reshape(b, (1, len(b))))


def calc_project(a: npt.NDArray, b: npt.NDArray) -> float:
    if a.shape[0] == 1:
        xk = matmul1d(
            (b @ a.T @ pinv(a @ a.T)[0]).values,
            a.flatten(),
        )
    else:
        xk = np.matmul(
            (b @ a.T @ pinv(a @ a.T)),
            a,
        )
    return np.diag(xk.T @ xk).sum()
