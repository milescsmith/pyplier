import random
from functools import singledispatch
from typing import Union

import numpy as np
from icontract import ensure, require
from pysmooth import smooth
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from rich import print as rprint

@ensure(lambda result: result > 0)
def num_pc(
    data: Union[dict[str, np.ndarray], np.ndarray],
    method: str = None,
    B: int = 20,
    seed: int = None,
) -> float:

    if method is None:
        method = "elbow"
    if method not in ("elbow", "permutation"):
        raise RuntimeError(
            f"method must be either 'elbow' or 'permutation', but \
                           {method} was passed"
        )

    if seed is not None:
        random.seed(seed)
    if not isinstance(data, dict):
        n = data.shape[1]  # nrows
        if n < 500:
            k = n
        else:
            k = int(max(200, n / 4))
    else:
        k = None

    uu, method = compute_uu(data, method=method, k=k)

    if (
        method == "permutation"
    ):  # not sure why this option is present in PLIER as it is not used
        rprint(
            "[red bold]WARNING!:[/red bold] using the 'permutation' method yields unreliable results.  This is only kept for compatibility with the R version of {PLIER}"
        )
        # nn = min(c(n, m))
        dstat = uu[0:k] ** 2 / sum(uu[0:k] ** 2)
        dstat0 = np.zeros(shape=(B, k))
        rng = np.random.default_rng()
        dat0 = np.copy(data)
        for i in range(B):
            dat0 = rng.permuted(dat0, axis=0).transpose()

            if k == n:
                uu0 = svd(dat0, compute_uv=False)
            else:
                _, uu0, _ = randomized_svd(M=dat0, n_components=k, n_iter=3)

            dstat0[i, :] = uu0[0:k] ** 2 / sum(uu0[0:k] ** 2)

        psv = np.ones(k)
        for i in range(k):
            psv[i] = np.count_nonzero(dstat0[:, i] >= dstat[i]) / dstat0.shape[0]

        for i in range(1, k):
            psv[i] = np.max([psv[(i - 1)], psv[i]])

        nsv = np.sum(psv[psv <= 0.1])
    elif method == "elbow":
        nsv = elbow(uu)

    return nsv


@singledispatch
def compute_uu(data, **kwargs):
    pass


@compute_uu.register
def _(data: np.ndarray, **kwargs) -> tuple[np.ndarray, str]:
    rprint("Computing svd")
    scaler = StandardScaler()
    data = scaler.fit_transform(data.T)
    uu = compute_svd(data, kwargs["k"])
    return uu, kwargs["method"]


@compute_uu.register
def _(data: dict, **kwargs):
    if data["d"] is not None:
        if kwargs["method"] == "permutation":
            rprint(
                "Original data is needed for permutation method.\nSetting method to elbow"
            )
            method = "elbow"
        else:
            method = kwargs["method"]
        uu = data["d"]
    return uu, method


@ensure(lambda result: result > 1)
def elbow(uu: np.ndarray) -> int:
    xraw = abs(np.diff(np.diff(uu)))
    rprint("Smoothing data")
    x = smooth(xraw, twiceit=True)
    # plot(x)

    return int((np.argwhere(x <= np.quantile(x, 0.5)))[1]) + 1


@require(lambda data: data.ndim >= 2)
def compute_svd(data: np.ndarray, k: int) -> np.ndarray:
    n = data.shape[1]  # nrows
    if n < 500:
        uu = svd(data.transpose(), compute_uv=False)
        return uu
    else:
        _, uu, _ = randomized_svd(M=data, n_components=k, n_iter=3, random_state=803)
        return uu
