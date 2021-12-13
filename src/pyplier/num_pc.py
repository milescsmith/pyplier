import random
import numpy as np
from scipy.linalg import svd
from .console import console
from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import scale

from pysmooth import smooth


def num_pc(data: np.ndarray, method: str = None, B: int = 20, seed: int = None) -> float:
  
    if method is None:
        method = "elbow"
    if method not in ("elbow","permutation"):
        raise RuntimeError(f"method must be either 'elbow' or 'permutation', but \
                           {method} was passed")
  
    if seed is not None:
        random.seed(seed)

    n = data.shape[1] # nrows
    if n < 500:
        k = n
    else:
        k = int(max(200, n / 4))
        
    if isinstance(data, np.ndarray):
        console.print("Computing svd")
        data = scale(data, axis=1)
        uu = compute_svd(data, k)
    elif isinstance(data, dict):
        if data["d"] is not None:
            if method == "permutation":
                console.print("Original data is needed for permutation method.\nSetting method to elbow")
                method = "elbow"

        uu = data

    if method == "permutation": # not sure why this option is present in PLIER as it is not used
        console.print("[red bold]WARNING!:[/red bold] using the 'permutation' method yields unreliable results.  This is only kept for compatibility with the R version of {PLIER}")
        # nn = min(c(n, m))
        dstat = uu[0:k]**2 / sum(uu[0:k]**2)
        dstat0 = np.zeros(shape=(B,k))
        rng = np.random.default_rng()
        dat0 = np.copy(data)
        for i in range(B):
            dat0 = rng.permuted(dat0, axis=0).transpose()
        
            if k == n:
                uu0 = svd(dat0, compute_uv=False)
            else:
                _, uu0, _ = randomized_svd(M=dat0, n_components=k, n_iter=3)
        
            dstat0[i,:] = uu0[0:k]**2 / sum(uu0[0:k]**2)
        
        psv = np.ones(k) 
        for i in range(k):
            psv[i] = np.count_nonzero(dstat0[:,i] >= dstat[i])/dstat0.shape[0]
        
        for i in range(1,k):
            psv[i] = np.max([psv[(i - 1)], psv[i]])

        nsv = np.sum(psv[psv <= 0.1])
    elif method == "elbow":
        # xraw = abs(np.diff(np.diff(uu)))
        # console.print("Smoothing data")
        # x = smooth(xraw, twiceit = True)
        # # plot(x)

        # nsv = int((np.argwhere(x <= np.quantile(x, 0.05)))[2])+1 
        
        nsv = elbow(uu)

    return nsv


def elbow(uu: np.ndarray) -> int:
    xraw = abs(np.diff(np.diff(uu)))
    console.print("Smoothing data")
    x = smooth(xraw, twiceit = True)
    # plot(x)

    return int((np.argwhere(x <= np.quantile(x, 0.5)))[1])+1 


def compute_svd(data: np.ndarray, k: int) -> np.ndarray:
    n = data.shape[1] # nrows
    if n < 500:
        uu = svd(data.transpose(), compute_uv=False)
        return uu
    else:
        _, uu, _ = randomized_svd(M=data, n_components=k, n_iter=3, random_state=803)
        return uu