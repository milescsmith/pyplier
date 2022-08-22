import warnings
from copy import deepcopy
from typing import Optional, TypedDict, Union

import numpy as np
import pandas as pd
from glmnet import ElasticNet
from scipy.stats import rankdata
from tqdm.auto import trange


class solveUReturnDict(TypedDict):
    U: pd.DataFrame
    L3: float


def solveU(
    Z,
    Chat,
    priorMat,
    penalty_factor,
    pathwaySelection: str = "fast",
    glm_alpha: float = 0.9,
    maxPath: int = 10,
    target_frac: float = 0.7,
    L3: Optional[float] = None,
) -> solveUReturnDict:
    """[summary]

    Parameters
    ----------
    Z : [type]
        current Z estimate
    Chat : [type]
        the inverse of the C matrix
    priorMat : [type]
        the prior pathway or C matrix
    penalty_factor : [type]
        Penalties for different pathways, must have size priorMat.shape[1].
    pathwaySelection : str, optional
        Method to use for pathway selection., by default "fast"
    glm_alpha : float, optional
        The elsatic net alpha parameter, by default 0.9
    maxPath : int, optional
        The maximum number of pathways to consider, by default 10
    target_frac : float, optional
        The target fraction on non-zero columns of, by default 0.7
    L3 : float, optional
        Solve with a given L3, otherwise search, by default None

    Returns
    -------
    [type]
        [description]
    """
    Ur = Chat @ Z  # get U by OLS

    Ur = Ur.rank(axis="index", ascending=False)  # rank

    if pathwaySelection != "fast":
        iip = np.where([Ur.min(axis=1) <= maxPath])[1]

    results = dict()

    if L3 is None:
        U = np.zeros(shape=(priorMat.shape[1], Z.shape[1]))

        lambdas = np.exp(np.arange(start=-4, stop=-12.125, step=-0.125))
        results = dict()
        lMat = np.full((len(lambdas), Z.shape[1]), np.nan)
        gres = ElasticNet(
            lambda_path=lambdas,
            lower_limits=0,
            standardize=False,
            fit_intercept=True,
            alpha=glm_alpha,
            max_features=150,
        )

        for i in range(Z.shape[1]):
            if pathwaySelection == "fast":
                iip = np.where([Ur.iloc[:, i] <= maxPath])[1]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gres.fit(
                    y=Z.iloc[:, i],
                    X=priorMat.iloc[:, iip],
                    relative_penalties=penalty_factor[iip],
                )

            gres.iip = iip
            lMat[:, i] = np.sum(np.where(gres.coef_path_ > 0, 1, 0), axis=0)
            results[i] = deepcopy(gres)

        fracs = np.mean(np.where(lMat > 0, 1, 0), axis=1)
        iibest = np.where(abs(target_frac - fracs) == abs((target_frac - fracs)).min())[
            0
        ][0]

        # yeah, so this is not very pythonic, but it matches the R code
        # TODO: replace this with something like our original attempt
        for i in trange(Z.shape[1]):
            U[results[i].iip, i] = results[i].coef_path_[:, iibest]

        U = pd.DataFrame(U, index=priorMat.columns, columns=Z.columns).fillna(0)
        L3 = lambdas[iibest]
    else:
        # do one fit with a given lambda
        gres = ElasticNet(
            lambda_path=[L3 * 0.9, L3, L3 * 1.1],
            lower_limits=0,
            standardize=False,
            fit_intercept=True,
            alpha=glm_alpha,
            max_features=150,
        )

        for i in range(Z.shape[1]):
            if pathwaySelection == "fast":
                iip = np.where([Ur.iloc[:, i] <= maxPath])[1]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # try:
                gres.fit(
                    y=Z.iloc[:, i],
                    X=priorMat.iloc[:, iip],
                    relative_penalties=penalty_factor[iip],
                )
            results[i] = pd.Series(data=gres.coef_path_[:, 1], index=Ur.index[iip])

        U = pd.DataFrame(results, index=priorMat.columns).fillna(0)

    return solveUReturnDict(U=U, L3=L3)
