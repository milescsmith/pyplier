import warnings
from copy import deepcopy
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from glmnet import ElasticNet
from tqdm.auto import trange


class SolveUReturnDict(TypedDict):
    U: pd.DataFrame
    l3: float


def solveU(
    z,
    chat,
    priorMat,
    penalty_factor,
    pathway_selection: Literal["complete", "fast"] = "fast",
    glm_alpha: float = 0.9,
    max_path: int = 10,
    target_frac: float = 0.7,
    l3: float | None = None,
    disable_progress: bool = False,
    persistent_progress: bool = True,
) -> SolveUReturnDict:
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
    l3 : float, optional
        Solve with a given l3, otherwise search, by default None
    persistent_progress : bool
        Should the progress bar progress be kept after completion? Defaults to True.
    disable_progress :
        Should progress bars be disabled? Defaults to False.

    Returns
    -------
    [type]
        [description]
    """
    ur = chat @ z  # get U by OLS

    ur = ur.rank(axis="index", ascending=False)  # rank

    if pathway_selection != "fast":
        iip = np.where([ur.min(axis=1) <= max_path])[1]

    results = {}

    if l3 is None:
        u = np.zeros(shape=(priorMat.shape[1], z.shape[1]))

        lambdas = np.exp(np.arange(start=-4, stop=-12.125, step=-0.125))
        results = {}
        l_mat = np.full((len(lambdas), z.shape[1]), np.nan)
        gres = ElasticNet(
            lambda_path=lambdas,
            lower_limits=0,
            standardize=False,
            fit_intercept=True,
            alpha=glm_alpha,
            max_features=150,
        )

        for i in trange(
            z.shape[1], disable=disable_progress, leave=persistent_progress, desc="Performing ElasticNet fit"
        ):
            if pathway_selection == "fast":
                iip = np.where([ur.iloc[:, i] <= max_path])[1]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gres.fit(
                    y=z.iloc[:, i],
                    X=priorMat.iloc[:, iip],
                    relative_penalties=penalty_factor[iip],
                )

            gres.iip = iip
            l_mat[:, i] = np.sum(np.where(gres.coef_path_ > 0, 1, 0), axis=0)
            results[i] = deepcopy(gres)

        fracs = np.mean(np.where(l_mat > 0, 1, 0), axis=1)
        iibest = np.where(abs(target_frac - fracs) == abs(target_frac - fracs).min())[0][0]

        # yeah, so this is not very pythonic, but it matches the R code
        # TODO: replace this with something like our original attempt
        for i in trange(
            z.shape[1], disable=disable_progress, leave=persistent_progress, desc="Storing ElasticNet fits"
        ):
            u[results[i].iip, i] = results[i].coef_path_[:, iibest]

        u = pd.DataFrame(u, index=priorMat.columns, columns=z.columns).fillna(0)
        l3 = lambdas[iibest]
    else:
        # do one fit with a given lambda
        gres = ElasticNet(
            lambda_path=[l3 * 0.9, l3, l3 * 1.1],
            lower_limits=0,
            standardize=False,
            fit_intercept=True,
            alpha=glm_alpha,
            max_features=150,
        )

        for i in range(z.shape[1]):
            if pathway_selection == "fast":
                iip = np.where([ur.iloc[:, i] <= max_path])[1]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                # try:
                gres.fit(
                    y=z.iloc[:, i],
                    X=priorMat.iloc[:, iip],
                    relative_penalties=penalty_factor[iip],
                )
            results[i] = pd.Series(data=gres.coef_path_[:, 1], index=ur.index[iip])

        u = pd.DataFrame(results, index=priorMat.columns).fillna(0)

    return SolveUReturnDict(U=u, l3=l3)
