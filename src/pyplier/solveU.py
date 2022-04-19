import warnings
from copy import deepcopy
from typing import Optional, TypedDict

import numpy as np
import pandas as pd
from glmnet import ElasticNet
from tqdm.auto import trange

from . import plier_logger as logger


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
    Urm = Ur.min(axis=1)

    U = pd.DataFrame(np.zeros(shape=(priorMat.shape[1], Z.shape[1])))
    if L3 is None:
        lambdas = np.exp(np.arange(start=-4, stop=-12.125, step=-0.125))
        results = dict()
        lMat = np.full((len(lambdas), Z.shape[1]), np.nan)

        for i in range(Z.shape[1]):
            if pathwaySelection == "fast":
                iip = np.where([Ur.iloc[:, i] <= maxPath])[1]
            else:
                iip = np.where([Urm <= maxPath])[1]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gres = ElasticNet(
                    lambda_path=lambdas,
                    lower_limits=0,
                    standardize=False,
                    fit_intercept=True,
                    alpha=glm_alpha,
                    max_features=150,
                )

                gres.fit(
                    y=Z.iloc[:, i].astype(np.float64).values,
                    X=priorMat.iloc[:, iip].astype(np.float64).values,
                    relative_penalties=[penalty_factor[_] for _ in iip],
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
            U.iloc[results[i].iip, i] = results[i].coef_path_[:, iibest]

        U.index = priorMat.columns
        U.columns = Z.columns
        # try:
        #     U = (pd.DataFrame(
        #             index=(priorMat.columns.set_names("pathway")).merge(pd.DataFrame(data={
        #                 i: pd.Series(
        #                     data=results[i].coef_path_[:, iibest],
        #                     index=Ur.index[results[i].iip].set_names("pathway")
        #                     )
        #                 for i in range(Z.shape[1])
        #             }, ),
        #             on="pathway",
        #             how="left",
        #         ).fillna(0)))
        # except KeyError:
        #     print("oops!")
        #     print(
        #         pd.DataFrame(data={
        #             i: pd.Series(
        #                 data=results[i].coef_path_[:, iibest],
        #                 index=Ur.index[results[i].iip],
        #             )
        #             for i in range(Z.shape[1])
        #         }, ).index.name)
        #     print(pd.DataFrame(index=priorMat.columns).index.name)

        # what is the point of this?  It is never used!
        # Utmp = solveU(Z, Chat, priorMat, penalty.factor,
        #     pathwaySelection = "fast", glm_alpha = 0.9, maxPath = 10,
        #     L3 = lambdas[iibest]
        #     )

        # stop()
        return solveUReturnDict(U=U, L3=lambdas[iibest])
    else:
        # do one fit with a given lambda
        results = dict()
        for i in range(Z.shape[1]):
            if pathwaySelection == "fast":
                iip = np.where([Ur.iloc[:, i] <= maxPath])[1]
            else:
                iip = np.where([Urm <= maxPath])[1]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gres = ElasticNet(
                    lambda_path=[L3 * 0.9, L3, L3 * 1.1],
                    lower_limits=0,
                    standardize=False,
                    fit_intercept=True,
                    alpha=glm_alpha,
                    max_features=150,
                )

                # try:
                gres.fit(
                    y=Z.iloc[:, i].astype(np.float64).values,
                    X=priorMat.iloc[:, iip].astype(np.float64).values,
                    relative_penalties=[penalty_factor[_] for _ in iip],
                )
            # except TypeError:
            #     print(f"iip: {iip}")
            #     print(f"sliced: {[penalty_factor[_] for _ in iip]}")
            #     print(f"penalty_factor: {penalty_factor}")

            # try:
            results[i] = pd.Series(
                data=[_[1] for _ in gres.coef_path_], index=Ur.index[iip]
            )
            # except AttributeError:
            #     print(dir(gres))

            # U[iip, i] = [_[1] for _ in gres.coef_path_]

        U = (
            pd.DataFrame(index=priorMat.columns.set_names("pathway"))
            .merge(
                pd.DataFrame(
                    {i: results[i] for i in range(Z.shape[1])},
                ).rename_axis(index="pathway", axis="index"),
                on="pathway",
                how="left",
            )
            .fillna(0)
        )

        return solveUReturnDict(U=U, L3=L3)
