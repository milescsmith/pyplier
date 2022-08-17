from math import floor
from typing import TypeVar, Union

import numpy as np
import pandas as pd
from rich import print as rprint
from scipy.linalg import solve
from scipy.optimize import brentq
from scipy.stats import mannwhitneyu, norm, rankdata
from statsmodels.stats.multitest import multipletests
from typeguard import typechecked

from .PLIERRes import PLIERResults
from .utils import copyMat, crossprod, tcrossprod

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")


@typechecked
def AUC(labels: pd.Series, values: pd.Series) -> dict[str, float]:
    posii = labels[labels > 0]
    negii = labels[labels <= 0]
    posn = len(posii)
    negn = len(negii)
    posval = values[posii.index]
    negval = values[negii.index]
    if posn > 0 and negn > 0:
        statistic, pvalue = mannwhitneyu(posval, negval, alternative="greater")
        conf_int_low, conf_int_high = mannwhitneyu_conf_int(
            posval, negval, alternative="greater"
        )
        res = {
            "low": conf_int_low,
            "high": conf_int_high,
            "auc": (statistic / (posn * negn)),
            "pval": pvalue,
        }
    else:
        res = {"auc": 0.5, "pval": np.nan}

    return res


@typechecked
def mannwhitneyu_conf_int(
    x: Union[list[float], pd.Series],
    y: Union[list[float], pd.Series],
    alpha: float = 0.05,
    tol_root: float = 1e-4,
    digits_rank: float = np.inf,
    correct: bool = False,
    alternative: str = "two_sided",
) -> tuple[float, float]:
    mumin = min(x) - max(y)
    mumax = max(x) - min(y)

    def W(d: float) -> float:
        len_x = len(x)
        len_y = len(y)
        dr = np.append(np.subtract(x, d), y)

        if np.isinf(digits_rank):
            dr = rankdata(dr)
        else:
            dr = rankdata(round(dr, digits_rank))

        _, nties_ci = np.unique(dr, return_counts=True)

        dz = (
            np.sum(dr[range(len_x)]) - (len_x * ((len_x + 1) / 2)) - (len_x * len_y / 2)
        )

        if correct:
            if alternative == "two_sided":
                correction_ci = np.sign(dz) * 0.5
            elif alternative == "greater":
                correction_ci = 0.5
            elif alternative == "less":
                correction_ci = -0.5
        else:
            correction_ci = 0

        sigma_ci = np.sqrt(
            (len_x * len_y / 12)
            * (
                (len_x + len_y + 1)
                - np.sum(np.power(nties_ci, 3) - nties_ci)
                / ((len_x + len_y) * (len_x + len_y - 1))
            )
        )

        if sigma_ci == 0:
            rprint("cannot compute confidence interval when all observations are tied")

        try:
            result = np.divide(np.subtract(dz, correction_ci), sigma_ci)
        except RuntimeWarning:
            rprint(
                f"dz: {dz}\n"
                f"correction_ci: {correction_ci}\n"
                f"sigma_ci: {sigma_ci}\n"
            )
        return result

    def wdiff(d: float, zq: float) -> float:
        return W(d) - zq

    Wmumin = W(mumin)
    Wmumax = W(mumax)

    def root(zq: float) -> float:
        f_lower = Wmumin - zq
        if f_lower <= 0:
            return mumin

        f_upper = Wmumax - zq
        if f_upper >= 0:
            return mumax

        try:
            return brentq(wdiff, mumin, mumax, zq, tol_root)
        except RuntimeError:
            try:
                return brentq(wdiff, mumin, mumax, zq, tol_root, maxiter=1000)
            except RuntimeError:
                return brentq(wdiff, mumin, mumax, zq, tol_root, maxiter=10000)

    if alternative == "two_sided":
        lower = root(norm.isf(alpha / 2))
        upper = root(norm.ppf(alpha / 2))
        cint = (lower, upper)
    elif alternative == "greater":
        lower = root(norm.isf(alpha))
        cint = (lower, np.PINF)
    elif alternative == "less":
        upper = root(norm.ppf(alpha))
        cint = (np.NINF, upper)
    else:
        cint = (np.nan, np.nan)

    return cint


def getAUC(
    plierRes: PLIERResults, data: pd.DataFrame, priorMat: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    Z = plierRes.Z
    Zcv = copyMat(Z)
    k = Z.shape[1]
    L1 = plierRes.L1
    L2 = plierRes.L2
    U = plierRes.U

    for i in range(5):
        ii = [
            (_ * 5 + i) + 1
            for _ in range(floor(data.shape[0] / 5))
            if (_ * 5 + i) + 1 <= Z.shape[0]
        ]
        Z_not_ii = Z.loc[~Z.index.isin(Z.iloc[ii, :].index)]
        data_not_ii = data.loc[~data.index.isin(data.iloc[ii, :].index)]
        solve_a = crossprod(Z_not_ii) + L2 * np.identity(k)
        Bcv = (
            solve(solve_a, np.identity(solve_a.shape[0]))
            @ Z_not_ii.transpose()
            @ data_not_ii
        )
        solve_Bcv = tcrossprod(Bcv) + L1 * np.identity(k)
        Zcv.iloc[ii, :] = (
            data.iloc[ii, :]
            @ Bcv.transpose()
            @ solve(solve_Bcv, np.identity(solve_Bcv.shape[0]))
        )

    out = pd.DataFrame(
        data=np.empty(shape=(0, 4)), columns=["pathway", "LV index", "AUC", "p-value"]
    )
    ii = U.loc[:, np.sum(a=U, axis=0) > 0].columns
    Uauc = copyMat(df=U, zero=True)
    Up = copyMat(df=U, zero=True)

    for i in ii:
        iipath = U.loc[(U.loc[:, i] > 0), i].index
        for j in iipath:
            aucres = AUC(priorMat.loc[:, j], Zcv.loc[:, i])

            out = pd.concat(
                [
                    out,
                    pd.DataFrame(
                        {
                            "pathway": [j],
                            "LV index": [i],
                            "AUC": [aucres["auc"]],
                            "p-value": [aucres["pval"]],
                        }
                    ),
                ],
                axis=0,
            )

            Uauc.loc[j, i] = aucres["auc"]
            Up.loc[j, i] = aucres["pval"]

    out["LV index"] = (
        out["LV index"].apply(lambda x: x.strip("LV")).astype(np.int64)
    )  # to match the output from {PLIER}
    _, fdr, *_ = multipletests(out.loc[:, "p-value"], method="fdr_bh")
    out.loc[:, "FDR"] = fdr
    return {"Uauc": Uauc, "Upval": Up, "summary": out.reset_index(drop=True)}
