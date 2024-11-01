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

from pyplier.plier_res import PLIERResults
from pyplier.utils import copyMat, crossprod, tcrossprod

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")


@typechecked
def auc(labels: pd.Series, values: pd.Series) -> dict[str, float]:
    posii = labels[labels > 0]
    negii = labels[labels <= 0]
    posn = len(posii)
    negn = len(negii)
    posval = values[posii.index]
    negval = values[negii.index]
    if posn > 0 and negn > 0:
        statistic, pvalue = mannwhitneyu(posval, negval, alternative="greater")
        conf_int_low, conf_int_high = mannwhitneyu_conf_int(posval, negval, alternative="greater")
        return {
            "low": conf_int_low,
            "high": conf_int_high,
            "auc": (statistic / (posn * negn)),
            "pval": pvalue,
        }
    else:
        return {"auc": 0.5, "pval": np.nan}


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
    """Essentially a straight-up transliteration of the the _wilcox.test function in R's {stats} library
    Necessary since the scipy.stats.mannwhitneyu does not return confidence intervals

    Parameters
    ----------
    x : Union[list[float], pd.Series]
        _description_
    y : Union[list[float], pd.Series]
        _description_
    alpha : float, optional
        _description_, by default 0.05
    tol_root : float, optional
        _description_, by default 1e-4
    digits_rank : float, optional
        _description_, by default np.inf
    correct : bool, optional
        _description_, by default False
    alternative : str, optional
        _description_, by default "two_sided"

    Returns
    -------
    tuple[float, float]
        _description_
    """
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

        dz = np.sum(dr[range(len_x)]) - (len_x * ((len_x + 1) / 2)) - (len_x * len_y / 2)

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
            * ((len_x + len_y + 1) - np.sum(np.power(nties_ci, 3) - nties_ci) / ((len_x + len_y) * (len_x + len_y - 1)))
        )

        if sigma_ci == 0:
            rprint("cannot compute confidence interval when all observations are tied")

        try:
            result = np.divide(np.subtract(dz, correction_ci), sigma_ci)
        except RuntimeWarning:
            rprint(f"dz: {dz}\ncorrection_ci: {correction_ci}\nsigma_ci: {sigma_ci}\n")
        return result

    def wdiff(d: float, zq: float) -> float:
        return W(d) - zq

    wmumin = W(mumin)
    wmumax = W(mumax)

    def root(zq: float) -> float:
        f_lower = wmumin - zq
        if f_lower <= 0:
            return mumin

        f_upper = wmumax - zq
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


def get_auc(plierRes: PLIERResults, data: pd.DataFrame, priorMat: pd.DataFrame) -> dict[str, pd.DataFrame]:
    z = plierRes.z
    zcv = copyMat(z)
    k = z.shape[1]
    l1 = plierRes.l1
    l2 = plierRes.l2
    u = plierRes.u

    for i in range(5):
        ii = [(_ * 5 + i) + 1 for _ in range(floor(data.shape[0] / 5)) if (_ * 5 + i) + 1 <= z.shape[0]]
        z_not_ii = z.loc[~z.index.isin(z.iloc[ii, :].index)]
        data_not_ii = data.loc[~data.index.isin(data.iloc[ii, :].index)]
        solve_a = crossprod(z_not_ii) + l2 * np.identity(k)
        bcv = solve(solve_a, np.identity(solve_a.shape[0])) @ z_not_ii.transpose() @ data_not_ii
        solve_bcv = tcrossprod(bcv) + l1 * np.identity(k)
        zcv.iloc[ii, :] = data.iloc[ii, :] @ bcv.transpose() @ solve(solve_bcv, np.identity(solve_bcv.shape[0]))

    out = pd.DataFrame(data=np.empty(shape=(0, 4)), columns=["pathway", "LV index", "AUC", "p-value"])
    ii = u.loc[:, np.sum(a=u, axis=0) > 0].columns
    uauc = copyMat(df=u, zero=True)
    up = copyMat(df=u, zero=True)

    for i in ii:
        iipath = u.loc[(u.loc[:, i] > 0), i].index
        for j in iipath:
            aucres = auc(priorMat.loc[:, j], zcv.loc[:, i])

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

            uauc.loc[j, i] = aucres["auc"]
            up.loc[j, i] = aucres["pval"]

    out["LV index"] = (
        out["LV index"].apply(lambda x: x.strip("LV")).astype(np.int64)
    )  # to match the output from {PLIER}
    _, fdr, *_ = multipletests(out.loc[:, "p-value"], method="fdr_bh")
    out.loc[:, "FDR"] = fdr
    return {"Uauc": uauc, "Upval": up, "summary": out.reset_index(drop=True)}
