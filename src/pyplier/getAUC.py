from math import floor
from typing import Dict

import numpy as np
import pandas as pd
from scipy.linalg import solve
from statsmodels.stats.multitest import multipletests

from .AUC import AUC
from .copyMat import copyMat
from .stubs import PLIERResults
from .utils import crossprod, tcrossprod


def getAUC(
    plierRes: PLIERResults, data: pd.DataFrame, priorMat: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    B = plierRes.B
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
                    )
                ],
                axis=0
            )

            Uauc.loc[j, i] = aucres["auc"]
            Up.loc[j, i] = aucres["pval"]

    out["LV index"] = (
        out["LV index"].apply(lambda x: x.strip("LV")).astype(np.int64)
    )  # to match the output from {PLIER}
    _, fdr, *_ = multipletests(out.loc[:, "p-value"], method="fdr_bh")
    out.loc[:, "FDR"] = fdr
    return {"Uauc": Uauc, "Upval": Up, "summary": out.reset_index(drop=True)}
