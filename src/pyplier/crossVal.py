from typing import Dict
from collections.abc import Iterable

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from .AUC import AUC
from .copyMat import copyMat
from .PLIERRes import PLIERResults


def crossVal(
    plierRes: PLIERResults, priorMat: pd.DataFrame, priorMatcv: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    title crossVal

    param priorMat the real prior info matrix
    param priorMatcv the zeroed-out prior info matrix used for PLIER
    computations

    """
    out = pd.DataFrame(
        data=np.empty(shape=(0, 4)), columns=["pathway", "LV index", "AUC", "p-value"]
    )
    out_dict = dict()
    ii = plierRes.U.loc[:, plierRes.U.sum(axis=0) > 0].columns
    Uauc = copyMat(df=plierRes.U, zero=True)
    Up = pd.DataFrame(np.ones(shape=plierRes.U.shape))

    for i in tqdm(ii):
    iipath = plierRes.U.loc[(plierRes.U.loc[:, i] > 0), i].index
    if len(iipath) > 1:
        for j in tqdm(iipath):
            iiheldout = (
                pd.concat([priorMat.loc[:, j], priorMatcv.loc[:, j]], axis=1)
                .apply(
                    lambda x: True
                    if (x[0] == 0) or ((x[0] > 0) and (x[1] == 0))
                    else np.nan,  # use np.nan instead of False so that we can drop entries in the chain
                    axis=1,
                )
                .dropna()
                .index
            )
            aucres = AUC(
                priorMat.loc[iiheldout, j], plierRes.Z.loc[iiheldout, i]
            )
            out_dict[j] = {
                "pathway": j,
                "LV index": i,
                "AUC": aucres["auc"],
                "p-value": aucres["pval"],
            }
            Uauc.loc[j, i] = aucres["auc"]
            Up.loc[j, i] = aucres["pval"]

        else:
            j = iipath
            iiheldout = (
                pd.concat([priorMat.loc[:, j], priorMatcv.loc[:, j]], axis=1)
                .apply(
                    lambda x: True
                    if (x[0] == 0) or ((x[0] > 0) and (x[1] == 0))
                    else np.nan,
                    axis=1,
                )
                .dropna()
                .index
            )
            aucres = AUC(priorMat.loc[iiheldout, j], plierRes.Z.loc[iiheldout, i])
            print(f"j: {j}")
            print(f"i: {i}")
            if isinstance(j, Iterable):
                for _ in j:
                    out_dict[_] = {
                        "pathway": _,
                        "LV index": i,
                        "AUC": aucres["auc"],
                        "p-value": aucres["pval"],
                    }
                    Uauc.loc[_, i] = aucres["auc"]
                    Up.loc[_, i] = aucres["pval"]
            elif isinstance(j, str):
                out_dict[j] = {
                    "pathway": j,
                    "LV index": i,
                    "AUC": aucres["auc"],
                    "p-value": aucres["pval"],
                }
                Uauc.loc[j, i] = aucres["auc"]
                Up.loc[j, i] = aucres["pval"]

    out = pd.DataFrame.from_dict(out_dict, orient="index")
    _, fdr, *_ = multipletests(out.loc[:, "p-value"], method="fdr_bh")
    out.loc[:, "fdr"] = fdr
    return {"Uauc": Uauc, "Upval": Up, "summary": out}
