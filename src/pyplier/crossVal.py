from collections.abc import Iterable
from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm.auto import tqdm

from .AUC import AUC
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
    Uauc = pd.DataFrame(
        np.zeros(shape=plierRes.U.shape),
        index=plierRes.U.index,
        columns=plierRes.U.columns,
    )
    Up = pd.DataFrame(
        np.ones(shape=plierRes.U.shape),
        index=plierRes.U.index,
        columns=plierRes.U.columns,
    )

    for i in tqdm(ii):
        iipath = plierRes.U.loc[(plierRes.U.loc[:, i] > 0), i].index
        if len(iipath) > 1:
            for j in tqdm(iipath):
                a = (
                    priorMat.loc[:, iipath]
                    .sum(axis=1)
                    .where(lambda x: x == 0)
                    .dropna()
                    .index
                )
                b = priorMat.loc[:, j].where(lambda x: x > 0).dropna().index
                c = priorMatcv.loc[:, j].where(lambda x: x == 0).dropna().index
                iiheldout = a.union(b.intersection(c))
                aucres = AUC(priorMat.loc[iiheldout, j], plierRes.Z.loc[iiheldout, i])
                out_dict[j] = {
                    "pathway": j,
                    "LV index": i,
                    "AUC": aucres["auc"],
                    "p-value": aucres["pval"],
                }
                Uauc.loc[j, i] = aucres["auc"]
                Up.loc[j, i] = aucres["pval"]

        else:
            j = iipath[0]
            a = priorMat.loc[:, iipath].where(lambda x: x == 0).dropna().index
            b = priorMat.loc[:, j].where(lambda x: x > 0).dropna().index
            c = priorMatcv.loc[:, j].where(lambda x: x == 0).dropna().index
            iiheldout = a.union(b.intersection(c))

            aucres = AUC(priorMat.loc[iiheldout, j], plierRes.Z.loc[iiheldout, i])
            if isinstance(j, Iterable) and not isinstance(j, str):
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

    out = pd.DataFrame.from_dict(out_dict, orient="index").set_index("pathway")
    _, fdr, *_ = multipletests(out.loc[:, "p-value"], method="fdr_bh")
    out.loc[:, "FDR"] = fdr
    return {"Uauc": Uauc, "Upval": Up, "summary": out}
