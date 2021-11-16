from typing import Dict

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from .AUC import AUC
from .copyMat import copyMat


def crossVal(
    plierRes: Dict[str, pd.DataFrame], priorMat: pd.DataFrame, priorMatcv: pd.DataFrame
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
    ii = plierRes["U"].loc[:, np.sum(a=plierRes["U"], axis=0) > 0].columns
    Uauc = copyMat(df=plierRes["U"], zero=True)
    Up = pd.DataFrame(np.ones(shape=plierRes["U"].shape))

    for i in ii:
        iipath = plierRes["U"].loc[(plierRes["U"].loc[:, i] > 0), i].index
        if len(iipath) > 1:
            for j in iipath:
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
                    priorMat.loc[iiheldout, j], plierRes["Z"].loc[iiheldout, i]
                )
                out = out.append(
                    other=pd.DataFrame(
                        {
                            "pathway": [j],
                            "LV index": [i],
                            "AUC": [aucres["auc"]],
                            "p-value": [aucres["pval"]],
                        }
                    )
                )
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
            aucres = AUC(priorMat.loc[iiheldout, j], plierRes["Z"].loc[iiheldout, i])
            out = out.append(
                other=pd.DataFrame(
                    {
                        "pathway": [j],
                        "LV index": [i],
                        "AUC": [aucres["auc"]],
                        "p-value": [aucres["pval"]],
                    }
                )
            )
            Uauc.loc[j, i] = aucres["auc"]
            Up.loc[j, i] = aucres["pval"]

    _, fdr, *_ = multipletests(out.loc[:, "p-value"], method="fdr_bh")
    out.loc[:, "fdr"] = fdr
    return {"Uauc": Uauc, "Upval": Up, "summary": out}
