import numpy as np
import pandas as pd
from icontract import require

from .getCutoff import getCutoff
from .PLIERRes import PLIERResults


@require(lambda ngenes: ngenes > 0)
def getEnrichmentVals(
    plierRes: PLIERResults,
    pathwayMat: pd.DataFrame,
    ngenes: int = 50,
    auc_cutoff: float = 0.7,
    fdr_cutoff: float = 0.01,
) -> pd.DataFrame:

    pathwayMat = pathwayMat.loc[plierRes.Z.index, plierRes.U.index]
    Uuse = np.where(plierRes.U < auc_cutoff, 0, plierRes.U)
    Uuse = np.where(plierRes.Up > getCutoff(plierRes, fdr_cutoff), 0, plierRes.U)
    intop = np.zeros(plierRes.Z.shape[1])
    inpath = np.zeros(plierRes.Z.shape[1])

    for i in range(intop):
        iipath = np.where(Uuse.iloc[:, i] > 0)
        if len(iipath) > 0:
            pathGenes = pathwayMat.loc[
                pathwayMat.iloc[:, iipath].apply(sum, axis="columns") > 0, :
            ].index
            topGenes = (
                plierRes.Z.iloc[:, i].sort_values(ascending=False)[1:ngenes].index
            )
            pathGenesInt = topGenes.intersection(pathGenes)
            inpath[i] = len(pathGenes)
            intop[i] = len(pathGenesInt)

    return pd.DataFrame(data={1: intop / inpath, 2: intop, 3: inpath})
