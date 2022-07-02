from typing import List, Optional, TypeVar

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from tqdm.auto import tqdm

from .PLIERRes import PLIERResults
from .utils import rowNorm

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")


def plotTopZ(
    plierRes: PLIERRes,
    data: pd.DataFrame,
    priorMat: pd.DataFrame,
    top: int = 10,
    index: Optional[List[str]] = None,
    regress: bool = False,
    allLVs: bool = False,
    *args,
    **kwargs
) -> None:
    data = data.loc[plierRes["Z"].index, :]
    priorMat = priorMat.loc[plierRes["Z"].index.intersection(priorMat.index), :]
    plierRes["U"].columns[np.where(plierRes["U"].sum(axis=0) > 0)]

    if not allLVs:
        if index is not None:
            ii = (
                plierRes["U"]
                .columns[np.where(plierRes["U"].sum(axis=0) > 0)]
                .intersection(index)
            )
    elif index is not None:
        ii = index

    tmp = plierRes["Z"].loc[:, ii].rank(ascending=False)

    nntmp = [tmp.index[np.where(tmp[i] <= top)].values for i in ii]
    nn = np.concatenate(nntmp)
    nncol = (
        plierRes["B"]
        .index[np.where(plierRes["Z"].columns.isin(ii))[0]]
        .repeat([len(_) for _ in nntmp])
        .str[:30]
    )
    nnpath = pd.concat(
        [
            priorMat.loc[
                x, plierRes["U"].loc[plierRes["U"].loc[:, y] > 0, y].index
            ].sum(axis=1)
            > 0
            for x, y in zip(nntmp, ii)
        ],
        axis=0,
    )
    nnindex = ii.repeat([len(_) for _ in nntmp])

    nnrep = np.unique(nn)[np.where(np.unique(nn, return_counts=True)[1] > 1)[0]]

    if len(nnrep) > 0:
        nnrep_im = np.where(nn == nnrep)[0]
        nn = np.delete(nn, nnrep_im)
        nncol = np.delete(nncol, nnrep_im)
        nnpath.drop(nnrep, inplace=True)
        nnindex = np.delete(nnindex, nnrep_im)

    nnpath.replace({True: "inPathway", False: "notInPathway"}, inplace=True)
    nncol = pd.DataFrame({"pathway": nncol, "present": nnpath})

    toPlot = rowNorm(data.loc[nn, :])

    if regress:
        for i in tqdm(ii):
            gi = np.where(nnindex == i)[0]
            toPlot.iloc[gi, :] = (
                sm.GLS(
                    toPlot.iloc[gi, :].T,
                    plierRes["B"]
                    .drop(index=plierRes["B"].index[0])
                    .loc[:, toPlot.columns]
                    .T,
                )
                .fit()
                .resid.T
            )
    # maxval = toPlot.abs().max().max()

    present_lut = {
        "inPathway": mcolors.to_rgb(mcolors.CSS4_COLORS["black"]),
        "notInPathway": mcolors.to_rgb(mcolors.CSS4_COLORS["beige"]),
    }

    pathway_pal = sns.husl_palette(nncol["pathway"].nunique(), s=2)
    pathway_lut = {x: y for x, y in zip(nncol["pathway"].unique(), pathway_pal)}

    nncol = nncol.replace({True: "inPathway", False: "notinPathway"})
    row_annotations = pd.DataFrame(
        {
            "pathway": nncol["pathway"].map(pathway_lut),
            "present": nncol["present"].map(present_lut),
        }
    )

    sns.clustermap(
        toPlot,
        cmap="viridis",
        robust=True,
        linecolor="black",
        row_colors=row_annotations,
    )
