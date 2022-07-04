from typing import Optional, TypeVar

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from tqdm.auto import tqdm

from .PLIERRes import PLIERResults
from .utils import rowNorm

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")


def plotTopZallPath(
    plierRes: PLIERRes,
    data: pd.DataFrame,
    priorMat: pd.DataFrame,
    top: int = 10,
    index: Optional[str] = None,
    regress: bool = False,
    fdr_cutoff: float = 0.2,
    *args,
    **kwargs
) -> None:
    """
    visualize the top genes contributing to the LVs similarily to [plotTopZ()].
        However in this case all the pathways contributing to each LV are show
        seperatly. Useful for seeing pathway usage for a single LV or
        understading the differences between two closely related LVs

    Parameters
    ----------
    plierRes : the result returned by PLIER
    data : the data to be displayed in a heatmap, typically the z-scored
    input : data (or some subset thereof)
    priorMat : the same gene by geneset binary matrix that was used to run PLIER
    top : the top number of genes to use
    index : the subset of LVs to display
    regress : remove the effect of all other LVs before plotting top genes,
        will take longer but can be useful to see distinct patterns in highly
        correlated genes.
    fdr_cutoff : Significance cutoff for a pathway to be plotted
    *args, **kwargs: Additional arguments to be passed to pheatmap, such as a
        column annotation data.frame
    """
    pval_cutoff = plierRes.summary.loc[
        plierRes.summary["FDR"] < fdr_cutoff, "p-value"
    ].max()

    ii = (plierRes.U.sum(axis=0) > 0).index
    if index is not None:
        ii = ii.intersection(index)

    tmp = plierRes.Z.loc[:, ii].rank(axis=0, ascending=False)
    Ustrict = plierRes.U.copy()
    Ustrict[plierRes.Up > pval_cutoff] = 0
    pathsUsed = Ustrict[Ustrict.loc[:, ii].sum(axis=1) > 0].index
    pathMat = np.zeros((0, len(pathsUsed)))

    nntmp = {i: tmp.index[np.where(tmp[i] <= top)[0]].values for i in ii}
    nn = np.concatenate(list(nntmp.values()))
    nncol = {
        i: plierRes.U.index[plierRes.U.loc[:, i] == plierRes.U.loc[:, i].max()].repeat(
            len(nntmp[i])
        )
        if plierRes.U.loc[:, i].max() > 0
        else pd.Index([i]).repeat(len(nntmp[i]))
        for i in ii
    }
    nnindex = np.concatenate([list(_) for _ in nncol.values()])
    pathMat = priorMat.loc[np.concatenate(list(nntmp.values())), pathsUsed]

    if any(pathMat.sum() > 1):
        pathMat.loc[:, pathMat.sum() > 0]
        pathsUsed = pathMat.columns

    pathMat = pathMat.astype("category")

    toPlot = data.loc[
        nn,
    ]

    if regress:
        for i in tqdm(ii):
            gi = np.where(nnindex == i)[0]
            toPlot.iloc[gi, :] = (
                sm.GLS(
                    toPlot.iloc[gi, :].T,
                    plierRes.B.drop(index=plierRes.B.index[0]).loc[:, toPlot.columns].T,
                )
                .fit()
                .resid.T
            )

    annotation_row = pathMat.replace(
        {
            0: mcolors.to_rgb(mcolors.CSS4_COLORS["beige"]),
            1: mcolors.to_rgb(mcolors.CSS4_COLORS["black"]),
        }
    )

    sns.clustermap(rowNorm(data.loc[nn, :]), row_colors=annotation_row, *args, **kwargs)
