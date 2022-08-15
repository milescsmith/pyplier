from typing import Optional, TypeVar

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from tqdm.auto import tqdm
from typeguard import typechecked

from .PLIERRes import PLIERResults
from .utils import rowNorm

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")


@typechecked
def plotMat(
    mat: pd.DataFrame,
    scale: bool = True,
    trim_names: Optional[int] = 50,
    cutoff: Optional[float] = None,
    *args,
    **kwargs
) -> dict[str, np.ndarray]:
    if trim_names is not None:
        mat.index = mat.index.str.slice(stop=trim_names)
        mat.columns = mat.columns.str.slice(stop=trim_names)

    if cutoff is not None:
        mat.where(lambda x: x > cutoff, 0, inplace=True)

    iirow = np.where(mat.abs().sum(axis=1) > 0)[0]
    mat = mat.iloc[iirow, :]
    iicol = np.where(mat.abs().sum(axis=0) > 0)[0]
    mat = mat.iloc[:, iicol]

    if scale:
        aa = mat.abs().max(axis=0)  # colMax
        mat = mat.divide(aa.replace(0, 1), axis=1)

    sns.clustermap(data=mat, *args, **kwargs)

    return {"iirow": iirow, "iicol": iicol}


def plotTopZ(
    plierRes: PLIERRes,
    data: pd.DataFrame,
    priorMat: pd.DataFrame,
    top: int = 10,
    index: Optional[list[str]] = None,
    regress: bool = False,
    allLVs: bool = False,
    *args,
    **kwargs
) -> None:
    data = data.loc[plierRes.Z.index, :]
    priorMat = priorMat.loc[plierRes.Z.index.intersection(priorMat.index), :]
    plierRes.U.columns[np.where(plierRes.U.sum(axis=0) > 0)]

    if not allLVs:
        if index is not None:
            ii = plierRes.U.columns[np.where(plierRes.U.sum(axis=0) > 0)].intersection(
                index
            )
    elif index is not None:
        ii = index

    tmp = plierRes.Z.loc[:, ii].rank(ascending=False)

    nntmp = [tmp.index[np.where(tmp[i] <= top)[0]].values for i in ii]
    nn = np.concatenate(nntmp)
    nncol = (
        plierRes.B.index[np.where(plierRes.Z.columns.isin(ii))[0]]
        .repeat([len(_) for _ in nntmp])
        .str[:30]
    )
    nnpath = pd.concat(
        [
            priorMat.loc[x, plierRes.U.loc[plierRes.U.loc[:, y] > 0, y].index].sum(
                axis=1
            )
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
                    plierRes.B.drop(index=plierRes.B.index[0]).loc[:, toPlot.columns].T,
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


def plotU(
    plierRes: PLIERRes,
    auc_cutoff: float = 0.6,
    fdr_cutoff: float = 0.05,
    indexCol: Optional[list[str]] = None,
    indexRow: Optional[list[str]] = None,
    top: int = 3,
    sort_row: bool = False,
    *args,
    **kwargs
) -> None:
    """
    plot the U matrix from a PLIER decomposition

    Parameters
    ----------
    plierRes : the result returned by PLIER
    auc.cutoff : the AUC cutoff for pathways to be displayed, increase to get a smaller subset of U
    fdr.cutoff : the significance cutoff for the pathway-LV association
    indexCol : restrict to a subset of the columns (LVs)
    indexRow : restrict to a subset of rows (pathways). Useful if only interested in pathways of a specific type
    top : the number of top pathways to display for each LV
    sort_row : do not cluster the matrix but instead sort it to display the positive values close to the diagonal
    *args, **kwargs : options to be passed to :class:seaborn.heatmap
    """
    if indexCol is None:
        indexCol = plierRes.U.columns

    if indexRow is None:
        indexRow = plierRes.U.index

    U = plierRes.U.copy()
    pval_cutoff = plierRes.summary.loc[
        plierRes.summary["FDR"] < fdr_cutoff, "p-value"
    ].max()
    U[plierRes.Uauc < auc_cutoff] = 0
    U[plierRes.Up > pval_cutoff] = 0

    U = U.loc[indexRow, indexCol].apply(replace_below_top, n=top, replace_val=0)

    if sort_row:
        Um = pd.DataFrame(
            {
                x: np.multiply((U.columns.get_loc(x) + 1) * 100, np.sign(y))
                for x, y in U.iteritems()
            }
        ).max(axis=1)
        plotMat(
            U.loc[Um.rank(ascending=False).sort_values().index],
            row_cluster=False,
            *args,
            **kwargs
        )
    else:
        plotMat(U, *args, **kwargs)


def replace_below_top(sr: pd.Series, n: int = 3, replace_val: int = 0) -> pd.Series:
    sr[sr < sr.nlargest(n)[-1]] = replace_val
    return sr
