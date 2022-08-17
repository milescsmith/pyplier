from typing import Optional, TypeVar

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import Patch
from tqdm.auto import tqdm
from typeguard import typechecked

from .PLIERRes import PLIERResults
from .utils import rowNorm

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")
ClusterGrid = TypeVar("ClusterGrid", bound="sns.matrix.ClusterGrid")


@typechecked
def plotMat(
    mat: pd.DataFrame,
    scale: bool = True,
    trim_names: Optional[int] = 50,
    cutoff: Optional[float] = None,
    *args,
    **kwargs,
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
    colormap: str = "viridis",
    annotation_row_palette: str = "hls",
    scale: Optional[int] = None,
    cluster_cols: bool = True,
    show_col_dendrogram: bool = False,
    cluster_rows: bool = False,
    show_row_dendrogram: bool = False,
    show_cbar: bool = False,
    show_legend: bool = True,
    shuffle_pathway_pal: bool = False,
    figsize: tuple[int] = (9, 9),
    *args,
    **kwargs,
) -> ClusterGrid:
    """Plot a heatmap of the top genes and the latent variables with which they are associated

    Parameters
    ----------
    plierRes : :class:`PLIERRes`
        result from :func:`pyplier.plier.PLIER`
    data : :class:`pd.DataFrame`
        the data to be displayed in a heatmap, typically the z-scored input data (or some subset thereof)
    priorMat : :class:`pd.DataFrame`
        the same gene by geneset binary matrix that was used to run PLIER
    top : int, default 10
        top number of genes to use
    index : list[str], default None
        Subset of LVs to display.  If None, show all.
    regress : bool, default False
        Remove the effect of all other LVs before plotting top genes.
        Will take longer, but can be useful to see distinct patterns in highly correlated LVs
    allLVs : bool, default False
        plot even the LVs that have no pathway association
    colormap : str, default 'viridis'
        color palette to use for displaying Z values in the heatmap
    annotation_row_palette : str, default "hls"
        color palette to use when generating label colors for the LV annotations
    scale : int, default None
        Whether to scale the values by rows (0) or columns(1) in the heatmap.  If None, no
        scaling is performed.  See :func:`sns.clustermap` for more.
    cluster_cols : bool, default True
        Should the samples be clustered?
    show_col_dendrogram : bool, default False
        Should the sample dendrogram be displayed?
    cluster_rows : bool, default False
        Should the genes be clustered?
    show_row_dendrogram : bool, default False
        Should the gene dendrogram be displayed?
    show_cbar : bool, default False
        Should the Z score color bar be displayed?
    show_legend : bool, default True
        Should the LV legend be displayed?
    shuffle_pathway_pal : bool, default False
        If true, the palette generated for the LV annotations is shuffled.  Can help
        if a lot of LVs are being displayed
    figsize : tuple[int], default (9,9)
        Figure size for the heatmap, expressed as (width, height)
    """

    # subset the original data matrix and the prior pathway matrix on the genes that were used by `PLIER`
    data = data.loc[plierRes.Z.index, :]
    priorMat = priorMat.loc[plierRes.Z.index.intersection(priorMat.index), :]
    plierRes.U.columns[np.where(plierRes.U.sum(axis=0) > 0)]

    if allLVs:
        if index is not None:
            ii = plierRes.U.columns.intersection(
                index
            )  # use `intersection` so we don't have an issue with trying to plot non-existent LVs
        else:
            ii = plierRes.U.columns
    elif index is not None:
        ii = plierRes.U.columns[plierRes.U.sum() > 0].intersection(index)
    else:
        ii = plierRes.U.columns[plierRes.U.sum() > 0]

    z_ranks = plierRes.Z.loc[:, ii].rank(ascending=False)
    nnz_ranks = [z_ranks[i].index[[z_ranks[i] <= top]].values for i in ii]

    nn = np.concatenate(nnz_ranks)

    nncol = plierRes.B.index[plierRes.Z.columns.isin(ii)].repeat(
        [len(_) for _ in nnz_ranks]
    )

    nnpath = pd.concat(
        [
            priorMat.loc[x, plierRes.U.loc[plierRes.U.loc[:, y] > 0, y].index].sum(
                axis=1
            )
            > 0
            for x, y in zip(nnz_ranks, ii)
        ],
        axis=0,
    )

    nnindex = ii.repeat([len(_) for _ in nnz_ranks])

    nnrep = np.unique(nn)[np.unique(nn, return_counts=True)[1] > 1]

    if len(nnrep) > 0:
        nnrep_im = np.intersect1d(nn, nnrep, return_indices=True)[1]
        nn = np.delete(nn, nnrep_im)
        nncol = np.delete(nncol, nnrep_im)
        nnpath = nnpath.iloc[[_ for _ in range(len(nnpath)) if _ not in nnrep_im]]
        nnindex = np.delete(nnindex, nnrep_im)
        nnpath.replace({True: "inPathway", False: "notInPathway"}, inplace=True)

    nnpath.replace({True: "inPathway", False: "notInPathway"}, inplace=True)

    pathway_gene_index = pd.MultiIndex.from_tuples(
        list(zip(*[nncol.tolist(), nn.tolist()])), names=["pathway", "gene"]
    )

    nncol = pd.DataFrame({"pathway": nncol, "present": nnpath})

    toplot = data.loc[nn, :]

    if regress:
        for i in tqdm(ii):
            gi = np.where(nnindex == i)[0]
            toplot.iloc[gi, :] = (
                sm.GLS(
                    toplot.iloc[gi, :].T,
                    plierRes.B.drop(index=plierRes.B.index[0]).loc[:, toplot.columns].T,
                )
                .fit()
                .resid.T
            )

    present_lut = {
        "inPathway": mcolors.to_rgb(mcolors.CSS4_COLORS["black"]),
        "notInPathway": mcolors.to_rgb(mcolors.CSS4_COLORS["beige"]),
    }

    pathway_pal = sns.color_palette(
        annotation_row_palette, n_colors=pathway_labels.unique().size
    )
    if shuffle_pathway_pal:
        np.random.shuffle(pathway_pal)
    pathway_lut = {x: y for x, y in zip(nncol["pathway"].unique(), pathway_pal)}

    nncol = nncol.replace({True: "inPathway", False: "notinPathway"})
    row_annotations = pd.DataFrame(
        {
            "pathway": nncol["pathway"].map(pathway_lut),
            "present": nncol["present"].map(present_lut),
        }
    ).set_index(pathway_gene_index)

    pathway_labels = row_annotations.index.get_level_values("pathway")

    g = sns.clustermap(
        data=toplot,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        cmap=colormap,
        robust=True,
        linecolor="black",
        row_colors=row_annotations.droplevel("pathway"),
        figsize=figsize,
        standard_scale=scale,
        xticklabels=False,
        **kwargs,
    )
    g.ax_col_dendrogram.set_visible(show_col_dendrogram)
    g.ax_row_dendrogram.set_visible(show_row_dendrogram)
    g.ax_cbar.set_visible(show_cbar)

    if show_legend:
        _ = g.ax_heatmap.legend(
            handles=[Patch(facecolor=pathway_lut[name]) for name in pathway_lut],
            labels=pathway_lut.keys(),
            ncol=1,
            loc="lower left",
            bbox_to_anchor=(1.05, 0.25),
        )

    return g


def plotTopZallPath(
    plierRes: PLIERRes,
    data: pd.DataFrame,
    priorMat: pd.DataFrame,
    top: int = 10,
    index: Optional[str] = None,
    regress: bool = False,
    fdr_cutoff: float = 0.2,
    *args,
    **kwargs,
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

    z_ranks = plierRes.Z.loc[:, ii].rank(axis=0, ascending=False)
    Ustrict = plierRes.U.copy()
    Ustrict[plierRes.Up > pval_cutoff] = 0
    pathsUsed = Ustrict[Ustrict.loc[:, ii].sum(axis=1) > 0].index
    pathMat = np.zeros((0, len(pathsUsed)))

    nntmp = {i: z_ranks.index[np.where(z_ranks[i] <= top)[0]].values for i in ii}
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
    **kwargs,
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
            **kwargs,
        )
    else:
        plotMat(U, *args, **kwargs)


def replace_below_top(sr: pd.Series, n: int = 3, replace_val: int = 0) -> pd.Series:
    sr[sr < sr.nlargest(n)[-1]] = replace_val
    return sr
