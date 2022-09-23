from typing import TypeVar

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.patches import Patch
from tqdm.auto import tqdm
from typeguard import typechecked

from pyplier.plier_res import PLIERResults
from pyplier.utils import zscore

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")
ClusterGrid = TypeVar("ClusterGrid", bound="sns.matrix.ClusterGrid")


@typechecked
def plotMat(
    mat: pd.DataFrame,
    scale: bool = True,
    trim_names: int = 50,
    cutoff: float | None = None,
    *args,
    **kwargs,
) -> dict[str, npt.arraylike]:
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
    prior_mat: pd.DataFrame,
    top: int = 10,
    index: list[str] | None = None,
    regress: bool = False,
    allLVs: bool = False,
    colormap: str = "viridis",
    annotation_row_palette: str = "hls",
    scale: int | None = None,
    cluster_cols: bool = True,
    show_col_dendrogram: bool = False,
    cluster_rows: bool = False,
    show_row_dendrogram: bool = False,
    show_cbar: bool = False,
    show_legend: bool = True,
    shuffle_pathway_pal: bool = False,
    figsize: tuple[int] = (9, 9),
    persistent_progress: bool = True,
    disable_progress: bool = False,
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
    persistent_progress : bool
        Should the progress bar progress be kept after completion? Defaults to True.
    disable_progress :
        Should progress bars be disabled? Defaults to False.
    """

    # subset the original data matrix and the prior pathway matrix on the genes that were used by `PLIER`
    data = data.loc[plierRes.z.index, :]
    prior_mat = prior_mat.loc[plierRes.z.index.intersection(prior_mat.index), :]
    plierRes.u.columns[np.where(plierRes.u.sum(axis=0) > 0)]

    if allLVs:
        if index is not None:
            ii = plierRes.u.columns.intersection(
                index
            )  # use `intersection` so we don't have an issue with trying to plot non-existent LVs
        else:
            ii = plierRes.u.columns
    elif index is not None:
        ii = plierRes.u.columns[plierRes.u.sum() > 0].intersection(index)
    else:
        ii = plierRes.u.columns[plierRes.u.sum() > 0]

    z_ranks = plierRes.z.loc[:, ii].rank(ascending=False)
    nnz_ranks = [z_ranks[i].index[[z_ranks[i] <= top]].values for i in ii]

    nn = np.concatenate(nnz_ranks)

    nncol = plierRes.b.index[plierRes.z.columns.isin(ii)].repeat([len(_) for _ in nnz_ranks]).to_series()

    nnpath = pd.concat(
        [
            prior_mat.loc[x, plierRes.u.loc[plierRes.u.loc[:, y] > 0, y].index].sum(axis=1) > 0
            for x, y in zip(nnz_ranks, ii, strict=False)
        ],
        axis=0,
    )

    nnindex = ii.repeat([len(_) for _ in nnz_ranks])

    nnrep = np.unique(nn)[np.unique(nn, return_counts=True)[1] > 1]

    if len(nnrep) > 0:
        nnrep_im = np.intersect1d(nn, nnrep, return_indices=True)[1]
        nn = np.delete(nn, nnrep_im)
        nncol = nncol.iloc[[_ for _ in range(len(nncol)) if _ not in nnrep_im]]
        nnpath = nnpath.iloc[[_ for _ in range(len(nnpath)) if _ not in nnrep_im]]
        nnindex = np.delete(nnindex, nnrep_im)

    nnpath.replace({True: "inPathway", False: "notInPathway"}, inplace=True)

    toplot = data.loc[nn, :]

    if regress:
        for i in tqdm(ii, disable=disable_progress, leave=persistent_progress, desc="Performing regression"):
            gi = np.where(nnindex == i)[0]
            toplot.iloc[gi, :] = (
                sm.GLS(
                    toplot.iloc[gi, :].T,
                    plierRes.b.drop(index=plierRes.b.index[0]).loc[:, toplot.columns].T,
                )
                .fit()
                .resid.T
            )

    present_lut = {
        "inPathway": mcolors.to_rgb(mcolors.CSS4_COLORS["black"]),
        "notInPathway": mcolors.to_rgb(mcolors.CSS4_COLORS["beige"]),
    }

    pathway_pal = sns.color_palette(annotation_row_palette, n_colors=nncol.unique().size)
    if shuffle_pathway_pal:
        np.random.shuffle(pathway_pal)
    pathway_lut = dict(zip(nncol.unique(), pathway_pal, strict=False))

    row_annotations = pd.DataFrame(
        {
            "pathway": nncol.map(pathway_lut).values,
            "present": nnpath.map(present_lut).values,
        },
        index=nnpath.index,
    )

    g = sns.clustermap(
        data=toplot,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        cmap=colormap,
        robust=True,
        linecolor="black",
        row_colors=row_annotations,
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
    index: str | None = None,
    regress: bool = False,
    fdr_cutoff: float = 0.2,
    persistent_progress: bool = True,
    disable_progress: bool = False,
    *args,
    **kwargs,
) -> None:
    """
    visualize the top genes contributing to the LVs similarily to [plotTopZ()].
        However in this case all the pathways contributing to each LV are shown
        seperately. Useful for seeing pathway usage for a single LV or
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
    persistent_progress : bool
        Should the progress bar progress be kept after completion? Defaults to True.
    disable_progress :
        Should progress bars be disabled? Defaults to False.
    *args, **kwargs: Additional arguments to be passed to pheatmap, such as a
        column annotation data.frame
    """
    pval_cutoff = plierRes.summary.loc[plierRes.summary["FDR"] < fdr_cutoff, "p-value"].max()

    ii = (plierRes.u.sum(axis=0) > 0).index
    if index is not None:
        ii = ii.intersection(index)

    z_ranks = plierRes.z.loc[:, ii].rank(axis=0, ascending=False)
    ustrict = plierRes.u.copy()
    ustrict[plierRes.up > pval_cutoff] = 0
    paths_used = ustrict[ustrict.loc[:, ii].sum(axis=1) > 0].index
    path_mat = np.zeros((0, len(paths_used)))

    nntmp = {i: z_ranks.index[np.where(z_ranks[i] <= top)[0]].values for i in ii}
    nn = np.concatenate(list(nntmp.values()))
    nncol = {
        i: plierRes.u.index[plierRes.u.loc[:, i] == plierRes.u.loc[:, i].max()].repeat(len(nntmp[i]))
        if plierRes.u.loc[:, i].max() > 0
        else pd.Index([i]).repeat(len(nntmp[i]))
        for i in ii
    }
    nnindex = np.concatenate([list(_) for _ in nncol.values()])
    path_mat = priorMat.loc[np.concatenate(list(nntmp.values())), paths_used]

    if any(path_mat.sum() > 1):
        path_mat.loc[:, path_mat.sum() > 0]
        paths_used = path_mat.columns

    path_mat = path_mat.astype("category")

    if regress:
        to_plot = data.loc[nn,]

        for i in tqdm(ii, disable=disable_progress, leave=persistent_progress, desc="Performing regression"):
            gi = np.where(nnindex == i)[0]
            to_plot.iloc[gi, :] = (
                sm.GLS(
                    to_plot.iloc[gi, :].T,
                    plierRes.b.drop(index=plierRes.b.index[0]).loc[:, to_plot.columns].T,
                )
                .fit()
                .resid.T
            )

    annotation_row = path_mat.replace(
        {
            0: mcolors.to_rgb(mcolors.CSS4_COLORS["beige"]),
            1: mcolors.to_rgb(mcolors.CSS4_COLORS["black"]),
        }
    )

    sns.clustermap(data.loc[nn, :].apply(zscore, axis=1), row_colors=annotation_row, *args, **kwargs)


def plotU(
    plierRes: PLIERRes,
    auc_cutoff: float = 0.6,
    fdr_cutoff: float = 0.05,
    index_col: list[str] | None = None,
    index_row: list[str] | None = None,
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
    if index_col is None:
        index_col = plierRes.u.columns

    if index_row is None:
        index_row = plierRes.u.index

    u = plierRes.u.copy()
    pval_cutoff = plierRes.summary.loc[plierRes.summary["FDR"] < fdr_cutoff, "p-value"].max()
    u[plierRes.uauc < auc_cutoff] = 0
    u[plierRes.up > pval_cutoff] = 0

    u = u.loc[index_row, index_col].apply(replace_below_top, n=top, replace_val=0)

    if sort_row:
        um = pd.DataFrame({x: np.multiply((u.columns.get_loc(x) + 1) * 100, np.sign(y)) for x, y in u.iteritems()}).max(
            axis=1
        )
        plotMat(
            u.loc[um.rank(ascending=False).sort_values().index],
            row_cluster=False,
            *args,
            **kwargs,
        )
    else:
        plotMat(u, *args, **kwargs)


def replace_below_top(sr: pd.Series, n: int = 3, replace_val: int = 0) -> pd.Series:
    sr[sr < sr.nlargest(n)[-1]] = replace_val
    return sr
