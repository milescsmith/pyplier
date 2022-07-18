from typing import Optional, TypeVar
import pandas as pd
import numpy as np
from .PLIERRes import PLIERResults
from .plotMat import plotMat
import seaborn as sns

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")


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
        Um = pd.DataFrame({x: np.multiply((U.columns.get_loc(x) + 1) * 100, np.sign(y)) for x, y in U.iteritems()}).max(axis=1)
        plotMat(U.loc[Um.rank(ascending=False).sort_values().index], row_cluster=False, *args, **kwargs)
    else:
        plotMat(U, *args, **kwargs)


def replace_below_top(sr: pd.Series, n: int = 3, replace_val: int = 0) -> pd.Series:
    sr[sr < sr.nlargest(n)[-1]] = replace_val
    return sr
