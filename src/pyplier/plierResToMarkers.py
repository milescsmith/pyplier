from typing import TypeVar

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# from .logger import plier_logger
from .PLIERRes import PLIERResults

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")


def plierResToMarkers(
    plierRes: PLIERRes, priorMat: pd.DataFrame, num: int = 20, index: list[str] = None
) -> pd.DataFrame:
    ii = plierRes.U.columns[
        np.where(plierRes.U.sum(axis=0) > 0)
    ]  # ii <- which(colsums(plierRes$U, parallel = TRUE) > 0)

    if index is not None:
        ii = np.intersect1d(ii, index)
    # if !is.null(index):
    #   ii <- intersect(ii, index)

    Zuse = plierRes.Z.loc[:, ii]  # Zuse <- plierRes$Z[, ii, drop = F]

    # for (i in seq_along(ii)):
    #   lv <- ii[i]
    #   paths <- names(which(plierRes$U[, lv] < 0.01))
    #   genes <- names(which(rowsums(x = priorMat[, paths], parallel = TRUE) > 0))
    #   genesNotInPath <- setdiff(rownames(Zuse), genes)
    #   Zuse[genesNotInPath, i] <- 0

    for i in tqdm(ii):
        paths = plierRes.U.index[np.where(plierRes.U.loc[:, i] < 0.01)[0]].values
        genes = priorMat[(priorMat.loc[:, paths].sum(axis=1) > 0)].index.values
        genesNotInPath = Zuse.index[~Zuse.index.isin(genes)]
        Zuse.loc[genesNotInPath, i] = 0

    tag = Zuse.rank(ascending=False)  # tag <- apply(-Zuse, 2, rank)
    tag.columns = plierRes.B.index[
        np.where(plierRes.U.sum(axis=0) > 0)
    ]  # colnames(tag) <- rownames(plierRes$B)[ii]
    iim = tag.min(axis=1)  # iim <- apply(tag, 1, min)
    iig = iim.index[np.where(iim <= num)[0]]  # iig <- which(iim <= num)
    tag = tag.loc[iig, :]  # tag <- tag[iig, ]
    iin = tag.apply(lambda x: x <= num).sum(
        axis=1
    )  # iin <- rowsums(tag <= num, parallel = TRUE)
    iimulti = iin.index[np.where(iin > 1)[0]]  # iimulti <- which(iin > 1)
    if len(iimulti) > 0:
        print(f"Genes not matched uniquely: {', '.join(iimulti.values)}")

    # if len(iimulti) > 0:
    #     message(paste0("Genes not matched uniquely: ", paste(names(iimulti), collapse = ", ")))

    tag = tag.apply(lambda x: x <= num).astype(int)  # tag <- (tag <= num) + 1 - 1

    return tag
