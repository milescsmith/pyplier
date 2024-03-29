import random
from math import ceil, floor
from typing import TypeVar, Dict, Literal

import numpy as np
import pandas as pd
from icontract import require
from numpy.random import default_rng
from rich import print as rprint
from scipy.linalg import solve, svd
from sklearn.utils.extmath import randomized_svd
from tqdm.auto import tqdm, trange

from .AUC import getAUC
from .crossVal import crossVal
from .nameB import nameB
from .num_pc import num_pc
from .PLIERRes import PLIERResults
from .regression import pinv_ridge
from .solveU import solveU
from .utils import crossprod, zscore, setdiff, tcrossprod

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")


@require(lambda pathwaySelection: pathwaySelection in ("complete", "fast"))
def PLIER(
    data: pd.DataFrame,  # for anndata objects, this will need to be transposed
    priorMat: pd.DataFrame,
    svdres: Dict[str, np.ndarray] = None,
    num_LVs: float = None,
    L1: float = None,
    L2: float = None,
    L3: float = None,
    frac: float = 0.7,
    max_iter: int = 350,
    trace: bool = False,
    scale: bool = True,
    Chat: np.ndarray = None,
    maxPath: int = 10,
    doCrossval: bool = True,
    penalty_factor: np.ndarray = None,
    glm_alpha: float = 0.9,
    minGenes: int = 10,
    tol: float = 1e-06,
    seed: int = 123456,
    allGenes: bool = False,
    rseed: int = None,
    pathwaySelection: Literal['complete', 'fast'] = "complete",
    persistent_progress: bool = True,
    disable_progress: bool = False,
) -> PLIERRes:
    """ "Main PLIER function

    Parameters
    ----------
    data : :class:`~pd.DataFrame`
        the data to be processed with genes in rows and samples in columns.
        Should be z-scored or set scale=True
    priorMat : :class:`~pd.DataFrame`
        the binary prior information matrix with genes in rows and
        pathways/genesets in columns
    svdres : dict[str, :class:`~np.ndarray`], optional
        Pre-computed result of the svd decomposition for data, by default None
    num_LVs : float, optional
        The number of latent variables to return, leave as None to be set
        automatically using the num_pc 'elbow' method, by default None
    L1 : float, optional
        L1 constant, leave as None to automatically select a value., by default None
    L2 : float, optional
        L2 constant, leave as None to automatically select a value, by default None
    L3 : float, optional
        L3 constant, leave as None to automatically select a value. Sparsity in U should be instead controlled by setting frac, by default None
    frac : float, optional
        The fraction of LVs that should have at least 1 prior inforamtion association, used to automatically set L3, by default 0.7
    max_iter : int, optional
        Maximum number of iterations to perform, by default 350
    trace : bool, optional
        Display progress information, by default False
    scale : bool, optional
        Z-score the data before processing, by default True
    Chat : :class:`~np.ndarray`, optional
        A ridge inverse of priorMat, used to select active pathways, expensive to compute so can be precomputed when running PLIER multiple times. Defaults to None.
    maxPath : int, optional
        The maximum number of active pathways per latent variable. Defaults to 10.
    doCrossval : bool, optional
        Whether or not to do real cross-validation with held-out pathway genes. Alternatively, all gene annotations are used and only pseudo-crossvalidation is done. The latter option may be preferable if some pathways of interest have few genes. Defaults to True.
    penalty_factor : np.ndarray, optional
        A vector equal to the number of columns in priorMat. Sets relative penalties for different pathway/geneset subsets. Lower penalties will make a pathway more likely to be used. Only the relative values matter. Internally rescaled. Defaults to None.
    glm_alpha : float, optional
        Set the alpha for elastic-net. Defaults to 0.9.
    minGenes : int, optional
        The minimum number of genes a pathway must have to be considered. Defaults to 10.
    tol : float, optional
        Convergence threshold. Defaults to 1e-06.
    seed : int, optional
        Set the seed for pathway cross-validation. Defaults to 123456.
    allGenes : bool, optional
        Use all genes. By default only genes in the priorMat matrix are used. Defaults to False.
    rseed : int, optional
        Set this option to use a random initialization, instead of SVD. Defaults to None.
    pathwaySelection : str, optional
        Pathways to be optimized with elstic-net penalty are preselected based on ridge regression results. 'Complete' uses all top pathways to fit individual LVs. 'Fast' uses only the top pathways for the single LV in question. Defaults to "complete".
    persistent_progress : bool
        Should the progress bar progress be kept after completion? Defaults to True.
    disable_progress : 
        Should progress bars be disabled? Defaults to False.

    Returns
    -------
    :class:`pyplier.PLIERResults`
        Object containing the following:
        - B
        - Z
        - U
        - C
        - L1
        - L2
        - L3
        - heldOutGenes
        - withPrior
        - Uauc
        - Up
        - summary
        - residual
    """

    if penalty_factor is None:
        penalty_factor = np.ones(priorMat.shape[1])

    if scale:
        Y = data.apply(zscore, axis=1)
    else:
        Y = data

    if (priorMat.shape[0] != data.shape[0]) or not all(priorMat.index == data.index):
        if not allGenes:
            cm = data.index.intersection(priorMat.index)
            rprint(f"Selecting common genes: {len(cm)}")
            priorMat = priorMat.loc[cm, :]
            Y = Y.loc[cm, :]
        else:
            extra_genes = setdiff(data.index, priorMat.index)
            eMat = pd.DataFrame(
                data=np.zeros((len(extra_genes), priorMat.shape[1])),
                columns=priorMat.columns,
                index=extra_genes,
            )
            priorMat = pd.concat([priorMat, eMat], axis=0)
            priorMat = priorMat.loc[data.index, :]

    numGenes = priorMat.sum(axis="rows")  # colsums

    heldOutGenes = dict()
    iibad = numGenes[numGenes < minGenes].index
    priorMat.loc[:, iibad] = 0
    rprint(f"Removing {len(iibad)} pathways with too few genes")
    if doCrossval:
        priorMatCV = priorMat.copy(deep=True)
        if seed is not None:
            random.seed(seed)
        for j in tqdm(priorMatCV.columns, disable=disable_progress, leave=persistent_progress, desc="Performing priorMat crossval"):
            iipos = priorMatCV.loc[:, j].where(lambda x: x > 0).dropna().index
            iiposs = random.sample(list(iipos), k=round(len(iipos) / 5))
            priorMatCV.loc[iiposs, j] = 0
            heldOutGenes[j] = list(iiposs)
        C = priorMatCV.copy(deep=True)
    else:
        C = priorMat.copy(deep=True)

    ns = data.shape[1]
    Bdiff = -1
    BdiffTrace = np.ndarray((0,), dtype=np.float64)
    BdiffCount = 0
    if Chat is None:
        Cp = crossprod(C)
        Chat = pinv_ridge(Cp, 5) @ C.transpose()
    # compute svd and use that as the starting point

    if (svdres is not None) and (svdres["v"].shape[1] != Y.shape[1]):
        rprint("SVD V has the wrong number of columns")
        svdres = None

    Y_arr = np.nan_to_num(Y)
    Y_arr = np.where(np.isposinf(Y_arr), np.finfo(np.float32).max, Y_arr)
    Y_arr = np.where(np.isneginf(Y_arr), np.finfo(np.float32).min, Y_arr)

    if svdres is None:
        svdres = dict()
        rprint("Computing SVD")
        if ns > 500:
            rprint("Using rsvd")
            # TODO: we *have* to accelerate this using pytorch or the like
            # sklearn is fragile and cannot handle it and a 30k x 30k matrix is killing it
            svdres["u"], svdres["d"], svdres["v"] = randomized_svd(
                M=Y_arr,
                n_components=ceil(min(ns, max(200, ns / 4))),
                n_iter=3,
                random_state=None,
            )
        else:
            svdres["u"], svdres["d"], svdres["v"] = svd(
                Y_arr, lapack_driver="gesdd"
            )  # the gesvd driver flips the sign for components > 6 in the v matrix as compared to R's svd function
        rprint("Done")
        svdres["v"] = svdres[
            "v"
        ].transpose()  # as compared to the output from R's svd, the v matrix is transposed.  Took me too long to figure this one out.

    if num_LVs is None:
        num_LVs = num_pc(svdres) * 2
        num_LVs = min(num_LVs, floor(Y_arr.shape[1] * 0.9))
        rprint(f"The number of LVs is set to {num_LVs}")

    if L2 is None:
        L2 = svdres["d"][num_LVs]
        rprint(f"L2 is set to {L2}")

    if L1 is None:
        L1 = L2 / 2
        rprint(f"L1 is set to {L1}")

    B = (
        svdres["v"][0 : Y_arr.shape[1], 0:num_LVs] @ np.diag(svdres["d"][0:num_LVs])
    ).transpose()

    # following two lines are equivalent to R's diag(x)
    # numpy.fill_diagonal modifies in place and does not
    # return a value, thus this workaround
    diag_mat = np.zeros((num_LVs, num_LVs))
    np.fill_diagonal(diag_mat, val=1)

    # for R's solve(), if b is missing, it uses the identity matrix of a
    # scipy.linalg.solve does not have a default for b, so just give it one
    Z = pd.DataFrame(
        np.dot(np.dot(Y, B.T), solve(a=np.dot(B, B.T) + L1 * diag_mat, b=diag_mat)),
        index=Y.index,
    )

    Z = Z.where(cond=lambda x: x > 0, other=0)

    if rseed is not None:
        rprint("using random start")
        random.seed(rseed)

        rng = default_rng()
        rng.shuffle(B, axis=1)  # B = t(apply(B, 1, sample))
        rng.shuffle(Z, axis=0)  # Z = apply(Z, 2, sample)
        Z = Z.transpose()

    U = np.zeros((C.shape[1], num_LVs))  # matrix(0, nrow = ncol(C), ncol = num_LVs)

    rprint(
        f"errorY (SVD based:best possible) = {((Y - np.dot(Z, B))**2).to_numpy().mean():.4f}"
    )

    iter_full_start = iter_full = 20

    if L3 is not None:
        L3_given = True
    else:
        L3_given = False

    for i in trange(max_iter, disable=disable_progress, leave=persistent_progress, desc="Calculating U"):
        if i >= iter_full_start:
            if i == iter_full and not L3_given:
                # update L3 to the target fraction
                Ulist = solveU(
                    Z=Z,
                    Chat=Chat,
                    priorMat=C,
                    penalty_factor=penalty_factor,
                    pathwaySelection=pathwaySelection,
                    glm_alpha=glm_alpha,
                    maxPath=maxPath,
                    disable_progress=disable_progress,
                    persistent_progress=persistent_progress,
                    target_frac=frac,
                )
                U = Ulist["U"]
                L3 = Ulist["L3"]
                rprint(f"New L3 is {L3}")
                iter_full = iter_full + iter_full_start
            else:
                U = solveU(
                    Z=Z,
                    Chat=Chat,
                    priorMat=C,
                    penalty_factor=penalty_factor,
                    pathwaySelection=pathwaySelection,
                    glm_alpha=glm_alpha,
                    maxPath=maxPath,
                    disable_progress=disable_progress,
                    persistent_progress=persistent_progress,
                    L3=L3,
                )["U"]

            Z1 = tcrossprod(Y, B)
            Z2 = L1 * (C @ U)

            Z1_nonzero = np.argwhere(np.asarray(Z1.T.stack()) > 0).flatten()
            Z2_nonzero = np.argwhere(np.asarray(Z2.T.stack()) > 0).flatten()

            ratio = np.median(
                np.divide(Z2, np.asarray(Z1))
                .T.stack()
                .values[np.intersect1d(Z2_nonzero, Z1_nonzero)]
            )

            Z = (Z1 + Z2) @ solve(a=(tcrossprod(B) + L1 * diag_mat), b=diag_mat)
        else:
            Z = tcrossprod(Y, B) @ solve(a=(tcrossprod(B) + L1 * diag_mat), b=diag_mat)

        Z[Z < 0] = 0

        oldB = B.copy()
        B = solve(a=(Z.transpose() @ Z + L2 * diag_mat), b=diag_mat) @ Z.transpose() @ Y

        Bdiff = ((B - oldB) ** 2).to_numpy().sum() / (B**2).to_numpy().sum()
        BdiffTrace = np.append(BdiffTrace, Bdiff)

        if trace & (i >= iter_full_start):
            rprint(
                f"iter {i} errorY = {np.mean((Y - Z @ B)**2):.4f} prior information ratio= {round(ratio,2)} Bdiff = {Bdiff:.4f} Bkappa= {np.linalg.cond(B):.4f};pos. col. U = {sum(U.sum(axis='index') > 0)}"
            )
        elif trace:
            rprint(
                f"iter {i} errorY = {np.mean((Y - Z @ B)**2):.4f} Bdiff = {np.linalg.cond(Bdiff):.4f} Bkappa = {np.linalg.cond(B):.4f}"
            )

        if (i > 52) and (Bdiff > BdiffTrace[i - 50]):
            BdiffCount += 1
            rprint("Bdiff is not decreasing")
        elif BdiffCount > 1:
            BdiffCount -= 1

        if Bdiff < tol:
            rprint(f"converged at iteration {i}")
            break
        if BdiffCount > 5:
            rprint(f"converged at iteration {i} Bdiff is not decreasing")
            break

    U.index = priorMat.columns
    U.columns = [f"LV{_+1}" for _ in range(num_LVs)]
    Z.columns = [f"LV{_+1}" for _ in range(num_LVs)]

    B.index = [f"LV{_+1}" for _ in range(num_LVs)]

    out = PLIERResults(
        residual=(Y - (Z.values @ B.values)),
        B=B,
        Z=Z,
        U=U,
        C=C,
        L1=L1,
        L2=L2,
        L3=L3,
        heldOutGenes=heldOutGenes,
    )

    if doCrossval:
        outAUC = crossVal(plierRes=out, priorMat=priorMat, priorMatcv=priorMatCV, disable_progress=disable_progress, persistent_progress=persistent_progress)
    else:
        rprint("Not using cross-validation. AUCs and p-values may be over-optimistic")
        outAUC = getAUC(out, Y, priorMat)

    out.withPrior = U.sum(axis="index")[U.sum(axis="index") > 0].to_dict()
    out.Uauc = outAUC["Uauc"]
    out.Up = outAUC["Upval"]
    out.summary = outAUC["summary"]
    tt = out.Uauc.max(axis="index")
    rprint(f"There are {sum(tt > 0.7)} LVs with AUC > 0.70")

    out.B.index = nameB(out)

    return out
