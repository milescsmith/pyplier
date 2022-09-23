import random
from math import ceil, floor
from typing import Literal, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from icontract import require
from numpy.random import default_rng
from rich import print as rprint
from scipy.linalg import solve, svd
from sklearn.utils.extmath import randomized_svd
from tqdm.auto import tqdm, trange

from pyplier.auc import getAUC
from pyplier.cross_val import crossVal
from pyplier.name_b import nameB
from pyplier.num_pc import num_pc
from pyplier.plier_res import PLIERResults
from pyplier.regression import pinv_ridge
from pyplier.solve_u import solveU
from pyplier.utils import crossprod, setdiff, tcrossprod, zscore

PLIERRes = TypeVar("PLIERRes", bound="PLIERResults")

AUC_CUTOFF = 0.7
MIN_BDIFF_COUNT = 5
MIN_ITERATIONS = 52
LARGE_NUMBER_OF_COLUMNS = 500


@require(lambda pathwaySelection: pathwaySelection in ("complete", "fast"))
def PLIER(
    data: pd.DataFrame,  # for anndata objects, this will need to be transposed
    prior_mat: pd.DataFrame,
    svdres: dict[str, npt.arraylike] | None = None,
    num_lvs: float | None = None,
    l1: float | None = None,
    l2: float | None = None,
    l3: float | None = None,
    frac: float = 0.7,
    max_iter: int = 350,
    trace: bool = False,
    scale: bool = True,
    chat: npt.arraylike | None = None,
    max_path: int = 10,
    doCrossval: bool = True,
    penalty_factor: npt.arraylike | None = None,
    glm_alpha: float = 0.9,
    min_genes: int = 10,
    tol: float = 1e-06,
    seed: int = 123456,
    all_genes: bool = False,
    rseed: int | None = None,
    pathway_selection: Literal["complete", "fast"] = "complete",
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
    svdres : dict[str, :class:`~npt.arraylike`], optional
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
    Chat : :class:`~npt.arraylike`, optional
        A ridge inverse of priorMat, used to select active pathways, expensive to compute so can be precomputed when running PLIER multiple times. Defaults to None.
    maxPath : int, optional
        The maximum number of active pathways per latent variable. Defaults to 10.
    doCrossval : bool, optional
        Whether or not to do real cross-validation with held-out pathway genes. Alternatively, all gene annotations are used and only pseudo-crossvalidation is done. The latter option may be preferable if some pathways of interest have few genes. Defaults to True.
    penalty_factor : npt.arraylike, optional
        A vector equal to the number of columns in priorMat. Sets relative penalties for different pathway/geneset subsets. Lower penalties will make a pathway more likely to be used. Only the relative values matter. Internally rescaled. Defaults to None.
    glm_alpha : float, optional
        Set the alpha for elastic-net. Defaults to 0.9.
    min_genes : int, optional
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
        - b
        - z
        - u
        - c
        - l1
        - l2
        - l3
        - held_out_genes
        - with_prior
        - uauc
        - up
        - summary
        - residual
    """

    if penalty_factor is None:
        penalty_factor = np.ones(prior_mat.shape[1])

    y = data.apply(zscore, axis=1) if scale else data
    if (prior_mat.shape[0] != data.shape[0]) or not all(prior_mat.index == data.index):
        if all_genes:
            extra_genes = setdiff(data.index, prior_mat.index)
            e_mat = pd.DataFrame(
                data=np.zeros((len(extra_genes), prior_mat.shape[1])),
                columns=prior_mat.columns,
                index=extra_genes,
            )
            prior_mat = pd.concat([prior_mat, e_mat], axis=0)
            prior_mat = prior_mat.loc[data.index, :]

        else:
            cm = data.index.intersection(prior_mat.index)
            rprint(f"Selecting common genes: {len(cm)}")
            prior_mat = prior_mat.loc[cm, :]
            y = y.loc[cm, :]
    num_genes = prior_mat.sum(axis="rows")  # colsums

    held_out_genes = {}
    iibad = num_genes[num_genes < min_genes].index
    prior_mat.loc[:, iibad] = 0
    rprint(f"Removing {len(iibad)} pathways with too few genes")
    if doCrossval:
        prior_mat_cv = prior_mat.copy(deep=True)
        if seed is not None:
            random.seed(seed)
        for j in tqdm(
            prior_mat_cv.columns,
            disable=disable_progress,
            leave=persistent_progress,
            desc="Performing prior_mat crossval",
        ):
            iipos = prior_mat_cv.loc[:, j].where(lambda x: x > 0).dropna().index
            iiposs = random.sample(list(iipos), k=round(len(iipos) / 5))
            prior_mat_cv.loc[iiposs, j] = 0
            held_out_genes[j] = list(iiposs)
        c = prior_mat_cv.copy(deep=True)
    else:
        c = prior_mat.copy(deep=True)

    ns = data.shape[1]
    bdiff = -1
    bdiff_trace = np.ndarray((0,), dtype=np.float64)
    bdiff_count = 0
    if chat is None:
        cp = crossprod(c)
        chat = pinv_ridge(cp, 5) @ c.transpose()
    # compute svd and use that as the starting point

    if (svdres is not None) and (svdres["v"].shape[1] != y.shape[1]):
        rprint("SVD V has the wrong number of columns")
        svdres = None

    y_arr = np.nan_to_num(y)
    y_arr = np.where(np.isposinf(y_arr), np.finfo(np.float32).max, y_arr)
    y_arr = np.where(np.isneginf(y_arr), np.finfo(np.float32).min, y_arr)

    if svdres is None:
        svdres = {}
        rprint("Computing SVD")
        if ns > LARGE_NUMBER_OF_COLUMNS:
            rprint("Using rsvd")
            # TODO: we *have* to accelerate this using pytorch or the like
            # sklearn is fragile and cannot handle it and a 30k x 30k matrix is killing it
            svdres["u"], svdres["d"], svdres["v"] = randomized_svd(
                M=y_arr,
                n_components=ceil(min(ns, max(200, ns / 4))),
                n_iter=3,
                random_state=None,
            )
        else:
            svdres["u"], svdres["d"], svdres["v"] = svd(
                y_arr, lapack_driver="gesdd"
            )  # the gesvd driver flips the sign for components > 6 in the v matrix as compared to R's svd function
        rprint("Done")
        svdres["v"] = svdres[
            "v"
        ].transpose()  # as compared to the output from R's svd, the v matrix is transposed.  Took me too long to figure this one out.

    if num_lvs is None:
        num_lvs = num_pc(svdres) * 2
        num_lvs = min(num_lvs, floor(y_arr.shape[1] * 0.9))
        rprint(f"The number of LVs is set to {num_lvs}")

    if l2 is None:
        l2 = svdres["d"][num_lvs]
        rprint(f"L2 is set to {l2}")

    if l1 is None:
        l1 = l2 / 2
        rprint(f"L1 is set to {l1}")

    b = (svdres["v"][0 : y_arr.shape[1], 0:num_lvs] @ np.diag(svdres["d"][:num_lvs])).transpose()

    # following two lines are equivalent to R's diag(x)
    # numpy.fill_diagonal modifies in place and does not
    # return a value, thus this workaround
    diag_mat = np.zeros((num_lvs, num_lvs))
    np.fill_diagonal(diag_mat, val=1)

    # for R's solve(), if b is missing, it uses the identity matrix of a
    # scipy.linalg.solve does not have a default for b, so just give it one
    z = pd.DataFrame(
        np.dot(np.dot(y, b.T), solve(a=np.dot(b, b.T) + l1 * diag_mat, b=diag_mat)),
        index=y.index,
    )

    z = z.where(cond=lambda x: x > 0, other=0)

    if rseed is not None:
        rprint("using random start")
        random.seed(rseed)

        rng = default_rng()
        rng.shuffle(b, axis=1)  # B = t(apply(B, 1, sample))
        rng.shuffle(z, axis=0)  # Z = apply(Z, 2, sample)
        z = z.transpose()

    u = np.zeros((c.shape[1], num_lvs))  # matrix(0, nrow = ncol(C), ncol = num_LVs)

    rprint(f"errorY (SVD based:best possible) = {((y - np.dot(z, b))**2).to_numpy().mean():.4f}")

    iter_full_start = iter_full = 20

    l3_given = l3 is not None
    for i in trange(max_iter, disable=disable_progress, leave=persistent_progress, desc="Calculating U"):
        if i >= iter_full_start:
            if i == iter_full and not l3_given:
                # update L3 to the target fraction
                u_list = solveU(
                    z=z,
                    chat=chat,
                    priorMat=c,
                    penalty_factor=penalty_factor,
                    pathway_selection=pathway_selection,
                    glm_alpha=glm_alpha,
                    max_path=max_path,
                    disable_progress=disable_progress,
                    persistent_progress=persistent_progress,
                    target_frac=frac,
                )
                u = u_list["U"]
                l3 = u_list["L3"]
                rprint(f"New L3 is {l3}")
                iter_full = iter_full + iter_full_start
            else:
                u = solveU(
                    z=z,
                    chat=chat,
                    priorMat=c,
                    penalty_factor=penalty_factor,
                    pathway_selection=pathway_selection,
                    glm_alpha=glm_alpha,
                    max_path=max_path,
                    disable_progress=disable_progress,
                    persistent_progress=persistent_progress,
                    L3=l3,
                )["U"]

            z1 = tcrossprod(y, b)
            z2 = l1 * (c @ u)

            z1_nonzero = np.argwhere(np.asarray(z1.T.stack()) > 0).flatten()
            z2_nonzero = np.argwhere(np.asarray(z2.T.stack()) > 0).flatten()

            ratio = np.median(np.divide(z2, np.asarray(z1)).T.stack().values[np.intersect1d(z2_nonzero, z1_nonzero)])

            z = (z1 + z2) @ solve(a=(tcrossprod(b) + l1 * diag_mat), b=diag_mat)
        else:
            z = tcrossprod(y, b) @ solve(a=(tcrossprod(b) + l1 * diag_mat), b=diag_mat)

        z[z < 0] = 0

        old_b = b.copy()
        b = solve(a=(z.transpose() @ z + l2 * diag_mat), b=diag_mat) @ z.transpose() @ y

        bdiff = ((b - old_b) ** 2).to_numpy().sum() / (b**2).to_numpy().sum()
        bdiff_trace = np.append(bdiff_trace, bdiff)

        if trace & (i >= iter_full_start):
            rprint(
                f"iter {i} errorY = {np.mean((y - z @ b)**2):.4f} prior information ratio= {round(ratio,2)} Bdiff = {bdiff:.4f} Bkappa= {np.linalg.cond(b):.4f};pos. col. U = {sum(u.sum(axis='index') > 0)}"
            )
        elif trace:
            rprint(
                f"iter {i} errorY = {np.mean((y - z @ b)**2):.4f} Bdiff = {np.linalg.cond(bdiff):.4f} Bkappa = {np.linalg.cond(b):.4f}"
            )

        if (i > MIN_ITERATIONS) and (bdiff > bdiff_trace[i - 50]):
            bdiff_count += 1
            rprint("Bdiff is not decreasing")
        elif bdiff_count > 1:
            bdiff_count -= 1

        if bdiff < tol:
            rprint(f"converged at iteration {i}")
            break
        if bdiff_count > MIN_BDIFF_COUNT:
            rprint(f"converged at iteration {i} Bdiff is not decreasing")
            break

    u.index = prior_mat.columns
    u.columns = [f"LV{_+1}" for _ in range(num_lvs)]
    z.columns = [f"LV{_+1}" for _ in range(num_lvs)]

    b.index = [f"LV{_+1}" for _ in range(num_lvs)]

    out = PLIERResults(
        residual=(y - (z.values @ b.values)),
        b=b,
        z=z,
        u=u,
        c=c,
        l1=l1,
        l2=l2,
        l3=l3,
        held_out_genes=held_out_genes,
    )

    if doCrossval:
        out_auc = crossVal(
            plierRes=out,
            priorMat=prior_mat,
            priorMatcv=prior_mat_cv,
            disable_progress=disable_progress,
            persistent_progress=persistent_progress,
        )
    else:
        rprint("Not using cross-validation. AUCs and p-values may be over-optimistic")
        out_auc = getAUC(out, y, prior_mat)

    out.with_prior = u.sum(axis="index")[u.sum(axis="index") > 0].to_dict()
    out.uauc = out_auc["Uauc"]
    out.up = out_auc["Upval"]
    out.summary = out_auc["summary"]
    tt = out.uauc.max(axis="index")
    rprint(f"There are {sum(tt > AUC_CUTOFF)} LVs with AUC > 0.70")

    out.b.index = nameB(out)

    return out
