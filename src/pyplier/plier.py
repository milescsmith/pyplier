import random
from math import floor

import numpy as np
import pandas as pd
from icontract import ensure, require
from numpy.random import default_rng
from rich import print as rprint
from scipy.linalg import solve, svd
from sklearn.preprocessing import normalize
from sklearn.utils.extmath import randomized_svd

from .commonRows import commonRows
from .num_pc import num_pc
from .pinv_ridge import pinv_ridge
from .stubs import PLIERResults
from .utils import crossprod, rowNorm, setdiff, tcrossprod
from .solveU import solveU


@require(lambda pathwaySelection: pathwaySelection in ("complete", "fast"))
def PLIER(
  data: pd.DataFrame,
  priorMat: pd.DataFrame,
  svdres=None,
  num_LVs: float=None,
  L1: float=None,
  L2: float=None,
  L3: float=None,
  frac: float=0.7,
  max_iter: int=350,
  trace: bool=False,
  scale: bool=True,
  Chat=None,
  maxPath: int=10,
  doCrossval: bool=True,
  penalty_factor: np.ndarray=None,
  glm_alpha: float=0.9,
  minGenes: int=10,
  tol: float=1e-06,
  seed: int=123456,
  allGenes: bool=False,
  rseed: int=None,
  pathwaySelection: str="complete"
  ):
    """"Main PLIER function

    Parameters
    ----------
    data : pd.DataFrame
        the data to be processed with genes in rows and samples in columns. 
        Should be z-scored or set scale=True
    priorMat : pd.DataFrame
        the binary prior information matrix with genes in rows and 
        pathways/genesets in columns
    svdres : [type], optional
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
    Chat : [type], optional
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
        Use all genes. By default only genes in the priorMat matrix are used. Defaults to F.
    rseed : int, optional
        Set this option to use a random initialization, instead of SVD. Defaults to None.
    pathwaySelection : str, optional
        Pathways to be optimized with elstic-net penalty are preselected based on ridge regression results. 'Complete' uses all top pathways to fit individual LVs. 'Fast' uses only the top pathways for the single LV in question. Defaults to "complete".

    Returns
    -------
    [type]
        [description]
    """
    

    if penalty_factor is None:
        penalty_factor = range(priorMat.shape[1])

    if scale:
        Y = rowNorm(arr=data)
    else:
        Y = data

    if (priorMat.shape[0] != data.shape[0]) or not all(priorMat.index == data.index):
        if not allGenes:
            cm = commonRows(data, priorMat)
            rprint(f"Selecting common genes: {len(cm)}")
            priorMat = priorMat.loc[cm,:]
            Y = Y.loc[cm,:]
        else:
            extra_genes = setdiff(data.index, priorMat.index)
            eMat = pd.DataFrame(
                data=np.zeros((len(extra_genes), priorMat.shape[1])),
                columns=priorMat.columns,
                index=extra_genes,
                )
            priorMat = priorMat.append(eMat)
            priorMat = priorMat.loc[data.index,:]

    numGenes = priorMat.sum(axis='rows') # colsums

    heldOutGenes = dict()
    iibad = numGenes[numGenes < minGenes].index
    priorMat.loc[:,iibad] = 0
    rprint(f"Removing {len(iibad)} pathways with too few genes")
    if doCrossval:
        priorMatCV = priorMat
        if seed is not None:
            random.seed(seed)
        for j in range(priorMatCV.shape[1]):
            current_col = priorMatCV.iloc[:,j]
            iipos = [current_col.index.get_loc(_) for _ in current_col[current_col > 0].index] # need the row number, not the row name
            
            iiposs = random.choices(iipos, k=len(iipos)/5)
            priorMatCV.iloc[iiposs, j] = 0
            heldOutGenes[priorMat.columns[j]] = priorMat.index[iiposs]
        C = priorMatCV
    else:
        C = priorMat

    ns = data.shape[1]
    Bdiff = -1
    BdiffTrace = np.ndarray((0,),dtype=np.float64)
    BdiffCount = 0
    if Chat is None:
        Cp = crossprod(C)
        Chat = pinv_ridge(Cp, 5) @ C.transpose()
    YsqSum = (Y**2).to_numpy().sum()
    # compute svd and use that as the starting point

    if (svdres is not None) and (svdres["v"] != Y.shape[1]):
        rprint("SVD V has the wrong number of columns")
        svdres = None

    if Y.isnull().to_numpy().any():
        Y.fillna(0, inplace=True)

    if svdres is None:
        svdres = dict()
        rprint("Computing SVD")
        if (ns > 500):
            rprint("Using rsvd")
            svdres["u"], svdres["d"], svdres["v"] = randomized_svd(M=Y.values, n_components = min(ns, max(200, ns / 4)), n_iter = 3)
        else:
            svdres["u"], svdres["d"], svdres["v"] = svd(Y, lapack_driver='gesdd') # the gesvd driver flips the sign for components > 6 in the v matrix as compared to R's svd function
        rprint("Done")
        svdres["v"] = svdres["v"].transpose() # as compared to the output from R's svd, the v matrix is transposed.  Took me too long to figure this one out.

    if num_LVs is None:
        num_LVs = num_pc(svdres) * 2
        num_LVs = min(num_LVs, floor(Y.shape[1] * 0.9))
        rprint(f"The number of LVs is set to {num_LVs}")

    if L2 is None:
        L2 = svdres[["d"]][num_LVs]
        rprint(f"L2 is set to {L2}")

    if L1 is None:
        L1 = L2 / 2
        rprint(f"L1 is set to {L1}")

    B = (svdres["v"][0:Y.shape[1], 0:num_LVs] @ np.diag(svdres["d"][0:num_LVs])).transpose()

    # following two lines are equivalent to R's diag(x)
    # numpy.fill_diagonal modifies in place and does not
    # return a value, thus this workaround
    diag_mat = np.zeros((num_LVs,num_LVs))
    np.fill_diagonal(diag_mat)
    
    # for R's solve(), if b is missing, it uses the identity matrix of a
    # scipy.linalg.solve does not have a default for b, so just give it one
    Z = np.dot(
        np.dot(Y, B.T),
        solve(a=np.dot(B,B.T) + L1 * diag_mat, b=diag_mat)
    )

    Z = Z.where(cond=lambda x: x > 0, other=0)
    
    if rseed is not None:
        rprint("using random start")
        random.seed(rseed)
        
        rng = default_rng()
        rng.shuffle(B, axis=1) # B = t(apply(B, 1, sample))
        rng.shuffle(Z, axis=0) # Z = apply(Z, 2, sample)
        Z = Z.transpose()

    U = np.zeros((C.shape[1], num_LVs)) # matrix(0, nrow = ncol(C), ncol = num_LVs)
    
    rprint(f"errorY (SVD based:best possible) = {((Y - np.dot(Z, B))**2).to_numpy().mean():.4f}")

    iter_full_start = iter_full = 20

    curfrac = 0
    nposlast = np.inf
    npos = -np.inf
    if L3 is not None:
        L3_given = True
    else:
        L3_given = False

    for i in range(max_iter):
        if (i >= iter_full_start):
            if (i == iter_full and not L3_given):
            # update L3 to the target fraction
                Ulist = solveU(Z, Chat, C, penalty_factor, pathwaySelection, glm_alpha, maxPath, target_frac = frac)
                U = Ulist[["U"]]
                L3 = Ulist[["L3"]]
                rprint(f"New L3 is {L3}")
                iter_full = iter_full + iter_full_start
            else:
                U = solveU(Z, Chat, C, penalty_factor, pathwaySelection, glm_alpha, maxPath, L3 = L3)
        
            # TODO: YOU ARE HERE
            curfrac = (npos = sum(apply(U, 2, max) > 0)) / num_LVs
            Z1 = Y @ t(B)
            Z2 = L1 * C @ U
            ratio = median((Z2 / Z1)[Z2 > 0 & Z1 > 0])
            Z = (Z1 + Z2) @ solve(tcrossprod(B) + L1 * diag(num_LVs))
        else
            Z = (Y @ t(B)) @ solve(tcrossprod(B) + L1 * diag(num_LVs))

        Z[Z < 0] = 0

        oldB = B
        B = solve(t(Z) @ Z + L2 * diag(num_LVs)) @ t(Z) @ Y

        Bdiff = sum((B - oldB)**2) / sum(B**2)
        BdiffTrace = c(BdiffTrace, Bdiff)

        err0 = sum((Y - Z @ B)**2) + sum((Z - C @ U)**2) * L1 + sum(B**2) * L2
        
        if (trace & i >= iter_full_start):
            rprint(f"iter {i} errorY= {round2(mean((Y - Z @ B)**2))} prior information ratio= {round(ratio,2)} Bdiff= {round2(Bdiff)} Bkappa= {round2(kappa(B)))};pos. col. U= {sum(colSums(U) > 0)}")
        elif trace
            rprint(f"iter {i} errorY= {round2(mean((Y - Z @ B)**2))} Bdiff= {round2(Bdiff)} Bkappa= {round2(kappa(B))}")

        if (i > 52 and Bdiff > BdiffTrace[i - 50]):
            BdiffCount = BdiffCount + 1
            message("Bdiff is not decreasing")
        elif BdiffCount > 1:
            BdiffCount = BdiffCount - 1

        if (Bdiff < tol):
            rprint(f"converged at iteration {i}")
            break
        if BdiffCount > 5:
            rprint(f"converged at iteration {i} Bdiff is not decreasing")
            break

    U.index = priorMat.columns
    U.columns = [f"LV{_+1}" for _ in range(num_LVs)]


    B.index = [f"LV{_+1}" for _ in range(num_LVs)]

    out = PLIERResults(
        "residual" = (Y - Z @ B),
        "B" = B,
        "Z" = Z,
        "U" = U,
        "C" = C,
        "L1" = L1,
        "L2" = L2,
        "L3" = L3,
        "heldOutGenes" = heldOutGenes
        )
        
    if doCrossval:
        outAUC = crossVal(out, Y, priorMat, priorMatCV)
    else:
        message("Not using cross-validation. AUCs and p-values may be over-optimistic")
        outAUC = getAUC(out, Y, priorMat)

    out["withPrior"] = which(colSums(out["U"]) > 0)
    out["Uauc"] = chuck(outAUC, "Uauc")
    out["Up"] = chuck(outAUC, "Upval")
    out["summary"] = chuck(outAUC, "summary")
    tt = colMaxs(out["Uauc"], value = True, parallel = True)
    rprint(f"There are {sum(tt > 0.7)} LVs with AUC > 0.70")

    rownames(out["B"]) = nameB(out)

    return out