def DataSmooth(svdres, k):
    """ SVD based smoothing for single cell RNAseq data

    Parameters
    ----------
    svdres : svd result
    k : number of components to use

    Returns
    -------
    """
    alfa = svdres["u"][:, 0:k].transpose()
    beta = (alfa * svdres["d"][:, 0:k]).transpose()
    gamma = beta @ svdres["v"][:, 0:k].transpose()
    return gamma
