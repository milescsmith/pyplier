#' @title DataSmooth
#' @description SVD based smoothing for single cell RNAseq data
#'
#' @param svdres svd result
#' @param k number of components to use
#'
#' @export
#'
def DataSmooth(svdres, k):
    alfa = svdres["u"][:, 0:k].transpose()
    beta = (alfa * svdres["d"][:, 0:k]).transpose()
    gamma = beta @ svdres["v"][:, 0:k].transpose()
    return gamma
