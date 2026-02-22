#' Bootstrap
#' @description The gene expression matrix was resampled by bootstrap method.
#'
#' @param datexpr Expression matrix (genes * samples). Numeric matrix with the dataset to infer the network.Rows contain genes and columns contain samples
#' @param num Number of the times of resampling with bootstrap method.The number of new datasets.Default: 5.
#' @param seed If set to TRUE, set.seed for keeping the results the same for each run. Default: TRUE.
#'
#' @return A list for the bootstrap result of gene expression matrix. Each one of objects contains a matrix of the resampled dataset.
#'
#' @details
#' Bootstrap is a resampling technique in statistical learning, which is used to estimate standard error, confidence interval and deviation. We resampled the gene expression matrix using bootstrap method to form multiple datasets.
#' @export
#'
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:10,1:10]
#' boot.sample <- bootstrap(datexpr)
bootstrap <- function(datexpr,num=5,seed=TRUE) {
  boot.sample <- list()
  for (i in 1:num) {
    if (seed) {
      set.seed(i)
      id <- sample(1:ncol(datexpr),replace = T)
      boot.sample[[i]] <- datexpr[,id]
      colnames(boot.sample[[i]]) <- NULL
    }
    else{
      id <- sample(1:ncol(datexpr),replace = T)
      boot.sample[[i]] <- datexpr[,id]
      colnames(boot.sample[[i]]) <- NULL
    }
  }
  return(boot.sample)
}


