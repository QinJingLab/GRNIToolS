
#' Calculate the predictions of GRN and the labels of the high-throughput golden standard.
#'
#' @description Using the gold standard data to evaluate the predicted GRN. Calculate the predictions of GRN and the labels of the high-throughput golden standard.
#' @param HGSmat A matrix which is the high-throughput golden standard. Rows contain genes and columns contain TFs ,the value between in one or zero.
#' @param grn A matrix which is the weighted adjacency matrix of the inferred GRN by this algorithm or the result from the function of out_grn. The matrix need the colnames(TF) and rownames(genes). 
#' @param sym If set to TRUE, only the regulatory relationship in the high-throughput gold standard is considered, and the value of 0 is not considered, so that the number of true positive and false positive is equal.
#' If set to FALSE, It is assumed that the regulatory relationships that do not exist in the high-throughput gold standard are non regulatory and 0.Default: TRUE.
#' @param num The number of gene regulator information. Default: NULL
#'
#' @return A data.frame. The first element, dat$predictions, is a vector of numerical predictions. The second element, dat$labels, is a vector of class labels.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(200,0,1),20,10)
#' B <- matrix(rnorm(1000,0,1),100)
#' X <- matrix(0,100,20)
#' s <- c(1:10)
#' TF <- paste0('G',1:20)
#' gene <- paste0('G',1:100)
#' res <- demo_reg(A, B, s, X, 'ADMM',max.steps = 200, TF, gene, cl.core = 2,file=NULL,verbose=FALSE)
#' grn <- res$grn
#' HGSmat <- matrix(0,100,20)
#' rownames(HGSmat) <- gene
#' colnames(HGSmat) <- TF
#' HGSmat[sample(2000,200)] <- 1
#' dat <- calpl(HGSmat,grn,sym=FALSE)
#' head(dat)
calpl <- function (HGSmat, grn, sym = TRUE, num = NULL) 
{
  grn <- as.data.frame(grn)
  if (ncol(grn) <= 3 & nrow(grn) != ncol(grn)) {
    a <- grn
    gtf <- sort(unique(a[, 1]))
    ggene <- sort(unique(a[, 2]))
    mat <- matrix(0, length(ggene), length(gtf))
    colnames(mat) <- gtf
    rownames(mat) <- ggene
    col <- match(a[, 1], gtf)
    row <- match(a[, 2], ggene)
    id <- cbind(row, col)
    mat[id] <- a[, 3]
    grn <- mat
  }
  grn <- as.matrix(grn)
  grn <- abs(grn)
  HGSmat <- abs(HGSmat)
  grn <- grn/max(grn)
  if (!is.null(num)) {
    diag(grn) <- 0
    mtf <- colnames(grn)
    mgene <- rownames(grn)
    id <- which(grn != 0, arr.ind = T)
    dat <- data.frame(TF = mtf[id[,2]], Gene = mgene[id[,1]], Weight = grn[id])
    dat <- dat[order(dat[, 3], decreasing = T), ]
    grn2 <- dat
    if (num > nrow(grn2)) {
      num <- nrow(grn2)
    }
    grn2 <- grn2[1:num, ]
    a <- grn2
    gtf <- sort(unique(a[, 1]))
    ggene <- sort(unique(a[, 2]))
    mat <- matrix(0, length(ggene), length(gtf))
    colnames(mat) <- gtf
    rownames(mat) <- ggene
    col <- match(a[, 1], gtf)
    row <- match(a[, 2], ggene)
    id <- cbind(row, col)
    mat[id] <- a[, 3]
    grn <- mat
  }
  if (ncol(grn) > ncol(HGSmat) | nrow(grn) > nrow(HGSmat))
  {
    id1 <- match(rownames(HGSmat), rownames(grn))
    id2 <- match(colnames(HGSmat), colnames(grn))
    grn <- grn[id1, id2]
  }else{ id1 <- match(rownames(grn), rownames(HGSmat))
  id2 <- match(colnames(grn), colnames(HGSmat))
  HGSmat <- HGSmat[id1, id2] }
  
  if (sym) {
    IND1 <- which(HGSmat == 1 & grn != 0, arr.ind = TRUE)
    N <- nrow(IND1)
    if (N != 0) {
      IND <- which(HGSmat == 0 & grn != 0, arr.ind = TRUE)
      N2 <- nrow(IND)
      if (N > N2) {
        N <- N2
      }
      set.seed(123)
      IND <- IND[sample(nrow(IND), N, replace = F), ]
      IND1 <- IND1[sample(nrow(IND1), N, replace = F), 
      ]
      IND <- rbind(IND1, IND)
      dat <- grn[IND]
      dat <- matrix(c(dat, c(rep(1, N), rep(0, N))), ncol = 2)
      dat <- as.data.frame(dat)
      colnames(dat) <- c("predictions", "labels")
    }
    else {
      dat <- NULL
    }
  }
  if (!sym) {
    id1 <- which(grn != 0)
    predictions <- grn[id1]
    labels <- HGSmat[id1]
    dat <- data.frame(predictions = as.numeric(predictions), 
                      labels = as.numeric(labels))
  }
  return(dat)
}
