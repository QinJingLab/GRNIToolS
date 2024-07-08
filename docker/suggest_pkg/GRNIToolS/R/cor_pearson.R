
#' Pearson correlation
#' @description Default parameters for pearson algorithm.
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details The weighted adjacency matrix of the inferred network by this algorithm keep the value which the p-value<0.05.
#'
#'
#' @return A matrix which is the weighted adjacency matrix of the inferred network by this algorithm.
#' @export
#' @references Hollander M. and Wolfe D.A. (1973). Nonparametric Statistical Methods. New York: Wiley.
#'
#' Press WH, Flannery BP, Teukolsky SA, Vetterling, WT (1988): Numerical Recipes in C. Cambridge: Cambridge University Press.
#'
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- cor_pearson(datexpr,gene)
cor_pearson <- function(datexpr,gene,TF = NULL,file = NULL){
  datexpr <- t(datexpr)
  res <- Hmisc::rcorr(as.matrix(datexpr),type = 'pearson')
  exprmat <- res$r
  exprp <- res$P
  exprp <- ifelse(is.na(exprp),1,exprp)
  exprmat[which(exprp > 0.05)] = 0
  if (is.null(TF)) {
    TF <- gene
  }
  get_Score <- function(gene,TF,grnmat){
    grnmat <- as.matrix(grnmat)
    id1 <- which(TF %in% gene)
    id2 <- which(gene %in% TF)
    Score <- matrix(0,length(gene),length(TF))
    colnames(Score) <- TF
    rownames(Score) <- gene
    Score[,id1] <- grnmat[,id2]
    return(Score)
  }

  exprmat <- get_Score(gene,TF,as.matrix(exprmat))

  exprmat <- as.data.frame(exprmat)
  if (! is.null(file)) {
    readr::write_delim(x = exprmat,file = file,delim = '\t',col_names = F)
  }
  grnmat <- exprmat
  return(grnmat)
}
