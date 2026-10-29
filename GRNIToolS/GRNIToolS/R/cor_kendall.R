
#' Kendall correlation
#' @description Default parameters for kendall algorithm.
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details The weighted adjacency matrix of the inferred network by this algorithm keep the value.
#'
#'
#' @return A matrix which is the weighted adjacency matrix of the inferred network by this algorithm.
#' @export
#' @references Becker, R. A., Chambers, J. M. and Wilks, A. R. (1988). The New S Language. Wadsworth & Brooks/Cole.
#'
#' Kendall, M. G. (1938). A new measure of rank correlation, Biometrika.
#'
#' Kendall, M. G. (1945). The treatment of ties in rank problems. Biometrika.
#'
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:10]
#' gene <- datalist$gene[1:100]
#' grn <- cor_kendall(datexpr,gene)
cor_kendall <- function(datexpr,gene,TF = NULL,file = NULL){
  datexpr <- t(datexpr)
  exprmat <- stats::cor(as.matrix(datexpr),method = 'kendall')
  if (is.null(TF)) {
    TF <- gene
  }

  exprmat <- get_score_matrix(gene,TF,as.matrix(exprmat))

  exprmat <- as.data.frame(exprmat)
  if (! is.null(file)) {
    .grn_write_network(exprmat, file = file)
  }
  grnmat <- exprmat
  return(grnmat)
}
