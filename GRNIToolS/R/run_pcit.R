
#' PCIT algorithm
#' @description Default parameters for PCIT algorithm
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details The Partial Correlation coefficient with Information Theory (PCIT) algorithm, combines the concept of partial correlation coefficient with information theory to identify significant gene-to-gene associations.
#'
#'
#' @return A matrix which is the weighted adjacency matrix of the network inferred by PCIT algorithm.
#' @export
#' @references Reverter, Antonio, and Eva KF Chan. "Combining partial correlation and an information theory approach to the reversed engineering of gene co-expression networks." Bioinformatics 24.21 (2008): 2491-2497.
#'
#'
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- run_pcit(datexpr,gene)
run_pcit <- function(datexpr,gene,TF = NULL,file = NULL){
  if (is.null(TF)) {
    TF <- gene
  }
  datexpr <- t(datexpr)
  datexpr <- stats::cor(datexpr)
  res <- PCIT::pcit(datexpr)
  id <- PCIT::idx(res)
  id2 <- c(1:length(as.numeric(datexpr)))[-id]
  grnmat <- datexpr
  grnmat[id2] <- 0
  diag(grnmat) <- 0
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
  grnmat <- get_Score(gene,TF,as.matrix(grnmat))
  grnmat <- as.data.frame(grnmat)
  if (! is.null(file)) {
    readr::write_delim(x = grnmat,file = file,delim = '\t',col_names = F)
  }
  return(grnmat)
}
