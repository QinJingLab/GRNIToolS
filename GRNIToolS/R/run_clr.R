#' Context Likelihood or Relatedness Network
#' @description Default parameters for CLR network inference algorithm.
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details CLR Gene regulatory network inference.
#' @return Matrix returns a matrix which is the weighted adjacency matrix of the network inferred by CLR algorithm.
#' @export
#'
#' @references Jeremiah J. Faith, Boris Hayete, Joshua T. Thaden, Ilaria Mogno, Jamey Wierzbowski, Guillaume Cottarel, Simon Kasif, James J. Collins, and Timothy S. Gardner. Large-scale mapping and validation of escherichia coli transcriptional regulation from a compendium of expression profiles. PLoS Biology, 2007.
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- run_clr(datexpr,gene)
run_clr <- function(datexpr,gene,TF = NULL,file = NULL){
  mi  <- parmigene::knnmi.all(datexpr)
  exprmat <- parmigene::clr(mi)
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
