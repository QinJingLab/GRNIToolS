#' CLR algorithm (from R package minet)
#' @description Default parameters for CLR network inference algorithm.
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details  CLR Gene regulatory network inference.
#' @return A matrix which is the weighted adjacency matrix of the network inferred by aracne algorithm. Rows contain genes and columns contain TF.
#' @export
#'
#' @references Jeremiah J. Faith, Boris Hayete, Joshua T. Thaden, Ilaria Mogno, Jamey Wierzbowski, Guillaume Cottarel, Simon Kasif, James J. Collins, and Timothy S. Gardner. Large-scale mapping and validation of escherichia coli transcriptional regulation from a compendium of expression profiles. PLoS Biology, 2007.
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- minet_clr(datexpr,gene)
minet_clr <- function(datexpr,gene,TF = NULL,file = NULL){
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
  cut_na <- function(expr){
    temp <- expr
    temp <- ifelse(is.na(temp),0,temp)
    expr <- temp	
    return(expr)
  }
  datexpr <- t(datexpr)
  mim <- minet::build.mim(datexpr)
  mim<- cut_na(mim)
  net <- minet::clr(mim)
  net <- net/max(net)
  exprmat <- get_Score(gene,TF,as.matrix(net))
  exprmat <- as.data.frame(exprmat)
  colnames(exprmat) = TF
  rownames(exprmat) = gene
  if (! is.null(file)) {
    readr::write_delim(x = exprmat,file = file,delim = '\t',col_names = F)
  }
  grnmat <- exprmat
  return(grnmat)
}
