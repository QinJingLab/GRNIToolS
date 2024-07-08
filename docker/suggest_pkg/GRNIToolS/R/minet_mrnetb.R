#' Maximum Relevance Minimum Redundancy Backward (from R package minet)
#' @description Default parameters for MRNETB network inference algorithm.
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details MRNETB takes the mutual information matrix as input in order to infer the network using the maximum relevance/minimum redundancy criterion combined with a backward elimination and a sequential replacement - see references. This method is a variant of mrnet.
#' @return A matrix which is the weighted adjacency matrix of the network inferred by MRNETB algorithm.
#' @export
#'
#' @references Patrick E. Meyer, Daniel Marbach, Sushmita Roy and Manolis Kellis. Information-Theoretic Inference of Gene Networks Using Backward Elimination. The 2010 International Conference on Bioinformatics and Computational Biology.
#'
#' Patrick E. Meyer, Kevin Kontos, Frederic Lafitte and Gianluca Bontempi. Information-theoretic inference of large transcriptional regulatory networks. EURASIP Journal on Bioinformatics and Systems Biology, 2007.
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- minet_mrnetb(datexpr,gene)
minet_mrnetb <- function(datexpr,gene,TF = NULL,file = NULL){
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
  net <- minet::mrnetb(mim)
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
