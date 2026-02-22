#' Maximum Relevance Minimum Redundancy algorithm (from R package minet)
#' @description Default parameters for MRNET network inference algorithm.
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details MRNET algorithm Gene regulatory network inference
#' @return A matrix which is the weighted adjacency matrix of the network inferred by aracne algorithm. Rows contain genes and columns contain TF.
#' @export
#' @references H. Peng, F.long and C.Ding. Feature selection based on mutual information: Criteria of max-dependency, max relevance and min redundancy. IEEE transaction on Pattern Analysis and Machine Intelligence, 2005.
#'
#' Patrick E. Meyer, Frederic Lafitte and Gianluca Bontempi. minet: A R/Bioconductor Package for Inferring Large Transcriptional Networks Using Mutual Information. BMC Bioinformatics, Vol 9, 2008.
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- minet_mrnet(datexpr,gene)
minet_mrnet <- function(datexpr,gene,TF = NULL,file = NULL){
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
  net <- minet::mrnet(mim)
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
