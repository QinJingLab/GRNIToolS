#' Maximum Relevance Minimum Redundancy
#' @description Default parameters for MRNET network inference algorithm.
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details MRNET algorithm Gene regulatory network inference
#' @return A matrix which is the weighted adjacency matrix of the network inferred by MRNET algorithm.
#' @export
#'
#' @references H. Peng, F.long and C.Ding. Feature selection based on mutual information: Criteria of max-dependency, max relevance and min redundancy. IEEE transaction on Pattern Analysis and Machine Intelligence, 2005.
#'
#' Patrick E. Meyer, Frederic Lafitte and Gianluca Bontempi. minet: A R/Bioconductor Package for Inferring Large Transcriptional Networks Using Mutual Information. BMC Bioinformatics, Vol 9, 2008.
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- run_mrnet(datexpr,gene)
run_mrnet <- function(datexpr,gene,TF = NULL,file = NULL){
  mi  <- parmigene::knnmi.all(datexpr)
  exprmat <- parmigene::mrnet(mi)
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
