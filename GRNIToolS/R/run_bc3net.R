#' BC3NET gene regulatory network inference
#'
#' @description Default parameters for the BC3NET network inference algorithm
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details BC3NET gene regulatory network inference
#'
#' @return A matrix which is the weighted adjacency matrix of the network inferred by BC3NET algorithm
#' @export
#' @references Altay G, Emmert-Streib F. Inferring the conservative causal core of gene regulatory networks. BMC Syst Biol. 2010 Sep 28;4:132.
#'
#' de Matos Simoes R, Emmert-Streib F. Bagging statistical network inference from large-scale gene expression data. PLoS One. 2012;7(3):e33624, Epub 2012 Mar 30, <doi:10.1371/journal.pone.0033624>.
#'
#' de Matos Simoes R, Emmert-Streib F. Influence of statistical estimators of mutual information and data heterogeneity on the inference of gene regulatory networks. PLoS One. 2011;6(12):e29279. Epub 2011 Dec 29.
#'
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- run_bc3net(datexpr,gene)
run_bc3net <- function(datexpr,gene,TF = NULL,file = NULL){
  if (is.null(TF)) {
    TF <- gene
  }
  datexpr <- as.matrix(datexpr)
  rownames(datexpr) <- gene
  bnet <- bc3net::bc3net(datexpr)
  output <- igraph::as_data_frame(bnet)
  cal_score_inet<- function(mat,namelist,TFlist){
    mat <- as.data.frame(mat)
    mat <- mat[mat[,1] %in% TFlist,]
    mat <- mat[mat[,2] %in% namelist,]
    mat[,3] <- abs(mat[,3])
    mat[,3] <- mat[,3] / max(mat[,3])
    mat_score <- matrix(0, nrow = length(namelist), ncol = length(TFlist))
    colnames(mat_score) <- TFlist
    rownames(mat_score) <- namelist
    for (i in 1:nrow(mat)) {
      temp <- mat[i,]
      mat_score[as.character(temp[,2]),as.character(temp[,1])] <- as.numeric(temp[,3])
    }
    return(mat_score)
  }
  exprmat <- cal_score_inet(output,gene,TF)
  exprmat <- as.data.frame(exprmat)
  if (! is.null(file)) {
    readr::write_delim(x = exprmat,file = file,delim = '\t',col_names = F)
  }
  grnmat <- exprmat
  return(grnmat)
}
