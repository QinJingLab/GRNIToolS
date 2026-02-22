#' Gene regulatory network inference for algorithm of ENNET
#' @description Default parameters for ENNET network inference algorithm.
#'
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @param cl.cores The number of cores in computer that you want to utilize. Default: 2.

#'
#' @details ENNET Gene regulatory network inference
#' @return A matrix which is the weighted adjacency matrix of the network inferred by ENNET algorithm.
#' @export
#'
#' @references Slawek, J., Arodz, T. ENNET: inferring large gene regulatory networks from expression data using gradient boosting. BMC Syst Biol 7, 106 (2013). https://doi.org/10.1186/1752-0509-7-106
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- run_ennet(datexpr,gene)
run_ennet <- function(datexpr,gene,TF = NULL,file = NULL,cl.cores=2){
  if (is.null(TF)) {
    TF <- gene
  }
  #doMC::registerDoMC(cl.cores)
  cl <- makeCluster(cl.cores)
  registerDoParallel(cl)
  colnames(datexpr) <- NULL
  rownames(datexpr) <- NULL
  exp <- as.matrix(datexpr)
  expr <- t(exp)
  id <- which(gene %in% TF)
  netMartrix <-ennet::ennet(E=expr,Tf=id)
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
  exprmat <- get_Score(gene,TF,as.matrix(netMartrix))
  exprmat <- as.data.frame(exprmat)
  if (! is.null(file)) {
    readr::write_delim(x = exprmat,file = file,delim = '\t',col_names = F)
  }
  grnmat <- exprmat
  stopCluster(cl)
  return(grnmat)

}
