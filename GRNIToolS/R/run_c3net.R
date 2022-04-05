#' C3NET gene regulatory network inference
#'
#' @description Default parameters for the C3NET network inference algorithm
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details C3NET gene regulatory network inference. For Step 1 of C3NET, there are three options that can be used. If methodstep1="cutoff" then cutoffMI input is taken as the significance threshold for step 1. In case cutoffMI is set to 0 then by default mean MI is taken as cutoffMI. If methodstep1 is set to "MTC" then the method from input MTCmethod, itnum for iteration number to get null distribution and alpha for statistical significance is taken for computations. If methodstep1 is set to "justp" or anything else then just alpha and itnum is used for computations.
#'
#'
#' @return Matrix A matrix which is the weighted adjacency matrix of the network inferred by C3NET algorithm
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
#' grn <- run_c3net(datexpr,gene)
run_c3net <- function(datexpr,gene,TF = NULL,file = NULL){
  if (is.null(TF)) {
    TF <- gene
  }
  datexpr <- as.matrix(datexpr)
  rownames(datexpr) <- gene
  exprmat <- c3net::c3net(datexpr, cutoffMI = 0,network = F)
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
