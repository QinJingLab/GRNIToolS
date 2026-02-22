#' ARACNE algorithm (from R package minet)
#' @description Default parameters for ARACNE algorithm for the reconstruction of gene interaction networks (additive model).
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details The motivation of the Algorithm for the Reconstruction of Accurate Cellular Networks (ARACNE) is that many similar measures between variables may be the result of indirect effects. In order to delete the indirect effect the algorithm relies on the "Data Processing Inequality", this process removes the weakest link in every triplet of connected variables.
#' @return A matrix which is the weighted adjacency matrix of the network inferred by aracne algorithm. Rows contain genes and columns contain TF.
#' @export
#'
#' @references Adam A. Margolin, Ilya Nemenman, Katia Basso, Chris Wiggins, Gustavo Stolovitzky, Riccardo Dalla Favera, and Andrea Califano. Aracne : An algorithm for the reconstruction of gene regulatory networks in a mammalian cellular context. BMC Bioinformatics, 2006.
#'
#' Meyer, Patrick E., Frederic Lafitte, and Gianluca Bontempi. "minet: AR/Bioconductor package for inferring large transcriptional networks using mutual information." BMC Bioinformatics 9.1 (2008): 461.
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- minet_aracne(datexpr,gene)
minet_aracne <- function(datexpr,gene,TF = NULL,file = NULL){
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
  net <- minet::aracne(mim)
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
