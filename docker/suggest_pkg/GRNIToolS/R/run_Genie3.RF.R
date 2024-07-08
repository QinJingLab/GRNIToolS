#' GENIE3.RF
#' @description Default parameters for GENIE3 algorithm (Random Forests)
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @param cl.cores The number of cores in computer that you want to utilize. Default: 2.
#' @details GENIE3 Infers a gene regulatory network (in the form of a weighted adjacency matrix) from expression data, using ensembles of regression trees(Random Forests).
#'
#' @return Matrix returns a matrix which is the weighted adjacency matrix of the network inferred by GENIE3 algorithm (Random Forests).
#' @export
#' @references Irrthum, Alexandre, Louis Wehenkel, and Pierre Geurts. "Inferring regulatory networks from expression data using tree-based methods." PloS one 5.9 (2010): e12776.
#'
#' Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.
#'
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' TF <- gene[1:20]
#' #grn <- run_GENIE3.RF(datexpr,gene,TF)
run_GENIE3.RF <- function(datexpr,gene,TF = NULL,file = NULL,cl.cores = 2){
  if (is.null(TF)) {
    TF <- gene
  }
  datexpr <- as.matrix(datexpr)
  rownames(datexpr) <- gene
  TFlist <- TF
  TF <- TF[TF %in% rownames(datexpr)]
  set.seed(123) #   For reproducibility of results
  weightMatrix <- GENIE3::GENIE3(exprMatrix = datexpr,regulators = TF,nCores = cl.cores,verbose = F,treeMethod = "RF")
  weightMatrix <- weightMatrix[TF,gene]
  weightMatrix <- t(weightMatrix)
  get_Score2 <- function(namelist,TFlist,grn){
    Score <- matrix(0,length(namelist),length(TFlist))
    colnames(Score) <- TFlist
    rownames(Score) <- namelist
    Score[,colnames(grn)] <- grn
    return(Score)
  }
  exprmat <- get_Score2(gene,TFlist,as.matrix(weightMatrix))
  exprmat <- as.data.frame(exprmat)
  if (! is.null(file)) {
    readr::write_delim(x = exprmat,file = file,delim = '\t',col_names = F)
  }
  grnmat <- exprmat
  unregister_dopar <- function() {
    env <- foreach:::.foreachGlobals
    rm(list=ls(name=env), pos=env)
  }
  unregister_dopar()
  return(grnmat)
}

