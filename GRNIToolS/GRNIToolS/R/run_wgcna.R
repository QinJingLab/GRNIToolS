#' WGCNA Weighted Correlation Network
#'
#' @description Default parameters for WGCNA algorithm
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @param cl.cores The number of cores in computer that you want to utilize. Default: 2.
#' @details WGCNA Gene regulatory network inference
#' @return A matrix which is the correlation matrix by WGCNA algorithm.
#' @export
#' @references Bin Zhang and Steve Horvath (2005) "A General Framework for Weighted Gene Co-Expression Network Analysis", Statistical Applications in Genetics and Molecular Biology: Vol. 4: No. 1, Article 17
#'
#' Nowick K, Gernat T, Almaas E, Stubbs L. Differences in human and chimpanzee gene expression patterns define an evolving network of transcription factors in brain. Proc Natl Acad Sci U S A. 2009 Dec 29;106(52):22358-63. doi: 10.1073/pnas.0911376106. Epub 2009 Dec 10.
#'
#' Gysi DM, Voigt A, Fragoso TM, Almaas E, Nowick K. wTO: an R package Wfor computing weighted topological overlap and a consensus network with integrated visualization tool. BMC Bioinformatics. 2018 Oct 24;19(1):392. doi: 10.1186/s12859-018-2351-7.
#' @import foreach parallel doParallel iterators
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- run_wgcna(datexpr,gene)
run_wgcna <- function(datexpr,gene,TF = NULL,file = NULL,cl.cores = 2){
  WGCNA::enableWGCNAThreads(cl.cores)
  rownames(datexpr) <- gene
  datexpr <- t(datexpr)
  if (is.null(TF)) {
    TF <- gene
  }
  powers = c(c(1:10), seq(from = 12, to=20, by=2))
  sft = WGCNA::pickSoftThreshold(datexpr, powerVector = powers, verbose = 0,indent = 0)
  RpowerTable=sft[[2]]
  cex1=0.9
  beta <-as.numeric(sft$powerEstimate)
  if (is.na(beta)) {
    warning("pickSoftThreshold() returned NA: no power in the tested range satisfies ",
            "the scale-free topology criterion for this dataset. The returned network ",
            "is degenerate (diagonal only).", call. = FALSE)
  }
  datexpr <- base::apply(datexpr, 2, as.numeric)
  adj <- WGCNA::adjacency(datexpr,power = beta)
  TOM <- WGCNA::TOMsimilarity(adj,verbose = 0,indent = 0)
  exprmat <- get_score_matrix(gene,TF,as.matrix(TOM))
  exprmat <- as.data.frame(exprmat)
  if (! is.null(file)) {
    .grn_write_network(exprmat, file = file)
  }
  grnmat <- exprmat
  doParallel::stopImplicitCluster()
  return(grnmat)
}
