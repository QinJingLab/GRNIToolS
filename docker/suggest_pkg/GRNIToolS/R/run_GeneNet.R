
#' GeneNet algorithm
#' @description Default parameters for GeneNet algorithm
#' @param datexpr Expression matrix (genes * samples). Every row is a gene, every column is a sample.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details GeneNEt uses an heuristic for learning statistically a causal network. It relies on a conversion of a network inferred through correlation into a partial correlation graph. Then, a partial ordering of the nodes is assigned by means of a multiple testing of the log-ratio of standardized partial variances. This allows identifying a directed acyclic causal network as a sub-graph of the partial correlation network.
#'
#'
#' @return Matrix  A matrix which is the weighted adjacency matrix of the network inferred by GeneNet algorithm.
#' @export
#' @references Opgen-Rhein, Rainer, and Korbinian Strimmer. "Inferring gene dependency networks from genomic longitudinal data: a functional data approach." RevStat 4.1 (2006): 53-65.
#'
#' Opgen-Rhein, Rainer, and Korbinian Strimmer. "Using regularized dynamic correlation to infer gene dependency networks from time-series microarray data." Proceedings of the 4th International Workshop on Computational Systems Biology (WCSB 2006), Tampere. Vol. 4. 2006.
#'
#' Sch\"afer, Juliane, and Korbinian Strimmer. "A shrinkage approach to large-scale covariance matrix estimation and implications for functional genomics." Statistical applications in genetics and molecular biology 4.1 (2005): 32.
#'
#' Opgen-Rhein, R., and K. Strimmer. (2007). From correlation to causation networks: a simple approximate learning algorithm and its application to high-dimensional plant gene expression data. BMC Syst. Biol. 1:37.
#'
#'
#' @examples
#' data(datalist)
#' datexpr <- datalist$B[1:100,1:100]
#' gene <- datalist$gene[1:100]
#' grn <- run_GeneNet(datexpr,gene)
run_GeneNet <- function (datexpr,gene,TF = NULL,file = NULL)
{
  if (is.null(TF)) {
    TF <- gene
  }
  datexpr <- t(datexpr)
  colnames(datexpr) <- gene
  inferred.pcor <- GeneNet::ggm.estimate.pcor(datexpr, method = "static",verbose = FALSE)
  co <- fdrtool::fdrtool(corpcor::sm2vec(inferred.pcor), plot = FALSE,
                         statistic = "correlation", verbose = FALSE)$param
  test.results <- GeneNet::network.test.edges(inferred.pcor, plot = FALSE,verbose = FALSE)
  grn <- GeneNet::extract.network(test.results, cutoff.ggm = co[1,"cutoff"], verbose = FALSE)
  if (nrow(grn)!=0) {
    grn <- GeneNet::network.make.graph(grn, node.labels = colnames(datexpr))
    grnmat <- methods::as(grn, "matrix")
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
    exprmat <- get_Score(gene,TF,as.matrix(grnmat))
  } else(exprmat <- grn)
  exprmat <- as.data.frame(exprmat)
  if (! is.null(file)) {
    readr::write_delim(x = exprmat,file = file,delim = '\t',col_names = F)
  }
  return(exprmat)
}

