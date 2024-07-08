#' Meta Methods that combine several different algorithms
#' @description This is the main function to call meta algorithms for network inference.
#'
#' @param algorithm The algorithm of inferring grn. We provide 31 algorithms in this function including
#' 'Pearson','Spearman','Kendall','WGCNA','BC3NET','C3NET','parmigene_ARACNE.a','parmigene_ARACNE.m','parmigene_CLR','parmigene_MRNET',
#' 'minet_ARACNE','minet_CLR','minet_MRNET','minet_MRNETB','GeneNet','GENIE3.ET','GENIE3.RF','ENNET,'PCIT','ADMM','ADMMHalf',ADMMHard', 'ITAHalf', 'ITAHard',
#' 'ISTA', 'SPGL0', 'SPGL1', 'CoSaMP', 'OMP', 'FoBa' and 'LARS'. Defaule: c('Pearson','parmigene_ARACNE.a','parmigene_CLR')
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @param cl.cores The number of cores in computer that you want to utilize. Default: 2.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param A Gene expression data of transcriptome factors (i.e. basis function in machine learning).
#' The class of A are required to be 'matrix' and the dimension of matrix A is m * n.
#' @param B Gene expression data of target genes (i.e. observation in machine learning).
#' The class of B are required to be 'matrix' and the dimension of matrix B is u * n.
#' @param X Gene expression data of Chromatin immunoprecipitation or zero matrix
#' (i.e. initial iterative point in machine learning).The class of X are required to be
#' 'matrix' and the dimension of matrix X is u * m.
#' @param s Sparsity level of solution. User can input a sequence of sparsity, i.e. 's <- c(1,2,3,4,5)'.
#' @param p If set to FALSE, this demo aims to use the algorithms in lasso type conveniently.If set to TRUE, this demo aims to use the algorithms in plasso type conveniently (The matrix X must need).Default: FALSE.
#' @param tau Tuning parameter of prior lasso (plasso); using cross-validation can get a great one to calculate. The default is 10.
#' @param meta Take the intersection or union of these methods. (union, intersection) Default: union.
#' @param max.steps Maximum iteration used in calculation. Default: 200.
#' @details For the existing gene regulatory network algorithms, select several algorithms, and take the intersection or union of the regulatory network results obtained by these methods to obtain a new regulatory network result.
#'
#' @return A matrix which is the weighted adjacency matrix of the network inferred by multiple algorithm.
#' @export
#'
#' @examples
#' A <- matrix(rnorm(200,0,1),20,10)
#' B <- matrix(rnorm(1000,0,1),100)
#' X <- matrix(0,100,20)
#' s <- c(1:10)
#' TF <- paste0('G',1:20)
#' gene <- paste0('G',1:100)
#' algorithm=c('Pearson','parmigene_ARACNE.a','parmigene_CLR','OMP')
#' grn <- run_meta(A,B,s,X,algorithm,meta='union',
#'                 max.steps = 200,gene,TF = TF,file = NULL,cl.cores=2)
run_meta <- function(A=NULL, B ,s=NULL, X=NULL,algorithm=c('Pearson','parmigene_ARACNE.a','parmigene_CLR'),meta='union',max.steps = 200,gene,TF = NULL,p=FALSE,tau=10,file = NULL,cl.cores=2){
  if (is.null(TF)) {
    TF <- gene
  }
  grnlist <- list()
  reg_alg <- c('ADMM', 'ADMMHalf','ADMMHard', 'ITAHalf', 'ITAHard', 'ISTA', 'SPGL0', 'SPGL1', 'CoSaMP', 'OMP', 'FoBa' , 'LARS')
  for (alg in algorithm) {
    if (alg %in% reg_alg) {
      temp <- demo_reg(A, B, s, X, alg,max.steps = max.steps,TF, gene,p=p,tau=tau, cl.cores,file=NULL,verbose=FALSE)
    }else {temp <- demo_other(datexpr=B,algorithm = alg,TF=TF,gene=gene,file=NULL,cl.cores=cl.cores,verbose=FALSE) }
    grn <- temp$grn
    grnlist <- c(grnlist,list(grn))
  }

  names(grnlist) <- algorithm
  if (meta == 'union') {
    grn2 <- NULL
    for (each in grnlist) {
      each <- as.matrix(each)
      each <- abs(each)
      each <- each /max(each)
      if (is.null(grn2)) {
        grn2 <- each
      }else {grn2 <- grn2 + each }
    }
    grn2 <- grn2 / length(algorithm)
  }
  if (meta == 'intersection') {
    grn2 <- NULL
    id <- 1: (nrow(grnlist[[1]]) * ncol(grnlist[[1]]))
    for (each in grnlist) {
      each <- as.matrix(each)
      each <- abs(each)
      id1 <- which(each != 0)
      id <- intersect(id,id1)
    }
    grn2 <- each
    grn2[] <- 0
    if (!length(id)==0) {
      for (each in grnlist) {
        each <- as.matrix(each)
        each <- abs(each)
        each <- each /max(each)
        grn2[id] <- grn2[id] + each[id]
      }
      grn2 <- grn2 / length(algorithm)
    }
  }
  grnmat <- grn2
  grnmat <- as.data.frame(grnmat)
  if (! is.null(file)) {
    readr::write_delim(x = grnmat,file = file,delim = '\t',col_names = F)
  }
  colnames(grnmat) <- TF
  rownames(grnmat) <- gene
  return(grnmat)
}
