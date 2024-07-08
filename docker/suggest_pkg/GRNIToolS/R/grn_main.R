#' grn_main
#' The pipeline of GRNIToolS.
#' @description A framework for benchmarking of gene regulatory network inference. This is the main function to call different algorithms for network inference and evaluate this network.
#'
#' @param A Gene expression data of transcriptome factors. Expression matrix (tfs x samples). Every row is a tf, every column is a sample. 
#' The class of A are required to be 'matrix' and the dimension of matrix A is m * n.
#' @param B Gene expression data of target genes. Expression matrix (genes x samples). Every row is a gene, every column is a sample. 
#' The class of B are required to be 'matrix' and the dimension of matrix B is u * n.
#' @param seq Sparsity level of solution. User can input a sequence of sparsity, i.e. 's <- c(1,2,3,4,5)'.
#' @param X Gene expression data of Chromatin immunoprecipitation or zero matrix.The class of X are required to be
#' 'matrix' and the dimension of matrix X is u * m.
#' @param max.steps Maximum iteration used in calculation. Default: 200.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @param p If set to FALSE, this demo aims to use the algorithms in lasso type conveniently.If set to TRUE, this demo aims to use the algorithms in plasso type conveniently(The matrix X must need).Default: FALSE.
#' @param tau Tuning parameter of prior lasso (plasso); using cross-validation can get a great one to calculate. The default is 10.
#' @param cl.cores The number of cores in computer that you want to utilize. Default: 2.
#' @param algorithm The algorithm of inferring grn. We provide 31 algorithms in this function including
#' 'Pearson','Spearman','Kendall','WGCNA','BC3NET','C3NET','parmigene_ARACNE.a','parmigene_ARACNE.m','parmigene_CLR','parmigene_MRNET',
#' 'minet_ARACNE','minet_CLR','minet_MRNET','minet_MRNETB','GeneNet','GENIE3.ET','GENIE3.RF','ENNET,'PCIT','ADMM','ADMMHalf',ADMMHard', 'ITAHalf', 'ITAHard',
#' 'ISTA', 'SPGL0', 'SPGL1', 'CoSaMP', 'OMP', 'FoBa' and 'LARS'. Defaule: c('Pearson','parmigene_ARACNE.a','parmigene_CLR')
#' @param HGSmat A matrix which is the high-throughput golden standard. Rows contain genes and columns contain tfs ,the value between in one or zero .
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param verbose If set to TRUE, a feedback on the progress of the progress of the calculations is given. Default: TRUE.
#' @param bootstrap_num Number of the times of resampling with bootstrap method.The number of new datasets.Default: 5.
#' @param bootstrap_seed If set to True, set.seed for keeping the results the same for each run. Default: TRUE.
#' @param sym If set to TRUE, only the regulatory relationship in the high-throughput gold standard is considered, and the value of 0 is not considered, so that the number of true positive and false positive is equal, and both false negative and true negative are 0.
#' If set to FALSE, It is assumed that the regulatory relationships that do not exist in the high-throughput gold standard are non regulatory and 0.Default: TRUE.
#' @param num The number of gene regulator information. Default: NULL
#' @param threshold Between 0 and 1. When the weight is over the threshold, If their weight value is over the threshold, they are considered to have a regulatory relationship, otherwise they are not considered to have regulatory relationship. Default: 0.
#' @return A data.frame with the evaluation parameters of the inferred gene regulatory network.
#' @export
#' @details This is a pipeline for benchmark of gene regulatory network inference and evaluation.Given the expression matrix, gene and high-throughput gold standard, then select the methods from the given network inference methods, and use these methods for network inference and evaluation.
#' This function provides a bootstrap method, which can repeatedly sample the expression matrix to form multiple new data to evaluate the stability of the inference method of regulatory network.
#' @examples
#' ntf <- 30
#' ngene <- 100
#' nsample <- 10
#' set.seed(123)
#' A <- matrix(rnorm(ntf * nsample,0,1),ntf,nsample)
#' TF <- paste0('G',1:ntf)
#' B <- matrix(rnorm(ngene * nsample,0,1),ngene,nsample)
#' X <- matrix(0,ngene,ntf)
#' s <-  c(1:10)
#' gene <- paste0('G',1:ngene)
#' HGSmat <- matrix(0,ngene,ntf)
#' colnames(HGSmat) <- TF
#' rownames(HGSmat) <- gene
#' HGSmat[sample(ngene * ntf,0.2 * ngene * ntf)] <- 1
#' algorithm <- c('Pearson','parmigene_ARACNE.a','minet_MRNET','PCIT','ADMM')
#' conlist <- grn_main(A=A,B=B,X=X,seq = s,gene=gene,TF=TF,algorithm = algorithm,
#'                     HGSmat = HGSmat,file=TRUE, verbose=TRUE,sym = FALSE)
#' print(conlist)                  
grn_main <- function(A=NULL, B ,seq=NULL, X=NULL, max.steps = 200, TF=NULL, gene,p=FALSE,tau=10, cl.cores=2, algorithm=c('Pearson','ARACNE.a','CLR'), HGSmat, file=TRUE, verbose=TRUE, bootstrap_num=5,bootstrap_seed=TRUE,sym = T,num = NULL,threshold = 0){
  if (is.null(TF)) {
    TF <- gene
  }
  auclist <- c()
  aucprlist <- c()
  timelist <- c()
  conlist <- c()
  g <- c()
  reg_algorithm <- c('ADMM','ADMMHalf','ADMMHard', 'ITAHalf', 'ITAHard', 'ISTA', 'SPGL0', 'SPGL1', 'CoSaMP', 'OMP', 'FoBa','LARS')
  if (bootstrap_num > 1) {
    B <- bootstrap(B,num = bootstrap_num,seed = bootstrap_seed)
  }else {B <- list(B)}
  for (alg in algorithm) {
    nt <- 0
    if (file) {
      for (datexpr in B) {

        if (bootstrap_num > 1) {
          nt <- nt + 1
          if (alg %in% reg_algorithm) {
            if (sum(abs(X))!=0) {
              reg_name <- paste0(alg,'_bootstrap',nt,'_CHIP.txt')
            }else{reg_name <- paste0(alg,'_bootstrap',nt,'.txt')}
            temp <- demo_reg(A,B=datexpr,seq,X,alg,max.steps,TF,gene,p=p,tau=10,cl.cores,file=reg_name,verbose = verbose)
          }else{ temp <- demo_other(datexpr=datexpr,algorithm = alg,TF=TF,gene=gene,file=paste0(alg,'_bootstrap',nt,'.txt'),cl.cores=cl.cores,verbose = verbose)}

        }else {
          if (alg %in% reg_algorithm) {
            if (sum(abs(X))!=0) {
              reg_name <- paste0(alg,'_CHIP.txt')
            }else{reg_name <- paste0(alg,'.txt')}
            temp <- demo_reg(A,B=datexpr,seq,X,alg,max.steps,TF,gene,p=p,tau=10,cl.cores=cl.cores,file=reg_name,verbose = verbose)
          }else {temp <- demo_other(datexpr=datexpr,algorithm = alg,TF=TF,gene=gene,file=paste0(alg,'.txt'),cl.cores=cl.cores,verbose = verbose)}}

        grn <- temp$grn
        colnames(grn) <- TF
        row.names(grn) <- gene
        time <- temp$time
        if (nrow(grn)!=0 ) {
          
          dat <- calpl(HGSmat,grn,sym = sym,num = num)
          if ( ! is.null(dat)) {
            con <- confusion(dat)
            auc <- calauc(dat)
          }else
            {con = c(rep(0,8))
              auc = c(0,0)}

        }else{
          con = c(rep(0,8))
          auc = c(0,0)
        }
        auclist <- c(auclist,auc[1])
        aucprlist <- c(aucprlist,auc[2])
        conlist <- rbind(conlist,con)
        timelist <- c(timelist,as.numeric(time))
        g <- c(g,alg)
      }
    }
    if (!file){
      for (datexpr in B) {
        if (alg %in% reg_algorithm) {
          temp <- demo_reg(A,B=datexpr,seq = seq,X,alg,max.steps,TF,gene,p=p,tau=10,cl.cores = cl.cores,file=NULL,verbose = verbose)
        }else{temp <- demo_other(datexpr=datexpr,algorithm = alg,TF=TF,gene=gene,file=NULL,cl.cores=cl.cores,verbose = verbose)}
        grn <- temp$grn
        time <- temp$time
        if (nrow(grn)!=0 ) {
          dat <- calpl(HGSmat,grn,sym = sym,num = num)
          if ( ! is.null(dat)) {
            con <- confusion(dat)
            auc <- calauc(dat)
          }else
          {con = c(rep(0,8))
          auc = c(0,0)}

        }else{
          con = c(rep(0,8))
          auc = c(0,0)
        }
        auclist <- c(auclist,auc[1])
        aucprlist <- c(aucprlist,auc[2])
        conlist <- rbind(conlist,con)
        timelist <- c(timelist,as.numeric(time))
        g <- c(g,alg)
      }
    }
  }
  conlist <- as.data.frame(conlist)
  conlist$AUROC <- auclist
  conlist$AUPR <- aucprlist
  conlist <- cbind(algorithm=g,conlist)
  rownames(conlist) <- c(1:nrow(conlist))
  conlist$Time <- timelist
  colnames(conlist)[1] <- 'Algorithm'
  return(conlist)

}
