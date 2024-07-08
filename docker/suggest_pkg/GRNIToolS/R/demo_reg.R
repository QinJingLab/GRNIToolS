#' Demo reg - The demo of regression method
#' @description This is the main function to call different algorithms of regression for gene regulator network inference.
#' @param A Gene expression data of transcriptome factors. Expression matrix (tfs x samples). Every row is a tf, every column is a sample. 
#' The class of A are required to be 'matrix' and the dimension of matrix A is m * n.
#' @param B Gene expression data of target genes. Expression matrix (genes x samples). Every row is a gene, every column is a sample. 
#' The class of B are required to be 'matrix' and the dimension of matrix B is u * n.
#' @param seq Sparsity level of solution. User can input a sequence of sparsity, i.e. 's <- c(1,2,3,4,5)'.
#' @param X Gene expression data of Chromatin immunoprecipitation or zero matrix.The class of X are required to be
#' 'matrix' and the dimension of matrix X is u * m.
#' @param algorithm We provide 12 algorithms in this function including 'ADMM', 'ADMMHalf',
#' 'ADMMHard', 'ITAHalf', 'ITAHard', 'ISTA', 'SPGL0', 'SPGL1', 'CoSaMP', 'OMP', 'FoBa' and 'LARS'. Default: ADMM.
#' @param max.steps Maximum iteration used in calculation. Default: 200.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @param p If set to FALSE, this demo aims to use the algorithms in lasso type conveniently.If set to TRUE, this demo aims to use the algorithms in plasso type conveniently(The matrix X must need).Default: FALSE.
#' @param tau Tuning parameter of prior lasso (plasso); using cross-validation can get a great one to calculate. The default is 10.
#' @param cl.cores The number of cores in computer that you want to utilize. Default: 2.
#' @param file A connection, or a character string naming the file to write to. The default not save the grn matrix. Default: NULL.
#' @param verbose If set to TRUE, a feedback on the progress of the progress of the calculations is given. Default: TRUE.
#' @return A list with the function of parameters ,algorithms, elapse time, grn which a matrix which is the weighted adjacency matrix of the inferred network by the regression algorithm.
#' @author Yaohua Hu <mayhhu@szu.edu.cn>
#'
#' Xinlin Hu <ttxinlinhu@qq.com>
#'
#' Yongqiang Zhou <zhouyq67@mail2.sysu.edu.cn>
#' @details The demo function aims to call algorithms to solve optimization problem.
#' This functions provides 12 algorithms for users.
#' This demo run parallelly and all cores can be utilized to speed up.
#' We import 'foba' and 'lars' from CRAN to run 'FoBa' and 'LARS' algorithms.
#'
#' LARS (least angle regression) algorithm is a variant of forward greedy methods proposed by
#' Efron et al. (2004) for approaching an approximate solution of problem:
#' \deqn{\begin{array}{l}\min ||Ax-b|{|^2}\\s.t.||x|{|_1} \le s\end{array}}
#' LARS is a hybrid of forward greedy selection and stagewise regression.
#'
#' Combining the ideas of forward and backward greedy algorithms, Zhang (2001) designed an
#' adaptive forward-backward greedy (FoBa) algorithm for solving problem:
#' \deqn{\begin{array}{l}\min ||Ax-b|{|^2}\\s.t.||x|{|_0} \le s\end{array}}
#' FoBa adopts OMP to select features and takes adaptive backward steps to remove any mistake
#' caused by earlier forward steps. The backward step shall be employed when the increase of
#' loss function is no more than half of the decrease of loss function in earlier forward steps.
#' This principle of backward steps guarantee removing any mistake caused by earlier forward steps
#' and avoiding to earse the gain made in the forward steps. FoBa can fix the fundamental flaws of
#' both forward and backward greedy methods, and shares their fast implementation advantage.
#'
#' @references Efron, B., Hastie, T., Johnstone, I., and Tibshirani, R. (2004). "Least angle
#' regression", Annals of Statistics ,32, 407-499.
#'
#' Zhang, T. (2011). "Adaptive forward-backward greedy algorithm for
#' learning sparse representations", IEEE Transactions on Information Theory, 57(7), 4689-4708.
#'
#' @import foba lars foreach parallel doParallel utils stringr
#' @importFrom stats predict
#' @export
#' @examples
#' A <- matrix(rnorm(200,0,1),20,10)
#' B <- matrix(rnorm(1000,0,1),100)
#' X <- matrix(0,100,20)
#' s <- c(1:10)
#' TF <- paste0('G',1:20)
#' gene <- paste0('G',1:100)
#' res <- demo_reg(A, B, s, X, 'ADMM',max.steps = 200,TF, gene,
#'                 p = FALSE,tau=10, cl.cores = 2,file=NULL,verbose=TRUE)
demo_reg <- function(A, B, seq, X, algorithm = 'ADMM', max.steps = 200, TF, gene,p = FALSE,tau=10, cl.cores = 2,file = NULL,verbose = TRUE){
  i <- c()
  reslist <- list()
  call <- match.call()
  path1 <- getwd()

  if (! is.null(file)) {
    dir <- 'Output'
    if (! dir.exists(dir)) {
      dir.create(dir)
    }
    Outpath <- str_c(dir,'/Output_', algorithm)
    if (! dir.exists(Outpath)) {
      dir.create(Outpath)
    }
    setwd(Outpath)
  }




  dm <- nrow(B)
  NoA <- norm(A, '2')
  A <- t(A)/NoA
    # ADMM method
    if(algorithm == 'ADMM' || algorithm == 'ADMMHalf' || algorithm == 'ADMMHard'){
      # mainADMM function
      mainADMM <- function(A, B, X, sparsity, algorithm, NoA, MaxIter, epsilon, i,p,tau){
        b <- B[i, ]/NoA
        x0 <- X[i, ]
        if (p) {
          x1 <- X[i,]
          b <- (b+tau*A%*%x1)/(1+tau)
        }
        # Call ADMM function
        if(algorithm == 'ADMM'){
          OutData <- ADMM(A, b, x0, sparsity, MaxIter, epsilon)
        }else if(algorithm == 'ADMMHalf'){
          OutData <- ADMMHalf(A, b, x0, sparsity, MaxIter, epsilon)
        }else if(algorithm == 'ADMMHard'){
          OutData <- ADMMHard(A, b, x0, sparsity, MaxIter, epsilon)
        }
        # Output
        xOut <- OutData[[1]]
        hist <- OutData[[2]]
        fOut <- hist[length(hist)]
        Output <- cbind(t(xOut), fOut)
        return(Output)
      }

      t1 <- Sys.time()
      for(s in seq){
        cl <- makeCluster(cl.cores)
        registerDoParallel(cl)


        solADMM <- foreach(i = 1:dm, .combine = rbind, .export = algorithm) %dopar%
          mainADMM(A, B, X, s, algorithm, NoA, max.steps, epsilon = 1e-9, i,p,tau)
        xADMM <- solADMM[, -ncol(solADMM)]
        FADMM <- solADMM[, ncol(solADMM)]

        if (verbose) {
          cat(sprintf('The %d -th sparsity is finished.\n', s))
        }

        stopCluster(cl)

        if (! is.null(file)) {
          write.table(xADMM, file = str_c('x', s, '.txt'),
                      row.names = FALSE, col.names = FALSE,
                      quote = F, sep="\t", fileEncoding = "ASCII")
        }
        reslist <- c(reslist,list(xADMM))

      }
      t2 <- Sys.time()
    }

    # ITA method
    if(algorithm == 'ISTA' || algorithm == 'ITAHalf' || algorithm == 'ITAHard'){
      # mainITA function
      mainITA <- function(A, B, X, sparsity, algorithm, NoA, MaxIter, epsilon, i,p,tau){
        b <- B[i, ]/NoA
        x0 <- X[i, ]
        if (p) {
          x1 <- X[i,]
          b <- (b+tau*A%*%x1)/(1+tau)
        }
        # Call ITA function
        if(algorithm == 'ISTA'){
          OutData <- ISTA(A, b, x0, sparsity, MaxIter, epsilon)
        }else if(algorithm == 'ITAHalf'){
          OutData <- ITAHalf(A, b, x0, sparsity, MaxIter, epsilon)
        }else{
          OutData <- ITAHard(A, b, x0, sparsity, MaxIter, epsilon)
        }
        # Output
        xOut <- OutData[[1]]
        hist <- OutData[[2]]
        fOut <- hist[length(hist)]
        Output <- cbind(t(xOut), fOut)
        return(Output)
      }

      t1 <- Sys.time()
      for(s in seq){
        cl <- makeCluster(cl.cores)
        registerDoParallel(cl)

        solITA <- foreach(i = 1:dm, .combine = rbind, .export = algorithm) %dopar%
          mainITA(A, B, X, s, algorithm, NoA, max.steps, epsilon = 1e-6, i,p,tau)
        xITA <- solITA[, -ncol(solITA)]
        FITA <- solITA[, ncol(solITA)]
        if (verbose) {
          cat(sprintf('The %d -th sparsity is finished.\n', s))
        }
        stopCluster(cl)
        if (! is.null(file)) {
          write.table(xITA, file = str_c('x', s, '.txt'),
                      row.names = FALSE, col.names = FALSE,
                      quote = F, sep="\t", fileEncoding = "ASCII")
        }
        reslist <- c(reslist,list(xITA))
      }
      t2 <- Sys.time()
    }

    # SPGL method
    if(algorithm == 'SPGL0' || algorithm == 'SPGL1'){
      # mainSPGL function
      mainSPGL <- function(A, B, X, sparsity, algorithm, NoA, MaxIter, epsilon, i,p,tau){
        b <- B[i, ]/NoA
        x0 <- X[i, ]
        if (p) {
          x1 <- X[i,]
          b <- (b+tau*A%*%x1)/(1+tau)
        }
        # Call SPGL function
        if(algorithm == 'SPGL0'){
          OutData <- SPGL0(A, b, x0, sparsity, MaxIter, epsilon)
        }else{
          OutData <- SPGL1(A, b, x0, sparsity, MaxIter, epsilon)
        }
        # Output
        xOut <- OutData[[1]]
        hist <- OutData[[2]]
        fOut <- hist[length(hist)]
        Output <- cbind(t(xOut), fOut)
        return(Output)
      }
      t1 <- Sys.time()
      for(s in seq){
        cl <- makeCluster(cl.cores)
        registerDoParallel(cl)


        solSPGL <- foreach(i = 1:dm, .combine = rbind, .export = algorithm) %dopar%
          mainSPGL(A, B, X, s, algorithm, NoA, max.steps, epsilon = 1e-6, i,p,tau)
        xSPGL <- solSPGL[, -ncol(solSPGL)]
        FSPGL <- solSPGL[, ncol(solSPGL)]
        if (verbose) {
          cat(sprintf('The %d -th sparsity is finished.\n', s))
        }
        stopCluster(cl)
        if (! is.null(file)) {
          write.table(xSPGL, file = str_c('x', s, '.txt'),
                      row.names = FALSE, col.names = FALSE,
                      quote = F, sep="\t", fileEncoding = "ASCII")
        }
        reslist <- c(reslist,list(xSPGL))
      }
      t2 <- Sys.time()
    }

    # CoSaMP
    if(algorithm == 'CoSaMP'){
      # mainCoSaMP function
      mainCoSaMP <- function(A, B, X, sparsity, NoA, MaxIter, epsilon1, epsilon2, i,p,tau){
        b <- B[i, ]/NoA
        if (p) {
          x1 <- X[i,]
          b <- (b+tau*A%*%x1)/(1+tau)
        }
        OutData <- CoSaMP(A, b, sparsity, MaxIter, epsilon1, epsilon2)
        # Output
        xOut <- OutData[[1]]
        hist <- OutData[[2]]
        fOut <- hist[length(hist)]
        Output <- cbind(t(xOut), fOut)
        return(Output)
      }
      t1 <- Sys.time()
      for(s in seq){
        cl <- makeCluster(cl.cores)
        registerDoParallel(cl)


        solCoSaMP <- foreach(i = 1:dm, .combine = rbind, .export = algorithm, .packages = 'MASS') %dopar%
          mainCoSaMP(A, B, X, s, NoA, max.steps, epsilon1 = 1e-6, epsilon2 = 1e-6, i,p,tau)
        xCoSaMP <- solCoSaMP[, -ncol(solCoSaMP)]
        FCoSaMP <- solCoSaMP[, ncol(solCoSaMP)]
        if (verbose) {
          cat(sprintf('The %d -th sparsity is finished.\n', s))
        }
        stopCluster(cl)
        if (! is.null(file)) {
          write.table(xCoSaMP, file = str_c('x', s, '.txt'),
                      row.names = FALSE, col.names = FALSE,
                      quote = F, sep="\t", fileEncoding = "ASCII")
        }
        reslist <- c(reslist,list(xCoSaMP))
      }
      t2 <- Sys.time()
    }


    # OMP
    if(algorithm == 'OMP'){
      # mainOMP function
      mainOMP <- function(A, B, X, sparsity, NoA, epsilon1, epsilon2, i,p,tau){
        b <- B[i, ]/NoA
        if (p) {
          x1 <- X[i,]
          b <- (b+tau*A%*%x1)/(1+tau)
        }
        OutData <- OMP(A, b, sparsity, epsilon1, epsilon2)
        # Output
        xOut <- OutData[[1]]
        hist <- OutData[[2]]
        fOut <- hist[length(hist)]
        Output <- cbind(t(xOut), fOut)
        return(Output)
      }

      t1 <- Sys.time()
      for(s in seq){
        cl <- makeCluster(cl.cores)
        registerDoParallel(cl)

        solOMP <- foreach(i = 1:dm, .combine = rbind, .export = algorithm) %dopar%
          mainOMP(A, B, X , s, NoA, epsilon1 = 1e-6, epsilon2 = 1e-6, i,p,tau)
        xOMP <- solOMP[, -ncol(solOMP)]
        FOMP <- solOMP[, ncol(solOMP)]
        if (verbose) {
          cat(sprintf('The %d -th sparsity is finished.\n', s))
        }
        stopCluster(cl)
        if (! is.null(file)) {
          write.table(xOMP, file = str_c('x', s, '.txt'),
                      row.names = FALSE, col.names = FALSE,
                      quote = F, sep="\t", fileEncoding = "ASCII")
        }
        reslist <- c(reslist,list(xOMP))
      }
      t2 <- Sys.time()
    }

    # FoBa
    if(algorithm == 'FoBa'){
      # mainFoBa function
      mainFoBa <- function(A, B, X , sparsity, NoA, MaxIter, i,p,tau){
        b <- t(B[i, ])/NoA
        if (p) {
          x1 <- X[i,]
          b <- (b+t(tau*A%*%x1))/(1+tau)
        }
        model <- foba(A, b, steps = MaxIter, intercept = FALSE)
        if(length(model$path) == MaxIter){
          py <- predict(model, A, k = sparsity, type = 'fit')
          xOut <- t(as.matrix(py$coefficients))
          fOut <- norm(t(A %*% py$coefficients) - b, '2')^2
          Output <- matrix(c(xOut, fOut), 1)
        }else{
          Output <- matrix(0, nrow = 1, ncol = (ncol(A)+1))
        }
        # Output
        return(Output)
      }
      t1 <- Sys.time()
      for(s in seq){
        cl <- makeCluster(cl.cores)
        registerDoParallel(cl)

        solFoBa <- foreach(i = 1:dm, .combine = rbind, .packages = 'foba') %dopar%
          mainFoBa(A, B, X , s, NoA, max.steps, i,p,tau)
        xFoBa <- solFoBa[, -ncol(solFoBa)]
        FFoBa <- solFoBa[, ncol(solFoBa)]
        if (verbose) {
          cat(sprintf('The %d -th sparsity is finished.\n', s))
        }
        stopCluster(cl)
        if (! is.null(file)) {
          write.table(xFoBa, file = str_c('x', s, '.txt'),
                      row.names = FALSE, col.names = FALSE,
                      quote = F, sep="\t", fileEncoding = "ASCII")
        }
        reslist <- c(reslist,list(xFoBa))
      }
      t2 <- Sys.time()
    }

    # Lars
    if(algorithm == 'LARS'){
      # mainLars function
      mainLars <- function(A, B, X , sparsity, NoA, i,p,tau){
        b <- t(B[i, ])/NoA
        if (p) {
          x1 <- X[i,]
          b <- (b+t(tau*A%*%x1))/(1+tau)
        }
        model <- lars(A, b, type = 'lar', use.Gram = FALSE, normalize = FALSE, intercept = FALSE)
        # mode default as "step"
        if(length(model$entry[model$entry != 0]) >= sparsity + 1){
          py <- predict(model, s = sparsity + 1, type = 'coef')
          xOut <- t(as.matrix(py$coefficients))
        }else if(length(model$entry[model$entry != 0]) <= dim(model$beta)[1]){
          py <- predict(model, s = length(model$entry[model$entry != 0]), type = 'coef')
          xOut <- t(as.matrix(py$coefficients))
        }else{
          py <- predict(model, s = dim(model$beta)[1], type = 'coef')
          xOut <- t(as.matrix(py$coefficients))
        }
        # Output
        fOut <- norm(t(A %*% t(xOut)) - b, '2')^2
        Output <- matrix(c(xOut, fOut),1)
        return(Output)
      }
      t1 <- Sys.time()
      for(s in seq){
        cl <- makeCluster(cl.cores)
        registerDoParallel(cl)

        solLars <- foreach(i=1:dm, .combine = rbind, .packages = 'lars') %dopar%
          mainLars(A, B, X , s, NoA, i,p,tau)
        xLars <- solLars[, -ncol(solLars)]
        FLars <- solLars[, ncol(solLars)]
        if (verbose) {
          cat(sprintf('The %d -th sparsity is finished.\n', s))
        }
        stopCluster(cl)
        if (! is.null(file)) {
          write.table(xLars, file = str_c('x', s, '.txt'),
                      row.names = FALSE, col.names = FALSE,
                      quote = F, sep="\t", fileEncoding = "ASCII")
        }
        reslist <- c(reslist,list(xLars))
      }
      t2 <- Sys.time()

    }

  calgrn <- function(reslist,algname, TF, gene,sparsity,verbose=verbose){
    i <- 0      
    for(s in sparsity){
      i <- i + 1
      Score <- matrix(0, nrow = length(gene), ncol = length(TF))
      A <- reslist[[i]]
      A <- as.matrix(A)
      A <- apply(as.matrix(A),2,as.numeric)
      A <- A/max(abs(A))
      A <- A/s
      Score[which(A == 0, arr.ind = TRUE)] <- 0
      Score[which(A != 0, arr.ind = TRUE)] <- 1/s
      if (i == 1) {
        Scoremax <- Score
        grn <- A
      }else {scorecopy <- Scoremax
             Scoremax <- pmax(Scoremax,Score)
             id <- which(scorecopy != Scoremax,arr.ind = T)  
             if (nrow(id) > 0) {
               grn[id] <- A[id]
             }
      }
    }
    # Maximum Score
   # Score <- Scoremax
    if (verbose) {
      print(sprintf("%s has completed", algname))
    }
   # return(Score)
    grn <- grn / max(abs(grn))
    return(grn)
  }

  grn <- calgrn(reslist,algname = algorithm,TF = TF,gene = gene,sparsity = seq,verbose=verbose)
  colnames(grn) <- TF
  rownames(grn) <- gene
  time <- as.numeric(t2) - as.numeric(t1)
  if (is.null(file)) {
    object <- list(call = call, algorithm = algorithm, time = time, grn = grn)
  }else{
    object <- list(call = call, algorithm = algorithm, time = time, savePath = Outpath,grn = grn)
  }
  if (! is.null(file)) {
    grn <- as.data.frame(grn)
    if (p) {
      readr::write_delim(x = grn,file = paste0('plasso_',file),delim = '\t',col_names = F)
    }else{readr::write_delim(x = grn,file =  paste0('lasso_',file),delim = '\t',col_names = F)}
  }
  setwd(path1)

  return(object)
}
