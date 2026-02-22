#' @title ADMMHard - alternating direction method of multipliers with hard thresholding operator
#' @description This function is the process of ADMM-Hard algorithm aims to solve \eqn{l_0} regularization optimization model.
#' @param A Gene expression data of transcriptome factors (i.e. basis function in machine learning).
#' @param b Gene expression data of target genes (i.e. observation in machine learning).
#' @param x0 Gene expression data of Chromatin immunoprecipitation or zero vector (i.e. initial iterative point in machine learning).
#' @param sparsity Sparsity level of solution.
#' @param MaxIter Maximum iteration used in calculation; we set default as 200.
#' @param epsilon Stopping rule in algorithm where the rule is \eqn{||Ax-b|{|_2^2} < \epsilon}; we set default as 1e-9.
#' @author Yaohua Hu <mayhhu@szu.edu.cn>
#' 
#' Xinlin Hu <ttxinlinhu@qq.com>
#' @details ADMM-Hard was introduced by Boyd et al.(2011). It is an extension of ADMM, where the hard thresholding operator is adopted to replace the soft one in ADMM. It is aimed to solve the \eqn{l_0} regularization problem:
#' \deqn{{\min}_{x \in R^n} ||Ax - b|{|_2^2} + \lambda ||x|{|_0}}
#' 
ADMMHard <- function(A, b, x0, sparsity, MaxIter = 200, epsilon = 1e-9){
  f <- c()
  x <- x0
  Am <- nrow(A)
  An <- ncol(A)
  y <- rep(0, Am)
  gamma <- 1.6
  beta <- Am/norm(as.matrix(b), '1')
  tau <- 0.5
  
  for(k in 1:MaxIter){
    err <- A %*% x - b
    r <- (y - beta * err)/(1 + beta)
    g <- t(A)%*%(err + r - y/beta)
    Bu <- x - tau * beta * g
    xlp <- sort(abs(Bu), decreasing = TRUE)
    xrank <- order(abs(Bu), decreasing = TRUE)
    
    x <- rep(0, An)
    # The second step: hard thresholding operator
    x[xrank[1:sparsity]] <- Bu[xrank[1:sparsity]]
    
    y <- y - gamma * beta * (A %*% x + r - b)
    print(sprintf('The %d -th iteration is finished.', k))
    
    f <- c(f, norm(A %*% x - b, '2')^2)
    if(f[length(f)] < epsilon){
      break
    }
  }
  # Collect history information
  Output <- list(x, f)
  return(Output)
}