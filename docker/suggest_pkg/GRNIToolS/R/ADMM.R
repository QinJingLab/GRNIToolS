#' @title ADMM - alternating direction method of multiplier
#' @description This function is the process of ADMM (alternating direction method of multipliers) aims to solve \eqn{l_1} regularization optimization model of linear regression (also called as Least Absolute Shrinkage and Selection Operator, LASSO). 
#' @param A Gene expression data of transcriptome factors (i.e. basis function in machine learning).
#' @param b Gene expression data of target genes (i.e. observation in machine learning).
#' @param x0 Gene expression data of Chromatin immunoprecipitation or zero vector (i.e. initial iterative point in machine learning).
#' @param sparsity Sparsity level of solution.
#' @param MaxIter Maximum iteration used in calculation; we set default as 200.
#' @param epsilon Stopping rule in algorithm where the rule is \eqn{||Ax-b|{|_2^2} < \epsilon}; we set default as 1e-9.
#' @author Yaohua Hu <mayhhu@szu.edu.cn> 
#' 
#' Xinlin Hu <ttxinlinhu@qq.com>
#' @details Alternating direction method of multipliers (ADMM) was introduced by Yang and Zhang (2011) to solve the \eqn{l_1} regularization problem:
#' \deqn{{\min}_{x \in R^n} ||Ax - b|{|_2^2} + \lambda ||x|{|_1}}
#' The idea of ADMM is to apply the Gauss-Seidel decomposition technique to solve the joint minimization problem of the augmented Lagrangian function. By applying the Gauss-Seidel decomposition technique, the augmented Lagrangian function minimization can be calculated via analytical formulae consisting of a projected gradient descent step, a soft thresholding operator and an updating rule of multipliers.
#' @references Yang, J. and Zhang, Y. (2011). "Alternating direction algorithms for \eqn{l_1}-problems in compressive sensing", SIAM Journal on Scientific Computing, 33(1), 250-278.
#' 
ADMM <- function(A, b, x0, sparsity, MaxIter = 200, epsilon = 1e-9){
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
    criterion <- xlp[sparsity+1]
    
    x <- rep(0, An)
    # The second step: soft thresholding operator
    x[xrank[1:sparsity]] <- Bu[xrank[1:sparsity]] - criterion %*% sign(Bu[xrank[1:sparsity]])
    
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
