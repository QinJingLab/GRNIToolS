#' @title ITAHard - iterative hard thresholding algorithm
#' @description This function is the process of ITA-Hard algorithm aims to solve \eqn{l_0} regularization optimization model, and it is an extension of ISTA.
#' @param A Gene expression data of transcriptome factors (i.e. basis function in machine learning).
#' @param b Gene expression data of target genes (i.e. observation in machine learning).
#' @param x0 Gene expression data of Chromatin immunoprecipitation or zero vector (i.e. initial iterative point in machine learning).
#' @param sparsity Sparsity level of solution.
#' @param MaxIter Maximum iteration used in calculation; we set default as 200.
#' @param epsilon Stopping rule in algorithm where the rule is \eqn{||Ax-b|{|_2^2} < \epsilon}; we set default as 1e-6.
#' @author Yaohua Hu <mayhhu@szu.edu.cn> 
#' 
#' Xinlin Hu <ttxinlinhu@qq.com>
#' @details Iterative hard thresholding algorithm (ITA-Hard) was introduced by Blumensath and Davies (2008), which is an extension of the ISTA for solving the \eqn{l_0} regularization problem:
#' \deqn{{\min}_{x \in R^n} ||Ax - b|{|_2^2} + \lambda ||x|{|_0}}
#' Similar to ISTA, the idea of ITA-Hard is to sequentially proceed a gradient descent step and a hard thresholding operator at each iteration.
#' @references Blumensath, T. and Davies, M. E. (2008). "Iterative thresholding for sparse approximations", Journal of Fourier Analysis and Applications, 14, 629-654.
#' 
ITAHard <- function(A, b, x0, sparsity, MaxIter = 200, epsilon = 1e-6){
    f <- c()
    x <- x0
    # mu is double of stepsize v, which we choose as 1/2
    mu <- 1 
    xn <- length(x)
    Bu1 <- mu * t(A) %*% b
    Bu2 <- mu * t(A) %*% A
    
    for(k in 1:MaxIter){
        # The first step: gradient step.
        Bu <- x + Bu1 - Bu2 %*% x
        xlp <- sort(abs(Bu), decreasing = TRUE)
        xrank <- order(abs(Bu), decreasing = TRUE)
        
        x <- rep(0, xn)
        # The second step: hard thresholding operator
        x[xrank[1:sparsity]] <- Bu[xrank[1:sparsity]]
        print(sprintf('The %d -th iteration is finished.',k))
        
        f <- c(f, norm(A %*% x - b, '2')^2)
        if(f[length(f)] < epsilon){
            break
        }
    }
    # Collect history information
    Output <- list(x, f)
    return(Output)
}