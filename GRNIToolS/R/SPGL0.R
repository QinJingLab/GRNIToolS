#' @title SPGL0 - spectral projected gradient method to solve L0 regularization problem
#' @description This function is the process of SPGL0 algorithm aims to solve \eqn{l_0} regularization optimization model, and it can be understood as an extension of SPGL1.
#' @param A Gene expression data of transcriptome factors (i.e. basis function in machine learning).
#' @param b Gene expression data of target genes (i.e. observation in machine learning).
#' @param x0 Gene expression data of Chromatin immunoprecipitation or zero vector (i.e. initial iterative point in machine learning).
#' @param sparsity Sparsity level of solution.
#' @param MaxIter Maximum iteration used in calculation; we set default as 200.
#' @param epsilon Stopping rule in algorithm where the rule is \eqn{||Ax-b|{|_2^2} < \epsilon}; we set default as 1e-6.
#' @author Yaohua Hu <mayhhu@szu.edu.cn> 
#' 
#' Xinlin Hu <ttxinlinhu@qq.com>
#' @details Projected gradient method (named as M-sparse algorithm) was introduced by Blumensath and Davies (2008). Inspired by the idea of Barzilai and Borwein stepsize rule Barzilai and Borwein (1988), we here propose and apply the SGPL0 (spectral projected gradient method) to solve the \eqn{l_0} regularization problem:
#' \deqn{{\min}_{x \in R^n} ||Ax - b|{|_2^2} + \lambda ||x|{|_0}}
#' The only difference from the M-sparse algorithm is to use the Barzilai-Borwein stepsize rule replace the constant stepsize. SPGL0 can also be understood as an extension of SGPL1, in which we use projection onto M coefficients with the largest magnitude instead of the projection onto the \eqn{l_1} norm ball.
#' @references Blumensath, T. and Davies, M. E. (2008). "Iterative thresholding for sparse approximations", Journal of Fourier Analysis and Applications, 14, 629-654.
#' 
#' Barzilai, J. and Borwein, J. M. (1988). "Two-point step size gradient methods", IMA Journal of Numerical Analysis, 8(1),141-148.
#' 
SPGL0 <- function(A, b, x0, sparsity, MaxIter = 200, epsilon = 1e-6){
    f <- c()
    x <- x0
    alphamax <- 2
    alphamin <- 0.1
    Am <- nrow(A)
    An <- ncol(A)
    g <- rep(0, An)
    alpha <- alphamax
    
    for(k in 1:MaxIter){
        xold <- x
        gold <- g
        ## gradient step 
        re <- b - A %*% x
        g <- -t(A) %*% re
        x <- x - alpha * g
        ## projection onto L-half ball
        y <- x
        xp <- sort(abs(y), decreasing = TRUE)
        rp <- order(abs(y), decreasing = TRUE)
        x <- rep(0, An)
        x[rp[1:sparsity]] <- y[rp[1:sparsity]]
        
        ## stop criterion
        f <- c(f, norm(A %*% x - b, '2')^2)
        if(f[length(f)] < epsilon){
            break
        }
        ## Barzilai-Borwein stepsize rule
        dx <- x - xold
        dg <- g - gold
        if(t(dx) %*% dg <= 0){
            alpha <- alphamax
        }else{
            BB <- t(dx) %*% dx/(t(dx) %*% dg)
            alpha <- min(alphamax, max(alphamin, BB))
        }
        print(sprintf('The %d -th iteration is finished.',k))
    }
    
    y <- x
    xp <- sort(abs(y), decreasing = TRUE)
    rp <- order(abs(y), decreasing = TRUE)
    x <- rep(0, An)
    x[rp[1:sparsity]] <- y[rp[1:sparsity]]
    # Collect history information
    Output <- list(x, f)
    return(Output)
}