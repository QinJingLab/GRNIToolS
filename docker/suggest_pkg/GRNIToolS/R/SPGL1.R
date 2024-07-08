#' @title SPGL1 - spectral projected gradient method to solve L1 regularization problem
#' @description This function is the process of SPGL1 algorithm aims to solve \eqn{l_1} regularization optimization model.
#' @param A Gene expression data of transcriptome factors (i.e. basis function in machine learning).
#' @param b Gene expression data of target genes (i.e. observation in machine learning).
#' @param x0 Gene expression data of Chromatin immunoprecipitation or zero vector (i.e. initial iterative point in machine learning).
#' @param sparsity Sparsity level of solution.
#' @param MaxIter Maximum iteration used in calculation; we set default as 200.
#' @param epsilon Stopping rule in algorithm where the rule is \eqn{||Ax-b|{|_2^2} < \epsilon}; we set default as 1e-6.
#' @author Yaohua Hu <mayhhu@szu.edu.cn> 
#' 
#' Xinlin Hu <ttxinlinhu@qq.com>
#' @details Spectral projected gradient method (SGPL1) was proposed by van den Berg and Friedlander (2008) to solve the \eqn{l_1} optimization problem:
#' \deqn{{\min}_{x \in R^n} ||Ax - b|{|_2^2} + \lambda ||x|{|_1}}
#' SPGL1 is indeed the projected gradient method with Barzilai-Borwein stepsize rule Barzilai and Borwein (1988), that is, first takes the gradient descent step with Barzilai-Borwein line-search scheme and then proceeds the projection onto the \eqn{l_1} norm ball at each iteration.
#' @references van den Berg, E. and Friedlander, M. P. (2008). "Probing the pareto frontier for basis pursuit solutions", SIAM Journal on Scientific Computing, 31(2), 890-912.
#' 
SPGL1 <- function(A, b, x0, sparsity, MaxIter = 200, epsilon = 1e-6){
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
        u <- sort(abs(x), decreasing = TRUE)
        rp <- order(abs(x), decreasing = TRUE)
        r <- sum(abs(x[rp[1:sparsity]]))
        sv <- cumsum(u)
        ind <- which(u > (sv - r)/t(1:length(u)))
        rho <- ind[length(ind)]
        theta <- max(0, (sv[rho] - r)/rho)
        at <- abs(x) - theta
        at[which(at <= 0)] <- 0
        x <- sign(x) * at
        
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