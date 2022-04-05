#' @title CoSaMP - compressive sampling matched pursuit
#' @description This function is the process of CoSaMP (Compressive sampling matched pursuit).
#' @param A Gene expression data of transcriptome factors (i.e. basis function in machine learning).
#' @param b Gene expression data of target genes (i.e. observation in machine learning).
#' @param sparsity Sparsity level of solution.
#' @param MaxIter Maximum iteration used in calculation; we set default as 200.
#' @param epsilon1 First stopping rule in algorithm where the rule is \eqn{Ax-b < \epsilon_1}; we set default as 1e-6.
#' @param epsilon2 Second stopping rule in algorithm where the rule is \eqn{\max{|A'(Ax-b)|} < \epsilon_2}; we set default as 1e-6.
#' @author Yaohua Hu <mayhhu@szu.edu.cn> 
#' 
#' Xinlin Hu <ttxinlinhu@qq.com>
#' @details Compressive sampling matched pursuit (CoSaMP) algorithm is an approximation algorithm proposed by Needell and Tropp (2009) for solving problem:
#' \deqn{\begin{array}{l}\min ||Ax-b|{|^2}\\s.t.||x|{|_0} \le s\end{array}}
#' 
#' CoSaMP is ultimately based on OMP, while it incorporates the sampling technique to accelerate the algorithm and to enjoy strong theoretical guarantees than OMP. At each iteration of CoSaMP, some random samples are added into the set of selected features, and then a least squares scheme is employed to estimate the approximation from the set of selected features. This process is repeated until the recoverable features have been found. CoSaMP shares the fast implementation advantage and remedies the correction inability of OMP.
#' @references 
#' Needell, D. and Tropp, J. (2009). "CoSaMP: Iterative signal recovery from incomplete and inaccurate samples", Applied and Computational Harmonic Analysis, 26(3), 301-321.
#' @import MASS
#' @importFrom MASS ginv

CoSaMP <- function(A, b, sparsity, MaxIter = 200, epsilon1 = 1e-6, epsilon2 = 1e-6){
    f <- c()
    Am <- nrow(A)
    An <- ncol(A)
    x <- rep(0, An) # store the recovered vector
    r <- b # initialize the residual with b
    cor <- t(A) %*% r
    
    for(k in 1:MaxIter){
        # -- Identification --
        Rm <- sort(abs(cor), decreasing = TRUE)
        ind <- order(abs(cor), decreasing = TRUE)
        indO <- ind[1:(2*sparsity)]
        indx <- which(x != 0)
        
        # -- Support Merger --
        index <- c(indO, indx)
        
        # -- Estimation --
        index <- unique(sort(index))
        AT <- A[, index]
        
        # -- The solution of Least Square --
        sol <- ginv(AT) %*% b
        
        # -- Pruning & Sample Update --
        y <- rep(0, An)
        y[index] <- sol
        Rm <- sort(abs(y), decreasing = TRUE)
        ind <- order(abs(y), decreasing = TRUE)
        x <- rep(0, An)
        x[ind[1:sparsity]] <- y[ind[1:sparsity]]
        
        # -- update the residual --
        r <- b - A %*% x
        cor <- t(A) %*% r
        
        print(sprintf('The %d -th iteration is finished.',k))
        
        f <- c(f, norm(A %*% x - b, '2')^2)
        if(norm(r, '2') < epsilon1 || max(abs(cor)) < epsilon2){
            break
        }
    }
    
    # Collect history information
    Output <- list(x, f)
    return(Output)
}