#' @title OMP - orthogonal matching pursuit
#' @description This function is the process of OMP (orthogonal matching pursuit).
#' @param A Gene expression data of transcriptome factors (i.e. basis function in machine learning).
#' @param b Gene expression data of target genes (i.e. observation in machine learning).
#' @param sparsity Sparsity level of solution.
#' @param epsilon1 First stopping rule in algorithm where the rule is \eqn{Ax-b < \epsilon_1}; we set default as 1e-6.
#' @param epsilon2 Second stopping rule in algorithm where the rule is \eqn{\max{|A'(Ax-b)|} < \epsilon_2}; we set default as 1e-6.
#' @author Yaohua Hu <mayhhu@szu.edu.cn> 
#' 
#' Xinlin Hu <ttxinlinhu@qq.com>
#' @details Orthogonal matching pursuit (OMP) algorithm was introduced by Tropp (2004) for the recovery of a high-dimensional sparse signal based on a small number of noisy linear measurements.
#' OMP is a kind of forward greedy methods for approaching an approximate solution of problem:
#' \deqn{\begin{array}{l}\min ||Ax - b|{|^2}\\s.t.||x|{|_0} \le s\end{array}}
#' OMP is a kind of forward greedy methods for approaching an approximate solution of convex problems. The idea of OMP is to iteratively select a column of A that is most correlated with the current residuals, and hence to make most significant progress at each iteration in order to reduce the loss function and achieve sparsity.
#' @references Tropp, J. A. (2004). "Greed is good: algorithmic results for sparse approximation", IEEE Transactions on Information Theory, 50(10), 2231-2242.
#'
OMP <- function(A, b, sparsity, epsilon1 = 1e-6, epsilon2 = 1e-6, verbose = FALSE){
    Am <- nrow(A)
    An <- ncol(A)
    r <- b
    cor <- t(A) %*% r
    index <- integer(0)

    for (iter in seq_len(sparsity)) {
        # pick the column with the largest |cor|; if already selected, move on to
        # the next unselected one
        cand <- order(abs(cor), decreasing = TRUE)
        ind <- cand[!cand %in% index][1]
        index <- c(index, ind)

        AT <- A[, index, drop = FALSE]
        # pseudo-inverse rather than solve(): t(AT) %*% AT is singular when
        # genes/TFs are collinear, and solve() fails there
        coef <- MASS::ginv(crossprod(AT)) %*% t(AT) %*% b
        r <- b - AT %*% coef
        cor <- t(A) %*% r

        if (verbose) print(sprintf('The %d -th variable is found.', length(index)))
        if (norm(r, '2') < epsilon1 || max(abs(cor)) < epsilon2) break
    }

    AT <- A[, index, drop = FALSE]
    xind <- MASS::ginv(crossprod(AT)) %*% t(AT) %*% b
    x <- rep(0, An)
    x[index] <- xind

    # result of OMP
    f <- norm(A %*% x - b, '2')^2

    # Collect history information
    Output <- list(x, f)
    return(Output)
}