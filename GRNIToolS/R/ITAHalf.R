#' @title ITAHalf - iterative half thresholding algorithm
#' @description This function is the process of ITA-Half algorithm aims to solve \eqn{l_{\frac{1}{2}}} regularization optimization model, and it is an extension of ISTA.
#' @param A Gene expression data of transcriptome factors (i.e. basis function in machine learning).
#' @param b Gene expression data of target genes (i.e. observation in machine learning).
#' @param x0 Gene expression data of Chromatin immunoprecipitation or zero vector (i.e. initial iterative point in machine learning).
#' @param sparsity Sparsity level of solution.
#' @param MaxIter Maximum iteration used in calculation; we set default as 200.
#' @param epsilon Stopping rule in algorithm where the rule is \eqn{||Ax-b|{|_2^2} < \epsilon}; we set default as 1e-6.
#' @author Yaohua Hu <mayhhu@szu.edu.cn> 
#' 
#' Xinlin Hu <ttxinlinhu@qq.com>
#' @details Iterative half thresholding algorithm (ITA-Half) was introduced by Xu et al. (2012), which is an extension of the ISTA for solving the \eqn{l_{\frac{1}{2}}} regularization problem:
#' \deqn{{\min}_{x \in R^n} ||Ax - b|{|_2^2} + \lambda ||x|{|_{1/2}^{1/2}}}
#' Similar to ISTA, the idea of ITA-Half is to sequentially proceed a gradient descent step and a half thresholding operator at each iteration.
#' @references Xu, Z., Chang, X., Xu, F., and Zhang, H. (2012). "\eqn{l_{\frac{1}{2}}} regularization: A thresholding representation theory and a fast solver", IEEE Transactions on Neural Networks and Learning Systems, 23, 1013-1027.
#' 
#' Hu, Y., Li, C., Meng, K., Qin, J., and Yang, X. (2017). "Group sparse optimization via \eqn{l_{p,q}} regularization", Journal of Machine Learning Research, 18(30), 1-52.
ITAHalf <- function(A, b, x0, sparsity, MaxIter = 200, epsilon = 1e-6){
    f <- c()
    x <- x0
    # mu is double of stepsize v, which we choose as 1/2
    mu <- 1 
    xn <- length(x)
    Va1 <- sqrt(96)/9
    Va2 <- 54^(1/3)/4
    Bu1 <- mu * t(A) %*% b
    Bu2 <- mu * t(A) %*% A
    
    for(k in 1:MaxIter){
        #the first step: gradient step.
        Bu <- x + Bu1 - Bu2 %*% x
        xlp <- sort(abs(Bu), decreasing = TRUE)
        xrank <- order(abs(Bu), decreasing = TRUE)
        BuV <- xlp[sparsity+1]^(3/2)
        lambda <- Va1 * BuV
        criterion <- xlp[sparsity+1]
        
        x <- rep(0, xn)
        # The second step: half thresholding operator
        for(i in 1:sparsity){
            phi <- acos(lambda * mu/8 * (abs(Bu[xrank[i]])/3)^(-1.5))
            x[xrank[i]] <- 2 * Bu[xrank[i]] * (1 + cos(2/3 * (pi - phi)))/3
        }
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
