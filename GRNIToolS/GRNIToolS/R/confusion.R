
#' Confusion Matrix
#' @description Calculate the confusion martix, takes the weighted adjacency matrix of the inferred network by this algorithm and the true or high-throughput golden standard network for validation.
#'
#' @param dat A data.frame. The first element, dat$predictions, is a vector of numerical predictions. The second element, dat$labels, is a vector of cordatponding class labels.
#' @param threshold Between 0 and 1. If their weight value is over the threshold, they are considered to have a regulatory relationship, otherwise they are not considered to have regulatory relationship. Default: 0.
#'
#' @return A vector of confusion matrix including some evalution parameters. (TP, FP, FN, precision, recall, F1-score).
#' @export
#'
#' @references G. Altay, F. Emmert-Streib, "Inferring the conservative causal core of gene regulatory networks", BMC Systems Biology, (2010) 4:132.
#' @examples
#' dat <- data.frame(predictions = runif(200, min=0, max=1),labels = c(rep(1,100),rep(0,100)))
#' score <- confusion(dat)
#' print(score)
#'
confusion <- function(dat,threshold=0){
  predictions <- dat$predictions
  labels <- dat$labels
  TP <- length(which(predictions >= threshold & labels==1))
  FP <- length(which(predictions >= threshold & labels==0))
  FN <- length(which(predictions < threshold & labels==1))
  precision <- TP/(TP + FP)
  recall <- TP/(TP + FN)
  #specificity <- TN/(TN + FP)
  F1score <- 2 * TP/(2 * TP + FP + FN)
  output <- c(precision, recall, F1score, TP,
              FP, FN)
  namesv <- c("Precision", "Recall", "F1-score",
              "TP", "FP", "FN")
  names(output) <- namesv
  output <- round(output,4)
  return(output)
}
