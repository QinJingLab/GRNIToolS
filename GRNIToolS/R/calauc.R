
#' Calculate the value of AUROC and AUPR.
#' @description Calculate the value of the area under the ROC-curve and the value of the area under the PR-curve.
#' @param dat A data.frame. The first element, dat$predictions, is a vector of numerical predictions. The second element, dat$labels, is a vector of cordatponding class labels.
#'
#' @return A vector of the value of AUROC and AUPR.
#' @details
#' AUROC:
#'
#' Area under the ROC curve. This is equal to the value of the Wilcoxon-Mann-Whitney test statistic and also the probability that the classifier will score are randomly drawn positive sample higher than a randomly drawn negative sample.
#'
#' AUPR:
#'
#' Area under the Precision/Recall curve.
#' @references
#' Fawcett, T. 2004. "ROC Graphs: Notes and Practical Considerations for Researchers." In HPL-2003-4, 89-96. HP Labs, Palo Alto, CA.
#'
#' Sing, Tobias, Niko Beerenwinkel, and Thomas Lengauer. 2004.
#'  "Learning Mixtures of Localized Rules by Maximizing the Area Under the Roc Curve."
#'  In In et Al Jose Hernandez-Orallo, Editor, 1st International Workshop on Roc Analysis in Artificial Intelligence, 89-96.
#' @export
#' @examples
#' dat <- data.frame(predictions = runif(200, min=0, max=1),labels = c(rep(1,100),rep(0,100)))
#' auc <- calauc(dat)
#' print(auc)
calauc <- function(dat){
  if (length(unique(dat$labels))==1) {
    output <- c(0,0)
  }else {
        pred <- ROCR::prediction(dat$prediction,dat$labels)
        perf <- ROCR::performance(pred, "tpr", "fpr")
        fpr <- perf@x.values[[1]]
        tpr <- perf@y.values[[1]]
        auc <- ROCR::performance(pred, "auc")
        auc <- auc@y.values[[1]]
        auc <- round(auc,5)

        perf <- ROCR::performance(pred, "prec", "rec")
        rec <- perf@x.values[[1]]
        prec <- perf@y.values[[1]]
        aucpr <- ROCR::performance(pred, "aucpr")
        aucpr <- aucpr@y.values[[1]]
        aucpr <- round(aucpr,5)
        output <- c(auc, aucpr)
  }

  namesv <- c('AUROC','AUPR')
  names(output) <- namesv
  return(output)
}
