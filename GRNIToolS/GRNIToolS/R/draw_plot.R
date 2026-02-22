#' draw_plot
#' @description Draw the curve for the gene regulatory network.
#' @param dat A data.frame. The first element, dat$predictions, is a vector of numerical predictions. The second element, dat$labels, is a vector of cordatponding class labels.
#' @param method Receiveroperating characteristic curve, precision-recall curve and sensitivity-specificity curve (ROC,PR, SSC). Default: ROC.
#' @param algname The algorithm of gene network inferring.
#'
#' @return The curve which selected, such as ROC and PR.
#' @export
#' @import ROCR RColorBrewer ggplot2
#'
#' @examples
#' set.seed(1)
#' dat <- data.frame(predictions = runif(200, min=0, max=1),labels = c(rep(1,100),rep(0,100)))
#' p <- draw_plot(dat = dat,method = 'ROC',algname ='ADMM')
#' print(p)
draw_plot <- function(dat,method = 'ROC',algname){
  pred <- ROCR::prediction(dat$prediction,dat$labels)
  plot_ROC <- function(pred,algname)
  {
    perf <- ROCR::performance(pred, "tpr", "fpr")
    fpr <- perf@x.values[[1]]
    tpr <- perf@y.values[[1]]
    auc <- ROCR::performance(pred, "auc")
    auc <- auc@y.values[[1]]
    auc <- round(auc,3)
    Roc <- data.frame(fpr = fpr, tpr = tpr, algname = stringr::str_c(algname,' (AUROC = ', auc, ')'))
    palatte <- RColorBrewer::brewer.pal(5, 'Set1')
    g <- ggplot(Roc, aes(x = fpr, y = tpr)) + geom_line(aes(color = algname), size = 1) + theme_bw() +
      scale_color_manual(values = palatte[1:2]) +
      geom_segment(x = 0, y = 0, xend = 1, yend = 1,
                   color = '#a9d6e5', linetype = 'dashed', size = 1) +
      labs(x = 'False Positive Rate', y = 'True Positive Rate',
           title = '') + ylim(0,1) +
      theme(plot.title = element_text(hjust = 0.5), legend.title = element_blank(),
            legend.justification = c(1, 0), legend.position = c(0.95, 0.05),
            legend.background = element_blank(), legend.key = element_blank(),
            legend.text = element_text(size = 10,face = 'bold'),
            axis.text = element_text(size = 12,face = 'bold'),
            axis.title = element_text(size = 15,face = 'bold'),
            axis.title.x = element_text(margin = margin(0.5,1,0,1,'cm')),
            axis.title.y = element_text(margin = margin(0.5,0.5,0,0.5,'cm')),
            plot.margin=unit(rep(1,4),'lines'),
            panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black"))
    return(g)
  }

  plot_PR <- function(pred,algname)
  {
    perf <- ROCR::performance(pred, "prec", "rec")
    rec <- perf@x.values[[1]]
    prec <- perf@y.values[[1]]
    aucpr <- ROCR::performance(pred, "aucpr")
    aucpr <- aucpr@y.values[[1]]
    aucpr <- round(aucpr,3)
    PR <- data.frame(Recall = rec, prec = prec, algname = stringr::str_c(algname,' (AUPR = ', aucpr, ')'))
    palatte <- RColorBrewer::brewer.pal(5, 'Set1')
    g <- ggplot(PR, aes(x = Recall, y = prec)) + geom_line(aes(color = algname), size = 1) + theme_bw() +
      scale_color_manual(values = palatte[1:2]) +
      geom_segment(x = 0, y = 1, xend = 1, yend = 0,
                   color = '#a9d6e5', linetype = 'dashed', size = 1) +
      labs(x = 'Recall', y = 'Precision',
           title = '') + ylim(0,1) +
      theme(plot.title = element_text(hjust = 0.5), legend.title = element_blank(),
            legend.justification = c(1, 0), legend.position = c(0.95, 0.05),
            legend.background = element_blank(), legend.key = element_blank(),
            legend.text = element_text(size = 10,face = 'bold'),
            axis.text = element_text(size = 12,face = 'bold'),
            axis.title = element_text(size = 15,face = 'bold'),
            axis.title.x = element_text(margin = margin(0.5,1,0,1,'cm')),
            axis.title.y = element_text(margin = margin(0.5,0.5,0,0.5,'cm')),
            plot.margin=unit(rep(1,4),'lines'),
            panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black"))
    return(g)
  }

  plot_SS <- function(pred,algname)
  {
    perf <- ROCR::performance(pred, "sens", "spec")
    specificity <- perf@x.values[[1]]
    sensitivity <- perf@y.values[[1]]
    SS <- data.frame(specificity = specificity, sensitivity= sensitivity)
    palatte <- RColorBrewer::brewer.pal(5, 'Set1')
    g <- ggplot(SS, aes(x = specificity, y = sensitivity)) + geom_line(aes(color = algname), size = 1) + theme_bw() +
      scale_color_manual(values = palatte[1:2]) +
      labs(x = 'Specificity', y = 'Sensitivity',
           title = '') + ylim(0,1) +
      theme(plot.title = element_text(hjust = 0.5), legend.title = element_blank(),
            legend.justification = c(1, 0), legend.position = c(0.95, 0.05),
            legend.background = element_blank(), legend.key = element_blank(),
            legend.text = element_text(size = 10,face = 'bold'),
            axis.text = element_text(size = 12,face = 'bold'),
            axis.title = element_text(size = 15,face = 'bold'),
            axis.title.x = element_text(margin = margin(0.5,1,0,1,'cm')),
            axis.title.y = element_text(margin = margin(0.5,0.5,0,0.5,'cm')),
            plot.margin=unit(rep(1,4),'lines'),
            panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
            panel.background = element_blank(), axis.line = element_line(colour = "black"))
    return(g)
  }

  if (method == 'ROC') {
    p <- plot_ROC(pred,algname)
  }
  if (method == 'PR') {
    p <- plot_PR(pred,algname)
  }
  if (method == 'SSC') {
    p <- plot_SS(pred,algname)
  }
  return(p)
}
