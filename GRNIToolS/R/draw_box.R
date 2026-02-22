#' draw_box
#' @description Show the plot of AUROC or AUPR by different algorithms of bootstrap.
#' @param dat The data.frame with the name of algorithm and the value of AUROC or AUPR.
#' @param name AUROC or AUPR
#'
#' @return The plot of AUROC or AUPR by different algorithm of bootstrap
#' @export
#' @import RColorBrewer ggplot2
#'
#' @examples
#' set.seed(123)
#' A <- matrix(rnorm(200,0,1),20,10)
#' B <- matrix(rnorm(1000,0,1),100)
#' X <- matrix(0,100,20)
#' s <- c(1:10)
#' TF <- paste0('G',1:20)
#' gene <- paste0('G',1:100)
#' HGSmat <- matrix(0,100,20)
#' colnames(HGSmat) <- TF
#' rownames(HGSmat) <- gene
#' HGSmat[sample(2000,1000)] <- 1
#' algorithm <- c('Pearson','parmigene_ARACNE.a','PCIT','minet_MRNETB','OMP')
#' conlist <- grn_main(A=A,B=B,X=X,seq = s,gene=gene,TF=TF,algorithm = algorithm,
#'                     HGSmat = HGSmat,file=FALSE,verbose=TRUE,sym=FALSE)
#' dat <- data.frame(Algorithm = algorithm,AUROC = conlist$AUROC)
#' p <- draw_box(dat,'AUROC')
#' print(p)
draw_box <- function(dat,name = 'AUROC'){
  colnames(dat) <- c('alg','auc')
  color = RColorBrewer::brewer.pal.info[RColorBrewer::brewer.pal.info$category == 'qual',]
  color = unlist(mapply(RColorBrewer::brewer.pal, color$maxcolors, rownames(color))) 
  color2 <- color[1:length(unique(dat$alg))]
  dat[,1] <- factor(dat[,1],levels = unique(dat[,1]))
  alg <- dat[,1]
  auc <- dat[,2]
  p <- ggplot(dat, aes(x = alg, y = auc, fill = alg))
  p2 <- p + theme_light() + geom_boxplot(width = 0.5,alpha=0.8,size=0.3,outlier.size = 0.5)  +
    labs(title = paste0("Plot of ",name," by different algorithm of bootstrap"), x = '', y = name,fill='Algorithm')+
    scale_y_continuous(limits = c(0.3,1)) +
    theme(
      plot.title = element_text(hjust = 0.5,colour="black",face = 'bold',size = 15),
      axis.text.x=element_text(angle = 45,hjust = 0.5,vjust = 0.3, colour="black",size =13,face = 'bold'),
      axis.text.y=element_text(hjust = 1,vjust = 0.3, colour="black",size =13,face = 'bold',margin = margin(0.5,0.5,0,0.5,'cm')),
      legend.title = element_text(colour="black",face = 'bold',size = 13),
      legend.text = element_text(colour="black",face = 'bold',size = 13),
      axis.title.x = element_text(margin = margin(0.5,1,0,1,'cm')),
      axis.title.y = element_blank(),
      panel.border=element_rect(colour = "black")) +
    scale_fill_manual(values = unique(color2))
  p2 <- p2  + geom_hline(aes(yintercept = 0.5),colour='red',linetype = 'dashed')
  return(p2)
}

