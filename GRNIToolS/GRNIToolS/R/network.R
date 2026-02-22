#' Visualization of network
#' @description Show the gene regulator network which you provided. Visualization of gene regulatory network
#' @param grn A matrix which is the weighted adjacency matrix of the inferred network by this algorithm or the result from the function of out_grn.
#' @param num The number of gene regulator imformation. Default: NULL.
#' @param file A connection, or a character string naming the file to write to. The default not save. Default: NULL.
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @details A labelled undirected network is plotted.
#' @return A D3 JavaScript force directed network graph.
#' @export
#'
#' @examples
#' B <- matrix(rnorm(1000,0,1),100)
#' TF <- paste0('G',1:20)
#' gene <- paste0('G',1:100)
#' grn <- demo_other(B,'parmigene_ARACNE.a',TF,gene)$grn
#' net <- network(grn,TF=TF,gene=gene)
network <- function(grn,num = NULL,TF,gene,file=NULL){
  if (ncol(grn) > 3) {
    diag(grn) <- 0
    mtf <- colnames(grn)
    mgene <- rownames(grn)
    id <- which(grn != 0,arr.ind = T)
    dat <- data.frame(TF = mtf[id[,2]],Gene = mgene[id[,1]],Weight = grn[id])
    dat <- dat[order(dat[,3],decreasing = T),]
    grn2 <- dat
    
    if (is.null(num)) {
      num <- nrow(grn2)
    }
    if (num > nrow(grn2)) {
      num <- nrow(grn2)
    }
    grn2 <- grn2[order(abs(grn2[,3]),decreasing = T),]
    grn2 <- grn2[1:num,]
    grn <- grn2
    colnames(grn) <- c('TF','Gene','Weight')
  }
  
  grn <- grn[order(abs(grn[,3]),decreasing = T),]
  if (is.null(num)) {
    num <- nrow(grn)
  }
  if (nrow(grn) > num) {
    grn <- grn[1:num,]
  }
  #grn <- as.matrix(grn)
  grn[,3] <- abs(grn[,3]) * 2
  grn[,1] <- match(grn[,1],TF)
  grn[,2] <- match(grn[,2],gene)
  grn[,1:2] <- grn[,1:2]-1
  
  name <- unique(c(TF,gene))
  TFid <- which(TF %in% name)
  geneid <- c(1:length(name))[-TFid]
  group <- c(1:length(name))
  group[TFid] <- 'TF'
  group[geneid] <- 'Gene'
  size <- c(1:length(name))
  size[TFid] <- 30
  size[geneid] <- 10
  
  datNodes <- data.frame(name = unique(c(TF,gene)),group = group,size = size)
  Colors <- 'd3.scaleOrdinal().range([ "#32CD32","#FFC0CB", "#8A2BE2", "#FF6347"])'
  net <- networkD3::forceNetwork(Links = grn,Nodes = datNodes,Source = "TF",Target = "Gene",Value = "Weight",
                                 NodeID = "name",Group = "group",Nodesize = "size" ,
                                 fontFamily="Times New Roman",fontSize = 20,linkColour="blue",
                                 charge = -100,opacity = 0.9,legend=T,arrows=T,
                                 colourScale = Colors,
                                 bounded=F,opacityNoHover=1.0,zoom = T)
  if (! is.null(file)) {
    networkD3::saveNetwork(net, file = file, selfcontained = TRUE)
  }
  return(net)
}

