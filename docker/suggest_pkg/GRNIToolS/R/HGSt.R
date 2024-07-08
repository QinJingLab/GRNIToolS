#' Transform the high-throughput gold standard data
#'
#' @description The high-throughput gold standard data is represented in matrix
#' @param TF The Transcription Factors (TFs) The default value NULL means that all the genes are used as candidate regulators. Default: NULL.
#' @param gene A vector of characters containing the target genes.
#' @param HGS The class of HGS is 'data.frame'. The golden standard gene regulator network. The first column is TFs list and the second one is target genes list. 
#' The value of third column should be 1 or 0.
#'
#' @details The three columns of high-throughput gold standard data are transformed into matrix to evaluate the inferred gene regulatory network.
#'
#' @return A matrix which is the high-throughput golden standard. Rows contain genes and columns contain TFs ,the value between in one or zero.
#' @export
#' @examples
#' data(datalist)
#' HGSmat <- HGSt(datalist$TF,datalist$gene,datalist$HGS)
HGSt <- function(TF,gene,HGS){
  # HGS
  HGS.r <- matrix(nrow = length(gene), ncol = length(TF))
  colnames(HGS.r) <- TF
  rownames(HGS.r) <- gene
  for(i in 1:nrow(HGS)){
    a <- HGS[i, 1]
    b <- HGS[i, 2]
    c <- match(b, gene)
    d <- match(a, TF)
    HGS.r[c, d] <- HGS[i, 3]
  }
  HGS.r[is.na(HGS.r)] <- 0
  HGSmat <- HGS.r
  return(HGSmat)
}
