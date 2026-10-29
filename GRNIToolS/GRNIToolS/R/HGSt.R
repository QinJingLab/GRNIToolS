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
  # Build the gene x TF gold-standard matrix in one vectorised pass. Row names
  # that do not appear in `gene` / `TF` are dropped with an explicit warning
  # rather than assigned by position, which would fail on NA subscripts.
  HGS.r <- matrix(0, nrow = length(gene), ncol = length(TF),
                  dimnames = list(gene, TF))
  # HGS columns 1-3: TF / target gene / weight (0/1 or -1/1)
  r <- match(as.character(HGS[, 2]), gene)   # rows: target genes
  c <- match(as.character(HGS[, 1]), TF)     # cols: TFs
  bad <- is.na(r) | is.na(c)
  if (any(bad)) {
    warning(sprintf("HGSt: %d of %d rows reference names absent from `gene`/`TF` and were ignored.",
                    sum(bad), length(bad)), call. = FALSE)
  }
  keep <- !bad
  if (any(keep)) {
    HGS.r[cbind(r[keep], c[keep])] <- as.numeric(HGS[keep, 3])
  }
  return(HGS.r)
}
