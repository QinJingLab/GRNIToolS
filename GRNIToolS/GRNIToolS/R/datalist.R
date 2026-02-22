#' @title Datalist of mESC from Bulk transcriptome
#' @description This datalist contains six dataset that authors used to calculate. The raw data 'A', 'B' and 'InitialX' are used to calculate via algorithms. The 'TFlist' and 'namelist' are the name list of Transcription Factors (TFs) name and Target Genes name, respectively. And the 'HGS', High-throughput Golden Standard, is data help authors evaluate the results.
#' @usage data(datalist)
#' @format The datalist was constructed as 'list'
#' \describe{
#' \item{\code{A}}{The class of A is 'data.frame'.}
#' \item{\code{B}}{The class of B is 'data.frame'.}
#' \item{\code{InitialX/X}}{The class of InitialX is 'data.frame'.}
#' \item{\code{HGS}}{The class of HGS is 'data.frame'. The golden standard mESC network was constructed from high-throughput ChIP-X and transcriptome data under TF perturbation (high-throughput golden standard). The first column is TFs list and the second one is Target Genes list.}
#' \item{\code{TFlist/TF}}{The class of TFlist is 'character'. The Transcription Factors (TFs) list}
#' \item{\code{namelist/gene}}{The class of namelist is 'character'. The Target Genes list}
#' }
#' @details datalist
#' @references Qin, J., Hu, Y., Xu, F., Yalamanchili, H. K., and Wang, J. (2014). "Inferring gene regulatory networks by integrating ChIP-seq/chip and transcriptome data via LASSO-type regularization methods", Methods, 67(3), 294-303.
#'
#' Qin, J., Li, M., Wang, P., Zhang, M., Wang J. (2011). "ChIP-Array: combinatory analysis of ChIP-seq/chip and microarray gene expression data to discover direct/indirect targets of a transcription factor", 39, W430-W436.
#'
#' Wang, P., Qin, J., Qin, Y., Zhu, Y., Wang, L., Li, M., Zhang, M., Wang, J. (2015). "ChIP-Array 2: Integrating multiple omics data to construct gene regulatory networks", Nucleic Acids Research, 43, W264-W269.
#' @examples
#' data(datalist)
#' @keywords datasets
datalist <- function(){
  utils::data(datalist)
}

