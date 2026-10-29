# Internal helper: reshape an algorithm's network matrix into a
# (gene x TF) score matrix (internal, not exported).
#
# Correlation, mutual information, TOM and partial-correlation methods all
# return a gene x gene adjacency matrix indexed by position in the gene list.
# GRNIToolS needs it trimmed and reordered to length(gene) x length(TF).
#
# Matching rules:
#   * rows (target genes): matched by name when the matrix has row names;
#     otherwise the row order is assumed to be the gene order
#   * cols (regulators): matched by name when the matrix has column names;
#     otherwise TF positions are taken within `gene` (ennet and friends clear
#     dimnames, and their column order is still the gene order)
#   * positions that match nothing stay 0, i.e. no edge predicted for that TF
#
# Note: not exported. Called by the 16 algorithm functions in this package;
# the get_score_matrix(gene, TF, grnmat) signature is kept stable.

get_score_matrix <- function(gene, TF, grnmat) {
  grnmat <- as.matrix(grnmat)
  Score  <- matrix(0, length(gene), length(TF), dimnames = list(gene, TF))

  # rows: target genes
  ridx <- if (is.null(rownames(grnmat))) seq_along(gene) else match(gene, rownames(grnmat))
  # cols: regulators (TFs)
  cidx <- if (is.null(colnames(grnmat))) match(TF, gene)  else match(TF, colnames(grnmat))

  okr <- !is.na(ridx)
  okc <- !is.na(cidx)
  if (any(okr) && any(okc)) {
    Score[okr, okc] <- grnmat[ridx[okr], cidx[okc], drop = FALSE]
  }
  Score
}
