# Internal helper: write a gene x TF score matrix to a self-describing TSV
# (internal, not exported).
#
# The output has `Gene` as its first column, one column per TF, and a header
# row, so the labels survive the round trip and the file can be used directly
# in Cytoscape. readr::write_delim() writes neither row names nor, with
# col_names = FALSE, any header.

.grn_write_network <- function(x, file, delim = "\t") {
  df  <- as.data.frame(x)
  out <- data.frame(Gene = rownames(df), df,
                    check.names = FALSE, stringsAsFactors = FALSE)
  readr::write_delim(out, file = file, delim = delim, col_names = TRUE)
}
