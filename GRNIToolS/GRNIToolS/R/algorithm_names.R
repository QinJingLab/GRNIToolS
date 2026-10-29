# Single source of truth for algorithm names (internal, not exported).
#
# Algorithm names are agreed-upon strings matched exactly by the dispatch
# branches in demo_reg() and demo_other(). Keeping the lists in one place means
# "which names are valid" and "which belong to the regression group" are
# maintained here only; the dispatch branches themselves remain, since each
# calls a different implementation.

# Regression / sparse-solver algorithms (implemented by demo_reg())
.grn_regression_algorithms <- c(
  "ADMM", "ADMMHalf", "ADMMHard",
  "ITAHalf", "ITAHard", "ISTA",
  "SPGL0", "SPGL1", "CoSaMP", "OMP", "FoBa", "LARS"
)

# Correlation / mutual information / machine learning / meta (demo_other())
.grn_other_algorithms <- c(
  "Pearson", "Spearman", "Kendall", "WGCNA", "BC3NET", "C3NET",
  "parmigene_ARACNE.a", "parmigene_ARACNE.m", "parmigene_CLR", "parmigene_MRNET",
  "minet_ARACNE", "minet_CLR", "minet_MRNET", "minet_MRNETB",
  "GeneNet", "GENIE3.ET", "GENIE3.RF", "ENNET", "PCIT"
)

# All valid algorithm names
.grn_all_algorithms <- c(.grn_regression_algorithms, .grn_other_algorithms)

# Validate a single algorithm name, reporting the available ones on failure.
.grn_check_algorithm <- function(algorithm, allowed, caller) {
  if (length(algorithm) != 1L || is.na(algorithm)) {
    stop(sprintf("`algorithm` in %s() must be a single algorithm name, got length %d.",
                 caller, length(algorithm)), call. = FALSE)
  }
  if (!algorithm %in% allowed) {
    hint <- ""
    if (algorithm %in% .grn_all_algorithms) {
      hint <- sprintf("\n  '%s' is a %s-type algorithm; call %s() instead.",
                      algorithm,
                      if (algorithm %in% .grn_regression_algorithms) "regression" else "other",
                      if (algorithm %in% .grn_regression_algorithms) "demo_reg" else "demo_other")
    }
    stop(sprintf("Unknown algorithm '%s' in %s().%s\n  Available (%d): %s",
                 algorithm, caller, hint, length(allowed), paste(allowed, collapse = ", ")),
         call. = FALSE)
  }
  invisible(algorithm)
}
