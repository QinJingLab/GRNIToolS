"""The one evaluation every method in GRNIToolS_DL is scored by.

Methods emit `predictions_<split>.tsv`; `evaluate.evaluate()` turns that file plus a
split file into AUROC and AUPRC. No method computes the numbers it is compared on.

See GRNIToolS/EVALUATION.md for the file format and the comparison with the R package
and GRNIToolS_scFM.
"""
