"""Cosine scoring of gene pairs from a gene-embedding dictionary.

Shared by ``modules.trainer.cos_similarity`` (which needs the ``label`` column
for metrics and for feature building) and ``modules.predictor.cos_similarity``
(which needs only the two gene columns).  Callers keep their own file layout
and their own column selection; only the scoring is shared here.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, paired_cosine_distances


def _to_numpy(gene_embeddings):
    return {k: np.array(v) for k, v in gene_embeddings.items()}


def read_pair_table(path):
    """Read a tab-separated, headerless gene-pair table as-is.

    Nothing is renamed and no case is folded here: several shipped split files
    use lowercase gene symbols, and callers pass the original spelling through
    to their outputs.
    """
    return pd.read_csv(path, header=None, sep='\t')


def pair_scores(gene_embeddings, gene1, gene2):
    """Cosine similarity per row, rounded to 5 decimals.

    Returns an unfiltered float64 array with one entry per input row; rows
    whose genes are missing from ``gene_embeddings`` score 0.0.  Gene symbols
    are upper-cased for lookup only.

    The rounding is done *after* the subtraction, in float64.  Embeddings that
    arrive as float32 (scGPT, scFoundation) would otherwise round inside
    float32, where five-decimal values are not representable.
    """
    emb_dict = _to_numpy(gene_embeddings)
    g1 = pd.Series(gene1).astype(str).str.upper()
    g2 = pd.Series(gene2).astype(str).str.upper()
    mask = g1.isin(emb_dict).to_numpy() & g2.isin(emb_dict).to_numpy()

    scores = np.zeros(len(g1), dtype=np.float64)
    if mask.any():
        vec1 = np.array([emb_dict[g] for g in g1[mask]])
        vec2 = np.array([emb_dict[g] for g in g2[mask]])
        sims = (1 - paired_cosine_distances(vec1, vec2)).astype(np.float64)
        scores[mask] = np.abs(np.round(sims, 5))
    return scores


def all_pair_frame(gene_embeddings):
    """Every gene pair of the embedding dictionary, by cosine similarity."""
    emb_dict = _to_numpy(gene_embeddings)
    keys = list(emb_dict.keys())
    feature_matrix = np.array(list(emb_dict.values()))
    cos_mat = cosine_similarity(feature_matrix).round(5)
    df_cos_mat = pd.DataFrame(cos_mat, columns=keys, index=keys)

    rows, cols = np.triu_indices(len(keys), k=1)
    return pd.DataFrame({
        'gene_1': df_cos_mat.index[rows],
        'gene_2': df_cos_mat.columns[cols],
        'label': df_cos_mat.values[rows, cols],
    })
