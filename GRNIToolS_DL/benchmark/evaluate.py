
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PREDICTION_COLUMNS = ('tf', 'gene', 'score')

# The shared split files under a dataset's GRNDL directory; three datasets renamed the test split.
SPLIT_CANDIDATES = {'valid': ('val.txt',), 'test': ('random_test.txt', 'test.txt'),
                    'tftest': ('TF_test.txt',)}


PREDICTION_CHUNK_ROWS = 1_000_000


# Unit separator, not NUL: pandas' string concat silently drops a NUL byte, which made the
# key the bare concatenation and let (ARNTL, CAT) collide with (ARNT, LCAT).
KEY_SEP = '\x1f'


def pair_key(frame):

    return (frame['tf'].str.upper().str.strip() + KEY_SEP
            + frame['gene'].str.upper().str.strip())


def read_predictions(predictions, wanted=None, chunksize=PREDICTION_CHUNK_ROWS):

    frames, columns = [], None
    for chunk in pd.read_csv(predictions, sep='\t', dtype=str, chunksize=chunksize):
        if columns is None:
            columns = _check_columns(list(chunk.columns), predictions)
        if wanted is not None:
            chunk = chunk[pair_key(chunk).isin(wanted)]
            if chunk.empty:
                continue
        chunk['score'] = pd.to_numeric(chunk['score'], errors='raise')
        frames.append(chunk)
    if columns is None:
        # An empty file yields no chunks at all. Read the header on its own so a file that
        # is header-only, or has no header, is treated exactly as the unchunked read did.
        columns = _check_columns(
            list(pd.read_csv(predictions, sep='\t', dtype=str, nrows=0).columns),
            predictions)
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=str) for c in columns})
    return pd.concat(frames, ignore_index=True)


def _check_columns(columns, predictions):
    missing = [c for c in PREDICTION_COLUMNS if c not in columns]
    if missing:
        raise ValueError(f'{predictions}: expected header tf/gene/score, found '
                         f'{list(columns)} (missing {missing})')
    return columns


def evaluate(predictions, split):

    # The split layout is the shared one: no header, gene names, columns tf/gene/label.
    lab = pd.read_csv(split, sep='\t', header=None, dtype=str)
    if lab.shape[1] < 3:
        raise ValueError(f'{split}: expected 3 tab-separated columns (tf, gene, label), '
                         f'found {lab.shape[1]} on a {lab.shape[0]}-row file')
    lab = lab.iloc[:, :3]
    lab.columns = ['tf', 'gene', 'label']
    lab['label'] = pd.to_numeric(lab['label'], errors='raise').astype(int)
    lab['key'] = pair_key(lab)
    pred = read_predictions(predictions, wanted=set(lab['key']))
    pred['key'] = pair_key(pred)

    # A pair scored twice is ambiguous. Keep the method's strongest statement about it.
    pred = pred.groupby('key', as_index=False).agg(score=('score', 'max'))

    n_rows = len(lab)
    conflict = lab.groupby('key')['label'].nunique()
    if (conflict > 1).any():
        bad = conflict[conflict > 1].index[:5].tolist()
        shown = [b.replace(KEY_SEP, '/') for b in bad]
        raise ValueError(f'{split}: {int((conflict > 1).sum())} pair(s) carry both labels, '
                         f'e.g. {shown}')
    lab_listed, lab = lab, lab.drop_duplicates('key')
    dup_labels = n_rows - len(lab)

    def pair_up(labels):
        """Left-join the scores onto `labels` and return the scored (label, score) arrays."""
        merged = labels.merge(pred[['key', 'score']], on='key', how='left')
        scored = merged['score'].notna()
        return merged.loc[scored, 'label'].values, merged.loc[scored, 'score'].values

    def metrics(y, s):
        """AUROC/AUPRC, or NaN with a reason when only one class is present."""
        if len(np.unique(y)) < 2:
            n_pos = int((y == 1).sum())
            return (float('nan'), float('nan'),
                    f'degenerate: {n_pos} positive, {len(y) - n_pos} negative')
        return (float(roc_auc_score(y, s)), float(average_precision_score(y, s)), '')

    y, s = pair_up(lab)
    out = {'n_total': int(len(lab)), 'n_rows': n_rows, 'dup_labels': dup_labels,
           'coverage': float(y.size / len(lab)) if len(lab) else float('nan')}
    auroc, auprc, reason = metrics(y, s)
    out.update(auroc=auroc, auprc=auprc, reason=reason)

    if dup_labels:
        yr, sr = pair_up(lab_listed)
        r_auroc, r_auprc, _ = metrics(yr, sr)
        out.update(n_raw=int(yr.size), auroc_raw=r_auroc, auprc_raw=r_auprc)
    else:
        out.update(n_raw=int(y.size), auroc_raw=auroc, auprc_raw=auprc)
    return out


def report(cell_dir, split_dir):
    """One line per split, for the dumps a method left in `cell_dir`."""
    lines = []
    for split, names in SPLIT_CANDIDATES.items():
        dump = os.path.join(cell_dir, f'predictions_{split}.tsv')
        truth = next((os.path.join(split_dir, n) for n in names
                      if os.path.exists(os.path.join(split_dir, n))), None)
        if truth is None or not os.path.exists(dump):
            continue
        r = evaluate(dump, truth)
        lines.append(f'{split:<9}AUROC={r["auroc"]:.5f}  AUPRC={r["auprc"]:.5f}'
                     + (f'  [{r["reason"]}]' if r['reason'] else ''))
    return lines


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(f'usage: {sys.argv[0]} <predictions.tsv> <split.tsv>\n'
                 f'       {sys.argv[0]} <cell_dir> <split_dir>')
    if os.path.isdir(sys.argv[1]) and os.path.isdir(sys.argv[2]):
        lines = report(sys.argv[1], sys.argv[2])
        if not lines:
            sys.exit(f'{sys.argv[1]}: no predictions_<split>.tsv for {sys.argv[2]}')
        print('\n'.join(lines))
    else:
        r = evaluate(sys.argv[1], sys.argv[2])
        print(f'AUROC={r["auroc"]:.5f}  AUPRC={r["auprc"]:.5f}'
              + (f'  [{r["reason"]}]' if r['reason'] else ''))
