import argparse
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# R's calpl() calls set.seed(123) right before drawing the balanced negatives.
DEFAULT_SEED = 123
# The edge lists are ~18M rows, so they are folded into the matrix chunk by chunk.
CHUNK_ROWS = 2_000_000


def get_parser():
    parser = argparse.ArgumentParser(
        description='Score an inferred GRN with the GRNIToolS R package protocol')
    parser.add_argument('--hgs', type=str, required=True,
                        help='gold standard: three columns (TF, gene, weight), no header')
    parser.add_argument('--grn', type=str, required=True,
                        help='inferred network: first three columns TF, gene, score')
    parser.add_argument('--tf-list', type=str, required=True, dest='tf_list',
                        help='the TF names, one per line (R: TFlist.tsv)')
    parser.add_argument('--gene-list', type=str, required=True, dest='gene_list',
                        help='the gene names, one per line (R: genelist.tsv)')
    parser.add_argument('--sym', type=str, choices=('balanced', 'unbalanced'), default='balanced',
                        help="R's calpl(sym=): 'balanced' (default) draws as many negatives as "
                             "there are positives, 'unbalanced' keeps every scored cell")
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'seed of the balanced draw (default {DEFAULT_SEED}, the seed R uses)')
    parser.add_argument('--output', type=str, required=True, help='Path to save the results')
    parser.add_argument('--method', type=str, required=False, default='DL',
                        help='Method you used for GRN inferred')
    parser.add_argument('--dataset', type=str, required=False, default='test',
                        help='Dataset you used for GRN inferred')
    return parser.parse_args()


def read_lines(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def sniff_sep(path):
    with open(path, errors='replace') as f:
        head = f.read(65536)
    return '\t' if head.count('\t') >= head.count(',') else ','


def edge_matrix(path, genes, tfs, header=0, chunksize=CHUNK_ROWS):
    gi = {name.upper(): i for i, name in enumerate(genes)}
    ti = {name.upper(): j for j, name in enumerate(tfs)}
    mat = np.zeros((len(genes), len(tfs)))
    stats = {'rows': 0, 'unknown_gene': 0, 'unknown_tf': 0, 'non_numeric': 0}
    reader = pd.read_csv(path, sep=sniff_sep(path), header=header,
                         names=['TF', 'Gene', 'Weight'], usecols=[0, 1, 2], dtype=str,
                         compression='infer', chunksize=chunksize, skip_blank_lines=True)
    for chunk in reader:
        stats['rows'] += len(chunk)
        row = chunk['Gene'].str.strip().str.upper().map(gi)
        col = chunk['TF'].str.strip().str.upper().map(ti)
        weight = pd.to_numeric(chunk['Weight'], errors='coerce').to_numpy(dtype=float)
        bad_gene = row.isna().to_numpy()
        bad_tf = col.isna().to_numpy()
        bad_weight = ~np.isfinite(weight)
        stats['unknown_gene'] += int(bad_gene.sum())
        stats['unknown_tf'] += int(bad_tf.sum())
        stats['non_numeric'] += int(bad_weight.sum())
        keep = ~(bad_gene | bad_tf | bad_weight)
        if keep.any():
            mat[row.to_numpy()[keep].astype(int), col.to_numpy()[keep].astype(int)] = weight[keep]
    return mat, stats


def calpl(hgs_mat, grn_mat, sym=True, seed=DEFAULT_SEED):
    grn = np.abs(np.asarray(grn_mat, dtype=float))
    hgs = np.abs(np.asarray(hgs_mat, dtype=float))
    peak = grn.max() if grn.size else 0.0
    if peak > 0:
        grn = grn / peak          # R divides by the global max; no ranking notices
    if not sym:
        rows, cols = np.nonzero(grn)
        return grn[rows, cols], hgs[rows, cols]
    pos_rows, pos_cols = np.nonzero((hgs == 1) & (grn != 0))
    n = pos_rows.shape[0]
    if n == 0:
        return None
    neg_rows, neg_cols = np.nonzero((hgs == 0) & (grn != 0))
    n = min(n, neg_rows.shape[0])
    rng = np.random.default_rng(seed)
    pick = rng.choice(neg_rows.shape[0], size=n, replace=False)     # R draws negatives first
    neg_rows, neg_cols = neg_rows[pick], neg_cols[pick]
    pick = rng.choice(pos_rows.shape[0], size=n, replace=False)
    pos_rows, pos_cols = pos_rows[pick], pos_cols[pick]
    return (np.concatenate([grn[pos_rows, pos_cols], grn[neg_rows, neg_cols]]),
            np.concatenate([np.ones(n), np.zeros(n)]))


def roc_points(predictions, labels):
    pred = np.asarray(predictions, dtype=float)
    lab = np.asarray(labels, dtype=float)
    finite = np.isfinite(pred)
    pred, lab = pred[finite], lab[finite]
    order = np.argsort(-pred, kind='stable')    # R's order(decreasing=TRUE) is stable too
    pred, lab = pred[order], lab[order]
    tp = np.cumsum(lab == 1)
    fp = np.cumsum(lab == 0)
    # R keeps the last row of each run of equal scores, so ties inside a run cannot matter.
    last = np.empty(pred.shape, dtype=bool)
    last[:-1] = pred[:-1] != pred[1:]
    last[-1] = True
    return np.concatenate([[0.0], fp[last]]), np.concatenate([[0.0], tp[last]])


def pr_points(fp, tp, n_pos):
    bounds = np.flatnonzero(np.concatenate([[True], tp[1:] != tp[:-1]]))
    tp = tp[bounds]
    fp = np.minimum.reduceat(fp, bounds)
    with np.errstate(invalid='ignore'):     # the first point is 0/0, set to 1 on the next line
        precision = tp / (fp + tp)
    recall = tp / n_pos
    if fp[0] == 0 and tp[0] == 0:
        precision[0] = 1.0
    finite = np.isfinite(precision) & np.isfinite(recall)
    return recall[finite], precision[finite], tp, fp


def rocr_auc(fp, tp, n_pos, n_neg):
    x = fp / n_neg
    y = tp / n_pos
    return np.sum(0.5 * (x[1:] - x[:-1]) * (y[1:] + y[:-1]))


def rocr_aucpr(fp, tp, n_pos):
    recall, precision, tp, fp = pr_points(fp, tp, n_pos)
    if len(recall) < 2:
        raise ValueError('Not enough distinct predictions to compute area under the PR curve.')
    recall, precision = list(recall), list(precision)
    # A jump of more than two true positives gets the intermediate PR points R would insert.
    for i in range(len(recall) - 1):
        jump = tp[i + 1] - tp[i]
        if jump > 2:
            skew = (fp[i + 1] - fp[i]) / jump
            x = np.arange(1, jump + 1)
            recall[i + 1:i + 1] = ((x + tp[i]) / n_pos).tolist()
            precision[i + 1:i + 1] = ((x + tp[i]) / (tp[i] + fp[i] + x + skew * x)).tolist()
    recall, precision = np.asarray(recall), np.asarray(precision)
    return np.sum(0.5 * (recall[1:] - recall[:-1]) * (precision[1:] + precision[:-1]))


def cal_auc(pred, true_label, output_dir, method, dataset):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if pred is None or len(pred) == 0:
        print(f'[evaluate] {method}/{dataset}: nothing to score, writing NA')
        with open(f'{output_dir}/{dataset}_auc.txt', 'w') as f:
            f.write('method\tdataset\tauroc\taupr\n')
            f.write(f'{method}\t{dataset}\tNA\tNA\n')
        return
    pred = np.asarray(pred, dtype=float)
    true_label = np.asarray(true_label, dtype=float)
    n_pos, n_neg = int(np.sum(true_label == 1)), int(np.sum(true_label == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f'only one class present (n_pos={n_pos}, n_neg={n_neg})')

    fp, tp = roc_points(pred, true_label)
    roc_auc = float(rocr_auc(fp, tp, n_pos, n_neg).round(5))
    pr_auc = float(rocr_aucpr(fp, tp, n_pos).round(5))

    plt.figure()
    plt.plot(fp / n_neg, tp / n_pos, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/{dataset}_roc_curve.png', dpi=300, bbox_inches='tight')

    recall, precision, _, _ = pr_points(fp, tp, n_pos)
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="best")
    plt.savefig(f'{output_dir}/{dataset}_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    with open(f'{output_dir}/{dataset}_auc.txt', 'w') as f:
        f.write('method\tdataset\tauroc\taupr\n')
        f.write(f'{method}\t{dataset}\t{roc_auc}\t{pr_auc}\n')
    print(f'[evaluate] {method}/{dataset}: {n_pos} positives, {n_neg} negatives, '
          f'AUROC={roc_auc} AUPR={pr_auc}')


if __name__ == '__main__':
    args = get_parser()
    genes = read_lines(args.gene_list)
    tfs = read_lines(args.tf_list)
    # The gold standard has no header (`header=FALSE` on the R side); the edge list does.
    hgs_mat, hgs_stats = edge_matrix(args.hgs, genes, tfs, header=None)
    if hgs_stats['unknown_gene'] or hgs_stats['unknown_tf']:
        warnings.warn('gold standard: {} of {} rows name a gene and {} a TF outside the '
                      'gene/TF lists; those cells stay 0, as in HGSt()'
                      .format(hgs_stats['unknown_gene'], hgs_stats['rows'],
                              hgs_stats['unknown_tf']))
    grn_mat, grn_stats = edge_matrix(args.grn, genes, tfs)
    if grn_stats['unknown_tf']:
        print(f'[evaluate] {grn_stats["unknown_tf"]} of {grn_stats["rows"]} scored pairs name a '
              f'TF outside the TF list and are outside the evaluated grid')

    dat = calpl(hgs_mat, grn_mat, sym=(args.sym == 'balanced'), seed=args.seed)
    if dat is None:
        cal_auc(None, None, args.output, args.method, args.dataset)
    else:
        cal_auc(dat[0], dat[1], args.output, args.method, args.dataset)
