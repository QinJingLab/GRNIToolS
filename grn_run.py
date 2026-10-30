#!/usr/bin/env python3
"""One entry point for the three packages; it dispatches and never re-scores."""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DL_DIR = os.path.join(HERE, 'GRNIToolS_DL')
SCFM_DIR = os.path.join(HERE, 'GRNIToolS_scFM')
R_DATA_DIR = os.path.join(HERE, 'GRNIToolS', 'GRNIToolS_data')

# The R package's two dispatch groups (R/algorithm_names.R); regression ones also take A and X.
R_REGRESSION = ['ADMM', 'ADMMHalf', 'ADMMHard', 'ITAHalf', 'ITAHard', 'ISTA',
                'SPGL0', 'SPGL1', 'CoSaMP', 'OMP', 'FoBa', 'LARS']
R_OTHER = ['Pearson', 'Spearman', 'Kendall', 'WGCNA', 'BC3NET', 'C3NET',
           'parmigene_ARACNE.a', 'parmigene_ARACNE.m', 'parmigene_CLR', 'parmigene_MRNET',
           'minet_ARACNE', 'minet_CLR', 'minet_MRNET', 'minet_MRNETB',
           'GeneNet', 'GENIE3.ET', 'GENIE3.RF', 'ENNET', 'PCIT']
R_METHODS = R_REGRESSION + R_OTHER

DL_METHODS = ['DeepSEM', 'DeepDRIM', 'GRNCNN', 'GRNResNet', 'STGRNS', 'GRNTrans',
              'GENELINK', 'GRNGCN', 'IGEGRNS', 'DeepRIG']

SCFM_METHODS = ['cos', 'logit', 'rf', 'xgb', 'MLP', 'ResNet', 'CNN',
                'Transformer', 'GCN']

# Canonical dataset name -> each package's own spelling; None means that package has no such data.
DATASETS = {
    'Ecoli':         {'dl': 'Ecoli',         'r': 'E. coli',          'scfm': None},
    'S.aureus':      {'dl': 'S.aureus',      'r': 'S.aureus',         'scfm': None},
    'S.cerevisiae':  {'dl': 'S.cerevisiae',  'r': 'S.cerevisiae',     'scfm': None},
    'SynTReN':       {'dl': 'SynTReN',       'r': 'SynTReN',          'scfm': None},
    'GeneNetWeaver': {'dl': 'GeneNetWeaver', 'r': 'GeneNetWeaver',    'scfm': None},
    'mESC-bulk':     {'dl': 'Bulk',          'r': 'mESC/mESC-bulk',   'scfm': 'mESC/bulk'},
    'mESC-sc-TS':    {'dl': 'SC-TS',         'r': 'mESC/mESC-sc-TS',  'scfm': 'mESC/SC-TS'},
    'mESC-sc-PC':    {'dl': 'SC-PC',         'r': 'mESC/mESC-sc-PC',  'scfm': 'mESC/SC-PC'},
    'A549':          {'dl': 'A549',          'r': 'A549',             'scfm': 'human/A549'},
    'HepG2':         {'dl': 'HepG2',         'r': 'HepG2',            'scfm': 'human/HepG2'},
    'MCF7':          {'dl': 'MCF7',          'r': 'MCF7',             'scfm': 'human/MCF7'},
}

# The R driver: load this dataset's matrices, call grn_main with the package's own defaults.
R_CODE = r'''
a <- commandArgs(trailingOnly = TRUE)
d <- a[1]; alg <- a[2]; out <- a[3]; opts <- a[4]
suppressMessages(library(GRNIToolS))
first <- function(bases) {
  for (b in bases) for (e in c('.txt', '.tsv', '.csv')) {
    p <- paste0(b, e); if (file.exists(p)) return(p)
  }
  stop('none of these exist: ', paste(paste0(bases[1], c('.txt', '.tsv', '.csv')), collapse = ', '))
}
rd <- function(p) {
  f <- if (endsWith(p, '.csv')) read.csv else read.delim
  as.matrix(f(p, header = FALSE))
}
A <- rd(first(file.path(d, 'A')))
B <- rd(first(file.path(d, 'B')))
readvec <- function(p) {
  x <- unlist(strsplit(trimws(readLines(p)), '\t', fixed = TRUE))
  x[nzchar(x)]
}
TF <- readvec(first(file.path(d, c('TFlist', 'TF'))))
gene <- readvec(first(file.path(d, c('genelist', 'gene'))))
HGS <- read.delim(first(file.path(d, 'HGS')), header = FALSE)
HGSmat <- HGSt(TF, gene, HGS)
xp <- file.path(d, 'InitialX.txt')
X <- if (file.exists(xp)) as.matrix(read.delim(xp, header = FALSE)) else matrix(0, nrow(B), nrow(A))
cat(sprintf('A %dx%d  B %dx%d  TF %d  gene %d  HGS nnz %d\n',
            nrow(A), ncol(A), nrow(B), ncol(B), length(TF), length(gene), sum(HGSmat != 0)))
if (sum(HGSmat != 0) == 0) stop('gold standard is empty: TF/gene names in ', d, ' do not match HGS')
# opts carries "name=value;name=value" overrides; empty means the package's own defaults.
extra <- list()
if (nzchar(opts)) for (piece in strsplit(opts, ';')[[1]]) { kv <- strsplit(piece, '=')[[1]]
  extra[[kv[1]]] <- type.convert(kv[2], as.is = TRUE) }
sp <- if (nrow(A) > 1) seq_len(min(10L, nrow(A) - 1L)) else stop('need at least two TFs')
cat(sprintf('sparsity 1:%d  (TF %d)\n', length(sp), nrow(A)))
res <- do.call(grn_main, c(list(A = A, B = B, seq = sp, X = X, TF = TF,
                                gene = gene, algorithm = alg, HGSmat = HGSmat,
                                file = FALSE, verbose = TRUE), extra))
write.table(res, out, sep = '\t', quote = FALSE, row.names = FALSE)
print(res)
'''


def backend_of(method):
    if method in DL_METHODS:
        return 'dl'
    if method in SCFM_METHODS:
        return 'scfm'
    if method in R_METHODS:
        return 'r'
    return None


def pick_file(directory, base, exts=('.txt', '.tsv', '.csv')):
    for e in exts:
        p = os.path.join(directory, base + e)
        if os.path.exists(p):
            return p
    return None


def dataset_dir(dataset, backend):
    ds = DATASETS[dataset][backend]
    if ds is None:
        sys.exit(f'{dataset} has no data for the {backend} backend; run --list for coverage')
    return ds


def build_dl(method, dataset, out):
    ds = dataset_dir(dataset, 'dl')
    g = os.path.join(DL_DIR, 'Input_data', ds, 'GRNDL')
    expr = os.path.join(g, 'expression.csv')
    if not os.path.exists(expr):
        sys.exit(f'no DL data for {dataset}: {expr} is missing')
    # main.py derives train/valid/test/tftest, --tf, --gene and --input_dir from --expr's directory.
    cmd = [sys.executable, 'main.py', '--method', method, '--expr', expr,
           '--output_dir', out]
    return cmd, DL_DIR


def build_scfm(method, dataset, out, pt):
    ds = dataset_dir(dataset, 'scfm')
    d = os.path.join(SCFM_DIR, 'input_data', ds)
    if not os.path.isdir(d):
        sys.exit(f'no scFM data for {dataset}: {d} is missing')
    expr = next((os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.endswith('expression.csv')), None)
    if expr is None:
        sys.exit(f'no expression matrix in {d}')
    # scFM runs with cwd=SCFM_DIR, so every path goes over relative to that directory
    rel = lambda p: os.path.relpath(p, SCFM_DIR)
    pt = rel(pt if os.path.isabs(pt) else os.path.join(HERE, pt))
    # scFM names the validation split valid.txt on the human sets and val.txt on the mESC ones.
    def split(name, alt=None):
        p = pick_file(d, name)
        if p is None and alt:
            p = pick_file(d, alt)
        if p is None:
            sys.exit(f'no {name} split in {d}')
        return rel(p)
    cmd = [sys.executable, 'main.py', 'train', '--method', method,
           '-expr', rel(expr),
           '-train', split('train'), '-valid', split('valid', 'val'),
           '-test', split('test'), '-tftest', split('TF_test'),
           '--pt', pt, '-o', out]
    return cmd, SCFM_DIR


def build_r(method, dataset, out, opts):
    ds = dataset_dir(dataset, 'r')
    d = os.path.join(R_DATA_DIR, ds)
    if not os.path.isdir(d):
        sys.exit(f'no R data for {dataset}: {d} is missing')
    rscript = shutil.which('Rscript')
    if rscript is None:
        sys.exit('Rscript not found; the R backend needs an R installation with GRNIToolS')
    cmd = [rscript, '-e', R_CODE, d, method, out, opts]
    return cmd, None


def check_env(backend):
    env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if env not in ('grnitools', 'GRN'):
        print(f'warning: CONDA_DEFAULT_ENV is {env!r}, not "grnitools"; the Python backends '
              f'spawn children with a bare `python` and will land in the wrong env',
              file=sys.stderr)


def show_lists():
    print(f'Methods ({len(DL_METHODS) + len(SCFM_METHODS) + len(R_METHODS)})\n')
    for name, methods in (('GRNIToolS_DL (python)', DL_METHODS),
                          ('GRNIToolS_scFM (python)', SCFM_METHODS),
                          ('GRNIToolS (R)', R_METHODS)):
        print(f'{name}: ' + ' '.join(methods))
        print()
    print('Datasets\n')
    print(f'{"":16}{"DL":6}{"scFM":6}{"R":6}')
    for name, m in DATASETS.items():
        print(f'{name:16}{"yes" if m["dl"] else "-":6}'
              f'{"yes" if m["scfm"] else "-":6}{"yes" if m["r"] else "-":6}')


def main():
    p = argparse.ArgumentParser(
        description='Run any GRNIToolS method through the package that implements it.',
        epilog='Everything after a bare `--` is forwarded to the backend command.')
    p.add_argument('--method')
    p.add_argument('--dataset')
    p.add_argument('--output', default='results')
    p.add_argument('--list', action='store_true', help='print methods and dataset coverage')
    p.add_argument('--dry-run', action='store_true', help='print the command, run nothing')
    p.add_argument('--r-bootstrap', type=int,
                   help='passed to grn_main as bootstrap_num (default 1, i.e. no resampling)')
    p.add_argument('--r-sym', choices=['true', 'false'], help='passed to grn_main as sym')
    p.add_argument('--r-threshold', type=float,
                   help='passed to grn_main as threshold (default 0, the package default)')
    p.add_argument('--r-cores', type=int,
                   help='passed to grn_main as cl.cores (default 2, the package default)')
    p.add_argument('--pt', default=os.path.join('GRNIToolS_scFM', 'model_file', 'GenePT',
                                                'GenePT_gene_embedding_ada_text.pickle'),
                   help='scFM gene-embedding file, relative to the repository root')
    p.add_argument('extra', nargs=argparse.REMAINDER)
    args = p.parse_args()

    if args.list:
        show_lists()
        return 0
    if not args.method or not args.dataset:
        p.error('--method and --dataset are both required (or use --list)')

    backend = backend_of(args.method)
    if backend is None:
        sys.exit(f'unknown method {args.method!r}; run with --list to see them')
    if args.dataset not in DATASETS:
        sys.exit(f'unknown dataset {args.dataset!r}; run with --list to see them')

    out = os.path.abspath(args.output)
    if backend == 'dl':
        check_env(backend)
        target = os.path.join(out, args.method)
        cmd, cwd = build_dl(args.method, args.dataset, out)
    elif backend == 'scfm':
        check_env(backend)
        target = os.path.join(out, args.method)
        cmd, cwd = build_scfm(args.method, args.dataset, target, args.pt)
    else:
        target = os.path.join(out, args.method)
        # bootstrap_num = 1 matches the published protocol: one pass over the full matrix.
        opts = [f'bootstrap_num={1 if args.r_bootstrap is None else args.r_bootstrap}']
        if args.r_sym is not None:
            opts.append(f'sym={args.r_sym.upper()}')
        if args.r_threshold is not None:
            opts.append(f'threshold={args.r_threshold}')
        if args.r_cores is not None:
            opts.append(f'cl.cores={args.r_cores}')
        cmd, cwd = build_r(args.method, args.dataset, os.path.join(target, 'metrics.tsv'),
                           ';'.join(opts))

    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    cmd += extra

    print(f'[{backend}] {args.method} on {args.dataset}')
    print('  ' + ' '.join(cmd if backend != 'r' else cmd[:2] + ['<driver>'] + cmd[3:]))
    if args.dry_run:
        return 0

    os.makedirs(target, exist_ok=True)
    return subprocess.run(cmd, cwd=cwd, check=False).returncode


if __name__ == '__main__':
    sys.exit(main())
