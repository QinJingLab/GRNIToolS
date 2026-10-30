import argparse
import os
import sys
import optuna
import torch
import numpy as np
import subprocess
import shutil
import socket
import time

# The unified evaluator, imported by path: the search maximises what the table reports.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark'))
from evaluate import SPLIT_CANDIDATES, evaluate  # noqa: E402

# The original author's marker: the methods are separate interpreters, so these do not reach them.
torch.manual_seed(42)
np.random.seed(42)

# Methods whose CLI accepts a seed, each with its own spelling; the other seven reject one.
SEED_FLAGS = {'GRNGCN': '--seed', 'GENELINK': '-seed'}

# Trial n is handed this plus n, so its result does not depend on when it was scheduled.
OPTUNA_BASE_SEED = 42

# Methods whose CLI reads the TF/gene name lists, and which of --tf/--gene each one takes.
NAME_LIST_METHODS = {'GENELINK': ('tf', 'gene'), 'DeepDRIM': ('gene',),
                     'DeepSEM': ('tf', 'gene')}

# Methods that take a directory of preprocessed inputs instead of the split files: DeepRIG.
AUX_DIR_METHODS = {'DeepRIG': 'DeepRIG'}

# The four split files under Input_data/<dataset>/GRNDL/, derived from --expr's directory.
SPLIT_FILES = {'train': ('train.txt',), 'valid': ('val.txt',),
               'test': ('random_test.txt', 'test.txt'), 'tftest': ('TF_test.txt',)}

# IGEGRNS takes its own splits: they index genes by row number and carry a header.
SPLIT_OVERRIDES = {'IGEGRNS': {'train': 'train.tsv', 'valid': 'val.tsv',
                               'test': 'test.tsv', 'tftest': 'tftest.tsv'}}

# Methods whose CLI takes the expression matrix transposed; see resolve_transposed_expr.
TRANSPOSED_EXPR_METHODS = {'DeepSEM': 'expression_t.csv'}

# Methods scored against something other than the --grn edge list; candidates tried in order.
GOLD_STANDARD_METHODS = {'DeepSEM': ('HGS.txt', 'grn.txt')}

# Attention heads per method when --head is not given; each matches that method's own default.
HEAD_DEFAULTS = {'GRNTrans': 4, 'STGRNS': 2}


def head_count(model_name, requested):
    return requested if requested is not None else HEAD_DEFAULTS.get(model_name, 4)

# --------------------------
# command-line arguments
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Gene Regulatory Network Inference with Deep Learning')
    # data paths
    parser.add_argument('--expr', type=str, default='./Input_data/Ecoli/GRNDL/expression.csv', help='Gene expression file')
    parser.add_argument('--train', type=str, default=None, help='Training data file (default: train.txt beside --expr)')
    parser.add_argument('--valid', type=str, default=None, help='Validation data file (default: val.txt beside --expr)')
    parser.add_argument('--test', type=str, default=None, help='Test data file (default: random_test.txt beside --expr)')
    parser.add_argument('--tftest', type=str, default=None, help='TF test data file (default: TF_test.txt beside --expr)')
    parser.add_argument('--grn', type=str, default='./Input_data/Ecoli/GRNDL/grn.txt', help='GRN data file')
    parser.add_argument("--tf", type = str, default=None, help="<file> Input tf list file (default: TFlist.tsv beside --expr)")
    parser.add_argument("--gene", type = str, default=None, help="<file> Input gene list file (default: genelist.tsv beside --expr)")
    parser.add_argument("--input_dir", type = str, required=False, default=None, help="<dir> Input directory (default: the DeepRIG directory beside --expr)")

    # model parameters
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units')
    parser.add_argument('--head', type=int, default=None,
                        help='Number of attention heads (default: 4 for GRNTrans, 2 for STGRNS)')
    # training parameters
    parser.add_argument('--opt', action='store_true', help='Enable Optuna hyperparameter tuning')
    parser.add_argument('--njobs', type=int, default=2,
                        help='Number of jobs to run in Optuna. Above 1 the search is faster but '
                             'not reproducible: a seeded sampler only reproduces the suggestion '
                             'sequence when trials run one at a time, because with concurrent '
                             'trials the suggestion-to-trial assignment follows completion order. '
                             'The run says so, in a warning and in best_params.txt.')
    parser.add_argument('--num_trials', type=int, default=30, help='The number of Optuna trials')
    parser.add_argument('--cleanup_trials', action='store_true',
                        help='After an --opt run, delete every trial_* directory except the best one. Off by default: those directories are the record of the search, and deletion cannot be undone.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='GRN_output', help='Output directory')
    parser.add_argument('--method', type=str, default='GRNCNN', help='Method for GRN inference')
    parser.add_argument('--network', type=str, default='0', help='GRNDL network prediction, default is False')
    parser.add_argument('--cuda_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES (comma-separated), e.g. 0,1,3')

    return parser.parse_args()


_allocated_ports = set()


def find_free_port():
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            port = s.getsockname()[1]
        if port not in _allocated_ports:
            _allocated_ports.add(port)
            return port


def read_roc_txt(path, split):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            fields = line.strip().split('\t')
            if len(fields) < 3 or fields[0] == 'method':
                continue
            if fields[1] == split or (split == 'test' and fields[1] == 'all'):
                try:
                    return float(fields[2])
                except ValueError:
                    return None
    return None


def constant_dump(path):
    """True when every row of a scored dump carries the same score."""
    try:
        with open(path) as f:
            col = f.readline().rstrip('\n').split('\t').index('score')
    except (OSError, ValueError):
        return False
    seen = set()
    with open(path) as f:
        next(f, None)
        for line in f:
            fields = line.rstrip('\n').split('\t')
            if len(fields) > col and fields[col]:
                seen.add(fields[col])
                if len(seen) > 1:
                    return False
    return len(seen) == 1


def gold_split(args, split):
    # The labels are always the name-based splits under <dataset>/GRNDL/, whichever
    # directory --expr points into; IGEGRNS's own splits hold gene row numbers instead.
    g = os.path.join(os.path.dirname(os.path.dirname(args.expr)), 'GRNDL')
    names = SPLIT_CANDIDATES[split]
    for name in names:
        if os.path.exists(os.path.join(g, name)):
            return os.path.join(g, name)
    return os.path.join(g, names[0])


def scoring_split(args):
    return gold_split(args, 'valid')


def rescore_roc_txt(output_dir, args):
    """Rewrite roc.txt from the dumps, so the file agrees with benchmark/evaluate.py.
    A method that writes no predictions_<split>.tsv (DeepSEM) keeps its own roc.txt."""
    rows = []
    for split in ('valid', 'test', 'tftest'):
        dump = os.path.join(output_dir, f'predictions_{split}.tsv')
        truth = gold_split(args, split)
        if not (os.path.exists(dump) and os.path.exists(truth)):
            continue
        try:
            scored = evaluate(dump, truth)
        except Exception as e:
            print(f'[GRNIToolS] {dump} could not be scored ({type(e).__name__}: {e}); '
                  f'roc.txt left as the method wrote it')
            return
        rows.append(f'{args.method}\t{split}\t{scored["auroc"]:.5f}\t{scored["auprc"]:.5f}')
    if not rows:
        return
    with open(os.path.join(output_dir, 'roc.txt'), 'w') as f:
        f.write('method\tdataset\tauroc\taupr\n' + '\n'.join(rows) + '\n')


def objective(trial, args):
    # data and arguments
    model_name = args.method
    expr_file, output_dir = args.expr, args.output_dir
    train_file, valid_file, test_file, tftest_file = args.train, args.valid, args.test, args.tftest
    output_dir = f'{output_dir}/{model_name}/'
    # the hyperparameters Optuna suggests for this trial
    params = {
        'lr': trial.suggest_float('lr', 0.00001, 0.001, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 256]),
        'epochs': trial.suggest_int('epochs', 20, 300)
    }
    
    # one output directory per trial
    trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
     
    # The command for this method; --cuda_devices is forwarded or a method's own default wins.
    if model_name == 'GRNCNN' or model_name == 'GRNResNet':
        command = ["python","./source/GRNDL/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(params['batch_size']), "--epochs", str(params['epochs']), "--lr", str(params['lr']),
                "--output_dir", trial_dir,
                "--model", model_name,
                "--cuda_devices", str(args.cuda_devices)]
    
    elif model_name == 'GRNTrans':
        command = ["python","./source/GRNTrans/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(params['batch_size']), "--epochs", str(params['epochs']), "--lr", str(params['lr']), 
                "--output", trial_dir, "--head", str(head_count(model_name, args.head)),
                "--cuda_devices", str(args.cuda_devices)]

    elif model_name == 'GRNGCN':
        command = ["python","./source/GRNGCN/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(params['batch_size']), "--epochs", str(params['epochs']), "--lr", str(params['lr']), 
                "--output", trial_dir,
                "--hidden", str(args.hidden),
                "--cuda_devices", str(args.cuda_devices)]

    elif model_name == 'GENELINK':
        command = ["python","./source/GENELINK/main.py",
                    "-batch_size", str(params['batch_size']), "-epochs", str(params['epochs']), "-lr", str(params['lr']), 
                    "-tf_path",args.tf,"-gene_path",args.gene,
                    '-network_path',args.train,"-valid_path",args.valid,
                    "-random_network_path",args.test,"-tf_network_path",args.tftest,"-expr_path",args.expr,
                    "-output_path", trial_dir,
                    "--cuda_devices", str(args.cuda_devices)]

        
    elif model_name == 'IGEGRNS':
        command = ['python','./source/IGEGRNS/src/Main.py','--geneDataName', expr_file, '--epochs', str(params['epochs']), '--lr', str(params['lr']), 
            '--batchSize', str(params['batch_size']), '--train_file',train_file, '--test_file', test_file, '--valid_file', valid_file,
            '--tftest_file', tftest_file, '--output', trial_dir]
        
    elif model_name == 'DeepRIG':
        command = ['python','./source/DeepRIG/main.py','--input_path', args.input_dir, '--output_path', trial_dir,'--learning_rate', str(params['lr']), '--epochs', str(params['epochs'])]

    elif model_name == 'STGRNS':
        command = ["python","./source/STGRNS/main.py", "--epochs",str(params['epochs']),"--lr",str(params['lr']),
                "--batch_size",str(params['batch_size']),
                "--head", str(head_count(model_name, args.head)),
                "--train", args.train,"--valid",args.valid,
                "--test",args.test,"--tftest",args.tftest,
                "--output",trial_dir,"--expr",args.expr,
                "--cuda_devices", str(args.cuda_devices)]
        
    elif model_name == 'DeepSEM':
        command = ["python","./source/DeepSEM/main.py","--task","non_celltype_GRN","--setting","test","--data_file",args.transposed_expr,"--save_name",trial_dir,
                "--n_epochs",str(params['epochs']),"--batch_size",str(params['batch_size']), "--lr", str(params['lr']), "--net_file", args.grn,
                "--hgs", args.hgs, "--tf_list", args.tf, "--gene_list", args.gene]

    elif model_name == 'DeepDRIM':
        # -cuda is forwarded so the method's own "0" default cannot override CUDA_VISIBLE_DEVICES.
        command = ["python","./source/DeepDRIM/main.py","--expr_file",expr_file,"--output", trial_dir,
                   "--gene_path", args.gene, "--train", args.train,"--valid",args.valid,
                    "--test",args.test,"--tftest",args.tftest,
                    "--epochs", str(params['epochs']), "--lr", str(params['lr']), "--batch_size", str(params['batch_size']),
                    "--cuda", str(args.cuda_devices)]


    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # A per-trial seed, taken from the trial number so trial n reproduces in isolation.
    if model_name in SEED_FLAGS:
        command += [SEED_FLAGS[model_name], str(OPTUNA_BASE_SEED + trial.number)]

    try:
        # One MASTER_PORT per trial, via env= so parallel trials cannot overwrite each other.
        trial_env = dict(os.environ, MASTER_PORT=str(find_free_port()))
        result = subprocess.run(command, check=True, capture_output=True, text=True, env=trial_env)
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} failed with error: {e.stderr}")
        return float('nan')
    except OSError as e:
        # The child could not be launched at all; report it as a failed trial, not a traceback.
        print(f"Trial {trial.number} could not run: {type(e).__name__}: {e}")
        return float('nan')

    # Score on valid, never test, and with the table's evaluator, which counts each pair once.
    valid_roc = None
    dump = f'{trial_dir}/predictions_valid.tsv'
    truth = scoring_split(args)
    if os.path.exists(dump) and os.path.exists(truth):
        try:
            valid_roc = evaluate(dump, truth)['auroc']
        except Exception as e:
            print(f"Trial {trial.number}: {dump} could not be scored "
                  f"({type(e).__name__}: {e}); falling back to roc.txt")

    if valid_roc is None:
        # A method that emits no dump, or a dump the evaluator rejects: fall back to the method's own number so the trial still counts.
        valid_roc = read_roc_txt(f'{trial_dir}/roc.txt', 'valid')
        if valid_roc is not None:
            print(f"Trial {trial.number}: scored from roc.txt, not the unified evaluator")

    if valid_roc is None:
        # Return nan instead of raising: an exception inside objective() aborts the whole study.
        return float('nan')
    if os.path.exists(dump) and constant_dump(dump):
        # Every pair scored the same, so 0.5 ranks nothing and must not win the search.
        print(f"Trial {trial.number}: constant dump, excluded from the search")
        raise optuna.TrialPruned()
    # Record the number the search ranked on; it comes from the evaluator, not the method's roc.txt.
    with open(f'{trial_dir}/objective.txt', 'w') as f:
        f.write(f'valid\t{valid_roc}\n')
    rescore_roc_txt(trial_dir, args)
    return valid_roc



# --------------------------
def create_model(args):
    model_name = args.method
    expr_file, output_dir = args.expr, args.output_dir
    train_file, valid_file, test_file, tftest_file = args.train, args.valid, args.test, args.tftest
    output_dir = f'{output_dir}/{model_name}/'
    if model_name == 'GRNCNN' or model_name == 'GRNResNet':
        command = ["python","./source/GRNDL/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(args.batch_size), "--epochs",  str(args.epochs), "--lr", str(args.lr),
                "--output_dir", output_dir,
                "--model", model_name, "--network", str(args.network),
                "--cuda_devices", str(args.cuda_devices)]
    
    elif model_name == 'GRNTrans':
        command = ["python","./source/GRNTrans/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(args.batch_size), "--epochs",  str(args.epochs), "--lr", str(args.lr), 
                "--output", output_dir, "--head", str(head_count(model_name, args.head)), "--network", str(args.network),
                "--cuda_devices", str(args.cuda_devices)]

    elif model_name == 'GRNGCN':
        command = ["python","./source/GRNGCN/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(args.batch_size), "--epochs",  str(args.epochs), "--lr", str(args.lr), 
                "--output", output_dir,
                "--hidden", str(args.hidden),
                "--cuda_devices", str(args.cuda_devices)]

    elif model_name == 'GENELINK':
        command = ["python","./source/GENELINK/main.py", "-epochs",str(args.epochs),"-lr",str(args.lr),"-batch_size",str(args.batch_size),
                        "-tf_path",args.tf,"-gene_path",args.gene,
                        '-network_path',args.train,"-valid_path",args.valid,
                        "-random_network_path",args.test,"-tf_network_path",args.tftest,
                        "-output_path",f'{output_dir}/',"-expr_path",args.expr,
                        "--cuda_devices", str(args.cuda_devices)]

    elif model_name == 'IGEGRNS':
        command = ['python',"./source/IGEGRNS/src/Main.py",'--geneDataName', expr_file, '--epochs', str(args.epochs), '--lr', str(args.lr), 
                   '--batchSize',str(args.batch_size), '--train_file',train_file, '--test_file', test_file, '--valid_file', valid_file,
                    '--tftest_file', tftest_file, '--output', output_dir]

    elif model_name == 'STGRNS':
        command = ["python","./source/STGRNS/main.py", "--epochs",str(args.epochs),"--lr",str(args.lr),
                "--batch_size",str(args.batch_size),
                "--head", str(head_count(model_name, args.head)),
                "--train", args.train,"--valid",args.valid,
                "--test",args.test,"--tftest",args.tftest,
                "--output",f'{output_dir}/',"--expr",args.expr,
                "--cuda_devices", str(args.cuda_devices)]

    elif model_name == 'DeepRIG':
        command = ['python',"./source/DeepRIG/main.py",'--input_path', args.input_dir, '--output_path', output_dir,'--learning_rate', str(args.lr), '--epochs', str(args.epochs)]

    elif model_name == 'DeepSEM':
        command = ["python","./source/DeepSEM/main.py","--task","non_celltype_GRN","--setting","test","--data_file",args.transposed_expr,"--save_name",output_dir,
                "--n_epochs",str(args.epochs),"--batch_size",str(args.batch_size), "--lr", str(args.lr), "--net_file", args.grn,
                "--hgs", args.hgs, "--tf_list", args.tf, "--gene_list", args.gene]

    elif model_name == 'DeepDRIM':
        # -cuda is forwarded here too; see the matching branch in objective() for why.
        command = ["python","./source/DeepDRIM/main.py","--expr_file",expr_file,"--output",output_dir,
                   "--gene_path", args.gene, "--train", args.train,"--valid",args.valid,
                    "--test",args.test,"--tftest",args.tftest,
                    "--epochs", str(args.epochs), "--lr", str(args.lr), "--batch_size", str(args.batch_size),
                    "--cuda", str(args.cuda_devices)]


    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return command

def setup_cuda_devices(cuda_devices):

    if cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        if torch.cuda.is_available():
            # device_count() already accounts for the filtering above, so it is what the children see.
            device_count = torch.cuda.device_count()
            requested_devices = [x.strip() for x in cuda_devices.split(',') if x.strip()]
            if device_count == 0:
                print(f"Warning: CUDA_VISIBLE_DEVICES='{cuda_devices}' exposes no device.")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                physical_id = requested_devices[i] if i < len(requested_devices) else '?'
                print(f"Device {i}: {device_name} (CUDA_VISIBLE_DEVICES={physical_id})")
        else:
            print("Warning: CUDA is not available. Using CPU instead.   ")
    else:
        print("Using default CUDA device settings.")
        
def summarise_trials(output_root):
    rows, objectives = [], []
    for dir_name in sorted(os.listdir(output_root)):
        if not dir_name.startswith('trial_'):
            continue
        obj_file = os.path.join(output_root, dir_name, 'objective.txt')
        if os.path.exists(obj_file):
            with open(obj_file) as f:
                split, _, value = f.read().strip().partition('\t')
            objectives.append((dir_name, split, value))
        roc_file = os.path.join(output_root, dir_name, 'roc.txt')
        if not os.path.exists(roc_file):
            continue
        with open(roc_file) as f:
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) >= 4 and fields[0] != 'method':
                    rows.append((dir_name,) + tuple(fields[:4]))
    if objectives:
        obj_path = os.path.join(output_root, 'objective_summary.tsv')
        with open(obj_path, 'w') as f:
            f.write('trial\tsplit\tauroc\n')
            for row in objectives:
                f.write('\t'.join(row) + '\n')
    if not rows:
        return None
    path = os.path.join(output_root, 'trials_summary.tsv')
    with open(path, 'w') as f:
        f.write('trial\tmethod\tdataset\tauroc\taupr\n')
        for row in rows:
            f.write('\t'.join(row) + '\n')
    return path


def cleanup_output_dir(args, best_trial_number):
    output_root = f'{args.output_dir}/{args.method}/'
    if not args.cleanup_trials:
        summary = summarise_trials(output_root)
        print(f'Keeping all trial directories under {output_root} '
              f'(pass --cleanup_trials to remove all but trial_{best_trial_number}).'
              + (f' Summary written to {summary}.' if summary else ''))
        return
    summary = summarise_trials(output_root)
    if summary:
        print(f'Wrote {summary} before removing the losing trials.')
    deleted_count = 0
    for dir_name in os.listdir(output_root):
        dir_path = os.path.join(output_root, dir_name)
        if os.path.isdir(dir_path):
            # the trial_<number> directories
            if dir_name.startswith("trial_"):
                try:
                    trial_num = int(dir_name.split("_")[1])
                    if trial_num != best_trial_number:
                        shutil.rmtree(dir_path)
                        deleted_count += 1
                except (ValueError, IndexError):
                    print(f"Skip: {dir_name}")
    print(f'Removed {deleted_count} trial director(y/ies); kept trial_{best_trial_number}.')


def resolve_name_lists(args):
    needed = NAME_LIST_METHODS.get(args.method)
    if not needed:
        return
    dataset_dir = os.path.dirname(args.expr)
    if 'tf' in needed and args.tf is None:
        args.tf = os.path.join(dataset_dir, 'TFlist.tsv')
    if 'gene' in needed and args.gene is None:
        args.gene = os.path.join(dataset_dir, 'genelist.tsv')
    print(f'[GRNIToolS] name lists for {args.method}: '
          + ' '.join(f'--{f} {getattr(args, f)}' for f in needed))
    for flag in needed:
        path = getattr(args, flag)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'--{flag} {path} does not exist. It is derived from the --expr '
                f'directory ({dataset_dir}); pass --{flag} explicitly if the name '
                f'list lives elsewhere.')


def resolve_aux_dir(args):
    if args.method not in AUX_DIR_METHODS or args.input_dir is not None:
        return
    dataset_dir = os.path.dirname(os.path.dirname(args.expr))  # <dataset>/GRNDL -> <dataset>
    args.input_dir = os.path.join(dataset_dir, AUX_DIR_METHODS[args.method])
    print(f'[GRNIToolS] input dir for {args.method}: --input_dir {args.input_dir}')
    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(
            f'--input_dir {args.input_dir} is not a directory. It is derived from the '
            f'--expr directory ({dataset_dir}); pass --input_dir explicitly if the '
            f'preprocessed inputs live elsewhere.')


def resolve_transposed_expr(args):
    name = TRANSPOSED_EXPR_METHODS.get(args.method)
    if name is None:
        return
    dataset_dir = os.path.dirname(args.expr)
    args.transposed_expr = os.path.join(dataset_dir, name)
    print(f'[GRNIToolS] data file for {args.method}: --data_file {args.transposed_expr}')
    if not os.path.exists(args.transposed_expr):
        raise FileNotFoundError(
            f'--data_file {args.transposed_expr} does not exist. It is derived from the '
            f'--expr directory ({dataset_dir}); DeepSEM needs the samples x genes '
            f'orientation, which has to sit beside the expression matrix.')


def resolve_gold_standard(args):
    names = GOLD_STANDARD_METHODS.get(args.method)
    if names is None:
        return
    dataset_dir = os.path.dirname(args.expr)
    missed = []
    for name in names:
        candidate = os.path.join(dataset_dir, name)
        if os.path.exists(candidate):
            args.hgs = candidate
            if missed:
                print(f'[GRNIToolS] {args.method}: {", ".join(missed)} is not beside '
                      f'--expr, falling back to --hgs {args.hgs}. Note this is the '
                      f'--grn edge list, not the R package\'s gold standard, so the '
                      f'score is not comparable with the R package\'s table.')
            else:
                print(f'[GRNIToolS] gold standard for {args.method}: --hgs {args.hgs}')
            return
        missed.append(name)
    raise FileNotFoundError(
        f'None of {", ".join(names)} exists in the --expr directory ({dataset_dir}), '
        f'so {args.method} has no gold standard to be scored against.')


def resolve_split_paths(args):
    dataset_dir = os.path.dirname(args.expr)
    layout = SPLIT_OVERRIDES.get(args.method)
    if layout and not os.path.exists(os.path.join(dataset_dir, layout['train'])):
        base = os.path.join(os.path.dirname(dataset_dir), 'IGEGRNS')
    else:
        base = dataset_dir
    paths = {}
    for flag, spec in (layout or SPLIT_FILES).items():
        names = (spec,) if isinstance(spec, str) else tuple(spec)
        found = next((os.path.join(base, n) for n in names
                      if os.path.exists(os.path.join(base, n))), None)
        paths[flag] = found or os.path.join(base, names[0])
    for flag, path in paths.items():
        if getattr(args, flag) is None:
            setattr(args, flag, path)
    print(f'[GRNIToolS] splits for {args.method}: '
          + ' '.join(f'--{f} {getattr(args, f)}' for f in paths))
    for flag in paths:
        path = getattr(args, flag)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'--{flag} {path} does not exist. It is derived from the --expr '
                f'directory ({dataset_dir}); pass --{flag} explicitly if the split '
                f'file lives elsewhere.')


def main():
    args = parse_args()
    resolve_split_paths(args)
    resolve_name_lists(args)
    resolve_aux_dir(args)
    resolve_transposed_expr(args)
    resolve_gold_standard(args)
    # GRNGCN indexes cuda:{--cuda_devices} into the full device list, so it needs this variable absent rather than set (same reason run_result_dl.sh runs it with env -u).
    setup_cuda_devices('' if args.method == 'GRNGCN' else args.cuda_devices)
    # working directory for this method's outputs
    output_dir = f'{args.output_dir}/{args.method}/'
    os.makedirs(output_dir, exist_ok=True)
    
    if args.opt:
        # Optuna search; TPESampler(seed=42) reproduces only while trials run one at a time.
        n_jobs = args.njobs
        serial = n_jobs == 1
        if not serial:
            print(f'WARNING: --njobs {n_jobs} > 1. The sampler is seeded, but with concurrent '
                  f'trials the suggestion-to-trial assignment follows completion order, so this '
                  f'search will not reproduce. Use --njobs 1 for a reproducible run.')
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, args), n_trials=args.num_trials, n_jobs=n_jobs)

        # Every trial failed: study.best_trial would raise, so say what happened instead.
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed:
            print(f'No trial completed successfully out of {len(study.trials)}. '
                  f'Every trial either failed or could not be scored; the per-trial reasons '
                  f'are above, and any partial output is under {output_dir}trial_*/')
            return

        # record the winning parameters
        best_params = study.best_params
        best_trial_number = study.best_trial.number
        with open(os.path.join(output_dir, 'best_params.txt'), 'w') as f:
            f.write('# objective: AUROC on the valid split, from benchmark/evaluate.py\n')
            f.write(f'# sampler TPESampler(seed=42), njobs={n_jobs}\n')
            if serial:
                f.write('# suggestion sequence: reproducible (trials run one at a time)\n')
                if args.method in SEED_FLAGS:
                    f.write(f'# per-trial seed: {args.method} {SEED_FLAGS[args.method]} '
                            f'{OPTUNA_BASE_SEED} + trial number\n')
                else:
                    f.write(f'# per-trial seed: none -- {args.method} has no seed flag, so its '
                            'trials are unseeded and their results will differ between runs\n')
            else:
                f.write('# suggestion sequence: NOT reproducible -- with concurrent trials the '
                        'suggestion-to-trial assignment follows completion order\n')
                f.write('# a single trial still reproduces on its own, but the search does not\n')
            f.write('# note: this says nothing about a method being bit-reproducible when it is\n'
                    '# seeded; that is a property of the method, measured separately\n')
            f.write(str(best_params))
        print(f"Best Parameters: {best_params}")
        print("\n[Optuna result]")
        print("best parameters:", study.best_params)
        print("best valid AUROC:", study.best_value)
        cleanup_output_dir(args, best_trial_number)

    else:
        command = create_model(args)
        print("Executing command:", ' '.join(command))
        start_time = time.time()
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            # capture_output swallows the child's stderr; log both streams before re-raising.
            with open(f'{output_dir}/log.txt', 'w') as f:
                f.write(f'command: {" ".join(command)}\nexit: {e.returncode}\n\n'
                        f'--- stdout ---\n{e.stdout or ""}\n--- stderr ---\n{e.stderr or ""}\n')
            print(f"Method failed (exit {e.returncode}); see {output_dir}/log.txt")
            raise
        res = result.stdout + (f'\n--- stderr ---\n{result.stderr}' if result.stderr else '')
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time:.5f} seconds")
        res += f'\nElapsed_time\t{elapsed_time:.5f} seconds'
        with open(f'{output_dir}/log.txt','w') as f:
            f.write(res)
        rescore_roc_txt(output_dir, args)

if __name__ == "__main__":
    main()