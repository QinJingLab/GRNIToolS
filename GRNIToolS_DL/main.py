import argparse
import os
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import subprocess
import shutil
import time

torch.manual_seed(42)
np.random.seed(42)

# --------------------------
# 命令行参数解析
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Gene Regulatory Network Inference with Deep Learning')
    # 数据路径参数
    parser.add_argument('--expr', type=str, default='./Input_data/Ecoli/GRNDL/expression.csv', help='Gene expression file')
    parser.add_argument('--train', type=str, default='./Input_data/Ecoli/GRNDL/train.txt', help='Training data file')
    parser.add_argument('--valid', type=str, default='./Input_data/Ecoli/GRNDL/val.txt', help='Validation data file')
    parser.add_argument('--test', type=str, default='./Input_data/Ecoli/GRNDL/random_test.txt', help='Test data file')
    parser.add_argument('--tftest', type=str, default='./Input_data/Ecoli/GRNDL/tftest.txt', help='TF test data file')
    parser.add_argument('--grn', type=str, default='./Input_data/Ecoli/GRNDL/grn.txt', help='GRN data file')
    parser.add_argument("--tf", type = str, default='./Input_data/Ecoli/GRNDL/TFlist.tsv', help="<file> Input tf list file")
    parser.add_argument("--gene", type = str, default='./Input_data/Ecoli/GRNDL/genelist.tsv', help="<file> Input gene list file")
    parser.add_argument("--input_dir", type = str, required=False, default='./Input_data/Ecoli/DeepRIG', help="<dir> Input directory")

    # 模型参数
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--head', type=int, default=2, help='Number of head units')
    # 训练参数
    parser.add_argument('--opt', action='store_true', help='Enable Optuna hyperparameter tuning')
    parser.add_argument('--njobs', type=int, default=2, help='Number of jobs to run in Optuna') 
    parser.add_argument('--num_trials', type=int, default=30, help='The number of Optuna trials')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='GRN_output', help='Output directory')
    parser.add_argument('--method', type=str, default='GRNCNN', help='Method for GRN inference')
    parser.add_argument('--network', type=str, default='0', help='GRNDL network prediction, default is False')
    parser.add_argument('--cuda_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES (comma-separated), e.g. 0,1,3')

    return parser.parse_args()


def objective(trial, args):
    # 加载数据与参数
    model_name = args.method
    expr_file, output_dir = args.expr, args.output_dir
    train_file, valid_file, test_file, tftest_file = args.train, args.valid, args.test, args.tftest
    output_dir = f'{output_dir}/{model_name}/'
    # Optuna生成超参数组合
    params = {
        'lr': trial.suggest_float('lr', 0.00001, 0.001, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 256]),
        'epochs': trial.suggest_int('epochs', 20, 300)
    }
    
    # 创建独立输出目录
    trial_dir = os.path.join(output_dir, f"trial_{trial.number}")
    os.makedirs(trial_dir, exist_ok=True)
     
    # 构造模型调用的命令
    if model_name == 'GRNCNN' or model_name == 'GRNResNet':
        command = ["python","./source/GRNDL/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(params['batch_size']), "--epoch", str(params['epochs']), "--lr", str(params['lr']), 
                "--output_dir", trial_dir, 
                "--hidden", str(args.hidden), "--dropout", str(args.dropout), "--head", str(args.head),
                "--model", model_name] 
    
    elif model_name == 'GRNTrans':
        command = ["python","./source/GRNTrans/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(params['batch_size']), "--epoch", str(params['epochs']), "--lr", str(params['lr']), 
                "--output", trial_dir, "--head", str(args.head),
                "--cuda_devices", str(args.cuda_devices)]

    elif model_name == 'GRNGCN':
        command = ["python","./source/GRNGCN/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(params['batch_size']), "--epoch", str(params['epochs']), "--lr", str(params['lr']), 
                "--output", trial_dir, 
                "--hidden", str(args.hidden), "--dropout", str(args.dropout)]        

    elif model_name == 'GENELINK':
        command = ["python","./source/GENELINK/main.py",
                    "-batch_size", str(params['batch_size']), "-epochs", str(params['epochs']), "-lr", str(params['lr']), 
                    "-tf_path",args.tf,"-gene_path",args.gene,
                    '-network_path',args.train,"-valid_path",args.valid,
                    "-random_network_path",args.test,"-tf_network_path",args.tftest,"-expr_path",args.expr,
                    "-output_path", trial_dir]

        
    elif model_name == 'IGEGRNS':
        command = ['python','./source/IGEGRNS/src/Main.py','--geneDataName', expr_file, '--epochs', str(params['epochs']), '--lr', str(params['lr']), 
            '--batchSize', str(params['batch_size']), '--train_file',train_file, '--test_file', test_file, '--valid_file', valid_file,
            '--tftest_file', tftest_file, '--output', trial_dir]
        
    elif model_name == 'DeepRIG':
        command = ['python','./source/DeepRIG/main.py','--input_path', args.input_dir, '--output_path', trial_dir,'--learning_rate', str(params['lr']), '--epochs', str(params['epochs'])]

    elif model_name == 'STGRNS':
        command = ["python","./source/STGRNS/main.py", "--epochs",str(params['epochs']),"--lr",str(params['lr']),
                "--batch_size",str(params['batch_size']),
                "--head", str(args.head),
                "--train", args.train,"--valid",args.valid,
                "--test",args.test,"--tftest",args.tftest,
                "--output",trial_dir,"--expr",args.expr]
        
    elif model_name == 'DeepSEM':
        command = ["python","./source/DeepSEM/main.py","--task","non_celltype_GRN","--setting","test","--data_file",expr_file,"--save_name",trial_dir,
                "--n_epochs",str(params['epochs']),"--batch_size",str(params['batch_size']), "--lr", str(params['lr']), "--net_file", args.grn]

    elif model_name == 'DeepDRIM':
        command = ["python","./source/DeepDRIM/main.py","--expr_file",expr_file,"--output", trial_dir,
                   "--gene_path", args.gene, "--train", args.train,"--valid",args.valid,
                    "--test",args.test,"--tftest",args.tftest,
                    "--epochs", str(params['epochs']), "--lr", str(params['lr']), "--batch_size", str(params['batch_size'])]


    else:
        raise ValueError(f"Unsupported model: {model_name}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Trial {trial.number} failed with error: {e.stderr}")
        return float('nan')
    
    # 解析评估结果
    roc_file = f'{trial_dir}/roc.txt'
    with open(roc_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            if line[0] == 'method':
                continue
            elif line[1] == 'valid':
                valid_roc = float(line[2])
            elif line[1] == 'test' or  line[1] == 'all':
                test_roc = float(line[2])
            elif line[1] == 'tftest':
                tftest_roc = float(line[2])

    return test_roc



# --------------------------
def create_model(args):
    model_name = args.method
    expr_file, output_dir = args.expr, args.output_dir
    train_file, valid_file, test_file, tftest_file = args.train, args.valid, args.test, args.tftest
    output_dir = f'{output_dir}/{model_name}/'
    if model_name == 'GRNCNN' or model_name == 'GRNResNet':
        command = ["python","./source/GRNDL/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(args.batch_size), "--epoch",  str(args.epochs), "--lr", str(args.lr), 
                "--output_dir", output_dir, 
                "--hidden", str(args.hidden), "--dropout", str(args.dropout), "--head", str(args.head),
                "--model", model_name, "--network", str(args.network),
                "--cuda_devices", str(args.cuda_devices)] 
    
    elif model_name == 'GRNTrans':
        command = ["python","./source/GRNTrans/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(args.batch_size), "--epoch",  str(args.epochs), "--lr", str(args.lr), 
                "--output", output_dir, "--head", str(args.head), "--network", str(args.network),
                "--cuda_devices", str(args.cuda_devices)]

    elif model_name == 'GRNGCN':
        command = ["python","./source/GRNGCN/main.py","--expr", expr_file, "--train", train_file, "--valid", valid_file,
                "--test", test_file, "--tftest", tftest_file,
                "--batch_size", str(args.batch_size), "--epoch",  str(args.epochs), "--lr", str(args.lr), 
                "--output", output_dir, 
                "--hidden", str(args.hidden), "--dropout", str(args.dropout),
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
                "--head", str(args.head),
                "--train", args.train,"--valid",args.valid,
                "--test",args.test,"--tftest",args.tftest,
                "--output",f'{output_dir}/',"--expr",args.expr,
                "--cuda_devices", str(args.cuda_devices)]

    elif model_name == 'DeepRIG':
        command = ['python',"./source/DeepRIG/main.py",'--input_path', args.input_dir, '--output_path', output_dir,'--learning_rate', str(args.lr), '--epochs', str(args.epochs)]

    elif model_name == 'DeepSEM':
        command = ["python","./source/DeepSEM/main.py","--task","non_celltype_GRN","--setting","test","--data_file",expr_file,"--save_name",output_dir,
                "--n_epochs",str(args.epochs),"--batch_size",str(args.batch_size), "--lr", str(args.lr), "--net_file", args.grn]

    elif model_name == 'DeepDRIM':
        command = ["python","./source/DeepDRIM/main.py","--expr_file",expr_file,"--output",output_dir,
                   "--gene_path", args.gene, "--train", args.train,"--valid",args.valid,
                    "--test",args.test,"--tftest",args.tftest,
                    "--epochs", str(args.epochs), "--lr", str(args.lr), "--batch_size", str(args.batch_size)]


    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return command

def setup_cuda_devices(cuda_devices):

    if cuda_devices:
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_devices
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            requested_devices = [int(x.strip()) for x in cuda_devices.split(',')]
            for device_id in requested_devices:
                if device_id >= device_count:
                    print(f"Warning: Device {device_id} not found. Available range: 0-{device_count-1}")
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                print(f"Device {i}: {device_name}")
        else:
            print("Warning: CUDA is not available. Using CPU instead.   ")
    else:
        print("Using default CUDA device settings.")
        
def cleanup_output_dir(args, best_trial_number):
    output_root = f'{args.output_dir}/{args.method}/'
    kept_count = 0
    deleted_count = 0
    for dir_name in os.listdir(output_root):
        dir_path = os.path.join(output_root, dir_name)
        if os.path.isdir(dir_path):
            # 匹配 trial_数字 格式的文件夹
            if dir_name.startswith("trial_"):
                try:
                    trial_num = int(dir_name.split("_")[1])
                    if trial_num != best_trial_number:
                        shutil.rmtree(dir_path)
                        deleted_count += 1
                    else:
                        kept_count += 1
                except (ValueError, IndexError):
                    print(f"Skip: {dir_name}")


def main():
    args = parse_args()
    setup_cuda_devices(args.cuda_devices)
    # 创建工作目录
    output_dir = f'{args.output_dir}/{args.method}/'
    os.makedirs(output_dir, exist_ok=True) 
    
    if args.opt:
        # Optuna超参数调优
        n_jobs = args.njobs
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: objective(trial, args), n_trials=args.num_trials, n_jobs=n_jobs)

        # 保存最佳参数
        best_params = study.best_params
        best_trial_number = study.best_trial.number
        with open(os.path.join(output_dir, 'best_params.txt'), 'w') as f:
            f.write(str(best_params))
        print(f"Best Parameters: {best_params}")
        print("\n[优化结果]")
        print("最佳参数组合:", study.best_params)
        print("最高AUROC:", study.best_value)
        cleanup_output_dir(args, best_trial_number)

    else:
        command = create_model(args)
        print("Executing command:", ' '.join(command))
        start_time = time.time()     
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        res = result.stdout
        end_time = time.time()     
        elapsed_time = end_time - start_time  
        print(f"Elapsed time: {elapsed_time:.5f} seconds")
        res += f'\nElapsed_time\t{elapsed_time:.5f} seconds'    
        with open(f'{output_dir}/log.txt','w') as f:
            f.write(res)

if __name__ == "__main__":
    main()