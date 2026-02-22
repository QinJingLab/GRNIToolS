import argparse
import os
import sys
import time
import subprocess
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='This process is used to construct gene regulator networks')
    parser.add_argument('--expr_file', required=False, default='./input_data/Ecoli/DeepDRIM/expression.csv', help='The file of the gene expression profile. Can be h5 or csv file, the format please refer the example data.')
    parser.add_argument('--output', required=False, default="./output/", help="Indicate the path for output.")
    parser.add_argument('--cuda', required=False, default="0", help="The device used in model")
    parser.add_argument('--epochs', type=int, default=120, help='Number of Epochs for training')
    parser.add_argument('--gene_path', required=False, default='./input_data/Ecoli/DeepDRIM/genelist.tsv', help="The path that includes gene name")
    parser.add_argument('--train', type =str, default = './input_data/Ecoli/DeepDRIM/train.txt', help="<file> Input train dataset")
    parser.add_argument('--test', type =str, default = './input_data/Ecoli/DeepDRIM/random_test.txt', help="<file> Input test dataset")
    parser.add_argument('--valid', type =str, default = './input_data/Ecoli/DeepDRIM/val.txt', help="<file> Input valid dataset")
    parser.add_argument('--tftest', type =str, default='./input_data/Ecoli/DeepDRIM/TF_test.txt', help="<file> Input tftest dataset")
   
    parser.add_argument('--lr', type=float, required=False, default=0.001, help="The learning rate used in model")
    parser.add_argument('--batch_size', type=int, required=False, default=32, help="The batch size used in model")
    parser.add_argument('--cross_validation_fold_divide_file', default=None, help="A file that indicate how to divide the x file into three-fold. The file include three line, each line list the ID of the x files for the folder (split by ',')")
    parser.add_argument('--to_predict', default=False, help="True or False. Default is False, then the code will do cross-validation evaluation. If set to True, we need to indicate weight_path for a trained model and the code will do prediction based on the trained model.")
    parser.add_argument('--weight_path', default=None, help="The path for a trained model.")
    parser.add_argument('--label_path', default=None, help="The path for a label file, to evaluate the prediction. Only need in predict model.")
    return parser.parse_args()


# args.gene_path = '/mnt/sdb/ZYQ/workspace/GRNITools2/grn_github/GRNIToolS/GRNIToolS_DL/Input_data/SynTReN/GRNDL/genelist.tsv'
# args.train = '/mnt/sdb/ZYQ/workspace/GRNITools2/grn_github/GRNIToolS/GRNIToolS_DL/Input_data/SynTReN/GRNDL/train.txt'
# args.valid = '/mnt/sdb/ZYQ/workspace/GRNITools2/grn_github/GRNIToolS/GRNIToolS_DL/Input_data/SynTReN/GRNDL/val.txt'
# args.test = '/mnt/sdb/ZYQ/workspace/GRNITools2/grn_github/GRNIToolS/GRNIToolS_DL/Input_data/SynTReN/GRNDL/random_test.txt'
# args.tftest = '/mnt/sdb/ZYQ/workspace/GRNITools2/grn_github/GRNIToolS/GRNIToolS_DL/Input_data/SynTReN/GRNDL/TF_test.txt'
# args.expr_file = '/mnt/sdb/ZYQ/workspace/GRNITools2/grn_github/GRNIToolS/GRNIToolS_DL/Input_data/SynTReN/GRNDL/expression.csv'

def run_DeepDRIM(args):
    # method_name = 'DeepDRIM'
    # output_dir = args.output + '/' + method_name
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    def cheat(gene_path,network_path,data_type,output_path):
        with open(gene_path,'r')as file:
            lines=file.readlines()
        output_deepdrim_input_dir = f'{output_path}/input/{data_type}/'
        if not os.path.exists(output_deepdrim_input_dir):
            os.makedirs(output_deepdrim_input_dir)

        geneMap=f'{output_deepdrim_input_dir}/geneMap.txt'
        with open(geneMap,'w')as file:
            for line in lines:
                line = line.strip()
                new_line = f"{line}\t{line}\n"  # 在每行后面添加一个副本
                file.write(new_line)

        with open(network_path,'r')as file:
            lines=file.readlines() 
            column_length=len(lines)    
        a=column_length
        divide=f'{output_deepdrim_input_dir}/network_divide.txt'
        with open(divide,'w') as file:
            file.write("0\n") 
            file.write(f"{int(a/2)}\n")
            file.write(f"{a}\n")  

    cheat(args.gene_path,args.train,'train',output_dir)
    cheat(args.gene_path,args.valid,'valid',output_dir)
    cheat(args.gene_path,args.test,'test',output_dir)
    cheat(args.gene_path,args.tftest,'TF_test',output_dir)

    output_deepdrim_input_dir = f'{output_dir}/input/'
    with open(f'{output_deepdrim_input_dir}/cross.txt','w') as file:
        file.write("0\n") 
        file.write("1\n")
        file.write("2\n") 

    def get_command(output_path, expr_file, network, data_type):
        command = ["python","./source/DeepDRIM/generate_input_realdata.py",
                "-out_dir",f'{output_path}/{data_type}/','-expr_file',expr_file,
                '-pairs_for_predict_file',network,
                '-geneName_map_file',f'{output_path}/{data_type}/geneMap.txt',
                '-flag_load_from_h5','False','-flag_load_split_batch_pos','True',
                '-TF_divide_pos_file',f'{output_path}/{data_type}/network_divide.txt',
                '-TF_num','2']
        return command
    start_time = time.time()      
    command_train = get_command(output_deepdrim_input_dir, args.expr_file, args.train, data_type='train')
    subprocess.run(command_train)    
    command_valid = get_command(output_deepdrim_input_dir, args.expr_file, args.valid, data_type='valid')
    subprocess.run(command_valid) 
    command_test = get_command(output_deepdrim_input_dir, args.expr_file, args.test, data_type='test')
    subprocess.run(command_test)    
    command_tftest = get_command(output_deepdrim_input_dir, args.expr_file, args.tftest, data_type='TF_test')
    subprocess.run(command_tftest) 

    output_path = f'{output_dir}/output/'
    def get_run_command(args, output_deepdrim_input_dir, output_path, network_path, data_type):
        return ["python","./source/DeepDRIM/DeepDRIM_novalid.py",
                    "-num_batches","2","-data_path",f'{output_deepdrim_input_dir}/{data_type}/version11/',
                    "-output_dir",f'{output_path}/{data_type}/',
                    "-to_predict","True","-weight_path",f'{output_path}/saved_models{args.epochs}/weights.keras',                     
                    "-label_path",network_path]
    
    command_train2 = ["python","./source/DeepDRIM/DeepDRIM_novalid.py",
                    "-num_batches","2","-data_path",f'{output_deepdrim_input_dir}/train/version11/',
                    "-output_dir",output_path,
                    "-cuda",str(args.cuda),"-epoch",str(args.epochs),"-lr",str(args.lr),"-batch_size",str(args.batch_size),
                    "-cross_validation_fold_divide_file", f'{output_deepdrim_input_dir}/cross.txt'
                    ]

    result = subprocess.run(command_train2, capture_output=True, text=True)
    command_valid2 = get_run_command(args, output_deepdrim_input_dir, output_path, args.valid, data_type='valid')
    result2 = subprocess.run(command_valid2, capture_output=True, text=True)
    command_test2 = get_run_command(args, output_deepdrim_input_dir, output_path, args.test, data_type='test')
    result3 = subprocess.run(command_test2, capture_output=True, text=True)
    command_tftest2 = get_run_command(args, output_deepdrim_input_dir, output_path, args.tftest, data_type='TF_test')
    result4 = subprocess.run(command_tftest2, capture_output=True, text=True)
    res = f'{result.stdout}\n{result2.stdout}\n{result3.stdout}\n{result4.stdout}'
    
    def get_auc(data_type, output_path):
        auc = []
        with open(f'{output_path}/{data_type}/auc_result.txt', 'r') as f:
            for line in f:
                line = line.strip().split(':')[1]
                line = float(line)
                line = round(line, 5)
                auc.append(line)
        return auc
    auroc_val, aupr_val = get_auc('valid', output_path)
    auroc_test, aupr_test = get_auc('test', output_path)
    auroc_tftest, aupr_tftest = get_auc('TF_test', output_path)
    res = 'method\tdataset\tauroc\taupr\n'
    res += f'DeepDRIM\tvalid\t{auroc_val}\t{aupr_val}\n'
    res += f'DeepDRIM\ttest\t{auroc_test}\t{aupr_test}\n'       
    res += f'DeepDRIM\ttftest\t{auroc_tftest}\t{aupr_tftest}\n'

    with open(f'{output_dir}/roc.txt', 'w') as f:
        f.write(res)

    end_time = time.time()
    elapsed_time = end_time - start_time   
    res += f'\nElapsed_time\t{elapsed_time:.6f} seconds'
    with open(f'{output_dir}/log.txt','w') as f:
        f.write(res)

if __name__ == '__main__':
    args = get_parser()
    run_DeepDRIM(args)