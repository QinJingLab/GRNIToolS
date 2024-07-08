import argparse
import subprocess
import os
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=120, help='Number of Epochs for training')
parser.add_argument('--method', type=str, default='GENELINK',
                    help='Determine which task to run. Select from DeepSEM, GENELINK,GRNDL,STGRNS,DeepDRIM.DeepDRIM should run under tensorflow environment')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size used in the training process.')
parser.add_argument('--expr_file', required=False, default='example/E.coli/Deeplearning_data/expression.csv',help='The input scRNA-seq gene expression file.')
parser.add_argument('--cuda', required=False, default="0",type=int, help="The device used in model")
parser.add_argument('--lr', type=float, required=False, default=0.0001, help="The learning rate used in model")
parser.add_argument('--tf_path', required=False, default='example/E.coli/TFlist.txt', help="The path that includes TF name")
parser.add_argument('--gene_path', required=False, default='example/E.coli/genelist.txt', help="The path that includes gene name")
parser.add_argument('--network_path', required=False, default='example/E.coli/Deeplearning_data/train_val_test/train.txt', help="The network indicates that releationship in TFs and genes,use for training")
parser.add_argument('--val_network_path', required=False, default='example/E.coli/Deeplearning_data/train_val_test/val.txt', help="The network indicates that releationship in TFs and genes,use for validation")
parser.add_argument('--test_network_path', required=False, default='example/E.coli/Deeplearning_data/train_val_test/random_test.txt', help="The network indicates that releationship in TFs and genes,use for test")
parser.add_argument('--TF_network_path', required=False, default='example/E.coli/Deeplearning_data/train_val_test/TF_test.txt', help="The network indicates that releationship in TFs and genes,use for TF_test")
parser.add_argument('--output_path', required=False,default='output/test/', help="output dictionary")
args=parser.parse_args()


if not os.path.exists(f'{args.output_path}/{args.method}/'):
    os.makedirs(f'{args.output_path}/{args.method}/')

if args.method == 'DeepSEM':
    subprocess.run(["python","DeepSEM/main.py","--task","non_celltype_GRN","--setting","test","--data_file",args.expr_file,"--save_name",f'{args.output_path}/{args.method}/',
                    "--n_epochs",str(args.epochs),"--batch_size",str(args.batch_size)])

if args.method == 'STGRNS':
    subprocess.run(["python","STGRNS/main1.py","-expression",args.expr_file,"-output_dir",f'{args.output_path}/{args.method}/input/train/',"-data_path",args.network_path])
    subprocess.run(["python","STGRNS/main1.py","-expression",args.expr_file,"-output_dir",f'{args.output_path}/{args.method}/input/test/',"-data_path",args.test_network_path])
    subprocess.run(["python","STGRNS/main1.py","-expression",args.expr_file,"-output_dir",f'{args.output_path}/{args.method}/input/TF_test/',"-data_path",args.test_network_path])
    if not os.path.exists(f'{args.output_path}/{args.method}/output/'):
        os.makedirs(f'{args.output_path}/{args.method}/output/')
    subprocess.run(["python","STGRNS/main2_acc.py","-cuda",str(args.cuda),"-epoch",str(args.epochs),"-lr",str(args.lr),"-batch_size",str(args.batch_size),
                    "-train_data_path",f'{args.output_path}/{args.method}/input/train/',
                    "-test_data_path",f'{args.output_path}/{args.method}/input/test/',
                    "-output_dir",f'{args.output_path}/{args.method}/output/',"-test_gene_pair",args.test_network_path,"-to_predict","False"])
    subprocess.run(["python","STGRNS/main2_acc.py","-cuda",str(args.cuda),
                    "-test_data_path",f'{args.output_path}/{args.method}/input/test/',
                    "-output_dir",f'{args.output_path}/{args.method}/output/test/',"-test_gene_pair",args.test_network_path,
                    "-to_predict","True","-weight_path",f'{args.output_path}/{args.method}/output/model.pth'])
    subprocess.run(["python","STGRNS/main2_acc.py","-cuda",str(args.cuda),
                    "-test_data_path",f'{args.output_path}/{args.method}/input/TF_test/',
                    "-output_dir",f'{args.output_path}/{args.method}/output/TF_test',"-test_gene_pair",args.TF_network_path,
                    "-to_predict","True","-weight_path",f'{args.output_path}/{args.method}/output/model.pth'])
    

if args.method == 'GENELINK':
    subprocess.run(["python","GENELINK/main.py","-cuda",str(args.cuda),"-epochs",str(args.epochs),"-lr",str(args.lr),"-batch_size",str(args.batch_size),
                    "-tf_path",args.tf_path,"-gene_path",args.gene_path,
                    '-network_path',args.network_path,"-valid_path",args.val_network_path,
                    "-random_network_path",args.test_network_path,"-tf_network_path",args.TF_network_path,
                    "-output_path",f'{args.output_path}/{args.method}/output/',"-expr_path",args.expr_file])
    

##GRNDL's expression is B.txt
if args.method == 'GRNCNN':
    subprocess.run(["python","GRNDL/main_TF.py","-cuda",str(args.cuda),"-epoch",str(args.epochs),"-lr",str(args.lr),"-batch_size",str(args.batch_size),
                    "-tf_path",args.tf_path,"-gene_path",args.gene_path,
                    '-network_path',args.network_path,"-valid_network_path",args.val_network_path,
                    "-random_network_path",args.test_network_path,"-tf_network_path",args.TF_network_path,
                    "-output_path",f'{args.output_path}/{args.method}/output/',"-expr_path",f'{os.path.dirname(args.expr_file)}/../B.txt',"-model_list","GRNCNN"])    
    
if args.method == 'GRNTrans':
    subprocess.run(["python","GRNDL/main_TF.py","-cuda",str(args.cuda),"-epoch",str(args.epochs),"-lr",str(args.lr),"-batch_size",str(args.batch_size),
                    "-tf_path",args.tf_path,"-gene_path",args.gene_path,
                    '-network_path',args.network_path,"-valid_network_path",args.val_network_path,
                    "-random_network_path",args.test_network_path,"-tf_network_path",args.TF_network_path,
                    "-output_path",f'{args.output_path}/{args.method}/output/',"-expr_path",f'{os.path.dirname(args.expr_file)}/../B.txt',"-model_list","Trans"])

if args.method == 'GRNResNet':
    subprocess.run(["python","GRNDL/main_TF.py","-cuda",str(args.cuda),"-epoch",str(args.epochs),"-lr",str(args.lr),"-batch_size",str(args.batch_size),
                    "-tf_path",args.tf_path,"-gene_path",args.gene_path,
                    '-network_path',args.network_path,"-valid_network_path",args.val_network_path,
                    "-random_network_path",args.test_network_path,"-tf_network_path",args.TF_network_path,
                    "-output_path",f'{args.output_path}/{args.method}/output/',"-expr_path",f'{os.path.dirname(args.expr_file)}/../B.txt',"-model_list","ResNet18"])    
#DeepDRIM should run in tf1.5
if args.method == 'DeepDRIM':

    def cheat(gene_path,network_path,type):
        with open(gene_path,'r')as file:
            lines=file.readlines()
        if not os.path.exists(f'{args.output_path}/{args.method}/input/{type}/'):
            os.makedirs(f'{args.output_path}/{args.method}/input/{type}/')
        geneMap=f'{args.output_path}/{args.method}/input/{type}/geneMap.txt'
        with open(geneMap,'w')as file:
            for line in lines:
                line = line.strip()
                new_line = f"{line}\t{line}\n"  # 在每行后面添加一个副本
                file.write(new_line)

        with open(network_path,'r')as file:
            lines=file.readlines() 
            column_length=len(lines)    
        a=column_length
        divide=f'{args.output_path}/{args.method}/input/{type}/network_divide.txt'
        with open(divide,'w') as file:
            file.write("0\n") 
            file.write(f"{int(a/2)}\n")
            file.write(f"{a}\n")  


    cheat(args.gene_path,args.network_path,'train')
    cheat(args.gene_path,args.val_network_path,'val')
    cheat(args.gene_path,args.test_network_path,'test')
    cheat(args.gene_path,args.TF_network_path,'TF_test')
    with open(f'{args.output_path}/{args.method}/input/cross.txt','w') as file:
        file.write("0\n") 
        file.write("1\n")
        file.write("2\n") 
    subprocess.run(["python","DeepDRIM/generate_input_realdata.py",
                    "-out_dir",f'{args.output_path}/{args.method}/input/train/','-expr_file',args.expr_file,
                    '-pairs_for_predict_file',args.network_path,
                    '-geneName_map_file',f'{args.output_path}/{args.method}/input/train/geneMap.txt',
                    '-flag_load_from_h5','False','-flag_load_split_batch_pos','True',
                    '-TF_divide_pos_file',f'{args.output_path}/{args.method}/input/train/network_divide.txt',
                    '-TF_num','2'])    
    subprocess.run(["python","DeepDRIM/generate_input_realdata.py",
                    "-out_dir",f'{args.output_path}/{args.method}/input/val/','-expr_file',args.expr_file,
                    '-pairs_for_predict_file',args.val_network_path,
                    '-geneName_map_file',f'{args.output_path}/{args.method}/input/val/geneMap.txt',
                    '-flag_load_from_h5','False','-flag_load_split_batch_pos','True',
                    '-TF_divide_pos_file',f'{args.output_path}/{args.method}/input/val/network_divide.txt',
                    '-TF_num','2']) 
    subprocess.run(["python","DeepDRIM/generate_input_realdata.py",
                    "-out_dir",f'{args.output_path}/{args.method}/input/test/','-expr_file',args.expr_file,
                    '-pairs_for_predict_file',args.test_network_path,
                    '-geneName_map_file',f'{args.output_path}/{args.method}/input/test/geneMap.txt',
                    '-flag_load_from_h5','False','-flag_load_split_batch_pos','True',
                    '-TF_divide_pos_file',f'{args.output_path}/{args.method}/input/test/network_divide.txt',
                    '-TF_num','2'])  
    subprocess.run(["python","DeepDRIM/generate_input_realdata.py",
                    "-out_dir",f'{args.output_path}/{args.method}/input/TF_test/','-expr_file',args.expr_file,
                    '-pairs_for_predict_file',args.TF_network_path,
                    '-geneName_map_file',f'{args.output_path}/{args.method}/input/TF_test/geneMap.txt',
                    '-flag_load_from_h5','False','-flag_load_split_batch_pos','True',
                    '-TF_divide_pos_file',f'{args.output_path}/{args.method}/input/TF_test/network_divide.txt',
                    '-TF_num','2'])  

    subprocess.run(["python","DeepDRIM/DeepDRIM_novalid.py",
                    "-num_batches","2","-data_path",f'{args.output_path}/{args.method}/input/train/version11/',
                    "-output_dir",f'{args.output_path}/{args.method}/output/',
                    "-cuda",str(args.cuda),"-epoch",str(args.epochs),"-lr",str(args.lr),"-batch_size",str(args.batch_size),
                    "-cross_validation_fold_divide_file", f'{args.output_path}/{args.method}/input/cross.txt'
                    ])  
    
    subprocess.run(["python","DeepDRIM/DeepDRIM_novalid.py",
                    "-num_batches","2","-data_path",f'{args.output_path}/{args.method}/input/val/version11/',
                    "-output_dir",f'{args.output_path}/{args.method}/output/val/',
                    "-to_predict","True","-weight_path",f'{args.output_path}/{args.method}/output/saved_models{args.epochs}/weights.hdf5',                     
                    "-label_path",args.val_network_path])
    
    subprocess.run(["python","DeepDRIM/DeepDRIM_novalid.py",
                    "-num_batches","2","-data_path",f'{args.output_path}/{args.method}/input/test/version11/',
                    "-output_dir",f'{args.output_path}/{args.method}/output/test/',
                    "-to_predict","True","-weight_path",f'{args.output_path}/{args.method}/output/saved_models{args.epochs}/weights.hdf5',                     
                    "-label_path",args.test_network_path])

    subprocess.run(["python","DeepDRIM/DeepDRIM_novalid.py",
                    "-num_batches","2","-data_path",f'{args.output_path}/{args.method}/input/TF_test/version11/',
                    "-output_dir",f'{args.output_path}/{args.method}/output/TF_test/',
                    "-to_predict","True","-weight_path",f'{args.output_path}/{args.method}/output/saved_models{args.epochs}/weights.hdf5',                     
                    "-label_path",args.TF_network_path])
                