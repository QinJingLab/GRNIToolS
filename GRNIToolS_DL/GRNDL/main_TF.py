import os 
from joblib import load
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
import data_generator_TF
import train
import evaluate 
from models import *
import argparse
import sys
parser = argparse.ArgumentParser(description="example")
import time

from sklearn.metrics import roc_auc_score, average_precision_score
parser.add_argument('-cuda', required=False, default="0",type=int, help="The device used in model")
parser.add_argument('-epoch', type=int, required=False, default=None, help="The epoch used in model")
parser.add_argument('-lr', type=float, required=False, default=None, help="The learning rate used in model")
parser.add_argument('-batch_size', type=int, required=False, default="64", help="The batch size used in model")
parser.add_argument('-expr_path', required=False, default='example/data/scRNA-seq_pCRISPR/B.txt', help="The path that includes train data")
parser.add_argument('-tf_path', required=False, default='example/data/scRNA-seq_pCRISPR/TF_2list.txt', help="The path that includes train data")
parser.add_argument('-gene_path', required=False, default='example/data/scRNA-seq_pCRISPR/genelist_2.txt', help="The path that includes test data")
parser.add_argument('-network_path', required=False, default='example/data/scRNA-seq_pCRISPR/train_val_test/train.txt', help="The output dir")
parser.add_argument('-valid_network_path', required=False, default='example/data/scRNA-seq_pCRISPR/train_val_test/val.txt', help="The test gene pair")
parser.add_argument('-random_network_path', required=False,default='example/data/scRNA-seq_pCRISPR/train_val_test/test.txt')
parser.add_argument('-tf_network_path', required=False,default='example/data/scRNA-seq_pCRISPR/train_val_test/TF_test.txt', help="The path for a trained model.")
parser.add_argument('-output_path', required=False,default='output/PC/', help="output dictionary")
parser.add_argument('-model_list', nargs='+',required=False,default=['GRNCNN','ResNet18','STGRNS','Trans','GRNGCN'], help="4 models can be choosed,default=['GRNCNN','ResNet18','STGRNS','Trans']")
parser.add_argument('-l1', type=float,required=False,default='0', help="l1 regularization index, default=0")
parser.add_argument('-batch_first' ,required=False,default=True, help="if trans use batchfirst")
args = parser.parse_args()


if __name__ == '__main__':
    model_list = args.model_list
    #model_list = ['GRNGCN']
    # step1 input data and save date to trainning
    #tf_path = f'example/data/ecoli/ecoli_tf_names.tsv'
    #gene_path = f'example/data/ecoli/ecoli_gene_names.tsv'
    #expr_path = f'example/data/ecoli/ecoli_data.tsv'
    #network_path = f'example/data/ecoli/ecoli_HGS.txt'
    tf_path = args.tf_path
    gene_path = args.gene_path
    expr_path = args.expr_path
    network_path = args.network_path
    valid_network_path=args.valid_network_path
    random_network_path=args.random_network_path
    tf_network_path=args.tf_network_path
    batch_size=args.batch_size

    for model_idx in model_list:
        start_time=time.time()
        save_path = f'{args.output_path}/train_res_{model_idx}/'
        save_train_path = f'{args.output_path}/train_res_{model_idx}/'
        save_evaluate_path = f'{args.output_path}/evaluate_res_{model_idx}'
        model_pth = f'{save_train_path}/state_dict_model_best.pth'


        if not os.path.exists(save_evaluate_path):
            os.makedirs(save_evaluate_path)  
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    

        train_dataloader,val_dataloader = data_generator_TF.load_save_data(save_path,tf_path,gene_path,expr_path,network_path,model_idx=model_idx,type='train',args=args)  
        #valid_dataloader = data_generator_TF.load_save_data(save_path,tf_path,gene_path,expr_path,valid_network_path,model_idx=model_idx,type='TF1',args=args) 
        #test1_dataloader = data_generator_TF.load_save_data(save_path,tf_path,gene_path,expr_path,random_network_path,model_idx=model_idx,type='TF2',args=args) 
        test2_dataloader = data_generator_TF.load_save_data(save_path,tf_path,gene_path,expr_path,tf_network_path,model_idx=model_idx,type='TF3',args=args) 
        train_eval_dataloader = data_generator_TF.load_save_data(save_path,tf_path,gene_path,expr_path,network_path,model_idx=model_idx,type='train_eval')#如果type是train的话会打乱顺序进行训练。所以重新生成顺序以便评估


        # step2 trainning
        save_train_path = f'{args.output_path}/train_res_{model_idx}/'
        if not os.path.exists(save_train_path):
            os.makedirs(save_train_path)
        # train_dataloader = load(f'{save_path}/train_dataloder')
        # valid_dataloader = load(f'{save_path}/valid_dataloder')


        train.trainning(save_path, save_train_path, train_dataloader, val_dataloader, model_idx=model_idx,args=args)

    # step3 evaluating
        
        auc1,aupr1=evaluate.evaluating(train_eval_dataloader, save_evaluate_path, 'train_eval', model_pth, model_idx = model_idx,args=args)
        auc2,aupr2=evaluate.evaluating(val_dataloader, save_evaluate_path, 'val', model_pth, model_idx = model_idx,args=args)
        #auc3,aupr3=evaluate.evaluating(valid_dataloader, save_evaluate_path, 'TF1', model_pth, model_idx = model_idx,args=args)
        #auc4,aupr4=evaluate.evaluating(test1_dataloader, save_evaluate_path, 'TF2', model_pth, model_idx = model_idx,args=args)
        auc5,aupr5=evaluate.evaluating(test2_dataloader, save_evaluate_path, 'TF3', model_pth, model_idx = model_idx,args=args)
        end_time=time.time()
        execution_time=(end_time - start_time)/60
        timefile=save_train_path+'time.txt'
        with open(timefile,'w',newline='',encoding='utf-8')as file:
            file.write(str(execution_time))
    
        with open(f'{save_evaluate_path}/auc_result.txt', 'w', newline='', encoding='utf-8') as file:
            file.write('train_eval_auc={:.5f}\ttrain_eval_aupr={:.5f}\n'.format(auc1, aupr1))
            file.write('val_auc={:.5f}\tval_aupr={:.5f}\n'.format(auc2, aupr2))
            #file.write('TF1_auc={:.5f}\tTF1_aupr={:.5f}\n'.format(auc3, aupr3))
            #file.write('TF2_auc={:.5f}\tTF2_aupr={:.5f}\n'.format(auc4, aupr4))
            file.write('TF3_auc={:.5f}\tTF3_aupr={:.5f}\n'.format(auc5, aupr5))
            file.write('time={:.5f}'.format(execution_time))
       # with open(f'{save_evaluate_path}/auc_result.txt','w',newline='',encoding='utf-8')as file:
       #     file.write('train_eval_auc='+str(auc1)+'\t'+'train_eval_aupr='+str(aupr1)+'\n')
       #     file.write('val_auc='+str(auc2)+'\t'+'val_aupr='+str(aupr2)+'\n')
       #     file.write('random_test_auc='+str(auc3)+'\t'+'random_test_aupr='+str(aupr3)+'\n')
       #     file.write('TF_test_auc='+str(auc4)+'\t'+'TF_test_aupr='+str(aupr4)+'\n')


