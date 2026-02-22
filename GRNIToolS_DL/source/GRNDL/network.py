import os 
import numpy as np
import argparse 
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
from joblib import load
from sklearn.metrics import roc_auc_score, average_precision_score
import data_generator
from models import *

def predict_network(args, save_path):

    test_file = args.test
    expr_path = args.expr
    batch_size = args.batch_size

    inputloader, tf_gene_dataset = data_generator.load_expr_data(expr_path, test_file, batch_size, tflist=args.tf, genelist=args.gene)

    model_pth = f'{save_path}/state_dict_model_best.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    if args.model == 'ResNet':
        model = ResNet()
    elif args.model == 'GRNCNN':
        model = GRNCNN()
    
    model = model.to(device)
    model.load_state_dict(torch.load(model_pth, weights_only=True))
    all_link_pred = []    
    model.eval()
    with torch.no_grad():
        for (data, _) in inputloader:
            data = data.to(device)
            output = model(data)
            link_pred = output.sigmoid()
            all_link_pred.extend(link_pred.detach().cpu().numpy().tolist())        

    updated_dataset = []
    lines = tf_gene_dataset.strip().split('\n')
    for line, pred in zip(lines, all_link_pred):
        updated_dataset.append(f"{line}\t{pred[0]:.5f}")
    tf_gene_dataset = '\n'.join(updated_dataset)

    with open(save_path + '/test_network_predict.txt', 'w') as f:
        f.write(tf_gene_dataset)


