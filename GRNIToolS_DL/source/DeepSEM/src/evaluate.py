import pandas as pd
import re
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import random 
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
 
# grn
# inferelator 取前3列为grn; pmf-grn为邻接矩阵; 读取其他矩阵，均有header

def get_parser():
    parser = argparse.ArgumentParser(description='Inferelator')
    parser.add_argument('--hgs', type=str, required=True, help='gold standard file')
    parser.add_argument('--grn', type=str, required=True, help='gene regulatory network file')
    parser.add_argument('--output', type=str, required=True, help='Path to save the results')
    parser.add_argument('--method', type=str, required=False, default= 'DL', help='Method you used for GRN inferred')
    parser.add_argument('--dataset', type=str, required=False, default= 'test', help='Dataset you used for GRN inferred')
    return parser.parse_args()

def get_data(hgs_file, grn_file):

    grn_file_suffix = grn_file.split('/')[-1].split('.')[-1]
    hgs = pd.read_csv(hgs_file, header=0)
    if hgs.shape[1] == 1:
        hgs = pd.read_csv(hgs_file, header=0, sep='\t', index_col= None)

    if re.search('gz', grn_file_suffix):
        grn = pd.read_csv(grn_file, compression='gzip', sep='\t')
    else:
        grn = pd.read_csv(grn_file, header=0, index_col= None)

    if grn.shape[1] == 1:
        grn = pd.read_csv(grn_file, header=0, sep='\t', index_col= None)
    if grn.shape[1] == 13:
        grn = grn.iloc[:, :3]
        hgs = pd.read_csv(hgs_file, header=0, index_col=0, sep ='\t')
    if grn.shape[1] > 3:
        grn = pd.read_csv(grn_file, header=0, sep='\t',index_col=0)  #行为TF，列为gene的adj mat
        hgs = pd.read_csv(hgs_file, header=0, index_col=0, sep ='\t')

    return grn, hgs

def mat_pl(grn, hgs):
    if grn.shape[1] > 3:
        edges = []
        for from_gene, row in grn.items():
            for to_gene, weight in row.items():
                if weight != 0:
                    edges.append({'TF': from_gene, 'gene': to_gene, 'weight': abs(weight)})
        grn = pd.DataFrame(edges)

    grn.columns = ['TF','Gene','Weight']
    grn['TF'] = grn['TF'].astype(str)
    grn['TF'] = grn['TF'].str.upper()
    grn['Gene'] = grn['Gene'].astype(str)
    grn['Gene'] = grn['Gene'].str.upper()
    grn.columns = ['TF','Gene','Weight']
    grn['TF'] = grn['TF'].str.upper()
    grn['Gene'] = grn['Gene'].str.upper()
    hgs.index = [i.upper() for i in hgs.index]
    hgs.columns = [i.upper() for i in hgs.columns]

    grn_mat = pd.DataFrame(float(0), index=hgs.index, columns=hgs.columns)
    for row in grn.iterrows():
        tf, gene, weight = row[1]['TF'], row[1]['Gene'], row[1]['Weight']
        if tf != gene:
            if tf in hgs.index and gene in hgs.columns:
                grn_mat.loc[tf,gene] = abs(float(weight))

    bool_index = (hgs == 1) & (grn_mat != 0)
    match_id = bool_index.stack()[bool_index.stack()].index.get_level_values(0), bool_index.stack()[bool_index.stack()].index.get_level_values(1)
    pos_score = [grn_mat.loc[match_id[0][i],match_id[1][i]] for i in range(len(match_id[0]))]
    bool_index = (hgs == 0) & (grn_mat != 0)
    match_id = bool_index.stack()[bool_index.stack()].index.get_level_values(0), bool_index.stack()[bool_index.stack()].index.get_level_values(1)
    neg_score = [grn_mat.loc[match_id[0][i],match_id[1][i]] for i in range(len(match_id[0]))]
 
    random.seed(42)
    if len(pos_score) > len(neg_score):
        n = len(neg_score)
        pos_index = random.sample(range(len(pos_score)), n)
        pos_score = [pos_score[i] for i in pos_index]
    else:
        n = len(pos_score)
        neg_index = random.sample(range(len(neg_score)), n)
        neg_score = [neg_score[i] for i in neg_index]
    pred = pos_score + neg_score
    true_label = [1] * len(pos_score) + [0] * len(neg_score)
    return pred, true_label

def col_pl(grn, hgs):

    hgs.columns = ['TF','Gene','Weight']
    hgs['TF'] = hgs['TF'].astype(str)
    hgs['TF'] = hgs['TF'].str.upper()
    hgs['Gene'] = hgs['Gene'].astype(str)
    hgs['Gene'] = hgs['Gene'].str.upper()
    hgs['Pair'] = hgs['TF'] + "_" + hgs['Gene']

    grn.columns = ['TF','Gene','Weight']
    grn['TF'] = grn['TF'].astype(str)
    grn['TF'] = grn['TF'].str.upper()
    grn['Gene'] = grn['Gene'].astype(str)
    grn['Gene'] = grn['Gene'].str.upper()
    grn.index = grn['TF'] + "_" + grn['Gene']

    pred, true_label = [], []

    for row in hgs.iterrows():
        pair, label = row[1]['Pair'], row[1]['Weight']
        if pair in grn.index:
            pred.append(abs(float(grn.loc[pair]['Weight'])))
            true_label.append(int(label))

    return pred, true_label


def cal_auc(pred, true_label, output_dir, method, dataset):

    fpr, tpr, thresholds = roc_curve(true_label, pred)
    roc_auc = auc(fpr, tpr).round(5)

    precision, recall, _ = precision_recall_curve(true_label, pred)
    pr_auc = average_precision_score(true_label, pred).round(5)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    roc_name = f'{output_dir}/{dataset}_roc_curve.png'
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(roc_name, dpi=300, bbox_inches='tight')  

    pr_name = f'{output_dir}/{dataset}_pr_curve.png'
    plt.figure()
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall curve')
    plt.legend(loc="best")
    plt.savefig(pr_name, dpi=300, bbox_inches='tight') 
    plt.close() 

    auc_name = f'{output_dir}/{dataset}_auc.txt'
    res = 'method\tdataset\tauroc\taupr\n'
    res += f'{method}\t{dataset}\t{roc_auc}\t{pr_auc}\n'
    with open(auc_name, 'w') as f:
        f.write(res)

if __name__ == '__main__':
    args = get_parser()
    output_dir = args.output
    hgs_file = args.hgs
    grn_file = args.grn
    method = args.method
    dataset = args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    grn, hgs = get_data(hgs_file, grn_file)

    pred, true_label = col_pl(grn, hgs)
    cal_auc(pred, true_label, output_dir, method, dataset)

