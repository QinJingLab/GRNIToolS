import numpy as np
import re
import os
import torch
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
import pandas as pd



class subDataset(Dataset.Dataset):

    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.FloatTensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        return data, label


def encode_data(network_path, dict_expr):
    df = pd.read_csv(
        network_path,
        sep='\t',
        header=None,
        names=['tf', 'gene', 'label'],
        dtype={'tf': str, 'gene': str, 'label': int}
    )
    df = df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
    df = df.dropna()
    tf_gene_dataset = ''
    total_matrix = []
    labels = []
    for row in df.iterrows():
        tf = row[1]['tf']
        gene = row[1]['gene']
        label = int(row[1]['label'])
        labels.append(label)
        if tf in dict_expr.keys() and gene in dict_expr.keys():
            tf_data = dict_expr[tf]
            gene_data = dict_expr[gene]
            tf_gene_dataset += f"{tf}\t{gene}\t{label}\t{tf_data}\t{gene_data}\n"
            gap = 30
            tf_gene_feature = []
            for i in range(0, len(tf_data), gap):
                feature = []
                x = tf_data[i:i+gap]
                y = gene_data[i:i+gap]
                feature.extend(x)
                feature.extend(y)
                feature = np.asanyarray(feature)
                if len(feature) == 2 * gap:
                    tf_gene_feature.append(feature)
                if len(tf_gene_feature) >= 60:
                    break
            if len(tf_gene_feature) < 60:
                for j in range((gap*2-len(tf_gene_feature))):
                    tf_gene_feature.append(np.asanyarray([0] * 60))
            tf_gene_feature = np.asanyarray(tf_gene_feature)
            total_matrix.append(np.asanyarray([tf_gene_feature]))
    return total_matrix, labels

def encode_data2net(tflist, genelist, dict_expr):
    tf_gene_dataset = ''
    total_matrix = []
    for tf in tflist:
        for gene in genelist:
            if tf in dict_expr.keys() and gene in dict_expr.keys():
                tf_data = dict_expr[tf]
                gene_data = dict_expr[gene]
                tf_gene_dataset += f"{tf}\t{gene}\n"
                gap = 30
                tf_gene_feature = []
                for i in range(0, len(tf_data), gap):
                    feature = []
                    x = tf_data[i:i+gap]
                    y = gene_data[i:i+gap]
                    feature.extend(x)
                    feature.extend(y)
                    feature = np.asanyarray(feature)
                    if len(feature) == 2 * gap:
                        tf_gene_feature.append(feature)
                    if len(tf_gene_feature) >= 60:
                        break
                if len(tf_gene_feature) < 60:
                    for j in range((gap*2-len(tf_gene_feature))):
                        tf_gene_feature.append(np.asanyarray([0] * 60))
                tf_gene_feature = np.asanyarray(tf_gene_feature)
                total_matrix.append(np.asanyarray([tf_gene_feature]))
    return total_matrix, tf_gene_dataset

def get_matrix(mat_expr_path,gene_list):  # return hash gene-expr, cell number
    dict_expr = {}
    with open(mat_expr_path, 'r') as f:
        for i, line in enumerate(f):
            if i == 0: continue
            line = line.strip().split(',')
            line = [float(ch) for ch in line[1:]]
            dict_expr[gene_list[i-1]] = line
    cell_num = len(line)
    return dict_expr, cell_num 

def get_normalized_expr(expr_path, gene_list):   #log10 normalized， 0 trans to -2 # return numpy matrix, row: genes, col: cells
    dict_expr, cell_num = get_matrix(expr_path, gene_list)
    expr_mat = np.zeros((len(dict_expr), cell_num))
    index_row = 0
    for gene in dict_expr:
        dict_expr[gene] = np.log10(np.array(dict_expr[gene]) + 10 ** -2)
        expr_mat[index_row] = dict_expr[gene]
        index_row += 1
    return expr_mat, dict_expr

def get_tf_gene_list(expr_path):  # return tf_list gene_list
    df = pd.read_csv(expr_path,index_col=0)
    gene_list = df.index.tolist()
    gene_list = [i.lower() for i in gene_list]
    return gene_list


def data_processing(network_path, expr_mat, dict_expr, batch_size, worker, model_name, shuffle = True):
    total_matrix, labels = encode_data(network_path, dict_expr)
    labels = [[i] for i in labels]
    t_dataset = subDataset(total_matrix, labels)
    return t_dataset
    # t_dataloader = DataLoader.DataLoader(t_dataset, batch_size= batch_size, shuffle = shuffle, drop_last=False, num_workers= worker)
    # return t_dataloader


def load_data(expr_path, train_file, valid_file, test_file, tftest_file, batch_size, model_name):
    worker = 2
    gene_list = get_tf_gene_list(expr_path)
    expr_mat, dict_expr = get_normalized_expr(expr_path,gene_list)
    train_data = data_processing(train_file, expr_mat, dict_expr, batch_size, worker, model_name, shuffle = True)
    valid_data = data_processing(valid_file, expr_mat, dict_expr, batch_size, worker, model_name, shuffle = True)
    test_data = data_processing(test_file, expr_mat, dict_expr, batch_size, worker, model_name, shuffle = False)
    tftest_data = data_processing(tftest_file, expr_mat, dict_expr, batch_size, worker, model_name, shuffle = False)

    return train_data, valid_data, test_data, tftest_data



def load_expr_data(expr_path, test_file, batch_size, tflist='', genelist=''):
    worker = 2
    gene_list = get_tf_gene_list(expr_path)
    expr_mat, dict_expr = get_normalized_expr(expr_path,gene_list)
    if tflist and genelist:
        tf = pd.read_csv(tflist, delimiter='\t', header=None) 
        tf_list = tf.iloc[:, 0].tolist()
        gene = pd.read_csv(genelist, delimiter='\t', header=None) 
        gene_list = gene.iloc[:, 0].tolist()
    else:
        df = pd.read_csv(
            test_file,
            sep='\t',
            header=None,
            names=['tf', 'gene', 'label'],
            dtype={'tf': str, 'gene': str, 'label': int}
        )
        df = df.apply(lambda x: x if x.dtype == "object" else x)
        df = df.dropna()        
        tf_list = df['tf'].drop_duplicates().tolist()
        gene_list = df['gene'].drop_duplicates().tolist()

    total_matrix, tf_gene_dataset = encode_data2net(tf_list, gene_list, dict_expr)
    labels =  [[1]] * len(total_matrix)
    t_dataset = subDataset(total_matrix, labels)
    t_dataloader = DataLoader.DataLoader(t_dataset, batch_size= batch_size, shuffle = False, drop_last=False, num_workers= worker)
    return t_dataloader, tf_gene_dataset