import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
from train import *
from util.utils import div_list
import time
import os
# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('model', 'DeepRIG', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('input_path', 'output_data_pre/mESC/SC-PC/DeepRIG', 'Input data path')
flags.DEFINE_string('output_path', './output/', 'Output data path')
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('cv', 1, 'Folds for cross validation.')
flags.DEFINE_integer('epochs', 500, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.7, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('ratio', 1, 'Ratio of negetive samples to positive samples.')
flags.DEFINE_integer('dim', 300, 'The size of latent factor vector.')

def computCorr(data, t = 0.0):

    genes = data.columns
    corr = data.corr(method = "spearman")
    adj = np.array(abs(corr.values))

    return adj

def prepareData(FLAGS, data_path, label_path, reverse_flags = 0):
    ###transpose for six datasets of BEELINE
    print("Read data completed! Normalize data now!")    # Reading data from disk
    label_file = pd.read_csv(label_path, header=0, sep = ',')
    data = pd.read_csv(data_path, header=0, index_col = 0).T    
    data = data.transform(lambda x: np.log(x + 1))
    data = data.loc[:, (data != 0).any(axis=0)]  
    print("Data normalized and logged!")
    
    var_names = data.columns.tolist()
    TF = set(label_file['Gene1'])
    gene = set(label_file['Gene2'])
    var_names2 = list(set(list(TF) + list(gene)))
    var_names3 = list(set(var_names + var_names2))
    existing_columns = [col for col in var_names3 if col in data.columns]
    var_names = existing_columns
    data = data[existing_columns]  # 筛选有效列
    # Adjacency matrix transformation
    labels = []    # tfid geneid label 
    if reverse_flags == 0:
        # var_names = list(data.columns)
        num_genes = len(var_names)
        AM = np.zeros([num_genes, num_genes])  # adj matrix
        for row_index, row in label_file.iterrows():   # 0 ; Gene1  AHR Gene2 110032F04RIK
            if int(row[2]) == 1:
                if row[0] in var_names and row[1] in var_names:
                    AM[var_names.index(row[0]), var_names.index(row[1])] = 1    # 矩阵AM, gene-gene ： 1
                    label_triplet = []
                    label_triplet.append(var_names.index(row[0]))
                    label_triplet.append(var_names.index(row[1]))
                    label_triplet.append(1)
                    labels.append(label_triplet)

    labels = np.array(labels)     # 使用金标准获取的adj矩阵
    print("Start to compute correlations between genes!")
    
    adj = computCorr(data)  # spearman 相关性adj
    node_feat = data.T.values  # 每个基因的exprssion
    return labels, adj, AM, var_names, TF, node_feat



# Preparing data for training

input_path = FLAGS.input_path
output_path = FLAGS.output_path
train_path = f'{input_path}/train.csv'
val_path = f'{input_path}/val.csv'
test_path = f'{input_path}/test.csv'
tftest_path = f'{input_path}/tftest.csv'
data_file = f'{input_path}/expressions.csv'
label_file = f'{input_path}/network.csv'
result_path = f'{output_path}'
os.makedirs(result_path, exist_ok=True) 
reverse_flags = 0   ###whether label file exists reverse regulations, 0 for DeepSem data, 1 for CNNC data
labels, adj, AM, gene_names, TF, node_feat = prepareData(FLAGS, data_file, label_file, reverse_flags)
train(FLAGS, adj, node_feat, train_path, val_path, test_path, tftest_path, labels, reverse_flags, gene_names, TF, result_path)
