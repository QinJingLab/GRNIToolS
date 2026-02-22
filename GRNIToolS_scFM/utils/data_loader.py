import os
import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch.utils.data.dataset as Dataset
import torch_geometric.transforms as Tr
from torch_geometric.data import Data  

def get_dataset(gene_embeddings, df):
    x_list = []
    y_list = []
    for i, row in df.iterrows():
        tf, gene = row['gene_1'].upper(), row['gene_2'].upper()  
        if tf in gene_embeddings and gene in gene_embeddings:
            # 处理Tensor类型输入
            tf_embed = gene_embeddings[tf]
            gene_embed = gene_embeddings[gene]
            
            # 如果是PyTorch Tensor则转numpy数组
            if isinstance(tf_embed, torch.Tensor):
                tf_embed = tf_embed.detach().cpu().numpy()
            if isinstance(gene_embed, torch.Tensor):
                gene_embed = gene_embed.detach().cpu().numpy()
            
            # 拼接特征向量
            x_list.append(np.concatenate([tf_embed, gene_embed]))
            y_list.append(row['label'])
    
    # 转换为numpy数组并调整形状
    x_list = np.array(x_list)
    y_list = np.array(y_list)
    x_list = x_list.reshape(x_list.shape[0], -1)
    return x_list, y_list

def get_dataset_pre(gene_embeddings, df):
    x_list = []
    tf_list = []
    gene_list = []
    for i, row in df.iterrows():
        tf, gene = row['gene_1'].upper(), row['gene_2'].upper()  
        if tf in gene_embeddings and gene in gene_embeddings:
            tf_list.append(tf)
            gene_list.append(gene)
            # 处理Tensor类型输入
            tf_embed = gene_embeddings[tf]
            gene_embed = gene_embeddings[gene]
            
            # 如果是PyTorch Tensor则转numpy数组
            if isinstance(tf_embed, torch.Tensor):
                tf_embed = tf_embed.detach().cpu().numpy()
            if isinstance(gene_embed, torch.Tensor):
                gene_embed = gene_embed.detach().cpu().numpy()
            
            # 拼接特征向量
            x_list.append(np.concatenate([tf_embed, gene_embed]))
    
    # 转换为numpy数组并调整形状
    x_list = np.array(x_list)
    x_list = x_list.reshape(x_list.shape[0], -1)
    return x_list, tf_list, gene_list
     

class subDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    #返回数据集大小
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        data = torch.FloatTensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        return data, label

def encode_dl(X_array, y_array, dl_method, batch_size = 64, shuffle = True, worker = 2, drop_last = False): # 

    def encode_data(mat_feature, y_array):
        pos_matrix = []
        neg_matrix = []
        for i in range(0, mat_feature.shape[0]):
            tf_gene_featrue = mat_feature[i]
            n_feature = tf_gene_featrue.shape[0] // 2
            tf_feature, gene_feature = tf_gene_featrue[:n_feature], tf_gene_featrue[n_feature:]
            gap = 30
            gap2 = 2 * gap
            tf_gene_feature = []
            for j in range(0, len(tf_feature), gap):
                feature = []
                x = tf_feature[j:j+gap]
                y = gene_feature[j:j+gap]
                x = np.pad(x, (0, (gap - len(x))), mode='constant', constant_values=0)
                y = np.pad(y, (0, (gap - len(y))), mode='constant', constant_values=0)
                feature.extend(x)
                feature.extend(y)
                feature = np.asanyarray(feature)
                if len(feature) == gap2:
                    tf_gene_feature.append(feature)
                if len(tf_gene_feature) >= gap2:
                    break
            if len(tf_gene_feature) < gap2:
                for _ in range((gap2 - len(tf_gene_feature))):
                    tf_gene_feature.append(np.asanyarray([0] * gap2))
            tf_gene_feature = np.asanyarray(tf_gene_feature)
            if int(y_array[i]) == 1:
                pos_matrix.append(np.asanyarray([tf_gene_feature]))
            elif int(y_array[i]) == 0:
                neg_matrix.append(np.asanyarray([tf_gene_feature]))

        return pos_matrix, neg_matrix    
  
    label1_matrix, label0_matrix = encode_data(X_array, y_array)
    label1 = [[1]] * len(label1_matrix)
    label0 = [[0]] * len(label0_matrix) 
    data_x = label1_matrix + label0_matrix
    data_y = label1 + label0
    t_dataset = subDataset(data_x, data_y)
    t_dataloader = DataLoader(t_dataset, batch_size= batch_size, shuffle = shuffle, drop_last=drop_last, num_workers= worker)
    return t_dataloader

def encode_mlp(x_array, y_array, batch_size, shuffle, drop_last=False):
    x_tensor = torch.tensor(x_array).float()
    y_tensor = torch.tensor(y_array).float()
    y_tensor = y_tensor.view(-1,1) 
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def encode_trans(x_array, y_array, batch_size, shuffle, drop_last=False):
    def split_array(x_array):
        array1 = []
        array2 = []
        for sub in x_array:
            mid = len(sub) // 2
            array1.append(sub[:mid])
            array2.append(sub[mid:])
        return array1, array2
    part1, part2 = split_array(x_array)
  # 将列表转换为 Tensor
    tflist_tensor = torch.as_tensor(np.array(part1), dtype=torch.float32)
    genelist_tensor = torch.as_tensor(np.array(part2), dtype=torch.float32)
    y_tensor = torch.as_tensor(np.array(y_array), dtype=torch.float32)
    # y_tensor = y_tensor.view(-1,1) 
    # 创建 TensorDataset
    dataset = TensorDataset(tflist_tensor, genelist_tensor, y_tensor)
    # 创建 DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader

def encode_gcn(feature, df, dict_gene_id):
    neg_edge = [[],[]]
    pos_edge = [[],[]]
    neg_label, pos_label = [], []
    for index, row in df.iterrows():  
        tf, gene, label, score = row
        tf, gene, label, score = str(tf).upper(), str(gene).upper(), str(label).upper(), str(score)
        if tf in dict_gene_id and gene in dict_gene_id:
            tf, gene = dict_gene_id[tf], dict_gene_id[gene]
        else:
            continue
        if label == '1':
            pos_edge[0].append(tf)
            pos_edge[1].append(gene)
            pos_label.append(1)
        elif label == '0':
            neg_edge[0].append(tf)
            neg_edge[1].append(gene)
            neg_label.append(0)
    pos_edge = np.array(pos_edge)
    neg_edge = np.array(neg_edge)
    label = np.array(pos_label + neg_label)
    pos_edge = torch.tensor(pos_edge)
    neg_edge = torch.tensor(neg_edge)
    label = torch.tensor(label).float()
    data = Data(x=feature, pos_edge_index = pos_edge, neg_edge_index = neg_edge, label = label)
    return data

