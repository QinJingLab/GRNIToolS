import numpy as np
import random
import os
import torch
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
import torch_geometric.transforms as Tr
from joblib import dump
from torch_geometric.data import Data  
from param import *
from torch_geometric.data import Batch

def get_tf_gene_list(tf_path,gene_path):  # return tf_list gene_list
    with open(tf_path, 'r') as f:
        tf_list = f.read().strip().lower().split('\n')
    with open(gene_path, 'r') as f:
        gene_list = f.read().strip().lower().split('\n')
    return tf_list, gene_list
 
def get_normalized_expr(expr_path, gene_list):   #log10 normalized， 0 trans to -2 # return numpy matrix, row: genes, col: cells
    dict_expr, cell_num = get_matrix(expr_path, gene_list)
    expr_mat = np.zeros((len(dict_expr), cell_num))
    index_row = 0
    for gene in dict_expr:
        dict_expr[gene] = np.log10(np.array(dict_expr[gene]) + 10 ** -2)
        expr_mat[index_row] = dict_expr[gene]
        index_row += 1
    return expr_mat, dict_expr

def get_matrix(mat_expr_path,gene_list):  # return hash gene-expr, cell number
    dict_expr = {}
    with open(mat_expr_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip().split('\t')
            line = [float(ch) for ch in line]
            dict_expr[gene_list[i]] = line
    cell_num = len(line)
    return dict_expr, cell_num 

def get_network(network_path):   # return hash tf-gene-label1 tf-gene-label0
    dict_tf_gene_1 = {}
    dict_tf_gene_0 = {}    
    all_gene = []
    with open(network_path,'r') as f:
        for line in f:
            tf, gene, label = line.strip().lower().split('\t')
            if abs(int(label)) == 1:
                if tf in dict_tf_gene_1:
                # if dict_tf_gene_1.get(tf):
                    dict_tf_gene_1[tf].append(gene)
                else:
                    dict_tf_gene_1[tf] = [gene]
            all_gene.append(gene)
            if abs(int(label)) == 0:
                if tf in dict_tf_gene_0:
                # if dict_tf_gene_1.get(tf):
                    dict_tf_gene_0[tf].append(gene)
                else:
                    dict_tf_gene_0[tf] = [gene]
            all_gene.append(gene)

    #all_gene = set(all_gene)
    #for tf, gene1 in dict_tf_gene_1.items():
    #    gene0 = all_gene - set(gene1)
    #    dict_tf_gene_0[tf] = list(gene0)

    return dict_tf_gene_1, dict_tf_gene_0

def embedding(tf_gene_pairs, expr_mat, dict_expr):
    tf_gene_dataset = ''
    total_matrix = []
    for tf, gene_list in tf_gene_pairs.items():
        gap = 100
        tf_data = dict_expr[tf]
        for gene in gene_list:
            gene_data = dict_expr[gene]
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
            tf_gene_feature = np.asanyarray(tf_gene_feature)
            total_matrix.append(tf_gene_feature)
            tf_gene_dataset += f"{tf}\t{gene}\t\n"
    return total_matrix, tf_gene_dataset


def embedding_cnn(tf_gene_pairs,expr_mat, dict_expr):
    # trans to piture size, h x w 60 * 60
    tf_gene_dataset = ''
    total_matrix = []
    for tf, gene_list in tf_gene_pairs.items():
        gap = 30
        tf_data = dict_expr[tf]
        for gene in gene_list:
            gene_data = dict_expr[gene]
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
            tf_gene_dataset += f"{tf}\t{gene}\t\n"

            
            # 控制样本数量，测试用，后续需要删掉
           # if len(total_matrix) >= 3000:
           #     break

    return total_matrix, tf_gene_dataset

def embedding_trans(tf_gene_pairs,expr_mat, dict_expr):
    # trans to piture size, h x w 60 * 60
    tf_gene_dataset = ''
    total_matrix = []
    for tf, gene_list in tf_gene_pairs.items():
        gap = 30
        tf_data = dict_expr[tf]
        for gene in gene_list:
            gene_data = dict_expr[gene]
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
            total_matrix.append(np.asanyarray(tf_gene_feature))
            tf_gene_dataset += f"{tf}\t{gene}\t\n"

            # 控制样本数量，测试用，后续需要删掉
            #if len(total_matrix) >= 3000:
            #    break

    return total_matrix, tf_gene_dataset



def embedding_trans_new(tf_gene_pairs,expr_mat, dict_expr):
    # trans to piture size, h x w 60 * 60
    tf_gene_dataset = ''
    total_matrix = []
    for tf, gene_list in tf_gene_pairs.items():
        gap = 512
        tf_data = dict_expr[tf]
        for gene in gene_list:
            gene_data = dict_expr[gene]
            list_x, list_y = [0] * gap, [0] * gap
            len_tf, len_gene = len(tf_data), len(gene_data)
            if len_tf <= gap:
                list_x[:len_tf], list_y[:len_gene] = tf_data, gene_data
            else:
                list_x, list_y = tf_data[:gap], gene_data[:gap]
            tf_gene_feature = np.append(list_x,list_y)
            total_matrix.append(tf_gene_feature)
            tf_gene_dataset += f"{tf}\t{gene}\t\n"

            # 控制样本数量，测试用，后续需要删掉
           # if len(total_matrix) >= 3000:
          #      break

    return total_matrix, tf_gene_dataset

def embedding_gcn(tf_gene_pairs,expr_mat, dict_expr, gene_list, gene_num):
    dict_gene_number = dict(zip(gene_list,range(gene_num)))
    F_mat = torch.tensor(expr_mat, dtype=torch.float)  
    edges = []
    for tf, genelist in tf_gene_pairs.items():
        for gene in genelist:
            edges.append([int(dict_gene_number[tf]), int(dict_gene_number[gene])])
    edge_index = np.asanyarray(edges).T
    edge_index = torch.tensor(np.array(edge_index), dtype=torch.long)      
    return F_mat, edge_index

#创建子类
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

def load_save_data(path,tf_path,gene_path,expr_path,network_path,model_idx='GRNCNN', type='train',args=None):
    dict_model = {'Trans': Config_Trans(), 'STGRNS':Config_STGRNS(), 'GRNCNN':Config_CNN(), 'ResNet18':Config_CNN(), 'GRNGCN':Config_GCN()}
    config = dict_model[model_idx]
    batch_size = args.batch_size if args else config.batch_size
    tf_list, gene_list = get_tf_gene_list(tf_path,gene_path)
    tf_num, gene_num = len(tf_list), len(gene_list)
    expr_mat, dict_expr = get_normalized_expr(expr_path,gene_list)
    tf_gene_label1, tf_gene_label0 = get_network(network_path)
    
    if model_idx == 'STGRNS':
        label1_matrix, tf_gene_dataset_1 = embedding(tf_gene_label1, expr_mat, dict_expr)
        label1 = [1] * len(label1_matrix)
        label0_matrix, tf_gene_dataset_0 = embedding(tf_gene_label0, expr_mat, dict_expr)
        label0 = [0] * len(label0_matrix)  

    if model_idx == 'GRNCNN' or model_idx == 'ResNet18':
        label1_matrix, tf_gene_dataset_1 = embedding_cnn(tf_gene_label1, expr_mat, dict_expr)  # label1_matrix[0].shape 1 * 60 * 60
        label1 = [1] * len(label1_matrix)
        label0_matrix, tf_gene_dataset_0 = embedding_cnn(tf_gene_label0, expr_mat, dict_expr) 
        label0 = [0] * len(label0_matrix)
        print(len(tf_gene_label1))
        print(len(tf_gene_label0))  
    
    if model_idx == 'Trans':
        label1_matrix, tf_gene_dataset_1 = embedding_trans(tf_gene_label1, expr_mat, dict_expr)
        label1 = [1] * len(label1_matrix)
        label0_matrix, tf_gene_dataset_0 = embedding_trans(tf_gene_label0, expr_mat, dict_expr)
        label0 = [0] * len(label0_matrix)  

        

    if model_idx == 'GRNGCN' and type =='train':
        gene_expr, edge_index = embedding_gcn(tf_gene_label1,expr_mat, dict_expr, gene_list, gene_num)

        data = Data(x=gene_expr, edge_index=edge_index) 
        print(data) 
        link_splitter = Tr.Compose([Tr.NormalizeFeatures(),Tr.RandomLinkSplit(num_val=0.125,num_test=0,add_negative_train_samples=False,is_undirected=True,neg_sampling_ratio=1.0,disjoint_train_ratio=0),]) #disjoint_train_ratio调节在“监督”阶段将使用多少条边作为训练信息。剩余的边将用于消息传递(网络中的信息传输阶段)。
        train_data, val_data, test_data = link_splitter(data) # val_data 会生成负样本，即使得到的结果正样本：负样本=1:1
        #print(train_datax)
        output=[]
        output_val=[]
        output_edge=[]
        for idx1, idx2, label in zip(train_data.edge_label_index[0].tolist(), train_data.edge_label_index[1].tolist(), train_data.edge_label):          
            output.append(f"{gene_list[idx1]}\t{gene_list[idx2]}\t{int(label)}")

        for idx1, idx2, label in zip(val_data.edge_label_index[0].tolist(), val_data.edge_label_index[1].tolist(), val_data.edge_label):          
            output_val.append(f"{gene_list[idx1]}\t{gene_list[idx2]}\t{int(label)}")
        save_path=f'{path}/{type}/'

        for idx1, idx2 in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist()):          
            output_edge.append(f"{gene_list[idx1]}\t{gene_list[idx2]}\t{int(1)}")
        save_path=f'{path}/{type}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)  
        with open(f'{path}/{type}/train_pair.txt', 'w') as file:
            file.write('\n'.join(output))
        with open(f'{path}/{type}/val_pair.txt', 'w') as file:
            file.write('\n'.join(output_val))
        with open(f'{path}/{type}/edge.txt', 'w') as file:
            file.write('\n'.join(output_edge))  
        print(train_data)
        print(val_data)
        return train_data,val_data
        '''
        data1=[]
        data0=[]
        gene_expr1, edge_index1 = embedding_gcn(tf_gene_label1,expr_mat, dict_expr, gene_list, gene_num)
        gene_expr0, edge_index0 = embedding_gcn(tf_gene_label0,expr_mat, dict_expr, gene_list, gene_num)
        data1 = Data(x=gene_expr1, edge_index=edge_index1)  
        data0 = Data(x=gene_expr0, edge_index=edge_index0) 
        data1.edge_label = torch.ones(data1.edge_index.size(1),dtype=torch.float32)
        data0.edge_label = torch.zeros(data0.edge_index.size(1),dtype=torch.float32)  
        data1.edge_label_index=data1.edge_index.clone()
        data0.edge_label_index=data0.edge_index.clone()

        #data_list = [data1, data0]
        #data_list_index=[data1.edge_label_index,data0.edge_label_index]
        #edge_indices = torch.cat(data_list_index,dim=1)

        #batch_data = Batch.from_data_list([data1, data0])
        #print(batch_data)
        #batch_data.edge_label_index=edge_indices
        total_samples1 = data1.edge_index.size(1) # 数据总量
        total_samples0 = data0.edge_index.size(1)
        train_size1 = int(0.875 * total_samples1)  # 训练集大小
        validation_size1 = int(0.125 * total_samples1)
        train_size0 = int(0.875 * total_samples0)  # 训练集大小
        validation_size0 = int(0.125 * total_samples0)
        indices1 = list(range(total_samples1))
        indices0 = list(range(total_samples0))
        random.shuffle(indices1) 
        random.shuffle(indices0) 
        train_indices1 = indices1[:train_size1]  # 训练集索引

        validation_indices1 = indices1[train_size1:train_size1 + validation_size1]  # 验证集索引

        train_indices0 = indices0[:train_size0]  # 训练集索引

        validation_indices0 = indices0[train_size0:train_size0 + validation_size0]  # 验证集索引
        train_data1=Data()
        train_data0=Data()
        validation_data1=Data()
        validation_data0=Data()

        train_data1.x=gene_expr1
        train_data1.edge_index=data1.edge_index[:,train_indices1]
        train_data1.edge_label = data1.edge_label[train_indices1]
        train_data1.edge_label_index = data1.edge_label_index[:, train_indices1]

        train_data0.x=gene_expr0
        train_data0.edge_index=data0.edge_index[:,train_indices0]
        train_data0.edge_label = data0.edge_label[train_indices0]
        train_data0.edge_label_index = data0.edge_label_index[:, train_indices0]
        print(train_data1)
        print(train_data0)
        validation_data1.x=gene_expr1
        validation_data1.edge_index=data1.edge_index[:,validation_indices1]
        validation_data1.edge_label = data1.edge_label[validation_indices1]
        validation_data1.edge_label_index = data1.edge_label_index[:, validation_indices1]
        validation_data0.x=gene_expr0
        validation_data0.edge_index=data0.edge_index[:,validation_indices0]
        validation_data0.edge_label = data0.edge_label[validation_indices0]
        validation_data0.edge_label_index = data0.edge_label_index[:, validation_indices0]
        data_list1=[train_data1,train_data0]
        data_list2=[validation_data1,validation_data0]
        batch_data1 = Batch.from_data_list(data_list1)
        batch_data2 = Batch.from_data_list(data_list2) 
        data_list_index1=[train_data1.edge_label_index,train_data0.edge_label_index]
        data_list_index0=[validation_data1.edge_label_index,validation_data0.edge_label_index]
        data_list_index11=[train_data1.edge_index,train_data0.edge_index]
        data_list_index00=[validation_data1.edge_index,validation_data0.edge_index]
        edge_indices1 = torch.cat(data_list_index1,dim=1)
        edge_indices0 = torch.cat(data_list_index0,dim=1)
        edge_indices11 = torch.cat(data_list_index11,dim=1)
        edge_indices00= torch.cat(data_list_index00,dim=1)
        batch_data1.edge_label_index=edge_indices1
        batch_data2.edge_label_index=edge_indices0
        batch_data1.edge_index=edge_indices11
        batch_data2.edge_index=edge_indices00
        #train_data = batch_data.__class__()
        #val_data = batch_data.__class__()



        #print(train_data)
        train_data = Data(x=gene_expr0, edge_index=batch_data1.edge_index, edge_label=batch_data1.edge_label, edge_label_index=batch_data1.edge_label_index)
        val_data = Data(x=gene_expr0, edge_index=batch_data2.edge_index, edge_label=batch_data2.edge_label, edge_label_index=batch_data2.edge_label_index)
        output=[]
        output_val=[]
        #data = Data(x=gene_expr, edge_index=edge_index)  
        #print(data)
        #link_splitter = Tr.RandomLinkSplit(num_val=0,num_test=0,add_negative_train_samples=False,is_undirected=True,neg_sampling_ratio=1.0,disjoint_train_ratio=0) #disjoint_train_ratio调节在“监督”阶段将使用多少条边作为训练信息。剩余的边将用于消息传递(网络中的信息传输阶段)。
        #train_data, val_data, test_data = link_splitter(data) # val_data 会生成负样本，即使得到的结果正样本：负样本=1:1
        #data1x = Data(x=gene_expr1, edge_index=edge_index1)
        #print(data1x)
        #train,_,_=link_splitter(data1x)
        #print(train)
        print(train_data)
        #print(val_data)
        #train_datax,_,_=link_splitter(train_data)
        #print(train_datax)
        for idx1, idx2, label in zip(train_data.edge_label_index[0].tolist(), train_data.edge_label_index[1].tolist(), train_data.edge_label):          
            output.append(f"{gene_list[idx1]}\t{gene_list[idx2]}\t{int(label)}")

        for idx1, idx2, label in zip(val_data.edge_label_index[0].tolist(), val_data.edge_label_index[1].tolist(), val_data.edge_label):          
            output_val.append(f"{gene_list[idx1]}\t{gene_list[idx2]}\t{int(label)}")
        save_path=f'{path}/{type}/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)  
        with open(f'{path}/{type}/train_pair.txt', 'w') as file:
            file.write('\n'.join(output))
        with open(f'{path}/{type}/val_pair.txt', 'w') as file:
            file.write('\n'.join(output_val))

        return train_data,val_data'''

    if model_idx== 'GRNGCN' and type !='train':
        data1=[]
        data0=[]
        gene_expr1, edge_index1 = embedding_gcn(tf_gene_label1,expr_mat, dict_expr, gene_list, gene_num)
        gene_expr0, edge_index0 = embedding_gcn(tf_gene_label0,expr_mat, dict_expr, gene_list, gene_num)
        data1 = Data(x=gene_expr1, edge_label_index=edge_index1)  
        data0 = Data(x=gene_expr0, edge_label_index=edge_index0) 
        data1.edge_label = torch.ones(data1.edge_label_index.size(1),dtype=torch.float32)
        data0.edge_label = torch.zeros(data0.edge_label_index.size(1),dtype=torch.float32)  

        
        train_network=f'{path}/train/edge.txt'
        tf_gene_label11, tf_gene_label00 = get_network(train_network)
        gene_expr11, edge_index11 = embedding_gcn(tf_gene_label11,expr_mat, dict_expr, gene_list, gene_num)

        data_list = [data1, data0]
        data_list_index=[data1.edge_label_index,data0.edge_label_index]

        edge_indices = torch.cat(data_list_index,dim=1)

        batch_data = Batch.from_data_list([data1, data0])

        #print(batch_data)
        batch_data.edge_label_index=edge_indices

        batch_data.x=gene_expr1
        batch_data.edge_index=edge_index11
        batch_data=Data(x=batch_data.x,edge_index=batch_data.edge_index,edge_label=batch_data.edge_label,edge_label_index=batch_data.edge_label_index)
        print(batch_data)

        output=[]
        #for idx1, idx2, label in zip(batch_data.edge_label_index[0].tolist(), batch_data.edge_label_index[1].tolist(), batch_data.edge_label):          
        #    output.append(f"{gene_list[idx1]}\t{gene_list[idx2]}\t{int(label)}")
        #save_path=f'{path}/{type}/'
        #if not os.path.exists(save_path):
        #    os.makedirs(save_path)  
        #with open(f'{path}/{type}/train_pair.txt', 'w') as file:
        #    file.write('\n'.join(output))

       # print(batch_data.edge_label_index)
        #print(batch_data)
        return batch_data

    save_path=f'{path}/{type}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)  
    np.save(f'{path}/{type}/matrix_positive.npy',label1_matrix)
    print(path)
    np.save(f'{path}/{type}/label_positive.npy',label1)
    np.save(f'{path}/{type}/matrix_negative.npy',label0_matrix)
    np.save(f'{path}/{type}/label_negative.npy',label0)
    with open(f'{path}/{type}/tf_gene_positive.txt','w') as f:
        f.write(tf_gene_dataset_1)
    with open(f'{path}/{type}/tf_gene_negative.txt','w') as f:
        f.write(tf_gene_dataset_0)

    # read the datasets
    # data_path = path
    # x_train_positive = np.load(data_path + 'matrix_positive.npy')    # ecoli shpae (2066, 8, 200)
    # y_train_positive = np.load(data_path + 'label_positive.npy')
    # x_train_negative = np.load(data_path + 'matrix_negative.npy')  
    # y_train_negative = np.load(data_path + 'label_negative.npy')  
    data_path = path
    x_train_positive = label1_matrix    # ecoli shpae (2066, 8, 200)
    y_train_positive = label1
    x_train_negative = label0_matrix
    y_train_negative = label0

    # 8：2 split dataset
    ratio = 0.89
    train_data_x = []
    train_data_y = []
    valid_data_x=[]
    valid_data_y=[]
    test_data_x = []
    test_data_y = []
    train_id=[]
    random.seed(10)
    min_sample_num = min(len(x_train_negative),len(x_train_positive))   # balance T number and F number
    sample_id = set(range(min_sample_num))         
    train_idx = random.sample(range(0,min_sample_num),int(ratio * min_sample_num))
    #train_idx = list(range(int(ratio * min_sample_num)))
    test_id = sample_id - set(train_idx) 
    for i in train_idx:
        train_data_x.append(x_train_positive[i])
        train_data_x.append(x_train_negative[i])
        train_data_y.append([y_train_positive[i]])
        train_data_y.append([y_train_negative[i]])

    for i in test_id:
        test_data_x.append(x_train_positive[i])
        test_data_x.append(x_train_negative[i])
        test_data_y.append([y_train_positive[i]])
        test_data_y.append([y_train_negative[i]])  

    with open(f'{path}/{type}/train_seed_id.txt','w') as f:
        f.write(str(train_idx).replace('[','').replace(']',''))
    
    gene_pair=[]   

    tf_gene_dataset_1 = tf_gene_dataset_1.split('\n')
    tf_gene_dataset_0 = tf_gene_dataset_0.split('\n')
    #print(tf_gene_dataset_0)
    with open(f'{path}/{type}/train_pair.txt','w',encoding='utf-8') as f:
        for idx in train_idx:
            f.write(tf_gene_dataset_1[idx]+str(y_train_positive[idx])+'\n')
            f.write(tf_gene_dataset_0[idx]+str(y_train_negative[idx])+'\n')

    with open(f'{path}/{type}/val_pair.txt','w',encoding='utf-8') as f:
        for idx in test_id:
            f.write(tf_gene_dataset_1[idx]+str(y_train_positive[idx])+'\n')
            f.write(tf_gene_dataset_0[idx]+str(y_train_negative[idx])+'\n')
    #with open(f'{data_path}test_seed_id.txt','w') as f:
        #f.write(str(test_id).replace('[','').replace(']',''))
    


    # create DataLoader iterator 


   # valid_dataset = subDataset(test_data_x, test_data_y)
    if type== 'train':
        train_dataset = subDataset(train_data_x, train_data_y)
        valid_dataset = subDataset(test_data_x, test_data_y)
        train_dataloader = DataLoader.DataLoader(train_dataset, batch_size= batch_size, shuffle = True, drop_last=False, num_workers= config.nworkers)
        valid_dataloader = DataLoader.DataLoader(valid_dataset, batch_size= batch_size, shuffle = False, drop_last=False, num_workers= config.nworkers)
        dump(train_dataloader,f'{path}/{type}/{type}_dataloder')
        dump(valid_dataloader,f'{path}/{type}/train_val_dataloder')
        return train_dataloader,valid_dataloader
    else:
        train_dataset = subDataset(train_data_x, train_data_y)
        train_dataloader = DataLoader.DataLoader(train_dataset, batch_size= batch_size, shuffle = False, drop_last=False, num_workers= config.nworkers)
        dump(train_dataloader,f'{path}/{type}/{type}_dataloder')
        return train_dataloader

