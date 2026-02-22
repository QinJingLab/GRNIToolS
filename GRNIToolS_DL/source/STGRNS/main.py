import numpy as np
import csv
from numpy import *
import os
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
from sklearn import metrics
import os
import csv
import math
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import TensorDataset, DataLoader
import torch.utils.data.dataset as Dataset
import torch.distributed as dist

# torch.set_default_tensor_type(torch.DoubleTensor)
warnings.filterwarnings('ignore')

def get_parser():
    parser = argparse.ArgumentParser(description='This process is used to construct gene regulator networks')
    parser.add_argument("-e", "--expr", type = str, default= './Input_data/Ecoli/GRNDL/expression.csv', help="<file> Input expression file, col for cell, row for gene") 
    parser.add_argument('--train', type =str, default = './Input_data/Ecoli/GRNDL/train.txt', help="<file> Input train dataset")
    parser.add_argument('--test',  type =str, default = './Input_data/Ecoli/GRNDL/random_test.txt', help="<file> Input test dataset")
    parser.add_argument('--valid', type =str, default = './Input_data/Ecoli/GRNDL/val.txt', help="<file> Input valid dataset")
    parser.add_argument('--tftest', type =str, default='./Input_data/Ecoli/GRNDL/TF_test.txt', help="<file> Input tftest dataset")
    parser.add_argument("-o", "--output", type = str, default='output/', help="<dir> Result directory")
    parser.add_argument('--lr', type=float, required=False, default=0.0003, help="The learning rate used in model")
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size used in the training process.')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--dim_models', type=int, default=200, help='Number of dimensions of model')
    parser.add_argument('--head', type=int, default=2, help="Number of head")
    parser.add_argument('--cuda_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES (comma-separated), e.g. 0,1,3')

    args= parser.parse_args()    
    return args


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 显式设置当前设备


def get_tf_list(tf_path):
    # return tf_list
    f_tf = open(tf_path)
    tf_reader = list(csv.reader(f_tf))
    tf_list = []
    for single in tf_reader[1:]:
        tf_list.append(single[0])
    print('Load ' + str(len(tf_list)) + ' TFs successfully!')
    return tf_list

def get_origin_expression_data(gene_expression_path):
    # return 1.tf-targets dict and pair-score dict
    #        2.number of timepoints
    f_expression = open(gene_expression_path, encoding="utf-8")
    expression_reader = list(csv.reader(f_expression))
    cells = expression_reader[0][1:]
    num_cells = len(cells)

    expression_record = {}
    num_genes = 0
    for single_expression_reader in expression_reader[1:]:
        if single_expression_reader[0] in expression_record:
            print('Gene name ' + single_expression_reader[0] + ' repeat!') 
        expression_record[single_expression_reader[0]] = list(map(float, single_expression_reader[1:]))
        num_genes += 1
    print(str(num_genes) + ' genes and ' + str(num_cells) + ' cells are included in origin expression data.')
    return expression_record, cells

def get_normalized_expression_data(gene_expression_path):
    # return 1.tf-targets dict and pair-score dict
    #        2.number of timepoints
    expression_record, cells = get_origin_expression_data(gene_expression_path)
    expression_matrix = np.zeros((len(expression_record), len(cells)))
    index_row = 0
    for gene in expression_record:
        expression_record[gene] = np.log10(np.array(expression_record[gene]) + 10 ** -2)
        expression_matrix[index_row] = expression_record[gene]
        index_row += 1

    # Heat map
    # plt.figure(figsize=(15,15))
    # sns.heatmap(expression_matrix[0:100,0:100])
    # plt.show()

    return expression_record, cells

def get_gene_ranking(gene_order_path, low_express_gene_list, gene_num, output_path,
                     flag):  # flag=True:write to output_path
    # 1.delete genes p-value>=0.01
    # 2.delete genes with low expression
    # 3.rank genes in descending order of variance
    # 4.return gene names list of top genes and variance_record of p-value<0.01
    f_order = open(gene_order_path)
    order_reader = list(csv.reader(f_order))
    if flag:
        f_rank = open(output_path, 'w', newline='\n')
        f_rank_writer = csv.writer(f_rank)
    variance_record = {}
    variance_list = []
    significant_gene_list = []
    for single_order_reader in order_reader[1:]:
        # column 0:gene name
        # column 1:p value
        # column 2:variance
        if float(single_order_reader[1]) >= 0.01:
            break
        if single_order_reader[0] in low_express_gene_list:
            continue
        variance = float(single_order_reader[2])
        if variance not in variance_record:  # 1 variance corresponding to 1 gene
            variance_record[variance] = single_order_reader[0]
        else:  # 1 variance corresponding to n genes
            print(str(variance_record[variance]) + ' and ' + single_order_reader[0] + ' variance repeat!')
            variance_record[variance] = [variance_record[variance]]
            variance_record[variance].append(single_order_reader[0])
        variance_list.append(variance)
        tstr = single_order_reader[0]
        single_order_reader[0] = tstr.upper()
        significant_gene_list.append(single_order_reader[0])
    print('After delete genes with p-value>=0.01 or low expression, ' + str(len(variance_list)) + ' genes left.')
    variance_list.sort(reverse=True)
    gene_rank = []
    for single_variance_list in variance_list[0:gene_num]:
        if type(variance_record[single_variance_list]) is str:  # 1 variance corresponding to 1 gene
            gene_rank.append(variance_record[single_variance_list])
        else:  # 1 variance corresponding to n genes
            gene_rank.append(variance_record[single_variance_list][0])
            del variance_record[single_variance_list][0]
            if len(variance_record[single_variance_list]) == 1:
                variance_record[single_variance_list] = variance_record[single_variance_list][0]
        if flag:
            f_rank_writer.writerow([variance_record[single_variance_list]])
    f_order.close()
    if flag:
        f_rank.close()
    return gene_rank, significant_gene_list

def get_filtered_gold(gold_network_path, rank_list, output_path, flag):
    # 1.Load origin gold file
    # 2.Delete genes not in rank_list
    # 3.return tf-targets dict and pair-score dict
    # Note: If no score in gold network, score=999
    f_gold = open(gold_network_path, encoding='UTF-8-sig')
    gold_reader = list(csv.reader(f_gold))
    for i in range(0, len(gold_reader) - 1):
        temp = gold_reader[i]
        s1 = str(temp[0])
        s2 = str(temp[1])

        temp[0] = s1.upper()
        temp[1] = s2.upper()

        gold_reader[i] = temp
    # print("gold_reader",gold_reader)
    # print("rank_list",rank_list)
    # print("gold_reader",gold_reader)
    print("gold_reader[0]", gold_reader[0])
    has_score = True
    if len(gold_reader[0]) < 3:
        has_score = False
    gold_pair_record = {}
    gold_score_record = {}
    unique_gene_list = []
    for single_gold_reader in gold_reader[1:]:
        # column 0: TF
        # column 1: target gene
        # column 2: regulate score
        if (single_gold_reader[0] not in rank_list) or (single_gold_reader[1] not in rank_list):
            continue
        gene_pair = [single_gold_reader[0], single_gold_reader[1]]
        str_gene_pair = single_gold_reader[0] + ',' + single_gold_reader[1]

        if single_gold_reader[0] not in unique_gene_list: unique_gene_list.append(single_gold_reader[0])
        if single_gold_reader[1] not in unique_gene_list: unique_gene_list.append(single_gold_reader[1])
        if str_gene_pair in gold_score_record:
            print('Gold pair repeat!')
        if has_score:
            print("single_gold_reader[2]", single_gold_reader[2])
            gold_score_record[str_gene_pair] = float(single_gold_reader[2])
        else:
            gold_score_record[str_gene_pair] = 999
        if gene_pair[0] not in gold_pair_record:
            gold_pair_record[gene_pair[0]] = [gene_pair[1]]
        else:
            gold_pair_record[gene_pair[0]].append(gene_pair[1])
    print("gold_pair_record", gold_pair_record)
    # Some statistics of gold_network
    print(str(len(gold_pair_record)) + ' TFs and ' + str(
        len(gold_score_record)) + ' edges in gold_network consisted of genes in rank_list.')
    print(str(len(unique_gene_list)) + ' genes are common in rank_list and gold_network.')

    rank_density = len(gold_score_record) / (len(gold_pair_record) * (len(rank_list)))
    gold_density = len(gold_score_record) / (len(gold_pair_record) * (len(unique_gene_list)))

    print('Rank genes density = edges/(TFs*(len(rank_gene)-1))=' + str(rank_density))
    print('Gold genes density = edges/(TFs*len(unique_gene_list))=' + str(gold_density))

    # write to file
    print("unique_gene_list", unique_gene_list)
    if flag:
        f_unique = open(output_path, 'w', encoding="utf-8", newline='\n')
        f_unique_writer = csv.writer(f_unique)
        out_unique = np.array(unique_gene_list).reshape(len(unique_gene_list), 1)
        f_unique_writer.writerows(out_unique)
        f_unique.close()
    return gold_pair_record, gold_score_record, unique_gene_list

def generate_filtered_gold(gold_pair_record, gold_score_record, output_path):
    # write filtered_gold to output_path
    # print("cnm")
    f_filtered = open(output_path, 'w', encoding="utf-8", newline='\n')
    f_filtered_writer = csv.writer(f_filtered)
    f_filtered_writer.writerow(['TF', 'Target', 'Score'])
    # print("cnm")
    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            single_output = [tf, target, gold_score_record[tf + ',' + target]]
            once_output.append(single_output)
        f_filtered_writer.writerows(once_output)
    f_filtered.close()

def get_gene_pair_list(unique_gene_list, gold_pair_record, gold_score_record, output_file):
    # positive is relationship that tf regulate target
    # negtive is reationship that same tf doesn's regulate target.
    # When same tf doesn't have enough negtive, borrow negtive from other TFs.
    # When negtive is not enough,stop and prove positive:negtive = 1:1

    # generate all negtive gene pairs of TFs
    all_tf_negtive_record = {}
    for tf in gold_pair_record:
        # print("tf",tf)
        all_tf_negtive_record[tf] = []
        for target in unique_gene_list:
            if target in gold_pair_record[tf]:
                continue
            all_tf_negtive_record[tf].append(target)

    # generate negtive record without borrow
    rank_negtive_record = {}
    for tf in gold_pair_record:
        num_positive = len(gold_pair_record[tf])
        if num_positive > len(all_tf_negtive_record[tf]):
            rank_negtive_record[tf] = all_tf_negtive_record[tf]
            all_tf_negtive_record[tf] = []
        else:
            # maybe random.sample(all_tf_negtive_record[tf],num_positive) to promote performance
            rank_negtive_record[tf] = all_tf_negtive_record[tf][:num_positive]
            all_tf_negtive_record[tf] = all_tf_negtive_record[tf][num_positive:]

    # output positive and negtive pairs
    f_gpl = open(output_file, 'w', newline='\n')
    f_gpl_writer = csv.writer(f_gpl)
    f_gpl_writer.writerow(['TF', 'Target', 'Label', 'Score'])
    stop_flag = False
    for tf in gold_pair_record:
        once_output = []
        for target in gold_pair_record[tf]:
            # output positive
            single_output = [tf, target, '1', gold_score_record[tf + ',' + target]]
            once_output.append(single_output)
            # output negtive
            if len(rank_negtive_record[tf]) == 0:
                # borrow negtive for other TFs
                find_negtive = False
                for borrow_tf in all_tf_negtive_record:
                    if len(all_tf_negtive_record[borrow_tf]) > 0:
                        find_negtive = True
                        single_output = [borrow_tf, all_tf_negtive_record[borrow_tf][0], 0, 0]
                        del all_tf_negtive_record[borrow_tf][0]
                        break
                # if not enough negtive of others,stop and prove positive:negtive = 1:1
                if not find_negtive:
                    stop_flag = True
                    break
            else:
                # negtive without borrow
                single_output = [tf, rank_negtive_record[tf][0], 0, 0]
                del rank_negtive_record[tf][0]
            once_output.append(single_output)
        if stop_flag:
            f_gpl_writer.writerows(once_output[:-1])
            print('Negtive not enough!')
            break
        f_gpl_writer.writerows(once_output)  # output positive and negtive of 1 TF at a time
    f_gpl.close()

def get_low_express_gene(origin_expression_record, num_cells):
    # get gene_list who were expressed in fewer than 10% of the cells
    gene_list = []
    threshold = num_cells // 10
    for gene in origin_expression_record:
        num = 0
        for expression in origin_expression_record[gene]:
            if expression != 0:
                num += 1
                if num > threshold:
                    break
        if num <= threshold:
            gene_list.append(gene)
    return gene_list


def loadData(gene_pair_list_path,gene_expression_path):

    origin_expression_record, cells = get_normalized_expression_data(gene_expression_path)
    print("len(origin_expression_record)", len(origin_expression_record))

    # Load gold_pair_record
    all_gene_list = []
    gold_pair_record = {}

    # f_genePairList = open(gene_pair_list_path, encoding='UTF-8')  ### read the gene pair and label file
    # for single_pair in list(csv.reader(f_genePairList))[1:]:
    #     # print("single_pair",single_pair)
    #     if single_pair[2] == '1':
    #         if single_pair[0] not in gold_pair_record:
    #             gold_pair_record[single_pair[0]] = [single_pair[1]]
    #         else:
    #             gold_pair_record[single_pair[0]].append(single_pair[1])
    #         # count all genes in gold edges
    #         if single_pair[0] not in all_gene_list:
    #             all_gene_list.append(single_pair[0])
    #         if single_pair[1] not in all_gene_list:
    #             all_gene_list.append(single_pair[1])
    # f_genePairList.close()



    with open(gene_pair_list_path, 'r') as f:
        for line in f:
            single_pair = line.strip().split('\t')
            if str(single_pair[2]) == '1':
                if single_pair[0] not in gold_pair_record:
                    gold_pair_record[single_pair[0]] = [single_pair[1]]
                else:
                    gold_pair_record[single_pair[0]].append(single_pair[1])
                if single_pair[0] not in all_gene_list:
                    all_gene_list.append(single_pair[0])
                if single_pair[1] not in all_gene_list:
                    all_gene_list.append(single_pair[1])
  

    # print dataset statistics
    print('All genes:' + str(len(all_gene_list)))
    print('TFs:' + str(len(gold_pair_record.keys())))
    print("len(single_pair)", len(single_pair))
    # Generate Pearson matrix
    label_list = []
    pair_list = []
    total_matrix = []
    num_tf = -1
    num_label1 = 0
    num_label0 = 0

    # control cell numbers by means of timepoints
    timepoints = len(cells)
    # timepoints=800
    x = []
    miss = 0 

    with open(gene_pair_list_path, 'r') as f:
        for line in f:
            single_pair = line.strip().split('\t')
            tf_name = single_pair[0]
            target_name = single_pair[1]
            flag = False
            if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(target_name)):
                flag = True
            if flag:    
                if str(single_pair[2]) == '1':
                    label = 1
                    num_label1 += 1
                elif str(single_pair[2]) == '0':
                    label = 0
                    num_label0 += 1
                label_list.append(label)
                pair_list.append(tf_name + ',' + target_name)
                tf_data = origin_expression_record[tf_name]
                target_data = origin_expression_record[target_name]
            else:
                miss = miss + 1
                continue
            
            single_tf_list = []
            gap = 100
            for k in range(0, len(tf_data), gap):
                feature = []
                a = tf_data[k:k + gap]
                b = target_data[k:k + gap]
                feature.extend(a)
                feature.extend(b)
                # single_tf_list.append(feature)
                feature = np.asarray(feature)
                # print("feature.shape", feature.shape)
                if (len(feature) == 2 * gap):
                    single_tf_list.append(feature)

            single_tf_list = np.asarray(single_tf_list)

            total_matrix.append(single_tf_list)

    total_matrix = np.asarray(total_matrix)
    label_list = np.array(label_list)
    # print("label_list.shape", label_list.shape)
    pair_list = np.array(pair_list)


    # for i in gold_pair_record:
    #     num_tf += 1
    #     for j in range(len(all_gene_list)):
    #         # for j in range(2):
    #         miss = 0
    #         # print('Generating matrix of gene pair ' + str(num_tf) + ' ' + str(j))
    #         tf_name = i
    #         target_name = all_gene_list[j]

    #         flag = False
    #         if (origin_expression_record.__contains__(tf_name) & origin_expression_record.__contains__(target_name)):
    #             flag = True

    #         if (flag):
    #             if tf_name in gold_pair_record and target_name in gold_pair_record[tf_name]:
    #                 label = 1
    #                 num_label1 += 1
    #             else:
    #                 label = 0
    #                 num_label0 += 1
    #             label_list.append(label)
    #             pair_list.append(tf_name + ',' + target_name)

    #             tf_data = origin_expression_record[tf_name]
    #             target_data = origin_expression_record[target_name]
    #         else:
    #             miss = miss + 1
    #             continue

    #         single_tf_list = []
    #         gap = 100
    #         for k in range(0, len(tf_data), gap):
    #             feature = []
    #             a = tf_data[k:k + gap]
    #             b = target_data[k:k + gap]
    #             feature.extend(a)
    #             feature.extend(b)
    #             # single_tf_list.append(feature)
    #             feature = np.asarray(feature)
    #             # print("feature.shape", feature.shape)
    #             if (len(feature) == 2 * gap):
    #                 single_tf_list.append(feature)

    #         single_tf_list = np.asarray(single_tf_list)

    #         total_matrix.append(single_tf_list)

    # total_matrix = np.asarray(total_matrix)
    # label_list = np.array(label_list)
    # # print("label_list.shape", label_list.shape)
    # pair_list = np.array(pair_list)

    print('PCC matrix generation finish.')
    print('Positive edges:' + str(num_label1))
    print('Negative edges:' + str(num_label0))
    print('Density=' + str(num_label1 / (num_label1 + num_label0)))

    return [total_matrix, label_list, pair_list]

##generating the data can be inputted by the STGRNS

def numpy2loader(X, y, batch_size):
    X_set = torch.from_numpy(X.astype(np.float32))  # 确保输入为 float32
    y_set = torch.from_numpy(y.astype(np.int64))
    X_loader = DataLoader(X_set, batch_size=batch_size)
    y_loader = DataLoader(y_set, batch_size=batch_size)

    return X_loader, y_loader

def loaderToList(data_loader):
    length = len(data_loader)
    data = []
    for i in data_loader:
        data.append(i)
    return data

def data2dataset(X, y, batch_size):
    # 添加维度转换，将输入数据转换为 (batch_size, seq_len, input_dim)
    X = X.transpose(0, 2, 1)  # 从 [samples, time_steps, features] 转为 [samples, features, time_steps]
    X_set = torch.from_numpy(X.astype(np.float32))  
    y_set = torch.from_numpy(y.astype(np.int64))
    
    # 创建TensorDataset时保持维度一致
    dataset = torch.utils.data.TensorDataset(X_set, y_set)
    return dataset

import torch
import torch.nn as nn
import math

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # self.register_buffer('pe', pe)
        self.register_parameter('pe', nn.Parameter(pe, requires_grad=False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class STGRNS(nn.Module):
    def __init__(self, input_dim, nhead=2, d_model=80, num_classes=2, dropout=0.1):
        super().__init__()
        self.prenet = nn.Linear(input_dim, d_model)
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=256, nhead=2, dropout=dropout
        )

        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, window_size):
        out = window_size.permute(1, 0, 2)
        out = self.positionalEncoding(out)
        out = self.encoder_layer(out)
        out = out.transpose(0, 1)
        stats = out.mean(dim=1)
        out = self.pred_layer(stats)
        return out


def STGRNSForGRNSRconstruction(rank, world_size, args, train_dataset, valid_dataset):
    # 初始化DDP环境
    setup(rank, world_size)
    torch.manual_seed(42)  # 保证所有进程初始权重一致

    x_train, y_train, gene_pair_train = train_dataset
    x_train = x_train.astype(np.float32)
    y_train = y_train.astype(np.int64)
    # X_trainloader, y_trainloader = numpy2loader(x_train, y_train, batch_size)
    train_data =  data2dataset(x_train, y_train, args.batch_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_data, num_replicas=world_size, rank=rank
    )
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        pin_memory=True
    )

    model = STGRNS(input_dim=200, nhead=args.head, d_model=args.dim_models, num_classes=2)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_epochs = args.epochs
    acc_record = {'train': [], 'dev': []}
    loss_record = {'train': [], 'dev': []}


    for epoch in range(n_epochs):
        train_sampler.set_epoch(epoch)
        ddp_model.train()
        train_loss = []
        all_labels = []
        all_probs = []  # 新增收集预测概率
        for data, labels in train_loader:
            data = data.to(torch.float32)
            data = data.permute(0, 2, 1).to(rank) 
            labels = labels.long().to(rank)
            logits = ddp_model(data)
            # labels = torch.tensor(labels, dtype=torch.long)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10) 
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            probs = F.softmax(logits, dim=1)[:, 1]  # 获取正类概率
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            train_loss.append(loss.item())
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        aupr = average_precision_score(all_labels, all_probs)

        train_loss = sum(train_loss) / len(train_loss)
        loss_record['train'].append(train_loss)

        if rank == 0:
            print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f} auc = {auc:.5f} aupr = {aupr:.5f}")
        # print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f} acc = {acc:.5f}")
    print('________________________________________________________________')
  
    model.eval()
    y_predict = []
    x_test, y_test, gene_pair_test = valid_dataset
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.int64)
    # X_trainloader, y_trainloader = numpy2loader(x_train, y_train, batch_size)
    valid_data =  data2dataset(x_test, y_test, args.batch_size)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_data, num_replicas=world_size, rank=rank
    )
    valid_loader = DataLoader(
        valid_data, 
        batch_size=args.batch_size,
        sampler=valid_sampler,
        pin_memory=True
    )
    with torch.no_grad():
        all_labels = []
        all_probs = []
        for data, labels in valid_loader:
            data = data.to(torch.float32)
            data = data.permute(0, 2, 1).to(rank) 
            labels = labels.long().to(rank)
            logits = model(data)
            
            # 与训练部分保持一致的指标收集逻辑
            probs = F.softmax(logits, dim=1)[:, 1]
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
            # 计算当前batch的准确率
            acc = (logits.argmax(dim=-1) == labels).float().mean()

        # 聚合所有进程数据后计算全局指标
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        auc = roc_auc_score(all_labels, all_probs)
        aupr = average_precision_score(all_labels, all_probs)

        if rank == 0:  # 只在主进程输出
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] acc = {acc:.5f} auc = {auc:.5f} aupr = {aupr:.5f}")
    if rank == 0:
        save_path = args.output
        model_state_name = f"{save_path}/model.pkl"
        torch.save(model.state_dict(), model_state_name)      
    dist.destroy_process_group()

###predict-----------------------------------------------
def predict(dataset_pred, save_path,d_models, batch_sizes, head=2):

    model = STGRNS(input_dim=200, nhead=head, d_model=d_models, num_classes=2)

    model.load_state_dict(torch.load(save_path +'/model.pkl'))
    model.eval()

    y_predict = []
    x_test, y_test, gene_pair_test = dataset_pred
    X_testloader, y_testloader = numpy2loader(x_test, y_test, batch_sizes)
    X_testList = loaderToList(X_testloader)
    y_testList = loaderToList(y_testloader)
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for k in range(0, len(X_testList)):
            data = X_testList[k].float()
            logits = model(data)
            predt = F.softmax(logits)
            temps = predt.cpu().numpy().tolist()
            for i in temps:
                t = i[1]
                y_predict.append(t)
    roc_auc = roc_auc_score(y_test,y_predict).round(5)
    pr_auc = average_precision_score(y_test, y_predict).round(5)

    network = 'TF,GENE,SCORE\n'
    for idx, tfgene in enumerate(gene_pair_test):
        tf, gene = tfgene.split(',')
        score = round(y_predict[idx],5)
        network += f"{tf},{gene},{score}\n"
    return roc_auc, pr_auc, network 




if __name__ == "__main__":
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    world_size = len(args.cuda_devices.split(','))

    ## load dataset
    train_dataset = loadData(args.train,args.expr)
    valid_dataset = loadData(args.valid,args.expr)
    test_dataset = loadData(args.test,args.expr)
    tftest_dataset = loadData(args.tftest,args.expr)


    ##training model and then predicting unknown network
    batch_sizes = args.batch_size
    epochs = args.epochs
    lr = args.lr
    d_models = args.dim_models
    n_head = args.head
    save_path = args.output
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # STGRNSForGRNSRconstruction(args.cuda_devices, batch_sizes, epochs, lr, n_head, d_models, train_dataset, valid_dataset, save_path)
    # 启动分布式训练
    torch.multiprocessing.spawn(
        STGRNSForGRNSRconstruction,
        args=(world_size, args, train_dataset, valid_dataset),
        nprocs=world_size
    )
    
    ### prediction 
    valid_auc, valid_aupr, valid_grn = predict(valid_dataset, save_path,d_models, batch_sizes, n_head)
    test_auc, test_aupr, test_grn = predict(test_dataset, save_path,d_models, batch_sizes, n_head)
    tftest_auc, tftest_aupr, tftest_grn = predict(tftest_dataset, save_path,d_models, batch_sizes, n_head)
    res = 'method\tdataset\tauroc\taupr\n'
    res += f'STGRNS\tvalid\t{valid_auc}\t{valid_aupr}\n'
    res += f'STGRNS\ttest\t{test_auc}\t{test_aupr}\n'
    res += f'STGRNS\ttftest\t{tftest_auc}\t{tftest_aupr}\n'
    with open(f"{save_path}/roc.txt", "w") as f:
        f.write(res)
    with open(f"{save_path}/test_network.txt", "w") as f:
        f.write(test_grn)
    with open(f"{save_path}/valid_network.txt", "w") as f:
        f.write(valid_grn)
    with open(f"{save_path}/tftest_network.txt", "w") as f:
        f.write(tftest_grn)    
        
