"""
-*- coding: utf-8 -*-
@Author : Smartpig
@Institution : DHU/DBLab
@Time : 2022/11/26 22:30
"""

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
import argparse
import os
from model import mymodel
from Load_data import gen_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score
from torch.optim.lr_scheduler import StepLR

np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


seed = 6
torch.manual_seed(seed) # 为CPU设置随机种子
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
np.random.seed(seed)  # Numpy module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



def parse_args():
    parser = argparse.ArgumentParser(description="请选择你的参数")
    #Parameters to split dataset
    parser.add_argument('--geneDataName', type=str, default = 'output_data_pre/mESC/SC-PC/IGEGRNS/expressions.csv', help='Gene expression data file name')
    # parser.add_argument('--refNetworkName', help='Gene regulator network file name', type=str, default='refNetwork.csv')
    parser.add_argument('-train', "--train_file", type =str, default = 'output_data_pre/mESC/SC-PC/IGEGRNS/train.tsv', help="<file> Input train dataset")
    parser.add_argument('-test', "--test_file", type =str, default = 'output_data_pre/mESC/SC-PC/IGEGRNS/test.tsv', help="<file> Input test dataset")
    parser.add_argument('-val', "--valid_file", type =str, default = 'output_data_pre/mESC/SC-PC/IGEGRNS/val.tsv', help="<file> Input valid dataset")
    parser.add_argument('-tftest', "--tftest_file", type =str, default='output_data_pre/mESC/SC-PC/IGEGRNS/tftest.tsv', help="<file> Input tftest dataset")
    parser.add_argument("-o", "--output", type = str, default='output/', help="<dir> Result directory")
    parser.add_argument('--lr', type=float, required=False, default=0.0001, help="The learning rate used in model")
    parser.add_argument('--epochs', type=int, default=200, required=False, help='Number of epochs')

    parser.add_argument('--datasetPath', help='Split folder file name', type=str, default='dataset')
    parser.add_argument('--k', help='k-fold cross validation', type=int, default=5)
    #Parameters to train model
    parser.add_argument('--batchSize', help="Batch size for training", type=int, default=32)
    parser.add_argument('--decodeDim', help="Output dimension of the encoder", type=int, default=256)
    parser.add_argument('--aggregator', help="Type of aggregator to use", choices=['mean', 'lstm', 'max'], default='mean')
    parser.add_argument('--normalize', help="Whether to normalize the encoder output", type=bool, default=True)
    parser.add_argument('--topkratio', help="Ratio for Top-K pooling", type=int, default=1)
    args = parser.parse_args()
    return args

def get_train_test_dataset(args):
    test_file = args.test_file
    train_file = args.train_file
    valid_file = args.valid_file
    tftest_file = args.tftest_file

    test_df = pd.read_csv(test_file, sep='\t', header=0, index_col=None)
    testEdge = torch.from_numpy(np.array(test_df.sample(frac=1), dtype='int64')) # 30000 * 3 分别为TF, GENE, LABEL

    valid_df = pd.read_csv(valid_file, sep='\t', header=0, index_col=None)
    validEdge = torch.from_numpy(np.array(valid_df.sample(frac=1), dtype='int64')) # 30000 * 3 分别为TF, GENE, LABEL
    
    tftest_df = pd.read_csv(tftest_file, sep='\t', header=0, index_col=None)
    tftestEdge = torch.from_numpy(np.array(tftest_df.sample(frac=1), dtype='int64')) # 30000 * 3 分别为TF, GENE, LABEL

    train_df = pd.read_csv(train_file, sep='\t', header=0, index_col=None)
    trainEdge = torch.from_numpy(np.array(train_df.sample(frac=1), dtype='int64')) 

    trueEdge = pd.DataFrame(train_df.loc[train_df['Label'] == 1, ['TF', 'Target']], dtype='int64') # ture edge; train数据的阳性边，两列数据TF , targer
    trueEdge = torch.from_numpy(np.array(trueEdge).T).long() # 2 * 31000 转成符合graph数据格式
    
    return trueEdge, trainEdge, validEdge, testEdge, tftestEdge


def main(args):
    # os.chdir(dataPath)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    geneExpData = pd.read_csv(args.geneDataName, sep=',', header=0, index_col=0)
    x = torch.from_numpy(np.array(geneExpData, dtype=np.float64)).float()
    AUROC, AUPRC = [],[]
    trueEdge, trainEdge, validEdge, testEdge, tftestEdge = get_train_test_dataset(args)

    data = Data(x=x, edge_index=trueEdge).to(device)
    trainEdge = trainEdge.to(device)
    model = mymodel(data.x.size()[1], args.decodeDim, args.aggregator, args.normalize, args.topkratio,
                    data.x.size()[0]).to(device)
    # lr = 0.005 
    lr = args.lr
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)  
    scheduler = StepLR(optimizer, step_size=5, gamma=0.8)
    for epoch in range(0, args.epochs):
        loss = train(data, model, optimizer, scheduler, trainEdge)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    print("OK")

    ### model.eval()
    res = f'method\tdataset\tAUROC\tAUPRC\n'
    # train
    trainEdge = trainEdge.to('cpu')
    score = predict(model, data, trainEdge[:, 0:2], args.batchSize)
    trueEdge = np.array(trainEdge[:, 2])
    auroc, auprc = comp_res(score, trueEdge)
    # res += f'train\t{auroc}\t{auprc}\n'
    # valid
    score = predict(model, data, validEdge[:, 0:2], args.batchSize)
    trueEdge = np.array(validEdge[:, 2])
    auroc, auprc = comp_res(score, trueEdge)
    auroc, auprc = round(auroc,5), round(auprc,5)
    res += f'IGEGRNS\tvalid\t{auroc}\t{auprc}\n'
    # test
    score = predict(model, data, testEdge[:, 0:2], args.batchSize)
    trueEdge = np.array(testEdge[:, 2])
    auroc, auprc = comp_res(score, trueEdge)
    auroc, auprc = round(auroc,5), round(auprc,5)
    res += f'IGEGRNS\ttest\t{auroc}\t{auprc}\n'
    # tftest
    score = predict(model, data, tftestEdge[:, 0:2], args.batchSize)
    trueEdge = np.array(tftestEdge[:, 2])
    auroc, auprc = comp_res(score, trueEdge)
    auroc, auprc = round(auroc,5), round(auprc,5)
    res += f'IGEGRNS\ttftest\t{auroc}\t{auprc}\n'

    return res

def train(data, model, optimizer, scheduler, trainEdge):
    model.train()
    # 梯度置零
    optimizer.zero_grad()
    x, topkarr = model.Sage_forword(data.x, data.edge_index)
    linkPredict = model.score(x, topkarr, trainEdge)
    linkLabels = trainEdge[:, 2].float()
    loss = F.binary_cross_entropy_with_logits(linkPredict, linkLabels)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss



@torch.no_grad()
def predict(model, data, testEdge, batchSize):
    model.eval()
    inputx, topkarr = model.Sage_forword(data.x, data.edge_index)

    # relEmb = getRelEmb(data.edge_index, inputx).view(1, inputx.shape[1])

    Score = np.empty(shape=0)
    stack_list = []
    times = 1
    for i in range(testEdge.shape[0]):
        a = inputx[int(testEdge[i][0])].view(1, model.decodeDim)
        b = inputx[int(testEdge[i][1])].view(1, model.decodeDim)
        hrt = torch.cat([a, b, topkarr], 0)
        stack_list.append(hrt)
        if times % batchSize == 0:  # 表示需要切分
            x = torch.stack(stack_list, 0)
            batchScore = model.predict(x)
            Score = np.hstack((Score, batchScore))
            stack_list.clear()
        times = times + 1
    if stack_list:  # 不为空，表示还存在部分数据未输入
        x = torch.stack(stack_list, 0)
        batchScore = model.predict(x)
        # 分数追加
        Score = np.hstack((Score, batchScore))
        # 清空已有数据
        stack_list.clear()
    Score = Score.reshape(Score.shape[0], 1)
    return Score



def comp_res(score, trueEdge):
    reslist = []
    for i in range(trueEdge.size):
        Edge = [score[i], trueEdge[i]]
        reslist.append(Edge)
    resDF = pd.DataFrame(reslist, columns=['pre', 'true'])
    fpr, tpr, thresholds = roc_curve(resDF['true'], resDF['pre'], pos_label=1)
    auroc = auc(fpr, tpr)
    precision, recall, thresholds_PR = precision_recall_curve(resDF['true'], resDF['pre'])
    auprc = auc(recall, precision)
    # return auroc, auprc
    return auroc,auprc



if __name__ == '__main__':

    args = parse_args()
    dataPath = args.output
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    res = main(args)
    with open(f'{dataPath}/roc.txt', 'w') as f:
        f.write(res)
