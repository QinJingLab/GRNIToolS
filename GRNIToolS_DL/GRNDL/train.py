import os 
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data  
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv  
from torch_geometric.utils import negative_sampling  
from joblib import load
from param import *
from models import *

def trainning(data_dir, train_dir, train_dataloader, valid_dataloader, model_idx='1',args=None):
    # set hyperparameter
    dict_model = {'Trans': Config_Trans(), 'STGRNS':Config_STGRNS(), 'GRNCNN':Config_CNN(), 'ResNet18':Config_CNN(), 'GRNGCN':Config_GCN()}
    config = dict_model[model_idx]
    lr = args.lr if args.lr else config.learning_rate
    device = args.cuda if args.cuda else config.device
    epochs = args.epoch if args.epoch else config.epochs


    num_class =  config.num_classes
    d_models = config.dim_model
    n_heads = config.num_head
    dropout =  config.dropout
    n_hidden = config.hidden
    input_size = config.input_size
    #hidden_dim=args.hidden_dim
    # load dataloader
    if model_idx == 'GRNGCN':
        train_data = train_dataloader
        val_data = valid_dataloader
    # else:
        # train_dataloader = load(f'{data_dir}/train_dataloder')
        # valid_dataloader = load(f'{data_dir}/valid_dataloder')
    # create trainning dir

    if not os.path.isdir(train_dir):
        os.mkdir(train_dir)

    if model_idx != 'GRNGCN':

        if model_idx == 'STGRNS':
            model = STGRNS(input_dim=input_size, nhead=n_heads, d_model=d_models, num_classes=num_class, dropout=dropout)

        if model_idx == 'GRNCNN':
            model = GRNCNN()

        if model_idx == 'ResNet18':
            model = ResNet18()

        if model_idx == 'Trans':
            model = GRNTrans(input_dim = input_size, num_head = n_heads, hidden_size = n_hidden , d_model = d_models, dropout=dropout,batch_first=args.batch_first)


        print("Start Training models " , model_idx)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss() #二分类
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        best_acc = 0
        l1=args.l1
        for epoch in range(epochs):
            model.train()
            print('\nEpoch: %d' % (epoch + 1))
            sum_loss = 0 
            correct = 0
            total = 0 
            dat_len = len(train_dataloader)
            for step, datas in enumerate(train_dataloader):
                # prepare dataset
                data, label = datas
                label = label.view(-1)
                data = data.to(device)
                label = label.to(device)
                # forward & backward
                output = model(data)                # 前向传播
                loss = criterion(output, label) 
                #l1_reg = torch.tensor(0.).to(device)
                #for param in model.parameters():
                #    l1_reg += torch.norm(param, 1)
                #loss += l1 * l1_reg
                    # 计算loss
                optimizer.zero_grad()               # 清空上一层的梯度
                # grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  #梯度裁剪, 防止梯度爆炸
                loss.backward()                     # 反向传播
                optimizer.step()                    # 更新优化器的学习率         
                sum_loss += loss.item()
                score, pred_y = torch.max(output.data, 1)
                #print(output.data)
                total += label.size(0)
                correct += pred_y.eq(label.data).to(device).sum()
                if step % 20 == 0:
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (epoch + 1, (step + 1 + epoch * dat_len), sum_loss / (step + 1), 100. * correct / total))
            
            print('Waiting Test...')
            with torch.no_grad():
                correct = 0
                total = 0
                model.eval()
                for (data, label) in valid_dataloader:
                    # prepare dataset
                    data, label = data.to(device), label.view(-1).to(device)
                    output = model(data)
                    _, pred_y = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += (pred_y == label).sum()
                print('Val\'s ac is: %.3f%%' % (100 * correct / total))
            # 保存网络参数
            if (100 * correct / total) >= best_acc:
                best_acc = (100 * correct / total)
                print("better model")
                torch.save(model.state_dict(), f'{train_dir}/state_dict_model_best.pth')

        print('Train has finished, total epoch is %d' % (epoch+1))

    elif model_idx == 'GRNGCN':

    
        def negative_sample():
            # 从训练集中采样与正边相同数量的负边
            neg_edge_index = negative_sampling(
                edge_index=train_data.edge_index, num_nodes=train_data.num_nodes,
                num_neg_samples=train_data.edge_label_index.size(1), method='sparse')
            # print(neg_edge_index.size(1))   # 3642条负边，即每次采样与训练集中正边数量一致的负边
            edge_label_index = torch.cat(
                [train_data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                train_data.edge_label,
                train_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)

            return edge_label, edge_label_index

        def accuracy(pred, target):
            pred = (pred >= 0.6).float()  # 使用阈值0.5将预测值转换为二分类标签
            correct = pred.eq(target).sum().item()
            acc = correct / target.size(0)
            return acc

        best_auc = 0
        model = GRNGCN(train_data.num_features, n_hidden, 64)
        model = model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        model.train()
        for epoch in range(epochs):
            print('\nEpoch: %d' % (epoch + 1))
            optimizer.zero_grad()
            edge_label, edge_label_index = negative_sample()
            edge_label, edge_label_index =edge_label.to(device), edge_label_index.to(device) 
            F_mat, A_mat = train_data.x.to(device), train_data.edge_index.to(device) 
            out = model(F_mat, A_mat, edge_label_index).view(-1)
            out_sigmoid=out.sigmoid()
            auc=roc_auc_score(edge_label.cpu().detach().numpy(),out_sigmoid.cpu().detach().numpy())
            acc=accuracy(out_sigmoid,edge_label)
            loss = criterion(out.cpu(), edge_label.cpu())
            loss.backward()
            optimizer.step()


            # validation
            model.eval()
            with torch.no_grad():
                F_mat_valid, A_mat_valid = val_data.x.to(device), val_data.edge_index.to(device)
                edge_label2,edge_label_index2 = val_data.edge_label.to(device),val_data.edge_label_index.to(device)
                z=model.encode(F_mat_valid,A_mat_valid)
                #out2 = model(F_mat_valid, A_mat_valid, edge_label_index2).view(-1).sigmoid()
                out2=model.decode(z,edge_label_index2).view(-1).sigmoid()
                auc2 = roc_auc_score(val_data.edge_label.cpu().numpy(), out2.cpu().numpy())
                acc2=accuracy(out2.cpu(),val_data.edge_label.cpu())
                #print(out2.cpu().tolist())
                print('train_loss {:.3f} train_acc {:.3f} val_acc {:.3f} '.format(loss.item(), acc,acc2))
            #if acc2 >= best_auc:
             #   best_auc = auc2
                torch.save(model.state_dict(), f'{train_dir}/state_dict_model_best.pth')
