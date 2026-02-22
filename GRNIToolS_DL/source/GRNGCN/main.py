import torch
import os
import argparse
import pandas as pd
import numpy as np
import random
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data  
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score

class GCNnet(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super(GCNnet, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward_ward(self, x, pos_edge_index, neg_edge_index):
        z = self.encode(x, pos_edge_index)
        return self.decode(z, pos_edge_index, neg_edge_index)

    def forward(self, data):
        x, pos_edge_index, neg_edge_index = data.x, data.pos_edge_index, data.neg_edge_index
        p = self.forward_ward(x, pos_edge_index, neg_edge_index)
        return p

def load_data_edge(feature, edge_file, dict_gene_id):
    df = pd.read_csv(edge_file, header=None, delimiter='\t')
    pos_edge = [[], []]
    neg_edge = [[], []]
    for index, row in df.iterrows():
        tf, gene, label = row
        tf, gene, label = str(tf).upper(), str(gene).upper(), str(label).upper()
        if tf in dict_gene_id and gene in dict_gene_id:
            tf, gene = dict_gene_id[tf], dict_gene_id[gene]
        else:
            continue
        if label == '1':
            pos_edge[0].append(tf)
            pos_edge[1].append(gene)
        elif label == '0':
            neg_edge[0].append(tf)
            neg_edge[1].append(gene)
    pos_edge = torch.tensor(pos_edge, dtype=torch.long)
    neg_edge = torch.tensor(neg_edge, dtype=torch.long)
    return pos_edge, neg_edge


def load_data_expr(expr_file, texp=False):
    df = pd.read_csv(expr_file, header=0, index_col=0)
    if texp:
        df = df.T
    gene_list = df.index
    gene_idx = dict(zip([i.upper() for i in gene_list.tolist()], range(len(gene_list))))
    feature = torch.from_numpy(df.to_numpy()).float()
    return feature, gene_idx


class EdgeDataset(Dataset):
    def __init__(self, pos_edge_index, neg_edge_index):
        assert pos_edge_index.size(1) > 0, "必须包含正样本"
        assert neg_edge_index.size(1) > 0, "必须包含负样本"
        self.pos_edge_index = pos_edge_index
        self.neg_edge_index = neg_edge_index

    def __len__(self):
        return self.pos_edge_index.shape[1] + self.neg_edge_index.shape[1]

    def __getitem__(self, idx):
        if idx < self.pos_edge_index.shape[1]:
            return self.pos_edge_index[:, idx], 1
        else:
            idx -= self.pos_edge_index.shape[1]
            return self.neg_edge_index[:, idx], 0


def collate_fn(batch):
    # pos_edge_index = torch.cat([edge.unsqueeze(1) for edge, label in batch if label == 1], dim=1)
    # neg_edge_index = torch.cat([edge.unsqueeze(1) for edge, label in batch if label == 0], dim=1)
    # labels = torch.tensor([label for _, label in batch], dtype=torch.float)
    # return pos_edge_index, neg_edge_index, labels

    pos_edges = [edge.unsqueeze(1) for edge, label in batch if label == 1]
    pos_edge_index = torch.cat(pos_edges, dim=1) if pos_edges else torch.tensor([], dtype=torch.long).view(2, 0)
    
    neg_edges = [edge.unsqueeze(1) for edge, label in batch if label == 0]
    neg_edge_index = torch.cat(neg_edges, dim=1) if neg_edges else torch.tensor([], dtype=torch.long).view(2, 0)
    
    labels = torch.tensor([label for _, label in batch], dtype=torch.float)
    return pos_edge_index, neg_edge_index, labels


def parser_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr', type=str, default='./Input_data/Ecoli/GRNDL/expression.csv', help='Gene expression file')
    parser.add_argument('--texp', type=bool, default=False, help='Matrix transposition, if row-cell, col-gene, set to True')
    parser.add_argument('--train', type=str, default='./Input_data/Ecoli/GRNDL/train.txt', help='Training data file')
    parser.add_argument('--valid', type=str, default='./Input_data/Ecoli/GRNDL/val.txt', help='Validation data file')
    parser.add_argument('--test', type=str, default='./Input_data/Ecoli/GRNDL/random_test.txt', help='Test data file')
    parser.add_argument('--tftest', type=str, default='./Input_data/Ecoli/GRNDL/tftest.txt', help='TF test data file')
    parser.add_argument("--tf", type = str, default='./Input_data/ecoli/GRNDL/TFlist.tsv', help="<file> Input tf list file")
    parser.add_argument("--gene", type = str, default='./Input_data/ecoli/GRNDL/genelist.tsv', help="<file> Input gene list file")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--hidden', type=int, default=256, help='Number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--output', type=str, default='model', help='Output directory')
    parser.add_argument('--cuda_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES (comma-separated), e.g. 0,1,3')

    args = parser.parse_args()
    return args

@torch.no_grad()
def predict(data, model, pos_edge_index, device):
    model.eval()
    data.x, pos_edge_index = data.x.to(device), pos_edge_index.to(device)
    z = model.encode(data.x, pos_edge_index)
    link_pred = model.decode(z, data.pos_edge_index, data.neg_edge_index)
    labels = data.label.to(device)  
    link_pred = link_pred.sigmoid()
    res_auc = roc_auc_score(labels.cpu(), link_pred.cpu())
    res_aupr = average_precision_score(labels.cpu(), link_pred.cpu())
    return round(res_auc, 5), round(res_aupr, 5)


def main():
    args = parser_opt()
    feature, dict_gene_id = load_data_expr(args.expr, args.texp)

    train_pos_edge, train_neg_edge = load_data_edge(feature, args.train, dict_gene_id)
    val_pos_edge, val_neg_edge = load_data_edge(feature, args.valid, dict_gene_id)
    test_pos_edge, test_neg_edge = load_data_edge(feature, args.test, dict_gene_id)
    tftest_pos_edge, tftest_neg_edge = load_data_edge(feature, args.tftest, dict_gene_id)

    train_dataset = EdgeDataset(train_pos_edge, train_neg_edge)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)



    # 设备初始化部分
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    device_ids = list(map(int, args.cuda_devices.split(',')))
    model = GCNnet(feature.shape[1], args.hidden, 64)
    # 自动判断是否并行
    if len(device_ids) > 1:
        device = torch.device(f"cuda:{device_ids[0]}")
        from torch_geometric.nn import DataParallel
        model = DataParallel(model)
    else:
        device = torch.device(f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = F.binary_cross_entropy_with_logits
    x=feature.to(device)
    best_val_auc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for pos_edge_index, neg_edge_index, labels in train_loader:
            pos_edge_index, neg_edge_index, labels = pos_edge_index.to(device), neg_edge_index.to(device), labels.to(device)
            optimizer.zero_grad()
            # 创建 Data 对象，包含所有必要的信息
            data = Data(
                x= x,  # 节点特征
                pos_edge_index=pos_edge_index,  # 正样本边
                neg_edge_index=neg_edge_index,  # 负样本边
                num_nodes=feature.size(0)  # 明确设置节点数量
            )
            if not hasattr(data, 'num_nodes'):
                data.num_nodes = feature.size(0)

            link_pred = model(data)
            loss = criterion(link_pred, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            z = model.encode(feature.to(device), train_pos_edge.to(device))
            link_pred = model.decode(z, val_pos_edge.to(device), val_neg_edge.to(device))
            val_loss = criterion(link_pred, torch.cat([torch.ones(val_pos_edge.shape[1]), torch.zeros(val_neg_edge.shape[1])]).to(device))
            link_pred = link_pred.sigmoid()
            val_auc = roc_auc_score(torch.cat([torch.ones(val_pos_edge.shape[1]), torch.zeros(val_neg_edge.shape[1])]).cpu(), link_pred.cpu())

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), f'{args.output}/model_parameters.pt')

        print(f'Epoch: {epoch+1:03d}, Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')


    train_labels = torch.cat([torch.ones(train_pos_edge.shape[1]), torch.zeros(train_neg_edge.shape[1])]).float()
    train_data = Data(x=feature, pos_edge_index=train_pos_edge, neg_edge_index=train_neg_edge, label=train_labels)

    val_labels = torch.cat([torch.ones(val_pos_edge.shape[1]), torch.zeros(val_neg_edge.shape[1])]).float()
    val_data = Data(x=feature, pos_edge_index=val_pos_edge, neg_edge_index=val_neg_edge, label=val_labels)

    test_labels = torch.cat([torch.ones(test_pos_edge.shape[1]), torch.zeros(test_neg_edge.shape[1])]).float()
    test_data = Data(x=feature, pos_edge_index=test_pos_edge, neg_edge_index=test_neg_edge, label=test_labels)

    tftest_labels = torch.cat([torch.ones(tftest_pos_edge.shape[1]), torch.zeros(tftest_neg_edge.shape[1])]).float()
    tftest_data = Data(x=feature, pos_edge_index=tftest_pos_edge, neg_edge_index=tftest_neg_edge, label=tftest_labels)

    valid_auc, valid_aupr = predict(val_data, model, train_pos_edge, device)
    test_auc, test_aupr = predict(test_data, model, train_pos_edge, device)
    tftest_auc, tftest_aupr = predict(tftest_data, model, train_pos_edge, device)
    
    res = 'method\tdataset\tauroc\taupr\n'
    res += f'GRNGCN\tvalid\t{valid_auc}\t{valid_aupr}\n'
    res += f'GRNGCN\ttest\t{test_auc}\t{test_aupr}\n'       
    res += f'GRNGCN\ttftest\t{tftest_auc}\t{tftest_aupr}\n'

    with open(f'{args.output}/roc.txt', 'w') as f:
        f.write(res)
    print(res)

if __name__ == "__main__":
    main()