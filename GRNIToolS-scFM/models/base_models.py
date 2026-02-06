
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
from torch_geometric.data import Data  
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, average_precision_score

class GRNCNN(nn.Module):  
    def __init__(self, input_channel=1, output_channel=64, kernel_size=3,stride=1,padding=1,hidden =128):  
        super(GRNCNN, self).__init__()  
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(output_channel, output_channel*2, kernel_size=kernel_size, stride=stride, padding=padding),  
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 15 * 15, 512),
            nn.ReLU(),
            nn.Linear(512, hidden),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):  
        # 输入x的shape应为[batch_size, 1, 60, 60]  
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

class BasicBlock(nn.Module):  
    expansion = 1  
  
    def __init__(self, in_planes, planes, stride=1):  
        super(BasicBlock, self).__init__()  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes)  
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(planes)  
  
        self.shortcut = nn.Sequential()  
        if stride != 1 or in_planes != self.expansion * planes:  
            self.shortcut = nn.Sequential(  
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),  
                nn.BatchNorm2d(self.expansion * planes)  
            )  
  
    def forward(self, x):  
        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.bn2(self.conv2(out))  
        out += self.shortcut(x)  
        out = F.relu(out)  
        return out  
  
class GRNResNet(nn.Module):  
    
    def __init__(self, num_classes=128):  
        super(GRNResNet, self).__init__()  
        self.in_planes = 64  
  
        # 更改第一个卷积层的输入通道数为1  
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(64)  
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  
        self.linear = nn.Linear(512 * 4, num_classes)  
        self.fc = nn.Linear(num_classes,1)


    def _make_layer(self, block, planes, num_blocks, stride):  
        strides = [stride] + [1] * (num_blocks - 1)  
        layers = []  
        for stride in strides:  
            layers.append(block(self.in_planes, planes, stride))  
            self.in_planes = planes * block.expansion  
        return nn.Sequential(*layers)  
  
    def forward(self, x):  
        out = F.relu(self.bn1(self.conv1(x)))  
        out = self.layer1(out)  
        out = self.layer2(out)  
        out = self.layer3(out)  
        out = self.layer4(out)  
        out = F.avg_pool2d(out, 4)  
        out = out.view(out.size(0), -1)  
        out = self.linear(out)  
        out = self.fc(out)
        return out 
    
class GRNGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden, out_channels):
        super(GRNGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden)
        self.conv2 = GCNConv(hidden, out_channels)

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        return self.conv2(x, edge_index)

    def decode(self, z, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)  # [2,E]  合并pos和neg的边
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)  # *：element-wise乘法 node1的特征和node2的特征点乘求和，获取两个节点存在边的概率

    def decode_all(self, z):
        prob_adj = z @ z.t()  # @：矩阵乘法，自动执行适合的矩阵乘法函数
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, pos_edge_index, neg_edge_index):
        return self.decode(self.encode(x, pos_edge_index), pos_edge_index, neg_edge_index)


class GRNMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim1 = 1024, hidden_dim2 = 256 , output_dim = 1):
        super(GRNMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)
        self.sigmoid = nn.Sigmoid() 
 
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        # out = self.sigmoid(out)
        return out   
    

class GRNTrans(nn.Module):
    def __init__(self, input_dim=805, d_model=128, nhead=2, num_layers=3):
        super().__init__()
        # 共享编码器
        self.gene_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU()
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.randn(1, 2, d_model))
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        # 编码基因特征 [batch_size, d_model]
        gene1_embed = self.gene_encoder(x1)
        gene2_embed = self.gene_encoder(x2)
        
        # 创建序列并添加位置编码 [batch_size, seq_len=2, d_model]
        sequence = torch.stack([gene1_embed, gene2_embed], dim=1)
        sequence += self.pos_encoder
        
        # 通过Transformer编码器
        encoded = self.transformer_encoder(sequence)
        
        output = self.classifier(encoded[:, 0])
        return output