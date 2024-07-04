import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import math
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.data import Data  
from sklearn.metrics import roc_auc_score
from torch_geometric.nn import GCNConv  
from torch_geometric.utils import negative_sampling  
import torch.optim as optm
from torch.nn import CosineSimilarity


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
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class STGRNS(nn.Module):
    def __init__(self, input_dim, nhead=8, d_model=200, num_classes=2, dropout=0.1):
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
    
class GRNCNN(nn.Module):  
    def __init__(self, input_channel=1, output_channel=64, kernel_size=3,stride=1,padding=1,num_classses=2):  
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
            nn.Linear(512, num_classses)
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
  
class ResNet18(nn.Module):  
    
    def __init__(self, num_classes=2):  
        super(ResNet18, self).__init__()  
        self.in_planes = 64  
  
        # 更改第一个卷积层的输入通道数为1  
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(64)  
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)  
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)  
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)  
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)  
        self.linear = nn.Linear(512 * 4, num_classes)  
  
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
        return out 
    
class GRNTrans_old(nn.Module):
    def __init__(self, input_size=60*60, hidden_size=256, num_layers=6, num_heads=8, dropout=0.1, num_classes=2):
        super(GRNTrans, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hidden_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, src):
        # src = src.view(src.size(0), -1)
        src = self.embedding(src)  # Embedding
        src = self.pos_encoder(src)  # 位置编码
        output = self.transformer_encoder(src)  # Transformer编码器
        output = torch.mean(output, dim=1)  # 对所有时间步的输出进行平均池化
        output = self.fc(output)  # 全连接层
        return output
  
class GRNTrans(nn.Module):
        
    def __init__(self, input_dim, num_head=6, d_model=60, hidden_size=256, num_layers = 6, num_classes=2, dropout=0.1,batch_first=True):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        print(batch_first)
        self.positionalEncoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_head, hidden_size, dropout,batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        out = self.embedding(x)
        out = self.positionalEncoding(out)
        out = self.transformer_encoder(out)
        out = torch.mean(out, dim=1) 
        out = self.fc(out)
        return out




class GRNGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, training=self.training,p = self.dropout)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_label_index):
        # z[edge_label_index[0]] (训练用的正样本数*2,64)   
        return (z[edge_label_index[0]] * z[edge_label_index[1]]).sum(dim=-1)  # product of a pair of nodes on each edge

    def decode_all(self, z):
        prob_adj = z @ z.t()
        return (prob_adj > 0).nonzero(as_tuple=False).t()

    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        z = self.decode(z, edge_label_index)
        return z

'''class GENELink(nn.Module):
    def __init__(self,input_dim,hidden1_dim,hidden2_dim,hidden3_dim,output_dim,num_head1,num_head2,
                 alpha,device,type,reduction):
        super(GENELink, self).__init__()
        self.num_head1 = num_head1
        self.num_head2 = num_head2
        self.device = device
        self.alpha = alpha
        self.type = type
        self.reduction = reduction

        if self.reduction == 'mean':
            self.hidden1_dim = hidden1_dim
            self.hidden2_dim = hidden2_dim
        elif self.reduction == 'concate':
            self.hidden1_dim = num_head1*hidden1_dim
            self.hidden2_dim = num_head2*hidden2_dim


        self.ConvLayer1 = [AttentionLayer(input_dim,hidden1_dim,alpha) for _ in range(num_head1)]
        for i, attention in enumerate(self.ConvLayer1):
            self.add_module('ConvLayer1_AttentionHead{}'.format(i),attention)

        self.ConvLayer2 = [AttentionLayer(self.hidden1_dim,hidden2_dim,alpha) for _ in range(num_head2)]
        for i, attention in enumerate(self.ConvLayer2):
            self.add_module('ConvLayer2_AttentionHead{}'.format(i),attention)

        self.tf_linear1 = nn.Linear(hidden2_dim,hidden3_dim)
        self.target_linear1 = nn.Linear(hidden2_dim,hidden3_dim)

        self.tf_linear2 = nn.Linear(hidden3_dim,output_dim)
        self.target_linear2 = nn.Linear(hidden3_dim, output_dim)



        if self.type == 'MLP':
            self.linear = nn.Linear(2*output_dim, 2)

        self.reset_parameters()

    def reset_parameters(self):
        for attention in self.ConvLayer1:
            attention.reset_parameters()

        for attention in self.ConvLayer2:
            attention.reset_parameters()

        nn.init.xavier_uniform_(self.tf_linear1.weight,gain=1.414)
        nn.init.xavier_uniform_(self.target_linear1.weight, gain=1.414)
        nn.init.xavier_uniform_(self.tf_linear2.weight, gain=1.414)
        nn.init.xavier_uniform_(self.target_linear2.weight, gain=1.414)




    def encode(self,x,adj):

        if self.reduction =='concate':
            x = torch.cat([att(x, adj) for att in self.ConvLayer1], dim=1)
            x = F.elu(x)

        elif self.reduction =='mean':
            x = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer1]), dim=0)
            x = F.elu(x)

        else:
            raise TypeError



        out = torch.mean(torch.stack([att(x, adj) for att in self.ConvLayer2]),dim=0)

        return out


    def decode(self,tf_embed,target_embed):

        if self.type =='dot':

            prob = torch.mul(tf_embed, target_embed)
            prob = torch.sum(prob,dim=1).view(-1,1)


            return prob

        elif self.type =='cosine':
            prob = torch.cosine_similarity(tf_embed,target_embed,dim=1).view(-1,1)

            return prob

        elif self.type == 'MLP':
            h = torch.cat([tf_embed, target_embed],dim=1)
            prob = self.linear(h)

            return prob
        else:
            raise TypeError(r'{} is not available'.format(self.type))


    def forward(self,x,adj,train_sample):

        embed = self.encode(x,adj)

        tf_embed = self.tf_linear1(embed)
        tf_embed = F.leaky_relu(tf_embed)
        tf_embed = F.dropout(tf_embed,p=0.01)
        tf_embed = self.tf_linear2(tf_embed)
        tf_embed = F.leaky_relu(tf_embed)

        target_embed = self.target_linear1(embed)
        target_embed = F.leaky_relu(target_embed)
        target_embed = F.dropout(target_embed, p=0.01)
        target_embed = self.target_linear2(target_embed)
        target_embed = F.leaky_relu(target_embed)

        self.tf_ouput = tf_embed
        self.target_output = target_embed


        train_tf = tf_embed[train_sample[:,0]]
        train_target = target_embed[train_sample[:, 1]]

        pred = self.decode(train_tf, train_target)

        return pred

    def get_embedding(self):
        return self.tf_ouput, self.target_output



class AttentionLayer(nn.Module):
    def __init__(self,input_dim,output_dim,alpha=0.2,bias=True):
        super(AttentionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha


        self.weight = nn.Parameter(torch.FloatTensor(self.input_dim, self.output_dim))
        self.weight_interact = nn.Parameter(torch.FloatTensor(self.input_dim,self.output_dim))
        self.a = nn.Parameter(torch.zeros(size=(2*self.output_dim,1)))


        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(self.output_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.weight_interact.data, gain=1.414)
        if self.bias is not None:
            self.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def _prepare_attentional_mechanism_input(self, x):

        Wh1 = torch.matmul(x, self.a[:self.output_dim, :])
        Wh2 = torch.matmul(x, self.a[self.output_dim:, :])
        e = F.leaky_relu(Wh1 + Wh2.T,negative_slope=self.alpha)
        return e


    def forward(self,x,adj):


        h = torch.matmul(x, self.weight)
        e = self._prepare_attentional_mechanism_input(h)

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj.to_dense()>0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        # attention = F.softmax(e, dim=1)

        attention = F.dropout(attention, training=self.training)
        h_pass = torch.matmul(attention, h)

        output_data = h_pass


        output_data = F.leaky_relu(output_data,negative_slope=self.alpha)
        output_data = F.normalize(output_data,p=2,dim=1)


        if self.bias is not None:
            output_data = output_data + self.bias

        return output_data'''



