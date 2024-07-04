from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from scGNN import GENELink
from torch.optim.lr_scheduler import StepLR
import scipy.sparse as sp
from utils import scRNADataset, load_data, adj2saprse_tensor, Evaluation,  Network_Statistic
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from PytorchTools import EarlyStopping
import numpy as np
import random
import glob
import os

import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('-epochs', type=int, default= 20, help='Number of epoch.')
parser.add_argument('-num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('-alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('-hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('-output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('-batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('-loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('-seed', type=int, default=8, help='Random seed')
parser.add_argument('-Type',type=str,default='dot', help='score metric')
parser.add_argument('-flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
parser.add_argument('-reduction',type=str,default='concate', help='how to integrate multihead attention')
parser.add_argument('-cuda',type=int,default='0', help='decide to use which gpu ')
parser.add_argument('-tf_path', required=False, default='/mnt/sdb/chenzhb/GRNDL_master/example/data/ecoli/TFlist.txt', help="The path that includes train data")
parser.add_argument('-gene_path', required=False, default='/mnt/sdb/chenzhb/GRNDL_master/example/data/ecoli/genelist.txt', help="The path that includes test data")
parser.add_argument('-network_path', required=False, default='/mnt/sdb/chenzhb/GRNDL_master/example/data/ecoli/Deeplearning data/train_val_test/train.txt', help="The output dir")
parser.add_argument('-valid_path', required=False, default='/mnt/sdb/chenzhb/GRNDL_master/example/data/ecoli/Deeplearning data/train_val_test/val.txt', help="The output dir")

parser.add_argument('-random_network_path', required=False,default='/mnt/sdb/chenzhb/GRNDL_master/example/data/ecoli/Deeplearning data/train_val_test/random_test.txt')
parser.add_argument('-tf_network_path', required=False,default='/mnt/sdb/chenzhb/GRNDL_master/example/data/ecoli/Deeplearning data/train_val_test/TF_test.txt')
parser.add_argument('-output_path', required=False,default='output/E.coli/', help="output dictionary")
parser.add_argument('-expr_path', required=False, default='/mnt/sdb/chenzhb/GRNDL_master/example/data/ecoli/Deeplearning data/expression.csv')

args = parser.parse_args()
seed = args.seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)


start_time=time.time()


def embed2file(tf_embed,tg_embed,gene_file,tf_path,target_path):
    tf_embed = tf_embed.cpu().detach().numpy()
    tg_embed = tg_embed.cpu().detach().numpy()
    print(tf_embed)
    gene_set = pd.read_csv(gene_file, index_col=0)
    tf_embed = pd.DataFrame(tf_embed,index=gene_set['Gene'].values)
    tg_embed = pd.DataFrame(tg_embed, index=gene_set['Gene'].values)
    tf_embed.to_csv(tf_path)
    tg_embed.to_csv(target_path)

def replace_with_index(arr, values):
    def replace(x):
        if x in values:
            return np.where(values == x)[0][0]
        return x
    return np.vectorize(replace)(arr)


exp_file = args.expr_path
tf_file = args.tf_path
target_file  =args.gene_path

train_file = args.network_path
val_file = args.valid_path
test1_file=args.random_network_path
test2_file=args.tf_network_path
output_path=args.output_path



data_input = pd.read_csv(exp_file,index_col=0)
loader = load_data(data_input)
feature = loader.exp_data()

#tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
tf1=pd.read_csv(tf_file,header=None).values
tf1=np.array(list(map(str.lower,np.concatenate(tf1))),dtype='object')

gene1=pd.read_csv(target_file,header=None).values
gene1=np.array(list(map(str.lower,np.concatenate(gene1))),dtype='object')
#target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
int_sequence = np.arange(1, len(gene1) + 1, dtype=np.int64)
target=np.array(int_sequence,dtype=np.int64)
result = np.zeros_like(tf1, dtype=int)
for i, a in enumerate(tf1):
    indices = np.where(gene1 == a)[0]  # 查找B中与A中字符串相同的位置
    if len(indices) > 0:
        result[i] = indices[0]

tf=result.astype(np.int64)

feature = torch.from_numpy(feature)
tf = torch.from_numpy(tf)







device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
data_feature = feature.to(device)
tf = tf.to(device)


#train_data = pd.read_csv(train_file, index_col=0).values
#validation_data = pd.read_csv(val_file, index_col=0).values
train_data = pd.read_csv(train_file, header=None,sep='\t').values
validation_data = pd.read_csv(val_file, header=None,sep='\t').values
train_data = np.char.lower(train_data.astype(str)).astype(object)
validation_data = np.char.lower(validation_data.astype(str)).astype(object)
train_data[:, 0] = replace_with_index(train_data[:, 0], gene1)
train_data[:, 1] = replace_with_index(train_data[:, 1], gene1)
validation_data[:, 0] = replace_with_index(validation_data[:, 0], gene1)
validation_data[:, 1] = replace_with_index(validation_data[:, 1], gene1)
test1_data = pd.read_csv(test1_file, header=None,sep='\t').values
test1_data = np.char.lower(test1_data.astype(str)).astype(object)
test1_data[:, 0] = replace_with_index(test1_data[:, 0], gene1)
test1_data[:, 1] = replace_with_index(test1_data[:, 1], gene1)

test2_data = pd.read_csv(test2_file, header=None,sep='\t').values
test2_data = np.char.lower(test2_data.astype(str)).astype(object)
test2_data[:, 0] = replace_with_index(test2_data[:, 0], gene1)
test2_data[:, 1] = replace_with_index(test2_data[:, 1], gene1)

train_data=train_data.astype(np.int64)
validation_data=validation_data.astype(np.int64)
test1_data=test1_data.astype(np.int64)
test2_data=test2_data.astype(np.int64)

train_load = scRNADataset(train_data, feature.shape[0], flag=args.flag)
adj = train_load.Adj_Generate(tf,loop=args.loop)
adj = adj2saprse_tensor(adj)

train_data = torch.from_numpy(train_data)
val_data = torch.from_numpy(validation_data)
test1_data = torch.from_numpy(test1_data)
test2_data = torch.from_numpy(test2_data)

model = GENELink(input_dim=feature.size()[1],
                hidden1_dim=args.hidden_dim[0],
                hidden2_dim=args.hidden_dim[1],
                hidden3_dim=args.hidden_dim[2],
                output_dim=args.output_dim,
                num_head1=args.num_head[0],
                num_head2=args.num_head[1],
                alpha=args.alpha,
                device=device,
                type=args.Type,
                reduction=args.reduction
                )


adj = adj.to(device)
model = model.to(device)
train_data = train_data.to(device)
validation_data = val_data.to(device)
test1_data=test1_data.to(device)
test2_data=test2_data.to(device)
optimizer = Adam(model.parameters(), lr=args.lr,weight_decay=1e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

model_path = output_path
if not os.path.exists(model_path):
    os.makedirs(model_path)

best_auc=0

for epoch in range(args.epochs):
    running_loss = 0.0

    for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
        model.train()
        optimizer.zero_grad()

        if args.flag:
            train_y = train_y.to(device)
        else:
            train_y = train_y.to(device).view(-1, 1)


        # train_y = train_y.to(device).view(-1, 1)
        pred = model(data_feature, adj, train_x)

        #pred = torch.sigmoid(pred)
        if args.flag:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        loss_BCE = F.binary_cross_entropy(pred, train_y)


        loss_BCE.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss_BCE.item()


    model.eval()
    score = model(data_feature, adj, validation_data)
    if args.flag:
        score = torch.softmax(score, dim=1)
    else:
        score = torch.sigmoid(score)

    # score = torch.sigmoid(score)
    
    AUC, AUPR, AUPR_norm = Evaluation(y_pred=score, y_true=validation_data[:, -1],flag=args.flag)
        #
    print('Epoch:{}'.format(epoch + 1),
            'train loss:{}'.format(running_loss),
            'AUC:{:.3F}'.format(AUC),
            'AUPR:{:.3F}'.format(AUPR))
    if AUC>best_auc:
        print("Save better model")
        best_auc=AUC
        torch.save(model.state_dict(), model_path +'/model.pkl')



model.load_state_dict(torch.load(model_path +'/model.pkl'))
model.eval()
#tf_embed, target_embed = model.get_embedding()
#embed2file(tf_embed,target_embed,target_file,tf_embed_path,target_embed_path)

score1 = model(data_feature, adj, train_data)
score2 = model(data_feature, adj, val_data)
score3 = model(data_feature, adj, test1_data)
score4 = model(data_feature, adj, test2_data)

if args.flag:
    score1 = torch.softmax(score1, dim=1)
    score2 = torch.softmax(score2, dim=1)
    score3 = torch.softmax(score3, dim=1)
    score4 = torch.softmax(score4, dim=1)
else:
    score1 = torch.sigmoid(score1)
    score2 = torch.sigmoid(score2)
    score3 = torch.sigmoid(score3)
    score4 = torch.sigmoid(score4)
# score = torch.sigmoid(score)




AUC1, AUPR1, AUPR_norm1= Evaluation(y_pred=score1, y_true=train_data[:, -1],flag=args.flag)
AUC2, AUPR2, AUPR_norm2= Evaluation(y_pred=score2, y_true=val_data[:, -1],flag=args.flag)
AUC3, AUPR3, AUPR_norm3= Evaluation(y_pred=score3, y_true=test1_data[:, -1],flag=args.flag)
AUC4, AUPR4, AUPR_norm4= Evaluation(y_pred=score4, y_true=test2_data[:, -1],flag=args.flag)
end_time=time.time()
execution_time=(end_time - start_time)/60
print('train_AUC:{}'.format(AUC1),
     'train_AUPRC:{}'.format(AUPR1),
     '\nval_AUC:{}'.format(AUC2),
     'val_AUPRC:{}'.format(AUPR2),
     '\ntest1_AUC:{}'.format(AUC3),
     'test1_AUPRC:{}'.format(AUPR3),
     '\ntest2_AUC:{}'.format(AUC4),
     'test2_AUPRC:{}'.format(AUPR4))

with open(f'{output_path}/auc_result.txt', 'w', newline='', encoding='utf-8') as file:
    file.write('train_eval_auc={:.5f}\ttrain_eval_aupr={:.5f}\n'.format(AUC1, AUPR1))
    file.write('val_auc={:.5f}\tval_aupr={:.5f}\n'.format(AUC2, AUPR2))
    file.write('random_test_auc={:.5f}\trandom_test_aupr={:.5f}\n'.format(AUC3, AUPR3))
    file.write('TF_test_auc={:.5f}\tTF_test_aupr={:.5f}\n'.format(AUC4, AUPR4))
    file.write('time={:.5f}'.format(execution_time))
























