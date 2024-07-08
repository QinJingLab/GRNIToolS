from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings
import time
warnings.filterwarnings('ignore')
from sklearn import metrics
import os
import csv
import math
import argparse
from torch.utils.data import (DataLoader)
torch.set_default_tensor_type(torch.DoubleTensor)
from tqdm import tqdm
start_time=time.time()

parser = argparse.ArgumentParser(description="example")
parser = argparse.ArgumentParser(description="")
parser.add_argument('-cuda', required=False, default="0",type=int, help="The device used in model")
parser.add_argument('-epoch', type=int, required=False, default="200", help="The epoch used in model")
parser.add_argument('-lr', type=float, required=False, default="0.0003", help="The learning rate used in model")
parser.add_argument('-batch_size', type=int, required=False, default="32", help="The batch size used in model")
parser.add_argument('-train_data_path', required=False, default="data/mESC-PC/train_val_test/input/train/", help="The path that includes train data")
parser.add_argument('-test_data_path', required=False, default="data/mESC-PC/train_val_test/input/test/", help="The path that includes test data")
parser.add_argument('-output_dir', required=False, default="data/mESC-PC/train_val_test/output/", help="The output dir")
parser.add_argument('-test_gene_pair', required=False, default="data/mESC-PC/train_val_test/test.txt", help="The test gene pair")
parser.add_argument('-to_predict',default='False', help="True or False. Default is False, then the code will do training. If set to True, we need to indicate weight_path for a trained model and the code will do prediction based on the trained model.")
parser.add_argument('-weight_path', default=None, help="The path for a trained model.")

args=parser.parse_args()

device=torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')
print(device)
def numpy2loader(X, y, batch_size):
    X_set = torch.from_numpy(X).to(device)
    X_loader = DataLoader(X_set, batch_size=batch_size)
    y_set = torch.from_numpy(y).to(device)
    y_loader = DataLoader(y_set, batch_size=batch_size)

    return X_loader, y_loader

def loaderToList(data_loader):
    length = len(data_loader)
    data = []
    for i in data_loader:
        data.append(i)
    return data

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



def STGRNSForGRNSRconstruction(batch_sizes, epochs,known_data_path,unknown_data_path,num_threads):
    data_path = known_data_path
    d_models = epochs
    torch.set_num_threads(50) #set num_threads
    batch_size = batch_sizes
    log_dir = args.output_dir
    if (not os.path.isdir(log_dir)):
        os.makedirs(log_dir)

    x_train = np.load(data_path + 'matrix.npy')
    y_train = np.load(data_path + 'label.npy')
    
    X_trainloader, y_trainloader = numpy2loader(x_train, y_train, batch_size)

    X_trainList = loaderToList(X_trainloader)
    y_trainList = loaderToList(y_trainloader)
 ##ues file data to calculate accuracy
    x2_train=np.load(data_path + 'matrix2.npy')
    y2_train = np.load(data_path + 'label2.npy')

    X2_trainloader, y2_trainloader = numpy2loader(x2_train, y2_train, batch_size)
 
    X2_trainList = loaderToList(X2_trainloader)
    y2_trainList = loaderToList(y2_trainloader)

    model = STGRNS(input_dim=200, nhead=2, d_model=d_models, num_classes=2).to(device)
    model_on_cuda = next(model.parameters()).is_cuda
    if model_on_cuda:
        print("model load in CUDA")
    else:
        print("model load out of CUDA")
    print
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, foreach=False)


    n_epochs=args.epoch
    acc_record = {'train': [], 'dev': []}
    loss_record = {'train': [], 'dev': []}
    model.train()
##save different accuracy model
    m0=0
    m1=0
    m2=0
    m3=0
    m4=0
    m5=0
    m6=0
    timefile=args.output_dir+'/model_time'
    for epoch in range(n_epochs):
        print('epoch '+str(epoch+1))
        train_loss = []
        correct=0
        total=0
        progress_bar = tqdm(range(len(X_trainList)), desc=f"[ Train | {epoch + 1:03d}/{n_epochs:03d}]",ncols=80, position=0)
        for j in progress_bar:
            data = X_trainList[j].to(device)
            labels = y_trainList[j].to(device)
            logits = model(data)
            labels = torch.tensor(labels, dtype=torch.long)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
            optimizer.step()
            acc = (logits.argmax(dim=-1) == labels).float().mean()
            train_loss.append(loss.item())
            progress_bar.set_postfix({'loss': train_loss[-1]})
#            correct+=(logits.argmax(dim=-1)==labels).sum().item()
#            total+=labels.size(0)
        model.eval()
        y_predicta=[]
        y_predict=[]
        predictions=[]
        for k in range(0,len(X2_trainList)):
            data2=X2_trainList[k].to(device)
            with torch.no_grad():
                logits=model(data2)
            predt = F.softmax(logits)
            temps=predt.cpu().numpy().tolist()
            for i in temps:
                t=i[1]
                y_predicta.append(t)
        y_predict=y_predicta[:len(y2_train)]
        y_predict=np.array(y_predict)
        fpr, tpr, thresholds = metrics.roc_curve(y2_train, y_predict, pos_label=1)
        auc=metrics.auc(fpr,tpr)
        print("auc="+str(auc))
       # bacc = metrics.balanced_accuracy_score(y_test, y_predict2)
        precision, recall, thresholds_PR = metrics.precision_recall_curve(y2_train, y_predict)
        AUPR = metrics.auc(recall, precision)
        print("aupr="+str(AUPR))
        y_predict2 = []
        for pre in y_predict:
            if(pre >0.001):
                y_predict2.append(1)
            else:
                y_predict2.append(0)
        acc = metrics.accuracy_score(y2_train, y_predict2)
        if (acc >= 0.7)and (m0==0):
            torch.save(model.state_dict(),log_dir+'0.7model.pth')
            m0=m0+1
            print("Saved model to"+log_dir+'0.7model.pth')
            end_time=time.time()
            execution_time=(end_time - start_time)/60
            with open(timefile,'a',newline='',encoding='utf-8')as file:
                file.write('epoch\t'+str(epoch)+'\ttime\t'+str(execution_time)+'min\n')
        if (acc>=0.75)and(m1==0):
            torch.save(model.state_dict(),log_dir+'0.75model.pth')
            m1=m1+1
            print("Saved model to"+log_dir+'0.75model.pth')
            end_time=time.time()
            execution_time=(end_time - start_time)/60
            with open(timefile,'a',newline='',encoding='utf-8')as file:
                file.write('epoch\t'+str(epoch)+'\ttime\t'+str(execution_time)+'min\n')
        if (acc>=0.8)and(m2==0):
            torch.save(model.state_dict(),log_dir+'0.8model.pth')
            m2=m2+1
            print("Saved model to"+log_dir+'0.8model.pth')
            end_time=time.time()
            execution_time=(end_time - start_time)/60
            with open(timefile,'a',newline='',encoding='utf-8')as file:
                file.write('epoch\t'+str(epoch)+'\ttime\t'+str(execution_time)+'min\n')
        if (acc>=0.85)and(m3==0):
            torch.save(model.state_dict(),log_dir+'0.85model.pth')
            m3=m3+1
            print("Saved model to"+log_dir+'0.85model.pth')
            end_time=time.time()
            execution_time=(end_time - start_time)/60
            with open(timefile,'a',newline='',encoding='utf-8')as file:
                file.write('epoch\t'+str(epoch)+'\ttime\t'+str(execution_time)+'min\n')
        if (acc>=0.9)and(m4==0):
            torch.save(model.state_dict(),log_dir+'0.9model.pth')
            m4=m4+1
            end_time=time.time()
            execution_time=(end_time - start_time)/60
            with open(timefile,'a',newline='',encoding='utf-8')as file:
                file.write('epoch\t'+str(epoch)+'\ttime\t'+str(execution_time)+'min\n')
        if (acc>=0.95)and(m5==0):
            torch.save(model.state_dict(),log_dir+'0.95model.pth')
            m5=m5+1
            print("Saved model to"+log_dir+'0.95model.pth')
            end_time=time.time()
            execution_time=(end_time - start_time)/60
            with open(timefile,'a',newline='',encoding='utf-8')as file:
                file.write('epoch\t'+str(epoch)+'\ttime\t'+str(execution_time)+'min\n')
        if (acc>=0.99)and(m6==0):
            torch.save(model.state_dict(),log_dir+'0.99model.pth')
            m6=m6+1
            print("Saved model to"+log_dir+'0.99model.pth')
            end_time=time.time()
            execution_time=(end_time - start_time)/60
            with open(timefile,'a',newline='',encoding='utf-8')as file:
                file.write('epoch\t'+str(epoch)+'\ttime\t'+str(execution_time)+'min\n')
            break
        print("acc="+str(acc))
        train_loss = sum(train_loss) / len(train_loss)
        loss_record['train'].append(train_loss)
        acc_record['train'].append(acc)
        print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f} accuracy={acc:.5f}")
    model.train()
    torch.save(model.state_dict(),log_dir+'model.pth')
    ###predict-----------------------------------------------
#    y_predict = []
 #   data_path = unknown_data_path
   # x_test = np.load(data_path + 'matrix.npy')
    #y_test = np.load(data_path + 'label.npy')
   # gene_pair=np.load(data_path + 'gene_pair.npy')
    #X_testloader, y_testloader = numpy2loader(x_test, y_test, batch_size)

#    X_testList = loaderToList(X_testloader)
#    y_testList = loaderToList(y_testloader)

#    model.eval()
#    predictions = []
#    for k in range(0, len(X_testList)):
#        data = X_testList[k]
#        with torch.no_grad():
#            logits = model(data)
#        predt = F.softmax(logits)
#        temps = predt.cpu().numpy().tolist()
#        for i in temps:
#            t = i[1]
#            y_predict.append(t)

    #print("y_predict", y_predict)

#    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
 #   auc = metrics.auc(fpr, tpr)
#    print("auc="+str(auc))
#    precision, recall, thresholds_PR = metrics.precision_recall_curve(y_test, y_predict)
#    AUPR = metrics.auc(recall, precision)
#    print("aupr="+str(AUPR))
#    y_predict2 = []
#    for pre in y_predict:
#        if(pre >0.001):
#            y_predict2.append(1)
#        else:
#            y_predict2.append(0)
#    acc = metrics.accuracy_score(y_test, y_predict2)
#    print("acc="+str(acc))
#    bacc = metrics.balanced_accuracy_score(y_test, y_predict2)
#    print("bacc="+str(bacc))
#    f1 = metrics.f1_score(y_test, y_predict2)
#    output_file1=log_dir+'test_predictions.csv'
#    output_file2=log_dir+'test_predictions_auc.txt'

#    gene_pairs_set=set()
#    with open(args.test_gene_pair,'r',newline='',encoding='utf-8') as second_csvfile:
#        reader=csv.reader(second_csvfile)
#        for row in reader:
#            gene_pairs_set.add((row[0],row[1]))
#    print('length'+str(len(y_predict)))
#    with open(output_file1,'w',newline='',encoding='utf-8')as csvfile:
#        writer=csv.writer(csvfile)
#        for i in range(len(y_predict)):
#          gene_names=gene_pair[i].split(',')
#          if (gene_names[0], gene_names[1])in gene_pairs_set:
#            data=[gene_names[0],gene_names[1],y_predict[i]]
#            writer.writerow(data)

#    output_file2=open(log_dir+'test_predictions_auc.txt','w')
  #  output_file2.write("auc="+str(auc))
 #   output_file2.write("aupr="+str(AUPR))
 #   output_file2.write("acc="+str(acc))
 #   output_file2.write("bacc="+str(bacc))
 #   output_file2.close()
 #   ##storing the predicted data
#    np.save(log_dir + 'y_test.npy', y_test)
  #  np.save(log_dir + 'y_predict.npy', y_predict)
    
     ##storing the predicted network
  #  np.save(log_dir + 'y_predict2.npy', y_predict2)

def modelpredict(batch_sizes, epochs,known_data_path,unknown_data_path,num_threads):
    print('modelpredict!')
    data_path = unknown_data_path
    d_models = epochs
    torch.set_num_threads(50) #set num_threads
    batch_size = batch_sizes
    log_dir = args.output_dir
    if (not os.path.isdir(log_dir)):
        os.makedirs(log_dir)
    model = STGRNS(input_dim=200, nhead=2, d_model=d_models, num_classes=2).to(device)
    model.load_state_dict(torch.load(args.weight_path,map_location='cuda:0'))
    y_predict = []
    data_path = unknown_data_path
    x_test = np.load(data_path + 'matrix2.npy')
    y_test = np.load(data_path + 'label2.npy')
    gene_pair=np.load(data_path + 'gene_pair2.npy')
    X_testloader, y_testloader = numpy2loader(x_test, y_test, batch_size)

    X_testList = loaderToList(X_testloader)
    y_testList = loaderToList(y_testloader)

    model.eval()
    predictions = []
    for k in range(0, len(X_testList)):
        data = X_testList[k]
        with torch.no_grad():
            logits = model(data)
        predt = F.softmax(logits)
        temps = predt.cpu().numpy().tolist()
        for i in temps:
            t = i[1]
            y_predict.append(t)

    #print("y_predict", y_predict)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_predict, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print("auc="+str(auc))
    precision, recall, thresholds_PR = metrics.precision_recall_curve(y_test, y_predict)
    AUPR = metrics.auc(recall, precision)
    print("aupr="+str(AUPR))
    y_predict2 = []
    for pre in y_predict:
        if(pre >0.001):
            y_predict2.append(1)
        else:
            y_predict2.append(0)
    acc = metrics.accuracy_score(y_test, y_predict2)
    print("acc="+str(acc))
    bacc = metrics.balanced_accuracy_score(y_test, y_predict2)
    print("bacc="+str(bacc))
    f1 = metrics.f1_score(y_test, y_predict2)
    output_file1=log_dir+'test_predictions.csv'
    output_file2=log_dir+'test_predictions_auc.txt'

    gene_pairs_set=set()
    with open(args.test_gene_pair,'r',newline='',encoding='utf-8') as second_txtfile:
        lines=second_txtfile.readlines()
        for line in lines:
            row=line.strip().split('\t')
            gene_pairs_set.add((row[0],row[1]))
    print('length'+str(len(y_predict)))
    with open(output_file1,'w',newline='',encoding='utf-8')as csvfile:
        writer=csv.writer(csvfile)
        for i in range(len(y_predict)):
          gene_names=gene_pair[i].split(',')
          if (gene_names[0], gene_names[1])in gene_pairs_set:
            data=[gene_names[0],gene_names[1],y_predict[i]]
            writer.writerow(data)

    output_file2=open(log_dir+'test_predictions_auc.txt','w')
    output_file2.write("auc="+str(auc))
    output_file2.write("aupr="+str(AUPR))
    output_file2.write("acc="+str(acc))
    output_file2.write("bacc="+str(bacc))
    output_file2.close()
    ##storing the predicted data
    np.save(log_dir + 'y_test.npy', y_test)
    np.save(log_dir + 'y_predict.npy', y_predict)
    
     ##storing the predicted network
    np.save(log_dir + 'y_predict2.npy', y_predict2)





##the data path of known data
data_path = args.train_data_path

##the data path of unknown data
unknown_data_path = args.test_data_path
num_threads = 1
##training model and then predicting unknown network
batch_sizes = args.batch_size
epochs = 200
if args.to_predict =='False':
  print('training!')
  STGRNSForGRNSRconstruction(batch_sizes,epochs,data_path,unknown_data_path,num_threads)

if args.to_predict =='True':
  print('predict!')
  modelpredict(batch_sizes,epochs,data_path,unknown_data_path,num_threads)
  
end_time=time.time()
execution_time=(end_time - start_time)/60
timefile=args.output_dir+'time'
with open(timefile,'w',newline='',encoding='utf-8')as file:
    file.write(str(execution_time))