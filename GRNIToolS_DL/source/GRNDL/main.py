import os 
import numpy as np
import argparse 
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data.dataloader as DataLoader
import torch.utils.data.dataset as Dataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from joblib import load
from sklearn.metrics import roc_auc_score, average_precision_score
import data_generator
import network
from models import *


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--expr', type=str, default='./Input_data/Ecoli/GRNDL/expression.csv', help='Gene expression file')
    parser.add_argument('--texp', type=bool, default=False, help='Matrix transposition, if row-cell, col-gene, set to True')
    parser.add_argument('--train', type=str, default='./Input_data/Ecoli/GRNDL/train.txt', help='Training data file')
    parser.add_argument('--valid', type=str, default='./Input_data/Ecoli/GRNDL/val.txt', help='Validation data file')
    parser.add_argument('--test', type=str, default='./Input_data/Ecoli/GRNDL/random_test.txt', help='Testing data file')
    parser.add_argument('--tftest', type=str, default='./Input_data/Ecoli/GRNDL/TF_test.txt', help='TF testing data file')
    parser.add_argument('--batch_size', type=int, required=False, default=128, help="The batch size used in model")
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('--head', type=int, default=6, help='Number of head units')
    parser.add_argument('--output_dir', type=str, default='GRNDL_output', help='Output file')
    parser.add_argument('--model', type=str, default='GRNCNN', help='DL model choice')
    parser.add_argument('--network', type=str, default='0', help='GRNDL network prediction, default is False')
    parser.add_argument('--cuda_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES (comma-separated), e.g. 0,1,3')
    
    args = parser.parse_args()
    return args


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # 显式设置当前设备


def predict(input_dataloader, args, save_path):
    model_pth = f'{save_path}/state_dict_model_best.pth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    if args.model == 'GRNResNet':
        model = GRNResNet()
    elif args.model == 'GRNCNN':
        model = GRNCNN()
    model.load_state_dict(torch.load(model_pth, weights_only=True))
    model = model.to(device)
    # model.load_state_dict(torch.load(model_pth))
    all_label = []
    all_link_pred = []    
    model.eval()
    with torch.no_grad():
        for (data, label) in input_dataloader:
            label = label.to(torch.float32)
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            link_pred = output.sigmoid()
            all_label.extend(label.cpu().numpy().tolist())
            all_link_pred.extend(link_pred.detach().cpu().numpy().tolist())        


    auc = roc_auc_score(all_label,all_link_pred)    
    aupr = average_precision_score(all_label, all_link_pred)
    return round(auc,5), round(aupr,5)



def train(rank, world_size, args, train_data, valid_data):
    setup(rank, world_size)
    torch.manual_seed(0)  
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = DataLoader.DataLoader(train_data, batch_size=args.batch_size, sampler=train_sampler)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
    valid_loader = DataLoader.DataLoader(valid_data, batch_size=args.batch_size, sampler=valid_sampler)

    model_name = args.model
    lr = args.lr
    epochs = args.epochs
    hidden_size= args.hidden
    dropout = args.dropout
    n_head = args.head
    if model_name == 'GRNResNet':
        model = GRNResNet().to(rank)
    elif model_name == 'GRNCNN':
        model = GRNCNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])   


    criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss() #二分类
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = F.binary_cross_entropy_with_logits 
    total_start = time.time()
    for epoch in range(epochs):
        model.train()
        # print('\nEpoch: %d' % (epoch + 1))
        sum_loss = 0 
        total = 0 
        sum_train_auc = 0
        for step, datas in enumerate(train_loader):
            # prepare dataset
            data, label = datas
            label = label.to(torch.float32)
            data = data.to(rank)
            label = label.to(rank)
            # forward & backward
            output = ddp_model(data)                
            loss = criterion(output, label)     
            optimizer.zero_grad()               
            loss.backward()                    
            optimizer.step()                    
            sum_loss += loss.item()
            link_pred = output.sigmoid()
            if len(np.unique(label.cpu())) < 2 or len(np.unique(link_pred.detach().cpu())) < 2:
                train_auc = 0.5        
            else:         
                train_auc = roc_auc_score(label.cpu(),link_pred.detach().cpu())        

            total += label.size(0)
            sum_train_auc += train_auc


        if rank == 0 and (epoch+10) % 5 == 0:
            print('[epoch:%d, Train]  Loss: %.03f | Auc: %.3f ' % (epoch + 1,  sum_loss / (step + 1), sum_train_auc / (step + 1)))
            # print('Waiting Test...')
        with torch.no_grad():
            total = 0
            model.eval()
            all_label = []
            all_link_pred = []
            for (data, label) in valid_loader:
                # prepare dataset
                # data, label = data.to(device), label.view(-1).to(device)
                label = label.to(torch.float32)
                data = data.to(rank)
                label = label.to(rank)
                output = model(data)
                link_pred = output.sigmoid()
                all_label.extend(label.cpu().numpy().tolist())
                all_link_pred.extend(link_pred.detach().cpu().numpy().tolist())
         
            valid_auc = roc_auc_score(all_label,all_link_pred)       
        if rank == 0 and (epoch+10) % 5 == 0:      
            print('[epoch:%d, Valid]  Loss: %.03f | Auc: %.3f ' % (epoch + 1, sum_loss / (step + 1), valid_auc))

    save_path = args.output_dir 
    if rank == 0:
        torch.save(model.state_dict(), f'{save_path}/state_dict_model_best.pth')        
        print('Train has finished, total epoch is %d' % (epoch+1))
        total_time = time.time() - total_start
        print(f"总训练时间: {total_time:.2f}秒")
    dist.destroy_process_group()



if __name__ == '__main__':
    args = parser_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    output_dir = args.output_dir
    train_file, valid_file, test_file, tftest_file = args.train, args.valid, args.test, args.tftest
    expr_path = args.expr
    batch_size = args.batch_size
    model_name = args.model
    save_path = f'{output_dir}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_data, valid_data, test_data, tftest_data = data_generator.load_data(expr_path, train_file, valid_file, test_file, tftest_file, batch_size, model_name)
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    torch.multiprocessing.spawn(
        train,
        args=(world_size, args, train_data, valid_data),
        nprocs=world_size  
    )
    

    
    valid_dataloader = DataLoader.DataLoader(valid_data, batch_size= batch_size, shuffle = False, drop_last=False, num_workers= 2)
    test_dataloader = DataLoader.DataLoader(test_data, batch_size= batch_size, shuffle = False, drop_last=False, num_workers= 2)
    tftest_dataloader = DataLoader.DataLoader(tftest_data, batch_size= batch_size, shuffle = False, drop_last=False, num_workers= 2)
    valid_auc, valid_aupr = predict(valid_dataloader, args, save_path)
    test_auc, test_aupr = predict(test_dataloader, args, save_path)
    tftest_auc, tftest_aupr = predict(tftest_dataloader, args, save_path)
    res = 'method\tdataset\tauroc\taupr\n'
    res += f'{args.model}\tvalid\t{valid_auc}\t{valid_aupr}\n'
    res += f'{args.model}\ttest\t{test_auc}\t{test_aupr}\n'       
    res += f'{args.model}\ttftest\t{tftest_auc}\t{tftest_aupr}\n'

    with open(f'{save_path}/roc.txt', 'w') as f:
        f.write(res)
    print(res)
    if int(args.network) == 1 :
        network.predict_network(args, save_path)

