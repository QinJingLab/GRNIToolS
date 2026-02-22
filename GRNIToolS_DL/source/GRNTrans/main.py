import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import roc_auc_score, average_precision_score  
from torch.utils.data import TensorDataset, DataLoader


def get_parser():
    parser = argparse.ArgumentParser(description='Gene Interaction Transformer Model')
    parser.add_argument('--expr',   type=str, 
                       default='./Input_data/Ecoli/GRNDL/expression.csv', 
                       required=False, 
                       help='Expression data file path')
    parser.add_argument('--train',  type=str, 
                       default='./Input_data/Ecoli/GRNDL/train.txt',
                       required=False, 
                       help='Training data file path')
    parser.add_argument('--valid',  type=str,
                       default='./Input_data/Ecoli/GRNDL/val.txt',
                       required=False,
                       help='Validation data file path')
    parser.add_argument('--test',   type=str,
                       default='/mnt/sdb/ZYQ/workspace/GRNITools2/  GRNIToolS/Input_data/Ecoli/GRNDL/random_test.txt',
                       required=False,
                       help='Test data file path')
    parser.add_argument('--tftest', type=str,
                       default='./Input_data/Ecoli/GRNDL/TF_test.txt',
                       required=False,
                       help='TF test data file path')
    parser.add_argument("--tf", type = str, default='', help="<file> Input tf list file")
    parser.add_argument("--gene", type = str, default='', help="<file> Input gene list file")
    parser.add_argument('-o', '--output', type=str, default='./output', help = 'Output')
    parser.add_argument('--network', type=str, default='0', help='GRNDL network prediction, default is False')

    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='Transformer model')
    parser.add_argument('--head', type=int, default=2, help='Number of head')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of labyers')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning ratio')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--normal', action='store_true', help='log2 normalization')
    parser.add_argument('--cuda_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES (comma-separated), e.g. 0,1,3')
    return parser.parse_args()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank) 



class GeneInteractionModel(nn.Module):
    def __init__(self, input_dim=805, d_model=128, nhead=2, num_layers=3):
        super().__init__()
        self.gene_encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU()
        ) 
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
        self.pos_encoder = nn.Parameter(torch.randn(1, 2, d_model))
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        gene1_embed = self.gene_encoder(x1)
        gene2_embed = self.gene_encoder(x2)
        sequence = torch.stack([gene1_embed, gene2_embed], dim=1)
        sequence += self.pos_encoder
        
        encoded = self.transformer_encoder(sequence)
        output = self.classifier(encoded[:, 0])
        return output
    
def load_data(expr_file, data_file, batch_size = 32, normal = False, shuffle = True):
    expr_data = pd.read_csv(expr_file, index_col=0)
    if expr_data.shape[1] > 4000:
        np.random.seed(42)
        col = np.random.choice(expr_data.columns, 4000, replace=False)
        expr_data = expr_data[col]
    if normal:
        expr_data = np.log2(expr_data + 1)
    
    dat = pd.read_csv(data_file, delimiter='\t')
    tflist, genelist, label = [], [], []
    for index, row in dat.iterrows():
        tf, gene, l = row.tolist()
        if tf not in expr_data.index or gene not in expr_data.index:
            continue  
        tflist.append(expr_data.loc[tf].tolist())
        genelist.append(expr_data.loc[gene].tolist())
        label.append(int(l))


    tflist_tensor = torch.tensor(tflist, dtype=torch.float32)
    genelist_tensor = torch.tensor(genelist, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.float32)
    dataset = TensorDataset(tflist_tensor, genelist_tensor, label_tensor)
    return dataset

def load_data_expr(expr_file, data_file, batch_size = 32, tflist = '', genelist ='', normal = False, shuffle = True):
    genelist = []
    tflist = []
    expr_data = pd.read_csv(expr_file, index_col=0)
    
    if expr_data.shape[1] > 4000:
        np.random.seed(42)
        col = np.random.choice(expr_data.columns, 4000, replace=False)
        expr_data = expr_data[col]

    if tflist and genelist:
        tf = pd.read_csv(tflist, delimiter='\t', header=None) 
        tf_list = tf.iloc[:, 0].tolist()
        gene = pd.read_csv(genelist, delimiter='\t', header=None) 
        gene_list = gene.iloc[:, 0].tolist()
    else:
        df = pd.read_csv(
            data_file,
            sep='\t',
            header=None,
            names=['tf', 'gene', 'label'],
            dtype={'tf': str, 'gene': str, 'label': int}
        )
        df = df.apply(lambda x: x if x.dtype == "object" else x)
        df = df.dropna()        
        tf_list = df['tf'].drop_duplicates().tolist()
        gene_list = df['gene'].drop_duplicates().tolist()
    tf_gene_dataset = ''
    label = []

    for tf in tf_list:
        for gene in gene_list:
            if tf not in expr_data.index or gene not in expr_data.index:
                continue          
            tflist.append(expr_data.loc[tf].tolist())
            genelist.append(expr_data.loc[gene].tolist())
            label.append(1)           
            tf_gene_dataset += f'{tf}\t{gene}\n'


    tflist_tensor = torch.tensor(tflist, dtype=torch.float32)
    genelist_tensor = torch.tensor(genelist, dtype=torch.float32)
    label_tensor = torch.tensor(label, dtype=torch.float32)

    dataset = TensorDataset(tflist_tensor, genelist_tensor, label_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader, tf_gene_dataset


def train(rank, world_size, args, input_dim):
    setup(rank, world_size)
    torch.manual_seed(0)  

    train_dataset = load_data(args.expr, args.train, 
                            batch_size=args.batch_size, normal=args.normal, shuffle=True)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    valid_dataset = load_data(args.expr, args.valid,
                                batch_size=args.batch_size, normal=args.normal, shuffle=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler)

    model = GeneInteractionModel(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.head,
        num_layers=args.num_layers
    ).to(rank)
    ddp_model = DDP(model, device_ids=[rank])   

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss() 
    epochs = args.epochs
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch) 
        ddp_model.train()
        for data in train_loader:
            gene1, gene2, labels = data
            gene1, gene2, labels = gene1.to(rank), gene2.to(rank), labels.to(rank)
            outputs = ddp_model(gene1, gene2).squeeze()
            loss = criterion(outputs, labels)            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()            
            if len(np.unique(labels.cpu())) < 2 or len(np.unique(outputs.detach().cpu())) < 2:
                auroc = 0.5        
            else:         
                auroc = roc_auc_score(labels.to('cpu').numpy(), outputs.detach().to('cpu').numpy())

        if rank == 0 and (epoch+10) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | AUROC: {auroc:.4f}")


    if rank == 0:
        val_auroc = predict(model, valid_loader, rank)[0]
        print(f"\nValidation AUROC: {val_auroc:.5f}")
        os.makedirs(args.output, exist_ok=True)
        model_save_path = f"{args.output}/model.pth"
        torch.save(model.state_dict(), model_save_path)

    dist.destroy_process_group()  



def predict(model, data_loader, device, has_labels=True):
    model.eval()
    all_probs = []
    all_labels = [] if has_labels else None
    
    with torch.no_grad():
        for data in data_loader:
            gene1, gene2 = data[:2]
            gene1, gene2 = gene1.to(device), gene2.to(device)
            outputs = model(gene1, gene2).squeeze()
            probs = outputs.cpu().numpy()
            all_probs.extend(np.atleast_1d(probs))
            
            if has_labels and len(data) > 2:
                labels = data[2].cpu().numpy()
                all_labels.extend(labels)
    
    if has_labels:
        auroc = round(roc_auc_score(all_labels, all_probs),5)
        aupr = round(average_precision_score(all_labels, all_probs),5)
        return auroc, aupr
    return np.array(all_probs)


def predict_network(model, data_loader, device, tf_gene_dataset, save_path, has_labels=True):
    model.eval()
    all_probs = []
    all_labels = [] if has_labels else None
    
    with torch.no_grad():
        for data in data_loader:
            gene1, gene2 = data[:2]
            gene1, gene2 = gene1.to(device), gene2.to(device)
            outputs = model(gene1, gene2).squeeze()
            probs = outputs.cpu().numpy()
            all_probs.extend(np.atleast_1d(probs))
            
            if has_labels and len(data) > 2:
                labels = data[2].cpu().numpy()
                all_labels.extend(labels)
    pred = np.array(all_probs)
    updated_dataset = []
    lines = tf_gene_dataset.strip().split('\n')
    for line, pred in zip(lines, all_probs):
        updated_dataset.append(f"{line}\t{pred:.5f}")
    tf_gene_dataset = '\n'.join(updated_dataset)
    with open(save_path + '/test_network_predict.txt', 'w') as f:
        f.write(tf_gene_dataset)


def create_dataloader(dataset, batch_size):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloder = DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    return sampler, dataloder

def main():
    args = get_parser()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    world_size = len(args.cuda_devices.split(','))

    def create_dataloader(dataset, batch_size):
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=len(args.cuda_devices.split(',')),
            rank=args.local_rank
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True)
    
    train_dataset = load_data(args.expr, args.train, 
                                batch_size=args.batch_size, normal=args.normal, shuffle=True)
    valid_dataset = load_data(args.expr, args.valid,
                                 batch_size=args.batch_size, normal=args.normal, shuffle=True)
    test_dataset = load_data(args.expr, args.test,
                                batch_size=args.batch_size, normal=args.normal, shuffle=False)
    tftest_dataset = load_data(args.expr,args.tftest,
                                batch_size=args.batch_size, normal=args.normal, shuffle=False)

    train_dataloader = create_dataloader(train_dataset, args.batch_size)
    valid_dataloader = create_dataloader(valid_dataset, args.batch_size)
    test_dataloader = create_dataloader(test_dataset, args.batch_size)
    tftest_dataloader = create_dataloader(tftest_dataset, args.batch_size)

    input_dim = pd.read_csv(args.expr, index_col=0).shape[1]  # 自动推断特征维度
    if input_dim > 4000:
        input_dim = 4000

    model = GeneInteractionModel(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.head,
        num_layers=args.num_layers
    ).to(device)
    model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=True)
  

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    
    trained_model = train(model, train_dataloader, valid_dataloader, 
                            optimizer, criterion, device, args.epochs)
    valid_auc, valid_aupr = predict(trained_model, valid_dataloader, device)
    test_auc, test_aupr = predict(trained_model, test_dataloader, device)
    tftest_auc, tftest_aupr = predict(trained_model, tftest_dataloader, device)

    os.makedirs(args.output, exist_ok=True)
    model_save_path = f"{args.output}/model.pth"
    if  args.local_rank == 0:  
        torch.save(trained_model.state_dict(), model_save_path)

    res = 'method\tdataset\tauroc\taupr\n'
    res += f'GRNTrans\tvalid\t{valid_auc}\t{valid_aupr}\n'
    res += f'GRNTrans\ttest\t{test_auc}\t{test_aupr}\n'       
    res += f'GRNTrans\ttftest\t{tftest_auc}\t{tftest_aupr}\n'
    with open(f'{args.output}/roc.txt', 'w') as f:
        f.write(res)

    if args.network != '0':
        data_loader, tf_gene_dataset = load_data_expr(args.expr, args.test, batch_size = args.batch_size, tflist = args.tf, genelist = args.gene, normal = False, shuffle = False)
        predict_network(model, data_loader, device, tf_gene_dataset, args.output)

if __name__ == '__main__':
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    valid_dataset = load_data(args.expr, args.valid,
                                 batch_size=args.batch_size, normal=args.normal, shuffle=True)
    test_dataset = load_data(args.expr, args.test,
                                batch_size=args.batch_size, normal=args.normal, shuffle=False)
    tftest_dataset = load_data(args.expr,args.tftest,
                                batch_size=args.batch_size, normal=args.normal, shuffle=False)
    input_dim = pd.read_csv(args.expr, index_col=0).shape[1]  # 自动推断特征维度
    if input_dim > 4000:
        input_dim = 4000

    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    torch.multiprocessing.spawn(
        train,
        args=(world_size, args, input_dim),
        nprocs=world_size  # 必须与可见GPU数量一致
    )
    
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    tftest_dataloader = DataLoader(tftest_dataset, batch_size=args.batch_size, shuffle=False)


    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GeneInteractionModel(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.head,
        num_layers=args.num_layers
    ).to(device)
    model_save_path = f"{args.output}/model.pth"
    model.load_state_dict(
        torch.load(model_save_path, weights_only=True)  
    )
    valid_auc, valid_aupr = predict(model, valid_dataloader, device)
    test_auc, test_aupr = predict(model, test_dataloader, device)
    tftest_auc, tftest_aupr = predict(model, tftest_dataloader, device)


    res = 'method\tdataset\tauroc\taupr\n'
    res += f'GRNTrans\tvalid\t{valid_auc}\t{valid_aupr}\n'
    res += f'GRNTrans\ttest\t{test_auc}\t{test_aupr}\n'       
    res += f'GRNTrans\ttftest\t{tftest_auc}\t{tftest_aupr}\n'
    with open(f'{args.output}/roc.txt', 'w') as f:
        f.write(res)

    if args.network != '0':
        data_loader, tf_gene_dataset = load_data_expr(args.expr, args.test, batch_size = args.batch_size, tflist = args.tf, genelist = args.gene, normal = False, shuffle = False)
        predict_network(model, data_loader, device, tf_gene_dataset, args.output)
    
    



