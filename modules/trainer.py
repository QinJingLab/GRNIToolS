import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics.pairwise import cosine_similarity, paired_cosine_distances
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from models import base_models as dl
from utils import gene_embedding as ge
from utils import data_loader as data_loader


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cos_similarity(gene_embeddings, output_dir, test_grn_path=None):

    embeddings_np = {k: np.array(v) for k, v in gene_embeddings.items()}

    if test_grn_path is None:
        keys = list(embeddings_np.keys())
        feature_matrix = np.array(list(embeddings_np.values()))
        cos_mat = cosine_similarity(feature_matrix).round(5)
        
        df_cos_mat = pd.DataFrame(cos_mat, columns=keys, index=keys)
        df_cos_mat.to_csv(f'{output_dir}/cos_grn.txt', sep='\t', header=True, index=True)
        return None
    else:
        df = pd.read_csv(test_grn_path, header=None, sep='\t')
        df.columns = ['gene_1', 'gene_2', 'label']
        
        gene1_list = df['gene_1'].str.upper().tolist()
        gene2_list = df['gene_2'].str.upper().tolist()
        
        valid_indices = []
        vec_a = []
        vec_b = []
        scores = np.zeros(len(df)) # 初始化分数为0

        for idx, (g1, g2) in enumerate(zip(gene1_list, gene2_list)):
            if g1 in embeddings_np and g2 in embeddings_np:
                valid_indices.append(idx)
                vec_a.append(embeddings_np[g1])
                vec_b.append(embeddings_np[g2])
        
        if valid_indices:
            vec_a = np.array(vec_a)
            vec_b = np.array(vec_b)
            pair_scores = 1 - paired_cosine_distances(vec_a, vec_b)
            np.put(scores, valid_indices, pair_scores)
        df['score'] = np.abs(np.round(scores, 5))
        df_filter = df[df['score'] > 0]
        return df_filter

def get_auc_cos(df_filter):
    if df_filter.empty:
        return 0.5, 0.0
    y_true = df_filter['label']
    y_scores = df_filter['score']
    roc_auc = round(roc_auc_score(y_true, y_scores), 5)
    pr_auc = round(average_precision_score(y_true, y_scores), 5)
    return roc_auc, pr_auc

def get_auroc_aupr(clf, x_array, y_true):
    if len(x_array) == 0:
        return 0.0, 0.0
    y_pred = clf.predict_proba(x_array)[:, 1]
    roc_auc = round(roc_auc_score(y_true, y_pred), 5)
    pr_auc = round(average_precision_score(y_true, y_pred), 5)
    return roc_auc, pr_auc

def predict(input_dataloader, dl_method, feature_len, save_path):
    if dl_method == 'MLP':
        model = dl.GRNMLP(feature_len)
    elif dl_method == 'ResNet':
        model = dl.GRNResNet()
    elif dl_method == 'CNN':
        model = dl.GRNCNN()
    elif dl_method == 'Transformer':
        model = dl.GRNTrans(feature_len // 2, 128, 4, 3)
    else:
        raise ValueError(f"Unknown dl_method: {dl_method}")

    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    
    all_label = []
    all_link_pred = []
    
    with torch.no_grad():
        for datas in input_dataloader:
            if len(datas) == 3:
                tf, gene, label = datas
                tf, gene = tf.to(DEVICE), gene.to(DEVICE)
                output = model(tf, gene).squeeze()
            else:
                data, label = datas
                data = data.to(DEVICE)
                output = model(data)
            
            label = label.to(torch.float32).to(DEVICE)
            
            link_pred = output.sigmoid().reshape(-1)
            all_label.extend(label.cpu().numpy().flatten().tolist())
            all_link_pred.extend(link_pred.detach().cpu().numpy().tolist())
            
    if not all_label: return 0.0, 0.0

    auc = roc_auc_score(all_label, all_link_pred)
    aupr = average_precision_score(all_label, all_link_pred)
    return round(auc, 5), round(aupr, 5)

def train(model, train_dataloader, valid_dataloader, epochs, lr, save_path):
    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_auc = 0
    
    for epoch in range(epochs):
        model.train()
        sum_loss = 0
        sum_train_auc = 0
        total_steps = len(train_dataloader)
        
        for step, datas in enumerate(train_dataloader):
            optimizer.zero_grad()
            
            # 数据解包
            if len(datas) == 3:
                tf, gene, label = datas
                tf, gene = tf.to(DEVICE), gene.to(DEVICE)
                label = label.to(torch.float32).to(DEVICE)
                output = model(tf, gene).squeeze()
            else:
                data, label = datas
                data = data.to(DEVICE)
                label = label.to(torch.float32).to(DEVICE)
                output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            
            sum_loss += loss.item()
            
            link_pred = output.sigmoid().detach().cpu()
            all_labels = label.cpu()
            
            try:
                if len(np.unique(all_labels)) >= 2:
                    train_auc = roc_auc_score(all_labels, link_pred)
                else:
                    train_auc = 0.5
            except ValueError:
                train_auc = 0.5
                
            sum_train_auc += train_auc

        avg_loss = sum_loss / total_steps
        avg_auc = sum_train_auc / total_steps
        print(f'[Epoch: {epoch + 1}, Train] Loss: {avg_loss:.3f} | Auc: {avg_auc:.3f}')

        # Validation
        model.eval()
        all_label = []
        all_link_pred = []
        
        with torch.no_grad():
            for data2 in valid_dataloader:
                if len(data2) == 3:
                    tf, gene, label = data2
                    tf, gene = tf.to(DEVICE), gene.to(DEVICE)
                    output = model(tf, gene).squeeze()
                else:
                    data, label = data2
                    data = data.to(DEVICE)
                    output = model(data)
                
                label = label.to(torch.float32).to(DEVICE)
                link_pred = output.sigmoid().reshape(-1)
                
                all_label.extend(label.cpu().numpy().flatten().tolist())
                all_link_pred.extend(link_pred.cpu().numpy().tolist())

        if len(all_label) > 0 and len(np.unique(all_label)) >= 2:
            valid_auc = roc_auc_score(all_label, all_link_pred)
        else:
            valid_auc = 0.0

        print(f'[Epoch: {epoch + 1}, Valid] Loss: {avg_loss:.3f} | Auc: {valid_auc:.3f}')
        
        if valid_auc > best_val_auc:
            best_val_auc = valid_auc
            torch.save(model.state_dict(), save_path)
            
    print(f'Train has finished, total epoch is {epochs}')

def predict_gcn(data, pos_edge, save_path, num_features, hidden):
    model = dl.GRNGCN(num_features, hidden, 64)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        data.x = data.x.to(DEVICE)
        pos_edge = pos_edge.to(DEVICE)
        data.pos_edge_index = data.pos_edge_index.to(DEVICE)
        data.neg_edge_index = data.neg_edge_index.to(DEVICE)
        
        z = model.encode(data.x, pos_edge)
        link_pred = model.decode(z, data.pos_edge_index, data.neg_edge_index)
        label = data.label.to(DEVICE)
        
        link_pred = link_pred.sigmoid()
        res_auc = roc_auc_score(label.cpu(), link_pred.cpu())
        res_aupr = average_precision_score(label.cpu(), link_pred.cpu())
        
    return round(res_auc, 5), round(res_aupr, 5)

def train_gcn(model, train_data, valid_data, save_path, epochs, lr):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    train_data = train_data.to(DEVICE)
    val_data = valid_data.to(DEVICE)
    
    best_val_auc = 0
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        z = model.encode(train_data.x, train_data.pos_edge_index)
        link_pred = model.decode(z, train_data.pos_edge_index, train_data.neg_edge_index)
        label = train_data.label
        
        loss = criterion(link_pred, label)
        loss.backward()
        optimizer.step()
        
        # Train Metric
        train_pred_prob = link_pred.sigmoid().detach()
        train_auc = roc_auc_score(label.cpu(), train_pred_prob.cpu())

        # Validation
        model.eval()
        with torch.no_grad():
            z_val = model.encode(val_data.x, train_data.pos_edge_index) 
            val_pred = model.decode(z_val, val_data.pos_edge_index, val_data.neg_edge_index)
            val_label = val_data.label
            val_loss = criterion(val_pred, val_label)
            val_pred_prob = val_pred.sigmoid()
            val_auc = roc_auc_score(val_label.cpu(), val_pred_prob.cpu())
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                torch.save(model.state_dict(), save_path)

        print(f'Epoch: {epoch+1:03d}, Train_loss: {loss.item():.4f}, Train_auc: {train_auc:.4f}, Val_loss: {val_loss.item():.4f}, Val_auc: {val_auc:.4f}')
    print("-" * 64)

def run_dl(gene_embeddings, df_train_grn, df_valid_grn, df_test_grn, df_tftest_grn, output_dir, batch_size, epochs, lr, llm, dl_method='ResNet'):
    dl_methods_list = ['MLP', 'ResNet', 'CNN', 'Transformer']
    dataset_path = f'{output_dir}/{dl_method}_data_size.txt'
    save_path = f'{output_dir}/{dl_method}_state_dict_model.pth'
    auc_path = f'{output_dir}/{dl_method}_grn_auc.txt'
    
    shuffle = True

    dfs = [df_train_grn, df_valid_grn, df_test_grn]
    if not df_tftest_grn.empty: dfs.append(df_tftest_grn)
    
    datasets = [data_loader.get_dataset(gene_embeddings, df) for df in dfs]
    X_train, y_train = datasets[0]
    X_valid, y_valid = datasets[1]
    X_test, y_test = datasets[2]
    
    tftest_exists = not df_tftest_grn.empty
    if tftest_exists:
        X_tftest, y_tftest = datasets[3]
        tftest_size = df_tftest_grn.shape[0]
    else:
        X_tftest, y_tftest = [], []
        tftest_size = 0

    dataset_dim = f'{llm}\t{dl_method}\t{df_train_grn.shape[0]}\t{df_valid_grn.shape[0]}\t{df_test_grn.shape[0]}\t{tftest_size}\n'
    # with open(dataset_path, 'w') as f:
    #     f.write(dataset_dim)

    feature_len = X_train.shape[1]
    print(f'Embedding length: {feature_len // 2}')

    if dl_method in dl_methods_list:
        def get_loader(X, y, method):
            if method == 'MLP':
                return data_loader.encode_mlp(X, y, batch_size, shuffle)
            elif method == 'Transformer':
                return data_loader.encode_trans(X, y, batch_size, shuffle)
            elif method in ['ResNet', 'CNN']:
                return data_loader.encode_dl(X, y, method, batch_size=batch_size, shuffle=shuffle, worker=2)
        
        train_loader = get_loader(X_train, y_train, dl_method)
        valid_loader = get_loader(X_valid, y_valid, dl_method)
        test_loader = get_loader(X_test, y_test, dl_method)
        tftest_loader = get_loader(X_tftest, y_tftest, dl_method) if tftest_exists else None

        # 模型初始化
        if dl_method == 'MLP':
            model = dl.GRNMLP(feature_len)
        elif dl_method == 'Transformer':
            model = dl.GRNTrans(feature_len//2, 128, 4, 3)
        elif dl_method == 'ResNet':
            model = dl.GRNResNet()
        elif dl_method == 'CNN':
            model = dl.GRNCNN()

        train(model, train_loader, valid_loader, epochs, lr, save_path)
        
        valid_roc_auc, valid_pr_auc = predict(valid_loader, dl_method, feature_len, save_path)
        test_roc_auc, test_pr_auc = predict(test_loader, dl_method, feature_len, save_path)
        
        if tftest_exists:
            tftest_roc_auc, tftest_pr_auc = predict(tftest_loader, dl_method, feature_len, save_path)
        else:
            tftest_roc_auc, tftest_pr_auc = 'none', 'none'

    elif dl_method == 'GCN':
        dict_gene_id = dict(zip([i.upper() for i in gene_embeddings.keys()], range(len(gene_embeddings))))
        
        feature_vals = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in gene_embeddings.values()]
        feature = torch.tensor(np.vstack(feature_vals), dtype=torch.float32)
        
        num_features = feature.shape[1]
        hidden = 128
        model = dl.GRNGCN(num_features, hidden, 64)
        
        train_data = data_loader.encode_gcn(feature, df_train_grn, dict_gene_id)
        valid_data = data_loader.encode_gcn(feature, df_valid_grn, dict_gene_id)
        test_data = data_loader.encode_gcn(feature, df_test_grn, dict_gene_id)
        tftest_data = data_loader.encode_gcn(feature, df_tftest_grn, dict_gene_id) if tftest_exists else None
        
        train_gcn(model, train_data, valid_data, save_path, epochs, lr)
        
        pos_edge = train_data.pos_edge_index
        valid_roc_auc, valid_pr_auc = predict_gcn(valid_data, pos_edge, save_path, num_features, hidden)
        test_roc_auc, test_pr_auc = predict_gcn(test_data, pos_edge, save_path, num_features, hidden)
        
        if tftest_exists:
            tftest_roc_auc, tftest_pr_auc = predict_gcn(tftest_data, pos_edge, save_path, num_features, hidden)
        else:
            tftest_roc_auc, tftest_pr_auc = 'none', 'none'

    s_file = (f'method\tdataset\tauroc\taupr\n'
              f'{dl_method}\tvalid\t{valid_roc_auc}\t{valid_pr_auc}\n'
              f'{dl_method}\ttest\t{test_roc_auc}\t{test_pr_auc}\n'
              f'{dl_method}\ttftest\t{tftest_roc_auc}\t{tftest_pr_auc}\n')
    
    with open(auc_path, 'w') as f:
        f.write(s_file)

def run_ml(gene_embeddings, df_train_grn, df_valid_grn, df_test_grn, df_tftest_grn, output_dir, llm, ml_method='rf'):
    save_path = f'{output_dir}/{ml_method}_model.pth'
    auc_path = f'{output_dir}/{ml_method}_grn_auc.txt'
    dataset_path = f'{output_dir}/{ml_method}_data_size.txt'
    
    models_map = {
        'rf': RandomForestClassifier,
        'xgb': XGBClassifier,
        'logit': LogisticRegression,
        'svm': lambda: SVC(probability=True)
    }
    
    if ml_method in models_map:
        clf = models_map[ml_method]()
    else:
        raise ValueError(f"Unknown ml_method: {ml_method}")

    X_train, y_train = data_loader.get_dataset(gene_embeddings, df_train_grn)
    X_valid, y_valid = data_loader.get_dataset(gene_embeddings, df_valid_grn)
    X_test, y_test = data_loader.get_dataset(gene_embeddings, df_test_grn)
    
    tftest_exists = not df_tftest_grn.empty
    if tftest_exists:
        X_tftest, y_tftest = data_loader.get_dataset(gene_embeddings, df_tftest_grn)
        tftest_size = df_tftest_grn.shape[0]
    else:
        X_tftest, y_tftest = [], []
        tftest_size = 0

    dataset_dim = f'{llm}\t{ml_method}\t{df_train_grn.shape[0]}\t{df_valid_grn.shape[0]}\t{df_test_grn.shape[0]}\t{tftest_size}\n'
    # with open(dataset_path, 'w') as f:
    #     f.write(dataset_dim)

    clf.fit(X_train, y_train)
    dump(clf, save_path)
    
    valid_roc_auc, valid_pr_auc = get_auroc_aupr(clf, X_valid, y_valid)
    test_roc_auc, test_pr_auc = get_auroc_aupr(clf, X_test, y_test)
    
    if tftest_exists:
        tftest_roc_auc, tftest_pr_auc = get_auroc_aupr(clf, X_tftest, y_tftest)
    else:
        tftest_roc_auc, tftest_pr_auc = 'none', 'none'

    s_file = (f'method\tdataset\tauroc\taupr\n'
              f'{ml_method}\tvalid\t{valid_roc_auc}\t{valid_pr_auc}\n'
              f'{ml_method}\ttest\t{test_roc_auc}\t{test_pr_auc}\n'
              f'{ml_method}\ttftest\t{tftest_roc_auc}\t{tftest_pr_auc}\n')
    
    with open(auc_path, 'w') as f:
        f.write(s_file)

def run_biollm(llm, pt, method, expr_path, train_path, valid_path, test_path, tftest_path, output, batch_size, epochs, lr, species='human'):
    ml_methods = ['logit', 'rf', 'svm', 'xgb']
    dl_methods = ['MLP', 'ResNet', 'CNN', 'Transformer', 'GCN']

    output_dir = f'{output}/{llm}/'
    os.makedirs(output_dir, exist_ok=True)
    
    gene_embeddings = ge.get_gene_embedding(llm, pt, expr_path, species, output_dir)
    
    if method == 'cos':
        df_valid_grn = cos_similarity(gene_embeddings, output_dir, valid_path)
        df_test_grn = cos_similarity(gene_embeddings, output_dir, test_path)
        df_tftest_grn = cos_similarity(gene_embeddings, output_dir, tftest_path) if tftest_path else pd.DataFrame()
        
        valid_roc_auc, valid_pr_auc = get_auc_cos(df_valid_grn)
        test_roc_auc, test_pr_auc = get_auc_cos(df_test_grn)
        
        if df_tftest_grn is not None and not df_tftest_grn.empty:
            tftest_roc_auc, tftest_pr_auc = get_auc_cos(df_tftest_grn)
        else:
            tftest_roc_auc, tftest_pr_auc = 'none', 'none'
            
        path = f'{output_dir}/cos_grn_auc.txt'
        s_file = (f'method\tdataset\tauroc\taupr\n'
                  f'{method}\tvalid\t{valid_roc_auc}\t{valid_pr_auc}\n'
                  f'{method}\ttest\t{test_roc_auc}\t{test_pr_auc}\n'
                  f'{method}\ttftest\t{tftest_roc_auc}\t{tftest_pr_auc}\n')
        with open(path, 'w') as f:
            f.write(s_file)

    elif method in ml_methods or method in dl_methods:
        df_train_grn = cos_similarity(gene_embeddings, output_dir, train_path)
        df_valid_grn = cos_similarity(gene_embeddings, output_dir, valid_path)
        df_test_grn = cos_similarity(gene_embeddings, output_dir, test_path)
        df_tftest_grn = cos_similarity(gene_embeddings, output_dir, tftest_path) if tftest_path else pd.DataFrame()
        
        if method in ml_methods:
            run_ml(gene_embeddings, df_train_grn, df_valid_grn, df_test_grn, df_tftest_grn, output_dir, llm, ml_method=method)
        elif method in dl_methods:
            run_dl(gene_embeddings, df_train_grn, df_valid_grn, df_test_grn, df_tftest_grn, output_dir, batch_size, epochs, lr, llm, dl_method=method)