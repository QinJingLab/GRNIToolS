import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity, paired_cosine_distances
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import load

# 引入项目依赖 (保持不变)
from models import base_models as dl
from utils import gene_embedding as ge
from utils import data_loader as data_loader

# 设置全局设备，避免重复检查
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def cos_similarity(gene_embeddings, test_grn_path=None):

    emb_dict = {k: np.array(v) for k, v in gene_embeddings.items()}
    
    if test_grn_path is None:
        keys = list(emb_dict.keys())
        feature_matrix = np.array(list(emb_dict.values()))
        cos_mat = cosine_similarity(feature_matrix).round(5)
        df_cos_mat = pd.DataFrame(cos_mat, columns=keys, index=keys)
        np.fill_diagonal(df_cos_mat.values, 0)
        
        rows, cols = np.triu_indices(len(keys), k=1)
        result_df = pd.DataFrame({
            'gene_1': df_cos_mat.index[rows],
            'gene_2': df_cos_mat.columns[cols],
            'label': df_cos_mat.values[rows, cols]
        })
        return result_df
    else:
        df = pd.read_csv(test_grn_path, header=None, sep='\t')
        df.columns = ['gene_1', 'gene_2']
        
        g1_series = df['gene_1'].astype(str).str.upper()
        g2_series = df['gene_2'].astype(str).str.upper()
        
        valid_mask = g1_series.isin(emb_dict) & g2_series.isin(emb_dict)
        scores = np.zeros(len(df))

        if valid_mask.any():
            valid_g1 = g1_series[valid_mask]
            valid_g2 = g2_series[valid_mask]
            
            vecs1 = np.array([emb_dict[g] for g in valid_g1])
            vecs2 = np.array([emb_dict[g] for g in valid_g2])
            
            sims = 1 - paired_cosine_distances(vecs1, vecs2)
            scores[valid_mask] = np.abs(sims.round(5))
            
        df['score'] = scores
        df_filter = df[df['score'] > 0]
        return df_filter

def get_embedding_net(gene_embeddings, test_grn_path=None):
    if test_grn_path is None:
        keys = list(gene_embeddings.keys())
        gene_num = len(keys)
        
        rows, cols = np.triu_indices(gene_num, k=1)
        
        result_df = pd.DataFrame({
            'gene_1': [keys[r] for r in rows],
            'gene_2': [keys[c] for c in cols],
            'label': 0.0
        })
        return result_df[['gene_1', 'gene_2']]
    else:
        df = pd.read_csv(test_grn_path, header=None, sep='\t')
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
        df.columns = ['gene_1', 'gene_2']
        return df

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
        return 0.5, 0.0
    y_pred = clf.predict_proba(x_array)[:, 1]
    roc_auc = round(roc_auc_score(y_true, y_pred), 5)
    pr_auc = round(average_precision_score(y_true, y_pred), 5)
    return roc_auc, pr_auc

def encode_gcn_predict(feature, df, dict_gene_id):

    tfs = df.iloc[:, 0].astype(str).str.upper()
    genes = df.iloc[:, 1].astype(str).str.upper()
    

    tf_ids = tfs.map(dict_gene_id)
    gene_ids = genes.map(dict_gene_id)
    
    valid_mask = tf_ids.notna() & gene_ids.notna()
    tf_ids = tf_ids[valid_mask].astype(int).values
    gene_ids = gene_ids[valid_mask].astype(int).values
    indices = np.where(valid_mask)[0] 
    
    labels_raw = (indices % 2 == 0).astype(int) 
    
    pos_mask = (labels_raw == 1)
    neg_mask = (labels_raw == 0)
    
    pos_edge = [tf_ids[pos_mask], gene_ids[pos_mask]]
    neg_edge = [tf_ids[neg_mask], gene_ids[neg_mask]]
    
    pos_edge_idx = torch.tensor(np.stack(pos_edge), dtype=torch.long) if len(pos_edge[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    neg_edge_idx = torch.tensor(np.stack(neg_edge), dtype=torch.long) if len(neg_edge[0]) > 0 else torch.empty((2, 0), dtype=torch.long)
    
    labels = torch.cat([torch.ones(pos_edge_idx.shape[1]), torch.zeros(neg_edge_idx.shape[1])]).float()

    data = Data(x=feature, pos_edge_index=pos_edge_idx, neg_edge_index=neg_edge_idx, label=labels)
    return data

def predict(input_dataloader, checkpoint, dl_method, feature_len):

    if dl_method == 'MLP':
        model = dl.GRNMLP(feature_len)
    elif dl_method == 'ResNet':
        model = dl.GRNResNet()
    elif dl_method == 'CNN':
        model = dl.GRNCNN()
    elif dl_method == 'Transformer':
        model = dl.GRNTrans(feature_len//2, 128, 4, 3)
    
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    
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
                
            link_pred = output.sigmoid().reshape(-1)
            all_link_pred.extend(link_pred.cpu().numpy().tolist())
            
    score = [round(x, 5) for x in all_link_pred]
    return score

def predict_gcn(data, pos_edge, checkpoint, num_features, hidden):
    model = dl.GRNGCN(num_features, hidden, 64) 
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        data.x = data.x.to(DEVICE)
        pos_edge = pos_edge.to(DEVICE)
        data.pos_edge_index = data.pos_edge_index.to(DEVICE)
        data.neg_edge_index = data.neg_edge_index.to(DEVICE)
        z = model.encode(data.x, pos_edge)
        link_pred = model.decode(z, data.pos_edge_index, data.neg_edge_index)
        link_pred = link_pred.sigmoid()
        
    score = [round(x, 5) for x in link_pred.cpu().numpy().tolist()]
    return score

def run_dl(gene_embeddings, checkpoint, df_test_grn, output_dir, llm, dl_method, df_train_grn=None):
    dl_methods = ['MLP', 'ResNet', 'CNN', 'Transformer']
    shuffle = False
    
    if dl_method in dl_methods:
        X_array_test, tf_list, gene_list = data_loader.get_dataset_pre(gene_embeddings, df_test_grn)
        y_array_test = np.zeros(len(X_array_test)) 
        feature_len = X_array_test.shape[1]
        batch_size = 64
        
        if dl_method == 'MLP':
            test_dataloader = data_loader.encode_mlp(X_array_test, y_array_test, batch_size, shuffle, False)
        elif dl_method == 'Transformer':
            test_dataloader = data_loader.encode_trans(X_array_test, y_array_test, batch_size, shuffle, False)
        elif dl_method in ['ResNet', 'CNN']:
            test_dataloader = data_loader.encode_dl(X_array_test, y_array_test, dl_method, batch_size=batch_size, shuffle=shuffle, drop_last=False, worker=2)

        score = predict(test_dataloader, checkpoint, dl_method, feature_len)
        
        predict_grn = pd.DataFrame({'TF': tf_list, 'Gene': gene_list, 'Score': score})
        predict_grn.to_csv(f'{output_dir}/{llm}_{dl_method}_inference_grn.txt', sep='\t', header=True, index=False)

    elif dl_method == 'GCN':
        if df_train_grn is None:
            raise ValueError("df_train_grn cannot be None when using GCN.")
            
        dict_gene_id = dict(zip([i.upper() for i in gene_embeddings.keys()], range(len(gene_embeddings))))

        feature_vals = [v.cpu().numpy() if isinstance(v, torch.Tensor) else v for v in gene_embeddings.values()]
        feature = torch.tensor(np.vstack(feature_vals), dtype=torch.float32)
        
        num_features = feature.shape[1]
        hidden = 128
        
        train_data = data_loader.encode_gcn(feature, df_train_grn, dict_gene_id)
        test_data = encode_gcn_predict(feature, df_test_grn, dict_gene_id)
        
        pos_edge = train_data.pos_edge_index
        score = predict_gcn(test_data, pos_edge, checkpoint, num_features, hidden)
        
        combined_edges = torch.cat([test_data.pos_edge_index, test_data.neg_edge_index], dim=1).cpu().numpy()
        tf_ids = combined_edges[0]
        gene_ids = combined_edges[1]
        
        id_to_gene = {v: k for k, v in dict_gene_id.items()}
        tf_list = [id_to_gene[i] for i in tf_ids]
        gene_list = [id_to_gene[i] for i in gene_ids]
        
        predict_grn = pd.DataFrame({'TF': tf_list, 'Gene': gene_list, 'Score': score})
        predict_grn.to_csv(f'{output_dir}/{llm}_{dl_method}_inference_grn.txt', sep='\t', header=True, index=False)

def run_ml(gene_embeddings, checkpoint, df_test_grn, output_dir, llm, ml_method):
    loaded_model = load(checkpoint)
    X_array_test, tf_list, gene_list = data_loader.get_dataset_pre(gene_embeddings, df_test_grn)
    
    if len(X_array_test) > 0:
        y_pred = loaded_model.predict_proba(X_array_test)[:, 1]
        y_pred = np.round(y_pred, 5)
    else:
        y_pred = []
        
    predict_grn = pd.DataFrame({'TF': tf_list, 'Gene': gene_list, 'Score': y_pred})
    predict_grn.to_csv(f'{output_dir}/{llm}_{ml_method}_inference_grn.txt', sep='\t', header=True, index=False)

def predict_biollm(llm, pt, method, model_path, expr_path, test_path, species, output, train_path=None):
    ml_methods = ['logit', 'rf', 'svm', 'xgb']
    dl_methods = ['MLP', 'ResNet', 'CNN', 'Transformer', 'GCN']

    output_dir = f'{output}/{llm}/'
    os.makedirs(output_dir, exist_ok=True)
    
    gene_embeddings = ge.get_gene_embedding(llm, pt, expr_path, species, output_dir)

    df_expr = pd.read_csv(expr_path, index_col=0, header=0)
    valid_genes = set(g.upper() for g in df_expr.index)
    gene_embeddings = {k: v for k, v in gene_embeddings.items() if k.upper() in valid_genes}

    if method == 'cos':
        df_test_grn = cos_similarity(gene_embeddings, test_path)
        if test_path is None:
            df_test_grn.to_csv(f'{output_dir}/cos_inference_grn_all.txt', sep='\t', header=True, index=False)
        else:
            df_test_grn.columns = ['TF', 'Gene', 'Score']
            df_test_grn.to_csv(f'{output_dir}/cos_inference_grn.txt', sep='\t', header=True, index=False)

    elif method in ml_methods or method in dl_methods:
        df_test_grn = get_embedding_net(gene_embeddings, test_path)
        
        df_train_grn = None
        if method == 'GCN' and train_path is not None:
            df_train_grn = pd.read_csv(train_path, header=None, sep='\t')
            if df_train_grn.shape[1] == 3:
                df_train_grn.columns = ['gene_1', 'gene_2', 'label']
                df_train_grn['score'] = 1 # 占位符，不作使用
            else:
                df_train_grn.columns = ['gene_1', 'gene_2', 'label', 'score']
 
        if method in ml_methods:
            run_ml(gene_embeddings, model_path, df_test_grn, output_dir, llm, method)
        elif method in dl_methods:
            run_dl(gene_embeddings, model_path, df_test_grn, output_dir, llm, method, df_train_grn)