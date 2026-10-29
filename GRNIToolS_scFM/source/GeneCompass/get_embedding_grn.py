import pandas as pd  
import anndata 
import scanpy as sc
from scipy.stats import zscore
import numpy as np
import pickle 
import tqdm
import os
import argparse
from datasets import Dataset,Features,Sequence,Value,load_from_disk
import torch
import pickle
from tqdm.notebook import trange
import pandas as pd
import torch.nn as nn
import numpy as np
import copy
import sys
from genecompass import BertForMaskedLM
from genecompass.utils import load_prior_embedding

    


# 从文件列表中读取指定物种线粒体基因、蛋白质基因、miRNA基因的基因名
# def get_gene_list(species: str, file_list: list[str]):
def get_gene_list(species, file_list):
    #species:human file_list:
    take_list = []
    protein_list = []
    miRNA_list = []
    mitochondria_list = []
    for i in file_list:
        if (i.split('/')[-1].split('_')[0] == species):
            take_list.append(i)
    for i in take_list:
        if (i.split('.')[-1] == "txt"):
            with open(i, "r") as f:
                name_temp = str(f.name.split('/')[-1].split('_')[1])
                if name_temp == 'protein':
                    for line in f:
                        first_word = line.split()[0]
                        protein_list.append(first_word)
                elif name_temp == 'miRNA.txt':
                    for line in f:
                        first_word = line.split()[0]
                        miRNA_list.append(first_word)
        elif i.split('.')[-1] == "xlsx":
            df = pd.read_excel(i)
            mitochondria_list = df.iloc[:,1].to_list()

    return protein_list, miRNA_list, mitochondria_list

# 根据gene_id过滤基因
def gene_id_filter(adata, gene_list) -> anndata:

    mask = adata.var.index.isin(gene_list)
    adata = adata[:, mask]
    return adata

def id_name_match(protein_list, miRNA_list):
    p_l = []
    for i in protein_list:
        if dict1.get(i) != None:
            p_l.append(dict1.get(i))

    m_l = []
    for i in miRNA_list:
        if dict1.get(i) != None:
            m_l.append(dict1.get(i))

    return p_l,m_l

def id_name_match1(name_list, dict1):
    n_l = []
    for i in name_list:
        if dict1.get(i) != None:
            n_l.append(dict1.get(i))
        else:
            n_l.append('delete')
    
    return n_l

def id_token_match(name_list, token_dict):
    m_l = []
    for i in name_list:
        if token_dict.get(i) != None:
            m_l.append(i)
        else:
            m_l.append('delete')
    return m_l

# 过滤基因总量或线粒体基因总量在所有细胞平均值三个标准差之外的的细胞
def normal_filter(adata, mito_list) -> anndata:
    total_counts = adata.X.sum(axis=1) # 计算每一行之和
    # adata.X : 50000 * 55655 行表示细胞，列表示基因
    # adata.X存储的是矩阵信息，它的结构虽然是一个数组，但是没有行名和列名信息。矩阵的行名信息存储在adata.obs中，列名信息存储在adata.var中。
    total_counts = np.array(total_counts).squeeze() #len = 50000

    idx = total_counts > 0 #创建一个布尔索引数组
    adata = adata[idx, :] #使用布尔索引idx筛选adata的行，只保留idx中对应位置为True的行数据
    total_counts = total_counts[idx] #只保留idx中对应位置为True的元素
    gene_name = adata.var.gene_symbols if 'gene_symbols' in adata.var else adata.var.feature_name
    gene_name = gene_name.tolist()
    index = [element in mito_list for element in gene_name] #根据mito_list中的元素是否存在于gene_name中，创建一个布尔索引列表index
    mito_adata = adata[:, index] #使用布尔索引index筛选adata的列，只保留index中对应位置为True的列数据

    # 计算每个细胞的线粒体基因表达之和
    mito_counts = mito_adata.X.sum(axis=1)
    mito_counts = np.array(mito_counts).squeeze()
    mito_percentage = mito_counts / total_counts

    # 将每个变量的值进行标准化，计算z-score
    total_counts_zscore = zscore(total_counts) #计算总基因表达量的z-score，即对总基因表达量进行标准化。
    mito_percentage_zscore = zscore(mito_percentage) #计算线粒体基因表达百分比的z-score，即对线粒体基因表达百分比进行标准化

    # 筛选出符合条件的细胞
    keep_cells = ((total_counts_zscore > -3) & (total_counts_zscore < 3) & (mito_percentage_zscore > -3) & (
                mito_percentage_zscore < 3))

    # 仅保留符合条件的细胞
    adata = adata[keep_cells, :] #筛选行
    return adata

# 过滤编码蛋白质或miRNA的基因数少于7个的细胞
def gene_number_filter(adata, gene_list):
    indices = [k in gene_list for k in adata.var.index] 
    f_adata = adata[:, indices]
    data = f_adata.X.toarray()
    mask = np.count_nonzero(data, axis=1) > 6
    #计算data数组每行中非零元素的个数，并创建一个布尔索引数组mask，其中对应行的非零元素个数大于6的为True，小于等于6的为False
    return adata[mask]

def tokenize_cell(gene_vector, gene_list, token_dict):
    """
    Convert normalized gene expression vector to tokenized rank value encoding.
    """
    nonzero_mask = np.nonzero(gene_vector)[0] #返回非零位置索引
    sorted_indices = np.argsort(-gene_vector[nonzero_mask]) #对gene_vector中非零元素进行降序排序，并将排序后的索引存储在sorted_indices中
    gene_list = np.array(gene_list)[nonzero_mask][sorted_indices] #从gene_list中选择相应的基因，并按照排序顺序存储在gene_list中

    f_token = [token_dict[gene] for gene in gene_list]
    value = gene_vector[nonzero_mask][sorted_indices]
    return f_token, value.tolist()

def Normalized(adata):

    matrix_a = adata.X
    row_sums = np.sum(matrix_a, axis=1, keepdims=True)  
    normalized_array = matrix_a / row_sums 

    # 定义一个函数来计算每行的非零中位数  
    def non_zero_median(row):  
        non_zero_elements = row[row != 0]  
        if len(non_zero_elements) == 0:  
            # 如果没有非零元素，可以返回一个特定的值，比如np.nan或者一个很大的数  
            # 这里我们返回np.nan来表示没有可计算的中位数  
            return np.nan  
        else:  
            return np.median(non_zero_elements)  
    
    # 使用apply_along_axis来计算每行的非零中位数  
    median_per_row = np.apply_along_axis(non_zero_median, 1, normalized_array)  
    median_per_row = [[i] for i in median_per_row.tolist()]
    normalized_array = normalized_array / median_per_row
    normalized_array = np.nan_to_num(normalized_array)
    adata.X = normalized_array
    return adata


def log1p(adata):
    sc.pp.log1p(adata, base=2) #底数为2
    return adata

def rank_value(adata, token):
    
    input_ids = np.zeros((len(adata.X), 2048))  # Initialize input_ids as a 2D array filled with zeros
    values = np.zeros((len(adata.X), 2048))  # Initialize values as a 2D array filled with zeros
    length = []

    gene_id = adata.var.index.to_list()

    # 按行遍历，一行为一个细胞
    for index, i in enumerate(tqdm.tqdm(adata.X)):
        i = np.squeeze(np.asarray(i))
        tokenizen, value = tokenize_cell(i, gene_id, token)
        # 处理2048截断
        if len(tokenizen) > 2048:
            input_ids[index] = tokenizen[:2048]
            values[index] = value[:2048]
            length.append(2048)
        else:
            input_ids[index, :len(tokenizen)] = tokenizen
            values[index, :len(value)] = value
            input_ids[index, len(tokenizen):] = 0  # Fill remaining elements with zeros
            values[index, len(value):] = 0  # Fill remaining elements with zeros
            length.append(len(tokenizen))

    return input_ids,length,values

def transfor_out(adata,specices_str,length,input_ids,values):
    if specices_str == 'human':
        specices_int = 0
    elif specices_str =='mouse':
        specices_int = 1
    specices_int = [specices_int] * adata.X.shape[0]
    specices_int = [[x] for x in specices_int]
    length = [[x] for x in length]

    data_out = {'input_ids': input_ids,'values':values,'length': length,'species': specices_int}

    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int32')),
        'values': Sequence(feature=Value(dtype='float32')),
        'length': Sequence(feature=Value(dtype='int16')),
        'species': Sequence(feature=Value(dtype='int16')),
    })
    dataset = Dataset.from_dict(data_out, features=features)
    return dataset

def save_disk(patch_str,dataset,length):
    dataset.save_to_disk(patch_str)
    sorted_list = sorted(length)
    out_path = patch_str + '/sorted_length.pickle'
    with open(out_path, 'wb') as f:
        pickle.dump(sorted_list, f)

def prepare_data(gene_id_name_path, gene_token_path, specices_str, expr_file, log_c):

    with open(gene_id_name_path, 'rb') as f:
        dict1 = pickle.load(f)
    with open(gene_token_path, 'rb') as f:
        token_dict = pickle.load(f)
    specices_str = 'human'
    dict1_rev = dict(zip(dict1.values(), dict1.keys()))
    dict1 = dict1_rev
    df = pd.read_csv(expr_file)
    df = df.rename(columns={
        'Unnamed: 0': 'cell_id'
    }).set_index('cell_id')

    adata = sc.AnnData(df)
    # id_name translate
    gene_id_l = id_name_match1(name_list = adata.var.index.to_list(), dict1 = dict1)
    adata.var['gene_symbols'] = adata.var.index
    adata.var.index = gene_id_l
    print(len(adata.obs))
    print(len(adata.var))
    adata = adata[:, ~(adata.var.index == "delete")]
    print(len(adata.obs))
    print(len(adata.var))

    #filter不在token字典里面的基因
    gene_id_name_m = id_token_match(adata.var.index.to_list(), token_dict)
    adata.var['gene_symbols'] = adata.var.index
    adata.var.index = gene_id_name_m
    print(len(adata.obs))
    print(len(adata.var))
    adata = adata[:, ~(adata.var.index == "delete")]
    print(len(adata.obs))
    print(len(adata.var))

    #Normalized
    adata = Normalized(adata)

    # log1p
    adata = log1p(adata)
    print(len(adata.obs))
    print(len(adata.var))

    #Rank
    input_ids,length,values = rank_value(adata,token_dict)
    #输出dataset
    datasets = transfor_out(adata,specices_str,length,input_ids,values)

    return datasets


def get_embedded(data, gene_id_name_path, gene_token_path, grn_gene_path = None):

    if grn_gene_path != None: 
        GRN_gene_file = open(grn_gene_path, 'rb')
        GRN_gene = pickle.load(GRN_gene_file)
        GRN_gene_file.close()

    file = open(gene_token_path, 'rb')
    id_token = pickle.load(file)
    file.close()
    
    file = open(gene_id_name_path, 'rb')
    gene = pickle.load(file)
    file.close()

    id_gene = {}
    for token, id in id_token.items():
        if token in gene:
            id_gene[id] = gene[token]

    knowledges = dict()
    out = load_prior_embedding(
    # name2promoter_human_path, name2promoter_mouse_path, id2name_human_mouse_path,
    # token_dictionary
    # prior_embedding_path
    )
    knowledges['promoter'] = out[0]
    knowledges['co_exp'] = out[1]
    knowledges['gene_family'] = out[2]
    knowledges['peca_grn'] = out[3]
    knowledges['homologous_gene_human2mouse'] = out[4]  

    checkpoint_path = './models/GeneCompass_Base'
    checkpoint_path = '/mnt/sdb/ZYQ/workspace/GRN_LLM/model_file/GeneCompass/GeneCompass_Base'
    model = BertForMaskedLM.from_pretrained(
        checkpoint_path,
        knowledges=knowledges,
        ignore_mismatched_sizes=True,
    ).to("cuda")

    model.eval()
    gene_emb = {}
    with torch.no_grad():
        for i in tqdm(range(len(data))):
            input_id = torch.tensor(data[i]['input_ids']).unsqueeze(0).cuda()
            values = torch.tensor(data[i]['values']).unsqueeze(0).cuda()
            species = torch.tensor(data[i]['species']).unsqueeze(0).cuda()
            
            new_emb = model.bert.forward(input_ids=input_id, values= values, species=species)[0]
            new_emb = new_emb[:,1:,:].cpu()
            emb = torch.squeeze(emb, dim = 0)
            for idx, gene_id in enumerate(data[i]['input_ids']):
                if gene_id in id_gene:
                    gene_name = id_gene[gene_id]
                    if gene_name in gene_emb:
                        gene_emb[gene_name] = (gene_emb[gene_name] + emb[idx]) / 2
                    else:
                        gene_emb[gene_name] = emb[idx]
        
    return gene_emb




    # # 计算cosine similarity
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    final = np.zeros((len(gene_name_index), len(gene_name_index)))
    for i in range(len(gene_name_index)):
        for j in range(len(gene_name_index)):
            
            final[i][j] = cos(emb[gene_name_index[i]].reshape(1, -1), emb[gene_name_index[j]].reshape(1, -1)).cpu().numpy()

    result = copy.deepcopy(final)
      
    # index = np.argsort(final.ravel())[::-1][499999]
    # pos = np.unravel_index(index, final.shape)  
    # result[final < final[pos]] = 0
    # result[final >= final[pos]] = 1

    final_emb = []
    for i in range(len(gene_name_index)):
        final_emb.append(emb[gene_name_index[i]].reshape(1, -1).numpy())

    final_emb = np.concatenate((final_emb))

    dict_id_gene = {'idx':gene_name_index, 'gene':gene_name}
    df_id_gene = pd.DataFrame(dict_id_gene)

    return final, final_emb, df_id_gene


def save_file(path, final, final_emb, df_id_gene):
    np.save(f'{path}/embedding.npy', final_emb, allow_pickle=True)
    np.save(f'{path}/cos_grn.npy', final, allow_pickle=True)
    df_id_gene.to_csv(f'{path}/gene_id_name.csv', index=False)

def get_opt():
    parser = argparse.ArgumentParser(description='Prepare data for GRN inference')
    parser.add_argument('--gene_id_name_path', type=str, required=False, default='/mnt/sdb/ZYQ/workspace/GRN/LLM/GeneCompass/demo_data/Gene_id_name_dict.pickle', help='Gene id and name path')
    parser.add_argument('--gene_token_path', type=str, required=False, default='/mnt/sdb/ZYQ/workspace/GRN/LLM/GeneCompass/demo_data/h_m_token2000W.pickle', help='Gene token path')
    parser.add_argument('--specices_str', type=str, required=False, default='human', help='Specices string')
    parser.add_argument('--expr_file', type=str, required=False, default='/mnt/sdb/ZYQ/workspace/GRN/LLM/GeneCompass/demo_data/BL--ExpressionData.csv', help='Expression file path')
    parser.add_argument('--log_c', type=bool, default=False, help='log1p or not')
    parser.add_argument('--output', type=str, required=False, default='./output_embedding', help='Output directory')
    args = parser.parse_args()
    return args

def main():
    args = get_opt()
    gene_id_name_path = '/mnt/sdb/ZYQ/workspace/GRN/LLM/GeneCompass/demo_data/Gene_id_name_dict.pickle'
    gene_token_path = '/mnt/sdb/ZYQ/workspace/GRN/LLM/GeneCompass/demo_data/h_m_token2000W.pickle'
    grn_gene_path = '/mnt/sdb/ZYQ/workspace/GRN/LLM/GeneCompass/demo_data/GRNgene.pkl'
    specices_str = 'human'
    gene_id_name_path = args.gene_id_name_path
    gene_token_path = args.gene_token_path
    specices_str = args.specices_str
    expr_file = args.expr_file   # 行细胞，列基因，含行名跟列名，csv文件格式
    log_c = args.log_c

    output = args.output
    if not os.path.exists(output):
        os.makedirs(output)
    datasets = prepare_data(gene_id_name_path, gene_token_path, specices_str, expr_file, log_c)
    final, final_emb, df_id_gene = get_embedded(datasets, gene_id_name_path, gene_token_path, grn_gene_path = None)
    save_file(output, final, final_emb, df_id_gene)
    


# if __name__ == '__main__':
#     main()