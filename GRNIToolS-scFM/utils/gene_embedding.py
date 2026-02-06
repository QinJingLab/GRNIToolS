import os
import sys
import pickle
import re
import gc
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

def get_project_root():
    try:
        path = Path(__file__).resolve().parent.parent
    except NameError:
        path = Path.cwd()
        
    if not (path / "model_file").exists():
        if (path.parent / "model_file").exists():
            path = path.parent
        else:
            print(f"Warning: 无法在 {path} 找到 'model_file' 目录。请检查你的运行路径。")
    return path

def clear_gpu_memory():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_cell_nums(sc_df, max_cell_size = 1000):
    np.random.seed(42)
    if sc_df.shape[1] > max_cell_size:
        column_indices = np.random.choice(sc_df.columns.tolist(), max_cell_size, replace=False)
        sampled_df = sc_df[column_indices]
    else:
        sampled_df = sc_df
    return sampled_df

def get_cus_gene_emb(expr_path, embedding):
    expr_df = pd.read_csv(expr_path,index_col=0)
    genelist = expr_df.index.tolist()
    gene_emb = {}
    for gene in genelist:
        gene = gene.upper()
        if gene in embedding:
            gene_emb[gene] = embedding[gene]
    return gene_emb

def get_emb_GenePT(pt_path, expr_path):
    with open(pt_path, "rb") as fp:
        GenePT_embeddings = pickle.load(fp)
    return get_cus_gene_emb(expr_path, GenePT_embeddings)

def get_emb_scBERT(pt_path, expr_path):
    root = get_project_root()
    source_path = str(root / "source" / "scBERT")

    gene_list = []
    with open(pt_path + '/genelist.txt','r') as f:
        for line in f:
            gene_list.append(line.strip())

    sys.path.insert(0, source_path)
    from performer_pytorch import PerformerLM
    model = PerformerLM(
        num_tokens = 7,
        dim = 200,
        depth = 6,
        heads = 10,
        max_seq_len = 24447,
        g2v_position_emb=True,
        g2v_file=pt_path + '/gene2vec_21249.npy'
        )
    emb =  model.pos_emb.emb.weight

    gene_emb = {}
    for i in range(emb.shape[0]-1):
        gene_emb[gene_list[i]] = emb[i].tolist()

    return get_cus_gene_emb(expr_path, gene_emb)

def get_emb_CellPLM(pt_path, expr_path):
    import joblib
    from CellPLM.utils import set_seed
    from CellPLM.pipeline.cell_embedding import CellEmbeddingPipeline
    import mygene
    def ensembl_to_symbol(gene_list):
        import mygene
        mg = mygene.MyGeneInfo()
        return mg.querymany(gene_list, scopes='ensembl.gene', fields='symbol', as_dataframe=True,
                    species='human').reset_index().drop_duplicates(subset='query')['symbol'].fillna('0').tolist()

    set_seed(42)
    PRETRAIN_VERSION = '20231027_85M'
    pipeline = CellEmbeddingPipeline(pretrain_prefix=PRETRAIN_VERSION, 
                                     pretrain_directory=pt_path)
    model = pipeline.model
    model.eval()
    emb = model.embedder.feat_enc.emb
    ensembl_list = model.gene_set 
    gene_list =  ensembl_to_symbol(ensembl_list)
    emb_list = [emb[i].tolist() for i in range(len(gene_list))]
    gene_emb = dict(zip(gene_list, emb_list))
    gene_emb = joblib.load(f'{pt_path}/gene_embeddings.pkl')   

    return get_cus_gene_emb(expr_path, gene_emb)

def get_emb_iSEEEK(pt_path, expr_path):
    from transformers import PreTrainedTokenizerFast, BertForMaskedLM

    tokenizer = PreTrainedTokenizerFast.from_pretrained(pt_path)
    model = BertForMaskedLM.from_pretrained(pt_path).bert
    gene_to_id = tokenizer.vocab 
    emb = model.embeddings.word_embeddings.weight
    gene_to_id = dict(sorted(gene_to_id.items(), key = lambda i:i[1]))
    gene_list = list(gene_to_id.keys())
    emb_list = [emb[i].tolist() for i in gene_to_id.values()]
    gene_emb = dict(zip(gene_list, emb_list))

    return get_cus_gene_emb(expr_path, gene_emb)

def get_emb_gene2vec(pt_path, expr_path):
    gene_emb = {}
    with open(pt_path, 'r') as f:
        for line in f:
            gene, vector = line.strip().split('\t')
            gene = gene.upper() 
            gene_emb[gene] = [float(v) for v in vector.split(' ')]
    return get_cus_gene_emb(expr_path, gene_emb)

def get_emb_scGPT(pt_path, expr_path):
    import json
    import torch
    import scanpy as sc
    from scgpt.model import TransformerModel
    from scgpt.tokenizer import GeneVocab
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pad_token = "<pad>"
    special_tokens = [pad_token, "<cls>", "<eoc>"]
    n_bins = 51
    pad_value = -2
    n_input_bins = n_bins
 
    # load model
    model_dir = Path(pt_path)
    model_config_file = model_dir / "args.json"
    model_file = model_dir / "best_model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    # Retrieve model parameters from config files
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)

    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]

    gene2idx = vocab.get_stoi()   

    config_path = os.path.join(pt_path, "args.json")
    with open(config_path, "r") as f:
        model_configs = json.load(f)

    vocab_path = os.path.join(pt_path, "vocab.json")
    vocab = GeneVocab.from_file(vocab_path)

    ntokens = len(vocab)  # size of vocabulary
    model = TransformerModel(
        ntokens,
        embsize,
        nhead,
        d_hid,
        nlayers,
        vocab=vocab,
        pad_value=pad_value,
        n_input_bins=n_input_bins,
    )
    try:
        model.load_state_dict(torch.load(model_file))
    except:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        for k, v in pretrained_dict.items():
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
    model.to(device)

    df = pd.read_csv(expr_path, index_col=0)
    adata = sc.AnnData(df.T)
    adata.var_names = adata.var_names.str.upper()
    adata.obs["batch"] = "batch1"
    adata.obs["celltype"] = "None" 
    adata.obs["final_annotation"] = "None" 
    adata.var_names_make_unique()

    gene_ids = np.array([id for id in gene2idx.values()])
    gene_embeddings = model.encoder(torch.tensor(gene_ids, dtype=torch.long).to(device))
    gene_embeddings = gene_embeddings.detach().cpu().numpy()
    gene_embeddings = {gene: gene_embeddings[i] for i, gene in enumerate(gene2idx.keys()) if gene in adata.var.index.tolist()}
    return gene_embeddings
  

def get_emb_Geneformer(pt_path, expr_path, output_dir):
    ### create loom file
    import loompy
    from geneformer import TranscriptomeTokenizer
    from geneformer import EmbExtractor
    dataset_path = f'{output_dir}/input'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    root = get_project_root()
    moldir_path = str(root / "model_file" / "Geneformer")

    with open(f'{moldir_path}/ensembl_mapping_dict_gc95M.pkl', "rb") as f:
        ensembl_gene_dict = pickle.load(f)
    with open(f'{moldir_path}/token_dictionary_gc95M.pkl', "rb") as f:
        gene_token_dict = pickle.load(f)

    df = pd.read_csv(expr_path, index_col=0)
    df = get_cell_nums(df)
    ensembl_list = []
    gene_list = []
    columns_to_remove = []
    for i, gene in enumerate(df.index):
        gene = gene.upper()
        if gene in ensembl_gene_dict:
            ensembl_id = ensembl_gene_dict[gene]
            if ensembl_id in gene_token_dict:
                ensembl_list.append(ensembl_id)
                gene_list.append(gene)
            else:
                columns_to_remove.append(i)
        else:
            columns_to_remove.append(i)    
    cell_id = df.columns.tolist()
    x = df.to_numpy()
    columns_to_keep = np.setdiff1d(np.arange(x.shape[0]), columns_to_remove)
    x2 = x[columns_to_keep,:]
    n_counts = x2.sum(axis=1).round().tolist()
    n_counts_col = x2.sum(axis=0).round().tolist()
    cell_type = ['mESC'] * x2.shape[1]
    org_type = ['embryo'] * x2.shape[1]
    row_attrs = {"gene_name": gene_list, "ensembl_id" : ensembl_list, 'n_counts' : n_counts}  
    col_attrs = {"cell_id": cell_id, "cell_type": cell_type, 'organ_major': org_type ,'n_counts' : n_counts_col} 
    
    filename = f"{dataset_path}/single_cell_data.loom"
    loompy.create(filename, x2, row_attrs, col_attrs)

    ## tokenizing
    if re.search('2048', pt_path):
        model_input_size = 2048
        batch_size = 200
    elif re.search('4096', pt_path):
        model_input_size = 4096
        batch_size = 100

    tk = TranscriptomeTokenizer({"cell_type": "cell_type", "organ_major": "organ"}, nproc=16, model_input_size = model_input_size)
    tk.tokenize_data(dataset_path, 
                     dataset_path,  
                     'output_prefix', 
                     file_format="loom")
    
    ## embedding
    embex = EmbExtractor(model_type="GeneClassifier",
                        emb_mode = 'gene',
                        num_classes=0,
                        emb_layer=-1,
                        max_ncells=1500,
                        forward_batch_size=batch_size,
                        nproc=16)
    dataset = f'{dataset_path}/output_prefix.dataset'
    embs = embex.extract_embs(pt_path,
                              dataset,
                              dataset_path,
                              "output_prefix")

    df_id = pd.read_csv(f'{moldir_path}/example_input_files_gene_info_table.csv')
    gene_ensembl_dict = dict(zip(df_id['ensembl_id'],df_id['gene_name'].tolist()))

    gene_emb = {}
    for ensembl_gene in embs.index:
        if ensembl_gene in gene_ensembl_dict:
            gene = gene_ensembl_dict[ensembl_gene]
            gene_emb[gene] = embs.loc[ensembl_gene].tolist()

    return gene_emb


def get_emb_GeneCompass(pt_path, expr_file, specices_str):
    import scanpy as sc
    import tqdm as tqdm
    import torch
    from datasets import Dataset,Features,Sequence,Value,load_from_disk
    from genecompass import BertForMaskedLM
    from genecompass.utils import load_prior_embedding
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

    def Normalized(adata):
        matrix_a = adata.X
        row_sums = np.sum(matrix_a, axis=1, keepdims=True)  
        normalized_array = matrix_a / row_sums 

        def non_zero_median(row):  
            non_zero_elements = row[row != 0]  
            if len(non_zero_elements) == 0:  
                return np.nan  
            else:  
                return np.median(non_zero_elements)  
            
        median_per_row = np.apply_along_axis(non_zero_median, 1, normalized_array)  
        median_per_row = [[i] for i in median_per_row.tolist()]
        normalized_array = normalized_array / median_per_row
        normalized_array = np.nan_to_num(normalized_array)
        adata.X = normalized_array
        return adata

    def log1p(adata):
        sc.pp.log1p(adata, base=2) 
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

    def transfor_out(adata,specices_str,length,input_ids,values):
        if specices_str == 'human':
            specices_int = 0
        elif specices_str =='mouse':
            specices_int = 1
        specices_int = 0 
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

    gene_id_name_path = 'model_file/GeneCompass/Gene_id_name_dict.pickle'
    gene_token_path = 'model_file/GeneCompass/h_m_token2000W.pickle'

    with open(gene_id_name_path, 'rb') as f:
        dict1 = pickle.load(f)
    with open(gene_token_path, 'rb') as f:
        token_dict = pickle.load(f)

    dict1_rev = dict(zip(dict1.values(), dict1.keys()))
    dict1 = dict1_rev

    df = pd.read_csv(expr_file, index_col=0)
    df = get_cell_nums(df)
    df = df.T
    df.index.name = 'cell_id'
    gene_upper = [g.upper() for g in df.columns.tolist()]
    df.columns = gene_upper

    adata = sc.AnnData(df)
    # id_name translate
    gene_id_l = id_name_match1(name_list = adata.var.index.to_list(), dict1 = dict1)
    adata.var['gene_symbols'] = adata.var.index
    adata.var.index = gene_id_l
    adata = adata[:, ~(adata.var.index == "delete")]

    #filter不在token字典里面的基因
    gene_id_name_m = id_token_match(adata.var.index.to_list(), token_dict)
    adata.var['gene_symbols'] = adata.var.index
    adata.var.index = gene_id_name_m
    adata = adata[:, ~(adata.var.index == "delete")]

    #Normalized
    adata = Normalized(adata)

    # log1p
    adata = log1p(adata)
    
    #Rank
    input_ids,length,values = rank_value(adata,token_dict)
    #输出dataset
    data = transfor_out(adata,specices_str,length,input_ids,values)
    
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
    checkpoint_path = pt_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForMaskedLM.from_pretrained(
        checkpoint_path,
        knowledges=knowledges,
        ignore_mismatched_sizes=True,
    ).to(device)
    model.eval()
    gene_emb = {}
    with torch.no_grad():
        for i in tqdm.trange(len(data)):
            input_id = torch.tensor(data[i]['input_ids']).unsqueeze(0).to(device)
            values = torch.tensor(data[i]['values']).unsqueeze(0).to(device)
            species = torch.tensor(data[i]['species']).unsqueeze(0).to(device)
            
            new_emb = model.bert.forward(input_ids=input_id, values= values, species=species)[0]
            new_emb = new_emb[:,1:,:].to('cpu')
            emb = torch.squeeze(new_emb, dim = 0)
            for idx, gene_id in enumerate(data[i]['input_ids']):
                if gene_id in id_gene:
                    gene_name = id_gene[gene_id]
                    if gene_name in gene_emb:
                        gene_emb[gene_name] = (gene_emb[gene_name] + emb[idx]) / 2
                    else:
                        gene_emb[gene_name] = emb[idx]
    return gene_emb

    
def get_emb_scFoundation(pt_path, expr_path, output_dir):
    import subprocess
    dataset_path = f'{output_dir}/input'
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    root = get_project_root()
    model_path = str(root / "model_file" / "scFoundation")
    script_dir = root / 'source' / 'scFoundation' / 'model'
    model_cpk_path = str(root / pt_path)
    ## data prepare
    gene_idx_file = f'{model_path}/OS_scRNA_gene_index.19264.tsv'
    gene_idx = pd.read_csv(gene_idx_file, header=0, delimiter='\t')
    gene_list = list(gene_idx['gene_name'])
    df = pd.read_csv(expr_path, index_col=0)
    df = get_cell_nums(df)
    df.index = [i.upper() for i in df.index]
    df = df.T
    intersection = list(set(gene_list).intersection(set(df.columns)))
    df = df[intersection]
    df.to_csv(f'{dataset_path}/scFoundation_expr.csv')

    ## embedding
    input_type = 'singlecell'
    normalized = 'F'
    command = [
        "python", "get_embedding.py",
        "--task_name", "emb",
        "--input_type", input_type,
        "--output_type", "gene",
        "--pool_type", 'all',
        '--tgthighres','a5',
        "--data_path", f'{root}/{dataset_path}/scFoundation_expr.csv',
        "--save_path", f'{root}/{dataset_path}',
        "--model_path", model_cpk_path,
        "--pre_normalized", normalized,
        "--version", "rd"
    ]
    result = subprocess.run(command, capture_output=True, text=True, cwd=str(script_dir))
    res = result.stdout
    with open(f'{output_dir}/log.txt','w') as f:
        f.write(res)

    npy_file = [f for f in os.listdir(dataset_path) if f.endswith('.npy')][0]
    gene_embedding = np.load(f'{dataset_path}/{npy_file}')
    if np.isnan(gene_embedding).any():
        gene_embedding_mean = np.nanmean(gene_embedding, axis=0)
    else:
        gene_embedding_mean = np.mean(gene_embedding, axis=0)
    gene_emb = dict(zip(gene_list, gene_embedding_mean))

    return gene_emb



def get_gene_embedding(llm, pt, expr_path, species, output_dir):
    if llm == 'GenePT':
        gene_embeddings = get_emb_GenePT(pt, expr_path)
    elif llm == 'scBERT':
        gene_embeddings = get_emb_scBERT(pt, expr_path)
    elif llm == 'Gene2vec':
        gene_embeddings = get_emb_gene2vec(pt, expr_path)
    elif llm == 'iSEEEK':
        gene_embeddings = get_emb_iSEEEK(pt, expr_path)  
    elif llm == 'CellPLM':
        gene_embeddings = get_emb_CellPLM(pt, expr_path)
    elif llm == 'GeneCompass':
        gene_embeddings = get_emb_GeneCompass(pt, expr_path, species)
    elif llm == 'Geneformer':
        gene_embeddings = get_emb_Geneformer(pt, expr_path, output_dir)
    elif llm == 'scFoundation':
        gene_embeddings = get_emb_scFoundation(pt, expr_path, output_dir)
    elif llm == 'scGPT':
        gene_embeddings = get_emb_scGPT(pt, expr_path)
    else:
        raise ValueError(f"Unsupported LLM model: {llm}")
    
    clear_gpu_memory()
    return gene_embeddings