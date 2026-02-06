import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, precision_recall_curve
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import seaborn as sns




def cal_auc(y_true, y_scores, output):

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['pdf.fonttype'] = 42  
    mpl.rcParams['ps.fonttype'] = 42
    
    # 字号设定
    base_size = 7
    mpl.rcParams['font.size'] = base_size
    mpl.rcParams['axes.labelsize'] = base_size + 1
    mpl.rcParams['axes.titlesize'] = base_size + 1
    mpl.rcParams['xtick.labelsize'] = base_size
    mpl.rcParams['ytick.labelsize'] = base_size
    mpl.rcParams['legend.fontsize'] = base_size
    
    # 线条设定
    mpl.rcParams['axes.linewidth'] = 0.75  
    mpl.rcParams['xtick.major.width'] = 0.75
    mpl.rcParams['ytick.major.width'] = 0.75
    mpl.rcParams['lines.linewidth'] = 1.5  

    # 确保输出目录存在
    if not os.path.exists(output):
        os.makedirs(output)

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    metrics_path = os.path.join(output, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'AUROC: {auroc:.5f}\nAUPR: {aupr:.5f}')

    # 定义颜色 
    colors = sns.color_palette("colorblind")
    main_color = colors[0] 
    base_color = '.5'       


    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=300)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    
    ax.plot(fpr, tpr, color=main_color, label=f'(AUROC = {auroc:.2f})', zorder=2)
    ax.plot([0, 1], [0, 1], color=base_color, linestyle='--', linewidth=1, label='Random', zorder=1)

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([-0.02, 1.02]) 
    ax.set_ylim([-0.02, 1.02])
    
    # 去除顶部和右侧边框 (Despine)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 设置图例 (无边框，左上角)
    ax.legend(frameon=False, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=300)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ratio = sum(y_true) / len(y_true)
    
    # 绘图
    ax.plot(recall, precision, color=colors[1], label=f'(AUPR = {aupr:.2f})', zorder=2) 
    ax.axhline(y=ratio, color=base_color, linestyle='--', linewidth=1, label='Baseline', zorder=1)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower left')
    
    # 保存
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
def evaluate(net, label, output):
    if os.path.exists(output) == False:
        os.makedirs(output)
    # Read the network file
    net_df = pd.read_csv(net, sep='\t', header=0)
    net_df.columns = ['TF', 'Target', 'Score']
    net_df['TF'] = net_df['TF'].str.upper()  
    net_df['Target'] = net_df['Target'].str.upper()

    label_df = pd.read_csv(label, sep='\t', header=None)
    label_df.columns = ['TF', 'Target', 'Label']
    label_df['TF'] = label_df['TF'].str.upper()  
    label_df['Target'] = label_df['Target'].str.upper()     

    try:
        y_true = label_df['Label']
        with_negative = 0 in y_true
    except:
        with_negative = False

    if with_negative:
        merged_df = pd.merge(net_df, label_df, on=['TF', 'Target'])
        if merged_df.shape[0] == 0:
            print("No matching rows found in the network file and label file.")
            return
        y_true = merged_df['Label']
        y_scores = merged_df['Score']
        cal_auc(y_true, y_scores, output)
    else:
        all_genes = pd.concat([label_df['TF'], label_df['Target']]).unique()
        net_genes = set(net_df['TF']).union(set(net_df['Target']))
        valid_genes = all_genes[np.isin(all_genes, list(net_genes))]
        gene_list = sorted(valid_genes)

        matrix_label = pd.DataFrame(data=0, index=gene_list, columns=gene_list, dtype=float)
        for _, row in label_df.iterrows():
            if row['TF'] in gene_list and row['Target'] in gene_list:
                matrix_label.loc[row['TF'], row['Target']] = 1.0  

        matrix_net = pd.DataFrame(data=0, index=gene_list, columns=gene_list, dtype=float)
        for _, row in net_df.iterrows():
            if row['TF'] in gene_list and row['Target'] in gene_list:
                matrix_net.loc[row['TF'], row['Target']] = row['Score']
        
        label_mask = (matrix_label != 0)
        net_mask = (matrix_net != 0)
        common_mask = label_mask & net_mask
        common_indices = matrix_label[common_mask].stack().index.tolist()
        common_df = pd.DataFrame(common_indices, columns=['TF', 'Target'])        
        pos_common_df = pd.merge(common_df, net_df[['TF', 'Target', 'Score']], 
                                 on=['TF', 'Target'], how='left')
        pos_num = pos_common_df.shape[0]


        neg_mask = (matrix_label == 0) & (matrix_net != 0)
        neg_indices = np.stack(np.where(neg_mask), axis=1)
        
        if neg_indices.shape[0] > pos_num:
            neg_num = pos_num
        else:
            neg_num = neg_indices.shape[0]
        np.random.seed(42)  
        selected_neg = neg_indices[np.random.choice(len(neg_indices), neg_num, replace=False)]
        
        neg_df = pd.DataFrame({
            'TF': matrix_label.index[selected_neg[:,0]],
            'Target': matrix_label.columns[selected_neg[:,1]]
        })
        neg_df = pd.merge(neg_df, net_df,
                        on=['TF', 'Target'], how='left')
        neg_df.columns = ['TF', 'Target', 'Score']

        true_label = np.concatenate([np.ones(pos_num), np.zeros(neg_num)])
        pred_score = np.concatenate([pos_common_df['Score'].values, neg_df['Score'].values])
        cal_auc(true_label, pred_score, output)

 