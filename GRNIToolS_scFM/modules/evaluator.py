import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import roc_auc_score, roc_curve




def cal_auc(y_true, y_scores, output):

    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    mpl.rcParams['pdf.fonttype'] = 42  
    mpl.rcParams['ps.fonttype'] = 42
    
    # font sizes
    base_size = 7
    mpl.rcParams['font.size'] = base_size
    mpl.rcParams['axes.labelsize'] = base_size + 1
    mpl.rcParams['axes.titlesize'] = base_size + 1
    mpl.rcParams['xtick.labelsize'] = base_size
    mpl.rcParams['ytick.labelsize'] = base_size
    mpl.rcParams['legend.fontsize'] = base_size
    
    # line widths
    mpl.rcParams['axes.linewidth'] = 0.75  
    mpl.rcParams['xtick.major.width'] = 0.75
    mpl.rcParams['ytick.major.width'] = 0.75
    mpl.rcParams['lines.linewidth'] = 1.5  

    # make sure the output directory exists
    if not os.path.exists(output):
        os.makedirs(output)

    auroc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)

    metrics_path = os.path.join(output, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f'AUROC: {auroc:.5f}\nAUPR: {aupr:.5f}')

    # colours
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
    
    # drop the top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # legend: no frame, lower right
    ax.legend(frameon=False, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(figsize=(4, 3.5), dpi=300)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ratio = sum(y_true) / len(y_true)
    
    # plot
    ax.plot(recall, precision, color=colors[1], label=f'(AUPR = {aupr:.2f})', zorder=2) 
    ax.axhline(y=ratio, color=base_color, linestyle='--', linewidth=1, label='Baseline', zorder=1)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='lower left')
    
    # save
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
    if label_df.shape[1] == 2:
        # a bare gold-standard edge list: every listed pair is an interaction
        label_df.columns = ['TF', 'Target']
        label_df['Label'] = 1
    else:
        label_df.columns = ['TF', 'Target', 'Label']
    label_df['TF'] = label_df['TF'].str.upper()
    label_df['Target'] = label_df['Target'].str.upper()

    # `0 in label_df['Label']` would test the *index*, which always contains 0,
    # so the dispatch has to compare values.
    y_true = label_df['Label']
    with_negative = (y_true == 0).any()

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
        
        net_values = matrix_net.values

        # Every listed edge among the shared genes is a positive, including the
        # ones the network did not score -- those count as 0 rather than being
        # dropped, which is what the merge-based path used to do.
        pos_indices = np.stack(np.where(matrix_label.values != 0), axis=1)
        pos_num = pos_indices.shape[0]
        pos_scores = net_values[pos_indices[:, 0], pos_indices[:, 1]]

        # Negatives come from every remaining pair, not only the pairs the
        # network happened to score.  The diagonal is excluded: a gene is not
        # its own regulator in this TF-by-target layout.
        neg_mask = (matrix_label.values == 0)
        np.fill_diagonal(neg_mask, False)
        neg_indices = np.stack(np.where(neg_mask), axis=1)

        neg_num = min(pos_num, neg_indices.shape[0])
        if pos_num == 0 or neg_num == 0:
            print("No usable positive/negative pairs found in the label file and network file.")
            return
        np.random.seed(42)
        selected_neg = neg_indices[np.random.choice(len(neg_indices), neg_num, replace=False)]
        neg_scores = net_values[selected_neg[:, 0], selected_neg[:, 1]]

        true_label = np.concatenate([np.ones(pos_num), np.zeros(neg_num)])
        pred_score = np.concatenate([pos_scores, neg_scores])
        cal_auc(true_label, pred_score, output)

 