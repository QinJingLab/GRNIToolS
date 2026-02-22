import pandas as pd
import random
import os
import argparse

def argsment():
    parser = argparse.ArgumentParser(description="preprocessing data")
    parser.add_argument("-e", "--expression", default='/mnt/sdb/ZYQ/workspace/GRN/DL/dataset/SC-PC/ExpressionData.csv', help="Input the gene expression matrix")
    parser.add_argument("-n", "--network", default='/mnt/sdb/ZYQ/workspace/GRN/DL/dataset/SC-PC/refNetwork.csv', help="Input the GRN")
    parser.add_argument("-c", "--cell_num", default=10000, help="Input the GRN")
    parser.add_argument("-o", "--output", default='data_pre_output', help="Output dir")
    return parser.parse_args()

def preprocess(expr_file, network_file, cell_num, output):
    if not os.path.exists(output):
        os.mkdir(output)

    # load the data and convert the gene to upper case
    df_expr = pd.read_csv(expr_file, delimiter = ',', index_col = 0)
    df_expr.index = [gene.upper() for gene in df_expr.index]
    df_network = pd.read_csv(network_file, delimiter = ',')
    df_network['Gene1'] = [TF.upper() for TF in df_network['Gene1']]
    df_network['Gene2'] = [TF.upper() for TF in df_network['Gene2']]

    # if the sample over the value such as 1000, randomly choose 1000
    df_expr2 = df_expr.T
    if df_expr2.shape[0] > cell_num:
        random.seed(123)
        id = random.sample(range(df_expr2.shape[0]), cell_num)
        df_expr_filter = df_expr2.iloc[id]
    else:
        df_expr_filter = df_expr2

    # delete the zero expression gene 

    column_sums = df_expr_filter.sum()  
    nonzero_columns = column_sums[column_sums != 0]  
    nonzero_column_names = nonzero_columns.index.tolist()  
    df_expr_filter_nonzero = df_expr_filter[nonzero_column_names]  

    ## save expression
    df_expr_filter_nonzero = df_expr_filter_nonzero.T
    path_file = output + '/expression.csv'
    df_expr_filter_nonzero.to_csv(path_file)

    ## save network which gene and tf in the expression
    df_expr = df_expr_filter_nonzero
    filtered_df_network = df_network[(df_network['Gene1'].isin(df_expr.index)) & (df_network['Gene2'].isin(df_expr.index))]  

    path_file = output + '/network.csv'
    filtered_df_network.to_csv(path_file, index = False)

def main():
    args = argsment()
    preprocess(args.expression, args.network, args.cell_num, args.output)   # 10000 is the cell number, you can change it according to your needs.

# python preprocess.py -e /mnt/sdb/ZYQ/workspace/GRN/DL/dataset/SC-PC/ExpressionData.csv -n /mnt/sdb/ZYQ/workspace/GRN/DL/dataset/SC-PC/refNetwork.csv  -c 10000 -o data_pre_output
if __name__ == '__main__':
    main()