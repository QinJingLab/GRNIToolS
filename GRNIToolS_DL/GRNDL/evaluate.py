import torch
import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import roc_curve, auc
from models import *
from param import *
from sklearn.metrics import average_precision_score
def draw_roc(label,pred,save_dir,filename):
    fpr, tpr, thread = roc_curve(label, pred)
    roc_auc = auc(fpr, tpr)
    #roc_auc = roc_auc_score(label, pred)
    # 绘图
    plt.figure()
    lw = 2
    
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(f'{save_dir}/{filename}_roc.png')



def evaluating(datasetloader, save_dir, filename, model_pth, model_idx = '1',args=None):

    dict_model = {'Trans': Config_Trans(), 'STGRNS':Config_STGRNS(), 'GRNCNN':Config_CNN(), 'ResNet18':Config_CNN(), 'GRNGCN':Config_GCN()}
    config = dict_model[model_idx]
    device = args.cuda if args else config.device
    n_heads = config.num_head
    d_models = config.dim_model
    num_class = config.num_classes
    input_size = config.input_size
    dropout = config.dropout
    n_hidden = config.hidden
    if model_idx == 'STGRNS':
        model = STGRNS(input_dim=input_size, nhead=n_heads, d_model=d_models, num_classes=num_class, dropout=dropout)
    if model_idx == 'GRNCNN':
        model = GRNCNN()
    if model_idx == 'ResNet18':
        model = ResNet18()
    if model_idx == 'Trans':
        model = GRNTrans(input_dim = input_size, num_head = n_heads, d_model = d_models ,hidden_size = n_hidden, dropout=dropout,batch_first=args.batch_first)
    if model_idx == 'GRNGCN':
        model = GRNGCN(datasetloader.num_features,config.hidden,64)
    print("Start evaluate model", model_idx, filename)
    model = model.to(device)
    model.load_state_dict(torch.load(model_pth))
    list_pred_y = []
    list_x=[]
    list_label = []
    y_score = []
    tp, tn, fp, fn = 0, 0, 0, 0
    if model_idx != 'GRNGCN':
        model.eval()
        for step, datas in enumerate(datasetloader):
            data, label = datas
            label = label.view(-1)
            data = data.to(device)
            label = label.to(device)
            output = model(data)
            list_label.extend(label.cpu().tolist())
            score, pred_y = torch.max(output,1)
            list_pred_y.extend(pred_y.cpu().tolist())
            list_x.extend(data.cpu().tolist())
        for pred_y, label in zip(list_label, list_pred_y):
            if pred_y == label == 1:
                tp += 1 
            if pred_y == label == 0:
                tn += 1
            if pred_y == 1 and pred_y != label:
                fp += 1
            if pred_y == 0 and pred_y != label:
                fn += 1
        res_mix_matrix = f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}"
        auc=roc_auc_score(list_label,list_pred_y)
        aupr = average_precision_score(list_label,list_pred_y)
        if (tp+fn)!=0 and (tn+fp)!=0 and (fp+tn)!=0 and (tp+fn)!=0:
            res_mix_matrix += '\nTPR: {:.3f}'.format(tp/(tp + fn))
            res_mix_matrix += '\nTNR: {:.3f}'.format(tn/(tn + fp))
            res_mix_matrix += '\nFPR: {:.3f}'.format(fp/(fp + tn))
            res_mix_matrix += '\nFNR: {:.3f}'.format(fn/(tp + fn))
            res_mix_matrix += '\nACC: {:.3f}'.format((tp + tn)/(tp + fp + fn + tn ))
        with open(f'{save_dir}/{filename}_mix_matrix.txt','w') as f:
            f.write(res_mix_matrix)
        print(auc)
        print(aupr)
        draw_roc(list_label,list_pred_y,save_dir,filename)

    elif model_idx == 'GRNGCN':
        #print("evaluate datasetloader")
        val_data = datasetloader

        def negative_sample():
            # 从训练集中采样与正边相同数量的负边
            neg_edge_index = negative_sampling(
                edge_index=val_data.edge_index, num_nodes=val_data.num_nodes,
                num_neg_samples=val_data.edge_label_index.size(1), method='sparse')
            # print(neg_edge_index.size(1))   # 3642条负边，即每次采样与训练集中正边数量一致的负边
            edge_label_index = torch.cat(
                [val_data.edge_label_index, neg_edge_index],
                dim=-1,
            )
            edge_label = torch.cat([
                val_data.edge_label,
                val_data.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)

            return edge_label, edge_label_index

        edge_label= val_data.edge_label
        edge_label_index =val_data.edge_label_index
        #print(edge_label_index)
        edge_label, edge_label_index = edge_label.to(device), edge_label_index.to(device) 
        model.eval()
        with torch.no_grad():
            F_mat_valid, A_mat_valid = val_data.x.to(device), val_data.edge_index.to(device)
            edge_label_index2 = edge_label_index.to(device)
            out2 = model(F_mat_valid, A_mat_valid, edge_label_index2).view(-1).sigmoid()    
            label, pred = edge_label.cpu().numpy(), out2.cpu().numpy()
            list_label = [int(i) for i in label]
            list_pred_y = pred.tolist()

            #print(list_pred_y)
            auc = roc_auc_score(edge_label.cpu().numpy(), out2.cpu().numpy())
            aupr = average_precision_score(edge_label.cpu().numpy(), out2.cpu().numpy())
            print(auc)
            print(aupr)
            draw_roc(list_label,list_pred_y,save_dir,filename)
            fpr, tpr, thresholds = roc_curve(list_label, list_pred_y)
            best_threshold = thresholds[np.argmax(tpr - fpr)]
            #best_threshold=0.6
            predicted_labels = [1 if prob >= best_threshold else 0 for prob in list_pred_y]
        for pred_y, label in zip(list_label, predicted_labels):
            if pred_y == label == 1:
                tp += 1 
            if pred_y == label == 0:
                tn += 1
            if pred_y == 1 and pred_y != label:
                fp += 1
            if pred_y == 0 and pred_y != label:
                fn += 1
        res_mix_matrix = f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}"
        if (tp+fn)!=0 and (tn+fp)!=0 and (fp+tn)!=0 and (tp+fn)!=0:
            res_mix_matrix += '\nTPR: {:.3f}'.format(tp/(tp + fn))
            res_mix_matrix += '\nTNR: {:.3f}'.format(tn/(tn + fp))
            res_mix_matrix += '\nFPR: {:.3f}'.format(fp/(fp + tn))
            res_mix_matrix += '\nFNR: {:.3f}'.format(fn/(tp + fn))
            res_mix_matrix += '\nACC: {:.3f}'.format((tp + tn)/(tp + fp + fn + tn ))
        with open(f'{save_dir}/{filename}_mix_matrix.txt','w') as f:
            f.write(res_mix_matrix)
    
        #for idx1, idx2, label in zip(edge_label_index[0].tolist(), edge_label_index[1].tolist(), list_label):
            #output.append(f"{idx1}\t{idx2}\t{label}")
        #with open(f'{save_dir}/{filename}_label.txt', 'w') as file:
         #   file.write('\n'.join(output))

        #for idx1, idx2, label in zip(edge_label_index[0].tolist(), edge_label_index[1].tolist(), list_pred_y):
       #     output1.append(f"{idx1}\t{idx2}\t{label}")
        #with open(f'{save_dir}/{filename}_predition.txt', 'w') as file:
        #    file.write('\n'.join(output1))

            
        



    # 读取另一个文件的前两列值
    '''new_columns = []
    if filename !='val':
        with open(f'{save_dir}/../train_res_{model_idx}/{filename}/train_pair.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
        # 根据适当的分隔符将每一行分割成列
                columns = line.strip().split('\t')  # 使用空格作为分隔符，可以根据实际情况进行修改
        # 获取前两列的值
                new_columns.append(columns[:2])
    else:
        with open(f'{save_dir}/../train_res_{model_idx}/train/val_pair.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
        # 根据适当的分隔符将每一行分割成列
                columns = line.strip().split('\t')  # 使用空格作为分隔符，可以根据实际情况进行修改
        # 获取前两列的值
                new_columns.append(columns[:2])
    # 组合新的列和list_label
    combined_data1 = [f"{col[0]} {col[1]} {label}" for col, label in zip(new_columns, list_label)]
    combined_data2 = [f"{col[0]} {col[1]} {label}" for col, label in zip(new_columns, list_pred_y)]
    #auc=roc_auc_score(list_label,list_pred_y)
    #print(auc)
    # 将组合数据写入新文件
    with open(f'{save_dir}/{filename}_label.txt','w') as file:
        file.write('\n'.join(combined_data1))
    
    with open(f'{save_dir}/{filename}_predict.txt','w') as file:
        file.write('\n'.join(combined_data2))
'''
    with open(f'{save_dir}/{filename}_auc.txt','w') as file:
        file.write('auc='+str(auc)+'\n'+'aupr='+str(aupr))
    return auc,aupr
    #with open(f'{save_dir}/{filename}_label.txt','w') as f:
    #    f.write('\n'.join(map(str, list_label)))
    #with open(f'{save_dir}/{filename}_predict.txt','w') as f:
    #    f.write('\n'.join(map(str, list_pred_y)))

    
