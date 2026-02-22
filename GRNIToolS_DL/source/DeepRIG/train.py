from __future__ import division
from __future__ import print_function
import numpy as np
import time
import tensorflow.compat.v1 as tf
from util.utils import *
from deeprig.models import DeepRIG
from sklearn.metrics import roc_auc_score, average_precision_score



def train_o(FLAGS, adj, features, train_arr, test_arr, labels, AM, gene_names, TF, result_path):
    # Load data
    adj, size_gene, logits_train, logits_test, train_mask, test_mask, labels= load_data(
        adj, train_arr, test_arr, labels, AM)
    
    # Some preprocessing
    if FLAGS.model == 'DeepRIG':
        model_func = DeepRIG
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    tf.compat.v1.disable_eager_execution()
    placeholders = {
        'adjacency_matrix': tf.placeholder(tf.int32, shape=adj.shape),
        'features': tf.placeholder(tf.float32, shape = features.shape),
        'labels': tf.placeholder(tf.float32, shape=(None, train_pos_logits.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'negative_mask': tf.placeholder(tf.int32)
    }
    size_gene = len(gene_names) 
    input_dim = features.shape[1]
    # Create model
    model = model_func(placeholders, input_dim, size_gene, FLAGS.dim)

    # Initialize session
    sess = tf.Session()

    # Define model evaluation function
    def evaluate(adj, features, labels, mask, negative_mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(adj, features, labels, mask, negative_mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], 1 - outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # Construct feed dictionary
        # negative_mask, label_neg = generate_mask(labels, FLAGS.ratio, len(train_arr), size_gene)
        # negative_mask: gene * gene的adj矩阵，再转1D mask = np.reshape(mask, [-1, 1])
        # label_neg: gene * 2, tf-gene 位置数组

        # logits_train, train_mask, negative_mask 来自get_mask

        feed_dict = construct_feed_dict(adj, features, train_pos_logits, train_pos_mask, train_neg_mask, placeholders)

        # feed_dict = construct_feed_dict(adj, features, logits_train, train_mask, negative_mask, placeholders)

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

        # Print results
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(1 - outs[2]),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    # # Save trained model
    # model.save(sess, './saved_models/')
    
    # Testing
    # test_negative_mask, test_label_neg = generate_mask(labels, FLAGS.ratio, len(test_arr), size_gene)
    test_cost, test_acc, test_duration = evaluate(adj, features, test_pos_logits, test_pos_mask, test_neg_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost),
          "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    #Save results
    feed_dict_val = construct_feed_dict(adj, features, test_pos_logits, test_pos_mask, test_neg_mask, placeholders)
    outs, reps = sess.run([model.outputs, model.hid], feed_dict=feed_dict_val)
    outs = np.array(outs)[:, 0]
    outs = outs.reshape((size_gene, size_gene))

    # save the predicted matrix
    train_pos_logits = train_pos_logits.reshape(outs.shape)
    TF_mask = np.zeros(outs.shape)
    for i, item in enumerate(gene_names):
        for j in range(len(gene_names)):
            if i == j or (train_pos_logits[i, j] != 0):
                continue
            if item in TF:
                TF_mask[i, j] = 1
    geneNames = np.array(gene_names)
    idx_rec, idx_send = np.where(TF_mask)
    results = pd.DataFrame(
        {'Gene1': geneNames[idx_rec], 'Gene2': geneNames[idx_send], 'EdgeWeight': (outs[idx_rec, idx_send])})
    results = results.sort_values(by = ['EdgeWeight'], axis = 0, ascending = False)
    return results


def get_mask(data_path, gene_names, reverse_flags=0):
    data_file = pd.read_csv(data_path, header=0, sep = ',')
    pos, neg = [[],[],[]], [[],[],[]]
    num_genes = len(gene_names)

    
    if reverse_flags == 0:
        for row_index, row in data_file.iterrows():   # 0 ; Gene1  AHR Gene2 110032F04RIK
            if not ( (row.iloc[0] in gene_names) and (row.iloc[1] in gene_names)):
                continue
            tf_id, gene_id = gene_names.index(row.iloc[0]), gene_names.index(row.iloc[1])
            label = int(row.iloc[2])
            if label == 1:
                pos[0].append(tf_id)
                pos[1].append(gene_id)
                pos[2].append(label)
            elif label == 0:
                neg[0].append(tf_id)
                neg[1].append(gene_id)
                neg[2].append(label)

    pos_logits_dataset = sp.csr_matrix((pos[2], (pos[0], pos[1])), shape = (num_genes, num_genes)).toarray()
    pos_logits_dataset = pos_logits_dataset.reshape([-1, 1])
    pos_mask = np.array(pos_logits_dataset[:, 0], dtype=np.bool_).reshape([-1, 1])

    neg_label = np.array(list(zip(neg[0],neg[1])))
    neg_mask = np.zeros((num_genes, num_genes))
    for i in neg_label:
        neg_mask[i[0], i[1]] = 1
    neg_mask = np.reshape(neg_mask, [-1, 1])
    return pos_logits_dataset, pos_mask, neg_mask 


def train(FLAGS, adj, features, train_path, val_path, test_path, tftest_path, labels, reverse_flags, gene_names, TF, result_path):

    # Load data
    train_pos_logits, train_pos_mask, train_neg_mask = get_mask(train_path, gene_names, reverse_flags)
    val_pos_logits, val_pos_mask, val_neg_mask = get_mask(val_path, gene_names, reverse_flags)
    test_pos_logits, test_pos_mask, test_neg_mask = get_mask(test_path, gene_names, reverse_flags)
    tftest_pos_logits, tftest_pos_mask, tftest_neg_mask = get_mask(tftest_path, gene_names, reverse_flags)

    # Some preprocessing
    if FLAGS.model == 'DeepRIG':
        model_func = DeepRIG
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    tf.compat.v1.disable_eager_execution()
    placeholders = {
        'adjacency_matrix': tf.placeholder(tf.int32, shape=adj.shape),
        'features': tf.placeholder(tf.float32, shape = features.shape),
        'labels': tf.placeholder(tf.float32, shape=(None, train_pos_logits.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'negative_mask': tf.placeholder(tf.int32)
    }
    size_gene = len(gene_names) 
    input_dim = features.shape[1]

    # Initialize session
    # sess = tf.Session()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = "1" 
    sess = tf.Session(config=config)

    # with tf.device('/device:GPU:1'):
    # Create model
    model = model_func(placeholders, input_dim, size_gene, FLAGS.dim)

    # Define model evaluation function
    def evaluate(adj, features, labels, mask, negative_mask, placeholders):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(adj, features, labels, mask, negative_mask, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        return outs_val[0], 1 - outs_val[1], (time.time() - t_test)

    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):
        t = time.time()
        # logits_train, train_mask, negative_mask 来自get_mask
        feed_dict = construct_feed_dict(adj, features, train_pos_logits, train_pos_mask, train_neg_mask, placeholders)
        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
        # Print results
        print("Epoch:", '%04d' % (epoch + 1),
              "train_loss=", "{:.5f}".format(outs[1]),
              "train_acc=", "{:.5f}".format(1 - outs[2]),
              "time=", "{:.5f}".format(time.time() - t))

    print("Optimization Finished!")

    # # Save trained model
    # 创建 Saver 对象
    saver = tf.train.Saver()
    # 保存模型到指定路径
    saver.save(sess, f'{result_path}/model.ckpt')

    
    #Save results
    # def save_results(adj, features, train_pos_logits, test_pos_logits, test_pos_mask, test_neg_mask, placeholders):
    #     test_cost, test_acc, test_duration = evaluate(adj, features, test_pos_logits, test_pos_mask, test_neg_mask, placeholders)
    #     print("Test set results:", "cost=", "{:.5f}".format(test_cost),
    #         "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
    #     feed_dict_val = construct_feed_dict(adj, features, test_pos_logits, test_pos_mask, test_neg_mask, placeholders)
    #     outs, reps = sess.run([model.outputs, model.hid], feed_dict=feed_dict_val)
    #     outs = np.array(outs)[:, 0]
    #     outs = outs.reshape((size_gene, size_gene))
    #     train_pos_logits = train_pos_logits.reshape(outs.shape)
    #     TF_mask = np.zeros(outs.shape)
    #     for i, item in enumerate(gene_names):
    #         for j in range(len(gene_names)):
    #             if i == j or (train_pos_logits[i, j] != 0):
    #                 continue
    #             if item in TF:
    #                 TF_mask[i, j] = 1
    #     geneNames = np.array(gene_names)
    #     idx_rec, idx_send = np.where(TF_mask)
    #     results = pd.DataFrame(
    #         {'Gene1': geneNames[idx_rec], 'Gene2': geneNames[idx_send], 'EdgeWeight': (outs[idx_rec, idx_send])})
    #     results['EdgeWeight'] = abs(results['EdgeWeight'])
    #     results = results.sort_values(by = ['EdgeWeight'], axis = 0, ascending = False)

    #     return results
    

    def save_results(adj, features, train_pos_logits, test_pos_logits, test_pos_mask, test_neg_mask, placeholders, data_path):
        test_cost, test_acc, test_duration = evaluate(adj, features, test_pos_logits, test_pos_mask, test_neg_mask, placeholders)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
            "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))
        feed_dict_val = construct_feed_dict(adj, features, test_pos_logits, test_pos_mask, test_neg_mask, placeholders)
        outs, reps = sess.run([model.outputs, model.hid], feed_dict=feed_dict_val)
        outs = np.array(outs)[:, 0]
        outs = outs.reshape((size_gene, size_gene))
        outs = abs(outs)

        data_file = pd.read_csv(data_path, header=0, sep = ',')
        pos, neg = [[],[],[]], [[],[],[]]
        for row_index, row in data_file.iterrows():   # 0 ; Gene1  AHR Gene2 110032F04RIK
            tf_id, gene_id = gene_names.index(row.iloc[0]), gene_names.index(row.iloc[1])
            label = int(row.iloc[2])
            if label == 1:
                pos[0].append(tf_id)
                pos[1].append(gene_id)
                pos[2].append(label)
            elif label == 0:
                neg[0].append(tf_id)
                neg[1].append(gene_id)
                neg[2].append(label)
        pred, label = [], []
        for i in range(len(pos[0])):
            tf, gene = pos[0][i], pos[1][i]
            label.append(int(pos[2][i]))
            pred.append(outs[tf,gene])

        for i in range(len(neg[0])):
            tf, gene = neg[0][i], neg[1][i]
            label.append(int(neg[2][i]))
            pred.append(outs[tf,gene])


        train_pos_logits = train_pos_logits.reshape(outs.shape)
        TF_mask = np.zeros(outs.shape)
        for i, item in enumerate(gene_names):
            for j in range(len(gene_names)):
                if i == j or (train_pos_logits[i, j] != 0):
                    continue
                if item in TF:
                    TF_mask[i, j] = 1
        geneNames = np.array(gene_names)
        idx_rec, idx_send = np.where(TF_mask)
        results = pd.DataFrame(
            {'Gene1': geneNames[idx_rec], 'Gene2': geneNames[idx_send], 'EdgeWeight': (outs[idx_rec, idx_send])})
        results['EdgeWeight'] = abs(results['EdgeWeight'])
        results = results.sort_values(by = ['EdgeWeight'], axis = 0, ascending = False)
        
        roc_auc = roc_auc_score(label,pred).round(5)
        pr_auc = average_precision_score(label, pred).round(5)
        return roc_auc, pr_auc, results


    roc_auc1, pr_auc1, res_valid = save_results(adj, features, train_pos_logits, val_pos_logits, val_pos_mask, val_neg_mask, placeholders, val_path)
    roc_auc2, pr_auc2, res_test = save_results(adj, features, train_pos_logits, test_pos_logits, test_pos_mask, test_neg_mask, placeholders, test_path)
    roc_auc3, pr_auc3, res_tftest = save_results(adj, features, train_pos_logits, tftest_pos_logits, tftest_pos_mask, tftest_neg_mask, placeholders, tftest_path)
     
    res_valid.to_csv( f'{result_path}/valid_network.csv', header=True, index=False)
    res_test.to_csv( f'{result_path}/test_network.csv', header=True, index=False)
    res_tftest.to_csv( f'{result_path}/tftest_network.csv',header=True, index=False) 
    res = 'method\tdataset\tauroc\taupr\n'
    res += f'DeepRIG\tvalid\t{roc_auc1}\t{pr_auc1}\n'
    res += f'DeepRIG\ttest\t{roc_auc2}\t{pr_auc2}\n'
    res += f'DeepRIG\ttftest\t{roc_auc3}\t{pr_auc3}\n'

    with open(f'{result_path}/roc.txt', 'w') as f:
        f.write(res)
