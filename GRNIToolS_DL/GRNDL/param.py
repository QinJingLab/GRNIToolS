import torch

class Config_CNN(object):
    #"""配置参数""" 
    def __init__(self):
        self.model_name = 'CNN'
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.1                                              # 随机失活
        self.num_classes = 2                                            # 类别数
        self.epochs = 100                                                # epoch数
        self.batch_size = 64                                            # mini-batch大小
        self.learning_rate = 0.001                                      # 学习率
        self.nworkers = 2
        self.dim_model = None
        self.num_head = None
        self.hidden = None
        self.input_size = 60

class Config_STGRNS(object):
    # # """配置参数""" #  
    def __init__(self):
        self.model_name = 'STGRNS'
        self.embedding_pretrained = None                                # 预训练词向量
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.1                                              # 随机失活
        self.num_classes = 2                                            # 类别数
        self.epochs = 100                                              # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.learning_rate = 0.0001                                    # 学习率
        self.embed = 200                                              # 词向量维度
        self.dim_model = 200
        self.hidden = 200
        self.last_hidden = 200
        self.num_head = 8
        self.num_encoder = 6
        self.nworkers = 2
        self.input_size = 200


class Config_Trans(object):
    # """配置参数""" #  
    def __init__(self):
        self.model_name = 'Transformer'
        self.embedding_pretrained = None                                # 预训练词向量
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.1                                            # 随机失活
        self.num_classes = 2                                            # 类别数
        self.epochs = 100                                              # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.learning_rate = 0.0001                                    # 学习率
        self.embed = 128                                            # 词向量维度
        self.dim_model = 128
        self.hidden = 128
        self.last_hidden = 128
        self.num_head = 2
        self.num_encoder = 6
        self.nworkers = 2
        self.input_size = 60
        self.batch_first=True

class Config_GCN(object):
    def __init__(self):
        self.model_name = 'GRNGCN'
        self.embedding_pretrained = None                                # 预训练词向量
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.1                                              # 随机失活
        self.num_classes = 2                                            # 类别数
        self.epochs = 300                                               # epoch数
        self.batch_size = 64                                           # mini-batch大小
        self.learning_rate = 0.001                                   # 学习率
        self.embed = 256                                               # 词向量维度
        self.dim_model = 200
        self.hidden = 128
        self.last_hidden = 512
        self.num_head = 8
        self.num_encoder = 6
        self.nworkers = 2
        self.input_size = 60

class Config_GENELINK(object):
    def __init__(self):
        self.model_name = 'GENELINK'
        self.embedding_pretrained = None                                # 预训练词向量
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # 设备
        self.hidden1_dim=128
        self.hidden2_dim=64
        self.hidden3_dim=32
        self.output_dim=16
        self.num_head1=3
        self.num_head2=3
        self.alpha=0.2
        self.reduction='concate'