# GRNIToolS_DL
Gene Regulatory Network Tools Selector, DeepLearning part.
# Introduction
Deep learning algorithms have seen significant success in a variety of industries in recent years, with the application of deep learning in bioinformatics receiving increasing attention. In this work, the benchmarking of six supervised (DeepDRIM, STGRNS, GRNCNN, GRNResNet, GRNTrans, GENELink) and one unsupervised (DeepSEM) deep learning method were conducted.

Here, DeepSEM，DeepDRIM, STGRNS, GENELink were proposed by others, but we made modifications to adapt to the work of the paper. Meanwhile, we have also constructed three models (GRNCNN, GRNTrans, GRNResNet) ourselves, although their performance may not be as good as other models, they can be benchmark tested.

# Environment bulid
### Dependency packages
All other methods can run under the unified framework of pytorch.The software can be used normally in the following environments:
```
python==3.10.13
matplotlib==3.8.2
numpy==1.24.1
pandas==2.1.4
scanpy==1.10.2
scikit-learn==1.3.2
tensorboard==2.16.2
torch==2.1.2+cu118
torch-geometric==2.5.3
tqdm==4.66.1
scipy==1.9.2
tensorflow=2.10.0
```
### Docker image build
Other packages can be easily installed through pip install and conda install. We have also built an image on Docker, which allows us to directly configure the environment through Docker.
```
docker pull pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
docker run -it --gpus all --name GRNIToolS_DL -v "dir you work":/workspace pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel /bin/bash
##docker exec -it GRNIToolS_DL /bin/bash 
pip install pandas
pip install scipy==1.9.2
pip install scikit-learn
pip install tensorboard
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/
torch-2.1.0+cu118.html
pip install matplotlib
pip install scanpy
pip install tensorfolw==2.10.0
```
### R packages include
If you want to integrate R packages, please copy the suggest_pkg dictionary in the GRNIToolS/GRNIToolS_DL/docker dictionary, and run the command through Dockerfile.
`cp -r ../docker/suggest_pkg docker/suggest_pkg`

`cd docker`

`docker build -t grnitools_dl .`

# Usage
We integrated 6 methods through GRNIToolS_DL.py and use the --method parameter to select the model to use. We can also adjust device, learning rate and batch size in the parameters. More details were written in the .py file. 

It is worth noting that some methods have specific hyperparameters that we have not adjusted. You can enter the folder of each method to view the usage methods.

`python GRNIToolS_DL -h`

```  
  -h, --help            show this help message and exit
  --epochs EPOCHS       Number of Epochs for training
  --method METHOD       Determine which task to run. Select from DeepSEM, GENELINK,GRNDL,STGRNS,DeepDRIM.DeepDRIM should run under tensorflow environment
  --batch_size BATCH_SIZE
                        The batch size used in the training process.
  --expr_file EXPR_FILE
                        The input scRNA-seq gene expression file.
  --cuda CUDA           The device used in model
  --lr LR               The learning rate used in model
  --tf_path TF_PATH     The path that includes TF name
  --gene_path GENE_PATH
                        The path that includes gene name
  --network_path NETWORK_PATH
                        The network indicates that releationship in TFs and genes,use for training
  --val_network_path VAL_NETWORK_PATH
                        The network indicates that releationship in TFs and genes,use for validation
  --test_network_path TEST_NETWORK_PATH
                        The network indicates that releationship in TFs and genes,use for test
  --TF_network_path TF_NETWORK_PATH
                        The network indicates that releationship in TFs and genes,use for TF_test
  --output_path OUTPUT_PATH
                        output dictionary 
```

For example, you can train and test on E.coli dataset by following command:
```
python GRNIToolS_DL --method STGRNS --epoch 200 --batch_size 64 --cuda 0 --lr 0.0001
--expr_file example/E.coli/Deeplearning_data/expression.csv
--tf_path example/E.coli/TFlist.txt
--gene_path example/E.coli/genelist.txt
--network_path example/E.coli/Deeplearning_data/train_val_test/train.txt
--val_network_path example/E.coli/Deeplearning_data/train_val_test/val.txt
--test_network_path example/E.coli/Deeplearning_data/train_val_test/random_test.txt
--TF_network_path example/E.coli/Deeplearning_data/train_val_test/TF_test.txt
--output_path output/test/
```
