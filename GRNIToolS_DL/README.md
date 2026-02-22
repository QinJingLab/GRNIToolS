# GRNIToolS_DL: Deep Learning for Gene Regulatory Network Inference

## Description

GRNIToolS_DL is a comprehensive toolkit for Gene Regulatory Network (GRN) inference using various deep learning approaches. This toolbox integrates multiple state-of-the-art algorithms to achieve efficient and accurate inference and analysis of gene regulatory networks from gene expression data.

## Overview

## Supported Models

| Method Name | Core Model | Key Feature  |
| :--- | :--- | :--- |
| **DeepSEM** | $\beta$-VAE | **Explicit Structure:** Encoder/decoder weights directly represent the adjacency matrix. |
| **DeepDRIM** | CNN + MLP | **Context Aware:** Uses "neighbor images" (adjacency matrices) to provide regulatory context. |
| **GRNCNN** | CNN | **Image Conversion:** Transforms expression vectors into concatenated images for CNN processing. |
| **GRNResNet** | ResNet | **Deep Architecture:** Uses ResNet18 on image-like matrices to prevent vanishing gradients. |
| **STGRNS** | Transformer | **Self-Attention:** Captures key features from TF-gene pairs via attention mechanisms. |
| **GRNTrans** | Transformer | **Sequence Modeling:** Encodes pairs as sequences with position encoding & multi-head attention. |
| **GENELink** | GAT + MLP | **Link Prediction:** GAT aggregates prior knowledge + scRNA-seq, followed by dual-channel MLPs. |
| **GRNGCN** | GCN | **Topology Encoding:** Two-layer GCN encoder captures topology; dot-product decoder predicts edges. |
| **IGEGRNS** | GraphSAGE + CNN | **Supervised Hybrid:** Combines GraphSAGE (neighbor aggregation) with a 3-layer CNN. |
| **DeepRIG** | GAE (GCN) | **Global Structure:** Graph Autoencoder embedding global structure from weighted co-expression networks. |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/QinJingLab/GRNIToolS.git
cd GRNIToolS_DL
```

2. Install required dependencies:
```bash
conda create -n grnitools python=3.10
pip install -r requirements.txt
```

3. For GPU support, ensure CUDA is properly installed and compatible with your PyTorch version.


## Usgae

### Quick Star
```bash
python main.py \
    --method GRNCNN \
    --expr ./Input_data/Ecoli/GRNDL/expression.csv \
    --train ./Input_data/Ecoli/GRNDL/train.txt \
    --valid ./Input_data/Ecoli/GRNDL/val.txt \
    --test ./Input_data/Ecoli/GRNDL/random_test.txt \
    --tftest ./Input_data/Ecoli/GRNDL/TF_test.txt \
    --output_dir results \
    --lr 0.0001 \
    --batch_size 64 \
    --epochs 100 
```

### Hyperparameter Optimization

Enable Optuna for automatic hyperparameter tuning:
```bash
python main.py \
    --method GRNCNN \
    --expr ./Input_data/Ecoli/GRNDL/expression.csv \
    --train ./Input_data/Ecoli/GRNDL/train.txt \
    --valid ./Input_data/Ecoli/GRNDL/val.txt \
    --test ./Input_data/Ecoli/GRNDL/random_test.txt \
    --tftest ./Input_data/Ecoli/GRNDL/tftest.txt \
    --output_dir results \
    --opt \
    --num_trials 30 \
    --njobs 2
```

## Parameters

### Data Parameters
- `--expr`: Gene expression matrix file path
- `--train`: Training data file path
- `--valid`: Validation data file path
- `--test`: Test data file path
- `--tftest`: TF test data file path
- `--tf`: TF list file path (for GENELINK)
- `--gene`: Gene list file path (for GENELINK)
- `--input_dir`: Input directory (for DeepRIG)
- `--grn`: GRN data file (for DeepSEM)

### Training Parameters
- `--batch_size`: Training batch size (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Initial learning rate (default: 0.0001)

### Optimization Parameters
- `--opt`: Enable Optuna hyperparameter optimization
- `--njobs`: Number of parallel jobs for Optuna (default: 2)
- `--num_trials`: Number of Optuna trials (default: 30)

### General Parameters
- `--output_dir`: Output directory path (default: GRN_output)

## Data Format

### Expression Data
The expression data should be in CSV format with genes as rows and samples/columns as columns.

### Network Data
The network data should be in text format with each line representing a TF-gene interaction pair.

### Train/Valid/Test Splits
The train, validation, and test files should contain TF-gene pairs in text format, one pair per line.




## Contact

* **Yongqiang Zhou**, School of Pharmaceutical Sciences (Shenzhen), Sun Yat-sen University. 

   **Email**: zhouyq67@mail2.sysu.edu.cn

* **Jing Qin**, School of Pharmaceutical Sciences (Shenzhen), Sun Yat-sen University. 
   
   **Email**: qinj29@mail.sysu.edu.cn
---