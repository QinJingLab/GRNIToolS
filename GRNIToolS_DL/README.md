# GRNIToolS_DL: Deep Learning for Gene Regulatory Network Inference

## Description

GRNIToolS_DL is a comprehensive toolkit for Gene Regulatory Network (GRN) inference using various deep learning approaches. This toolbox integrates multiple state-of-the-art algorithms to achieve efficient and accurate inference and analysis of gene regulatory networks from gene expression data.

## Overview

Every method writes a scored edge list in the same schema — `predictions_<split>.tsv`, or
`predictions_all.tsv` for DeepSEM — and is scored by one evaluator, `benchmark/evaluate.py`,
so their numbers share a definition: AUROC and AUPRC over the pairs a method managed to score.

Two caveats on comparing across methods. **DeepSEM is unsupervised** — it never sees the
labels — while every other method here is trained on the labelled training split, so its
row belongs to a different task and is not comparable to the rest. And several methods
cannot score every pair they are handed, so an AUROC over a method-chosen subset is not
comparable to one over the whole split.

## Supported Models

| Method Name | Core Model | Key Feature  |
| :--- | :--- | :--- |
| **DeepSEM** | $\beta$-VAE | **Explicit Structure:** Encoder/decoder weights directly represent the adjacency matrix. |
| **DeepDRIM** | CNN + MLP | **Context Aware:** Uses "neighbor images" (adjacency matrices) to provide regulatory context. |
| **GRNCNN** | CNN | **Image Conversion:** Transforms expression vectors into concatenated images for CNN processing. |
| **GRNResNet** | ResNet | **Deep Architecture:** Uses ResNet18 on image-like matrices to prevent vanishing gradients. |
| **STGRNS** | Transformer | **Self-Attention:** Captures key features from TF-gene pairs via attention mechanisms. |
| **GRNTrans** | Transformer | **Sequence Modeling:** Encodes pairs as sequences with position encoding & multi-head attention. |
| **GENELINK** | GAT + MLP | **Link Prediction:** GAT aggregates prior knowledge + scRNA-seq, followed by dual-channel MLPs. |
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
conda activate grnitools
pip install -r requirements.txt
```

3. For GPU support, ensure CUDA is properly installed and compatible with your PyTorch version.


## Usage

### Quick Start
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
    --batch_size 64
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
    --tftest ./Input_data/Ecoli/GRNDL/TF_test.txt \
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
- `--tf`: TF list file path (GENELINK and DeepSEM)
- `--gene`: Gene list file path (GENELINK, DeepDRIM and DeepSEM)
- `--input_dir`: Input directory (DeepRIG). DeepRIG reads none of `--expr`, `--train`,
  `--valid`, `--test` or `--tftest`; this directory is its entire data interface
- `--grn`: GRN data file (DeepSEM). Defaults to `./Input_data/Ecoli/GRNDL/grn.txt` and is



### Training Parameters
- `--batch_size`: Training batch size (default: 64)
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Initial learning rate (default: 0.0001)

### Optimization Parameters
- `--opt`: Enable Optuna hyperparameter optimization
- `--njobs`: Number of parallel jobs for Optuna (default: 2)
- `--num_trials`: Number of Optuna trials (default: 30)

### General Parameters
- `--method`: Method to run (default: GRNCNN). One of the names in the table above; `GRNCNN` and `GRNResNet` both dispatch to `source/GRNDL`
- `--hidden`: Number of hidden units (GRNGCN, default: 128)
- `--network`: GRNCNN and GRNResNet only. When 1, reload the best checkpoint once training ends
  and write `test_network_predict.txt` — the test pairs with a predicted score appended
  (default: 0)
- `--head`: Number of attention heads (GRNTrans, default: 4)
- `--output_dir`: Output directory path (default: GRN_output)
- `--cuda_devices`: CUDA_VISIBLE_DEVICES, comma-separated for multi-GPU (default: 0). It is
  exported to the environment, so every method except GRNGCN inherits it; the six methods whose
  CLI takes a flag (GRNCNN, GRNResNet, GRNTrans, GRNGCN, GENELINK, STGRNS) also get it on the
  command line, and DeepDRIM is given `--cuda`. GRNGCN is excluded from the export because it
  indexes the flag's value into the full device list, so the variable has to stay unset
- `--cleanup_trials`: After an `--opt` run, delete every `trial_*` directory except the best
  one. Off by default — those directories record what the search tried. Before anything is
  removed, each trial's `roc.txt` is collected into `trials_summary.tsv`

### Evaluating the output

A method that emits a scored edge list writes `predictions_<split>.tsv` beside its `roc.txt`,
under `--output_dir/<method>/`: tab-separated, with the header `tf`, `gene`, `score` and one
row per scored pair. Higher scores mean more likely an edge, and any monotone scale works.
There is deliberately no label column — the labels come from the split file, which
`benchmark/evaluate.py` reads itself. DeepSEM is the exception — it writes
`predictions_all.tsv`, with no split suffix.

Once the run ends, `main.py` scores the `predictions_<split>.tsv` dumps with
`benchmark/evaluate.py` and writes `roc.txt` from them, so the file and the evaluator agree.
DeepSEM keeps its own one-row `roc.txt`, from the R package's protocol.

```bash
python main.py --method GRNResNet --output_dir results \
    --expr ./Input_data/Ecoli/GRNDL/expression.csv \
    --train ./Input_data/Ecoli/GRNDL/train.txt \
    --valid ./Input_data/Ecoli/GRNDL/val.txt \
    --test ./Input_data/Ecoli/GRNDL/random_test.txt \
    --tftest ./Input_data/Ecoli/GRNDL/TF_test.txt

# one split, one line
python benchmark/evaluate.py results/GRNResNet/predictions_test.tsv \
    ./Input_data/Ecoli/GRNDL/random_test.txt

# or the whole cell at once: a directory of dumps against a directory of splits
python benchmark/evaluate.py results/GRNResNet ./Input_data/Ecoli/GRNDL
```

The second form prints one line per split:

```
valid    AUROC=0.64124  AUPRC=0.67371
test     AUROC=0.66613  AUPRC=0.69218
tftest   AUROC=0.57974  AUPRC=0.62712
```

## Data Format

### Expression Data
The expression data should be in CSV format with genes as rows and samples as columns.

### Network Data
The network data should be in text format with each line representing a TF-gene interaction pair.

### Train/Valid/Test Splits
The train, validation, and test files should contain one TF-gene pair per line, as three
tab-separated columns: TF, target, and the label (1 for an edge, 0 otherwise).




## Contact

* **Yongqiang Zhou**, School of Pharmaceutical Sciences (Shenzhen), Sun Yat-sen University. 

   **Email**: zhouyq67@mail2.sysu.edu.cn

* **Jing Qin**, School of Pharmaceutical Sciences (Shenzhen), Sun Yat-sen University. 
   
   **Email**: qinj29@mail.sysu.edu.cn
---
