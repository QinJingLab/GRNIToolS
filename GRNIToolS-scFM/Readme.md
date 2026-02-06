# GRNIToolS_scFM

A benchmarking tool for Gene Regulatory Network (GRN) inference using single-cell foundation models.

## Overview

GRNIToolS_scFM is a comprehensive tool for inferring gene regulatory networks from single-cell data using single-cell foundation models. The tool supports multiple inference methods and provides network performance evaluation.


## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd GRNIToolS_scFM
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The tool operates in three modes: train, predict, and evaluate.

## Example Workflow

Here's a complete example of how to use GRNIToolS_scFM:

1. **Prepare your data**:
   - Expression data: CSV file with genes as rows and cells as columns
   - Training/validation/test data/TF_test: Tab-separated files with gene pairs and labels (1 for interaction, 0 for no interaction)

2. **Train a model**:
```bash
python main.py train \
    --llm GenePT \
    --method MLP \
    --output output/A549 \
    --train_path input_data/human/A549/train.txt \
    --valid_path input_data/human/A549/valid.txt \
    --test_path input_data/human/A549/test.txt \
    --tftest_path input_data/human/A549/TF_test.txt \
    --expr_path input_data/human/A549/A549_expression.csv \
    --species human \
    --lr 0.0001 \
    --epochs 50 \
    --batch_size 64
```

3. **Predict GRN**:
```bash
python main.py predict \
    --llm GenePT \
    --method MLP \
    --model_path output/A549/GenePT/MLP_state_dict_model.pth \
    --test_path input_data/human/A549/test.txt \
    --expr_path input_data/human/A549/A549_expression.csv \
    --species human \
    --output output/A549
```

4. **Evaluate the results**:
```bash
python main.py evaluate \
    --net output/A549/GenePT/GenePT_MLP_inference_grn.txt \
    --label input_data/human/A549/test.txt \
    --output output/A549/evaluation_GenePT_MLP/
```

## Command Line Arguments

### Training Mode Arguments

- `--llm`: Biological language model to use (default: GenePT)
- `--pt`: Path to pre-trained model embeddings
- `--method`: GRN inference method (default: cos)
- `--output`: Output directory (default: output/A549)
- `--lr`: Learning rate (default: 0.0001)
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--train_path`: Path to training data
- `--valid_path`: Path to validation data
- `--test_path`: Path to test data
- `--tftest_path`: Path to unseen transcription factor test data
- `--expr_path`: Path to expression data
- `--species`: Species of the input data (default: human)

### Prediction Mode Arguments

- `--llm`: Single-cell foundation model to use (default: GenePT)
- `--pt`: Path to pre-trained model 
- `--method`: GRN inference method (default: cos)
- `--model_path`: Path to trained model checkpoint
- `--test_path`: Path to test data
- `--train_path`: Path to training data (required for GCN method)
- `--expr_path`: Path to expression data
- `--species`: Species of the input data (default: human)
- `--output`: Output directory (default: output/A549)

### Evaluation Mode Arguments

- `--net`: Path to predicted network
- `--label`: Path to ground truth labels
- `--output`: Evaluation metrics output directory

## Supported Models

- GenePT
- CellPLM
- Gene2vec
- GeneCompass
- Geneformer
- iSEEEK
- scBERT
- scFoundation
- scGPT

## Supported Inference Methods

- cos: Cosine similarity
- MLP: Multi-layer Perceptron
- CNN: Convolutional Neural Network
- GCN: Graph Convolutional Network
- ResNet: Residual Network
- Transformer: Transformer architecture
- rf: Random Forest
- xgb: XGBoost
- logit: Logistic Regression

## Output Files

- Training mode: Model checkpoints and performance metrics
- Prediction mode: Predicted gene regulatory networks with confidence scores
- Evaluation mode: Performance metrics (AUROC, AUPR) and visualization plots
