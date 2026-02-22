import argparse
import os
import sys
import time
from modules.trainer import run_biollm
from modules.predictor import predict_biollm
from modules.evaluator import evaluate
import warnings
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")


def get_parser():
    parser = argparse.ArgumentParser(description='A Benchmarking Tool for Gene Regulatory Network Inference')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')
    
    train_parser = subparsers.add_parser('train', help='tranning mode')
    train_parser.add_argument('-l', '--llm', required = False, type = str, default = 'GenePT', help = 'choose biological large language models')
    train_parser.add_argument('--pt', required = False, type = str, default = './model_file/GenePT/GenePT_gene_embedding_ada_text.pickle', help = 'input model path')
    train_parser.add_argument('-o', '--output', required = False, type = str, default = 'output/A549', help = 'output file directory')
    train_parser.add_argument('-m', '--method', required = False, type = str, default = 'cos', help = 'choose method for grn inference')
    train_parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of epoch')
    train_parser.add_argument('--batch_size', type=int, default=64, help='The size of each batch')
    train_parser.add_argument('-train', '--train_path', default='input_data/human/A549/train.txt')
    train_parser.add_argument('-valid', '--valid_path', default='input_data/human/A549/val.txt')
    train_parser.add_argument('-test', '--test_path', required=False,default='input_data/human/A549/test.txt')
    train_parser.add_argument('-tftest', '--tftest_path', required=False,default='input_data/human/A549/TF_test.txt')
    train_parser.add_argument('-expr', '--expr_path', required=False, default='input_data/human/A549/A549_expression.csv')
    train_parser.add_argument('-sp', '--species', required= False, default='human', help="The species of the input data")
    
    predict_parser = subparsers.add_parser('predict', help='predict mode')
    predict_parser.add_argument('-l', '--llm', required = False, type = str, default = 'GenePT', help = 'choose biological large language models')
    predict_parser.add_argument('--pt', required = False, type = str, default = './model_file/GenePT/GenePT_gene_embedding_ada_text.pickle', help = 'input model path')
    predict_parser.add_argument('-m', '--method', required = False, type = str, default = 'cos', help = 'choose method for grn inference')
    predict_parser.add_argument('--model_path', required=False, help='checkpoint path')
    predict_parser.add_argument('-test', '--test_path', required=False,default=None)
    predict_parser.add_argument('-train', '--train_path', required=False,default=None, help='Path to training data need for the method of GCN')
    predict_parser.add_argument('-expr', '--expr_path', required=False, default='input_data/human/A549/A549_expression.csv')
    predict_parser.add_argument('-sp', '--species', required= False, default='human', help="The species of the input data")
    predict_parser.add_argument('-o', '--output', required = False, type = str, default = 'output/A549', help = 'output file directory')

    eval_parser = subparsers.add_parser('evaluate', help='evaluation mode')
    eval_parser.add_argument('--net', required=True, type=str, help='Path to the predicted network')
    eval_parser.add_argument('--label', type=str, default='input_data/human/A549/test.txt', help='Path to ground truth labels')
    eval_parser.add_argument('-o', '--output', type=str, default='output/A549', help='Evaluation metrics output directory')

    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    if args.command == 'train':
        llm, pt, method = args.llm, args.pt, args.method
        expr_path = args.expr_path
        train_path, valid_path, test_path, tftest_path = args.train_path, args.valid_path, args.test_path, args.tftest_path
        output, batch_size = args.output, args.batch_size
        epochs, lr, species = args.epochs, args.lr, args.species
        run_biollm(llm, pt, method, expr_path, train_path, valid_path, test_path, tftest_path, output, batch_size, epochs, lr, species)
        
    elif args.command == 'predict':
        llm = args.llm
        pt = args.pt
        method = args.method
        model_path = args.model_path
        test_path = args.test_path
        expr_path = args.expr_path
        species = args.species
        output = args.output
        train_path = args.train_path
        predict_biollm(llm, pt, method, model_path, expr_path, test_path, species, output, train_path)

    elif args.command == 'evaluate':
        net = args.net
        label = args.label
        output = args.output
        evaluate(net, label, output)

if __name__ == "__main__":
    main()