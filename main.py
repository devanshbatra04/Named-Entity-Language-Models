AWDM_LSTM_PATH = "./awd-lstm-lm/"
import sys
import os
sys.path.insert(0, AWDM_LSTM_PATH)
import argparse
from data_utils import load_datasets, model_load, model_save
from awd_lstm.utils import batchify
from build_model import get_model, train_and_eval
import torch

DATA_WITH_TYPES = "./data_with_type"
DATA_WITHOUT_TYPES = "./data_without_type"
EVAL_BATCH_SIZE = 10
TEST_BATCH_SIZE = 1
BATCH_SIZE = 80

EMBEDDING_SIZE = 400
NUM_HIDDEN_UNITS_PER_LAYER = 1150
NUM_LAYERS = 3
MODEL_TYPE = 'LSTM'

LR = 30
OPTIMIZER = 'adam'
WDECAY = 1.2e-6
EPOCHS = 8000

parser = argparse.ArgumentParser(description='PyTorch Named Entity Language Model (re-implemented by Yash and Devansh)')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
args = parser.parse_args()
args.tied = True


if __name__ == "__main__":
    corpus_with_types, corpus_without_types = load_datasets(DATA_WITH_TYPES, DATA_WITHOUT_TYPES)

    model_with_type, criterion_model_with_type, params_model_with_type \
        = get_model(MODEL_TYPE, corpus_with_types, EMBEDDING_SIZE, NUM_HIDDEN_UNITS_PER_LAYER, NUM_LAYERS, args)
    model_without_type, criterion_model_without_type, params_model_without_type \
        = get_model(MODEL_TYPE, corpus_with_types, EMBEDDING_SIZE, NUM_HIDDEN_UNITS_PER_LAYER, NUM_LAYERS, args)

    optimizer_model_with_type = None
    optimizer_model_without_type = None
    # Ensure the optimizer is optimizing params, which includes both the model's weights as well as the criterion's
    # weight (i.e. Adaptive Softmax)
    if OPTIMIZER == 'sgd':
        optimizer_model_with_type = torch.optim.SGD(params_model_with_type, lr=LR, weight_decay=WDECAY)
        optimizer_model_without_type = torch.optim.SGD(params_model_without_type, lr=LR, weight_decay=WDECAY)
    if OPTIMIZER == 'adam':
        optimizer_model_with_type = torch.optim.Adam(params_model_with_type, lr=LR, weight_decay=WDECAY)
        optimizer_model_without_type = torch.optim.Adam(params_model_without_type, lr=LR, weight_decay=WDECAY)

    if not (os.path.exists('model_with_types.pt')):
        train_and_eval(model_with_type, MODEL_TYPE, corpus_with_types, optimizer_model_with_type,
                       criterion_model_with_type, params_model_with_type, EPOCHS, LR, args, 'model_with_types.pt')

    if not (os.path.exists('model_without_types.pt')):
        train_and_eval(model_without_type, MODEL_TYPE, corpus_without_types, optimizer_model_without_type,
                       criterion_model_without_type, params_model_without_type, EPOCHS, LR, args,
                       'model_without_types.pt')

