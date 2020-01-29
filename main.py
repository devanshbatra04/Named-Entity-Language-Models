import argparse
from data_utils import load_datasets
from awd_lstm.utils import batchify

DATA_WITH_TYPES = "./data_with_type"
DATA_WITHOUT_TYPES = "./data_without_type"
EVAL_BATCH_SIZE = 10
TEST_BATCH_SIZE = 1
BATCH_SIZE = 80

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
args = parser.parse_args()
args.tied = True


if __name__ == "__main__":
    corpus_with_types, corpus_without_types = load_datasets(DATA_WITH_TYPES, DATA_WITHOUT_TYPES)

    train_data_with_types = batchify(corpus_with_types.train, BATCH_SIZE, args)
    val_data_with_types = batchify(corpus_with_types.valid, BATCH_SIZE, args)
    test_data_with_types = batchify(corpus_with_types.test, BATCH_SIZE, args)

    train_data_without_types = batchify(corpus_without_types.train, BATCH_SIZE, args)
    val_data_without_types = batchify(corpus_without_types.valid, BATCH_SIZE, args)
    test_data_without_types = batchify(corpus_without_types.test, BATCH_SIZE, args)
