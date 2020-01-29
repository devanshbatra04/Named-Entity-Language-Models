import torch
import awd_lstm.data as data
import os
import hashlib
from copy import deepcopy

###############################################################################
# Load data
###############################################################################

def model_save(fn):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)


def load_datasets(dataset_without_types, path_to_type_list):
    encoded_data_without_type_path = 'corpus.{}.data'.format(hashlib.md5(dataset_without_types.encode()).hexdigest())
    if os.path.exists(encoded_data_without_type_path):
        print('Loading cached dataset of raw data without types...')
        corpus_without_types = torch.load(encoded_data_without_type_path)
    else:
        print('Producing dataset of raw data without types...')
        corpus_without_types = data.Corpus(dataset_without_types)
        torch.save(corpus_without_types, encoded_data_without_type_path)

    corpus_with_types = deepcopy(corpus_without_types)
    corpus_with_types.tokenize(path_to_type_list)

    return corpus_with_types, corpus_without_types


