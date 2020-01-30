import torch
import awd_lstm.data as data
import os
import hashlib

###############################################################################
# Load data
###############################################################################


def model_save(fn, model, criterion, optimizer):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer], f)


def model_load(fn):
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)
        return model, criterion, optimizer


def load_datasets(dataset_with_types, dataset_without_types):
    encoded_data_with_type_path = 'corpus.{}.data'.format(hashlib.md5(dataset_with_types.encode()).hexdigest())
    if os.path.exists(encoded_data_with_type_path):
        print('Loading cached dataset of raw data with types...')
        corpus_with_types = torch.load(encoded_data_with_type_path)
    else:
        print('Producing dataset of raw data with types...')
        corpus_with_types = data.Corpus(dataset_with_types)
        torch.save(corpus_with_types, encoded_data_with_type_path)

    encoded_data_without_type_path = 'corpus.{}.data'.format(hashlib.md5(dataset_without_types.encode()).hexdigest())
    if os.path.exists(encoded_data_without_type_path):
        print('Loading cached dataset of raw data without types...')
        corpus_without_types = torch.load(encoded_data_without_type_path)
    else:
        print('Producing dataset of raw data without types...')
        corpus_without_types = data.Corpus(dataset_without_types)
        torch.save(corpus_without_types, encoded_data_without_type_path)

    return corpus_with_types, corpus_without_types


