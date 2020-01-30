AWDM_LSTM_PATH = "./awd-lstm-lm/"
import sys
import time
import math
import numpy as np
import torch
sys.path.insert(0, AWDM_LSTM_PATH)
from awd_lstm.model import RNNModel
from awd_lstm.utils import get_batch, repackage_hidden, batchify

###############################################################################
# Build the models
###############################################################################

from awd_lstm.splitcross import SplitCrossEntropyLoss


def get_model(model_type, corpus, em_size, n_hid, n_layers, args):
    n_tokens = len(corpus.dictionary)
    model = RNNModel(model_type, n_tokens, em_size, n_hid, n_layers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)
    ###

    splits = []
    if n_tokens > 500000:
        # One Billion
        # This produces fairly even matrix mults for the buckets:
        # 0: 11723136, 1: 10854630, 2: 11270961, 3: 11219422
        splits = [4200, 35000, 180000]
    elif n_tokens > 75000:
        # WikiText-103
        splits = [2800, 20000, 76000]
    print('Using', splits)
    criterion = SplitCrossEntropyLoss(em_size, splits=splits, verbose=False)

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    params = list(model.parameters()) + list(criterion.parameters())

    return model, criterion, params


def evaluate(model, criterion, model_type, data_source, args, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if model_type == 'QRNN':
        model.reset()
    total_loss = 0
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


def train(model, model_type, corpus, optimizer, criterion, params, epoch, args, batch_size=80, alpha=2, beta=1, clip =0.25, log_interval=200):
    # Turn on training mode which enables dropout.
    if model_type == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    train_data = batchify(corpus.train, args.batch_size, args)
    hidden = model.init_hidden(batch_size)
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        # seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        optimizer.zero_grad()

        output, hidden, rnn_hs, dropped_rnn_hs = model(data, hidden, return_h=True)
        raw_loss = criterion(model.decoder.weight, model.decoder.bias, output, targets)

        loss = raw_loss
        # Activiation Regularization
        if alpha: loss = loss + sum(alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
        # Temporal Activation Regularization (slowness)
        if beta: loss = loss + sum(beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if clip: torch.nn.utils.clip_grad_norm_(params, clip)
        optimizer.step()

        total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len
