AWDM_LSTM_PATH = "./awd-lstm-lm/"
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
sys.path.insert(0, AWDM_LSTM_PATH)
from awd_lstm.model import RNNModel
from awd_lstm.utils import get_batch, repackage_hidden, batchify
from data_utils import model_load, model_save

###############################################################################
# Build the models
###############################################################################

from awd_lstm.splitcross import SplitCrossEntropyLoss


def get_model(model_type, corpus, em_size, n_hid, n_layers, args):
    n_tokens = len(corpus.dictionary)
    model = RNNModel(model_type, n_tokens, em_size, n_hid, n_layers, args.dropout, args.dropouth, args.dropouti,
                     args.dropoute, args.wdrop, args.tied)
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


def evaluate(model, criterion, model_type, args, data_source, batch_size=10):
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


def train(model, model_type, corpus, optimizer, criterion, params, epoch, args, batch_size=80, alpha=2, beta=1,
          clip=0.25, log_interval=200):
    # Turn on training mode which enables dropout.
    if model_type == 'QRNN': model.reset()
    total_loss = 0
    start_time = time.time()
    train_data = batchify(corpus.train, batch_size, args)
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


def train_and_eval(model, model_type, corpus, optimizer, criterion, params, epochs, lr, args, save_path, eval_batch_size=10,
                   test_batch_size=1):
    val_data = batchify(corpus.valid, eval_batch_size, args)
    test_data = batchify(corpus.test, test_batch_size, args)

    best_val_loss = []
    stored_loss = 100000000

    try:
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            train(model, model_type, corpus, optimizer, criterion, params, epoch, args)
            if 't0' in optimizer.param_groups[0]:
                tmp = {}
                for prm in model.parameters():
                    tmp[prm] = prm.data.clone()
                    prm.data = optimizer.state[prm]['ax'].clone()

                val_loss2 = evaluate(model, criterion, model_type, args, val_data)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss2, math.exp(val_loss2), val_loss2 / math.log(2)))
                print('-' * 89)

                if val_loss2 < stored_loss:
                    model_save(save_path)
                    print('Saving Averaged!')
                    stored_loss = val_loss2

                for prm in model.parameters():
                    prm.data = tmp[prm].clone()

            else:
                val_loss = evaluate(model, criterion, model_type, args, val_data, eval_batch_size)
                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                      'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
                    epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss), val_loss / math.log(2)))
                print('-' * 89)

                if val_loss < stored_loss:
                    model_save(save_path)
                    print('Saving model (new best validation)')
                    stored_loss = val_loss

                # if args.optimizer == 'sgd' and 't0' not in optimizer.param_groups[0] and (
                #         len(best_val_loss) > args.nonmono and val_loss > min(best_val_loss[:-args.nonmono])):
                #     print('Switching to ASGD')
                #     optimizer = torch.optim.ASGD(model.parameters(), lr=lr, t0=0, lambd=0.,
                #                                  weight_decay=args.wdecay)

                if epoch in args.when:
                    print('Saving model before learning rate decreased')
                    model_save('{}.e{}'.format(save_path, epoch))
                    print('Dividing learning rate by 10')
                    optimizer.param_groups[0]['lr'] /= 10.

                best_val_loss.append(val_loss)

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    model_load(args.save)

    # Run on test data.
    test_loss = evaluate(model, criterion, model_type, args, test_data, test_batch_size)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
        test_loss, math.exp(test_loss), test_loss / math.log(2)))
    print('=' * 89)

def get_symbol_table(data, types):
    id_map ={}
    i = 0
    for pos, tp in zip(data, types):
        id_map.update({pos.data[0]:tp.data[0]})
    return id_map


def evaluate_both(type_model, entity_composite_model, data_source_without_type, data_source_type, corpus_without_types, corpus_with_types, args, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    type_model.eval()
    entity_composite_model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_loss2 = 0
    total_loss_cb = 0

    n_tokens_in_typed_corpus = len(corpus_with_types.dictionary)
    n_tokens_in_untyped_corpus = len(corpus_without_types.dictionary)
    hidden_state_entity_composite_model = entity_composite_model.init_hidden(batch_size)
    hidden_state_type_model = type_model.init_hidden(batch_size)
    # mcq_ids = [corpus_without_types.dictionary.word2idx[w] for w in mcq_wrd]

    for batch, i in enumerate(range(0, data_source_without_type.size(0) - 1, args.bptt)):
        data_without_type, targets_without_type = get_batch(data_source_without_type, i, args, evaluation=True)
        data_with_type, targets_with_type = get_batch(data_source_type, i, args, evaluation=True)

        data_type, targets_type = get_batch(data_source_type, i, args, evaluation=True)

        if batch_size == 1:
            hidden_state_entity_composite_model = entity_composite_model.init_hidden(batch_size)
            hidden_state_type_model = type_model.init_hidden(batch_size)

        #TODO Correct following type_model and entity_composite_model inference
        output_type_model, hidden_state_type_model = type_model(data_with_type, hidden_state_type_model)
        output_entity_composite_model, hidden_state_entity_composite_model = entity_composite_model(data_without_type, data_type, hidden_state_entity_composite_model)

        output_type_model_flat = output_type_model.view(-1, n_tokens_in_typed_corpus)
        output_entity_composite_model_flat = output_entity_composite_model.view(-1, n_tokens_in_untyped_corpus)

        candidates_ids = set([i.data[0] for i in targets_without_type])

        numwords = output_entity_composite_model_flat.size()[0]
        symbol_table = get_symbol_table(targets_without_type, targets_with_type)

        output_flat_combined = output_entity_composite_model_flat.clone()
        for idxx in range(numwords):
            for pos in candidates_ids:  # for all candidates

                tp = symbol_table[pos]
                var_prob = output_flat_combined.data[idxx][pos]
                type_prob = output_type_model_flat.data[idxx][tp]
                new_prob1 = 2 * var_prob  # just to scale values, empirical

                if corpus_without_types.dictionary.idx2word[pos] != corpus_with_types.dictionary.idx2word[tp]:
                    new_prob1 = (var_prob + type_prob)  # / 2
                output_flat_combined.data[idxx][pos] = new_prob1

        total_loss += len(data_without_type) * criterion(output_entity_composite_model_flat, targets_without_type).data
        total_loss2 += len(data_with_type) * criterion(output_type_model_flat, targets_with_type).data
        total_loss_cb += len(data_without_type) * criterion(output_flat_combined, targets_without_type).data


        # print (' soccer: ', len(data) * criterion(output_flat, targets).data), ' my: ',  len(data) * criterion(output_flat_cb, targets).data
        if batch % 500 == 0:
            # print(' only ingred not avg')
            # print ("done batch ", batch, ' of ', len(data_source)/ eval_batch_size)
            test_loss_cb = total_loss_cb[0] / len(data_source_without_type)
            test_loss = total_loss[0] / len(data_source_without_type)
            test_loss2 = total_loss2[0] / len(data_source_without_type)
            p = (100 * batch) / (33000)
            print('=' * 160)
            print(
                '| after: {:5.2f}% | test var loss {:5.2f} | test var ppl {:8.2f} | test type loss {:5.2f} | test type ppl {:8.2f} | test cb loss {:5.2f} | test cb ppl {:8.2f}'.format(
                    p, test_loss, math.exp(test_loss), test_loss2, math.exp(test_loss2), test_loss_cb,
                    math.exp(test_loss_cb)))
            print('=' * 160)

        hidden_state_entity_composite_model = repackage_hidden(hidden_state_entity_composite_model)
        hidden_state_type_model = repackage_hidden(hidden_state_type_model)

    return total_loss[0] / len(data_source_without_type), total_loss2[0] / len(data_source_type), total_loss_cb[0] / len(data_source_without_type)

