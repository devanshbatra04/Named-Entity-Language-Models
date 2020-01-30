AWDM_LSTM_PATH = "./awd-lstm-lm/"
import sys
sys.path.insert(0, AWDM_LSTM_PATH)
from awd_lstm.model import RNNModel
from awd_lstm.utils import get_batch, repackage_hidden

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
