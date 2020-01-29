from awd_lstm.model import RNNModel
###############################################################################
# Build the models
###############################################################################

from awd_lstm.splitcross import SplitCrossEntropyLoss


def get_model(model_type, corpus, em_size, n_hid, n_layers,args):
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
    criterion = SplitCrossEntropyLoss(args.emsize, splits=splits, verbose=False)

    if args.cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    params = list(model.parameters()) + list(criterion.parameters())

    return model, criterion, params
