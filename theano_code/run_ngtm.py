import os
import sys
import json
import codecs
import datetime
from optparse import OptionParser
import numpy as np
from sklearn.preprocessing import normalize
import ngtm
import common_theano
import file_handling as fh


def main():
    usage = "%prog input_dir train_prefix output_prefix"
    parser = OptionParser(usage=usage)
    parser.add_option('--de', dest='de', default=500,
                      help='Size of MLP hidden layer(s) or embeddings: default=%default')
    parser.add_option('--n_topics', dest='n_topics', default=50,
                      help='Number of topics: default=%default')
    parser.add_option('--n_classes', dest='n_classes', default=1,
                      help='Number of classes (labels): default=%default')
    parser.add_option('--encoder_layers', dest='encoder_layers', default=1,
                      help='Number of layers in encoder (1 or 2): default=%default')
    parser.add_option('--encoder_shortcut', action="store_true", dest="encoder_shortcut", default=False,
                      help='Use direct connection in the encoder: default=%default')
    parser.add_option('--generator_layers', dest='generator_layers', default=1,
                      help='Number of layers in generator (0, 1 or 2): default=%default')
    parser.add_option('--generator_shortcut', action="store_true", dest="generator_shortcut", default=False,
                      help='Use direct connection in the generator: default=%default')
    parser.add_option('--transform', dest='transform', default=None,
                      help='Transformation before output layer [None|tanh|relu|softmax]: default=%default')
    parser.add_option('--use_interactions', action="store_true", dest="use_interactions", default=False,
                      help='Use interactions: default=%default')
    parser.add_option('--no_bias', action="store_true", dest="no_bias", default=False,
                      help="Don't use a bias term in output layer: default=%default")
    parser.add_option('--encode_labels', action="store_true", dest="encode_labels", default=False,
                      help='Feed labels (if present) into encoder: default=%default')
    parser.add_option('--l1_penalty', dest='l1_penalty', default=0.0,
                      help='L1 penalty on sparse deviations: default=%default')
    parser.add_option('--l1_inter_factor', dest='l1_inter_factor', default=1.0,
                      help='Factor by which to multiply penalty on interactions: default=%default')
    parser.add_option('--init_scale', dest='init_scale', default=6.0,
                      help='Scaling to initialize model parameters: default=%default')
    parser.add_option('--optimizer', dest='optimizer', default='adagrad',
                      help='Optimizer [sgd|sgdm|adagrad]: default=%default')
    parser.add_option('--lr', dest='lr', default=0.05,
                      help='Learning rate: default=%default')
    parser.add_option('--clip', action="store_true", dest="clip_gradients", default=False,
                      help='Clip gradients: default=%default')
    parser.add_option('--train_bias', action="store_true", dest="train_bias", default=False,
                      help='Update the output bias during training: default=%default')
    parser.add_option('--min_epochs', dest='min_epochs', default=50,
                      help='Minimum number of epochs: default=%default')
    parser.add_option('--max_epochs', dest='max_epochs', default=200,
                      help='Maximum number of epochs: default=%default')
    parser.add_option('--patience', dest='patience', default=10,
                      help='Stop if no improvement after this many epochs: default=%default')
    parser.add_option('--n_dev', dest='n_dev', default=500,
                      help='Number of random documents to use for a validation set: default=%default')
    parser.add_option('--sort', action="store_true", dest="sort", default=False,
                      help='Sort on first epoch: default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random seed (must be > 0): default=%default')
    parser.add_option('--sparsity_target', dest='sparsity_target', default=None,
                      help='Adapt l1 penalty to try to achieve this sparsity (requires l1_penalty > 0): default=%default')
    parser.add_option('--train_percent', dest='train_percent', default=1.0,
                      help='Proportion of training data to use (random if < 1.0): default=%default')
    parser.add_option('--time_penalty', action="store_true", dest="time_penalty", default=False,
                      help='Use a temporal difference penalty: default=%default')
    parser.add_option('--test', dest='test_prefix', default=None,
                      help='Test prefix for evaluation: default=%default')

    options, args = parser.parse_args()

    input_dir = args[0]
    input_prefix = args[1]
    output_prefix = args[2]

    de = int(options.de)
    n_topics = int(options.n_topics)
    n_classes = int(options.n_classes)
    encoder_layers = int(options.encoder_layers)
    encoder_shortcut = options.encoder_shortcut
    generator_layers = int(options.generator_layers)
    generator_shortcut = options.generator_shortcut
    transform = options.transform
    use_interactions = options.use_interactions
    no_bias = options.no_bias
    encode_labels = options.encode_labels
    l1_strength = np.array(options.l1_penalty, dtype=np.float32)
    l1_inter_factor = float(options.l1_inter_factor)
    init_scale = float(options.init_scale)
    optimizer_name = options.optimizer
    learning_rate = np.array(options.lr, dtype=np.float32)
    clip_gradients = options.clip_gradients
    train_bias = options.train_bias
    min_epochs = int(options.min_epochs)
    max_epochs = int(options.max_epochs)
    patience = int(options.patience)
    n_dev = int(options.n_dev)
    sort_on_first_epoch = options.sort
    alternate = False
    train_percent = float(options.train_percent)
    sparsity_target = options.sparsity_target
    if sparsity_target is not None:
        sparsity_target = float(sparsity_target)
    time_penalty = options.time_penalty
    test_prefix = options.test_prefix
    seed = options.seed
    if seed is None:
        seed = np.random.randint(0, int(1e8))
    else:
        seed = int(seed)

    if n_topics < 1:
        n_topics = 1
    if n_classes < 1:
        n_classes = 1

    model_file = os.path.join(input_dir, output_prefix + '.npz')
    log_file = os.path.join(input_dir, output_prefix + '.log')
    pred_file = os.path.join(input_dir, output_prefix + '.pred.npz')
    with codecs.open(log_file, 'w', encoding='utf-8') as f:
        # note start time in log file
        f.write('Creating log file at %s\n' % datetime.datetime.now())
        # save command line options
        f.write(json.dumps(vars(options)) + '\n')

    # load data
    train_X, vocab, index_arrays, train_labels = load_data(input_dir, input_prefix, log_file)
    if test_prefix is not None:
        test_X, _, test_index_arrays, test_labels = load_data(input_dir, test_prefix, log_file, vocab)
    else:
        test_X = None
        test_index_arrays = None
        test_labels = None
    log_word_freq = load_background_freq(input_dir, input_prefix, vocab)
    n_items, dv = train_X.shape

    if train_percent < 1.0:
        random_subset = np.random.choice(np.arange(n_items), int(n_items * train_percent), replace=False)
        log(log_file, "Randomly sampling %d items" % len(random_subset))
        train_X = train_X[random_subset, :]
        index_arrays = [index_arrays[i] for i in random_subset]
        train_labels = train_labels[random_subset, :]

    if no_bias:
        init_bias = np.zeros(len(vocab))
    else:
        init_bias = log_word_freq

    # get optimizer
    optimizer, opti_params = common_theano.get_optimizer(optimizer_name)

    # get random variables
    np_rng, th_rng = common_theano.get_rngs(seed)

    # create the model
    print(dv, de, n_topics, n_classes)

    model = ngtm.NGTM(dv, de, n_topics, optimizer, opti_params, np_rng, th_rng, n_classes, encoder_layers=encoder_layers, generator_layers=generator_layers, generator_transform=transform, use_interactions=use_interactions, clip_gradients=clip_gradients, init_bias=init_bias, train_bias=train_bias, scale=init_scale, encode_labels=encode_labels, l1_inter_factor=l1_inter_factor, time_penalty=time_penalty, encoder_shortcut=encoder_shortcut, generator_shortcut=generator_shortcut)

    # train
    train(model, train_X, vocab, index_arrays, train_labels, learning_rate, l1_strength, min_epochs, max_epochs, patience, n_dev, sort_on_first_epoch, model_file, log_file, sparsity_target, alternate)

    # reload best params to print final topics
    print("Loading best parameters")
    model.load_params(model_file)
    log(log_file, "Final topics:")
    print_topics(model, vocab, log_file, write_to_log=True)

    # evaluate
    if test_prefix is not None:
        eval_nvdm(model, model_file, train_labels, test_X, test_index_arrays, test_labels, log_file)
        print("Saving representations")
        combined_X = np.concatenate([train_X, test_X], axis=0)
        combined_labels = np.concatenate([train_labels, test_labels], axis=0)
        save_mean_representations(model, model_file, combined_X, combined_labels, pred_file)


def load_data(input_dir, input_prefix, log_file, vocab=None):
    print("Loading data")
    temp = fh.load_sparse(os.path.join(input_dir, input_prefix + '.npz')).todense()
    X = np.array(temp, dtype='float32')
    if vocab is None:
        vocab = fh.read_json(os.path.join(input_dir, input_prefix + '.vocab.json'))
    lists_of_indices = fh.read_json(os.path.join(input_dir, input_prefix + '.indices.json'))
    index_arrays = [np.array(l, dtype='int32') for l in lists_of_indices]
    n_items, vocab_size = X.shape
    print(n_items, len(index_arrays))
    assert vocab_size == len(vocab)
    assert n_items == len(index_arrays)
    log(log_file, "Loaded %d documents with %d features" % (n_items, vocab_size))

    label_file = os.path.join(input_dir, input_prefix + '.labels.npz')
    if os.path.exists(label_file):
        print("Loading labels")
        temp = fh.load_sparse(label_file).todense()
        labels = np.array(temp, dtype='float32')
    else:
        print("Label file not found")
        labels = np.zeros([n_items, 1], dtype='float32')
    assert len(labels) == n_items

    counts_sum = X.sum(axis=0)
    order = list(np.argsort(counts_sum).tolist())
    order.reverse()
    print("Most common words: ", ' '.join([vocab[i] for i in order[:10]]))

    return X, vocab, index_arrays, labels


def load_background_freq(input_dir, input_prefix, vocab):
    word_freq_file = os.path.join(input_dir, input_prefix + '.word_freq.json')
    if os.path.exists(word_freq_file):
        print("Loading background frequencies")
        log_word_freq = np.log(np.array(fh.read_json(word_freq_file)))
        order = np.argsort(log_word_freq)
        for i in range(10):
            print('%d %s %0.3f' % (i, vocab[order[-i-1]], np.exp(log_word_freq[order[-i-1]])))
    else:
        print("*** Background word frequency file not found! ***")
        log_word_freq = None
    return log_word_freq


def train(model, train_X, vocab, index_arrays, labels, lr, l1_strength, min_epochs, max_epochs, patience, n_dev, sort_on_first_epoch, model_file, log_file, sparsity_target, alternate):
    log(log_file, "Training")
    n_items, dv = train_X.shape
    n_topics = model.d_t

    # normalize input vectors
    train_X = normalize(np.array(train_X, dtype='float32'), axis=1)

    items = range(n_items)
    dev_indices = np.random.choice(items, size=n_dev, replace=False)
    train_indices = list(set(items) - set(dev_indices))

    # sort the training examples by length for the first epoch
    if sort_on_first_epoch:
        train_sums = [len(index_arrays[i]) for i in train_indices]
        order = np.argsort(train_sums).tolist()
        temp = [train_indices[i] for i in order]
        assert set(temp) == set(train_indices)
        train_indices = temp

    min_bound = np.inf
    best_model = None
    epochs_since_improvement = 0
    kl_grow_epochs = 0
    kl_base, step = np.float32(len(train_indices) * kl_grow_epochs), 0

    for epoch in range(max_epochs):
        log(log_file, "Epoch %d" % epoch)

        # shuffle the order of the data
        if epoch > 0:
            print("Shuffling data")
            np.random.shuffle(train_indices)

        log(log_file, "epoch item nll KLD l1p running_av_cost")
        running_cost = 0

        # do training one item at a time
        for i, item in enumerate(train_indices):
            y = labels[item]

            step += 1
            if epoch < kl_grow_epochs:
                kl_strength = np.float32(step / kl_base)
            else:
                kl_strength = 1.
            if n_topics > 1:
                if alternate:
                    if epoch % 2 == 0:
                        nll, KLD, penalty = model.train_decoder(train_X[item, :], index_arrays[item], y, lr, l1_strength, 1)
                    else:
                        nll, KLD, penalty = model.train_not_decoder(train_X[item, :], index_arrays[item], y, lr, l1_strength, 1)
                else:
                    nll, KLD, penalty = model.train(train_X[item, :], index_arrays[item], y, lr, l1_strength, kl_strength)
            else:
                nll, penalty = model.train_label_only(index_arrays[item], y, lr, l1_strength)
                KLD = 0
            # KLD = 0
            running_cost += nll + KLD
            if (i+1) % 500 == 0:
                log(log_file, "%d %d %0.4f %0.4f %0.4f %0.6f" % (epoch, i+1, nll, KLD, penalty, running_cost / float(i+1)))

            # check for nans and infs
            if np.isnan(nll):
                log(log_file, "NaN encountered in document %d" % item)
                words = [vocab[i] for i in index_arrays[item]]
                words.sort()
                print("%d %d %0.4f %0.4f %0.4f" % (epoch, i+1, nll, KLD, penalty))
                sys.exit()
            if np.isinf(nll):
                log(log_file, "inf encountered in in document %d" % item)
                words = [vocab[i] for i in index_arrays[item]]
                words.sort()
                print(' '.join(words))
                print("%d %d %0.4f %0.4f %0.4f" % (epoch, i+1, nll, KLD, penalty))
                sys.exit()

        print_topics(model, vocab, log_file)
        sparsity = check_sparsity(model)
        log(log_file, "\nOverall sparity = %0.3f" % sparsity)

        if sparsity_target is not None:
            l1_strength = np.array(update_l1(l1_strength, sparsity, sparsity_target), dtype=np.float32)
            print("Sparisty target = %0.3f" % sparsity_target)
            print("New l1 strength = %0.4f" % l1_strength)

        # estimate bound on dev set
        bound = 0
        bound_mu = 0
        avg_kld = 0
        predictions = []
        dev_labels = []
        for i, item in enumerate(dev_indices):
            sample_nlls = []
            sample_nlls_mu = []
            klds = []
            y = labels[item]
            dev_labels.append(np.argmax(y))
            for j in range(20):
                if n_topics > 1:
                    nll, KLD = model.neg_log_likelihood(train_X[item, :], index_arrays[item], y)
                    nll_mu, KLD = model.neg_log_likelihood_mu(train_X[item, :], index_arrays[item], y)
                else:
                    nll = model.neg_log_likelihood_label_only(index_arrays[item], y)
                    nll_mu = nll
                    KLD = 0
                # KLD = 0
                KLD *= kl_strength
                klds.append(KLD)
                sample_nlls.append(nll + KLD)
                sample_nlls_mu.append(nll_mu + KLD)
            bound += np.mean(sample_nlls) / float(len(index_arrays[item]))
            bound_mu += np.mean(sample_nlls_mu) / float(len(index_arrays[item]))
            avg_kld += np.mean(klds) / float(len(index_arrays[item]))

        bound = np.exp(bound/float(len(dev_indices)))
        bound_mu = np.exp(bound_mu/float(len(dev_indices)))
        log(log_file, "\nEstimated perplexity upper bound on validation set = %0.3f" % bound)

        if epoch < kl_grow_epochs:
            print("Growing kl factor")
        elif bound < min_bound:
            log(log_file, "\nNew best dev bound = %0.3f" % bound)
            min_bound = bound
            print("Saving model")
            model.save_params(model_file)
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            log(log_file, "No improvement in %d epoch(s)" % epochs_since_improvement)

        if epochs_since_improvement > patience and epoch > min_epochs:
            break

    print("Finished training")
    log(log_file, "Best bound on dev data = %0.3f" % min_bound)


def eval_nvdm(model, model_filename, train_labels, test_X, index_arrays, labels, log_file):
    n_items, dv = test_X.shape
    n_classes = model.n_classes
    n_topics = model.d_t

    test_X = normalize(np.array(test_X, dtype='float32'), axis=1)

    print("Loading best parameters")
    model.load_params(model_filename)

    print("Estimating bound")
    # evaluate bound on test set
    bound = 0
    item_mus = []
    bound_mu = 0
    for item in range(n_items):
        y = labels[item]
        sample_nlls = []
        sample_nlls_mu = []
        for j in range(20):
            if n_topics > 1:
                nll, KLD = model.neg_log_likelihood(test_X[item, :], index_arrays[item], y)
                mu, log_sigma = model.encode(test_X[item, :], y)
                nll_mu, KLD = model.neg_log_likelihood_mu(test_X[item, :], index_arrays[item], y)
            else:
                nll = model.neg_log_likelihood_label_only(index_arrays[item], y)
                nll_mu = nll
                KLD = 0
            # KLD = 0
            sample_nlls.append(nll + KLD)
            sample_nlls_mu.append(nll_mu + KLD)
        bound += np.mean(sample_nlls) / float(len(index_arrays[item]))
        bound_mu += np.mean(sample_nlls_mu) / float(len(index_arrays[item]))

        # save the mean document representation
        r_mu = model.get_mean_doc_rep(test_X[item, :], y)
        item_mus.append(np.array(r_mu))

    bound = np.exp(bound/float(n_items))
    bound_mu = np.exp(bound_mu/float(n_items))
    log(log_file, "\nEstimated perplexity upper bound on test set = %0.3f" % bound)


def save_mean_representations(model, model_filename, X, labels, pred_file):
    n_items, dv = X.shape
    n_classes = model.n_classes
    n_topics = model.d_t

    # try normalizing input vectors
    test_X = normalize(np.array(X, dtype='float32'), axis=1)

    model.load_params(model_filename)

    # evaluate bound on test set
    item_mus = []
    for item in range(n_items):
        y = labels[item]

        # save the mean document representation
        r_mu = model.get_mean_doc_rep(test_X[item, :], y)
        item_mus.append(np.array(r_mu))

    # write all the test doc representations to file
    if pred_file is not None and n_topics > 1:
        np.savez_compressed(pred_file, X=np.array(item_mus), y=labels)


def print_topics(model, vocab, log_file, write_to_log=False, sparsity_threshold=1e-3):
    n_topics = model.d_t
    if n_topics > 1:
        log(log_file, "Topics:", write_to_log)
        weights = np.array(model.W_decoder.get_value())
        mean_sparsity = 0.0
        for j in range(model.d_t):
            order = list(np.argsort(weights[:, j]).tolist())
            order.reverse()
            highest = ' '.join([vocab[i] for i in order[:7]])
            lowest = ' '.join([vocab[i] for i in order[-4:]])
            min_w = weights[:, j].min()
            max_w = weights[:, j].max()
            mean_w = weights[:, j].mean()
            sparsity = np.array(np.abs(weights[:, j]) < sparsity_threshold, dtype=float).sum() / float(model.d_v)
            mean_sparsity += sparsity
            log(log_file, "%d %s / %s (%0.3f, %0.3f, %0.3f) [%0.5f]" % (j, highest, lowest, min_w, mean_w, max_w, sparsity), write_to_log)
        sparsity = np.array(np.abs(weights) < sparsity_threshold, dtype=float).sum() / float(model.d_v * model.d_t)
        log(log_file, "Topic sparsity = %0.3f" % sparsity, write_to_log)

    n_classes = model.n_classes
    if n_classes > 1:
        log(log_file, "\nClasses:", write_to_log)
        weights = np.array(model.W_decoder_label.get_value())
        mean_sparsity = 0.0
        for j in range(n_classes):
            order = list(np.argsort(weights[:, j]).tolist())
            order.reverse()
            highest = ' '.join([vocab[i] for i in order[:7]])
            lowest = ' '.join([vocab[i] for i in order[-4:]])
            min_w = weights[:, j].min()
            max_w = weights[:, j].max()
            mean_w = weights[:, j].mean()
            sparsity = np.array(np.abs(weights[:, j]) < sparsity_threshold, dtype=float).sum() / float(model.d_v)
            mean_sparsity += sparsity
            log(log_file, "%d %s / %s (%0.3f, %0.3f, %0.3f) [%0.5f]" % (j, highest, lowest, min_w, mean_w, max_w, sparsity), write_to_log)
        sparsity = np.array(np.abs(weights) < sparsity_threshold, dtype=float).sum() / float(model.d_v * n_classes)
        log(log_file, "Covariate sparsity = %0.3f" % sparsity, write_to_log)

        if model.use_interactions:
            weights = np.array(model.W_decoder_inter.get_value())
            sparsity = np.array(np.abs(weights) < sparsity_threshold, dtype=float).sum() / float(model.d_v * model.d_t * model.n_classes)
            log(log_file, "Interaction sparsity = %0.3f" % sparsity, write_to_log)


def check_sparsity(model, sparsity_threshold=1e-3):
    """
    Return the proportion of decoder weights that are approximately zero (less than sparsity_threshold)
    :param model:  A NVDM model
    :param sparsity_threshold: the threshold below which to consider things to be zero
    :return: The proportion of weights that are approximately zero (float)
    """
    num_zero = 0
    num_elements = 0
    if model.d_t > 1:
        weights = np.array(model.W_decoder.get_value())
        num_elements += model.d_v * model.d_t
        num_zero += np.array(np.abs(weights) < sparsity_threshold, dtype=float).sum()
    if model.n_classes > 1:
        weights = np.array(model.W_decoder_label.get_value())
        num_elements += model.d_v * model.n_classes
        num_zero += np.array(np.abs(weights) < sparsity_threshold, dtype=float).sum()
        if model.use_interactions:
            weights = np.array(model.W_decoder_inter.get_value())
            num_elements += model.d_v * model.d_t * model.n_classes
            num_zero += np.array(np.abs(weights) < sparsity_threshold, dtype=float).sum()
    return num_zero / float(num_elements)


def update_l1(current_l1, sparsity, sparsity_target):
    """
    A method to update the l1 penalty to try to achieve a target sparsity
    :param current_l1: current value of l1
    :param sparsity: current level of sparsity
    :param sparsity_target: target level of sparsity
    :return: new l1 value
    """
    diff = sparsity_target - sparsity
    new_l1 = current_l1 * 2.0 ** diff
    return new_l1


def log(logfile, text, write_to_log=True):
    print(text)
    if write_to_log:
        with codecs.open(logfile, 'a', encoding='utf-8') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()
