import os
import re
import sys
from optparse import OptionParser
from collections import Counter
import numpy as np
from scipy import sparse
from scipy.io import savemat
from spacy.en import English

import file_handling as fh


def main():
    usage = "%prog train.json test.json output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--vocab_size', dest='vocab_size', default=2000,
                      help='Size of the vocabulary (by most common): default=%default')
    parser.add_option('--malletstop', action="store_true", dest="malletstop", default=False,
                     help='Use Mallet stopwords: default=%default')
    parser.add_option('--only_alpha', action="store_true", dest="only_alpha", default=False,
                      help='Drop all tokens that contain characters outside of [a-z]: default=%default')
    parser.add_option('--keep_nonalphanum', action="store_true", dest="keep_nonalphanum", default=False,
                      help='Filter out non-alpha-numeric tokens: default=%default')
    parser.add_option('--replace_num', action="store_true", dest="replace_num", default=False,
                      help='Replace numbers with <NUM>: default=%default')
    parser.add_option('--lemmatize', action="store_true", dest="lemmatize", default=False,
                      help='Use lemmas: default=%default')
    parser.add_option('--min_length', action="store_true", dest="min_length", default=1,
                      help='Minimum token length: default=%default')
    parser.add_option('--log_transform', action="store_true", dest="log_transform", default=False,
                      help='Transform counts by round(log(1 + count)): default=%default')

    (options, args) = parser.parse_args()

    if len(args) != len(usage.split())-1:
        print("Please provide all input arguments")

    train_infile = args[0]
    test_infile = args[1]
    output_dir = args[2]
    vocab_size = int(options.vocab_size)
    use_mallet_stopwords = options.malletstop
    only_alpha = options.only_alpha
    keep_nonalphanum = options.keep_nonalphanum
    replace_num = options.replace_num
    lemmatize = options.lemmatize
    min_length = int(options.min_length)
    log_transform = options.log_transform

    if not os.path.exists(output_dir):
        sys.exit("Error: output directory does not exist")

    preprocess_data(train_infile, test_infile, output_dir, vocab_size, use_mallet_stopwords=use_mallet_stopwords, replace_num=replace_num, lemmatize=lemmatize, log_transform=log_transform, keep_nonalphanum=keep_nonalphanum, only_alpha=only_alpha, min_length=min_length)


def preprocess_data(train_infile, test_infile, output_dir, vocab_size, use_mallet_stopwords=False, replace_num=False, lemmatize=False, log_transform=False, keep_nonalphanum=False, only_alpha=False, min_length=1):

    print("Loading SpaCy")
    parser = English()
    train_X, train_vocab, train_indices, train_y, label_list, word_freqs, train_dat, train_mallet_strings, train_sage_output, train_svm_strings = load_and_process_data(train_infile, vocab_size, parser, use_mallet_stopwords=use_mallet_stopwords, replace_num=replace_num, lemmatize=lemmatize, log_transform=log_transform, keep_nonalphanum=keep_nonalphanum, only_alpha=only_alpha, min_length=min_length)
    test_X, _, test_indices, test_y, _, _, test_dat, test_mallet_strings, test_sage_output, test_svm_strings = load_and_process_data(test_infile, vocab_size, parser, vocab=train_vocab, label_list=label_list, use_mallet_stopwords=use_mallet_stopwords, replace_num=replace_num, lemmatize=lemmatize, log_transform=log_transform, keep_nonalphanum=keep_nonalphanum, only_alpha=only_alpha, min_length=min_length)
    fh.save_sparse(train_X, os.path.join(output_dir, 'train.npz'))
    fh.write_to_json(train_vocab, os.path.join(output_dir, 'train.vocab.json'))
    fh.write_to_json(train_indices, os.path.join(output_dir, 'train.indices.json'))
    fh.save_sparse(train_y, os.path.join(output_dir, 'train.labels.npz'))
    fh.save_sparse(test_X, os.path.join(output_dir, 'test.npz'))
    fh.write_to_json(test_indices, os.path.join(output_dir, 'test.indices.json'))
    fh.save_sparse(test_y, os.path.join(output_dir, 'test.labels.npz'))
    n_labels = len(label_list)
    label_dict = dict(zip(range(n_labels), label_list))
    fh.write_to_json(label_dict, os.path.join(output_dir, 'train.label_list.json'))
    fh.write_to_json(list(word_freqs.tolist()), os.path.join(output_dir, 'train.word_freq.json'))

    # save output for David Blei's lda-c code
    fh.write_list_to_text(train_dat, os.path.join(output_dir, 'train.dat'))
    fh.write_list_to_text(test_dat, os.path.join(output_dir, 'test.dat'))

    # save output for Mallet
    fh.write_list_to_text(train_mallet_strings, os.path.join(output_dir, 'train.mallet.txt'))
    fh.write_list_to_text(test_mallet_strings, os.path.join(output_dir, 'test.mallet.txt'))

    # save output for Jacob Eisenstein's SAGE code:
    train_sage_output['te_data'] = test_sage_output['tr_data']
    train_sage_output['te_aspect'] = test_sage_output['tr_aspect']
    savemat(os.path.join(output_dir, 'sage.mat'), train_sage_output)

    # save output in SVM format
    fh.write_list_to_text(train_svm_strings, os.path.join(output_dir, 'train.svm.txt'))
    fh.write_list_to_text(test_svm_strings, os.path.join(output_dir, 'test.svm.txt'))


def load_and_process_data(infile, vocab_size, parser, strip_html=False, vocab=None, label_list=None, use_mallet_stopwords=False, replace_num=False, lemmatize=False, log_transform=False, keep_nonalphanum=False, only_alpha=False, min_length=1):

    mallet_stopwords = None
    if use_mallet_stopwords:
        print("Using MALLET stopwords")
        mallet_stopwords = fh.read_text('mallet_stopwords.txt')
        mallet_stopwords = {s.strip() for s in mallet_stopwords}

    print("Reading data files")
    item_dict = fh.read_json(infile)
    n_items = len(item_dict)

    parsed = []
    labels = []

    print("Parsing %d documents" % n_items)
    word_counts = Counter()
    doc_counts = Counter()
    keys = list(item_dict.keys())
    keys.sort()
    for i, k in enumerate(keys):
        item = item_dict[k]
        if i % 1000 == 0 and i > 0:
            print(i)

        text = item['text']
        label = item['label']
        labels.append(label)

        if strip_html:
            # remove each pair of angle brackets and everything within them
            text = re.sub('<[^>]+>', '', text)

        parse = parser(text)
        # remove white space from tokens
        if lemmatize:
            words = [re.sub('\s', '', token.lemma_) for token in parse]
        else:
            words = [re.sub('\s', '', token.orth_) for token in parse]
        # convert to lower case and drop empty strings
        words = [word.lower() for word in words if len(word) >= min_length]
        # remove stop words
        if use_mallet_stopwords:
            words = [word for word in words if word not in mallet_stopwords]
        # remove tokens that don't contain letters or numbers
        if only_alpha:
            words = [word for word in words if re.match('^[a-zA-A]*$', word) is not None]
        if not keep_nonalphanum:
            words = [word for word in words if re.match('[a-zA-A0-9]', word) is not None]
        # convert numbers to a number symbol
        if replace_num:
            words = ['<NUM>' if re.match('[0-9]', word) is not None else word for word in words]
        # store the parsed documents
        parsed.append(words)
        # keep track fo the number of documents with each word
        word_counts.update(words)
        doc_counts.update(set(words))

    print("Size of full vocabulary=%d" % len(word_counts))

    if vocab is None:
        most_common = doc_counts.most_common(n=vocab_size)
        words, counts = zip(*most_common)
        print("Most common words:")
        for w in range(20):
            print(words[w], doc_counts[words[w]], word_counts[words[w]])
        vocab = list(words)
        vocab.sort()
        total_words = np.sum(list(word_counts.values()))
        word_freqs = np.array([word_counts[v] for v in vocab]) / float(total_words)
    else:
        word_freqs = None

    vocab_index = dict(zip(vocab, range(vocab_size)))

    if label_list is None:
        label_list = list(set(labels))
        label_list.sort()

    n_labels = len(label_list)
    label_index = dict(zip(label_list, range(n_labels)))

    X = np.zeros([n_items, vocab_size], dtype=int)
    y = []

    dat_strings = []
    svm_strings = []
    mallet_strings = []

    lists_of_indices = []  # an alternative representation of each document as a list of indices

    print("First document:")
    print(' '.join(parsed[0]))

    counter = Counter()
    print("Converting to count representations")
    count = 0
    total_tokens = 0
    for i, words in enumerate(parsed):
        indices = [vocab_index[word] for word in words if word in vocab_index]
        word_subset = [word for word in words if word in vocab_index]
        counter.clear()
        counter.update(indices)
        # only include non-empty documents
        if len(counter.keys()) > 0:
            # udpate the counts
            mallet_strings.append(str(i) + '\t' + 'en' + '\t' + ' '.join(word_subset))
            values = list(counter.values())
            if log_transform:
                # apply the log transform from Salakhutdinov and Hinton
                values = np.array(np.round(np.log(1 + np.array(values, dtype='float'))), dtype=int)
            X[np.ones(len(counter.keys()), dtype=int) * count, list(counter.keys())] += values
            total_tokens += len(word_subset)
            y_vector = np.zeros(n_labels)
            y_vector[label_index[labels[i]]] = 1
            y.append(y_vector)
            #y.append(label_index[labels[i]])
            # save the list of indices
            lists_of_indices.append(indices)
            dat_string = str(int(len(counter))) + ' '
            dat_string += ' '.join([str(int(k)) + ':' + str(int(v)) for k, v in zip(list(counter.keys()), list(counter.values()))])
            dat_strings.append(dat_string)
            svm_string = 'target '
            svm_string += ' '.join([vocab[int(k)] + ':' + str(int(v)) for k, v in zip(list(counter.keys()), list(counter.values()))])
            svm_strings.append(svm_string)
            #text_map[count] = words
            count += 1

    print("Found %d non-empty documents" % count)
    print("Total tokens = %d" % total_tokens)

    # drop the items that don't have any words in the vocabualry
    X = np.array(X[:count, :], dtype=int)

    temp = np.array(y)
    y = np.array(temp[:count], dtype=int)
    sparse_y = sparse.csr_matrix(y)

    # convert to a sparse representation
    sparse_X = sparse.csr_matrix(X)
    sparse_X_sage = sparse.csr_matrix(X, dtype=float)

    vocab_for_sage = np.zeros((vocab_size,), dtype=np.object)
    vocab_for_sage[:] = vocab

    tr_aspect = np.ones([n_items, 1], dtype=float)
    widx = np.arange(vocab_size, dtype=float) + 1

    sage_output = {'tr_data': sparse_X_sage, 'tr_aspect': tr_aspect, 'widx': widx, 'vocab': vocab_for_sage}

    return sparse_X, vocab, lists_of_indices, sparse_y, label_list, word_freqs, dat_strings[:count], mallet_strings[:count], sage_output, svm_strings[:count]


if __name__ == '__main__':
    main()
