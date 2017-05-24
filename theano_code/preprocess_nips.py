import os
import re
import sys
import json
import codecs
from optparse import OptionParser
from collections import Counter
import numpy as np
from scipy import sparse
from scipy.io import savemat
from spacy.en import English

import file_handling as fh


def main():
    usage = "%prog train.jsonlist output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--vocab_size', dest='vocab_size', default=10000,
                      help='Size of the vocabulary (by most common): default=%default')
    parser.add_option('--test_prop', dest='test_prop', default=0.2,
                      help='Proportion of documents for test set: default=%default')
    parser.add_option('--group_size', dest='group_size', default=3,
                      help='Number of years to group together: default=%default')
    parser.add_option('--labels', dest='labels', default='years',
                      help='Labels to use [year|authors|both]: default=%default')
    parser.add_option('--non_alpha', action="store_true", dest="non_alpha", default=False,
                      help='Drop all tokens that contain characters outside of [a-z]: default=%default')
    parser.add_option('--malletstop', action="store_true", dest="malletstop", default=False,
                      help='Use Mallet stopwords: default=%default')
    parser.add_option('--replace_num', action="store_true", dest="replace_num", default=False,
                      help='Replace numbers with <NUM>: default=%default')
    parser.add_option('--min_length', action="store_true", dest="min_length", default=3,
                      help='Minimum token length: default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Randomization seed: default=%default')

    (options, args) = parser.parse_args()

    if len(args) != len(usage.split())-1:
        print("Please provide all input arguments")

    train_infile = args[0]
    output_dir = args[1]

    vocab_size = int(options.vocab_size)
    test_prop = float(options.test_prop)
    label_type = options.labels
    only_alpha = not options.non_alpha
    use_mallet_stopwords = options.malletstop
    replace_num = options.replace_num
    seed = int(options.seed)
    group_size = int(options.group_size)
    min_length = int(options.min_length)

    np.random.seed(seed)

    if not os.path.exists(output_dir):
        sys.exit("Error: output directory does not exist")

    preprocess_data(train_infile, output_dir, vocab_size, label_type, test_prop, use_mallet_stopwords=use_mallet_stopwords, replace_num=replace_num, group_size=group_size, only_alpha=only_alpha, min_length=min_length)


def preprocess_data(train_infile, output_dir, vocab_size, label_type, test_prop, use_mallet_stopwords=False, replace_num=False, group_size=1, only_alpha=False, min_length=3):

    print("Loading SpaCy")
    parser = English()

    with codecs.open(train_infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    n_items = len(lines)
    n_test = int(test_prop * n_items)
    n_train = n_items - n_test
    train_indices = np.random.choice(range(n_items), n_train, replace=False)
    test_indices = list(set(range(n_items)) - set(train_indices))

    train_X, train_vocab, train_indices, train_y, label_list, word_freqs, train_dat, train_mallet_strings, train_sage_output, train_svm_strings, label_index = load_and_process_data(train_infile, vocab_size, parser, label_type, train_indices, use_mallet_stopwords=use_mallet_stopwords, replace_num=replace_num, group_size=group_size, only_alpha=only_alpha, min_length=min_length)
    test_X, _, test_indices, test_y, _, _, test_dat, test_mallet_strings, test_sage_output, test_svm_strings, _ = load_and_process_data(train_infile, vocab_size, parser, label_type, test_indices, vocab=train_vocab, label_list=label_list, label_index=label_index, use_mallet_stopwords=use_mallet_stopwords, replace_num=replace_num, group_size=group_size, only_alpha=only_alpha, min_length=min_length)
    fh.save_sparse(train_X, os.path.join(output_dir, 'train.npz'))
    fh.write_to_json(train_vocab, os.path.join(output_dir, 'train.vocab.json'))
    fh.write_to_json(train_indices, os.path.join(output_dir, 'train.indices.json'))
    fh.save_sparse(train_y, os.path.join(output_dir, 'train.labels.npz'))
    fh.save_sparse(test_X, os.path.join(output_dir, 'test.npz'))
    fh.write_to_json(test_indices, os.path.join(output_dir, 'test.indices.json'))
    fh.save_sparse(test_y, os.path.join(output_dir, 'test.labels.npz'))
    fh.write_to_json(list(word_freqs.tolist()), os.path.join(output_dir, 'train.word_freq.json'))
    fh.write_list_to_text(train_dat, os.path.join(output_dir, 'train.dat'))
    n_labels = len(label_list)
    label_dict = dict(zip(range(n_labels), label_list))
    fh.write_to_json(label_dict, os.path.join(output_dir, 'train.label_list.json'))

    fh.write_list_to_text(train_mallet_strings, os.path.join(output_dir, 'train.mallet.txt'))
    fh.write_list_to_text(test_mallet_strings, os.path.join(output_dir, 'test.mallet.txt'))

    train_sage_output['te_data'] = test_sage_output['tr_data']
    train_sage_output['te_aspect'] = test_sage_output['tr_aspect']
    savemat(os.path.join(output_dir, 'sage.mat'), train_sage_output)

    fh.write_list_to_text(train_svm_strings, os.path.join(output_dir, 'train.svm.txt'))
    fh.write_list_to_text(test_svm_strings, os.path.join(output_dir, 'test.svm.txt'))


def load_and_process_data(infile, vocab_size, parser, label_type, data_indices, strip_html=False, vocab=None, label_list=None, label_index=None, use_mallet_stopwords=False, replace_num=False, lemmatize=False, keep_nonalphanum=False, only_alpha=False, group_size=1, min_length=3):

    mallet_stopwords = []
    if use_mallet_stopwords:
        print("Using MALLET stopwords")
        mallet_stopwords = fh.read_text('mallet_stopwords.txt')
        mallet_stopwords = {s.strip() for s in mallet_stopwords}

    print("Reading data files")
    with codecs.open(infile, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    author_counts = Counter()
    author_set = set()
    year_set = set()
    item_dict = {}
    for l_i, l in enumerate(data_indices):
        line = lines[l]
        item = json.loads(line)
        new_item = {'text': item['text'], 'year': str(item['date']), 'authors': item['authors']}
        author_set.update(item['authors'])
        author_counts.update(item['authors'])
        year_set.update([str(item['date'])])
        item_dict[l_i] = new_item

    if label_index is None:
        years = range(1987, 2017, group_size)
        n_years = len(years)
        year_index = {}
        index = 0
        for year in years:
            for offset in range(group_size):
                year_index[str(int(year + offset))] = index
            index += 1
        print(type(year_index[list(year_index.keys())[0]]))
        print('%d years' % n_years)

        author_counts = Counter({a: c for a, c in author_counts.items() if c >= 20})
        authors = list(author_counts.keys())
        authors.sort()
        n_authors = len(authors)
        author_index = dict(zip(authors, np.arange(n_authors, dtype=int)))
        print('%d authors with at least 20 papers' % n_authors)
        print(authors)

        if label_type == 'years':
            label_index = year_index
            label_list = [str(int(y)) for y in years]
        elif label_type == 'author':
            label_index = author_index
            label_list = authors
        elif label_type == 'both':
            label_index = year_index
            author_index = dict(zip(authors, np.arange(n_authors, dtype=int) + int(n_years)))
            label_index.update(author_index)
            label_list = [str(int(y)) for y in years] + authors
        n_labels = len(label_list)
    else:
        n_labels = len(label_list)

    print(label_list)
    print(label_index)

    n_items = len(item_dict)

    parsed = []
    label_vectors = []

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
        if label_type == 'years':
            year = item['year']
            label_vector = np.zeros(n_labels)
            label_vector[label_index[str(year)]] = 1
            label_vectors.append(label_vector)
        elif label_type == 'authors':
            authors = item['authors']
            label_vector = np.zeros(n_labels)
            for a in authors:
                if a in label_index:
                    label_vector[label_index[a]] = 1
            if np.sum(label_vector) > 0:
                label_vector = label_vector / float(label_vector.sum())
            label_vectors.append(label_vector)
        elif label_type == 'both':
            year = item['year']
            authors = item['authors']
            label_vector = np.zeros(n_labels)
            try:
                label_vector[label_index[str(year)]] = 1
            except:
                print(year)
                print(label_index[str(year)])
                label_vector[label_index[str(year)]] = 1
            for a in authors:
                if a in label_index:
                    label_vector[label_index[a]] = 1
            label_vectors.append(label_vector)

        if strip_html:
            # remove each pair of angle brackets and everything within them
            text = re.sub('<[^>]+>', '', text)

        parse = parser(text)
        if lemmatize:
            words = [re.sub('\s', '', token.lemma_) for token in parse]
        else:
            words = [re.sub('\s', '', token.orth_) for token in parse]
        # convert to lower case and drop short/empty strings
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

    X = np.zeros([n_items, vocab_size], dtype='int32')
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
            mallet_strings.append(str(i) + '\t' + 'en' + '\t' + ' '.join(word_subset))
            total_tokens += len(word_subset)
            # udpate the counts
            X[np.ones(len(counter.keys()), dtype=int) * count, list(counter.keys())] += list(counter.values())
            y.append(label_vectors[i])

            # save the list of indices
            lists_of_indices.append(indices)
            dat_string = str(int(len(counter))) + ' '
            dat_string += ' '.join([str(int(k)) + ':' + str(int(v)) for k, v in zip(list(counter.keys()), list(counter.values()))])
            dat_strings.append(dat_string)
            svm_string = 'target '
            svm_string += ' '.join([vocab[int(k)] + ':' + str(int(v)) for k, v in zip(list(counter.keys()), list(counter.values()))])
            svm_strings.append(svm_string)
            count += 1

    print("Found %d non-empty documents" % count)

    # drop the items that don't have any words in the vocabualry
    X = np.array(X[:count, :], dtype='int32')
    temp = np.array(y)
    y = np.array(temp[:count], dtype='int32')
    sparse_y = sparse.csr_matrix(y)

    # convert to a sparse representation
    sparse_X = sparse.csr_matrix(X)
    sparse_X_sage = sparse.csr_matrix(X, dtype=float)

    vocab_for_sage = np.zeros((vocab_size,), dtype=np.object)
    vocab_for_sage[:] = vocab

    tr_aspect = np.ones([count, 1], dtype=float)
    widx = np.arange(vocab_size, dtype=float) + 1

    sage_output = {'tr_data': sparse_X_sage, 'tr_aspect': tr_aspect, 'widx': widx, 'vocab': vocab_for_sage}

    return sparse_X, vocab, lists_of_indices, sparse_y, label_list, word_freqs, dat_strings[:count], mallet_strings[:count], sage_output, svm_strings[:count], label_index


if __name__ == '__main__':
    main()
