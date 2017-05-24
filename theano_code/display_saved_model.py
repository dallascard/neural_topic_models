from optparse import OptionParser
import numpy as np
import file_handling as fh


def main():
    usage = "%prog model_file.npz vocab_file.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--sparsity_thresh', dest='sparsity_thresh', default=1e-3,
                      help='Sparsity threshold: default=%default')
    parser.add_option('--interactions', action="store_true", dest="interactions", default=False,
                      help='Print interaction topics: default=%default')
    parser.add_option('--n_pos', dest='n_pos', default=7,
                      help='Number of positive terms to display: default=%default')
    parser.add_option('--n_neg', dest='n_neg', default=4,
                      help='Number of negative terms to display: default=%default')
    parser.add_option('--max_classes', dest='max_classes', default=None,
                      help='Maximum number of classes to display: default=%default')

    (options, args) = parser.parse_args()

    model_file = args[0]
    vocab_file = args[1]

    params = np.load(model_file)
    vocab = fh.read_json(vocab_file)
    n_pos = int(options.n_pos)
    n_neg = int(options.n_neg)
    max_classes = options.max_classes

    sparsity_threshold = options.sparsity_thresh
    interactions = options.interactions

    dv = params['d_v']
    n_topics = params['d_t']
    n_classes = params['n_classes']
    if max_classes is not None:
        n_classes = int(max_classes)

    if n_topics > 1:
        print("\nTopics:")
        weights = np.array(params['W_decoder'])
        mean_sparsity = 0.0
        for j in range(n_topics):
            order = list(np.argsort(weights[:, j]).tolist())
            order.reverse()
            highest = ' '.join([vocab[i] for i in order[:n_pos]])
            lowest = ' '.join([vocab[i] for i in order[-n_neg:]])
            min_w = weights[:, j].min()
            max_w = weights[:, j].max()
            mean_w = weights[:, j].mean()
            sparsity = np.array(np.abs(weights[:, j]) < sparsity_threshold, dtype=float).sum() / float(dv)
            mean_sparsity += sparsity
            print("%d %s / %s (%0.3f, %0.3f, %0.3f) [%0.5f]" % (j, highest, lowest, min_w, mean_w, max_w, sparsity))
        sparsity = np.array(np.abs(weights) < sparsity_threshold, dtype=float).sum() / float(dv * n_topics)
        print("Topic sparsity = %0.3f" % sparsity)


    if n_classes > 1:
        print("\nClasses:")
        weights = np.array(params['W_decoder_label'])
        mean_sparsity = 0.0
        for j in range(n_classes):
            order = list(np.argsort(weights[:, j]).tolist())
            order.reverse()
            highest = ' '.join([vocab[i] for i in order[:n_pos]])
            lowest = ' '.join([vocab[i] for i in order[-n_neg:]])
            min_w = weights[:, j].min()
            max_w = weights[:, j].max()
            mean_w = weights[:, j].mean()
            sparsity = np.array(np.abs(weights[:, j]) < sparsity_threshold, dtype=float).sum() / float(dv)
            mean_sparsity += sparsity
            print("%d %s / %s (%0.3f, %0.3f, %0.3f) [%0.5f]" % (j, highest, lowest, min_w, mean_w, max_w, sparsity))
        sparsity = np.array(np.abs(weights) < sparsity_threshold, dtype=float).sum() / float(dv * n_classes)
        print("Covariate sparsity = %0.3f" % sparsity)

    if params['use_interactions']:
        print("\nInteractions:")
        interaction_weights = np.array(params['W_decoder_inter'])
        if interactions:
            mean_sparsity = 0.0
            for j in range(n_topics):
                for k in range(n_classes):
                    index = k + j * n_classes
                    weights_sum = interaction_weights[:, index]
                    order = list(np.argsort(weights_sum).tolist())
                    order.reverse()
                    highest = ' '.join([vocab[i] for i in order[:n_pos]])
                    lowest = ' '.join([vocab[i] for i in order[-n_neg:]])
                    min_w = weights_sum.min()
                    max_w = weights_sum.max()
                    mean_w = weights_sum.mean()
                    sparsity = np.array(np.abs(weights_sum) < sparsity_threshold, dtype=float).sum() / float(dv)
                    mean_sparsity += sparsity
                    print("%d/%d %s / %s (%0.3f, %0.3f, %0.3f) [%0.5f]" % (j, k, highest, lowest, min_w, mean_w, max_w, sparsity))

        sparsity = np.array(np.abs(interaction_weights) < sparsity_threshold, dtype=float).sum() / float(dv * n_topics * n_classes)
        print("Interaction sparsity = %0.3f" % sparsity)

        print("\nWith interactions (but no labels):")
        topic_weights = np.array(params['W_decoder'])
        interaction_weights = np.array(params['W_decoder_inter'])
        if interactions:
            mean_sparsity = 0.0
            for j in range(n_topics):
                print(j)
                for k in range(n_classes):
                    index = k + j * n_classes
                    weights_sum = topic_weights[:, j] + interaction_weights[:, index]
                    order = list(np.argsort(weights_sum).tolist())
                    order.reverse()
                    highest = ' '.join([vocab[i] for i in order[:n_pos]])
                    lowest = ' '.join([vocab[i] for i in order[-n_neg:]])
                    min_w = weights_sum.min()
                    max_w = weights_sum.max()
                    mean_w = weights_sum.mean()
                    sparsity = np.array(np.abs(weights_sum) < sparsity_threshold, dtype=float).sum() / float(dv)
                    mean_sparsity += sparsity
                    print("%d/%d %s / %s (%0.3f, %0.3f, %0.3f) [%0.5f]" % (j, k, highest, lowest, min_w, mean_w, max_w, sparsity))


if __name__ == '__main__':
    main()
