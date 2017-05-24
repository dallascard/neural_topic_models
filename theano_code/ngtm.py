import sys
import numpy as np
import theano
import theano.tensor as T
import common_theano

# set this to allow deep copy of model
sys.setrecursionlimit(5000)


class NGTM:

    def __init__(self, d_v, d_e, d_t, optimizer, optimizer_args, np_rng, th_rng, n_classes=0, encoder_layers=1, generator_layers=0, generator_transform=None, use_interactions=False, clip_gradients=False, init_bias=None, train_bias=False, scale=6.0, encode_labels=False, l1_inter_factor=1.0, time_penalty=False, encoder_shortcut=False, generator_shortcut=False):

        self.d_v = d_v  # vocabulary size
        self.d_e = d_e  # dimensionality of encoder
        self.d_t = d_t  # number of topics
        self.n_classes = n_classes  # number of classes
        assert encoder_layers == 1 or encoder_layers == 2
        self.n_encoder_layers = encoder_layers
        assert generator_layers == 0 or generator_layers == 1 or generator_layers == 2 or generator_layers == 4
        self.n_generator_layers = generator_layers

        # set various options
        self.generator_transform = generator_transform   # transform to apply after the generator
        self.use_interactions = use_interactions  # use interactions between topics and labels
        self.encode_labels = encode_labels  # feed labels into the encoder
        self.l1_inter_factor = l1_inter_factor  # factor by which to multiply L1 penalty on interactions
        self.encoder_shortcut = encoder_shortcut
        self.generator_shortcut = generator_shortcut

        # create parameter matrices and biases
        self.W_encoder_1 = common_theano.init_param('W_encoder_1', (d_e, d_v), np_rng, scale=scale)
        self.b_encoder_1 = common_theano.init_param('b_encoder_1', (d_e, ), np_rng, scale=0.0)

        if n_classes > 1:
            self.W_encoder_label = common_theano.init_param('W_encoder_label', (d_e, n_classes), np_rng, scale=scale)
        else:
            self.W_encoder_label = common_theano.init_param('W_encoder_label', (d_e, n_classes), np_rng, values=np.zeros((d_e, n_classes), dtype=np.float32))

        self.W_encoder_2 = common_theano.init_param('W_encoder_2', (d_e, d_e), np_rng, scale=scale)
        self.b_encoder_2 = common_theano.init_param('b_encoder_2', (d_e, ), np_rng, scale=0.0)

        self.W_encoder_shortcut = common_theano.init_param('W_encoder_shortcut', (d_e, d_v), np_rng, scale=scale)

        self.W_mu = common_theano.init_param('W_mu', (d_t, d_e), np_rng, scale=scale)
        self.b_mu = common_theano.init_param('b_mu', (d_t, ), np_rng, scale=0.0)

        self.W_sigma = common_theano.init_param('W_sigma', (d_t, d_e), np_rng, scale=scale, values=np.zeros((d_t, d_e)))
        self.b_sigma = common_theano.init_param('b_sigma', (d_t, ), np_rng, scale=0.0,
                                                values=np.array([-4] * d_t))

        self.W_generator_1 = common_theano.init_param('W_generator_1', (d_t, d_t), np_rng, scale=scale)
        self.b_generator_1 = common_theano.init_param('b_generator_1', (d_t, ), np_rng, scale=0.0)

        self.W_generator_2 = common_theano.init_param('W_generator_2', (d_t, d_t), np_rng, scale=scale)
        self.b_generator_2 = common_theano.init_param('b_generator_2', (d_t, ), np_rng, scale=0.0)
        
        self.W_generator_3 = common_theano.init_param('W_generator_3', (d_t, d_t), np_rng, scale=scale)
        self.b_generator_3 = common_theano.init_param('b_generator_3', (d_t, ), np_rng, scale=0.0)
        
        self.W_generator_4 = common_theano.init_param('W_generator_4', (d_t, d_t), np_rng, scale=scale)
        self.b_generator_4 = common_theano.init_param('b_generator_4', (d_t, ), np_rng, scale=0.0)

        self.W_decoder = common_theano.init_param('W_decoder', (d_v, d_t), np_rng, scale=scale)
        self.b_decoder = common_theano.init_param('b_decoder', (d_v, ), np_rng, scale=0.0)

        self.W_decoder_label = common_theano.init_param('W_decoder_label', (d_v, n_classes), np_rng, scale=scale)
        self.W_decoder_inter = common_theano.init_param('W_decoder_inter', (d_v, d_t * n_classes), np_rng, scale=scale)

        # set the decoder bias to the background frequency
        if init_bias is not None:
            self.b_decoder = common_theano.init_param('b_decoder', (d_v,), np_rng, values=init_bias)

        # create basic sets of parameters which we will use to tell the model what to update
        self.params = [self.W_encoder_1, self.b_encoder_1,
                       self.W_mu, self.b_mu,
                       self.W_sigma, self.b_sigma,
                       self.W_decoder]
        self.param_shapes = [(d_e, d_v), (d_e,),
                             (d_t, d_e), (d_t,),
                             (d_t, d_e), (d_t,),
                             (d_v, d_t)]

        self.encoder_params = [self.W_encoder_1, self.b_encoder_1,
                               self.W_mu, self.b_mu,
                               self.W_sigma, self.b_sigma]
        self.encoder_param_shapes = [(d_e, d_v), (d_e,),
                                     (d_t, d_e), (d_t,),
                                     (d_t, d_e), (d_t,)]

        self.generator_params = []
        self.generator_param_shapes = []

        # add additional parameters to sets, depending on configuration
        if train_bias:
            self.params.append(self.b_decoder)
            self.param_shapes.append((d_v,))
            self.decoder_params = [self.W_decoder, self.b_decoder]
            self.decoder_param_shapes = [(d_v, d_t), (d_v,)]
        else:
            self.decoder_params = [self.W_decoder]
            self.decoder_param_shapes = [(d_v, d_t)]

        # add parameters for labels (covariates)
        if self.n_classes > 1:
            self.params.append(self.W_decoder_label)
            self.param_shapes.append((d_v, n_classes))
            self.decoder_params.extend([self.W_decoder_label])
            self.decoder_param_shapes.extend([(d_v, n_classes)])
            if use_interactions:
                self.params.append(self.W_decoder_inter)
                self.param_shapes.append((d_v, d_t * n_classes))
                self.decoder_params.extend([self.W_decoder_inter])
                self.decoder_param_shapes.extend([(d_v, d_t * n_classes)])
            if encode_labels:
                self.params.append(self.W_encoder_label)
                self.param_shapes.append((d_e, n_classes))
                self.encoder_params.extend([self.W_encoder_label])
                self.encoder_param_shapes.extend([(d_e, n_classes)])
        self.label_only_params = [self.W_decoder_label]
        self.label_only_param_shapes = [(d_v, n_classes)]

        # add encoder parameters depending on number of layers
        if self.n_encoder_layers > 1:
            self.params.extend([self.W_encoder_2, self.b_encoder_2])
            self.param_shapes.extend([(d_e, d_e), (d_e,)])
            self.encoder_params.extend([self.W_encoder_2, self.b_encoder_2])
            self.encoder_param_shapes.extend([(d_e, d_e), (d_e,)])
        if self.encoder_shortcut:
            self.params.extend([self.W_encoder_shortcut])
            self.param_shapes.extend([(d_e, d_v)])
            self.encoder_params.extend([self.W_encoder_shortcut])
            self.encoder_param_shapes.extend([(d_e, d_v)])

        # add generator parameters depending on number of layers
        if self.n_generator_layers > 0:
            self.params.extend([self.W_generator_1, self.b_generator_1])
            self.param_shapes.extend([(d_t, d_t), (d_t,)])
            self.generator_params.extend([self.W_generator_1, self.b_generator_1])
            self.generator_param_shapes.extend([(d_t, d_t), (d_t,)])

        if self.n_generator_layers > 1:
            self.params.extend([self.W_generator_2, self.b_generator_2])
            self.param_shapes.extend([(d_t, d_t), (d_t,)])
            self.generator_params.extend([self.W_generator_2, self.b_generator_2])
            self.generator_param_shapes.extend([(d_t, d_t), (d_t,)])

        if self.n_generator_layers > 2:
            self.params.extend([self.W_generator_3, self.b_generator_3, self.W_generator_4, self.b_generator_4])
            self.param_shapes.extend([(d_t, d_t), (d_t,), (d_t, d_t), (d_t,)])
            self.generator_params.extend([self.W_generator_3, self.b_generator_3, self.W_generator_4, self.b_generator_4])
            self.generator_param_shapes.extend([(d_t, d_t), (d_t,), (d_t, d_t), (d_t,)])

        # declare variables that will be given as inputs to functions to be declared below
        x = T.vector('x', dtype=theano.config.floatX)  # normalized vector of counts for one item
        y = T.vector('y', dtype=theano.config.floatX)  # vector of labels for one item
        indices = T.ivector('x')  # vector of vocab indices (easier to evaluate log prob)
        lr = T.fscalar('lr')  # learning rate
        l1_strength = T.fscalar('l1_strength')  # l1_strength
        kl_strength = T.fscalar('kl_strength')  # l1_strength

        n_words = T.shape(indices)
        # the two variables below are just for debugging
        n_words_print = theano.printing.Print('n_words')(T.shape(indices)[0])  # for debugging
        x_sum = theano.printing.Print('x_sum')(T.sum(x))  # for debugging

        # encode one item to mean and variance vectors
        mu, log_sigma_sq = self.encoder(x, y)

        # take a random sample from the corresponding multivariate normal
        h = self.sampler(mu, log_sigma_sq, th_rng)

        # compute the KL divergence from the prior
        KLD = -0.5 * T.sum(1 + log_sigma_sq - T.square(mu) - T.exp(log_sigma_sq))

        # generate a document representation of dimensionality == n_topics
        r = self.generator(h)

        # decode back into a distribution over the vocabulary
        p_x_given_h = self.decoder(r, y)

        # evaluate the likelihood
        nll_term = -T.sum(T.log(p_x_given_h[T.zeros(n_words, dtype='int32'), indices]) + 1e-32)

        # compute the loss
        loss = nll_term + KLD * kl_strength

        # add an L1 penalty to the decoder terms
        if time_penalty and n_classes > 1:
            penalty = common_theano.col_diff_L1(l1_strength, self.W_decoder_label, n_classes)
        else:
            penalty = common_theano.L1(l1_strength, self.W_decoder)
            if n_classes > 1:
                penalty += common_theano.L1(l1_strength, self.W_decoder_label)
                if use_interactions:
                    penalty += common_theano.L1(l1_strength * self.l1_inter_factor, self.W_decoder_inter)

        # declare some alternate function for decoding from the mean
        r_mu = self.generator(mu)
        p_x_given_x = self.decoder(r_mu, y)
        nll_term_mu = -T.sum(T.log(p_x_given_x[T.zeros(n_words, dtype='int32'), indices]) + 1e-32)
        
        # declare some alternate functions for pretraining from a fixed document representation (r)
        pretrain_r = T.vector('pretrain_r', dtype=theano.config.floatX)
        p_x_given_pretrain_h = self.decoder(pretrain_r, y)
        pretrain_loss = -T.sum(T.log(p_x_given_pretrain_h[T.zeros(n_words, dtype='int32'), indices]) + 1e-32)

        # declare some alternate functions for only using labels
        p_x_given_y_only = self.decoder_label_only(y)
        nll_term_y_only = -T.sum(T.log(p_x_given_y_only[T.zeros(n_words, dtype='int32'), indices]) + 1e-32)

        # compute gradients
        gradients = [T.cast(T.grad(loss + penalty, param, disconnected_inputs='warn'), dtype=theano.config.floatX) for param in self.params]
        encoder_gradients = [T.cast(T.grad(loss, param, disconnected_inputs='warn'), dtype=theano.config.floatX) for param in self.encoder_params]
        generator_gradients = [T.cast(T.grad(loss, param, disconnected_inputs='warn'), dtype=theano.config.floatX) for param in self.generator_params]
        decoder_gradients = [T.cast(T.grad(loss + penalty, param, disconnected_inputs='warn'), dtype=theano.config.floatX) for param in self.decoder_params]
        pretrain_gradients = [T.cast(T.grad(pretrain_loss + penalty, param, disconnected_inputs='warn'), dtype=theano.config.floatX) for param in self.decoder_params]
        label_only_gradients = [T.cast(T.grad(nll_term_y_only + penalty, param, disconnected_inputs='warn'), dtype=theano.config.floatX) for param in self.label_only_params]

        # optionally clip gradients
        if clip_gradients:
            gradients = common_theano.clip_gradients(gradients, 5)
            encoder_gradients = common_theano.clip_gradients(encoder_gradients, 5)
            generator_gradients = common_theano.clip_gradients(generator_gradients, 5)
            decoder_gradients = common_theano.clip_gradients(decoder_gradients, 5)
            pretrain_gradients = common_theano.clip_gradients(pretrain_gradients, 5)
            label_only_gradients = common_theano.clip_gradients(label_only_gradients, 5)

        # create the updates for various sets of parameters
        updates = optimizer(self.params, self.param_shapes, gradients, lr, optimizer_args)
        encoder_updates = optimizer(self.encoder_params, self.encoder_param_shapes, encoder_gradients, lr, optimizer_args)
        generator_updates = optimizer(self.generator_params, self.generator_param_shapes, generator_gradients, lr, optimizer_args)
        decoder_updates = optimizer(self.decoder_params, self.decoder_param_shapes, decoder_gradients, lr, optimizer_args)
        other_updates = optimizer(self.encoder_params + self.generator_params, self.encoder_param_shapes + self.generator_param_shapes, encoder_gradients + generator_gradients, lr, optimizer_args)
        pretrain_updates = optimizer(self.decoder_params, self.decoder_param_shapes, pretrain_gradients, lr, optimizer_args)
        label_only_updates = optimizer(self.label_only_params, self.label_only_param_shapes, label_only_gradients, lr, optimizer_args)

        # declare the available methods for this class
        self.test_input = theano.function(inputs=[x, indices], outputs=[n_words_print, x_sum])
        self.train = theano.function(inputs=[x, indices, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD, penalty], updates=updates, on_unused_input='ignore')
        self.train_encoder = theano.function(inputs=[x, indices, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD, penalty], updates=encoder_updates, on_unused_input='ignore')
        self.train_generator = theano.function(inputs=[x, indices, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD, penalty], updates=generator_updates, on_unused_input='ignore')
        self.train_decoder = theano.function(inputs=[x, indices, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD, penalty], updates=decoder_updates, on_unused_input='ignore')
        self.train_not_decoder = theano.function(inputs=[x, indices, y, lr, l1_strength, kl_strength], outputs=[nll_term, KLD, penalty], updates=other_updates, on_unused_input='ignore')
        self.pretrain_decoder = theano.function(inputs=[indices, y, pretrain_r, lr, l1_strength, kl_strength], outputs=[pretrain_loss], updates=pretrain_updates, on_unused_input='ignore')
        self.encode = theano.function(inputs=[x, y], outputs=[mu, log_sigma_sq], on_unused_input='ignore')
        self.decode = theano.function(inputs=[pretrain_r, y], outputs=[p_x_given_pretrain_h], on_unused_input='ignore')
        self.sample = theano.function(inputs=[x, y], outputs=h, on_unused_input='ignore')
        self.get_mean_doc_rep = theano.function(inputs=[x, y], outputs=r_mu, on_unused_input='ignore')
        self.encode_and_decode = theano.function(inputs=[x, y], outputs=p_x_given_x, on_unused_input='ignore')
        self.neg_log_likelihood = theano.function(inputs=[x, indices, y], outputs=[nll_term, KLD], on_unused_input='ignore')
        self.neg_log_likelihood_mu = theano.function(inputs=[x, indices, y], outputs=[nll_term_mu, KLD], on_unused_input='ignore')
        self.train_label_only = theano.function(inputs=[indices, y, lr, l1_strength], outputs=[nll_term_y_only, penalty], updates=label_only_updates)
        self.neg_log_likelihood_label_only = theano.function(inputs=[indices, y], outputs=nll_term_y_only)

    def encoder(self, x, y):
        if self.n_encoder_layers == 1:
            temp = T.dot(self.W_encoder_1, x) + self.b_encoder_1
            if self.n_classes > 1 and self.encode_labels:
                temp += T.dot(self.W_encoder_label, y)
            pi = T.nnet.relu(temp)
            if self.encoder_shortcut:
                pi += T.dot(self.W_encoder_shortcut, x)
        else:
            temp = T.dot(self.W_encoder_1, x) + self.b_encoder_1
            if self.n_classes > 1 and self.encode_labels:
                temp += T.dot(self.W_encoder_label, y)
            temp2 = T.nnet.relu(temp)
            pi = T.nnet.relu(T.dot(self.W_encoder_2, temp2) + self.b_encoder_2)
            if self.encoder_shortcut:
                pi += T.dot(self.W_encoder_shortcut, x)

        mu = T.dot(self.W_mu, pi) + self.b_mu
        log_sigma_sq = T.dot(self.W_sigma, pi) + self.b_sigma
        return mu, log_sigma_sq

    def sampler(self, mu, log_sigma_sq, theano_rng):
        eps = theano_rng.normal(mu.shape)
        sigma = T.sqrt(T.exp(log_sigma_sq))
        h = mu + sigma * eps
        return h

    def generator(self, h):
        if self.n_generator_layers == 0:
            r = h
        elif self.n_generator_layers == 1:
            temp = T.dot(self.W_generator_1, h) + self.b_generator_1
            if self.generator_shortcut:
                r = T.tanh(temp) + h
            else:
                r = temp
        elif self.n_generator_layers == 2:
            temp = T.tanh(T.dot(self.W_generator_1, h) + self.b_generator_1)
            temp2 = T.dot(self.W_generator_2, temp) + self.b_generator_2
            if self.generator_shortcut:
                r = T.tanh(temp2) + h
            else:
                r = temp2
        else:
            temp = T.tanh(T.dot(self.W_generator_1, h) + self.b_generator_1)
            temp2 = T.tanh(T.dot(self.W_generator_2, temp) + self.b_generator_2)
            temp3 = T.tanh(T.dot(self.W_generator_3, temp2) + self.b_generator_3)
            temp4 = T.dot(self.W_generator_4, temp3) + self.b_generator_4
            if self.generator_shortcut:
                r = T.tanh(temp4) + h
            else:
                r = temp4

        # transform r
        if self.generator_transform == 'softmax':
            return T.nnet.softmax(r)[0]
        elif self.generator_transform == 'tanh':
            return T.tanh(r)
        elif self.generator_transform == 'relu':
            return T.nnet.relu(r)
        else:  # if self.generator_transform is None
            return r

    def decoder(self, r, y):
        temp = T.dot(self.W_decoder, r) + self.b_decoder
        if self.n_classes > 1:
            temp += T.dot(self.W_decoder_label, y)
            if self.use_interactions:
                temp += T.dot(self.W_decoder_inter, T.reshape(T.outer(r, y), (self.d_t * self.n_classes, )))
        p_x_given_h = T.nnet.softmax(temp)
        return p_x_given_h

    def decoder_label_only(self, y):
        temp = self.b_decoder
        temp += T.dot(self.W_decoder_label, y)
        p_x_given_h = T.nnet.softmax(temp)
        return p_x_given_h

    def save_params(self, filename):
        np.savez_compressed(filename,
                            d_v=self.d_v,
                            d_e=self.d_e,
                            d_t=self.d_t,
                            n_classes=self.n_classes,

                            n_encoder_layers=self.n_encoder_layers,
                            n_generator_layers=self.n_generator_layers,
                            generator_transform=self.generator_transform,
                            use_interactions=self.use_interactions,
                            encode_labels=self.encode_labels,
                            encoder_shortcut=self.encoder_shortcut,
                            generator_shortcut=self.generator_shortcut,

                            W_encoder_1=self.W_encoder_1.get_value(),
                            W_encoder_2=self.W_encoder_2.get_value(),
                            W_encoder_label=self.W_encoder_label.get_value(),
                            W_encoder_shortcut=self.W_encoder_shortcut.get_value(),

                            W_mu=self.W_mu.get_value(),
                            W_sigma=self.W_sigma.get_value(),

                            W_generator_1=self.W_generator_1.get_value(),
                            W_generator_2=self.W_generator_2.get_value(),
                            W_generator_3=self.W_generator_3.get_value(),
                            W_generator_4=self.W_generator_4.get_value(),


                            W_decoder=self.W_decoder.get_value(),
                            W_decoder_label=self.W_decoder_label.get_value(),
                            W_decoder_inter=self.W_decoder_inter.get_value(),

                            b_encoder_1=self.b_encoder_1.get_value(),
                            b_encoder_2=self.b_encoder_2.get_value(),
                            b_mu=self.b_mu.get_value(),
                            b_sigma=self.b_sigma.get_value(),
                            b_generator_1=self.b_generator_1.get_value(),
                            b_generator_2=self.b_generator_2.get_value(),
                            b_generator_3=self.b_generator_3.get_value(),
                            b_generator_4=self.b_generator_4.get_value(),
                            b_decoder=self.b_decoder.get_value(),)

    def load_params(self, filename):
        # load parameters
        params = np.load(filename)
        # load the rest of the parameters
        self.d_e = params['d_e']
        self.d_t = params['d_t']
        self.n_classes = params['n_classes']

        self.n_encoder_layers = params['n_encoder_layers']
        self.n_generator_layers = params['n_generator_layers']
        self.generator_transform = params['generator_transform']
        self.use_interactions = params['use_interactions']
        self.encode_labels = params['encode_labels']
        self.encoder_shortcut = params['encoder_shortcut']
        self.generator_shortcut = params['generator_shortcut']

        self.W_encoder_1.set_value(params['W_encoder_1'])
        self.W_encoder_2.set_value(params['W_encoder_2'])
        self.W_encoder_label.set_value(params['W_encoder_label'])
        self.W_encoder_shortcut.set_value(params['W_encoder_shortcut'])

        self.W_mu.set_value(params['W_mu'])
        self.W_sigma.set_value(params['W_sigma'])

        self.W_generator_1.set_value(params['W_generator_1'])
        self.W_generator_2.set_value(params['W_generator_2'])
        self.W_generator_3.set_value(params['W_generator_3'])
        self.W_generator_4.set_value(params['W_generator_4'])

        self.W_decoder.set_value(params['W_decoder'])
        self.W_decoder_label.set_value(params['W_decoder_label'])
        self.W_decoder_inter.set_value(params['W_decoder_inter'])

        self.b_encoder_1.set_value(params['b_encoder_1'])
        self.b_encoder_2.set_value(params['b_encoder_1'])
        self.b_mu.set_value(params['b_mu'])
        self.b_sigma.set_value(params['b_sigma'])
        self.b_generator_1.set_value(params['b_generator_1'])
        self.b_generator_2.set_value(params['b_generator_2'])
        self.b_generator_3.set_value(params['b_generator_3'])
        self.b_generator_4.set_value(params['b_generator_4'])
        self.b_decoder.set_value(params['b_decoder'])
