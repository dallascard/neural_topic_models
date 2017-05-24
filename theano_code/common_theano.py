import numpy as np
import theano
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from optimization import sgd, sgdm, adagrad

transforms = {'tanh': 1, 'relu': 2, 'softmax': 3}

def get_rngs(seed):
    assert seed > 0
    # initialize random and np.random with seed
    np.random.seed(seed)
    np_rng = np.random.RandomState(np.random.randint(2 ** 30))
    theano_rng = RandomStreams(np.random.randint(2 ** 30))
    return np_rng, theano_rng


def init_param(name, shape, np_rng, scale=6.0, borrow=True, values=None):
    size = np.sum(shape, dtype=float)
    if values is None:
        values = np.asarray(
            np_rng.uniform(
                low=-np.sqrt(scale / size),
                high=np.sqrt(scale / size),
                size=shape
            ),
            dtype=theano.config.floatX
        )
    else:
        values = values.astype(theano.config.floatX)
    return theano.shared(value=values, name=name, borrow=borrow)


def feed_forward(activation, weights, bias, input_):
    return T.cast(activation(T.dot(input_, weights) + bias), dtype=theano.config.floatX)


def feed_forward_no_bias(activation, weights, input_):
    return T.cast(activation(T.dot(input_, weights)), dtype=theano.config.floatX)


def L1(L1_reg, param):
    return L1_reg * (abs(param).sum())


def L2(L2_reg, param):
    return L2_reg * ((param ** 2).sum())


def col_diff_L1(L1_reg, params, n_cols, offset=1):
    diff = params[:, :n_cols - offset] - params[:, offset:n_cols]
    return L1_reg * (abs(diff).sum())


def clip_gradients(gradients, clip):
    """
    If clip > 0, clip the gradients to be within [-clip, clip]

    Args:
        gradients: the gradients to be clipped
        clip: the value defining the clipping interval

    Returns:
        the clipped gradients
    """
    if T.gt(clip, 0):
        gradients = [T.clip(g, -clip, clip) for g in gradients]
    return gradients


def get_optimizer(name, momentum=0.9, epsilon=0.00001, rho=0.95):
    if name == 'sgdm':
        print("Using SGD with momentum =", momentum)
        f = sgdm
        extra_args = momentum
    elif name == 'adagrad':
        print("Using adagrad with epsilon =", epsilon)
        f = adagrad
        extra_args = epsilon
    elif name == 'adadelta':
        print("Using adadelta")
        f = adadelta
        extra_args = (epsilon, rho)
    elif name == 'adam':
        print("Using adam")
        f = adam
        b1, b2, e, gamma = (0.9, 0.999, 1e-8, 1.0-1e-8)
        extra_args = (b1, b2, e, gamma)
    elif name == 'rmsprop':
        print("Using RMSprop")
        f = rmsprop
        extra_args = None
    else:  # sgd
        print("Using vanilla SGD")
        f = sgd
        extra_args = None

    return f, extra_args


