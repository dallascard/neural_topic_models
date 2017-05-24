import numpy as np
import theano
from theano import tensor as T


def sgd(shared_params, param_shapes, gradients, lr, extra_args=None):
    """
    Stochastic gradient descent

    Args:
        shared_params: the (theano shared memory) parameters to be updated
        param_shapes: the shapes of shared_params (not used by this optimization function; just a placeholder)
        gradients: the gradients wrt each parameter
        lr: the current global learning rate (float32)
        extra_args: None

    Returns:
        the resulting updates
    """
    updates = [(p, p - lr * g) for p, g in zip(shared_params, gradients)]
    return updates


def sgdm(shared_params, param_shapes, gradients, lr, extra_args=0.9):
    """
    Stochastic gradient descent with momentum

    Args:
        shared_params: the (theano shared memory) parameters to be updated
        param_shapes: the shapes of shared_params
        gradients: the gradients wrt each parameter
        lr: the initial learning rate
        extra_args: [
            momentum: the momenutem value
            ]

    Returns:
        the resulting updates
    """
    momentum = extra_args

    velocity = [
        theano.shared(
            value=np.zeros(p_shape, dtype=theano.config.floatX),
            borrow=True,
            name="momentum:" + param.name
        )
        for p_shape, param in zip(param_shapes, shared_params)
        ]

    velocity_t = [momentum * v + lr * g for v, g in zip(velocity, gradients)]
    velocity_updates = [(v, v_t) for v, v_t in zip(velocity, velocity_t)]
    param_updates = [(param, param - v_t) for param, v_t in zip(shared_params, velocity_t)]
    updates = velocity_updates + param_updates
    return updates


def adagrad(shared_params, param_shapes, gradients, lr, extra_args=0.00001):
    """
    adagrad

    Args:
        shared_params: the (theano shared memory) parameters to be updated
        param_shapes: the shapes of shared_params
        gradients: the gradients wrt each parameter
        lr: the initial learning rate
        extra_args: [
            epsilon: small number to avoid division by zero
            ]

    Returns:
        the resulting updates
    """

    epsilon = extra_args

    grad_histories = [
        theano.shared(
            value=np.zeros(param_shape, dtype=theano.config.floatX),
            borrow=True,
            name="grad_hist:" + param.name
        )
        for param_shape, param in zip(param_shapes, shared_params)
        ]

    new_grad_histories = [g_hist + T.sqr(g) for g_hist, g in zip(grad_histories, gradients)]
    grad_history_updates = list(zip(grad_histories, new_grad_histories))

    param_updates = [(param, param - (lr / (T.sqrt(g_hist) + epsilon)) * param_grad)
                     for param, param_grad, g_hist in zip(shared_params, gradients, new_grad_histories)]

    updates = grad_history_updates + param_updates

    return updates

