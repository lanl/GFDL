import numpy as np
import scipy


def relu(z):
    return np.maximum(0, z)


def tanh(z):
    return np.tanh(z)


def sigmoid(z):
    return scipy.special.expit(z)


def identity(z):
    return z


def softmax(z):
    # softmax over each sample
    return scipy.special.softmax(z, axis=-1)


def softmin(z):
    return softmax(-z)


def log_sigmoid(z):
    return scipy.special.log_expit(z)


def log_softmax(z):
    return scipy.special.log_softmax(z, axis=-1)


ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "identity": identity,
    "linear": identity,
    "softmax": softmax,
    "softmin": softmin,
    "log_sigmoid": log_sigmoid,
    "log_softmax": log_softmax,
}


def resolve_activation(activation):
    name = activation.strip().lower()
    try:
        fn = ACTIVATIONS[name]
    except KeyError as e:
        allowed = sorted(ACTIVATIONS.keys())
        raise ValueError(
            f"activation='{activation}' is not supported; choose from {allowed}"
        ) from e
    return name, fn
