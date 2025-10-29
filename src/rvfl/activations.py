# rvfl/activations.py
import numpy as np


def relu(z):
    return np.maximum(0, z)


def tanh(z):
    return np.tanh(z)


def sigmoid(z):
    out = 1 / (1 + np.exp(-z))
    return out


def identity(z):
    return z


def softmax(z):
    # softmax over each sample
    ez = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return ez / np.sum(ez, axis=-1, keepdims=True)


def softmin(z):
    return softmax(-z)


def log_sigmoid(z):
    return -np.logaddexp(0.0, -z)


def log_softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    logsumexp = np.log(np.sum(np.exp(z), axis=-1, keepdims=True))
    return z - logsumexp


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
