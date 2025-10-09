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


ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh,
    "sigmoid": sigmoid,
    "identity": identity,
    "linear": identity,
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
