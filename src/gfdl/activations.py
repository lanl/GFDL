"""
Supported activations functions for Gradient Free Deep Learning  estimators.
"""

import numpy as np
import scipy


def relu(z):
    """
    The rectified linear unit activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    numpy.ndarray
        The output array with only positive values.
    """
    return np.maximum(0, z)


def tanh(z):
    """
    The hyperbolic tangent activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    numpy.ndarray
        The output array with hyperbolic tangent values.

    See Also
    --------
    numpy.tanh : The hyperbolic tangent function.
    """
    return np.tanh(z)


def sigmoid(z):
    """
    The logistic sigmoid activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    numpy.ndarray
        The output array with the function values.

    See Also
    --------
    scipy.special.expit : The logistic sigmoid function.
    """
    return scipy.special.expit(z)


def identity(z):
    """
    The identity activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    numpy.ndarray
        The output array same as input values.
    """
    return z


def softmax(z):
    """
    The softmax activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    numpy.ndarray
        The output array with the function values.

    See Also
    --------
    scipy.special.softmax : The softmax function.
    """
    return scipy.special.softmax(z, axis=-1)


def softmin(z):
    """
    The softmin activation function.

    It is the softmax function applied to negative of the input values.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    numpy.ndarray
        The output array with the function values.

    See Also
    --------
    scipy.special.softmax : The softmax function.
    """
    return softmax(-z)


def log_sigmoid(z):
    """
    The logarithm of logistic sigmoid activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    numpy.ndarray
        The output array with the function values.

    See Also
    --------
    scipy.special.log_expit : The logistic sigmoid function.
    """
    return scipy.special.log_expit(z)


def log_softmax(z):
    """
    The log softmax activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    numpy.ndarray
        The output array with the function values.

    See Also
    --------
    scipy.special.log_softmax : The logarithm of softmax function.
    """
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
    # numpydoc ignore=GL08
    name = activation.strip().lower()
    try:
        fn = ACTIVATIONS[name]
    except KeyError as e:
        allowed = sorted(ACTIVATIONS.keys())
        raise ValueError(
            f"activation='{activation}' is not supported; choose from {allowed}"
        ) from e
    return name, fn
