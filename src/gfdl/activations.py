"""
Activation functions for Gradient Free Deep Learning estimators.
"""

import scipy
from sklearn.utils._array_api import get_namespace_and_device


def relu(z):
    """
    The rectified linear unit activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    array
        The output array with only positive values.
    """
    xp, _, device = get_namespace_and_device(z)
    zero = xp.asarray(0, dtype=z.dtype, device=device)
    return xp.maximum(zero, z)


def tanh(z):
    """
    The hyperbolic tangent activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    array
        The output array with hyperbolic tangent values.

    See Also
    --------
    numpy.tanh : The hyperbolic tangent function.
    """
    xp, _, device = get_namespace_and_device(z)
    return xp.tanh(z)


def sigmoid(z):
    """
    The logistic sigmoid activation function.

    Parameters
    ----------
    z : array_like
        Input array.

    Returns
    -------
    array
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
    array
        The input array is returned unchanged.
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
    array
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
    array
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
    array
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
    array
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
