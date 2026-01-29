"""
Weight functions for Gradient Free Deep Learning estimators.
"""

import numpy as np


def zeros(d, h, **kwargs):
    """
    The weight function setting all weights to zero.

    This function is useful to test out the effect of
    the data features in isolation.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    Returns
    -------
    ndarray or scalar
        All zeros.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.
    """
    return np.zeros((h, d))


def uniform(d, h, *, rng, **kwargs):
    """
    The weight function returning samples drawn from uniform distribution.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    rng : np.random.Generator
        A NumPy random number generator instance.

    Returns
    -------
    ndarray or scalar
        Draw samples from the uniform distribution between ``[0, 1)``.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.

    Notes
    -----
    uniform() had to be split out for pickle/serialization
    for conformance with the sklearn estimator API:
    https://scikit-learn.org/stable/developers/develop.html#developing-scikit-learn-estimators
    """
    return rng.uniform(0, 1, (h, d))


def range(d, h, **kwargs):
    """
    The weight function returning samples drawn from discrete uniform distribution.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    Returns
    -------
    ndarray or scalar
        Drawn samples from the discrete uniform distribution ``[0, d*h)``.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.
    """
    s = np.arange(d * h)
    s = np.subtract(s, np.mean(s))
    s /= np.std(s)
    s = np.nan_to_num(s)
    return s.reshape(h, d)


def he_uniform(d, h, *, rng, **kwargs):
    """
    The weight function returning samples drawn from He uniform distribution.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    rng : np.random.Generator
        A NumPy random number generator instance.

    Returns
    -------
    ndarray or scalar
        Draw samples from the He uniform distribution between
        ``[sqrt(6/h), sqrt(6/h))``.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.

    Notes
    -----
    This implementation deviates from the standard expression where
    the number of input features (d) are used to compute the limit.
    https://faroit.com/keras-docs/2.0.0/initializers/#he_uniform
    However, using the standard form returned a different
    answer from GrafoRVFL, which uses the output size i.e. hidden
    layer size instead (from ChatGPT). Needs further exploration
    of why they deviate from the standard form.
    If we choose to use the standard form, then our tests cannot be
    used to compare against GrafoRVFL as the results could be order
    one difference or higher.
    """

    limit = np.sqrt(6 / h)
    return rng.uniform(-limit, limit, (h, d))


def lecun_uniform(d, h, *, rng, **kwargs):
    """
    The weight function returning samples drawn from Lecun uniform distribution.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    rng : np.random.Generator
        A NumPy random number generator instance.

    Returns
    -------
    ndarray or scalar
        Draw samples from the Lecun uniform distribution between
        ``[sqrt(3/h), sqrt(3/h))``.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.

    Notes
    -----
    Same comment as "he_uniform"
    https://faroit.com/keras-docs/2.0.0/initializers/#lecun_uniform
    """

    limit = np.sqrt(3 / h)
    return rng.uniform(-limit, limit, (h, d))


def glorot_uniform(d, h, *, rng, **kwargs):
    """
    The weight function returning samples drawn from Glorot uniform distribution.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    rng : np.random.Generator
        A NumPy random number generator instance.

    Returns
    -------
    ndarray or scalar
        Draw samples from the Glorot uniform distribution between
        ``[-sqrt(6/(d+h)), sqrt(6/(d+h)))``.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.

    Notes
    -----
    https://faroit.com/keras-docs/2.0.0/initializers/#glorot_uniform
    """

    fan_avg = 0.5 * (d + h)
    limit = np.sqrt(3 / fan_avg)
    return rng.uniform(-limit, limit, (h, d))


def normal(d, h, *, rng, **kwargs):
    """
    The weight function returning samples drawn from normal distribution.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    rng : np.random.Generator
        A NumPy random number generator instance.

    Returns
    -------
    ndarray or scalar
        Draw samples from the normal distribution with
        mean ``0`` and standard deviation ``1``.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.
    """
    return rng.normal(0, 1, (h, d))


def he_normal(d, h, *, rng, **kwargs):
    """
    The weight function returning samples drawn from He normal distribution.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    rng : np.random.Generator
        A NumPy random number generator instance.

    Returns
    -------
    ndarray or scalar
        Draw samples from the He normal distribution with
        mean ``0`` and standard deviation ``sqrt(2/d)``.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.

    Notes
    -----
    Same comment as "he_uniform"
    https://faroit.com/keras-docs/2.0.0/initializers/#he_normal
    """

    var = np.sqrt(2 / h)
    return rng.normal(0, var, (h, d))


def lecun_normal(d, h, *, rng, **kwargs):
    """
    The weight function returning samples drawn from Lecun normal distribution.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    rng : np.random.Generator
        A NumPy random number generator instance.

    Returns
    -------
    ndarray or scalar
        Draw samples from the Lecun normal distribution
        with mean ``0`` and standard deviation ``1/sqrt(h)``.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.

    Notes
    -----
    Same comment as "he_uniform"
    https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal
    """

    var = 1 / np.sqrt(h)
    return rng.normal(0, var, (h, d))


def glorot_normal(d, h, *, rng, **kwargs):
    """
    The weight function returning samples drawn from Glorot normal distribution.

    Parameters
    ----------
    d : int
        Number of features in a sample or number of neurons in the previous
        hidden layer.

    h : int
        Number of neurons in the current hidden layer.

    rng : np.random.Generator
        A NumPy random number generator instance.

    Returns
    -------
    ndarray or scalar
        Draw samples from the Glorot normal distribution with
        mean ``0`` and standard deviation ``sqrt(2/(d+h))``.

    Other Parameters
    ----------------
    **kwargs : dict
        Other keyword arguments.

    Notes
    -----
    https://faroit.com/keras-docs/2.0.0/initializers/#glorot_normal
    """

    fan_avg = 0.5 * (d + h)
    var = np.sqrt(1 / fan_avg)
    return rng.normal(0, var, (h, d))


WEIGHTS = {
    "zeros": zeros,
    "uniform": uniform,
    "range": range,
    "normal": normal,
    "he_uniform": he_uniform,
    "lecun_uniform": lecun_uniform,
    "glorot_uniform": glorot_uniform,
    "he_normal": he_normal,
    "lecun_normal": lecun_normal,
    "glorot_normal": glorot_normal,
}


def resolve_weight(weight):
    # numpydoc ignore=GL08
    name = weight.strip().lower()
    try:
        w = WEIGHTS[name]
    except KeyError as e:
        allowed = sorted(WEIGHTS.keys())
        raise ValueError(
            f"weight scheme='{weight}' is not supported; choose from {allowed}"
        ) from e
    return w
