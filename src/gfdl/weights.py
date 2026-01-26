import numpy as np


def zeros(d, h, **kwargs):
    return np.zeros((h, d))


def uniform(d, h, *, rng, **kwargs):
    # NOTE: uniform() had to be split out for pickle/serialization
    # for conformance with the sklearn estimator API:
    # https://scikit-learn.org/stable/developers/develop.html#developing-scikit-learn-estimators
    return rng.uniform(0, 1, (h, d))


def range(d, h, **kwargs):
    s = np.arange(d * h)
    s = np.subtract(s, np.mean(s))
    s /= np.std(s)
    s = np.nan_to_num(s)
    return s.reshape(h, d)


def he_uniform(d, h, *, rng, **kwargs):
    # This implementation deviates from the standard expression where
    # the number of input features (d) are used to compute the limit.
    # https://faroit.com/keras-docs/2.0.0/initializers/#he_uniform
    # However, using the standard form returned a different
    # answer from GrafoRVFL, which uses the output size i.e. hidden
    # layer size instead (from ChatGPT). Needs further exploration
    # of why they deviate from the standard form.
    # If we choose to use the standard form, then our tests cannot be
    # used to compare against GrafoRVFL as the results could be order
    # one difference or higher.
    limit = np.sqrt(6 / h)
    return rng.uniform(-limit, limit, (h, d))


def lecun_uniform(d, h, *, rng, **kwargs):
    # Same comment as "he_uniform"
    # https://faroit.com/keras-docs/2.0.0/initializers/#lecun_uniform
    limit = np.sqrt(3 / h)
    return rng.uniform(-limit, limit, (h, d))


def glorot_uniform(d, h, *, rng, **kwargs):
    # https://faroit.com/keras-docs/2.0.0/initializers/#glorot_uniform
    fan_avg = 0.5 * (d + h)
    limit = np.sqrt(3 / fan_avg)
    return rng.uniform(-limit, limit, (h, d))


def normal(d, h, *, rng, **kwargs):
    return rng.normal(0, 1, (h, d))


def he_normal(d, h, *, rng, **kwargs):
    # Same comment as "he_uniform"
    # https://faroit.com/keras-docs/2.0.0/initializers/#he_normal
    var = np.sqrt(2 / h)
    return rng.normal(0, var, (h, d))


def lecun_normal(d, h, *, rng, **kwargs):
    # Same comment as "he_uniform"
    # https://www.tensorflow.org/api_docs/python/tf/keras/initializers/LecunNormal
    var = 1 / np.sqrt(h)
    return rng.normal(0, var, (h, d))


def glorot_normal(d, h, *, rng, **kwargs):
    # https://faroit.com/keras-docs/2.0.0/initializers/#glorot_normal
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
    name = weight.strip().lower()
    try:
        w = WEIGHTS[name]
    except KeyError as e:
        allowed = sorted(WEIGHTS.keys())
        raise ValueError(
            f"weight scheme='{weight}' is not supported; choose from {allowed}"
        ) from e
    return w
