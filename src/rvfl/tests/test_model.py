# tests/test_model.py

import graforvfl
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rvfl.model import RVFL

activations = ["relu", "tanh", "sigmoid", "identity"]
weights = ["zeros", "uniform", "range"]


@pytest.mark.parametrize(
        "hidden_layer_sizes",
        [(10,), (10, 10), (5, 10, 15, 20), (100,)]
        )
@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("direct_links", [0, 1])
@pytest.mark.parametrize("activation", activations)
@pytest.mark.parametrize("weight_scheme", weights)
def test_model(hidden_layer_sizes, n_classes, activation, weight_scheme, direct_links):
    N, d = 60, 10
    X, y = make_classification(n_samples=N,
                               n_features=d,
                               n_classes=n_classes,
                               n_informative=8,
                               random_state=42)

    model = RVFL(hidden_layer_sizes, activation, weight_scheme, direct_links, 0)

    model.fit(X, y)

    assert len(model.W) == len(hidden_layer_sizes)
    assert model.W[0].T.shape == (d, hidden_layer_sizes[0])

    for layer, w, b, i in zip(
        hidden_layer_sizes[1:],
        model.W[1:],
        model.b[1:],
        range(len(model.W) - 1), strict=False
        ):
        assert w.T.shape == (hidden_layer_sizes[i], layer)
        assert b.shape == (layer,)

    if direct_links:
        assert model.beta.shape == (
            hidden_layer_sizes[-1] + d, len(np.arange(n_classes))
            )
    else:
        assert model.beta.shape == (hidden_layer_sizes[-1], len(np.arange(n_classes)))

    pred = model.predict(X[:10])
    assert set(np.unique(pred)).issubset(set(np.arange(n_classes)))
    np.testing.assert_array_equal(np.unique(y), np.arange(n_classes))

    P = model.predict_proba(X[:10])
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P >= 0).all() and (P <= 1).all()
    np.testing.assert_array_equal(pred, model.classes[np.argmax(P, axis=1)])


@pytest.mark.parametrize("weight_scheme", weights)
@pytest.mark.parametrize(
        "hidden_layer_size",
        [(10,), (2, 3, 2, 1), (5, 10, 15, 20, 15, 10), (100,)]
        )
def test_multilayer_math(weight_scheme, hidden_layer_size):

    N, d = 60, 10
    X, y = make_classification(n_samples=N,
                               n_features=d,
                               n_classes=3,
                               n_informative=8,
                               random_state=42)

    model = RVFL(
        hidden_layer_sizes=hidden_layer_size,
        activation="identity",
        weight_scheme=weight_scheme,
        direct_links=False,
        seed=0
        )

    model.fit(X, y)

    scl = StandardScaler()
    X = scl.fit_transform(X)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Y = enc.fit_transform(y.reshape(-1, 1))

    # collapsing weights and biases for representation as linear operation
    weights = [w.T for w in model.W]
    T = np.eye(weights[-1].shape[1])
    bias = model.b[-1].copy()

    for i in range(len(model.b) - 2, -1, -1):
        T = weights[i + 1] @ T
        bias += model.b[i] @ T

    weights = np.linalg.multi_dot(weights) if len(weights) > 1 else weights[0]

    expected_phi = (X @ weights) + bias

    expected_beta = np.linalg.pinv(expected_phi) @ Y

    for exp, act in zip(expected_beta, model.beta, strict=False):
        np.testing.assert_allclose(exp, act)


def test_invalid_activation_weight():
    with pytest.raises(ValueError, match="is not supported"):
        RVFL(100, "bogus_activation", "random_normal", 0, 0)
    with pytest.raises(ValueError, match="is not supported"):
        RVFL(100, "identity", "bogus_weight", 0, 0)


@pytest.mark.parametrize("hidden_layer_sizes", [(10,), (100,)])
@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("activation", activations)
def test_classification_against_grafo(hidden_layer_sizes, n_classes, activation):
    # test binary and multi-class classification against
    # the open source graforvfl library on some synthetic
    # datasets
    X, y = make_classification(n_classes=n_classes,
                               n_informative=8)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    model = RVFL(hidden_layer_sizes=hidden_layer_sizes,
                 activation=activation,
                 weight_scheme="uniform",
                 direct_links=1,
                 seed=0)
    model.fit(X_train, y_train)

    scl = StandardScaler()

    grafo_act = "none" if activation == "identity" else activation
    grafo_rvfl = graforvfl.RvflClassifier(size_hidden=hidden_layer_sizes[0],
                                          act_name=grafo_act,
                                          weight_initializer="random_uniform",
                                          reg_alpha=None,
                                          seed=0)

    grafo_rvfl.fit(scl.fit_transform(X_train), y_train)

    actual_proba = model.predict_proba(X_test)
    expected_proba = grafo_rvfl.predict_proba(scl.transform(X_test))

    np.testing.assert_allclose(actual_proba, expected_proba)
