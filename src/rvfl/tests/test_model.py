# tests/test_model.py

import graforvfl
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

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

    np.testing.assert_allclose(model.beta, expected_beta)


@pytest.mark.parametrize("hidden_layer_sizes, activation, weight_scheme, exp_auc", [
    # when direct links are absent (ELM), we expect the
    # ROC AUC to increase with multi-layer network complexity
    # up to a reasonable degree, when the width of the layers is
    # quite small
     ((2,), "relu", "uniform", 0.516163655),
     ((2, 2), "relu", "uniform", 0.60763808),
     # start hitting diminishing returns here:
     ((2, 2, 2, 2), "relu", "uniform", 0.60891569),
     ((2, 2, 2, 2, 2, 2, 2), "relu", "uniform", 0.609660066),
     # effectively no improvement here:
     ((2, 2, 2, 2, 2, 2, 2, 2, 2), "relu", "uniform", 0.609660066),
     ]
 )
def test_multilayer_progression(weight_scheme,
                                hidden_layer_sizes,
                                activation,
                                exp_auc):
    X, y = make_classification(n_samples=400,
                               n_features=100,
                               n_classes=5,
                               n_informative=26,
                               random_state=42,
                               class_sep=0.5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    model = RVFL(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        weight_scheme=weight_scheme,
        direct_links=False,
        seed=0
        )
    model.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)
    actual_auc = roc_auc_score(y_test, y_score, multi_class="ovo")
    assert_allclose(actual_auc, exp_auc)


def test_against_shi2021():
    # test multilayer classification against
    # the results given in Shi et al. (2021) DOI 10.1016/j.patcog.2021.107978
    # dataset obtained from UCI ML repo
    abalone = fetch_ucirepo(id=1)

    X = abalone.data.features
    y = abalone.data.targets

    X = X.assign(
            Sex=lambda d: d["Sex"].map({"M": 0, "F": 1, "I": 2}).astype("int8")
            )

    X, y = np.array(X), np.array(y).reshape(-1)

    # Shi et al. only use 3 classes, but all samples are used, which implies binning
    # they do not specify the bins used, so I used my best judgement
    y = np.digitize(y, bins=[7, 11])

    # Shi et al. used half of the titanic dataset to tune
    # so I assumed they did the same for this dataset
    X_tune, X_eval, y_tune, y_eval = train_test_split(
        X, y, test_size=0.5, random_state=0)

    # Shi et al. used 4 folds on the titanic dataset
    # I inferred that they also used 4 folds for this dataset
    K = 4

    # X_tune and y_tune were used to find the hyperparameters
    # using Shi et al.'s two-stage tuning method, disregarding
    # parameter C and tuning the activation function instead
    """
    For RVFL based models, we use a two-stage tuning method to obtain
    their best hyperparameter configurations. The two-stage tuning can be
    performed by the following steps: 1) Fix the number layers to 2, and
    then select the optimal number of neurons (N*) and regularization
    parameter (C*) using a coarse range for N and C. 2) Tune the number
    of layers and fine tune the N, C parameters by considering only a fine
    range in the neighborhood of N* and C*.

    Shi et al. (2021) https://doi.org/10.1016/j.patcog.2021.107978
    """
    best_layer, best_neuron, best_act = 3, 512, "tanh"

    # The actual splits used in the paper were not specified
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    acc = 0
    for train_index, test_index in skf.split(X_eval, y_eval):
        X_train = X_eval[train_index]
        y_train = y_eval[train_index]
        X_test = X_eval[test_index]
        y_test = y_eval[test_index]

        hidden_layer_sizes = [best_neuron] * best_layer

        model = RVFL(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=best_act,
            weight_scheme="uniform",
            direct_links=True,
            seed=0
            )
        model.fit(X_train, y_train)

        y_hat = model.predict(X_test)

        acc += accuracy_score(y_test, y_hat)

    acc /= K

    # not an exact match because they don't specify their activation
    # nor do they mention the best hyperparameter configuration
    # and they're using ridge

    # tightest bound for both rel and abs
    assert acc == pytest.approx(.6633, rel=2e-2, abs=2e-2)


def test_invalid_activation_weight():
    with pytest.raises(ValueError, match="is not supported"):
        RVFL(100, "bogus_activation", "random_normal", 0, 0)
    with pytest.raises(ValueError, match="is not supported"):
        RVFL(100, "identity", "bogus_weight", 0, 0)


def test_invalid_alpha():
    with pytest.raises(ValueError, match=r"Negative reg\_alpha"):
        RVFL(100, "identity", "uniform", 0, 0, -10)


@pytest.mark.parametrize("hidden_layer_sizes", [(10,), (100,)])
@pytest.mark.parametrize("n_classes", [2, 5])
@pytest.mark.parametrize("activation", activations)
@pytest.mark.parametrize("alpha", [None, 0.5, 1])
def test_classification_against_grafo(hidden_layer_sizes, n_classes, activation, alpha):
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
                 seed=0,
                 reg_alpha=alpha)
    model.fit(X_train, y_train)

    scl = StandardScaler()

    grafo_act = "none" if activation == "identity" else activation
    grafo_rvfl = graforvfl.RvflClassifier(size_hidden=hidden_layer_sizes[0],
                                          act_name=grafo_act,
                                          weight_initializer="random_uniform",
                                          reg_alpha=alpha,
                                          seed=0)

    grafo_rvfl.fit(scl.fit_transform(X_train), y_train)

    actual_proba = model.predict_proba(X_test)
    expected_proba = grafo_rvfl.predict_proba(scl.transform(X_test))

    np.testing.assert_allclose(actual_proba, expected_proba)
