# tests/test_model.py

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks
from ucimlrepo import fetch_ucirepo

from rvfl.model import EnsembleRVFLClassifier, RVFLClassifier

activations = ["relu", "tanh", "sigmoid", "identity", "softmax", "softmin",
               "log_sigmoid", "log_softmax"]
weights = ["zeros", "range", "uniform", "normal", "he_uniform", "lecun_uniform",
           "glorot_uniform", "he_normal", "lecun_normal", "glorot_normal"]


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

    model = RVFLClassifier(hidden_layer_sizes, activation, weight_scheme,
                           direct_links, 0)

    model.fit(X, y)

    assert len(model.W_) == len(hidden_layer_sizes)
    assert model.W_[0].T.shape == (d, hidden_layer_sizes[0])

    for layer, w, b, i in zip(
        hidden_layer_sizes[1:],
        model.W_[1:],
        model.b_[1:],
        range(len(model.W_) - 1), strict=False
        ):
        assert w.T.shape == (hidden_layer_sizes[i], layer)
        assert b.shape == (layer,)

    if direct_links:
        assert model.coeff_.shape == (
            sum(hidden_layer_sizes) + d, len(np.arange(n_classes))
            )
    else:
        assert model.coeff_.shape == (sum(hidden_layer_sizes),
                                      len(np.arange(n_classes)))

    pred = model.predict(X[:10])
    assert set(np.unique(pred)).issubset(set(np.arange(n_classes)))
    np.testing.assert_array_equal(np.unique(y), np.arange(n_classes))

    P = model.predict_proba(X[:10])
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P >= 0).all() and (P <= 1).all()
    np.testing.assert_array_equal(pred, model.classes_[np.argmax(P, axis=1)])


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

    model = RVFLClassifier(
        hidden_layer_sizes=hidden_layer_size,
        activation="identity",
        weight_scheme=weight_scheme,
        direct_links=False,
        seed=0
        )

    model.fit(X, y)

    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    Y = enc.fit_transform(y.reshape(-1, 1))

    # collapsing weights and biases for representation as linear operation
    weights = [w.T for w in model.W_]
    Ts, cs = [], []
    T = np.eye(X.shape[1])
    c = np.zeros((X.shape[1],))

    for w, b in zip(weights, model.b_, strict=False):
        T = T @ w
        c = c @ w + b
        Ts.append(T)
        cs.append(c)

    # design matrix with ALL layers concatenated
    expected_phi = np.hstack([X @ T_l + c_l for T_l, c_l in zip(Ts, cs, strict=False)])

    expected_beta = np.linalg.pinv(expected_phi) @ Y

    np.testing.assert_allclose(model.coeff_, expected_beta)


@pytest.mark.parametrize("hidden_layer_sizes, activation, weight_scheme, exp_auc", [
    # when direct links are absent (ELM), we expect the
    # ROC AUC to increase with multi-layer network complexity
    # up to a reasonable degree, when the width of the layers is
    # quite small
     ((2,), "relu", "uniform", 0.5598328634285958),
     ((2, 2), "relu", "uniform", 0.5666639967533855),
     # start hitting diminishing returns here:
     ((2, 2, 2, 2), "relu", "uniform", 0.5666639967533855),
     ((2, 2, 2, 2, 2, 2, 2), "relu", "uniform", 0.5666639967533855),
     # effectively no improvement here:
     ((2, 2, 2, 2, 2, 2, 2, 2, 2), "relu", "uniform", 0.5666639967533855),
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
    model = RVFLClassifier(
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


@pytest.mark.parametrize(
        "Classifier, target",
        [(RVFLClassifier, 0.7161), (EnsembleRVFLClassifier, 0.7132)]
        )
def test_against_shi2021(Classifier, target):
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

    # values determined using method outlined above
    hidden_layer_sizes = [512, 512]
    reg = 16

    model = Classifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation="relu",
        weight_scheme="uniform",
        reg_alpha=reg,
        seed=0
        )

    scl = StandardScaler()

    # The actual splits used in the paper were not specified
    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    acc = 0
    for train_index, test_index in skf.split(X_eval, y_eval):
        X_train = X_eval[train_index]
        y_train = y_eval[train_index]
        X_test = X_eval[test_index]
        y_test = y_eval[test_index]

        model.fit(scl.fit_transform(X_train), y_train)

        y_hat = model.predict(scl.transform(X_test))

        acc += accuracy_score(y_test, y_hat)

    acc /= K

    # not an exact match because they don't specify their activation
    # nor do they mention the best hyperparameter configuration
    # and they're using ridge

    # tightest bound for both rel and abs
    # values in paper:
    # dRVFL accuracy: 66.33%
    # edRVFL accuracy: 65.81%
    assert acc == pytest.approx(target, rel=1e-4, abs=0)


def test_soft_and_hard():
    N, d = 60, 10
    X, y = make_classification(n_samples=N,
                               n_features=d,
                               n_classes=3,
                               n_informative=8,
                               random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

    model = EnsembleRVFLClassifier(
        hidden_layer_sizes=(5, 5, 5),
        activation="tanh",
        weight_scheme="uniform",
        seed=0,
        reg_alpha=0.1
    )
    model.fit(X_train, y_train)

    y_soft = model.predict(X_test)

    P = model.predict_proba(X_test)
    y_from_mean = model.classes_[np.argmax(P, axis=1)]
    np.testing.assert_equal(y_soft, y_from_mean)

    model.voting = "hard"
    y_hard = model.predict(X_test)

    np.testing.assert_equal(y_soft, y_hard)


def test_hard_vote_proba_error():
    X, y = make_classification(n_samples=60,
                               n_features=10,
                               n_classes=3,
                               n_informative=8,
                               random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    model = EnsembleRVFLClassifier(
        hidden_layer_sizes=(5, 5, 5),
        activation="tanh",
        weight_scheme="uniform",
        seed=0,
        reg_alpha=0.1,
        voting="hard",
    )
    model.fit(X_train, y_train)
    with pytest.raises(AttributeError, match="predict_proba"):
        model.predict_proba(X_test)


@pytest.mark.parametrize("alpha", [None, 0.1])
def test_soft_and_hard_can_differ(alpha):
    N, d = 60, 10
    X, y = make_classification(n_samples=N,
                               n_features=d,
                               n_classes=3,
                               n_informative=8,
                               random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)

    # adding more layers (heads) increases the chance of disagreement
    # between the two voting methods
    model = EnsembleRVFLClassifier(
        hidden_layer_sizes=(3, 3, 3, 3),
        activation="tanh",
        weight_scheme="uniform",
        seed=0,
        reg_alpha=alpha
    )
    model.fit(X_train, y_train)
    y_soft = model.predict(X_test)
    model.voting = "hard"
    y_hard = model.predict(X_test)
    difference = [
        True, True, True, True, True, True, True, True, True, True, False, True
        ]

    np.testing.assert_array_equal(y_soft == y_hard, difference)


@pytest.mark.parametrize("Classifier", [RVFLClassifier, EnsembleRVFLClassifier])
def test_invalid_activation_weight(Classifier):
    X = np.zeros((30, 4))
    y = np.zeros((30,))
    invalid_act = Classifier(hidden_layer_sizes=100,
                             activation="bogus_activation",
                             weight_scheme="uniform")
    invalid_weight = Classifier(hidden_layer_sizes=100,
                                activation="identity",
                                weight_scheme="bogus_weight")
    # the sklearn estimator API bans input validation in __init__,
    # so we need to call fit() for error handling to kick in:
    # https://scikit-learn.org/stable/developers/develop.html#developing-scikit-learn-estimators
    with pytest.raises(ValueError, match="is not supported"):
        invalid_act.fit(X, y)
    with pytest.raises(ValueError, match="is not supported"):
        invalid_weight.fit(X, y)


@pytest.mark.parametrize("Classifier", [RVFLClassifier, EnsembleRVFLClassifier])
def test_invalid_alpha(Classifier):
    # the sklearn estimator API bans input validation in __init__,
    # so we need to call fit() for error handling to kick in:
    # https://scikit-learn.org/stable/developers/develop.html#developing-scikit-learn-estimators
    X = np.zeros((30, 4))
    y = np.zeros((30,))
    bad_est = Classifier(hidden_layer_sizes=100,
                             activation="identity",
                             weight_scheme="uniform",
                             reg_alpha=-10)
    with pytest.raises(ValueError, match=r"Negative reg\_alpha"):
        bad_est.fit(X, y)


@pytest.mark.parametrize("""hidden_layer_sizes,
                            n_classes,
                            activation,
                            weight_scheme,
                            alpha,
                            exp_proba_shape,
                            exp_proba_median,
                            exp_proba_min""", [

                    # expected values are from graforvfl library
                    ([10,], 2, "relu", "uniform", None, (20, 2), 0.5, 0.0444571694),
                    ([100,], 2, "tanh", "normal", None, (20, 2), 0.5, 0.02538905725),
                    ([10,], 5, "softmax", "lecun_uniform", None, (20, 5),
                     0.186506112, 0.08469873),
])
def test_classification_against_grafo(hidden_layer_sizes, n_classes, activation,
                                      weight_scheme, alpha, exp_proba_shape,
                                      exp_proba_median, exp_proba_min):
    # test binary and multi-class classification against expected values
    # from the open source graforvfl library on some synthetic
    # datasets
    X, y = make_classification(n_classes=n_classes,
                               n_informative=8, random_state=0)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    model = RVFLClassifier(hidden_layer_sizes=hidden_layer_sizes,
                 activation=activation,
                 weight_scheme=weight_scheme,
                 direct_links=1,
                 seed=0,
                 reg_alpha=alpha)
    model.fit(X_train, y_train)

    actual_proba = model.predict_proba(X_test)
    actual_proba_shape = actual_proba.shape
    actual_proba_median = np.median(actual_proba)
    actual_proba_min = np.min(actual_proba)

    np.testing.assert_allclose(actual_proba_shape, exp_proba_shape)
    np.testing.assert_allclose(actual_proba_median, exp_proba_median)
    np.testing.assert_allclose(actual_proba_min, exp_proba_min)


@parametrize_with_checks([RVFLClassifier(), EnsembleRVFLClassifier()])
def test_sklearn_api_conformance(estimator, check):
    check(estimator)
