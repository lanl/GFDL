# tests/test_model.py
import sys

import graforvfl
import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")

from rvfl.model import RVFL

activations = ["relu", "tanh", "sigmoid", "identity"]
weights = ["zeros", "uniform", "range"]


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("n_hidden", [10])
@pytest.mark.parametrize("direct_links", [0, 1])
@pytest.mark.parametrize("activation", activations)
@pytest.mark.parametrize("weight_scheme", weights)
def test_model(n_hidden, activation, weight_scheme, direct_links, seed):
    N, d, c = 60, 5, 4
    X, y = make_classification(n_samples=N,
                               n_features=d,
                               n_classes=c,
                               n_informative=3,
                               random_state=42)

    model = RVFL(n_hidden, activation, weight_scheme, direct_links, seed)

    model.fit(X, y)
    assert model.W.shape == (d, n_hidden)
    assert model.b.shape == (n_hidden,)
    if direct_links:
        assert model.beta.shape == (n_hidden + d, len(np.arange(c)))
    else:
        assert model.beta.shape == (n_hidden, len(np.arange(c)))

    pred = model.predict(X[:10])
    assert set(np.unique(pred)).issubset(set(np.arange(c)))
    np.testing.assert_array_equal(np.unique(y), np.arange(c))

    P = model.predict_proba(X[:10])
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-6)
    assert (P >= 0).all() and (P <= 1).all()
    np.testing.assert_array_equal(pred, model.classes[np.argmax(P, axis=1)])


def test_invalid_activation_weight():
    with pytest.raises(ValueError, match="is not supported"):
        RVFL(100, "bogus_activation", "random_normal", 0, 0)
    with pytest.raises(ValueError, match="is not supported"):
        RVFL(100, "identity", "bogus_weight", 0, 0)


@pytest.mark.parametrize("n_enhancement", [10, 100])
@pytest.mark.parametrize("n_classes", [2, 5])
def test_classification_against_grafo(n_enhancement, n_classes):
    # test binary and multi-class classification against
    # the open source graforvfl library on some synthetic
    # datasets
    X, y = make_classification(n_classes=n_classes,
                               n_informative=8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=0)
    model = RVFL(n_hidden=n_enhancement,
                 activation="identity",
                 weight_scheme="uniform",
                 direct_links=1,
                 seed=0)
    model.fit(X_train, y_train)
    grafo_rvfl = graforvfl.RvflClassifier(size_hidden=n_enhancement,
                                          act_name="none",
                                          weight_initializer="random_uniform",
                                          reg_alpha=None,
                                          seed=0)
    grafo_rvfl.fit(X_train, y_train)
    actual_proba = model.predict_proba(X_test)
    expected_proba = grafo_rvfl.predict_proba(X_test)
    np.testing.assert_allclose(actual_proba, expected_proba)
