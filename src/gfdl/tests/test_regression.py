import numpy as np
import pytest
from graforvfl import RvflRegressor
from sklearn.base import clone
from sklearn.datasets import fetch_openml, make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from gfdl.model import GFDLRegressor

activations = ["relu", "tanh", "sigmoid", "identity", "softmax", "softmin",
             "log_sigmoid", "log_softmax"]
weights = ["uniform", "normal", "he_uniform", "lecun_uniform",
         "glorot_uniform", "he_normal", "lecun_normal", "glorot_normal"]


@pytest.mark.parametrize("n_samples", [100, 1000])
@pytest.mark.parametrize("n_targets", [10, 100])
@pytest.mark.parametrize("hidden_layer_sizes", [(100,), (1000,)])
@pytest.mark.parametrize("activation", activations)
@pytest.mark.parametrize("weight_scheme", weights)
@pytest.mark.parametrize("reg_alpha", [1, 10])
def test_regression_against_grafo(n_samples, n_targets, hidden_layer_sizes,
                                  activation, weight_scheme, reg_alpha):
    N, d = n_samples, n_targets
    RNG = 42
    X, y = make_regression(n_samples=N,
                           n_features=d,
                           n_informative=d,
                           n_targets=n_targets,
                           noise=0.0,
                           bias=0.0,
                           random_state=RNG)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.25,
                                                        random_state=RNG)

    # Preprocessing (use the SAME scaler for all models that need it)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    grafo_act = "none" if activation == "identity" else activation
    if weight_scheme == "uniform":
        grafo_wts = "random_uniform"
    elif weight_scheme == "normal":
        grafo_wts = "random_normal"
    else:
        grafo_wts = weight_scheme

    # Define models
    models = {
        "GrafoRVFL": RvflRegressor(
        size_hidden=hidden_layer_sizes[0],
        act_name=grafo_act,
        weight_initializer=grafo_wts,
        reg_alpha=reg_alpha,
        seed=RNG
        ),
        "GFDL": GFDLRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            weight_scheme=weight_scheme,
            direct_links=1,
            seed=RNG,
            reg_alpha=reg_alpha
        )
    }

    # Fit + predict
    preds = {}

    for name, model in models.items():
        Xtr, Xte = (X_train_s, X_test_s)
        model.fit(Xtr, y_train)
        yhat = model.predict(Xte)
        preds[name] = yhat

    # Compare GrafoRVFL and GFDL results
    grf_results = preds["GrafoRVFL"]
    cur_results = preds["GFDL"]

    # Test results
    np.testing.assert_allclose(cur_results, grf_results, atol=1e-07)


@parametrize_with_checks([GFDLRegressor()])
def test_sklearn_api_conformance(estimator, check):
    check(estimator)


@pytest.mark.parametrize("reg_alpha, expected", [
    (0.1, 0.78550376),
    # NOTE: for Moore-Penrose, a large singular value
    # cutoff (rtol) is required to achieve reasonable R2 with
    # the Boston Housing dataset
    (None, 0.73452466),
])
def test_regression_boston(reg_alpha, expected):
    # real-world data test with multi-layer RVFL
    boston = fetch_openml(name="boston", version=1, as_frame=False)
    X, y = boston.data, boston.target.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = GFDLRegressor(
            hidden_layer_sizes=[800] * 10,
            activation="tanh",
            weight_scheme="uniform",
            direct_links=1,
            seed=0,
            reg_alpha=reg_alpha,
            rtol=1e-3,  # has no effect for `Ridge`
        )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    # RandomForestRegressor() with default params scores
    # 0.8733907 here; multi-layer GFDL with above params is a bit
    # worse, but certainly better than random chance:
    actual = r2_score(y_test, y_pred)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("hidden_layer_sizes", [(10,), (5, 5)])
@pytest.mark.parametrize("direct_links", [True, False])
@pytest.mark.parametrize("activation", activations)
@pytest.mark.parametrize("weight_scheme", weights[1:])
@pytest.mark.parametrize("alpha", [None, 0.1, 0.5, 1])
def test_partial_fit_regressor(
    hidden_layer_sizes, direct_links, activation, weight_scheme, alpha
):
    # Test coefficient equivalence between partial_fit() and fit() as long
    # as D.T@D is well-conditioned

    # partial_fit is equivalent to accumulating normal equations
    X, y = make_regression(
        n_samples=600,
        n_features=20,
        n_informative=10,
        random_state=0,
    )

    ff_model = GFDLRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        weight_scheme=weight_scheme,
        direct_links=direct_links,
        seed=0,
        reg_alpha=alpha,
    )
    pf_model = clone(ff_model)

    ff_model.fit(X, y)

    batch = 25
    for start in range(0, len(X), batch):
        end = min(start + batch, len(X))
        Xb = X[start:end]
        yb = y[start:end]
        pf_model.partial_fit(Xb, yb)

    np.testing.assert_allclose(ff_model.coeff_, pf_model.coeff_, rtol=1e-5, atol=4e-5)
