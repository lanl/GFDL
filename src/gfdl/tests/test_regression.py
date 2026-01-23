import numpy as np
import pytest
from graforvfl import RvflRegressor
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
        "GrafoGFDL": RvflRegressor(
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

    # Compare GrafoGFDL and GFDL results
    grf_results = preds["GrafoGFDL"]
    cur_results = preds["GFDL"]

    # Test results
    np.testing.assert_allclose(cur_results, grf_results, atol=1e-07)


@parametrize_with_checks([GFDLRegressor()])
def test_sklearn_api_conformance(estimator, check):
    check(estimator)


def test_regression_boston():
    # real-world data test with multi-layer GFDL
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
            reg_alpha=0.1,
        )
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    # RandomForestRegressor() with default params scores
    # 0.8733907 here; multi-layer GFDL with above params is a bit
    # worse, but certainly better than random chance:
    actual = r2_score(y_test, y_pred)
    np.testing.assert_allclose(actual, 0.78550376)
