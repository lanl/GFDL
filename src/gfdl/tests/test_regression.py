import numpy as np
import pytest
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


@pytest.mark.parametrize("""n_samples,
                            n_targets,
                            hidden_layer_sizes,
                            activation,
                            weight_scheme,
                            reg_alpha,
                            exp_preds_shape,
                            exp_preds_median,
                            exp_preds_min""", [
    # expected values are from the graforvfl library
    (100, 10, (100,), "relu", "glorot_normal", 10, (25, 10),
     -29.31478018, -490.5751822),
    (100, 10, (100,), "tanh", "uniform", 100, (25, 10),
     -34.613165002, -327.82362807),
])
def test_regression_against_grafo(n_samples, n_targets, hidden_layer_sizes,
                                  activation, weight_scheme, reg_alpha,
                                  exp_preds_shape, exp_preds_median,
                                  exp_preds_min):
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

    model = GFDLRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            weight_scheme=weight_scheme,
            direct_links=1,
            seed=RNG,
            reg_alpha=reg_alpha
        )
    model.fit(X_train_s, y_train)
    actual_preds = model.predict(X_test_s)
    actual_preds_shape = actual_preds.shape
    actual_preds_median = np.median(actual_preds)
    actual_preds_min = actual_preds.min()
    np.testing.assert_allclose(actual_preds_shape, exp_preds_shape)
    np.testing.assert_allclose(actual_preds_median, exp_preds_median)
    np.testing.assert_allclose(actual_preds_min, exp_preds_min)


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
