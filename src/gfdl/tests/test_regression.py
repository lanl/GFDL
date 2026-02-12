import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import fetch_openml, make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from gfdl.model import GFDLRegressor


@pytest.mark.parametrize("""n_samples,
                            n_targets,
                            hidden_layer_sizes,
                            activation,
                            weight_scheme,
                            reg_alpha,
                            exp_preds_shape,
                            exp_preds_median,
                            exp_preds_min,
                            exp_preds_r2""", [
    # expected values are from the graforvfl library
    (100, 10, (100,), "relu", "glorot_normal", 10, (25, 10),
     -29.31478018, -490.57518221, 0.97537085),
    (100, 10, (100,), "tanh", "uniform", 1, (25, 10),
     -43.03897314, -504.32794352, 0.98411997),
    (100, 10, (100,), "log_softmax", "uniform", 1, (25, 10),
      -30.56871963218171, -558.1388909597706, 0.9999532782125536),
    (100, 10, (100,), "log_sigmoid", "normal", 10, (25, 10),
      -19.5976250350991, -574.1699708675857, 0.9853855947182326),
    (100, 10, (1000,), "softmin", "he_uniform", 1, (25, 10),
     -57.91870287977487, -589.6707200160679, 0.9656730623177637),
    (100, 10, (1000,), "softmax", "lecun_uniform", 10, (25, 10),
     -51.938696542946786, -513.4094105001416, 0.9589931777194366),
    (100, 100, (100,), "sigmoid", "glorot_uniform", 1, (25, 100),
     -46.92889730988215, -1585.2331437646524, 0.6496204322668526),
    (100, 100, (100,), "tanh", "he_normal", 10, (25, 100),
     -5.531248709518545, -1131.5021652659007, 0.6018381457540279),
    (100, 100, (1000,), "relu", "lecun_normal", 1, (25, 100),
     -24.857674257413233, -1241.941403822942, 0.5954067650339964),
    (100, 100, (1000,), "identity", "glorot_normal", 10, (25, 100),
     -49.66037744636776, -1418.0996396366454, 0.6387637880009253),
    (1000, 10, (100,), "log_softmax", "glorot_normal", 1, (250, 10),
     -2.157983014856103, -821.8910528092026, 0.999999671320564),
    (1000, 10, (100,), "log_sigmoid", "lecun_normal", 10, (250, 10),
     -2.25281191108881, -813.3197346939389, 0.9998208055604957),
    (1000, 10, (1000,), "softmin", "he_normal", 1, (250, 10),
     -2.932635323616438, -819.9889270165279, 0.9999535335431835),
    (1000, 10, (1000,), "softmax", "glorot_uniform", 10, (250, 10),
     -3.27895924524588, -809.0526184106433, 0.9996980844468629),
    (1000, 100, (100,), "sigmoid", "lecun_uniform", 1, (250, 100),
     40.193814730616296, -2003.2760146757932, 0.9999864051131802),
    (1000, 100, (100,), "tanh", "he_normal", 10, (250, 100),
     38.349789631939906, -1968.7361166078529, 0.9984649082549426),
    (1000, 100, (1000,), "relu", "normal", 1, (250, 100),
     47.91240910167704, -2194.259205351918, 0.8620693547752554),
    (1000, 100, (1000,), "identity", "uniform", 10, (250, 100),
     39.788475103832646, -2004.3219743138504, 0.9999999882159872)
])
def test_regression_against_grafo(n_samples, n_targets, hidden_layer_sizes,
                                  activation, weight_scheme, reg_alpha,
                                  exp_preds_shape, exp_preds_median,
                                  exp_preds_min, exp_preds_r2):
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
    actual_preds_r2 = r2_score(y_test, actual_preds)
    np.testing.assert_allclose(actual_preds_shape, exp_preds_shape)
    np.testing.assert_allclose(actual_preds_median, exp_preds_median)
    np.testing.assert_allclose(actual_preds_min, exp_preds_min)
    np.testing.assert_allclose(actual_preds_r2, exp_preds_r2)


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
