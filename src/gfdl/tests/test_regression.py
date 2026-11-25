import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import fetch_openml, make_regression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from gfdl.activations import ACTIVATIONS
from gfdl.model import GFDLRegressor
from gfdl.weights import WEIGHTS


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
     -32.14209690766766, -481.55516141966143, 0.9712776377191723),
    (100, 10, (100,), "tanh", "uniform", 1, (25, 10),
     -41.92442138866774, -516.0313054982047, 0.9850538345738263),
    (100, 10, (100,), "log_softmax", "uniform", 1, (25, 10),
      -30.503849433404522, -557.5713457973524, 0.9999227232741242),
    (100, 10, (100,), "log_sigmoid", "normal", 10, (25, 10),
      -17.120248131915773, -569.1707430220065, 0.9778201399555199),
    (100, 10, (1000,), "softmin", "he_uniform", 1, (25, 10),
     -57.91874831027026, -589.6708942251743, 0.9656729060672852),
    (100, 10, (1000,), "softmax", "lecun_uniform", 10, (25, 10),
     -51.938708293813356, -513.4094120325685, 0.9589931742470019),
    (100, 100, (100,), "sigmoid", "glorot_uniform", 1, (25, 100),
     -47.26272058402997, -1585.5809945232877, 0.6502971105082866),
    (100, 100, (100,), "tanh", "he_normal", 10, (25, 100),
     -23.90761417540218, -1166.2865694494212, 0.6170406929979582),
    (100, 100, (1000,), "relu", "lecun_normal", 1, (25, 100),
     -30.08457703541013, -1232.8644759164906, 0.5927312717547109),
    (100, 100, (1000,), "identity", "glorot_normal", 10, (25, 100),
     -47.96807460502515, -1418.8499820911748, 0.6388489682458863),
    (1000, 10, (100,), "log_softmax", "glorot_normal", 1, (250, 10),
     -2.1563412717684343, -822.0126340622721, 0.9999995341600142),
    (1000, 10, (100,), "log_sigmoid", "lecun_normal", 10, (250, 10),
     -2.2826174459209545, -813.4046754272948, 0.9998234420452545),
    (1000, 10, (1000,), "softmin", "he_normal", 1, (250, 10),
     -2.932641851958226, -819.9890066758794, 0.9999535340299637),
    (1000, 10, (1000,), "softmax", "glorot_uniform", 10, (250, 10),
     -3.2789483326689974, -809.0525880888754, 0.9996980842991656),
    (1000, 100, (100,), "sigmoid", "lecun_uniform", 1, (250, 100),
     40.3160362219667, -2002.5934355519378, 0.9999868485449435),
    (1000, 100, (100,), "tanh", "he_normal", 10, (250, 100),
     36.92046171062485, -1967.0768888545792, 0.9985159789609054),
    (1000, 100, (1000,), "relu", "normal", 1, (250, 100),
     49.75134721707206, -2172.597661612917, 0.8544979511989379),
    (1000, 100, (1000,), "identity", "uniform", 10, (250, 100),
     39.799548825825056, -2004.3257538044973, 0.9999999880953903)
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
    # NOTE: the R2 scores for this test can range from
    # within the low 60% to low 80% due to statistical
    # noise from changes in seed or to the RNG mechanism.
    # We changed the estimator's seed so that expected R2 scores
    # remain high in the 0.70 - 0.80 range.
    # See https://github.com/lanl/GFDL/pull/157#discussion_r4647180466.
    (0.1, 0.7484237873692346),
    # NOTE: for Moore-Penrose, a large singular value
    # cutoff (rtol) is required to achieve reasonable R2 with
    # the Boston Housing dataset
    (None, 0.7943910014161282),
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
            seed=42,
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
@pytest.mark.parametrize("activation", ACTIVATIONS.keys())
@pytest.mark.parametrize("weight_scheme", list(WEIGHTS.keys())[1:])
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

    np.testing.assert_allclose(pf_model.coeff_, ff_model.coeff_, rtol=1e-5, atol=1e-4)


@pytest.mark.parametrize("reg_alpha, expected", [
    (0.1, 0.7225086527162101),
    # NOTE: for Moore-Penrose, a large singular value
    # cutoff (rtol) is required to achieve reasonable R2 with
    # the Boston Housing dataset
    (None, 0.736705709057082),
])
def test_partial_fit_boston(reg_alpha, expected):
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
            hidden_layer_sizes=[800] * 2,
            activation="tanh",
            weight_scheme="uniform",
            direct_links=1,
            seed=0,
            reg_alpha=reg_alpha,
            rtol=1e-6,  # when rtol is squared (1e-3 -> 1e-6)
                        # we achieve similar results as full fit
        )
    batch = 100
    for start in range(0, len(X_train_s), batch):
        end = min(start + batch, len(X_train_s))
        model.partial_fit(X_train_s[start:end], y_train[start:end])
    y_pred = model.predict(X_test_s)
    # Note that we can reach the same score as the full fit using
    # hidden_layer_sizes=[800] * 10 and reg_alpha=0.1,
    # but it's terribly inefficient in the case of reg_alpha=None
    # hence the reduced network size

    # RandomForestRegressor() with default params scores
    # 0.8733907 here; multi-layer GFDL with above params is a bit
    # worse, but certainly better than random chance:
    actual = r2_score(y_test, y_pred)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("hidden_layer_sizes", [
    (-1, -1, -1),
    (0, 1, 1),
])
def test_gh_85_regressor(hidden_layer_sizes):
    X, y = make_regression(random_state=42)
    model = GFDLRegressor(hidden_layer_sizes=hidden_layer_sizes, seed=0)
    with pytest.raises(ValueError, match="must be > 0"):
        model.fit(X, y)


@pytest.mark.parametrize("hidden_layer_sizes", [3, 100])
def test_gh_136_regressor(hidden_layer_sizes):
    # see gh-136
    X, y = make_regression(random_state=42)
    model = GFDLRegressor(hidden_layer_sizes=hidden_layer_sizes, seed=0)
    model.fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5,)
    assert model.hidden_layer_sizes == hidden_layer_sizes


@pytest.mark.parametrize("hidden_layer_sizes", [3, 100])
def test_gh_136_regressor_partial_fit(hidden_layer_sizes):
    # see gh-136
    X, y = make_regression(n_samples=60, random_state=42)
    model = GFDLRegressor(hidden_layer_sizes=hidden_layer_sizes, seed=0)
    model.partial_fit(X, y)
    preds = model.predict(X[:5])
    assert preds.shape == (5,)
    assert model.hidden_layer_sizes == hidden_layer_sizes


def test_preserve_class_inputs():
    # see: gh-111
    model = GFDLRegressor(seed=0)
    actual = model.get_params()
    expected = {"activation": "identity",
                "direct_links": True,
                "hidden_layer_sizes": (100,),
                "reg_alpha": None,
                "rtol": None,
                "seed": 0,
                "weight_scheme": "uniform"}
    for k, v in actual.items():
        assert v == expected[k]
        assert isinstance(v, type(expected[k]))
