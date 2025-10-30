# rvfl/model.py
import numpy as np
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from rvfl.activations import resolve_activation


class RVFL(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        direct_links: bool = True,
        seed: int = None,
        reg_alpha: float = None
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.direct_links = direct_links
        self.seed = seed
        self.weight_scheme = weight_scheme
        self.reg_alpha = reg_alpha

    def _uniform(self, d, h, *, first_layer=False, **kwargs):
        # NOTE: _uniform() had to split out for pickle/serialization
        # for conformance with the sklearn estimator API:
        # https://scikit-learn.org/stable/developers/develop.html#developing-scikit-learn-estimators
        if first_layer:
            self._rng = self.get_generator(self.seed)
            return self._rng.uniform(0, 1, (h, d))
        return self._rng.uniform(0, 1, (h, d))

    def fit(self, X, Y):
        # Assumption : X, Y have been pre-processed.
        # X shape: (n_samples, n_features)
        # Y shape: (n_samples, n_classes-1)
        fn = resolve_activation(self.activation)[1]
        self._activation_fn = fn
        if self.reg_alpha is not None and self.reg_alpha < 0.0:
            raise ValueError("Negative reg_alpha. Expected range : None or [0.0, inf).")
        self._N = X.shape[1]
        hidden_layer_sizes = np.asarray(self.hidden_layer_sizes)
        self._weights(self.weight_scheme)

        # weights shape: (n_layers,)
        # biases shape: (n_layers,)
        self.W_ = []
        self.b_ = []

        self.W_.append(
            self._weight_mode(self._N, hidden_layer_sizes[0], first_layer=True)
            )
        self.b_.append(
            self._weight_mode(hidden_layer_sizes[0], 1, first_layer=True)
            .reshape(-1)
            )
        for i, layer in enumerate(hidden_layer_sizes[1:]):
            # (n_hidden, n_features)
            self.W_.append(self._weight_mode(hidden_layer_sizes[i], layer))
            # (n_hidden,)
            self.b_.append(self._weight_mode(layer, 1).reshape(-1))

        # hypothesis space shape: (n_layers,)
        Hs = []
        H_prev = X
        for w, b in zip(self.W_, self.b_, strict=False):
            Z = H_prev @ w.T + b  # (n_samples, n_hidden)
            H_prev = self._activation_fn(Z)
            Hs.append(H_prev)

        # design matrix shape: (n_samples, n_hidden_final+n_features)
        # or (n_samples, n_hidden_final)
        D = np.concatenate((Hs[-1], X), axis=1) if self.direct_links else Hs[-1]

        # beta shape: (n_hidden_final+n_features, n_classes-1)
        # or (n_hidden_final, n_classes-1)

        # If reg_alpha is None, use direct solve using
        # MoorePenrose Pseudo-Inverse, otherwise use ridge regularized form.
        if self.reg_alpha is None:
            self.beta_ = np.linalg.pinv(D) @ Y
        else:
            ridge = Ridge(alpha=self.reg_alpha, fit_intercept=False)
            ridge.fit(D, Y)
            self.beta_ = ridge.coef_.T
        return self

    def predict(self, X):
        check_is_fitted(self)
        Hs = []
        H_prev = X
        for W, b in zip(self.W_, self.b_, strict=False):
            Z = H_prev @ W.T + b  # (n, m)
            H_prev = self._activation_fn(Z)
            Hs.append(H_prev)

        D = np.concatenate((Hs[-1], X), axis=1) if self.direct_links else Hs[-1]

        out = D @ self.beta_

        return out

    def get_generator(self, seed):
        return np.random.default_rng(seed)

    def _weights(self, weight_scheme):

        name = weight_scheme.strip().lower()
        match name:
            case "zeros":
                def _zeros(d, h, **kwargs):
                    return np.zeros((h, d))
                self._weight_mode = _zeros
            case "uniform":
                self._weight_mode = self._uniform
            case "range":
                def _range(d, h, **kwargs):
                    s = np.arange(d * h)
                    s = np.subtract(s, np.mean(s))
                    s /= np.std(s)
                    s = np.nan_to_num(s)
                    return s.reshape(h, d)
                self._weight_mode = _range
            case _:
                allowed = {"zeros", "uniform", "range"}
                raise ValueError(
                    f"weight scheme='{weight_scheme}' is not supported;\
                    choose from {allowed}"
                )


class RVFLClassifier(RVFL):
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        direct_links: bool = True,
        seed: int = None,
        reg_alpha: float = None
    ):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                       activation=activation,
                       weight_scheme=weight_scheme,
                       direct_links=direct_links,
                       seed=seed,
                       reg_alpha=reg_alpha)

    def fit(self, X, y):
        # shape: (n_samples, n_features)
        X, Y = validate_data(self, X, y)
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(X)
        self.classes_ = unique_labels(y)

        # onehot y
        # (this is necessary for everything beyond binary classification)
        self.enc_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # shape: (n_samples, n_classes-1)
        Y = self.enc_.fit_transform(np.asarray(y).reshape(-1, 1))

        # call base fit method
        super().fit(X, Y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        out = self.predict_proba(X)
        y_hat = self.classes_[np.argmax(out, axis=1)]
        return y_hat

    def predict_proba(self, X):
        check_is_fitted(self)
        X = self._scaler.transform(X)
        out = super().predict(X)
        out = np.exp(out - logsumexp(out, axis=1, keepdims=True))
        return out
