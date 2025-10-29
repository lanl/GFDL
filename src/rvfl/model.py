# rvfl/model.py
import numpy as np
from scipy.special import logsumexp
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rvfl.activations import resolve_activation


class RVFL:
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        direct_links: bool = True,
        seed: int = None,
        reg_alpha: float = None
    ):
        self.hidden_layer_sizes = np.array(hidden_layer_sizes)
        name, fn = resolve_activation(activation)
        self.activation = name
        self._activation_fn = fn
        self.direct_links = direct_links
        self.seed = seed
        self._weights(weight_scheme)

        if reg_alpha is not None and reg_alpha < 0.0:
            raise ValueError("Negative reg_alpha. Expected range : None or [0.0, inf).")
        self.reg_alpha = reg_alpha

    def fit(self, X, Y):
        # Assumption : X, Y have been pre-processed.
        # X shape: (n_samples, n_features)
        # Y shape: (n_samples, n_classes-1)
        self.N = X.shape[1]

        # weights shape: (n_layers,)
        # biases shape: (n_layers,)
        self.W = []
        self.b = []

        self.W.append(
            self.weight_mode(self.N, self.hidden_layer_sizes[0], first_layer=True)
            )
        # self.b.append(
        #    self.weight_mode(self.hidden_layer_sizes[0], 1, first_layer=True)
        #    .reshape(-1)
        #    )
        self.b.append(
            self.weight_mode(1, self.hidden_layer_sizes[0], first_layer=True)
            .reshape(-1)
            )
        for i, layer in enumerate(self.hidden_layer_sizes[1:]):
            # (n_hidden, n_features)
            self.W.append(self.weight_mode(self.hidden_layer_sizes[i], layer))
            # (n_hidden,)
            # self.b.append(self.weight_mode(layer, 1).reshape(-1))
            self.b.append(self.weight_mode(1, layer).reshape(-1))

        # hypothesis space shape: (n_layers,)
        Hs = []
        H_prev = X
        for w, b in zip(self.W, self.b, strict=False):
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
            self.beta = np.linalg.pinv(D) @ Y
        else:
            ridge = Ridge(alpha=self.reg_alpha, fit_intercept=False)
            ridge.fit(D, Y)
            self.beta = ridge.coef_.T

    def predict(self, X):
        Hs = []
        H_prev = X
        for W, b in zip(self.W, self.b, strict=False):
            Z = H_prev @ W.T + b  # (n, m)
            H_prev = self._activation_fn(Z)
            Hs.append(H_prev)

        D = np.concatenate((Hs[-1], X), axis=1) if self.direct_links else Hs[-1]

        out = D @ self.beta

        return out

    def get_generator(self, seed):
        return np.random.default_rng(seed)

    def _weights(self, weight_scheme):

        name = weight_scheme.strip().lower()
        match name:
            case "zeros":
                def _zeros(d, h, **kwargs):
                    return np.zeros((h, d))
                self.weight_mode = _zeros
            case "range":
                def _range(d, h, **kwargs):
                    s = np.arange(d * h)
                    s = np.subtract(s, np.mean(s))
                    s /= np.std(s)
                    s = np.nan_to_num(s)
                    return s.reshape(h, d)
                self.weight_mode = _range
            case "uniform":
                def _uniform(d, h, *, first_layer=False, **kwargs):
                    if first_layer:
                        self.rng = self.get_generator(self.seed)
                        return self.rng.uniform(0, 1, (h, d))
                    return self.rng.uniform(0, 1, (h, d))
                self.weight_mode = _uniform
            case "he_uniform":
                def _he_uniform(d, h, *, first_layer=False, **kwargs):
                    # This implementation deviates from the standard expression where
                    # the number of input features (d) are always used to compute the
                    # limit. However, using the standard form returned a different
                    # answer from GrafoRVFL, which uses the output size i.e. hidden
                    # layer size instead (from ChatGPT). Needs further exploration
                    # of why they deviate from the standard form.
                    # If we choose to use the standard form, then our tests cannot be
                    # used to compare against GrafoRVFL as the results could be order
                    # one difference or higher.
                    limit = np.sqrt(6 / h)
                    if first_layer:
                        self.rng = self.get_generator(self.seed)
                        return self.rng.uniform(-limit, limit, (h, d))
                    return self.rng.uniform(-limit, limit, (h, d))
                self.weight_mode = _he_uniform
            case "lecun_uniform":
                def _lecun_uniform(d, h, *, first_layer=False, **kwargs):
                    # Same comment as "he_uniform"
                    limit = np.sqrt(3 / h)
                    if first_layer:
                        self.rng = self.get_generator(self.seed)
                        return self.rng.uniform(-limit, limit, (h, d))
                    return self.rng.uniform(-limit, limit, (h, d))
                self.weight_mode = _lecun_uniform
            case "glorot_uniform":
                def _glorot_uniform(d, h, *, first_layer=False, **kwargs):
                    fan_avg = 0.5 * (d + h)
                    limit = np.sqrt(3 / fan_avg)
                    if first_layer:
                        self.rng = self.get_generator(self.seed)
                        return self.rng.uniform(-limit, limit, (h, d))
                    return self.rng.uniform(-limit, limit, (h, d))
                self.weight_mode = _glorot_uniform
            case "normal":
                def _normal(d, h, *, first_layer=False, **kwargs):
                    if first_layer:
                        self.rng = self.get_generator(self.seed)
                        return self.rng.normal(0, 1, (h, d))
                    return self.rng.normal(0, 1, (h, d))
                self.weight_mode = _normal
            case "he_normal":
                def _he_normal(d, h, *, first_layer=False, **kwargs):
                    # Same comment as "he_uniform"
                    var = np.sqrt(2 / h)
                    if first_layer:
                        self.rng = self.get_generator(self.seed)
                        return self.rng.normal(0, var, (h, d))
                    return self.rng.normal(0, var, (h, d))
                self.weight_mode = _he_normal
            case "lecun_normal":
                def _lecun_normal(d, h, *, first_layer=False, **kwargs):
                    # Same comment as "he_uniform"
                    var = 1 / np.sqrt(h)
                    if first_layer:
                        self.rng = self.get_generator(self.seed)
                        return self.rng.normal(0, var, (h, d))
                    return self.rng.normal(0, var, (h, d))
                self.weight_mode = _lecun_normal
            case "glorot_normal":
                def _glorot_normal(d, h, *, first_layer=False, **kwargs):
                    fan_avg = 0.5 * (d + h)
                    var = np.sqrt(1 / fan_avg)
                    if first_layer:
                        self.rng = self.get_generator(self.seed)
                        return self.rng.normal(0, var, (h, d))
                    return self.rng.normal(0, var, (h, d))
                self.weight_mode = _glorot_normal
            case _:
                allowed = {"zeros", "uniform", "range", "normal"}
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
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)
        self.classes = np.unique(y)

        # onehot y
        # (this is necessary for everything beyond binary classification)
        self.enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # shape: (n_samples, n_classes-1)
        Y = self.enc.fit_transform(y.reshape(-1, 1))

        # call base fit method
        super().fit(X, Y)

    def predict(self, X):
        out = self.predict_proba(X)
        y_hat = self.classes[np.argmax(out, axis=1)]

        return y_hat

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        out = super().predict(X)
        out = np.exp(out - logsumexp(out, axis=1, keepdims=True))

        return out
