# rvfl/model.py
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rvfl.activations import resolve_activation


class RVFL:
    def __init__(
        self,
        n_hidden: int = 10,
        activation: str = "identity",
        weight_scheme: str = "uniform",
        direct_links: bool = True,
        seed: int = None
    ):
        self.n_hidden = n_hidden
        name, fn = resolve_activation(activation)
        self.activation = name
        self._activation_fn = fn
        self.direct_links = direct_links
        self.seed = seed
        self._weights(weight_scheme)

    def fit(self, X, y):
        # shape: (n_samples, n_features)
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.N = X.shape[1]

        self.classes = np.unique(y)

        # onehot y
        # (this is necessary for everything beyond binary classification)
        self.enc = OneHotEncoder(handle_unknown="ignore")
        # shape: (n_samples, n_classes-1)
        Y = self.enc.fit_transform(y.reshape(-1, 1))

        # weights shape: (n_hidden, n_features)
        self.W = self.weight_mode(self.N, self.n_hidden)
        # biases shape: (n_hidden,)
        self.b = self.weight_mode(self.n_hidden, 1).reshape(-1)

        # hypothesis space shape: (n_samples, n_hidden)
        H = self._activation_fn(X @ self.W.T + self.b)

        # phi shape: (n_samples, n_hidden+n_features)
        # or (n_samples, n_hidden)
        Phi = np.concatenate((H, X), axis=1) if self.direct_links else H

        # beta shape: (n_hidden+n_features, n_classes-1)
        # or (n_hidden, n_classes-1)
        self.beta = np.linalg.pinv(Phi) @ Y

    def predict(self, X):
        X = self.scaler.transform(X)

        H = self._activation_fn(X @ self.W.T + self.b)
        Phi = np.concatenate((H, X), axis=1) if self.direct_links else H
        out = Phi @ self.beta

        y_hat = self.classes[np.argmax(out, axis=1)]

        return y_hat

    def predict_proba(self, X):
        X = self.scaler.transform(X)

        H = self._activation_fn(X @ self.W.T + self.b)
        Phi = np.concatenate((H, X), axis=1) if self.direct_links else H
        out = Phi @ self.beta
        out = np.exp(out) / np.exp(out).sum(axis=1, keepdims=True)

        return out

    def get_generator(self, seed):
        return np.random.default_rng(seed)

    def _weights(self, weight_scheme):

        name = weight_scheme.strip().lower()
        match name:
            case "zeros":
                def _zeros(d, h):
                    return np.zeros((h, d))
                self.weight_mode = _zeros
            case "uniform":
                def _uniform(d, h):
                    return self.get_generator(self.seed).uniform(0, 1, (h, d))
                self.weight_mode = _uniform
            case "range":
                def _range(d, h):
                    s = np.arange(d * h)
                    s = np.subtract(s, np.mean(s))
                    s /= np.std(s)
                    return s.reshape(h, d)
                self.weight_mode = _range
            case _:
                allowed = {"zeros", "uniform", "range"}
                raise ValueError(
                    f"weight scheme='{weight_scheme}' is not supported;\
                    choose from {allowed}"
                )
