# rvfl/model.py
import numpy as np
from scipy.special import logsumexp
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
        reg_alpha: float = 0.0
    ):
        self.hidden_layer_sizes = np.array(hidden_layer_sizes)
        name, fn = resolve_activation(activation)
        self.activation = name
        self._activation_fn = fn
        self.direct_links = direct_links
        self.seed = seed
        self.reg_alpha = reg_alpha
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

        # weights shape: (n_layers,)
        # biases shape: (n_layers,)
        self.W = []
        self.b = []

        self.W.append(
            self.weight_mode(self.N, self.hidden_layer_sizes[0], first_layer=True)
            )
        self.b.append(
            self.weight_mode(self.hidden_layer_sizes[0], 1, first_layer=True)
            .reshape(-1)
            )
        for i, layer in enumerate(self.hidden_layer_sizes[1:]):
            # (n_hidden, n_features)
            self.W.append(self.weight_mode(self.hidden_layer_sizes[i], layer))
            # (n_hidden,)
            self.b.append(self.weight_mode(layer, 1).reshape(-1))

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

        # If reg_alpha is zero or very close to zero,
        # use direct solve using MoorePenrose Pseudo-Inverse
        # Otherwise, use ridge regularized form of solution
        tol = 1e-14
        if abs(self.reg_alpha) < tol:
            self.beta = np.linalg.pinv(D) @ Y
        else:
            DT = D.transpose()

            if (D.shape[1] <= D.shape[0]):
                scaledIMat = self.reg_alpha * np.identity(D.shape[1])
                DTD = DT @ D
                DTY = DT @ Y
                self.beta = np.linalg.inv(DTD + scaledIMat) @ DTY
            else:
                scaledIMat = self.reg_alpha * np.identity(D.shape[0])
                DDT = D @ DT
                self.beta = DT @ np.linalg.inv(DDT + scaledIMat) @ Y

    def predict(self, X):

        out = self.predict_proba(X)
        y_hat = self.classes[np.argmax(out, axis=1)]

        return y_hat

    def predict_proba(self, X):
        X = self.scaler.transform(X)

        Hs = []
        H_prev = X
        for W, b in zip(self.W, self.b, strict=False):
            Z = H_prev @ W.T + b  # (n, m)
            H_prev = self._activation_fn(Z)
            Hs.append(H_prev)

        Phi = np.concatenate((Hs[-1], X), axis=1) if self.direct_links else Hs[-1]

        out = Phi @ self.beta
        out = np.exp(out - logsumexp(out, axis=1, keepdims=True))

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
            case "uniform":
                def _uniform(d, h, *, first_layer=False, **kwargs):
                    if first_layer:
                        self.rng = self.get_generator(self.seed)
                        return self.rng.uniform(0, 1, (h, d))
                    return self.rng.uniform(0, 1, (h, d))
                self.weight_mode = _uniform
            case "range":
                def _range(d, h, **kwargs):
                    s = np.arange(d * h)
                    s = np.subtract(s, np.mean(s))
                    s /= np.std(s)
                    s = np.nan_to_num(s)
                    return s.reshape(h, d)
                self.weight_mode = _range
            case _:
                allowed = {"zeros", "uniform", "range"}
                raise ValueError(
                    f"weight scheme='{weight_scheme}' is not supported;\
                    choose from {allowed}"
                )
