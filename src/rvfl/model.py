import numpy as np
from scipy.special import logsumexp
from scipy.stats import mode
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from rvfl.activations import resolve_activation
from rvfl.weights import resolve_weight


class RVFL(BaseEstimator):
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

    def fit(self, X, Y):
        # Assumption : X, Y have been pre-processed.
        # X shape: (n_samples, n_features)
        # Y shape: (n_samples, n_classes-1)
        if self.reg_alpha is not None and self.reg_alpha < 0.0:
            raise ValueError("Negative reg_alpha. Expected range : None or [0.0, inf).")
        fn = resolve_activation(self.activation)[1]
        self._activation_fn = fn
        self._N = X.shape[1]
        hidden_layer_sizes = np.asarray(self.hidden_layer_sizes)
        self._weight_mode = resolve_weight(self.weight_scheme)

        # weights shape: (n_layers,)
        # biases shape: (n_layers,)
        self.W_ = []
        self.b_ = []
        rng = self.get_generator(self.seed)

        self.W_.append(
            self._weight_mode(
                self._N, hidden_layer_sizes[0], rng=self.get_generator(self.seed)
                )
            )
        self.b_.append(
            self._weight_mode(1, hidden_layer_sizes[0], rng=rng)
            .reshape(-1)
            )
        for i, layer in enumerate(hidden_layer_sizes[1:]):
            # (n_hidden, n_features)
            self.W_.append(
                self._weight_mode(hidden_layer_sizes[i], layer, rng=rng,)
                )
            # (n_hidden,)
            self.b_.append(
                self._weight_mode(1, layer, rng=rng,).reshape(-1)
                )

        # hypothesis space shape: (n_layers,)
        Hs = []
        H_prev = X
        for w, b in zip(self.W_, self.b_, strict=False):
            Z = H_prev @ w.T + b  # (n_samples, n_hidden)
            H_prev = self._activation_fn(Z)
            Hs.append(H_prev)

        # design matrix shape: (n_samples, sum_hidden+n_features)
        # or (n_samples, sum_hidden)
        if self.direct_links:
            Hs.append(X)
        D = np.hstack(Hs)

        # beta shape: (sum_hidden+n_features, n_classes-1)
        # or (sum_hidden, n_classes-1)

        # If reg_alpha is None, use direct solve using
        # MoorePenrose Pseudo-Inverse, otherwise use ridge regularized form.
        if self.reg_alpha is None:
            self.coeff_ = np.linalg.pinv(D) @ Y
        else:
            ridge = Ridge(alpha=self.reg_alpha, fit_intercept=False)
            ridge.fit(D, Y)
            self.coeff_ = ridge.coef_.T
        return self

    def predict(self, X):
        check_is_fitted(self)
        Hs = []
        H_prev = X
        for W, b in zip(self.W_, self.b_, strict=False):
            Z = H_prev @ W.T + b  # (n, m)
            H_prev = self._activation_fn(Z)
            Hs.append(H_prev)

        if self.direct_links:
            Hs.append(X)
        D = np.hstack(Hs)
        out = D @ self.coeff_

        return out

    def get_generator(self, seed):
        return np.random.default_rng(seed)


class RVFLClassifier(ClassifierMixin, RVFL):
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        direct_links: bool = True,
        seed: int = None,
        reg_alpha: float = None
    ):
        """
        Random vector functional link network classifier.

        Parameters
        ----------

        hidden_layer_sizes : array-like of shape (n_layers,)
          The ith element represents the number of neurons in the ith
          hidden layer.
        activation : str, default="identity"
          Activation function for the hidden layers.
            - 'identity' or 'identity', no-op activation
            - 'tanh', the hyperbolic tan function
            - 'relu', the rectified linear unit function
            - 'sigmoid', a non-linear function that squashes input into
              the range `[0, 1]`
            - 'softmax', the normalized exponential function
            - 'softmin', `softmax` of the negated input
            - 'log_sigmoid', the natural logarithm of the sigmoid function
            - 'log_softmax', the logarithm of the standard softmax function
        weight_scheme : str, default="uniform"
          The random weight initialization scheme.
            - 'zeros', all weights set to zero
            - 'uniform', weights assigned from uniform distribution within `[0, 1)`
            - 'range', weights assigned from ``np.arange()`` normalized to the number
              of neurons
            - 'normal', weights assigned from a Gaussian distribution with mean `0`
              and standard deviation `1`.
            - 'he_uniform', He uniform variance scaling weights.
            - 'lecun_uniform', LeCun uniform weights.
            - 'glorot_uniform', Glorot uniform weights, also called Xavier uniform
              weights.
            - 'he_normal', He normal initialized weights.
            - 'lecun_normal', Lecun normal initialized weights.
            - 'glorot_normal', Glorot normal initialized weights, also called
              Xavier normal initialized weights.
        direct_links : bool, default=`True`
          Direct links are used in RVFL networks, while setting the value
          to `False` will reusult in an ELM network.
        seed : int, default=`None`
          Random seed used to initialize the network.
        reg_alpha : float, default=`None`
          When `None`, use Moore-Penrose inversion to solve for the output
          weights of the network. Otherwise, it specifies the constant that
          multiplies the L2 term of `sklearn` `Ridge`, controlling the
          regularization strength. `reg_alpha` must be a non-negative float.
        """
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                       activation=activation,
                       weight_scheme=weight_scheme,
                       direct_links=direct_links,
                       seed=seed,
                       reg_alpha=reg_alpha)

    def fit(self, X, y):
        """
        Build a gradient-free neural network from the training set (X, y).

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
          The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          The target values (class labels).

        Returns
        -------
        object
          Fitted estimator.
        """
        # shape: (n_samples, n_features)
        X, Y = validate_data(self, X, y)
        self.classes_ = unique_labels(Y)

        # onehot y
        # (this is necessary for everything beyond binary classification)
        self.enc_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # shape: (n_samples, n_classes-1)
        Y = self.enc_.fit_transform(Y.reshape(-1, 1))

        # call base fit method
        super().fit(X, Y)
        return self

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          The input samples.

        Returns
        -------
        ndarray
          The predicted classes, with shape (n_samples,) or (n_samples, n_outputs).
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        out = self.predict_proba(X)
        y_hat = self.classes_[np.argmax(out, axis=1)]
        return y_hat

    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        out = super().predict(X)
        out = np.exp(out - logsumexp(out, axis=1, keepdims=True))
        return out


class EnsembleRVFL(RVFL):
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        seed: int = None,
        reg_alpha: float = None
    ):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,
                         weight_scheme=weight_scheme,
                         direct_links=True,
                         seed=seed,
                         reg_alpha=reg_alpha)

    def fit(self, X, Y):

        if self.reg_alpha is not None and self.reg_alpha < 0.0:
            raise ValueError("Negative reg_alpha. Expected range : None or [0.0, inf).")

        fn = resolve_activation(self.activation)[1]
        self._activation_fn = fn
        self._N = X.shape[1]
        hidden_layer_sizes = np.asarray(self.hidden_layer_sizes)
        self._weight_mode = resolve_weight(self.weight_scheme)

        self.W_ = []
        self.b_ = []
        rng = self.get_generator(self.seed)

        self.W_.append(
            self._weight_mode(
                self._N, hidden_layer_sizes[0], rng=self.get_generator(self.seed)
                )
            )
        self.b_.append(
            self._weight_mode(1, hidden_layer_sizes[0], rng=rng)
            .reshape(-1)
            )

        for i, layer in enumerate(hidden_layer_sizes[1:]):
            # (n_hidden, n_features)
            self.W_.append(
                self._weight_mode(hidden_layer_sizes[i] + self._N, layer, rng=rng)
                )
            # (n_hidden,)
            self.b_.append(
                self._weight_mode(1, layer, rng=rng,).reshape(-1)
                )

        self.coeffs_ = []
        D = X

        for W, b in zip(self.W_, self.b_, strict=False):
            Z = D @ W.T + b  # (n_samples, n_hidden_layer_i)
            H = self._activation_fn(Z)
            # design matrix shape: (n_samples, n_hidden_layer_i+n_features)
            # or (n_samples, n_hidden_final)
            D = np.hstack((H, X))

            # beta shape: (n_hidden_final+n_features, n_classes-1)
            # or (n_hidden_final, n_classes-1)

            # If reg_alpha is None, use direct solve using
            # MoorePenrose Pseudo-Inverse, otherwise use ridge regularized form.
            if self.reg_alpha is None:
                coeff = np.linalg.pinv(D) @ Y
            else:
                ridge = Ridge(alpha=self.reg_alpha, fit_intercept=False)
                ridge.fit(D, Y)
                coeff = ridge.coef_.T
            self.coeffs_.append(coeff)

        return self

    def _forward(self, X):
        check_is_fitted(self)
        outs = []

        D = X
        for W, b, coeff in zip(self.W_, self.b_, self.coeffs_, strict=True):
            Z = D @ W.T + b
            H = self._activation_fn(Z)

            D = np.hstack((H, X))

            out = D @ coeff
            outs.append(out)

        return outs


class EnsembleRVFLClassifier(ClassifierMixin, EnsembleRVFL):
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        seed: int = None,
        reg_alpha: float = None,
        voting: str = "soft",    # "soft" or "hard"
    ):
        """
        Ensemble random vector functional link network classifier.

        Parameters
        ----------

        hidden_layer_sizes : array-like of shape (n_layers,)
          The ith element represents the number of neurons in the ith
          hidden layer.
        activation : str, default="identity"
          Activation function for the hidden layers.
            - 'identity' or 'identity', no-op activation
            - 'tanh', the hyperbolic tan function
            - 'relu', the rectified linear unit function
            - 'sigmoid', a non-linear function that squashes input into
              the range `[0, 1]`
            - 'softmax', the normalized exponential function
            - 'softmin', `softmax` of the negated input
            - 'log_sigmoid', the natural logarithm of the sigmoid function
            - 'log_softmax', the logarithm of the standard softmax function
        weight_scheme : str, default="uniform"
          The random weight initialization scheme.
            - 'zeros', all weights set to zero
            - 'uniform', weights assigned from uniform distribution within `[0, 1)`
            - 'range', weights assigned from ``np.arange()`` normalized to the number
              of neurons
            - 'normal', weights assigned from a Gaussian distribution with mean `0`
              and standard deviation `1`.
            - 'he_uniform', He uniform variance scaling weights.
            - 'lecun_uniform', LeCun uniform weights.
            - 'glorot_uniform', Glorot uniform weights, also called Xavier uniform
              weights.
            - 'he_normal', He normal initialized weights.
            - 'lecun_normal', Lecun normal initialized weights.
            - 'glorot_normal', Glorot normal initialized weights, also called
              Xavier normal initialized weights.
        seed : int, default=`None`
          Random seed used to initialize the network.
        reg_alpha : float, default=`None`
          When `None`, use Moore-Penrose inversion to solve for the output
          weights of the network. Otherwise, it specifies the constant that
          multiplies the L2 term of `sklearn` `Ridge`, controlling the
          regularization strength. `reg_alpha` must be a non-negative float.
        voting : str, default=`"soft"`
          Whether to use soft or hard voting in the ensemble.

        Notes
        -----
        The implementation is based on the one described by Shi et al. in [1]_.

        References
        ----------
        .. [1] Shi, Katuwal, Suganthan, Tanveer, "Random vector functional
           link neural network based ensemble deep learning." Pattern Recognition,
           vol. 117, pp. 107978, 2021. :doi:`10.1016/j.patcog.2021.107978,`.
        """
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,
                         weight_scheme=weight_scheme,
                         seed=seed,
                         reg_alpha=reg_alpha,
                         )
        self.voting = voting

    def fit(self, X, y):
        # shape: (n_samples, n_features)
        X, Y = validate_data(self, X, y)
        self.classes_ = unique_labels(Y)

        # onehot y
        # (this is necessary for everything beyond binary classification)
        self.enc_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        # shape: (n_samples, n_classes-1)
        Y = self.enc_.fit_transform(Y.reshape(-1, 1))

        # call base fit method
        super().fit(X, Y)
        return self

    def _check_voting(self):
        if self.voting == "hard":
            raise AttributeError(
                f"predict_proba is not available when voting={self.voting!r}"
            )
        return True

    @available_if(_check_voting)
    def predict_proba(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        outs = self._forward(X)
        probs = []

        for out in outs:
            p = np.exp(out - logsumexp(out, axis=1, keepdims=True))
            probs.append(p)

        return np.mean(probs, axis=0)

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        if self.voting == "soft":
            P = self.predict_proba(X)
            return self.classes_[np.argmax(P, axis=1)]

        outs = self._forward(X)
        votes = []

        for out in outs:
            p = np.exp(out - logsumexp(out, axis=1, keepdims=True))
            votes.append(self.classes_[np.argmax(p, axis=1)])

        votes = np.stack(votes, axis=1)
        m = mode(votes, axis=1, keepdims=False)
        return m.mode


class RVFLRegressor(RegressorMixin, MultiOutputMixin, RVFL):
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        direct_links: bool = True,
        seed: int = None,
        reg_alpha: float = None
    ):
        """
        Random vector functional link network regressor.

        Parameters
        ----------

        hidden_layer_sizes : array-like of shape (n_layers,)
          The ith element represents the number of neurons in the ith
          hidden layer.
        activation : str, default="identity"
          Activation function for the hidden layers.
            - 'identity' or 'identity', no-op activation
            - 'tanh', the hyperbolic tan function
            - 'relu', the rectified linear unit function
            - 'sigmoid', a non-linear function that squashes input into
              the range `[0, 1]`
            - 'softmax', the normalized exponential function
            - 'softmin', `softmax` of the negated input
            - 'log_sigmoid', the natural logarithm of the sigmoid function
            - 'log_softmax', the logarithm of the standard softmax function
        weight_scheme : str, default="uniform"
          The random weight initialization scheme.
            - 'zeros', all weights set to zero
            - 'uniform', weights assigned from uniform distribution within `[0, 1)`
            - 'range', weights assigned from ``np.arange()`` normalized to the number
              of neurons
            - 'normal', weights assigned from a Gaussian distribution with mean `0`
              and standard deviation `1`.
            - 'he_uniform', He uniform variance scaling weights.
            - 'lecun_uniform', LeCun uniform weights.
            - 'glorot_uniform', Glorot uniform weights, also called Xavier uniform
              weights.
            - 'he_normal', He normal initialized weights.
            - 'lecun_normal', Lecun normal initialized weights.
            - 'glorot_normal', Glorot normal initialized weights, also called
              Xavier normal initialized weights.
        direct_links : bool, default=`True`
          Direct links are used in RVFL networks, while setting the value
          to `False` will reusult in an ELM network.
        seed : int, default=`None`
          Random seed used to initialize the network.
        reg_alpha : float, default=`None`
          When `None`, use Moore-Penrose inversion to solve for the output
          weights of the network. Otherwise, it specifies the constant that
          multiplies the L2 term of `sklearn` `Ridge`, controlling the
          regularization strength. `reg_alpha` must be a non-negative float.
        """
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                       activation=activation,
                       weight_scheme=weight_scheme,
                       direct_links=direct_links,
                       seed=seed,
                       reg_alpha=reg_alpha)

    def fit(self, X, y):
        """
        Train the gradient-free neural network on the training set (X, y).

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
          The training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          The target values.

        Returns
        -------
        object
          The fitted estimator.
        """
        X, Y = validate_data(self, X, y, multi_output=True)
        super().fit(X, Y)
        return self

    def predict(self, X):
        """
        Predict regression target for X.

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
          The input samples.

        Returns
        -------
        ndarray
          The predicted values. Should have shape (n_samples,) or
          (n_samples, n_outputs).
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        return super().predict(X)
