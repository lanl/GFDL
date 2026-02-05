"""
Estimators for gradient free deep learning.
"""

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
from sklearn.utils import column_or_1d
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_is_fitted, validate_data

from gfdl.activations import resolve_activation
from gfdl.weights import resolve_weight


class GFDL(BaseEstimator):
    """Base class for GFDL for classification and regression."""
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        direct_links: bool = True,
        seed: int = None,
        reg_alpha: float = None,
        rtol: float | None = None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.direct_links = direct_links
        self.seed = seed
        self.weight_scheme = weight_scheme
        self.reg_alpha = reg_alpha
        self.rtol = rtol

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
            self.coeff_ = np.linalg.pinv(D, rtol=self.rtol) @ Y
        else:
            ridge = Ridge(alpha=self.reg_alpha, fit_intercept=False)
            ridge.fit(D, Y)
            self.coeff_ = ridge.coef_.T
        return self

    def partial_fit(self, X, y):
        # Moore Penrose Pseudoinverse:
        # D+ = (D.T @ D)^-1 @ D.T
        #
        # Least squares solution:
        # D+ @ y = (D.T @ D)^-1 @ D.T @ y
        #
        # We're persisting the gram (D.T @ D) and moment (D.T @ y) matrices
        # and updating them by adding the gram and moment matrices of
        # each consecutive batch.
        #
        # Consider the summation representation of matrix multiplication:
        # (B.T @ B)_ij = sum_k (B.T_ik @ B_kj)
        #
        # Now suppose the full design matrix is formed by vertically stacking
        # two batches, written as [B_1 | B_2]:
        # [B_1 | B_2].T @ [B_1 | B_2] = B_1.T @ B_1 + B_2.T @ B_2
        #
        # It's evident that it is indeed possible to update normal equations
        # through summation

        if self.reg_alpha is not None and self.reg_alpha < 0.0:
            raise ValueError("Negative reg_alpha. Expected range : None or [0.0, inf).")
        # Assumption : X, Y have been pre-processed.
        # X shape: (n_samples, n_features)
        # Y shape: (n_samples, n_classes-1)

        if not hasattr(self, "W_"):
            # initialize params only on first call
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

        # design matrix shape: (n_samples, sum_hidden+n_features)
        # or (n_samples, sum_hidden)
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

        if not hasattr(self, "A"):
            self.A = np.zeros((D.shape[1], D.shape[1]))
            self.B = np.zeros((D.shape[1], y.shape[1]))

        self.A += D.T @ D
        self.B += D.T @ y

        # beta shape: (sum_hidden+n_features, n_classes-1)
        # or (sum_hidden, n_classes-1)

        # If reg_alpha is None, use direct solve using
        # MoorePenrose Pseudo-Inverse, otherwise use ridge regularized form.

        if self.reg_alpha is None:
            if self.rtol is None:
                self.coeff_ = np.linalg.pinv(self.A) @ self.B
            else:
                self.coeff_ = np.linalg.pinv(self.A, rcond=self.rtol) @ self.B
        else:
            reg = np.identity(self.A.shape[0]) * self.reg_alpha
            self.coeff_ = np.linalg.solve(self.A + reg, self.B)
        if self.coeff_.ndim == 2 and self.coeff_.shape[1] == 1:
            self.coeff_ = self.coeff_.ravel()
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


class GFDLClassifier(ClassifierMixin, GFDL):
    """
    Random vector functional link network classifier.

    This model fits a feedforward neural network with fixed random hidden-layer
    parameters and solves for the output weights using linear least squares or
    ridge regression. When direct links are disabled, the model architecture corresponds
    to an Extreme Learning Machine (ELM) architecture.

    Parameters
    ----------
    hidden_layer_sizes : array-like of shape (n_layers,), default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : str, default='identity'
        Activation function for the hidden layers.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'tanh': :func:`tanh <gfdl.activations.tanh>`.

        - 'relu': :func:`relu <gfdl.activations.relu>`.

        - 'sigmoid': :func:`sigmoid <gfdl.activations.sigmoid>`.

        - 'softmax': :func:`softmax <gfdl.activations.softmax>`.

        - 'softmin': :func:`softmin <gfdl.activations.softmin>`.

        - 'log_sigmoid': :func:`log_sigmoid <gfdl.activations.log_sigmoid>`.

        - 'log_softmax': :func:`log_softmax <gfdl.activations.log_softmax>`.

    weight_scheme : str, default='uniform'
        Distribution used to initialize the random hidden-layer weights.

        The initialization functions generate weight matrices of shape
        (n_hidden_units, n_features), where values are drawn
        according to the selected scheme.

        - 'zeros': set weights to zeros (:func:`zeros <gfdl.weights.zeros>`).

        - 'range': set weights to normalized np.arange
          (:func:`range <gfdl.weights.range>`).

        - 'uniform': uniform distribution (:func:`uniform <gfdl.weights.uniform>`).

        - 'he_uniform': He uniform distribution
          (:func:`he_uniform <gfdl.weights.he_uniform>`).

        - 'lecun_uniform': Lecun uniform distribution
          (:func:`lecun_uniform <gfdl.weights.lecun_uniform>`).

        - 'glorot_uniform': Glorot uniform distribution
          (:func:`glorot_uniform <gfdl.weights.glorot_uniform>`).

        - 'normal': Normal distribution (:func:`normal <gfdl.weights.normal>`).

        - 'he_normal': He normal distribution
          (:func:`he_normal <gfdl.weights.he_normal>`).

        - 'lecun_normal': Lecun normal distribution
          (:func:`lecun_normal <gfdl.weights.lecun_normal>`).

        - 'glorot_normal': Glorot normal distribution
          (:func:`glorot_normal <gfdl.weights.glorot_normal>`).

    direct_links : bool, default=True
        Whether to connect input layer to output nodes.
        When set to False, only the hidden-layer activations are used, corresponding
        to the Extreme Learning Machine (ELM) architecture.

    seed : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        initialization.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    reg_alpha : float, default=None
        Amount of ridge shrinkage to apply in order to improve
        conditioning during Ridge regression. When set to zero or `None`,
        model uses direct solve using Moore-Penrose Pseudo-Inverse.

    rtol : float, default=None
      Cutoff for small singular values for the Moore-Penrose
      pseudo-inverse. Only applies when ``reg_alpha=None``.
      When ``rtol=None``, the array API standard default for
      ``pinv`` is used.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    classes_ : ndarray or list of ndarray of shape (n_classes,)
        Class labels for each output.

    W_ : list of ndarray of shape (n_layers,)
        Weight matrices of the hidden layers. The ith element in the list represents the
        weight matrix corresponding to layer i.

    b_ : list of ndarray of shape (n_layers,)
        Bias vectors of the hidden layers. The ith element in the list represents the
        bias term corresponding to layer i.

    coeff_ : ndarray of shape (n_features_out, n_outputs)
        Output weight matrix learned by fit method.

    See Also
    --------
    GFDLRegressor : Regressor variant for the RVFL architecture.

    Examples
    --------
    >>> from gfdl.model import GFDLClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_classification(n_samples=100, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
    ...                                                     random_state=1)
    >>> clf = GFDLClassifier(seed=1).fit(X_train, y_train)
    >>> clf.predict_proba(X_test[:1])
    array([[0.46123716, 0.53876284]])
    >>> clf.predict(X_test[:5, :])
    array([1, 0, 1, 0, 1])
    """
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        direct_links: bool = True,
        seed: int = None,
        reg_alpha: float = None,
        rtol: float = None
    ):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                       activation=activation,
                       weight_scheme=weight_scheme,
                       direct_links=direct_links,
                       seed=seed,
                       reg_alpha=reg_alpha,
                       rtol=rtol)

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

    def partial_fit(self, X, y, classes=None):
        """
        Build a gradient-free neural network from the batched training set (X, y).

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
          The batched training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          The batched target values (class labels).

        classes : array of shape (n_classes,), default=None
            Classes across all calls to partial_fit.
            Can be obtained via `np.unique(y_all)`, where y_all is the
            target vector of the entire dataset.
            This argument is required for the first call to partial_fit
            and can be omitted in the subsequent calls.

        Returns
        -------
        object
            The partially fitted estimator.
        """
        # shape: (n_samples, n_features)
        X, Y = validate_data(self, X, y, reset=not hasattr(self, "n_features_in_"))
        Y = column_or_1d(Y, warn=True)
        check_classification_targets(Y)

        if not hasattr(self, "classes_"):
            if classes is None:
                raise TypeError("Classes must not be None for first partial_fit call")
            self.classes_ = classes
            self.enc_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.enc_.fit(self.classes_.reshape(-1, 1))

        if not set(unique_labels(Y)) <= set(self.classes_):
            raise ValueError(
                f"Expected only labels in classes_ = {list(self.classes_)!r}, "
                f"but got {unique_labels(Y)!r}."
            )
        Y = self.enc_.transform(Y.reshape(-1, 1))

        super().partial_fit(X, Y)
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
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          The input samples.

        Returns
        -------
        ndarray
          The class probabilities of the input samples. The order of the classes
          corresponds to that in the attribute ``classes_``. The ndarray should
          have shape (n_samples, n_classes).
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)
        out = super().predict(X)
        out = np.exp(out - logsumexp(out, axis=1, keepdims=True))
        return out


class EnsembleGFDL(BaseEstimator):
    """Base class for ensemble GFDL model for classification and regression."""
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        seed: int = None,
        reg_alpha: float = None,
        rtol: float | None = None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.weight_scheme = weight_scheme
        self.seed = seed
        self.reg_alpha = reg_alpha
        self.rtol = rtol

    def get_generator(self, seed):
        return np.random.default_rng(seed)

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
                coeff = np.linalg.pinv(D, rtol=self.rtol) @ Y
            else:
                ridge = Ridge(alpha=self.reg_alpha, fit_intercept=False)
                ridge.fit(D, Y)
                coeff = ridge.coef_.T
            self.coeffs_.append(coeff)

        return self

    def partial_fit(self, X, Y):

        if self.reg_alpha is not None and self.reg_alpha < 0.0:
            raise ValueError("Negative reg_alpha. Expected range : None or [0.0, inf).")

        if not hasattr(self, "W_"):
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

        if not hasattr(self, "As"):
            self.As = []
            self.Bs = []

        for i, (W, b) in enumerate(zip(self.W_, self.b_, strict=False)):
            Z = D @ W.T + b  # (n_samples, n_hidden_layer_i)
            H = self._activation_fn(Z)
            # design matrix shape: (n_samples, n_hidden_layer_i+n_features)
            # or (n_samples, n_hidden_final)
            D = np.hstack((H, X))

            if len(self.As) == i:
                self.As.append(np.zeros((D.shape[1], D.shape[1])))
                self.Bs.append(np.zeros((D.shape[1], Y.shape[1])))

            self.As[i] += D.T @ D
            self.Bs[i] += D.T @ Y

            # beta shape: (sum_hidden+n_features, n_classes-1)
            # or (sum_hidden, n_classes-1)

            # If reg_alpha is None, use direct solve using
            # MoorePenrose Pseudo-Inverse, otherwise use ridge regularized form.

            if self.reg_alpha is None:
                if self.rtol is None:
                    coef_ = np.linalg.pinv(self.As[i]) @ self.Bs[i]
                else:
                    coef_ = np.linalg.pinv(self.As[i], rcond=self.rtol) @ self.Bs[i]
            else:
                reg = np.identity(self.As[i].shape[0]) * self.reg_alpha
                coef_ = np.linalg.solve(self.As[i] + reg, self.Bs[i])
            if coef_.ndim == 2 and coef_.shape[1] == 1:
                coef_ = coef_.ravel()

            self.coeffs_.append(coef_)

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


class EnsembleGFDLClassifier(ClassifierMixin, EnsembleGFDL):
    """
    Ensemble random vector functional link network classifier.

    Parameters
    ----------

    hidden_layer_sizes : array-like of shape (n_layers,)
      The ith element represents the number of neurons in the ith
      hidden layer.

    activation : str, default='identity'
        Activation function for the hidden layers.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'tanh': :func:`tanh <gfdl.activations.tanh>`.

        - 'relu': :func:`relu <gfdl.activations.relu>`.

        - 'sigmoid': :func:`sigmoid <gfdl.activations.sigmoid>`.

        - 'softmax': :func:`softmax <gfdl.activations.softmax>`.

        - 'softmin': :func:`softmin <gfdl.activations.softmin>`.

        - 'log_sigmoid': :func:`log_sigmoid <gfdl.activations.log_sigmoid>`.

        - 'log_softmax': :func:`log_softmax <gfdl.activations.log_softmax>`.

    weight_scheme : str, default='uniform'
        Distribution used to initialize the random hidden-layer weights.

        The initialization functions generate weight matrices of shape
        (n_hidden_units, n_features), where values are drawn
        according to the selected scheme.

        - 'zeros': set weights to zeros (:func:`zeros <gfdl.weights.zeros>`).

        - 'range': set weights to normalized np.arange
          (:func:`range <gfdl.weights.range>`).

        - 'uniform': uniform distribution (:func:`uniform <gfdl.weights.uniform>`).

        - 'he_uniform': He uniform distribution
          (:func:`he_uniform <gfdl.weights.he_uniform>`).

        - 'lecun_uniform': Lecun uniform distribution
          (:func:`lecun_uniform <gfdl.weights.lecun_uniform>`).

        - 'glorot_uniform': Glorot uniform distribution
          (:func:`glorot_uniform <gfdl.weights.glorot_uniform>`).

        - 'normal': Normal distribution (:func:`normal <gfdl.weights.normal>`).

        - 'he_normal': He normal distribution
          (:func:`he_normal <gfdl.weights.he_normal>`).

        - 'lecun_normal': Lecun normal distribution
          (:func:`lecun_normal <gfdl.weights.lecun_normal>`).

        - 'glorot_normal': Glorot normal distribution
          (:func:`glorot_normal <gfdl.weights.glorot_normal>`).

    seed : int, default=`None`
      Random seed used to initialize the network.

    reg_alpha : float, default=`None`
      When `None`, use Moore-Penrose inversion to solve for the output
      weights of the network. Otherwise, it specifies the constant that
      multiplies the L2 term of `sklearn` `Ridge`, controlling the
      regularization strength. `reg_alpha` must be a non-negative float.

    rtol : float, default=None
      Cutoff for small singular values for the Moore-Penrose
      pseudo-inverse. Only applies when ``reg_alpha=None``.
      When ``rtol=None``, the array API standard default for
      ``pinv`` is used.

    voting : str, default=`"soft"`
      Whether to use soft or hard voting in the ensemble.

    Notes
    -----
    The implementation is based on the one described by Shi et al. in [1]_.

    References
    ----------
    .. [1] Shi, Katuwal, Suganthan, Tanveer, "Random vector functional
       link neural network based ensemble deep learning." Pattern Recognition,
       vol. 117, pp. 107978, 2021, https://doi.org/10.1016/j.patcog.2021.107978.

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from gfdl.model import EnsembleGFDLClassifier
    >>> X, y = make_classification(n_samples=1000, n_features=4,
    ...                            n_informative=2, n_redundant=0,
    ...                            random_state=0, shuffle=False)
    >>> clf = EnsembleGFDLClassifier(seed=0)
    >>> clf.fit(X, y)
    >>> print(clf.predict([[0, 0, 0, 0]]))
    [1]
    """
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        seed: int = None,
        reg_alpha: float = None,
        rtol: float = None,
        voting: str = "soft",    # "soft" or "hard"
    ):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                         activation=activation,
                         weight_scheme=weight_scheme,
                         seed=seed,
                         reg_alpha=reg_alpha,
                         rtol=rtol
                         )
        self.voting = voting

    def fit(self, X, y):
        """
        Train the ensemble of connected RVFL networks on the training set (X, y).

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

    def partial_fit(self, X, y, classes=None):
        """
        Train the ensemble of connected RVFL networks on the batched
        training set (X, y).

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
          The batched training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          The batched target values.

        Returns
        -------
        object
          The partially fitted estimator.
        """
        # shape: (n_samples, n_features)
        X, Y = validate_data(self, X, y, reset=not hasattr(self, "n_features_in_"))
        Y = column_or_1d(Y, warn=True)
        check_classification_targets(Y)

        if not hasattr(self, "classes_"):
            if classes is None:
                raise TypeError("Classes must not be None for first partial_fit call")
            self.classes_ = classes
            self.enc_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            self.enc_.fit(self.classes_.reshape(-1, 1))

        if not set(np.unique(Y)) <= set(self.classes_):
            raise ValueError(
                f"Expected only labels in classes_ = {list(self.classes_)!r}, "
                f"but got {unique_labels(Y)!r}."
            )

        Y = self.enc_.transform(Y.reshape(-1, 1))

        # call base fit method
        super().partial_fit(X, Y)
        return self

    def _check_voting(self):
        if self.voting == "hard":
            raise AttributeError(
                f"predict_proba is not available when voting={self.voting!r}"
            )
        return True

    @available_if(_check_voting)
    def predict_proba(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
          The input samples.

        Returns
        -------
        ndarray
          The class probabilities of the input samples. The order of the classes
          corresponds to that in the attribute ``classes_``. The ndarray should
          have shape (n_samples, n_classes).
        """
        check_is_fitted(self)
        X = validate_data(self, X, reset=False)

        outs = self._forward(X)
        probs = []

        for out in outs:
            p = np.exp(out - logsumexp(out, axis=1, keepdims=True))
            probs.append(p)

        return np.mean(probs, axis=0)

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


class GFDLRegressor(RegressorMixin, MultiOutputMixin, GFDL):
    """
    Random vector functional link network regressor.

    This model fits a feedforward neural network with fixed random hidden-layer
    parameters and solves for the output weights using linear least squares or
    ridge regression. When direct links are disabled, the model architecture corresponds
    to an Extreme Learning Machine (ELM) architecture.

    Parameters
    ----------
    hidden_layer_sizes : array-like of shape (n_layers,), default=(100,)
        The ith element represents the number of neurons in the ith
        hidden layer.

    activation : str, default='identity'
        Activation function for the hidden layers.

        - 'identity', no-op activation, useful to implement linear bottleneck,
          returns f(x) = x

        - 'tanh': :func:`tanh <gfdl.activations.tanh>`.

        - 'relu': :func:`relu <gfdl.activations.relu>`.

        - 'sigmoid': :func:`sigmoid <gfdl.activations.sigmoid>`.

        - 'softmax': :func:`softmax <gfdl.activations.softmax>`.

        - 'softmin': :func:`softmin <gfdl.activations.softmin>`.

        - 'log_sigmoid': :func:`log_sigmoid <gfdl.activations.log_sigmoid>`.

        - 'log_softmax': :func:`log_softmax <gfdl.activations.log_softmax>`.

    weight_scheme : str, default='uniform'
        Distribution used to initialize the random hidden-layer weights.

        The initialization functions generate weight matrices of shape
        (n_hidden_units, n_features), where values are drawn
        according to the selected scheme.

        - 'zeros': set weights to zeros (:func:`zeros <gfdl.weights.zeros>`).

        - 'range': set weights to normalized np.arange
          (:func:`range <gfdl.weights.range>`).

        - 'uniform': uniform distribution (:func:`uniform <gfdl.weights.uniform>`).

        - 'he_uniform': He uniform distribution
          (:func:`he_uniform <gfdl.weights.he_uniform>`).

        - 'lecun_uniform': Lecun uniform distribution
          (:func:`lecun_uniform <gfdl.weights.lecun_uniform>`).

        - 'glorot_uniform': Glorot uniform distribution
          (:func:`glorot_uniform <gfdl.weights.glorot_uniform>`).

        - 'normal': Normal distribution (:func:`normal <gfdl.weights.normal>`).

        - 'he_normal': He normal distribution
          (:func:`he_normal <gfdl.weights.he_normal>`).

        - 'lecun_normal': Lecun normal distribution
          (:func:`lecun_normal <gfdl.weights.lecun_normal>`).

        - 'glorot_normal': Glorot normal distribution
          (:func:`glorot_normal <gfdl.weights.glorot_normal>`).

    direct_links : bool, default=True
        Whether to connect input layer to output nodes.

        When set to False, only the hidden-layer activations are used, corresponding
        to the Extreme Learning Machine (ELM) architecture.

    seed : int, RandomState instance, default=None
        Determines random number generation for weights and bias
        initialization.
        Pass an int for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    reg_alpha : float, default=None
        Amount of ridge shrinkage to apply in order to improve
        conditioning during Ridge regression. When set to zero or `None`,
        model uses direct solve using Moore-Penrose Pseudo-Inverse.

    rtol : float, default=None
        Cutoff for small singular values for the Moore-Penrose
        pseudo-inverse. Only applies when ``reg_alpha=None``.
        When ``rtol=None``, the array API standard default for
        ``pinv`` is used.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    W_ : list of ndarray of shape (n_layers,)
        Weight matrices of the hidden layers. The ith element in the list represents the
        weight matrix corresponding to layer i.

    b_ : list of ndarray of shape (n_layers,)
        Bias vectors of the hidden layers. The ith element in the list represents the
        bias term corresponding to layer i.

    coeff_ : ndarray of shape (n_features_out, n_outputs)
        Output weight matrix learned by the fit method.

    See Also
    --------
    GFDLClassifier : Classifier variant for the RVFL architecture.

    Examples
    --------
    >>> from gfdl.model import GFDLRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(n_samples=200, n_features=20, random_state=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=1)
    >>> regr = GFDLRegressor(seed=1)
    >>> regr.fit(X_train, y_train)
    GFDLRegressor(seed=1)
    >>> regr.predict(X_test[:2])
    array([  18.368, -278.014])
    """
    def __init__(
        self,
        hidden_layer_sizes: np.typing.ArrayLike = (100,),
        activation: str = "identity",
        weight_scheme: str = "uniform",
        direct_links: bool = True,
        seed: int = None,
        reg_alpha: float = None,
        rtol: float | None = None,
    ):
        super().__init__(hidden_layer_sizes=hidden_layer_sizes,
                       activation=activation,
                       weight_scheme=weight_scheme,
                       direct_links=direct_links,
                       seed=seed,
                       reg_alpha=reg_alpha,
                       rtol=rtol)

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

    def partial_fit(self, X, y):
        """
        Train the gradient-free neural network on the batched training set (X, y).

        Parameters
        ----------

        X : array-like of shape (n_samples, n_features)
          The batched training input samples.
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
          The batched target values.

        Returns
        -------
        object
            Returns the partially fitted estimator.
        """
        # shape: (n_samples, n_features)
        X, Y = validate_data(self, X, y, reset=not hasattr(self, "n_features_in_"))
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        super().partial_fit(X, Y)
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
