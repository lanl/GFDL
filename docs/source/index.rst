.. gfdl documentation master file, created by
   sphinx-quickstart on Fri Jan  9 14:32:26 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gradient Free Deep Learning 
===========================

Gradient Free Deep Learning (GFDL) is a Python package for performing
machine learning classification and regression tasks using neural networks
that do not require backpropagation.

It specifically provides:

- A scikit-learn compatible estimator for classification
  (:class:`gfdl.model.GFDLClassifier`)
- A scikit-learn compatible estimator for regression
  (:class:`gfdl.model.GFDLRegressor`)
- A scikit-learn compatible estimator for ensemble classification
  (:class:`gfdl.model.EnsembleGFDLClassifier`)
- A small set of neural-network style activations (:mod:`gfdl.activations`)

Getting started
---------------

Install:

.. code-block:: bash

   pip install gfdl

Quick example
-------------

.. code-block:: python

   import numpy as np
   from gfdl.model import GFDLClassifier

   rng = np.random.default_rng(seed=42)
   X = rng.standard_normal(size=(100, 10))
   y = (X[:, 0] > 0).astype(int)

   clf = GFDLClassifier()
   clf.fit(X, y)

.. toctree::
   :hidden:
   :maxdepth: 2

   api/index
   release
   dev
