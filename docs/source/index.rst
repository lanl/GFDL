.. gfdl documentation master file, created by
   sphinx-quickstart on Fri Jan  9 14:32:26 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gradient Free Deep Learning
===========================

Gradient Free Deep Learning is a Python package for ...

It provides:

- A scikit-learn compatible estimator (:class:`gfdl.model.GFDLClassifier`)
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

   X = np.random.randn(100, 10)
   y = (X[:, 0] > 0).astype(int)

   clf = GFDLClassifier()
   clf.fit(X, y)


.. toctree::
   :hidden:
   :maxdepth: 2

   api/index