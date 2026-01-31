.. gfdl documentation master file, created by
   sphinx-quickstart on Fri Jan  9 14:32:26 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Gradient Free Deep Learning 
===========================

Gradient Free Deep Learning (GFDL) is a Python package for ...

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

Process for Making a Release
============================

#. Start preparing the PR that drafts the release notes. This involves manually
   scanning all of the merged and appropriately milestoned PRs with an
   "enhancement" label, so that we don't miss mentioning new features. For other
   merged PRs and closed issues (i.e., bug fixes), the ``tools/gh_lists.py`` script
   will capture them in a raw list. The list of authors in the release notes
   may be populated by running ``tools/authors.py`` script over the appropriate
   commit range. If there are new authors in this release, you may also need
   to adjust ``.mailmap`` accordingly to avoid duplication, other issues. Try
   to add 2-3 new summary "highlights" at the top of the release notes for
   each release. Do not merge the release notes PR into ``main`` branch until
   you are ready to proceed with the release process proper.
#. Start preparing the PR to bump the ``main`` branch to the next version
   number, but DO NOT merge it until: the above release notes PR has been merged,
   the new ``maintenance`` branch has been pushed up. ``maintenance`` branches
   are named following this convention: ``maintenance/0.1.x`` for the ``0.1.x``
   release series, ``maintenance/0.2.x`` for the ``0.2.x`` release series, and
   so on. This allows bug fixes to be backported to maintenance branches while
   the ``main`` branch continues to gain new features.
#. After pushing the new ``maintenance`` branch up, that branch may need
   backport PR(s) for dependency version pins (upper bounds to avoid future breaks),
   fixups to release notes, etc.
#. As our project is currently small with few consumers, we will not use release
   candidates at this time, and will proceed directly with feature releases. As
   a precaution, try to work in a Python environment with the minimum version
   of each dependency when performing the release process.
#. Adjust the version number to remove the trailing ``.dev0`` in ``pyproject.toml``.
#. Now create a release commit similar to i.e., ``REL: GFDL 0.1.0 release commit``
   based on the current release version, but DO NOT push it yet.
#. Tag the release locally with ``git tag -s <v0.x.y>`` (the ``-s`` ensures
   the tag is signed--make sure you are properly setup for signing tags).
#. Continue with building release artifacts.
#. Since we are currently a pure Python project, go ahead and build the source
   distribution and binary (wheel) locally: ``python -m build``. If that
   succeeds, go ahead and push the release commit to the GFDL repo.
#. Now, start uploading release artifacts (you may need to ask for appropriate
   permissions if you are a new release manager). For PyPI, upload first the
   portable wheel, and then the sdist. ``twine upload /path/to/*.whl`` and
   then ``twine upload /path/to/*.tar.gz``. For the GitHub releases, use
   the GUI at https://github.com/lanl/GFDL/releases to create a release and
   upload all release artifacts. At this stage you can push up the release
   tag so that it may be used for the GitHub release: ``git push origin v0.1.0``.
#. Update the deployed documentation following the release.
#. For now, we will only announce the release within our small team by email
   with a short message and a copy of the release notes.
#. After the release is done, forward port relevant changes to release notes, build
   scripts, author name mapping, and tools scripts that were only made on the
   ``maintenance`` branch back to ``main``.
#. Finally, after the release, create a PR against the ``maintenance`` branch
   to draft the release notes skeleton for the next bug fix release on that
   branch, and bump to the next ``y`` in the version number (``0.x.y.dev0``)
   in ``pyproject.toml``.
