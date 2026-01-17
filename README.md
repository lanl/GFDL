# Gradient Free Deep Learning (GFDL) -- LANL O5013 

## Introduction
This is a Python library that provides a variety of `scikit-learn` conformant
machine learning estimators that do not use backpropagation. The most
prominent estimator types we support provide access to single and multi layer
random vector functional link (RVFL) networks and extreme learning machines
(ELMs). There is a considerable background literate on these two types
of gradient free networks, and one obvious advantage of being gradient free
is that expensive hardware devices are not required to train the models
efficiently.

## Contribution Guidelines

1. We use an open source license that is compatible with the rest of the
scientific Python ecosystem, so please do not provide contributions that
have a potential to be copyleft. For example, do not copy or even read code
from libraries that have a GPL or other copyleft-style license, as we cannot
accept it while retaining our more liberal software license.
2. At the moment it is not acceptable to use machine learning/AI/LLMs as
part of the code contribution/review process. The reason is related to provenance
and licensing---we cannot know for sure if the material being contributed
originated or partially originated from code that had a copyleft license.
3. Please make an effort to format your PR titles and commit messages
according to the [guidelines used provided by NumPy](https://numpy.org/devdocs/dev/development_workflow.html#writing-the-commit-message). This helps keep our commit
history readable and easier to debug.
4. Please try to avoid merging your own code---we aim to provide timely
code reviews and have developers merge the code of others when they are
satisfied.
5. Please add regression tests for bug fixes and new features, and avoid
making unrelated changes (i.e., formatting changes to other parts of the
code alongside a bug fix or improvement).
