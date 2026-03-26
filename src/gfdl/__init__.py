import platform

import numpy as np
import packaging
import scipy
import sklearn
from packaging.version import Version

from .model import GFDL

__all__ = ["GFDL"]

packaging_version = Version(packaging.__version__)
min_packaging_version = "24.0"
if packaging_version < Version(min_packaging_version):
    raise ImportError(f"{packaging_version=}, but {min_packaging_version=}")

python_version = Version(platform.python_version())
min_python_version = "3.12"
if python_version < Version(min_python_version):
    raise ImportError(f"{python_version=}, but {min_python_version=}")

numpy_version = Version(np.__version__)
min_numpy_version = "2.0.0"
if numpy_version < Version(min_numpy_version):
    raise ImportError(f"{numpy_version=}, but {min_numpy_version=}")

sklearn_version = Version(sklearn.__version__)
min_sklearn_version = "1.5.0"
if sklearn_version < Version(min_sklearn_version):
    raise ImportError(f"{sklearn_version=}, but {min_sklearn_version=}")

scipy_version = Version(scipy.__version__)
min_scipy_version = "1.13.0"
if scipy_version < Version(min_scipy_version):
    raise ImportError(f"{scipy_version=}, but {min_scipy_version=}")
