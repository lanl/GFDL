# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GFDL'
html_title = project
copyright = '2026, Emma Viani, Navamita Ray, Tyler Reddy'
author = 'Emma Viani, Navamita Ray, Tyler Reddy'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'numpydoc',
    'sphinx_copybutton',
    'sphinx_design',
    'matplotlib.sphinxext.plot_directive',
    "sphinx.ext.linkcode",
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []
numpydoc_class_members_toctree = False
add_module_names = False



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = []

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

# For linking code to the apis 
import inspect
import os
from pathlib import Path

GITHUB_REPO_URL = "https://github.com/lanl/GFDL"   
GITHUB_REF = "treddy_conform_with_numpydoc" 

# conf.py is docs/source/conf.py -> repo root is usually parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]

def linkcode_resolve(domain, info):
    """Return a GitHub URL for the documented Python object."""
    if domain != "py":
        return None

    modname = info.get("module")
    fullname = info.get("fullname")
    if not modname:
        return None

    try:
        module = __import__(modname, fromlist=["*"])
    except Exception:
        return None

    obj = module
    for part in (fullname or "").split("."):
        if not part:
            continue
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        obj = inspect.unwrap(obj)
        filename = inspect.getsourcefile(obj)
        if not filename:
            return None
        filename = Path(filename).resolve()
        source, start_line = inspect.getsourcelines(obj)
    except Exception:
        return None

    try:
        rel_path = filename.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None

    end_line = start_line + len(source) - 1
    return f"{GITHUB_REPO_URL}/blob/{GITHUB_REF}/{rel_path}#L{start_line}-L{end_line}"
