import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../.."))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "oats"
copyright = f"2022 - {datetime.now().year}, Georgian Partners LP"
author = "Benjmain Ye, Georgian Partners LP"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "m2r2",
]

templates_path = ["templates"]
exclude_patterns = []

# In order to also have the docstrings of __init__() methods included
autoclass_content = "both"
# autosummary_generate = True

autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": None,
    "exclude-members": "hyperopt_model, hyperopt_ws",
}


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**/modules.rst",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["static"]
html_logo = "static/oats.png"
