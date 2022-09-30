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

project = "tanom-detect"
copyright = f"2021 - {datetime.now().year}, Georgian Partners LP"
author = "Benjmain Ye, Georgian Partners LP"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autosummary", 
              "sphinx.ext.autodoc", 
              "sphinx.ext.viewcode", 
              "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = []
# In order to also have the docstrings of __init__() methods included
autoclass_content = "both"
autosummary_generate = True

autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": None,
    "exclude-members": "Data, UcrData, SimpleDartsModel, DartsModel, PyODModel"
}



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
