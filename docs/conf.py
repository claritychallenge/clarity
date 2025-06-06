"""Sphinx configuration file for pyClarity documentation."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(".."))

project = "pyClarity"
copyright = "2020-2025, pyClarity authors"
author = "pyClarity authors"
try:
    from importlib.metadata import version as get_version

    _full_release = get_version(
        "pyClarity"
    )  # Replace 'pyClarity' with your actual package name if different
    release = str(_full_release)
    version = str(".".join(release.split(".")[:2]))
except Exception:
    # Fallback if package is not installed or version cannot be found
    release = "0.0.0+unknown"
    version = "0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]
autosummary_generate = True

autodoc_default_options = {
    "members": True,  # Document members (functions, classes, methods)
    "undoc-members": (
        True
    ),  # Document members without docstrings (optional, but can be useful)
    "inherited-members": (
        False
    ),  # Set to True if you want docstrings inherited from base classes
    "show-inheritance": True,  # Show base classes of classes
    # 'private-members': False, # Usually keep this False unless specifically needed
    # 'special-members': False, # Usually keep this False unless specifically needed (e.g., __call__)
    # 'no-value': True,       # Don't display default values for attributes
}
np.float_ = np.float64
autodoc_mock_imports = [
    "speechbrain",
    "asteroid",
    "hyperpyyaml",
    "local",
    "infer",
    "train",
    "eval",
    "enhance",
    "evaluate",
    "eval",
    "whisper",
    "fastdtw",
    "transformer_cpc1_ensemble_decoder",
    "transformer_cpc1_decoder",
    "jiwer",
]

html_theme_options = {
    "collapse_navigation": False,  # Keep the navigation fully expanded
    "sticky_navigation": (
        True
    ),  # (Optional) Keep the navigation pane fixed when scrolling
    "navigation_depth": (
        4
    ),  # (Optional) Set a deeper level for navigation items to appear
    # 'includehidden': True,       # (Optional) Show toctree entries that are hidden by default
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]
