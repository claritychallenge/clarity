"""Sphinx configuration file for pyClarity documentation."""

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../clarity/enhancer"))


project = "pyClarity"
copyright = "2020-2025, pyClarity authors"
author = "pyClarity authors"

_default_release = "0.0.0+unknown"
_default_version = "0.0"

try:
    from importlib.metadata import version as get_version

    _full_release = get_version("pyclarity")
    release = _full_release
    version = ".".join(release.split(".")[:2])
except Exception:
    # Fallback if package is not installed or version cannot be found
    release = _default_release
    version = _default_version

release = str(release)
version = str(version)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

root_doc = "index"
master_doc = "index"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "myst_parser",
]
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": False,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "speechbrain",
    "asteroid",
    "hyperpyyaml",
    "local",
    "infer",
    "train",
    "enhance",
    "evaluate",
    "eval",
    "whisper",
    "fastdtw",
    "transformer_cpc1_ensemble_decoder",
    "transformer_cpc1_decoder",
    "jiwer",
    "huggingface_hub",
    "museval",
    "musdb",
]

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
    # 'includehidden': True,  # (Opt) Show toctree entries that are hidden by default
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
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",  # Add this line for Markdown files
}
