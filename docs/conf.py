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
    "sphinx_multiversion",
]
autosummary_generate = True

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "inherited-members": False,
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
    "huggingface_hub",
    "museval",
    "musdb",
    "car_noise_simulator",
]

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 4,
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

# --- sphinx-multiversion configuration ---

# Build only the 'main' branch
smv_branch_whitelist = r"^(main)$"

# Build only the specified tags
# The 'r' before the string
# denotes a raw string, useful for regex.
# The '^' and '$' anchor the match to the start and end of the tag name.
# The '|' acts as an OR operator.
smv_tag_whitelist = r"^(0\.7\.1|0\.6\.3|0\.5\.0|0\.4\.1|0\.3\.4|0\.2\.1|0\.1\.1)$"

# Optional: The version to show as "latest" in the version selector dropdown.
# You might want this to be 'main' or your latest stable tag (e.g., '0.7.1').
smv_latest_version = "main"  # Or '0.7.1' if that's your most recent stable release
