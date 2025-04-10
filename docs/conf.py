# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "pyClarity"
copyright = "2025, pyClarity authors"
author = "pyClarity authors"
release = "v0.6.3"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
autosummary_generate = True

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",  # for Google or NumPy-style docstrings
    "sphinx.ext.viewcode",  # optional: adds links to source code
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "style_nav_header_background": "white",
    "version_selector": True,
}
html_logo = "images/challenges.png"
