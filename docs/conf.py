# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath("."))


# -- Project information -----------------------------------------------------

project = "ANTS"
copyright = "2026, Ben Whewell"
author = "Ben Whewell"

# The full version, including alpha/beta/rc tags
release = "0.2.0"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named "sphinx.ext.*") or your custom
# ones.
extensions = [
    "sphinx.ext.mathjax",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.tikz",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
]

bibtex_bibfiles = ["refs.bib"]

# Custom LaTeX macros used in the math-heavy chapters (e.g. the collision-based
# hybrid method). Defined centrally so the same shorthand works across pages.
# Macros with an optional argument (\sig, \hsig) use the MathJax 3
# [definition, num_args, default_for_optional] form.
mathjax3_config = {
    "tex": {
        "macros": {
            "sig": [r"\sigma_{#1}^{\mathrm{#2}}", 2, ""],
            "hsig": [r"\hat{\sigma}_{#1}^{\mathrm{#2}}", 2, ""],
            "bx": r"\mathbf{x}",
            "bn": r"\mathbf{n}",
            "bsOmega": r"\boldsymbol{\Omega}",
            "bbS": r"\mathbb{S}",
            "Psiu": r"\Psi^{\mathrm{u}}",
            "Psic": r"\Psi^{\mathrm{c}}",
            "Psit": r"\Psi^{\mathrm{t}}",
            "Phiu": r"\Phi^{\mathrm{u}}",
            "Phic": r"\Phi^{\mathrm{c}}",
            "psiu": r"\psi^{\mathrm{u}}",
            "psic": r"\psi^{\mathrm{c}}",
            "psit": r"\psi^{\mathrm{t}}",
            "phiu": r"\phi^{\mathrm{u}}",
            "phic": r"\phi^{\mathrm{c}}",
            "Qu": r"Q^{\mathrm{u}}",
            "Qc": r"Q^{\mathrm{c}}",
            "Qt": r"Q^{\mathrm{t}}",
            "qu": r"q^{\mathrm{u}}",
            "qc": r"q^{\mathrm{c}}",
            "qt": r"q^{\mathrm{t}}",
            "hg": r"\hat{g}",
            "hG": r"\hat{G}",
            "hm": r"\hat{m}",
            "hM": r"\hat{M}",
            "hw": r"\hat{w}",
            "hE": r"\hat{E}",
            "hOmega": r"\hat{\Omega}",
            "cG": r"\mathcal{G}",
            "cM": r"\mathcal{M}",
            "chG": r"\hat{\mathcal{G}}",
            "chM": r"\hat{\mathcal{M}}",
            "FOM": r"\mathrm{FOM}",
            "hquad": r"\quad",
            "quand": r"\quad\text{and}\quad",
            "bA": r"\mathbf{A}",
            "keff": r"k_{\mathrm{eff}}",
            "bbR": r"\mathbb{R}",
            "Tilde": [r"\widetilde{#1}", 1],
        }
    }
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/bwhewe-13/ants",
            "icon": "fa-brands fa-github",
        },
    ],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "show_toc_level": 2,
    "collapse_navigation": False,
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["basic.css"]


latex_elements = {
    "preamble": r"""
    \usepackage{tikz}
    \usepackage{tikzscale}
    """,
}
