# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root to the path for autodoc
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
project = 'insideLLMs'
copyright = '2024, insideLLMs Contributors'
author = 'insideLLMs Contributors'
release = '0.1.0'
version = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx_autodoc_typehints',
]

# Generate autosummary even if no references
autosummary_generate = True

# Napoleon settings for Google and NumPy docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Autodoc settings
autodoc_member_order = 'groupwise'
autodoc_typehints = 'both'
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to ignore when looking for source files
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'
html_title = 'insideLLMs Documentation'
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'light_css_variables': {
        'color-brand-primary': '#2563eb',
        'color-brand-content': '#2563eb',
    },
    'dark_css_variables': {
        'color-brand-primary': '#60a5fa',
        'color-brand-content': '#60a5fa',
    },
}

# -- Options for todo extension ----------------------------------------------
todo_include_todos = True
