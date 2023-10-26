# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'dwiml_doc'
copyright = '2023, dwiml'
author = 'dwiml'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Could not make it work with either sphinx_book_theme nor sphinx_rtd_theme.
html_theme = 'classic'   # 'pyramid', 'bizstyle', 'sphinxdoc' all work
html_static_path = ['_static']
html_theme_options = {'body_max_width': '80%'}
