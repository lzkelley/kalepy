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
import os
import sys
# sys.path.insert(0, os.path.abspath('../../kalepy/'))
sys.path.insert(0, os.path.abspath('../..'))   # if docs/source/conf.py

with open('../../kalepy/VERSION.txt') as inn:
    version = inn.read().strip()


# -- Project information -----------------------------------------------------

project = 'kalepy'
copyright = '2022, Luke Zoltan Kelley and Contributors'
author = 'Luke Zoltan Kelley'

# The full version, including alpha/beta/rc tags
# release = '0.4'
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'nbsphinx',                # convert notebooks to sphinx output
    # 'sphinx.ext.napoleon',     # allow numpy/google style docstrings
    'numpydoc',     # allow numpy/google style docstrings
    'sphinx.ext.autodoc',      # auto-generate documentation from docstrings
    'sphinx.ext.mathjax',      # render math in html using mathjax
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',  # link to other packages' documentations
]
autosummary_generate = True    # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# NOTE: `numpy` is actually needed, otherwise things break
autodoc_mock_imports = ['pytest', 'scipy', 'six', 'matplotlib', 'numba']

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Extensions Parameters ----------------------------------------------------

# autodoc
autoclass_content = 'both'
autodoc_default_options = {
    'autoclass_content': 'both',
    'special-members': '__init__',
}

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
}

numpydoc_show_class_members = False

# Report warnings for all validation checks
# numpydoc_validation_checks = {"all"}

# Report warnings for all checks *except* those listed
numpydoc_validation_checks = {
    "all",
    "GL08",   # The object does not have a docstring
    "ES01",   # No extended summary found
    "PR01",   # Parameters { ... } not documented
    "PR02",   # Unknown parameters { ... }
    "PR07",   # Parameter has no description
    "PR10",   # Parameter "___" requires a space before the colon separating the parameter name and type
    "RT01",   # No Returns section found
    "RT03",   # Return value has no description
    "SS01",   # No summary found (a short summary in a single line should be present at the beginning of the docstring)

    "SA01",   # See Also section not found
    "EX01",   # No examples section found

    "GL01",   # Docstring text (summary) should start in the line immediately after the opening quotes (not in the same line, or leaving a blank line in between)
    "GL02",   # Closing quotes should be placed in the line after the last text in the docstring (do not close the quotes in the same line as the text, or leave a blank line between the last text and the quotes)
    "GL03",   # Double line break found; please use only one blank line to separate sections or paragraphs, and do not leave blank lines at the end of docstrings
    "PR05",   # Parameter "___" type should not finish with "."
    "PR08",   # Parameter "weights" description should start with a capital letter
    "PR09",   # Parameter "___" description should finish with "."
    "RT02",   # The first line of the Returns section should contain only the type, unless multiple values are being returned
    "RT05",   # Return value description should finish with ".
    "SS03",   # Summary does not end with a period
    "SS05",   # Summary must start with infinitive verb, not third person (e.g. use "Generate" instead of "Generates")
}

# Napoleon settings
# napoleon_google_docstring = True
# napoleon_numpy_docstring = True
# napoleon_include_init_with_doc = False
# napoleon_include_private_with_doc = False
# napoleon_include_special_with_doc = True
# napoleon_use_admonition_for_examples = False
# napoleon_use_admonition_for_notes = False
# napoleon_use_admonition_for_references = False
# napoleon_use_ivar = False
# napoleon_use_param = True
# napoleon_use_rtype = True


def run_apidoc(_):
    """

    https://github.com/readthedocs/readthedocs.org/issues/1139#issuecomment-312626491

    """
    from sphinx.ext.apidoc import main
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.join(cur_dir, os.path.pardir))
    output_dir = os.path.join(cur_dir, "apidoc_modules")
    # docs/source ==> /docs ==> kalepy/
    input_dir = os.path.join(cur_dir, os.path.pardir, os.path.pardir, "kalepy")
    main(['-e', '-o', output_dir, input_dir, '--force'])
    return


def setup(app):
    app.connect('builder-inited', run_apidoc)
    return
