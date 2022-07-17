# Holodeck Package Documentation and Documents (`kalepy/docs`)

This directory contains documentation-related material:
- `media/`
  - demonstration plots and kalepy logo
- `paper/`
  - the JOSS published kalepy paper (Kelley-2020)
- `reference/`
  - PDFs of useful reference material (mostly published papers)
- `source/`
  - documentation source material using sphinx, e.g. for readthedocs.io


## sphinx / readthedocs

The source material lives in the `source/` subfolder.  `docs.sh` is a script to automatically update the sphinx documentation, using `make` commands described in the local `Makefile`.  Sphinx's `autodoc` is used for automatic documentation generation using docstrings within the python source code.

* `sphinx-apidoc` : generates .rst files with autodoc directives from the source code.
  * This command need to be rerun everytime the source-code changes.  Note that it will not overwrite existing .rst files unless the `-f` argument is used.
* `sphinx.ext.autosummary` : creates reference tables summarizing python objects within existing .rst files.  This is enabled by including `autosummary` directive in those .rst files, and is added when `make html` is run.
