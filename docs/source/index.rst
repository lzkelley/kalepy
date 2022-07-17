Introduction
============

*Multidimensional kernel density estimation for distribution functions, resampling, and plotting.*

`kalepy on github <https://github.com/lzkelley/kalepy>`_

|travis| |codecov| |rtd| |joss|

.. |travis| image:: https://travis-ci.org/lzkelley/kalepy.svg?branch=master
.. |codecov| image:: https://codecov.io/gh/lzkelley/kalepy/branch/master/graph/badge.svg
.. |rtd| image:: https://readthedocs.org/projects/kalepy/badge/?version=latest
.. |joss| image:: https://joss.theoj.org/papers/10.21105/joss.02784/status.svg
   :target: https://doi.org/10.21105/joss.02784

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/logo_anim_small.gif

.. contents:: :local:

Installation
------------

.. code-block:: bash

    pip install kalepy


or from source, for development:

.. code-block:: bash

    git clone https://github.com/lzkelley/kalepy.git
    pip install -e kalepy



Quickstart
----------

| Basic examples are shown below.
| `The top-level API for KDE is here, <kde_api.html>`_
| `and for plotting is here, <plot_api.html>`_
| `The README file on github also includes installation and quickstart examples. <https://github.com/lzkelley/kalepy/blob/master/README.md>`_

One dimensional kernel density estimation:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import kalepy as kale
   import matplotlib.pyplot as plt
   points, density = kale.density(data, points=None)
   plt.plot(points, density, 'k-', lw=2.0, alpha=0.8, label='KDE')

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/kde1d.png
    :alt: my-picture1


One dimensional resampling:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Draw new samples from the KDE reconstructed PDF
    samples = kale.resample(data)
    plt.hist(samples, density=True, alpha=0.5, label='new samples', color='0.65', edgecolor='b')

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/resamp1d.png


Multi-dimensional kernel density estimation:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    # Construct a KDE instance from data, shaped (N, 3) for `N` data points, and 3 dimensions
    kde = kale.KDE(data)
    # Build a corner plot using the `kalepy` plotting submodule
    corner = kale.corner(kde)

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/kde3dresamp.png



Documentation
-------------

A number of examples are included in `the package notebooks <https://github.com/lzkelley/kalepy/tree/master/notebooks>`_, and the `readme file <https://github.com/lzkelley/kalepy/blob/master/README.md>`_.  Some background information and references are included in `the JOSS paper <https://joss.theoj.org/papers/10.21105/joss.02784>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Kernel Density Estimation (KDE) API <kde_api>
   Plotting API <plot_api>
   Full Package Documentation <apidoc_modules/kalepy>
   Modules list <apidoc_modules/modules>


Development & Contributions
---------------------------

Please visit the `github page to make contributions to the package. <https://github.com/lzkelley/kalepy>`_  Particularly if you encounter any difficulties or bugs in the code, please `submit an issue <https://github.com/lzkelley/kalepy/issues>`_, which can also be used to ask questions about usage, or to submit general suggestions and feature requests.  Direct additions, fixes, or other contributions are very welcome which can be done by submitting `pull requests <https://github.com/lzkelley/kalepy/pulls>`_.  If you are considering making a contribution / pull-request, please open an issue first to make sure it won't clash with other changes in development or planned for the future.  Some known issues and indended future-updates are noted in the `change-log <https://github.com/lzkelley/kalepy/blob/master/CHANGES.md>`_ file.  If you are looking for ideas of where to contribute, this would be a good place to start.

Updates and changes to the newest version of `kalepy` will not always be backwards compatible.  The package is consistently versioned, however, to ensure that functionality and compatibility can be maintained for dependencies.  Please consult the `change-log <https://github.com/lzkelley/kalepy/blob/master/CHANGES.md>`_ for summaries of recent changes.

Test Suite
^^^^^^^^^^

If you are making, or considering making, changes to the `kalepy` source code, the are a large number of built in continuous-integration tests, both in the `kalepy/tests <https://github.com/lzkelley/kalepy/tree/master/kalepy/tests>`_ directory, and in the `kalepy notebooks <https://github.com/lzkelley/kalepy/tree/master/notebooks>`_.  Many of the notebooks are automatically converted into test scripts, and run during continuous integration.  If you are working on a local copy of `kalepy`, you can run the tests using the `tester.sh script (i.e. '$ bash tester.sh') <https://github.com/lzkelley/kalepy/tree/master/tester.sh>`_, which will include the test notebooks.


Attribution
-----------

A JOSS paper has been published on the `kalepy` package.  If you have found this package useful in your research, please add a reference to the code paper:

.. code-block:: tex

    @article{Kelley2021,
      doi = {10.21105/joss.02784},
      url = {https://doi.org/10.21105/joss.02784},
      year = {2021},
      publisher = {The Open Journal},
      volume = {6},
      number = {57},
      pages = {2784},
      author = {Luke Zoltan Kelley},
      title = {kalepy: a Python package for kernel density estimation, sampling and plotting},
      journal = {Journal of Open Source Software}
    }

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
