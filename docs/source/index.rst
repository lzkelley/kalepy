Introduction
============

*Multidimensional kernel density estimation for distribution functions, resampling, and plotting.*

`kalepy on github <https://github.com/lzkelley/kalepy>`_

|travis| |codecov| |rtd|

.. |travis| image:: https://travis-ci.org/lzkelley/kalepy.svg?branch=master
.. |codecov| image:: https://codecov.io/gh/lzkelley/kalepy/branch/master/graph/badge.svg
.. |rtd| image:: https://readthedocs.org/projects/kalepy/badge/?version=latest

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
| `The top-level API is documented here, with many KDE and plotting examples, <api.html>`_
| `The README file on github also includes installation and quickstart examples. <https://github.com/lzkelley/kalepy/blob/master/README.md>`_

One dimensional kernel density estimation:
******************************************

.. code-block:: python

   import kalepy as kale
   import matplotlib.pyplot as plt
   points, density = kale.density(data, points=None)
   plt.plot(points, density, 'k-', lw=2.0, alpha=0.8, label='KDE')

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/kde1d.png
    :alt: my-picture1


One dimensional resampling:
***************************

.. code-block:: python

    # Draw new samples from the KDE reconstructed PDF
    samples = kale.resample(data)
    plt.hist(samples, density=True, alpha=0.5, label='new samples', color='0.65', edgecolor='b')

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/resamp1d.png


Multi-dimensional kernel density estimation:
********************************************

.. code-block:: python

    # Construct a KDE instance from data, shaped (N, 3) for `N` data points, and 3 dimensions
    kde = kale.KDE(data)
    # Build a corner plot using the `kalepy` plotting submodule
    corner = kale.corner(kde)

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/kde3dresamp.png



Documentation
-------------

A number of examples are included in `the package notebooks <https://github.com/lzkelley/kalepy/tree/master/notebooks>`_, and the `readme file <https://github.com/lzkelley/kalepy/blob/master/README.md>`_.  Some background information and references are included in `the JOSS paper <>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   introduction <index>
   kalepy API <api>
   Full Package Documentation <kalepy>


Development & Contributions
---------------------------

Please visit the `github page <https://github.com/lzkelley/kalepy>`_ for issues or bug reports.  Contributions and feedback are very welcome.


Attribution
-----------

A JOSS paper has been submitted.  If you have found this package useful in your research, please add a reference to the code paper:

.. code-block:: tex

    @article{kalepy,
      author = {Luke Zoltan Kelley},
      title = {kalepy: a python package for kernel density estimation and sampling},
      journal = {The Journal of Open Source Software},
      publisher = {The Open Journal},
    }


.. Indices and tables
.. ==================
.. 
.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
