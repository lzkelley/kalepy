kalepy
======

*Multidimensional kernel density estimation for calculating distribution functions and resampling.*

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

One dimensional kernel density estimation:
******************************************

.. code-block:: python

   import kalepy as kale
   import matplotlib.pyplot as plt
   # For a one-dimensional data-set:
   #   If the evaluation `points` aren't provided then `kalepy` automatically generates them
   points, density = kale.density(data, points=None)

   plt.plot(points, density, 'k-', lw=2.0, alpha=0.8, label='KDE')

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_files/demo_8_0.png
    :alt: my-picture1


One dimensional resampling:
***************************

.. code-block:: python

    # Draw new samples from the KDE reconstructed PDF
    samples = kale.resample(data)

    # Plot new samples
    plt.hist(samples, density=True, alpha=0.5, label='new samples', color='0.65', edgecolor='b')

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_files/demo_11_0.png


Multi-dimensional kernel density estimation:
********************************************

.. code-block:: python

    # Construct a KDE instance from data, shaped (N, 3) for `N` data points, and 3 dimensions
    kde = kale.KDE(data)

    # Build a corner plot using the `kalepy` plotting submodule
    corner = kale.plot.Corner(kde, figsize=[10, 10])

.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_files/demo_13_1.png



Documentation
-------------

A number of examples are included in `the package notebooks <https://github.com/lzkelley/kalepy/tree/master/notebooks>`_, and the `readme file <https://github.com/lzkelley/kalepy/blob/master/README.md>`_.  Some background information and references are included in `the JOSS paper <>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
  api

* :ref:`modindex`


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
