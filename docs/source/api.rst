==========
kalepy API
==========

.. contents:: :local:

Kernel Density Estimation
=========================

The primary API is two functions in the top level package: `kalepy.density` and `kalepy.resample`.  Additionally, `kalepy.pdf` is included which is a shorthand for `kalepy.density(..., probability=True)` --- i.e. a normalized density distribution.

Each of these functions constructs a `KDE` (kalepy.kde.KDE) instance, calls the corresponding member function, and returns the results.  If multiple operations will be done on the same data set, it will be more efficient to construct the `KDE` instance manually and call the methods on that.  i.e.

.. code-block:: python

    kde = kalepy.KDE(data)            # construct `KDE` instance
    points, density = kde.density()   # use `KDE` for density-estimation
    new_samples = kde.resample()      # use same `KDE` for resampling


.. include:: kde_api.rst



Plotting Distributions
======================

For more extended documentation, see the `kalepy.plot submodule documentation. <kalepy_plot.html>`_

.. include:: plot_api.rst


.. Module contents
.. ---------------
.. 
.. .. automodule:: kalepy
..    :members:
..    :undoc-members:
..    :show-inheritance:
