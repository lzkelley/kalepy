Top-Level API
=============

Kernel Density Estimation
-------------------------

The primary API is two functions in the top level package: `kalepy.density` and `kalepy.resample`.  Additionally, `kalepy.pdf` is included which is a shorthand for `kalepy.density(..., probability=True)` --- i.e. a normalized density distribution.


   kalepy.density(data, points=None, ...):
       This function 1) constructs a kernel-density estimate of the distribution function from which the given `data` were sampled; then 2) returns the values of the distribution function at `points` (which are automatically generated if they are not given).
       
       Returns:
       
       * `points` : the locations at which the density function is sampled.
       * `density` : the values of the density function at the sample points.


   kalepy.resample(data, size=None, ...):
        This function 1) constructs a kernel-density estimate of the distribution function from which the given `data` were sampled; then, 2) resamples `size` data points from that function.  If `size` is not given, then the same number of points are returned as in the input `data`.
        
        Returns:
        
        * `samples` : newly sampled data points.
        
    
Each of these functions constructs a `KDE` (kalepy.kde.KDE) instance, calls the corresponding member function, and returns the results.  If multiple operations will be done on the same data set, it will be more efficient to construct the `KDE` instance manually and call the methods on that.  i.e.

.. code-block:: python

    kde = kalepy.KDE(data)            # construct `KDE` instance
    points, density = kde.density()   # use `KDE` for density-estimation
    new_samples = kde.resample()      # use same `KDE` for resampling


Plotting
--------



Module contents
---------------

.. automodule:: kalepy
   :members:
   :undoc-members:
   :show-inheritance:
