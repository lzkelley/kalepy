## To-Do
- Optimization drastically needed.  Things are done in (generally) the simplest ways, currently, need to be optimized for performance (both speed and memory [e.g. with reflections]).
- `kdes/`
    - Allow for calculating PDF and resampling in only particular dimensions/parameters.
        - FIX: Doesn't work quite right for non-fixed bandwidth, bandwidth needs to be re-calculated for different number of dimensions
    - `tests/`
        - No tests currently check that proper errors are raised.
        - Make sure tests check both cases of `num_points > num_data` and visa-versa (e.g. in PDF calculation).




## Current
- Kernels are now implemented as their own classes, allowing for easy subclassing.  Currently `Guassian` and `Box` seem to be working.
- `kdes/`
    - `bandwidths.py`
    - `kernels.py`
        - Classes for each different kernel.
        - `class Gaussian` 
            - Standard Gaussian kernel.
        - `class Box`
            - Boxcar/rectangle kernel with finite support.
    - `utils.py`
        - `stats_str()`
            - Method for calculating percentiles of given data and returing them as a str.




## v0.1 – 2019/05/19
- `kdes/`
    - `__init__.py`
        - `class KDE`
            - Base class for KDE calculations, modeled roughly on the `scipy.stats.gaussian_kde` class.
            - Allows for multidimensional PDF calculation and resampling of data, in multi-dimensional parameter spaces.
            - Reflecting boundary conditions are available in multiple dimensions, both for PDF calculation and resampling.
    - `utils.py`
        - General utility functions for the package.  Methods extracted from the `zcode` package.
        - `midpoints()`
            - Calculate the midpoints between values in an array, either in log or linear space.
        - `minmax()`
            - Calculate the extrema of a given dataset.  Allows for comparison with previous extrema, setting limits, or 'stretching' the return values by a given amount.
        - `spacing()`
            - Construct a linear or log spacing between the given extrema.
    - `tests/`
        - `test_kde.py`
            - Basic tests for the `KDE` base class and its operations.
        - `test_util.py`
            - Basic tests for the utility methods.

- `notebooks/`
    - `kde.ipynb`
        - Includes basic examples and tests with plots.  Mostly the same tests as in the `kdes/tests/` directory, but with plots.
