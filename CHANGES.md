## To-Do / Known-Issues
- **Optimization desperately needed**.  Things are done in (generally) the simplest ways, currently, need to be optimized for performance (both speed and memory [e.g. with reflections]).  Especially in the case of finite-support kernels, the calculations can be drastically sped up.  Can also use an approximation for infinite-support kernels, truncating at some threshold value of sigma (or percentile; etc).
- Try using `sp.stats.rv_continuous` as base-class for 'Distribution' to provide functionality like 'ppf' etc.
- `Triweight` kernel is currently NOT-WORKING
  - The distribution is non-unitary for 2D distributions.  This might be a normalization issue when constructing the PDF (i.e. in `Triweight._evaluate()`) --- is this scaling for the nball correct??
- Differences between covariance-matrix elements of numerous orders of magnitude can cause spurious results, in particular in the PDF marginalized over parameters.  See "KDE::Dynamic Range" docstrings.  Currently this is checked for in the `KDE._finalize()` method, at the end of initialization, and a warning is given if the dynamic range seems too large. 

- `kalepy/`
    - Allow for calculating PDF and resampling in only particular dimensions/parameters.
        - FIX: Doesn't work quite right for non-fixed bandwidth, bandwidth needs to be re-calculated for different number of dimensions
    - Re-enable the `Kernel.pdf_grid` method, and have `KDE.pdf_grid` call that.
    - `tests/`
        - No tests currently check that proper errors are raised.
        - Make sure tests check both cases of `num_points > num_data` and visa-versa (e.g. in PDF calculation).
    - `kernels.py`
        - Use meta-classes to register subclasses of `Distribution`.



## Current


## v0.2.4 - 2020/03/25
- `Triweight` kernel temporarily disabled as it's having normalization problems in ND > 1.

- `kalepy/`
    - `kde_base.py`
        - `class KDE`
            - Addition (uncaught) keyword-arguments are passed from `KDE` initialization to `Kernel` initialization, so that additional arguments (e.g. `chunk`) can be passed along.
    - `kernels.py`
        - BUG: `Triweight` kernel is not working --> disabled kernel.
        - `class Kernel`
            - Implemented 'chunking' for resampling calculation.  Currently only reflection.
                - This produces an *extreme* memory and time performance increase.  For certain parameters, empirically a chunk size of ~ 1e5 seems to work best.
            - `resample()`
                - BUG: non-integer values of `size` would result in an error.
        - `class Distribution`
            - Significant improvements to the way CDFs are handled.
            - `ppf()`  [new-function]
                - "Percent point function" the inverse of the CDF (returns quantiles given cumulative-probabilities).
    - `utils.py`
        - `bound_indices()`
            - BUG: error in boolean logic.
        - `check_path()`  [new-function]
            - Create the given path if it does not already exist.
        - `cumtrapz()`  [new-function]
            - Cumulative summation using the trapezoid-rule.  Light wrapper around  the `trapz_dens_to_mass()` function.
        - `modify_exists()`  [new-function]
            - Modify the given filename if it already exists.
        - `run_if()`  [new-function]
            - New functions for running passed methods if the current environment is the target environment.
        - `save_fig()`  [new-function]
            - Save a `matplotlib` figure adding convenience features.
- `docs/`
    - `logo/`
        - Logo associated data files.
- `notebooks/`
    - `performance.ipynb`  [new-file]
        - New notebook for performance checks, comparisons and diagnostics.



## v0.2.3 - 2019/06/17
- Added code producing a `kalepy` logo, which is added to the attached media and README file.
- Updated notebooks to fix a few minor errors.


## v0.2.2 - 2019/06/11
- Significant improvement in memory and speed while resampling with reflecting boundaries by implementing chunking.


## v0.2.1 - 2019/06/09
- `kalepy/`
    - `__init__.py`
        - Import desired API methods into module namespace.  Use `__all__` in both `kernels.py` and `utils.py`.
    - `kde_base.py`
        - `class KDE`
            - Introduce `helper` argument upon initialization which determines if extra checks and verbose feedback are given.
            - Introcuce `bw_rescale` initialization argument to rescale the bw-matrix by some factor (matrix, or array).
            - `pdf_grid()`  [new-function]
                - Convenience / wrapper function to calculate the PDF given the edges of a grid.
    - `kernels.py`
        - Introduce `helper` parameter, see `class KDE`
        - Allow the `keep` parameter to be `True` in which case all parameters are kept, or `False` and none are kept (same as `None`).
        - `_check_reflect()`
            - Add additional checks for where the reflection boundaries are relative to the data-values and bandwidth.
        - `_resample_reflect()`
            - BUG: reflection was actually a periodic boundary (ish), instead of reflection.  Not sure why it was still behaving well in testing...
            - BUG: reflection was unnecessarily duplicating (already reflected) data, making fewer new points valid.
    - `utils.py`
        - `ave_std()`  [new-function]
            - Calculation of (optionally) *weighted* average and standard-deviation.
        - `bound_indices()`
            - Allow boundaries to be `None` (for no boundaries)
        - `percentiles()`  [new-function]
            - Copied from `zcode.math.statistic`, allows for weighted percentiles.
        - `stats_str()`
            - Copied function from `zcode.math.math_core` with more extended functionality.
        - `trapz_dens_to_mass()`
            - New argument `axis` to integrate only along target axes.
        - `trapz_nd()`
            - New argument `axis` to integrate only along target axes.
- `notebooks/`
    - `init.ipy`     [new-file]
        - Convenience script for setting up the imports in each notebook file
    - `utils.ipynb`  [new-file]
        - New notebook for testing/exploring the `utils.py` submodule.




## v0.2 – 2019/06/03
- Module renamed from `kdes` to `kalepy`.
- Notebooks are now included in travis unit testing. 
- Added skeleton for sphinx documentation; not written yet.

- `README.md`
    - Added installation information and basic examples.
- `kalepy/`
    - `bandwidths.py`
    - `kde_base.py`  [new-file]
        - `class KDE`  [new-class]
            - Primary API for using the `kalepy` package.  Uses passed data and options to construct KDEs by interfacing with `Kernel` instances.
            - The `KDE` class calculates the bandwidth and constructs a `kernel` instance, and handles passing the data and covariance matrix to the kernel as needed.
            - `pdf()`
                - Interface to the kernel instance method: `kernel.pdf()`
            - `resample()`
                - Interface to the kernel instance method: `kernel.resample()`
    - `kernels.py`  [new-file]
        - Stores classes and methods for handling the kernels and their underlying distribution functions.
        - NOTE: some of the scaling and normalization does not work properly in multi-dimensions for all kernels.
        - `class Kernel`
            - Stores a covariance-matrix and uses it as needed with a `Distribution` class instance.
        - `class Distribution`
            - Subclassed to implement particular distribution functions to use in a kernel.
            - Agnostic of the data and covariance.  The `Kernel` class handles the covariance matrix and appropriately transforming the data.
        - `class Gaussian(Distribution)`
            - Gaussian/Normal distribution function with infinite support.
        - `class Box_Asym(Distribution)`
            - Boxcar/rectangle/uniform function with finite support.
        - `class Parabola(Distribution)`
            - Epanechnikov kernel-function with finite support.
        - `class Triweight`
            - Cubic kernel, similar to Parabola but with additional smooth derivatives.
            - WARNING: does not currently work in multiple-dimensions (normalization is off).
        - `get_all_distribution_classes()`
            - Method to retrieve a list of all `Distribution` sub-classes.  Mostly used for testing.
        - `get_distribution_class()`
            - Convert from the argument to a `Distribution` subclass as needed.  This argument can convert from a string specification of a distribution function to return the actual class.
    - `utils.py`
        - `class Test_Base`
            - Base-class to use in unittests.
        - `add_cov()`
            - Given a covariance matrix, use a Cholesky decomposition to transform the given data to have that covariance.
        - `allclose()`   [new-function]
            - Convenience function for unittests.
        - `alltrue()`    [new-function]
            - Convenience function for unittests.
        - `array_str()`  [new-function]
            - Format an array (or elements of) for printing.
        - `bins()`       [new-function]
            - Generate bin- edges, centers and widths all together.
        - `bound_indices()`
            - Find the indices of parameter space arrays within given bounds.
        - `cov_from_var_cor()`
            - Construct a covariance matrix given a set of variances of parameters, and the correlations between them.
        - `matrix_invert()`
            - Invert a matrix, following back to SVD if it initially fails.
        - `rem_cov()`
            - Given a covariance matrix, use a Cholesky decomposition to remove that covariance from the given data.
        - `stats_str()`  [new-function]
            - Method for calculating percentiles of given data and returning them as a str.
        - `trapz_dens_to_mass()`
            - Use the ndimensional trapezoid rule to convert from densities on a grid to masses (e.g. PDF to PMF).
    - `tests/`
        - `test_distributions.py`
            - Test the underlying distribution functions.
        - `test_kde.py`
            - Test the top-level KDE class and the accuracy of KDE calculation of PDFs and resampling.
        - `test_kernels.py` [new-file]
            - Tests of the kernels directly.
        - `test_utils.py`
            - Test the utility functions.


- `notebooks/`
    - `kernels.ipynb`  [new-file]
        - Examining / testing the behavior of different kernels specifically.
    - `demo.ipynb`     [new-file]
        - Currently includes the material used in the `README.rst`, should be expanded as a quick demonstration / tutorial of the package.





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
