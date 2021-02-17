"""Multidimensional kernel density estimation for distribution functions, resampling, and plotting.

Copyright (C) 2020 Luke Zoltan Kelley and Contributors.
"""

import os

# ---- For Testing: uncomment to raise errors on all warnings
# import warnings
# import numpy as np
# cat = Warning
# cat = np.VisibleDeprecationWarning
# warnings.simplefilter("error", category=cat)
# -----------------------------------------------------------

# Load the version information stored within the package contents
_path = os.path.dirname(os.path.abspath(__file__))
_vers_path = os.path.join(_path, "VERSION.txt")
with open(_vers_path) as inn:
    _version = inn.read().strip()

__version__ = _version
__author__ = "Luke Zoltan Kelley <lzkelley@northwestern.edu>"
__copyright__ = "Copyright (C) 2020 Luke Zoltan Kelley and Contributors"
# __contributors__ = []
__bibtex__ = (
    """
    @article{kalepy,
      author = {Luke Zoltan Kelley},
      title = {kalepy: a python package for kernel density estimation and sampling},
      journal = {The Journal of Open Source Software},
      publisher = {The Open Journal},
    }
    """
)

# Numerical padding parameter (e.g. to avoid edge issues, etc)
_NUM_PAD = 1e-8
# Zero-out the PDF of kernels with infinite-support beyond this probability
_TRUNCATE_INFINITE_KERNELS = 1e-8
# Default bandwidth calculation method
_BANDWIDTH_DEFAULT = 'scott'

_PATH_NB = os.path.join(_path, os.path.pardir, 'notebooks', '')
_PATH_NB_OUT = os.path.join(_PATH_NB, 'output', '')

from kalepy import kernels  # noqa
from kalepy import utils    # noqa
from kalepy.kde import KDE  # noqa
import kalepy.plot          # noqa
from kalepy.plot import *   # noqa

# cleanup imports and objects so they're not visible in the imported package
del os
del inn
del _version
del _vers_path
del _path


# High Level API Functions
# -----------------------------------

def density(data, points=None, weights=None, reflect=None, probability=False, grid=False, **kwargs):
    """Use a KDE to calculate the density of the given data.

    This function (1) constructs a kernel-density estimate of the distribution function from which the given `data` were sampled; then (2) returns the values of the distribution function at `points` (which are automatically generated if they are not given).

    Arguments
    ---------
    dataset : array_like (N,) or (D,N,)
        Dataset from which to construct the kernel-density-estimate.
        For multivariate data with `D` variables and `N` values, the data must be shaped (D,N).
        For univariate (D=1) data, this can be a single array with shape (N,).

    points : ([D,]M,) array_like of float, or (D,) set of array_like point specifications
        The locations at which the PDF should be evaluated.  The number of dimensions `D` must
        match that of the `dataset` that initialized this class' instance.
        NOTE: If the `params` kwarg (see below) is given, then only those dimensions of the
        target parameters should be specified in `points`.

        The meaning of `points` depends on the value of the `grid` argument:

        * `grid=True`  : `points` must be a set of (D,) array_like objects which each give the
          evaluation points for the corresponding dimension to produce a grid of values.
          For example, for a 2D dataset,
          `points=([0.1, 0.2, 0.3], [1, 2])`,
          would produce a grid of points with shape (3, 2):
          `[[0.1, 1], [0.1, 2]], [[0.2, 1], [0.2, 2]], [[0.3, 1], [0.3, 2]]`,
          and the returned values would be an array of the same shape (3, 2).

        * `grid=False` : `points` must be an array_like (D,M) describing the position of `M`
          sample points in each of `D` dimensions.
          For example, for a 3D dataset:
          `points=([0.1, 0.2], [1.0, 2.0], [10, 20])`,
          describes 2 sample points at the 3D locations, `(0.1, 1.0, 10)` and `(0.2, 2.0, 20)`,
          and the returned values would be an array of shape (2,).

    weights : array_like (N,), None
        Weights corresponding to each `dataset` point.  Must match the number of points `N` in
        the `dataset`.
        If `None`, weights are uniformly set to 1.0 for each value.

    reflect : (D,) array_like, None
        Locations at which reflecting boundary conditions should be imposed.
        For each dimension `D`, a pair of boundary locations (for: lower, upper) must be
        specified, or `None`.  `None` can also be given to specify no boundary at that
        location.  See class docstrings:`Reflection` for more information.

    params : int, array_like of int, None
        Only calculate the PDF for certain parameters (dimensions).
        See class docstrings:`Projection` for more information.

    grid : bool,
        Evaluate the KDE distribution at a grid of points specified by `points`.
        See `points` argument description above.

    probability : bool, normalize the results to sum to unity


    Returns
    -------
    points : array_like of scalar
        Locations at which the PDF is evaluated.
    vals : array_like of scalar
        PDF evaluated at the given points

    """
    kde = KDE(data, weights=weights, reflect=reflect, **kwargs)
    points, vals = kde.density(points, probability=probability, grid=grid)
    return points, vals


def pdf(data, points=None, weights=None, reflect=None, params=None, grid=False, **kwargs):
    """Use a KDE to calculate the probability-density of the given data.

    Wrapper for `kalepy.density(..., probability=True)`.

    Arguments
    ---------
    dataset : array_like (N,) or (D,N,)
        Dataset from which to construct the kernel-density-estimate.
        For multivariate data with `D` variables and `N` values, the data must be shaped (D,N).
        For univariate (D=1) data, this can be a single array with shape (N,).

    points : ([D,]M,) array_like of float, or (D,) set of array_like point specifications
        The locations at which the PDF should be evaluated.  The number of dimensions `D` must
        match that of the `dataset` that initialized this class' instance.
        NOTE: If the `params` kwarg (see below) is given, then only those dimensions of the
        target parameters should be specified in `points`.

        The meaning of `points` depends on the value of the `grid` argument:

        * `grid=True`  : `points` must be a set of (D,) array_like objects which each give the
          evaluation points for the corresponding dimension to produce a grid of values.
          For example, for a 2D dataset,
          `points=([0.1, 0.2, 0.3], [1, 2])`,
          would produce a grid of points with shape (3, 2):
          `[[0.1, 1], [0.1, 2]], [[0.2, 1], [0.2, 2]], [[0.3, 1], [0.3, 2]]`,
          and the returned values would be an array of the same shape (3, 2).

        * `grid=False` : `points` must be an array_like (D,M) describing the position of `M`
          sample points in each of `D` dimensions.
          For example, for a 3D dataset:
          `points=([0.1, 0.2], [1.0, 2.0], [10, 20])`,
          describes 2 sample points at the 3D locations, `(0.1, 1.0, 10)` and `(0.2, 2.0, 20)`,
          and the returned values would be an array of shape (2,).

    weights : array_like (N,), None
        Weights corresponding to each `dataset` point.  Must match the number of points `N` in
        the `dataset`.
        If `None`, weights are uniformly set to 1.0 for each value.

    reflect : (D,) array_like, None
        Locations at which reflecting boundary conditions should be imposed.
        For each dimension `D`, a pair of boundary locations (for: lower, upper) must be
        specified, or `None`.  `None` can also be given to specify no boundary at that
        location.  See class docstrings:`Reflection` for more information.

    params : int, array_like of int, None
        Only calculate the PDF for certain parameters (dimensions).
        See class docstrings:`Projection` for more information.

    grid : bool,
        Evaluate the KDE distribution at a grid of points specified by `points`.
        See `points` argument description above.


    Returns
    -------
    points : array_like of scalar
        Locations at which the PDF is evaluated.
    vals : array_like of scalar
        PDF evaluated at the given points


    """
    pdf = density(data, points=points, weights=weights, reflect=reflect, params=params, grid=grid,
                  probability=True, **kwargs)
    return pdf


def resample(data, size=None, weights=None, reflect=None, keep=None, **kwargs):
    """Use a KDE to resample from a reconstructed density function of the given data.

    This function (1) constructs a kernel-density estimate of the distribution function from which the given `data` were sampled; then, (2) resamples `size` data points from that function.  If `size` is not given, then the same number of points are returned as in the input `data`.

    Arguments
    ---------
    data : ([D,]N,) array_like of scalar, the data to be resampled,
        The input data can be either:
        * 1D array_like (N,) with `N` data points, or
        * 2D array_like (D,N) with `D` parameters, and `N` data points

    size : int, None (default)
        The number of new data points to draw.  If `None`, then the number of `datapoints` is
        used.

    weights : (N,) array_like or `None`,
        The weights of which each data point in `data`.

    reflect : (D,) array_like, None (default)
        Locations at which reflecting boundary conditions should be imposed.
        For each dimension `D`, a pair of boundary locations (for: lower, upper) must be
        specified, or `None`.  `None` can also be given to specify no boundary at that
        location.

    keep : int, array_like of int, None (default)
        Parameters/dimensions where the original data-values should be drawn from, instead of
        from the reconstructed PDF.
        TODO: add more information.

    Returns
    -------
    samples : ([D,]L) ndarray of float
        Newly drawn samples from the PDF, where the number of points `L` is determined by the
        `size` argument.
        If `squeeze` is True (default), and the number of dimensions in the original dataset
        `D` is one, then the returned array will have shape (L,).

    """
    kde = KDE(data, weights=weights, reflect=reflect, **kwargs)
    samps = kde.resample(size=size, keep=keep)
    return samps


'''
def cdf(data, edges=None, **kwargs):
    """Use a KDE to calculate a CDF of the given data.

    Arguments
    ---------
    edges : array_like of scalar or None
        Locations at which to evaluate the CDF.
        If `None`: edges are constructed using the `KDE._guess_edges()` method.
    kwargs : dict
        Additional key-value pair arguments passed to the `KDE.__init__` constructor.

    Returns
    -------
    edges : (N,) array_like of scalar
        Locations at which the CDF is evaluated.
    vals : (N,) array_like of scalar
        CDF evaluated at the given points

    """
    kde = KDE(data, **kwargs)
    # NOTE/FIX: dont need to interpolate to `_guess_edges` as they are the interpolation points!
    if edges is None:
        edges = kde._guess_edges()
        if len(edges) == 1:
            import numpy as np
            edges = np.array(edges).squeeze()

    vals = kde.cdf(edges)
    return edges, vals
'''
