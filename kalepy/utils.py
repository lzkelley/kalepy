"""kalepy's internal, utility functions.
"""
import logging
import copy

import numpy as np
import scipy as sp
import scipy.linalg  # noqa


# =================================================================================================
# ====    Primary / API Functions    ====
# =================================================================================================


def add_cov(data, cov):
    try:
        color_mat = sp.linalg.cholesky(cov)
    except Exception:
        logging.error("Cholesky decomposition failed!")
        logging.error("cov matrix: {}".format(cov))
        raise

    color_data = np.dot(color_mat.T, data)
    return color_data


def array_str(data, num=3, format=':.2e'):
    spec = "{{{}}}".format(format)

    def _astr(vals):
        try:
            temp = ", ".join([spec.format(dd) for dd in vals])
        except TypeError:
            logging.error("Failed to format object of type: {}, shape: {}!".format(
                type(vals), np.shape(vals)))
            logging.error("Object = '{}'".format(str(vals)))
            raise

        return temp

    if len(data) <= 2*num:
        rv = _astr(data)
    else:
        rv = _astr(data[:num]) + " ... " + _astr(data[-num:])

    rv = '[' + rv + ']'
    return rv


def bins(*args, **kwargs):
    """Calculate `np.linspace(*args)` and return also centers and widths.

    Returns
    -------
    xe : (N,) bin edges
    xc : (N-1,) bin centers
    dx : (N-1,) bin widths

    """
    xe = np.linspace(*args, **kwargs)
    xc = midpoints(xe)
    dx = np.diff(xe)
    return xe, xc, dx


def bound_indices(data, bounds, outside=False):
    """Find the indices of the `data` array that are bounded by the given `bounds`.

    If `outside` is True, then indices for values *outside* of the bounds are returned.
    """
    data = np.atleast_2d(data)
    bounds = np.atleast_2d(bounds)
    ndim, nvals = np.shape(data)
    idx = np.ones(nvals, dtype=bool)
    for ii, bnd in enumerate(bounds):
        if bnd is None or ((len(bnd) == 1) and (bnd[0] is None)):
            idx = idx & ~outside
            continue

        if outside:
            lo = (data[ii, :] < bnd[0]) if (bnd[0] is not None) else False
            hi = (bnd[1] < data[ii, :]) if (bnd[1] is not None) else False
            idx = idx & (lo | hi)
        else:
            lo = True if (bnd[0] is None) else (bnd[0] < data[ii, :])
            hi = True if (bnd[1] is None) else (data[ii, :] < bnd[1])
            idx = idx & (lo & hi)

    return idx


def centroids(edges, data):
    """Calculate the centroids (centers of mass) of each cell in the given grid.

    Parameters
    ----------
    edges : (D,) array_like of array_like
    data : (...) ndarray of scalar

    Returns
    -------
    coms : ndarray (D, ...)

    """
    data = np.asarray(data)
    if really1d(edges):
        edges = [edges]

    ndim = data.ndim

    # shape of vertices ('corners') of each bin
    shp_corners = [2, ] * ndim
    # shape of bins
    shp_bins = [sh - 1 for sh in data.shape]

    # ---- Get the y-values (densities) for each corner, for each bin

    # for a 2D grid, `zz[0, 0, :, :]` would be the lower-left,
    # while `zz[1, 0, :, :]` would be the lower-right
    zz = np.zeros(shp_corners + shp_bins)
    # iterate over all permutations of corners
    #     get a tuple specifying left/right edge for each dimension, e.g.
    #     (0, 1, 0) would be (left, right, left) for 3D
    for idx in np.ndindex(tuple(shp_corners)):
        cut = []
        # for each dimension, get a slicing object to get the left or right edges along that dim
        for dd, ii in enumerate(idx):
            # ii=0 ==> s=':-1'   ii=1 ==> s='1:'
            jj = (ii + 1) % 2     # ii=0 ==> jj=1   ii=1 ==> jj=0
            s = slice(ii, data.shape[dd] - jj)
            cut.append(s)

        # for this corner (`idx`) select the y-values (densities) at that corner
        zz[idx] = data[tuple(cut)]

    # ---- Calculate the centers of mass in each dimension

    coms = np.zeros([ndim, ] + shp_bins)
    for ii in range(ndim):
        # sum over both corners, for each dimension *except* for `ii`
        jj = np.arange(ndim).tolist()
        jj.pop(ii)
        # y1 is the left  corner along this dimension, marginalized (summed) over all other dims
        # y2 is the right corner along this dimension
        y1, y2 = np.sum(zz, axis=tuple(jj))

        # bin width in this dimension, for each bin
        dx = np.diff(edges[ii])
        # make `dx` broadcastable to the same shape as bins (i.e. `shp_bins`)
        cut = [np.newaxis for dd in range(ndim-1)]
        cut.insert(ii, slice(None))
        cut = tuple(cut)
        _dx = dx[cut]

        xstack = [edges[ii][:-1], edges[ii][1:]]
        xstack = [np.asarray(xs)[cut] for xs in xstack]
        xstack = np.asarray(xstack)
        ystack = [y1, y2]
        # we need to know which direction each triangle is facing, find the index of the min y-value
        #     0 is left, 1 is right
        idx_min = np.argmin(ystack, axis=0)[np.newaxis, ...]

        # get the min and max y-values; doesn't matter if left or right for these
        y1, y2 = np.min(ystack, axis=0), np.max(ystack, axis=0)

        # ---- Calculate center of mass for trapezoid

        # - We have marginalized over all dimensions except for this one, so we can consider the 1D
        #   case that looks like this:
        #
        #       /| y2
        #      / |
        #     /  |
        #    |---| y1
        #    |   |
        #    |___|
        #
        # - We will calculate the COM for the rectangle and the triangle separately, and then get
        #   the weighted COM between the two, where the weights are given by the areas
        # - `a1` and `x1` will be the area (i.e. mass) and x-COM for the rectangle.
        #   The x-COM is just the midpoint, because the y-values are the same
        # - `a2` and `x2` will be the area and x-COM for the triangle
        #   NOTE: for the triangle, it's direction matters.  For each bin, `idx_min` tells the
        #         direction: 0 means increasing (left-to-right), and 1 means decreasing.
        a1 = _dx * y1
        a2 = 0.5 * _dx * (y2 - y1)

        x1 = np.mean(xstack, axis=0)
        # get the x-value for the low y-value
        xlo = np.take_along_axis(xstack, idx_min, 0)[0]
        idx_min = idx_min[0]
        # make `dx` for each bin positive or negative, depending on the orientation of the triangle
        x2 = xlo + (2.0/3.0)*_dx*(1 - 2*idx_min)

        x1 = np.broadcast_to(x1, x2.shape)
        # when both areas are zero, use un-weighted average of x1 and x2
        bd = ((a1 + a2) == 0.0)
        coms[ii][bd] = 0.5 * (x1[bd] + x2[bd])
        gd = ~bd
        coms[ii][gd] = (x1[gd] * a1[gd] + x2[gd] * a2[gd]) / (a1[gd] + a2[gd])

    return coms


def cov_keep_vars(matrix, keep, reflect=None):
    matrix = np.array(matrix)
    if (keep is None) or (keep is False):
        return matrix

    if keep is True:
        keep = np.arange(matrix.shape[0])

    keep = np.atleast_1d(keep)
    for pp in keep:
        matrix[pp, :] = 0.0
        matrix[:, pp] = 0.0
        # Make sure this isn't also a reflection axis
        if (reflect is not None) and (reflect[pp] is not None):
            err = "Cannot both 'keep' and 'reflect' about dimension '{}'".format(pp)
            raise ValueError(err)

    return matrix


def cumsum(vals, axis=None):
    """Perform a cumulative sum without flattening the input array.

    See: https://stackoverflow.com/a/60647166/230468

    Arguments
    ---------
    vals : array_like of scalar
        Input values to sum over.
    axis : None or int
        Axis over which to perform the cumulative sum.

    Returns
    -------
    res : ndarray of scalar
        Same shape as input `vals`

    """

    vals = np.asarray(vals)
    nd = _ndim(vals)
    if (axis is not None) or (nd == 1):
        return np.cumsum(vals, axis=axis)

    res = vals.cumsum(-1)
    for ii in range(2, nd+1):
        np.cumsum(res, axis=-ii, out=res)

    return res


def cumtrapz(pdf, edges, prepend=True, axis=None):
    """Perform a cumulative integration using the trapezoid rule.

    Arguments
    ---------
    pdf : array_like of scalar
        Input values (e.g. a PDF) to be integrated.
    edges : [D,] list of (array_like of scalar)
        Edges defining bins along each dimension.
        This should be an array/list of edges for each of `D` dimensions.
    prepend : bool
        Whether or not to prepend zero values along the integrated dimensions.
    axis : None or int
        Axis/Dimension over which to integrate.

    Returns
    -------
    cdf : ndarray of scalar
        Values integrated over the desired axes.
        Shape:
        * If `prepend` is False, the shape of `cdf` will be one smaller than the input `pdf`
        * in all dimensions integrated over.
        * If `prepend` is True, the shape of `cdf` will match that of the input `pdf`.

    """
    # Convert from density to mass using trapezoid rule in each bin
    pmf = trapz_dens_to_mass(pdf, edges, axis=axis)
    # Perform cumulative sumation
    cdf = cumsum(pmf, axis=axis)

    # Prepend zeros to output array
    if prepend:
        ndim = _ndim(cdf)
        temp = [1, 0] if axis is None else [0, 0]
        padding = [temp for ii in range(ndim)]
        if axis is not None:
            padding[axis][0] = 1
        # cdf = _pre_pad_zero(cdf, axis=axis)
        cdf = np.pad(cdf, padding, constant_values=0)

    return cdf


def histogram(data, bins=None, weights=None, density=False, probability=False):
    if bins is None:
        bins = 'auto'
    hist, edges = np.histogram(data, bins=bins, weights=weights, density=False)
    if density:
        hist = hist.astype(float) / np.diff(edges)
    if probability:
        tot = data.size if (weights is None) else np.sum(weights)
        hist = hist.astype(float) / tot
    return hist, edges


def matrix_invert(matrix, helper=True):
    try:
        matrix_inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        if helper:
            logging.warning("singular `matrix`, trying SVD...")
        matrix_inv = np.linalg.pinv(matrix)

    return matrix_inv


def meshgrid(*args, indexing='ij', **kwargs):
    return np.meshgrid(*args, indexing=indexing, **kwargs)


def midpoints(arr, log=False, axis=-1, squeeze=False):
    """Return the midpoints between values in the given array.

    If the given array is N-dimensional, midpoints are calculated from the last dimension.

    Arguments
    ---------
    arr : ndarray of scalars,
        Input array.
    log : bool or None,
        Find midpoints in log-space
    axis : int, sequence, or `None`,
        The axis about which to find the midpoints.  If `None`, find the midpoints along all axes.
        If a sequence (tuple, list, or array), take the midpoints along each specified axis.

    Returns
    -------
    mids : ndarray of floats,
        The midpoints of the input array.
        The resulting shape will be the same as the input array `arr`, except that
        `mids.shape[axis] == arr.shape[axis]-1`.

    """

    if axis is None:
        axis = [ii for ii in range(np.ndim(arr))]
    else:
        axis = np.atleast_1d(axis)

    # Convert to log-space
    if log:
        mids = np.log10(arr)
    else:
        mids = np.array(arr)

    # Take the midpoints along each of the desired axes
    for ax in axis:
        mids = _midpoints_1d(mids, axis=ax)

    if log:
        mids = np.power(10.0, mids)
    if squeeze:
        mids = mids.squeeze()

    return mids


def minmax(data, positive=False, prev=None, stretch=None, log_stretch=None, limit=None):
    if prev is not None:
        assert len(prev) == 2, "`prev` must have length 2."
    if limit is not None:
        assert len(limit) == 2, "`limit` must have length 2."

    # If there are no elements (left), return `prev` (`None` if not provided)
    if np.size(data) == 0:
        return prev

    # Find extrema
    idx = (data > 0.0) if positive else slice(None)
    minmax = np.array([np.min(data[idx]), np.max(data[idx])])

    # Add stretch (relative to center point)
    if (stretch is not None) or (log_stretch is not None):
        if (stretch is not None) and (log_stretch is not None):
            raise ValueError("Only `stretch` OR `log_stretch` can be applied!")

        # Choose the given value
        fact = stretch if (stretch is not None) else log_stretch

        # If a single stretch value is given, duplicate it for both lo and hi sides
        if (fact is not None) and np.isscalar(fact):
            fact = [fact, fact]
        # Make sure size is right
        if (fact is not None) and np.size(fact) != 2:
            raise ValueError("`log_stretch` and `stretch` must be None, scalar or (2,)!")

        # Use log-values as needed (stretching in log-space)
        _minmax = np.log10(minmax) if (log_stretch is not None) else minmax
        # Find the center, and stretch relative to that
        cent = np.average(_minmax)
        _minmax[0] = cent - (1.0 + fact[0])*(cent - _minmax[0])
        _minmax[1] = cent + (1.0 + fact[1])*(_minmax[1] - cent)
        # Convert back to normal-space as needed
        minmax = np.power(10.0, _minmax) if (log_stretch is not None) else _minmax

    # Compare to previous extrema, if given
    if prev is not None:
        if prev[0] is not None:
            minmax[0] = np.min([minmax[0], prev[0]])
        if prev[1] is not None:
            minmax[1] = np.max([minmax[1], prev[1]])

    # Compare to limits, if given
    if limit is not None:
        if limit[0] is not None:
            minmax[0] = np.max([minmax[0], limit[0]]) if not np.isnan(minmax[0]) else limit[0]
        if limit[1] is not None:
            minmax[1] = np.min([minmax[1], limit[1]]) if not np.isnan(minmax[1]) else limit[1]

    return minmax


def parse_edges(data, edges=None, extrema=None, weights=None, params=None,
                nmin=5, nmax=1000, pad=None, refine=1.0, bw=None):
    """
    """
    if _ndim(data) not in [1, 2]:
        err = (
            "`data` (shape: {}) ".format(np.shape(data)) +
            "must have shape (N,) or (D, N) for `N` data points and `D` parameters!"
        )
        raise ValueError(err)

    squeeze = (_ndim(data) == 1)
    data = np.atleast_2d(data)
    npars = np.shape(data)[0]

    if pad is None:
        pad = 1 if extrema is None else 0

    extrema = _parse_extrema(data, extrema=extrema, warn=False, params=params)

    # If `edges` provides a specification for each dimension, convert to npars*[edges]
    if (_ndim(edges) == 0) or (really1d(edges) and (np.size(edges) != npars)):
        edges = [edges] * npars
    elif len(edges) != npars:
        err = "length of `edges` ({}) does not match number of data dimensions ({})!".format(
            len(edges), npars)
        raise ValueError(err)

    if bw is None:
        bw = np.empty((npars, npars), dtype=object)
    edges = [_get_edges_1d(edges[ii], data[ii], extrema[ii],
                           npars, nmin, nmax, pad, weights=weights, refine=refine, bw=bw[ii, ii])
             for ii in range(npars)]

    if squeeze:
        edges = np.squeeze(edges)

    return edges


def iqrange(data, log=False, weights=None):
    """Calculate inter-quartile range of the given data."""
    if log:
        data = np.log10(data)
    iqr = np.subtract(*quantiles(data, percs=[0.75, 0.25], weights=weights))
    return iqr


def quantiles(values, percs=None, sigmas=None, weights=None, axis=None, values_sorted=False):
    """Compute weighted quartiles.

    Taken from `zcode.math.statistics`
    Based on @Alleo answer: http://stackoverflow.com/a/29677616/230468

    Arguments
    ---------
    values: (N,)
        input data
    percs: (M,) scalar [0.0, 1.0]
        Desired percentiles of the data.
    weights: (N,) or `None`
        Weighted for each input data point in `values`.
    values_sorted: bool
        If True, then input values are assumed to already be sorted.

    Returns
    -------
    percs : (M,) float
        Array of percentiles of the weighted input data.

    """
    if (percs is None) and (sigmas is None):
        raise ValueError("Either `percs` or `sigmas` must be provided!")

    values = np.array(values)
    if percs is None:
        percs = sp.stats.norm.cdf(sigmas)

    if _ndim(values) > 1:
        if axis is None:
            values = values.flatten()

    percs = np.array(percs)
    if weights is None:
        weights = np.ones_like(values)
    weights = np.array(weights)
    assert np.all(percs >= 0.0) and np.all(percs <= 1.0), 'percentiles must be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values, axis=axis)
        values = np.take_along_axis(values, sorter, axis=axis)
        weights = np.take_along_axis(weights, sorter, axis=axis)

    weighted_quantiles = np.cumsum(weights, axis=axis) - 0.5 * weights
    weighted_quantiles /= np.sum(weights, axis=axis)[..., np.newaxis]
    if axis is None:
        percs = np.interp(percs, weighted_quantiles, values)
    else:
        values = np.moveaxis(values, axis, -1)
        weighted_quantiles = np.moveaxis(weighted_quantiles, axis, -1)
        percs = [np.interp(percs, weighted_quantiles[idx], values[idx])
                 for idx in np.ndindex(values.shape[:-1])]
        percs = np.array(percs)

    return percs


def really1d(arr):
    """Test whether an array_like is really 1D (i.e. not a jagged ND array).

    Test whether the input array is uniformly one-dimensional, as apposed to (e.g.) a ``ndim == 1``
    list or array of irregularly shaped sub-lists/sub-arrays.  True for an empty list `[]`.

    Arguments
    ---------
    arr : array_like
        Array to be tested.

    Returns
    -------
    bool
        Whether `arr` is purely 1D.

    """
    if _ndim(arr) != 1:
        return False
    # Empty list or array
    if len(arr) == 0:
        return True
    # Each element must be a scalar
    if np.any([np.shape(tt) != () for tt in arr]):
        return False

    return True


def flatten(arr):
    """Flatten a ND array, whether jagged or not, into a 1D array.
    """
    # If this is not an iterable, or it's actually 1D then it's already 'flat'
    if (not np.iterable(arr)) or really1d(arr):
        return arr

    # Flatten each component and combine recursively
    return np.concatenate([flatten(aa) for aa in arr])


def flatlen(arr):
    if not np.iterable(arr):
        return 1

    if really1d(arr):
        return len(arr)

    return np.sum([flatlen(aa) for aa in arr])


def isjagged(arr):
    """Test if the given array is jagged.
    """
    # if np.isscalar(arr) or (np.size(arr) == len(flatten(arr))):
    if np.isscalar(arr) or (np.size(arr) == flatlen(arr)):
        return False
    return True


def isinteger(val, iterable=True):
    """Test whether the given variable is an integer (i.e. `numbers.integral` subclass).

    Parameters
    ----------
    val : object,
        Variable to test.
    iterable : bool,
        Allow argument to be an iterable.

    Returns
    -------
    bool : whether or not the input is an integer, or integer iterable

    """
    if not np.isscalar(val) and not iterable:
        return False

    import numbers
    try:
        dtype = np.asarray(val).dtype.type
    except AttributeError:
        dtype = type(val)

    return issubclass(dtype, numbers.Integral)


def jshape(arr, level=0, printout=False, prepend="", indent="  "):
    """Print the complete shape (even if jagged) of the given array.
    """
    arr = np.asarray(arr, dtype=object)
    if printout:
        print(prepend + indent*level + str(np.shape(arr)))

    if not isjagged(arr):
        return np.shape(arr)

    shape = []
    for aa in arr:
        sh = jshape(aa, level+1, prepend=prepend, indent=indent, printout=printout)
        shape.append(sh)

    shape = tuple([np.shape(arr), tuple(shape)])
    return shape


def rem_cov(data, cov=None):
    if cov is None:
        cov = np.cov(*data)
    color_mat = sp.linalg.cholesky(cov)
    uncolor_mat = np.linalg.inv(color_mat)
    white_data = np.dot(uncolor_mat.T, data)
    return white_data


def run_if(func, target, *args, otherwise=None, **kwargs):
    env = _python_environment()
    if env.startswith(target):
        return func(*args, **kwargs)
    elif otherwise is not None:
        return otherwise(*args, **kwargs)

    return None


def run_if_notebook(func, *args, otherwise=None, **kwargs):
    target = 'notebook'
    return run_if(func, target, *args, otherwise=otherwise, **kwargs)


def run_if_script(func, *args, otherwise=None, **kwargs):
    target = 'script'
    return run_if(func, target, *args, otherwise=otherwise, **kwargs)


def spacing(data, scale='log', num=None, dex=10, **kwargs):
    DEF_NUM_LIN = 20

    if scale.startswith('log'):
        log_flag = True
    elif scale.startswith('lin'):
        log_flag = False
    else:
        raise RuntimeError("``scale`` '%s' unrecognized!" % (scale))

    # Find extrema of values
    span = minmax(data, **kwargs)

    if (num is None):
        if log_flag:
            num_dex = np.fabs(np.diff(np.log10(span)))
            num = int(np.ceil(num_dex * dex)) + 1
        else:
            num = DEF_NUM_LIN

    num = int(num)
    if log_flag:
        spaced = np.logspace(*np.log10(span), num=num)
    else:
        spaced = np.linspace(*span, num=num)

    return spaced


def stats(data, shape=True, sample=3, stats=True):
    rv = ""
    failure = True
    if shape:
        failure = False
        rv += str(np.shape(data))
    if (sample is not None) and (sample is not False):
        failure = False
        rv += " - " + array_str(data, num=sample)
    if stats:
        failure = False
        rv += " - " + stats_str(data)

    if failure:
        raise ValueError("No stats requested!")

    return rv


def stats_str(data, percs=[0.0, 0.16, 0.50, 0.84, 1.00], ave=False, std=False, weights=None,
              format=None, log=False, label_log=True):
    """Return a string with the statistics of the given array.

    Arguments
    ---------
    data : ndarray of scalar
        Input data from which to calculate statistics.
    percs : array_like of scalars in {0, 100}
        Which percentiles to calculate.
    ave : bool
        Include average value in output.
    std : bool
        Include standard-deviation in output.
    format : str
        Formatting for all numerical output, (e.g. `":.2f"`).
    log : bool
        Convert values to log10 before printing.

    Output
    ------
    out : str
        Single-line string of the desired statistics.

    """
    # data = np.array(data).astype(np.float)
    data = np.array(data)

    if log:
        data = np.log10(data)

    percs = np.atleast_1d(percs)

    percs_flag = False
    if (percs is not None) and len(percs):
        percs_flag = True

    out = ""

    if format is None:
        allow_int = False if (ave or std) else True
        format = _guess_str_format_from_range(data, allow_int=allow_int)

    # If a `format` is given, but missing the colon, add the colon
    if len(format) and not format.startswith(':'):
        format = ':' + format
    form = "{{{}}}".format(format)

    # Add average
    if ave:
        out += "ave = " + form.format(np.average(data))
        if std or percs_flag:
            out += ", "

    # Add standard-deviation
    if std:
        out += "std = " + form.format(np.std(data))
        if percs_flag:
            out += ", "

    # Add percentiles
    if percs_flag:
        tiles = quantiles(data, percs, weights=weights).astype(data.dtype)
        out += "(" + ", ".join(form.format(tt) for tt in tiles) + ")"
        out += ", for (" + ", ".join("{:.0f}%".format(100*pp) for pp in percs) + ")"

    # Note if these are log-values
    if log and label_log:
        out += " (log values)"

    return out


def subdivide(xx, num=1, log=False):
    """Subdivide the giving array (e.g. bin edges) by the given factor.

    Arguments
    ---------
    xx : (X,) array_like of scalar,
        Input array to be subdivided.
    num : int,
        Subdivide each bin by this factor.  Subdividing "once" (i.e. num=1) produces 2x number of bins.  In general
        the number of output bins is ``X * (num + 1)``.
    log : bool,
        Subdivide evenly in log-space, instead of linear space (e.g. [0, 10.0] ==> [0.0, 3.16, 10.0])

    Returns
    -------
    div : (X * `num`+1,) ndarray of float
        Subdivided array with a number of elements equal to the length of the input array 'X' times one plus the
        subdivision factor `num`.

    """
    div = np.asarray(xx)
    if log:
        div = np.log10(div)

    dd = np.diff(np.concatenate([div, [0.0]]))[:, np.newaxis]
    dd = dd * np.linspace(0.0, 1.0, num+1, endpoint=False)[np.newaxis, :]
    div = div[:, np.newaxis] + dd
    div = div.flatten()[:-num]
    if log:
        div = 10.0 ** div

    return div


def trapz_nd(data, edges, axis=None):

    if np.isscalar(edges[0]):
        edges = np.atleast_2d(edges)
    shp = [len(ee) for ee in edges]
    if not np.all(np.array(shp) == np.shape(data)):
        err = "Shape of `edges` ({}) does not match data ({})!".format(shp, np.shape(data))
        raise ValueError(err)

    ndim = len(shp)
    if axis is None:
        axis = np.arange(ndim)
    else:
        axis = np.atleast_1d(axis)

    axis = sorted(axis)[::-1]
    tot = np.array(data)
    for ii in axis:
        xx = edges[ii]
        tot = np.trapz(tot, x=xx, axis=ii)

    return tot


def trapz_dens_to_mass(pdf, edges, axis=None):
    """Convert from density to mass, for values on the corner of a grid, using the trapezoid rule.

    Arguments
    ---------
    pdf : array_like
        Density values, computed at the grid edges specified by the `edges` list-of-lists.
    edges : array_like of array_like
        List of edge-locations along each dimension specifying the grid of values at which `pdf`
        are located.
        e.g. `[[x0, x1, ... xn], [y0, y1, ... ym], ...]`
        The length of each sub-list in `edges`, must match the shape of `pdf`.
        e.g. if `edges` is a (3,) list, composed of sub-lists with lengths: `[N, M, L,]` then
        the shape of `pdf` must be `(N, M, L,)`.
    axis : int, array_like int, or None
        Along which axes to convert from density to mass.
        If `None`, apply to all axes.

    Returns
    -------
    mass : array_like
        The `mass` array has as many dimensions as `pdf`, with each dimension one element shorter.
        e.g. if the shape of `pdf` is (N, M, ...), then the shape of `mass` is (N-1, M-1, ...).

    """
    import functools

    # ---- Sanitize / Process arguments
    pdf = np.asarray(pdf)

    # Make sure `edges` is a list/array of list/arrays
    if np.isscalar(edges[0]):
        edges = np.atleast_2d(edges)

    # Make sure the lengths of the `edges` array matches the shape of `pdf`
    shp_inn = np.array([len(ed) for ed in edges])
    ndim = len(shp_inn)
    if not np.all(np.shape(pdf) == shp_inn):
        err = "Shape of pdf ({}) does not match edges ({})!".format(np.shape(pdf), shp_inn)
        raise ValueError(err)

    # if `axis = None` apply to all axess
    if axis is None:
        axis = np.arange(ndim)

    axis = np.atleast_1d(axis)
    axis_ndim = len(axis)
    axis = sorted(axis)

    # ---- Determine basic properties

    # get list of axes not being integrated over
    not_axis = np.arange(ndim)
    for ii in axis[::-1]:
        not_axis = np.delete(not_axis, ii)

    # Determine final output shape: that of `pdf` but one-less along each dimension
    shp_out = np.zeros(ndim, dtype=int)
    # `widths` will be the widths of bins along each axis; broadcastable to final shape `shp_out`
    widths = []
    for ii in range(ndim):
        dim_len_inn = shp_inn[ii]
        if ii in axis:
            shp_out[ii] = dim_len_inn - 1
            wid = np.diff(edges[ii])
        else:
            shp_out[ii] = dim_len_inn
            wid = np.ones(dim_len_inn)

        # Create new axes along all by the current dimension, slice along the current dimension
        cut = [np.newaxis for ii in range(ndim)]
        cut[ii] = slice(None)
        temp = wid[tuple(cut)]
        widths.append(temp)

    # Multiply the widths along each dimension to get the volume of each grid cell
    # `np.product` fails when a dimension has length 1, do manual operation if so
    # See: https://github.com/numpy/numpy/issues/20612
    volumes = functools.reduce(np.multiply, widths)
    '''
    print(f"reduce: {volumes.shape=}")
    try:
        volumes = np.product(np.array(widths, dtype=object), axis=0).astype(float)
        print("VOLUMES SUCCESS = ", np.shape(volumes))
    except ValueError as err:
        logging.info(f"WARNING: using manual multiple after error on `np.product` '{err}'")
        op = np.multiply
        volumes = op.identity
        print(f"{np.shape(volumes)=}")
        for aa in range(len(widths)):
            volumes = op(volumes, widths[ii])
            print(f"\t{np.shape(volumes)=}, {np.shape(widths[ii])=}")
        print("VOLUMES FAILURE = ", np.shape(volumes))

    # NOTE
    print(f"widths={[np.shape(ww) for ww in widths]}")
    print(f"{volumes=}")
    print(f"{shp_out=}")
    '''
    err = f"BAD `volume` shape (volumes={np.shape(volumes)}, shp_out={shp_out})!"
    assert np.all(np.shape(volumes) == shp_out), err

    # ---- Integrate each cell to convert from density to mass

    mass = np.zeros(shp_out)
    # Iterate over left and right edges over all dimensions,
    #    e.g.  ..., [... 0 0], [... 0 1], [... 1 0], [... 1 1]
    for inds in np.ndindex(*([2]*ndim)):
        # We only need a single slice along axes we are *not* integrating
        if np.any([inds[jj] > 0 for jj in not_axis]):
            continue
        # Designate axes we are *not* integrating specially: as `-1`
        inds = [-1 if (ii in not_axis) else inds[ii] for ii in range(ndim)]

        # Along each dimension, take the left-side slice, or the right-side slice if we are
        #    integrating over that dimension, otherwise take the full slice
        cut = [
            slice(0, -1, None) if ii == 0 else
            slice(1, None, None) if ii == 1 else
            slice(None)
            for ii in inds
        ]
        temp = pdf[tuple(cut)]
        mass += (temp * volumes)

    # Normalize the average
    mass /= (2**axis_ndim)

    return mass


# =================================================================================================
# ====    Internal Functions    ====
# =================================================================================================


def _midpoints_1d(arr, axis=-1):
    """Return the midpoints between values in the given array.

    If the given array is N-dimensional, midpoints are calculated from the last dimension.

    Arguments
    ---------
    arr : ndarray of scalars,
        Input array.
    frac : float,
        Fraction of the way between intervals (e.g. `0.5` for half-way midpoints).
    axis : int,
        Which axis about which to find the midpoints.

    Returns
    -------
    mids : ndarray of floats,
        The midpoints of the input array.
        The resulting shape will be the same as the input array `arr`, except that
        `mids.shape[axis] == arr.shape[axis]-1`.

    """

    if not np.isscalar(axis):
        raise ValueError("Input `axis` argument must be an integer less than ndim={np.ndim(arr)}!")

    if (np.shape(arr)[axis] < 2):
        raise RuntimeError("Input ``arr`` does not have a valid shape!")

    # diff = np.diff(arr, axis=axis)

    # skip the last element, or the last axis
    left = [slice(None)] * arr.ndim
    left[axis] = slice(0, -1, None)
    left = tuple(left)
    right = [slice(None)] * arr.ndim
    right[axis] = slice(1, None, None)
    right = tuple(right)

    # start = arr[left]
    # mids = start + frac*diff
    mids = 0.5 * (arr[left] + arr[right])

    return mids


def _get_edges_1d(edges, data, extrema, ndim, nmin, nmax, pad, weights=None, refine=1.0, bw=None):
    """

    Arguments
    ---------
    edges : None, int, or array or scalar
        Specification for bin-edges.
        `None` : number of bins is automatically calculated from `data`
        int : used as number of bins, span is calculated from `data`
        array : used as fixed bin edges, ignores `data`
    data : 1D array of scalar
        Data points from which to calculate bin-edges
    ndim : `None` or int
        Number of dimensions of the data-set, used to calculate an effective number of data-points.

    """

    if _ndim(edges) == 0:
        num_bins = edges
    elif really1d(edges):
        return edges
    else:
        err = "1D `edges` (shape: {}) must be `None`, an integer or a 1D array of edges!".format(
            np.shape(edges))
        raise ValueError(err)

    _num_bins, bin_width, _span = _guess_edges(
        data, extrema=extrema, weights=weights,
        ndim=ndim, num_min=nmin, num_max=nmax, refine=refine, bw=bw)

    # print("utils.py:_get_edges_1d():")
    # print("\t", "num_bins = ", num_bins, "_num_bins = ", _num_bins, "refine = ", refine,
    #       "bin_width = ", bin_width, "_span = ", _span)

    if num_bins is None:
        num_bins = _num_bins

    if pad is not None:
        pad_width = pad * bin_width
        extrema = [extrema[0] - pad_width, extrema[1] + pad_width]

    edges = np.linspace(*extrema, num_bins + 1, endpoint=True)
    return edges


def _guess_edges(data, extrema=None, ndim=None, weights=None, num_min=None, num_max=None, refine=1.0, bw=None):
    if weights is None:
        num_eff = data.size
    else:
        if (not really1d(weights)) or (np.size(weights) != np.size(data)):
            err = "Shape of `weights` ({}) does not match `data` ({})!".format(
                np.shape(weights), np.shape(data))
            raise ValueError(err)

        num_eff = np.sum(weights)**2 / np.sum(weights**2)

    if (ndim is not None) and (num_eff > 100):
        # num_eff = np.power(num_eff, 1.0 / ndim)
        _num_eff = num_eff / ndim**2
        num_eff = np.clip(_num_eff, 100, None)

    any_inf = [0.0] if not np.iterable(extrema) else [ex for ex in extrema if ex is not None]
    any_inf = np.any(~np.isfinite(any_inf))
    if (extrema is None) or any_inf:
        if any_inf:
            err = "Given extrema ({}) contain non-finite values!  Overriding!".format(extrema)
            logging.error(err)
        idx = np.isfinite(data)
        extrema = [data[idx].min(), data[idx].max()]
    span_width = np.diff(extrema)[0]

    # Sturges histogram bin estimator.
    w1 = span_width / (np.log2(num_eff) + 1.0)
    # Freedman-Diaconis histogram bin estimator
    iqr = iqrange(data, log=False, weights=weights)               # get interquartile range
    w2 = 2.0 * iqr * num_eff ** (-1.0 / 3.0)
    w3 = bw / np.sqrt(num_eff) if (bw is not None) else 0.0

    widths = [w1, w2, w3]
    bin_width = [bw for bw in widths if bw > 0.0]
    if len(bin_width) > 0:
        bin_width = min(bin_width)
        bin_width = bin_width / refine
    else:
        err = "`bin_width` is not positive (w1 = {}, w2 = {})!".format(w1, w2)
        logging.warning(err)
        if np.allclose(data, data[0]):
            bin_width = 1e-16
            logging.warning("WARNING: all data is identical! Choosing arbitrary `bin_width`")
        else:
            raise ValueError(err)

    if span_width <= 0.0:
        err = "`span_width` is not positive (span_width={})!".format(span_width)
        logging.warning(err)
        if np.allclose(data, data[0]):
            span_width = 10*bin_width
            logging.warning("WARNING: all data is identical! Choosing arbitrary `span_width`")
        else:
            raise ValueError(err)

    num_bins = int(np.ceil(span_width / bin_width))
    if (num_min is not None) or (num_max is not None):
        num_bins = np.clip(num_bins, num_min, num_max)

    # print(f"{num_bins=}, {bin_width=}, {extrema=}, {widths=}")

    return num_bins, bin_width, extrema


def _guess_str_format_from_range(arr, prec=2, log_limit=2, allow_int=True):
    """
    """

    try:
        extr = np.log10(np.fabs(minmax(arr[arr != 0.0])))
    # string values will raise a `TypeError` exception
    except (TypeError, AttributeError):
        return ":"

    if any(extr < -log_limit) or any(extr > log_limit):
        form = ":.{precision:d}e"
    elif np.issubdtype(arr.dtype, np.integer) and allow_int:
        form = ":d"
    else:
        form = ":.{precision:d}f"

    form = form.format(precision=prec)

    return form


def _prep_msg(msg=None):
    if (msg is None) or (msg is True):
        msg_fail = "FAILURE:: arrays do not match!"
        if msg is True:
            msg_succ = "SUCC:: arrays match"
        else:
            msg_succ = None
    else:
        msg_fail = "FAILURE:: " + msg.format(fail="not ") + "!"
        msg_succ = "SUCCESS:: " + msg.format(fail="")

    return msg_succ, msg_fail


def _python_environment():
    """Tries to determine the current python environment, one of: 'jupyter', 'ipython', 'terminal'.
    """
    try:
        # NOTE: `get_ipython` is builtin (i.e. should not be explicitly imported from anything)
        ipy_str = str(type(get_ipython())).lower()  # type: ignore
        if 'zmqshell' in ipy_str:
            return 'notebook'
        if 'terminal' in ipy_str:
            return 'ipython'
    except NameError:
        return 'script'


def _ndim(vals):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.ndim(vals)


def _parse_extrema(data, extrema=None, params=None, warn=True):
    """Get extrema (min and max) consistent with the given `data`.

    `data` must have shape (D, N) for `D` parameters/dimensions, and `N` data points.

    `extrema` can be:
        None: extrema are calculated
        (2,): extrema taken as the same for each dimension
        (D, 2): extrema given for each dimension

        In the latter two cases, any values of `extrema` that are 'None' are filled in using
        extrema from the data.

    """

    # `data` must be shaped as (D, N)
    if _ndim(data) != 2:
        err = "`data` must be shaped (D, N) for `D` dimensions/parameters and `N` data points!"
        raise ValueError(err)

    npars = np.shape(data)[0]

    data_extrema = [minmax(dd) for dd in data]
    if extrema is not None:
        extrema = copy.deepcopy(extrema)

    if extrema is None:
        extrema = data_extrema
    elif (params is not None) and (len(extrema) != npars):
        extrema = [extrema[pp] for pp in params]

    # Check components of given `extrema` to make sure they are valid
    #   fill in any `None` values with extrema from the data
    else:
        # Convert from (2,) ==> (D, 2)
        if really1d(extrema) and (np.shape(extrema) == (2,)):
            extrema = [extrema for ii in range(npars)]
        # If already (D, 2) we're good, keep going
        elif np.shape(np.array(extrema, dtype=object)) == (npars, 2):
            pass
        # If jagged (D,) array
        elif len(extrema) == npars:
            if really1d(extrema) or not np.all([(ee is None) or (len(ee) == 2) for ee in extrema]):
                err = "Each element of jagged `extrema` must be `None` or have length 2!"
                raise ValueError(err)
        # Otherwise bad
        else:
            err = "`extrema` shape '{}' unrecognized for {} parameters!".format(
                jshape(extrema), npars)
            raise ValueError(err)

        # Fill in `None` values and check if given `extrema` is out of bounds for data
        for dd in range(npars):
            if extrema[dd] is None:
                extrema[dd] = [None, None]

            for ii in range(2):
                if extrema[dd][ii] is None:
                    extrema[dd][ii] = data_extrema[dd][ii]

            if warn:
                if (extrema[dd][0] > data_extrema[dd][0]):
                    msg = "lower `extrema` in dimension {} ({}) is above data min: {}!".format(
                        dd, extrema[dd][0], data_extrema[dd][0])
                    logging.warning(msg)

                if (extrema[dd][1] < data_extrema[dd][1]):
                    msg = "lower `extrema` in dimension {} ({}) is below data max: {}!".format(
                        dd, extrema[dd][1], data_extrema[dd][1])
                    logging.warning(msg)

    return extrema


# =================================================================================================
# ====    Internal Convenience Functions    ====
# =================================================================================================


def _random_data_1d_01(num=1e4):
    num = int(num)
    np.random.seed(12345)
    _d1 = np.random.normal(4.0, 1.0, num//2)
    _d2 = np.random.lognormal(0, 0.5, size=num - _d1.size)
    data = np.concatenate([_d1, _d2])

    xx = np.linspace(0.0, 7.0, 200)[1:]
    yy = 0.5*np.exp(-(xx - 4.0)**2/2) / np.sqrt(2*np.pi)
    yy += 0.5 * np.exp(-np.log(xx)**2/(2*0.5**2)) / (0.5*xx*np.sqrt(2*np.pi))

    truth = [xx, yy]
    return data, truth


def _random_data_1d_02(num=1e4):
    num = int(num)
    np.random.seed(12345)
    _d1 = np.random.normal(1.0, 1.0, num//2)
    _d2 = np.random.uniform(1.0, 3.0, size=num - _d1.size)
    data = np.concatenate([_d1, _d2])

    # xx = np.linspace(0.0, 7.0, 200)[1:]
    # yy = 0.5*np.exp(-(xx - 4.0)**2/2) / np.sqrt(2*np.pi)
    # yy += 0.5 * np.exp(-np.log(xx)**2/(2*0.5**2)) / (0.5*xx*np.sqrt(2*np.pi))
    #
    # truth = [xx, yy]
    return data


def _random_data_2d_01(num=1e3, noise=0.2):
    num = int(num)

    sigma = [1.0, 0.2]
    corr = [
        [+1.5, -0.5],
        [-0.5, +1.0]
    ]

    cov = np.zeros_like(corr)
    for (ii, jj), cc in np.ndenumerate(corr):
        cov[ii, jj] = cc * sigma[ii] * sigma[jj]

    data = np.random.multivariate_normal(np.zeros_like(sigma), cov, num).T
    dd = data[1, :]
    dd = (dd - dd.min())/dd.max()
    data *= np.sqrt(dd)[np.newaxis, :]

    nn = int(num*noise)
    noise = [np.random.normal(data[ii, :].mean(), 2*ss, nn) for ii, ss in enumerate(sigma)]
    sel = np.random.choice(num, nn, replace=False)
    data[:, sel] = np.array(noise)
    return data


def _random_data_2d_02(num=1e3, noise=0.2):
    num = int(num)

    xx = np.random.normal(1.0, 0.5, num)
    yy = np.random.uniform(0.0, 2.0, num)
    data = [xx, yy]
    data = [np.sort(aa) for aa in data]
    xx, yy = data
    idx = np.arange(num)
    sel = np.random.choice(num, num//4, replace=False)
    idx[sel] = np.random.choice(num, num//4, replace=False)
    # np.random.shuffle(yy[np.random.choice(num, num//2, replace=False)])
    data = [xx, yy[idx]]

    # idx = np.arange(num)
    # np.random.shuffle(idx)
    # data = [zz[idx] for zz in data]
    # idx = np.random.choice(num, num//2, replace=False)
    # np.random.shuffle(np.array(data)[:, idx])

    return data


def _random_data_2d_03(num=1e3):
    num = int(num)

    aa = np.random.lognormal(0.0, 0.3, size=3*num) - 1
    aa += np.random.uniform(-1.0, 1.0, aa.size)
    aa = aa[aa > 0.0]
    aa = np.random.choice(aa, size=num, replace=False)
    cc = np.random.power(2, size=num)

    # Make `aa` and `cc` strongly covariant
    COV = 0.1    # the smaller the value, the stronger the covariance
    aa = np.sort(aa)
    c1 = cc / cc.max()
    a1 = (aa / aa.max())**(1/4)
    xx = a1**2 + c1**2 + np.random.normal(0.0, COV, size=a1.size)
    idx = np.argsort(xx)
    cc = cc[idx]

    #    unsort `aa` and keep covariance
    idx = np.arange(aa.size)
    np.random.shuffle(idx)
    aa = aa[idx]
    cc = cc[idx]

    data = [aa, cc]
    return data


def _random_data_3d_01(num=1e3):
    num = int(num)

    sigma = [1.0, 0.2, 1.5]
    corr = [
        [+1.4, +0.8, +0.4],
        [+0.8, +1.0, -0.25],
        [+0.4, -0.25, +1.0]
    ]

    cov = np.zeros_like(corr)
    for (ii, jj), cc in np.ndenumerate(corr):
        cov[ii, jj] = cc * sigma[ii] * sigma[jj]

    data = np.random.multivariate_normal(np.zeros_like(sigma), cov, num).T
    dd = data[1, :]
    dd = (dd - dd.min())/dd.max()
    data *= np.sqrt(dd)[np.newaxis, :]

    pc = 0
    extr = [np.percentile(dd, [0+pc, 100-pc]) for dd in data]
    noise = [np.random.uniform(*ex, num//5) for ex in extr]
    data = np.append(data, noise, axis=1)

    return data


def _random_data_3d_02(num=1e3, noise=0.2):
    num = int(num)

    sigma = [1.0, 0.2, 1.5]
    corr = [
        [+1.4, +0.8, +0.4],
        [+0.8, +1.0, -0.25],
        [+0.4, -0.25, +1.0]
    ]

    cov = np.zeros_like(corr)
    for (ii, jj), cc in np.ndenumerate(corr):
        cov[ii, jj] = cc * sigma[ii] * sigma[jj]

    data = np.random.multivariate_normal(np.zeros_like(sigma), cov, num).T
    dd = data[1, :]
    dd = (dd - dd.min())/dd.max()
    data *= np.sqrt(dd)[np.newaxis, :]

    nn = int(num*noise)
    noise = [np.random.normal(data[ii, :].mean(), 2*ss, nn) for ii, ss in enumerate(sigma)]
    sel = np.random.choice(num, nn, replace=False)
    data[:, sel] = np.array(noise)
    return data


def _random_data_3d_03(num=1e3, par=[0.0, 0.5], cov=0.1):
    num = int(num)

    aa = np.random.lognormal(*par, size=3*num) - 1
    aa = aa[aa > 0.0]
    aa = np.random.choice(aa, size=num, replace=False)
    bb = np.random.normal(scale=par[1], size=num)
    cc = np.random.power(2, size=num)

    # Make `aa` and `cc` strongly covariant
    # cov = 0.1    # the smaller the value, the stronger the covariance
    aa = np.sort(aa)
    c1 = cc / cc.max()
    a1 = (aa / aa.max())**(1/4)
    xx = a1**2 + c1**2 + np.random.normal(0.0, cov, size=a1.size)
    idx = np.argsort(xx)
    cc = cc[idx]

    #    unsort `aa` and keep covariance
    idx = np.arange(aa.size)
    np.random.shuffle(idx)
    aa = aa[idx]
    cc = cc[idx]

    data = [aa, bb, cc]
    return data
