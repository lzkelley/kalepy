"""Simple utility methods.
"""
import logging
import os
import re

import numpy as np
import scipy as sp
import scipy.linalg  # noqa


def add_cov(data, cov):
    color_mat = sp.linalg.cholesky(cov)
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


def assert_true(val, msg=None):
    msg_succ, msg_fail = _prep_msg(msg)
    if not val:
        raise AssertionError(msg_fail)

    if msg_succ is not None:
        print(msg_succ)

    return


def allclose(xx, yy, msg=None, **kwargs):
    msg_succ, msg_fail = _prep_msg(msg)
    xx = np.atleast_1d(xx)
    # yy = np.atleast_1d(yy)
    idx = np.isclose(xx, yy, **kwargs)
    if not np.all(idx):
        logging.error("bads : " + array_str(np.where(~idx)[0], format=':d'))
        logging.error("left : " + array_str(xx[~idx]))
        try:
            logging.error("right: " + array_str(yy[~idx]))
        except (TypeError, IndexError):
            logging.error("right: " + str(yy))

        raise AssertionError(msg_fail)

    if msg_succ is not None:
        print(msg_succ)

    return


def alltrue(xx, msg=None):
    msg_succ, msg_fail = _prep_msg(msg)
    xx = np.atleast_1d(xx)
    idx = (xx == True)
    if not np.all(idx):
        logging.error("bads : " + array_str(np.where(~idx)[0], format=':d'))
        logging.error("vals : " + array_str(xx[~idx]))
        raise AssertionError(msg_fail)

    if msg_succ is not None:
        print(msg_succ)

    return


def ave_std(values, weights=None, **kwargs):
    """
    Return the weighted average and (biased[1]) standard deviation.

    [1]: i.e. we are dividing by the size `n` of values, not `n-1`.
    """
    average = np.average(values, weights=weights, **kwargs)
    variance = np.average((values - average)**2, weights=weights, **kwargs)
    return average, np.sqrt(variance)


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


# NOTE: this is SLOWER
def _bound_indices(data, bounds, outside=False):
    """Find the indices of the `data` array that are bounded by the given `bounds`.

    If `outside` is True, then indices for values *outside* of the bounds are returned.
    """
    data = np.atleast_2d(data)
    bounds = np.atleast_2d(bounds)
    ndim, nvals = np.shape(data)
    # shape = (ndim, 2, nvals)
    shape = (ndim, nvals)

    if outside:
        idx = np.zeros(shape, dtype=int)
    else:
        idx = np.ones(shape, dtype=int)

    for ii, bnd in enumerate(bounds):
        if ((len(bnd) == 1) and (bnd[0] is None)):
            continue

        if bnd[0] is None:
            bnd[0] = -np.inf
        if bnd[1] is None:
            bnd[1] = +np.inf

        temp = np.searchsorted(bnd, data[ii, :])
        temp = (temp + outside) % 2
        idx[ii, :] = temp

    idx = np.product(idx, axis=0).astype(bool)

    return idx


def check_path(fname):
    """Make sure the given path exists. Create directories as needed.
    """
    path, fname = os.path.split(fname)
    if len(path) > 0 and not os.path.exists(path):
        os.makedirs(path)
    return


def cov_from_var_cor(var, corr):
    var = np.atleast_1d(var)
    assert np.ndim(var) == 1, "`var` should be 1D!"
    ndim = len(var)
    # Covariance matrix diagonals should be the variance (of each parameter)
    cov = np.identity(ndim) * var

    if np.isscalar(corr):
        corr = corr * np.ones((ndim, ndim))
    elif np.shape(corr) != (ndim, ndim):
        raise ValueError("`corr` must be either a scalar or (D,D) matrix!")

    # Set the off-diagonals to be the correlation, times the product of standard-deviations
    for ii, jj in np.ndindex(cov.shape):
        if ii == jj:
            continue
        cov[ii, jj] = np.sqrt(var[ii]) * np.sqrt(var[jj]) * corr[ii, jj]

    return cov


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
    nd = np.ndim(vals)
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
            If `prepend` is False, the shape of `cdf` will be one smaller than the input `pdf`
            in all dimensions integrated over.
            If `prepend` is True, the shape of `cdf` will match that of the input `pdf`.

    """
    # Convert from density to mass using trapezoid rule in each bin
    pmf = trapz_dens_to_mass(pdf, edges, axis=axis)
    # Perform cumulative sumation
    cdf = cumsum(pmf, axis=axis)

    # Prepend zeros to output array
    if prepend:
        cdf = _pre_pad_zero(cdf, axis=axis)

    return cdf


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


def midpoints(data, scale='lin', frac=0.5, axis=-1, squeeze=True):
    """Return the midpoints between values in the given array.
    """

    if (np.shape(data)[axis] < 2):
        raise RuntimeError("Input ``arr`` does not have a valid shape!")

    if scale.lower().startswith('lin'):
        log = False
    elif scale.lower().startswith('log'):
        log = True
    else:
        raise ValueError("`scale` must be either 'lin' or 'log'!")

    # Convert to log-space
    if log:
        data = np.log10(data)
    else:
        data = np.array(data)

    diff = np.diff(data, axis=axis)

    cut = [slice(None)] * data.ndim
    cut[axis] = slice(0, -1, None)
    cut = tuple(cut)

    start = data[cut]
    mids = start + frac*diff

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


def modify_exists(path_fname):
    """Modify the given filename if it already exists.
    """
    path_fname = os.path.abspath(path_fname)
    if not os.path.exists(path_fname):
        return path_fname

    fname, vers = _fname_match_vers(path_fname)
    vers = 0 if (vers is None) else vers + 1
    fname = fname.format(vers)
    return fname


def parse_edges(edges, data):

    if np.ndim(data) == 1:
        edges = _get_edges_1d(edges, data, ndim=None)
        return edges
    elif np.ndim(data) != 2:
        raise ValueError("`data` (shape: {}) must be 1D or 2D!".format(np.shape(data)))

    npars = np.shape(data)[0]
    # If `edges` provides a specification for each dimension, convert to npars*[edges]
    if (np.ndim(edges) == 0) or (really1d(edges) and (np.size(edges) != npars)):
        edges = [edges] * npars
    elif len(edges) != npars:
        err = "length of `edges` ({}) does not match number of data dimensions ({})!".format(
            len(edges), npars)
        raise ValueError(err)

    edges = [_get_edges_1d(ee, dd, ndim=npars) for dd, ee in zip(data, edges)]
    return edges


def _get_edges_1d(edges, data, ndim=1):
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

    if np.ndim(edges) == 0:
        num_bins = edges
    elif really1d(edges):
        return edges
    else:
        err = "1D `edges` (shape: {}) must be `None`, an integer or a 1D array of edges!".format(
            np.shape(edges))
        raise ValueError(err)

    num_eff = data.size
    if (ndim is not None):
        # num_eff = np.power(num_eff, 1.0 / ndim)
        num_eff /= ndim**2

    # span = np.fabs(data.max() - data.min())
    span = [data.min(), data.max()]
    span_width = np.diff(span)[0]

    # Sturges histogram bin estimator.
    w1 = span_width / (np.log2(num_eff) + 1.0)
    # Freedman-Diaconis histogram bin estimator
    iqr = np.subtract(*np.percentile(data, [75, 25]))
    w2 = 2.0 * iqr * num_eff ** (-1.0 / 3.0)

    bin_width = min(w1, w2)

    if num_bins is None:
        num_bins = int(np.ceil(span_width / bin_width))

    edges = np.linspace(*span, num_bins + 1, endpoint=True)
    return edges


def percentiles(values, percs=None, sigmas=None, weights=None, axis=None, values_sorted=False):
    """Compute weighted percentiles.

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
    values = np.array(values)
    if percs is None:
        percs = sp.stats.norm.cdf(sigmas)

    if np.ndim(values) > 1:
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
    if np.ndim(arr) != 1:
        return False
    # Empty list or array
    if len(arr) == 0:
        return True
    if np.any(np.vectorize(np.ndim)(arr)):
        return False
    return True


def rem_cov(data, cov=None):
    if cov is None:
        cov = np.cov(*data)
    color_mat = sp.linalg.cholesky(cov)
    uncolor_mat = np.linalg.inv(color_mat)
    white_data = np.dot(uncolor_mat.T, data)
    return white_data


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
            num = np.int(np.ceil(num_dex * dex)) + 1
        else:
            num = DEF_NUM_LIN

    num = int(num)
    if log_flag:
        spaced = np.logspace(*np.log10(span), num=num)
    else:
        spaced = np.linspace(*span, num=num)

    return spaced


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
        tiles = percentiles(data, percs, weights=weights).astype(data.dtype)
        out += "(" + ", ".join(form.format(tt) for tt in tiles) + ")"
        out += ", for (" + ", ".join("{:.0f}%".format(100*pp) for pp in percs) + ")"

    # Note if these are log-values
    if log and label_log:
        out += " (log values)"

    return out


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

    Returns
    -------
    mass : array_like
        The `mass` array has as many dimensions as `pdf`, with each dimension one element shorter.
        e.g. if the shape of `pdf` is (N, M, ...), then the shape of `mass` is (N-1, M-1, ...).

    """

    # Make sure `edges` is a list/array of list/arrays
    if np.isscalar(edges[0]):
        edges = np.atleast_2d(edges)

    # Make sure the lengths of the `edges` array matches the shape of `pdf`
    shp_inn = np.array([len(ed) for ed in edges])
    ndim = len(shp_inn)
    if not np.all(np.shape(pdf) == shp_inn):
        err = "Shape of pdf ({}) does not match edges ({})!".format(np.shape(pdf), shp_inn)
        raise ValueError(err)

    if axis is None:
        axis = np.arange(ndim)

    axis = np.atleast_1d(axis)
    axis_ndim = len(axis)
    axis = sorted(axis)
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
    volumes = np.product(widths, axis=0)
    # NOTE
    assert np.all(np.shape(volumes) == shp_out), "BAD `volume` shape!"

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
        cut = [slice(0, -1, None) if ii == 0 else slice(1, None, None) if ii == 1 else slice(None)
               for ii in inds]
        temp = pdf[tuple(cut)]
        mass += (temp * volumes)
        # Store each left/right approximation, in each dimension
        # mass.append(temp * volumes)

    # Normalize the average
    mass /= (2**axis_ndim)
    # Take the average of each approximation
    # mass = np.sum(mass, axis=0) / (2**ndim)

    return mass


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


def _is_notebook():
    return _python_environment().startswith('notebook')


def _is_script():
    return _python_environment().startswith('script')


def _fname_match_vers(path_fname, digits=2):
    path, fname = os.path.split(path_fname)
    match = re.search('_[0-9]{1,}', fname)
    if match is None:
        fname_comps = fname.split('.')
        idx = 0 if (len(fname_comps) == 1) else -2
        fname_comps[idx] = fname_comps[idx] + '_{{:0{:}d}}'.format(digits)
        fname = ".".join(fname_comps)
        fname = os.path.join(path, fname)
        fname, num = _fname_match_vers(fname.format(0), digits=digits)
        num = num if os.path.exists(fname.format(0)) else None
        return fname, num

    match_str = match.group()
    match_str = match_str.strip('_')
    num_digits = len(match_str)
    num = int(match_str)

    form = "_{{:0{:}d}}".format(num_digits)
    fname = fname.replace('_' + match_str, form)
    fname = os.path.join(path, fname)
    num_max = 10**num_digits - 1
    while os.path.exists(fname.format(num+1)) and (num < num_max - 1):
        num += 1

    return fname, num


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


def _pre_pad_zero(aa, axis=None):
    if axis is None:
        return np.pad(aa, [1, 0])

    aa = np.moveaxis(aa, axis, 0)
    aa = np.concatenate([[np.zeros_like(aa[0])], aa], axis=0)
    aa = np.moveaxis(aa, 0, axis)
    return aa


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
        ipy_str = str(type(get_ipython())).lower()  # noqa
        if 'zmqshell' in ipy_str:
            return 'notebook'
        if 'terminal' in ipy_str:
            return 'ipython'
    except NameError:
        return 'script'


class _DummyError(Exception):
    pass


class Test_Base(object):

    DEF_SEED = 1234

    @classmethod
    def setup_class(cls):
        np.random.seed(cls.DEF_SEED)
        return

    # Override `__getattribute__` to print the class and function name whenever they are called
    def __getattribute__(self, attr):
        value = object.__getattribute__(self, attr)
        if attr.startswith('__'):
            return value

        name = object.__getattribute__(self, "__class__").__name__
        if callable(value):
            print("\n|{}:{}|".format(name, attr))

        np.random.seed(object.__getattribute__(self, "DEF_SEED"))
        return value
