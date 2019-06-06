"""Simple utility methods.
"""
import logging

import numpy as np
import scipy as sp
import scipy.linalg  # noqa

__all__ = [
    'add_cov', 'array_str', 'bins', 'midpoints',
    'minmax', 'rem_cov', 'spacing', 'stats_str',
    'trapz_nd', 'trapz_dens_to_mass'
]


def add_cov(data, cov):
    color_mat = sp.linalg.cholesky(cov)
    color_data = np.dot(color_mat.T, data)
    return color_data


def array_str(data, num=3, fmt=':.2e'):
    spec = "{{{}}}".format(fmt)

    def _astr(vals):
        temp = ", ".join([spec.format(dd) for dd in vals])
        return temp

    if len(data) <= 2*num:
        rv = _astr(data)
    else:
        rv = _astr(data[:num]) + " ... " + _astr(data[-num:])

    rv = '[' + rv + ']'
    return rv


def allclose(xx, yy, msg=None, **kwargs):
    msg_succ, msg_fail = _prep_msg(msg)
    xx = np.atleast_1d(xx)
    # yy = np.atleast_1d(yy)
    idx = np.isclose(xx, yy, **kwargs)
    if not np.all(idx):
        logging.error("bads : " + array_str(np.where(~idx)[0], fmt=':d'))
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
        logging.error("bads : " + array_str(np.where(~idx)[0], fmt=':d'))
        logging.error("vals : " + array_str(xx[~idx]))
        raise AssertionError(msg_fail)

    if msg_succ is not None:
        print(msg_succ)

    return


def bins(*args):
    xe = np.linspace(*args)
    xc = midpoints(xe)
    dx = np.diff(xe)
    return xe, xc, dx


def bound_indices(data, bounds, outside=False):
    ndim, nvals = np.shape(data)
    idx = np.ones(nvals, dtype=bool)
    for ii, bnd in enumerate(bounds):
        if outside:
            idx = idx & (data[ii, :] < bnd[0]) & (bnd[1] < data[ii, :])
        else:
            idx = idx & (bnd[0] < data[ii, :]) & (data[ii, :] < bnd[1])
    return idx


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


def matrix_invert(matrix, quiet=True):
    try:
        matrix_inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        if quiet:
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


def minmax(data, prev=None, stretch=None, log_stretch=None, limit=None):
    if prev is not None:
        assert len(prev) == 2, "`prev` must have length 2."
    if limit is not None:
        assert len(limit) == 2, "`limit` must have length 2."

    # If there are no elements (left), return `prev` (`None` if not provided)
    if np.size(data) == 0:
        return prev

    # Find extrema
    minmax = np.array([np.min(data), np.max(data)])

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

    if log_flag:
        spaced = np.logspace(*np.log10(span), num=num)
    else:
        spaced = np.linspace(*span, num=num)

    return spaced


def stats_str(data, percs=[0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0]):
    vals = np.percentile(data, percs)
    rv = ", ".join(["{:.2e}".format(xx) for xx in vals])
    rv = "[" + rv + "]"
    return rv


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
