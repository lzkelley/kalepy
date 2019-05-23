"""Simple utility methods.
"""
import logging

import numpy as np


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


def bound_indices(data, bounds):
    ndim, nvals = np.shape(data)
    idx = np.ones(nvals, dtype=bool)
    for ii, bnd in enumerate(bounds):
        idx = idx & (bnd[0] < data[ii, :]) & (data[ii, :] < bnd[1])
    return idx


def stats_str(data, percs=[0.0, 5.0, 25.0, 50.0, 75.0, 95.0, 100.0]):
    vals = np.percentile(data, percs)
    rv = ", ".join(["{:.2e}".format(xx) for xx in vals])
    rv = "[" + rv + "]"
    return rv


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


def allclose(xx, yy, **kwargs):
    idx = np.isclose(xx, yy, **kwargs)
    if not np.all(idx):
        logging.error("bads : " + array_str(np.where(~idx)[0]))
        logging.error("left : " + array_str(xx[~idx]))
        try:
            logging.error("right: " + array_str(yy[~idx]))
        except TypeError:
            logging.error("right: " + str(yy))

        raise AssertionError("Arrays do not match!")

    return


def alltrue(xx):
    idx = (xx == True)
    if not np.all(idx):
        logging.error("bads : " + array_str(np.where(~idx)[0]))
        logging.error("vals : " + array_str(xx[~idx]))
        raise AssertionError("Not all elements are True!")

    return


def bins(*args):
    xe = np.linspace(*args)
    xc = midpoints(xe)
    dx = np.diff(xe)
    return xe, xc, dx
