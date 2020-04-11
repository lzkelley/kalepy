"""
"""

import os

_path = os.path.dirname(os.path.abspath(__file__))
_vers_path = os.path.join(_path, "VERSION.txt")
with open(_vers_path) as inn:
    _version = inn.read().strip()

__version__ = _version
__author__ = "Luke Zoltan Kelley <lzkelley@northwestern.edu>"
__copyright__ = "Copyright 2019 - Luke Zoltan Kelley and contributors"
# __contributors__ = []
# __bibtex__ = ""

# Numerical padding parameter (e.g. to avoid edge issues, etc)
_NUM_PAD = 1e-8
# Zero-out the PDF of kernels with infinite-support beyond this probability
_TRUNCATE_INFINITE_KERNELS = 1e-8
# Default bandwidth calculation method
_BANDWIDTH_DEFAULT = 'scott'

_PATH_NB_OUT = os.path.join(_path, os.path.pardir, 'notebooks', 'output')


from kalepy import kernels   # noqa
from kalepy import utils   # noqa
from kalepy.kde import KDE  # noqa

del os
del inn
del _version
del _vers_path
del _path


# High Level API Functions
# -----------------------------------

def pdf(data, edges=None, reflect=None, **kwargs):
    """Use a KDE to calculate a PDF of the given data.

    Arguments
    ---------
    edges : array_like of scalar or None
        Locations at which to evaluate the PDF.
        If `None`: edges are constructed using the `KDE._guess_edges()` method.
    kwargs : dict
        Additional key-value pair arguments passed to the `KDE.__init__` constructor.

    Returns
    -------
    edges : (N,) array_like of scalar
        Locations at which the PDF is evaluated.
    vals : (N,) array_like of scalar
        PDF evaluated at the given points

    """
    kde = KDE(data, **kwargs)
    if edges is None:
        edges = kde._guess_edges()
        if len(edges) == 1:
            import numpy as np
            edges = np.array(edges).squeeze()

    vals = kde.pdf(edges, reflect=reflect)
    return edges, vals


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
