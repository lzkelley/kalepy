"""Kernal basis functions for KDE calculations.
"""
import logging
import six
from collections import OrderedDict

import numpy as np
import scipy as sp   # noqa
import scipy.stats   # noqa

from kalepy import utils

_NUM_PAD = 1e-8


class Kernel_Dist(object):

    _FINITE = None

    '''
    def __init__(self, matrix=None):
        if matrix is None:
            matrix = 1.0
            logging.warning("No `matrix` provided, setting to [[1.0]]!")

        matrix = np.atleast_2d(matrix)
        self._matrix = matrix
        self._matrix_inv = utils.matrix_invert(matrix)
        # NOTE: should this raise an error if the determinant is negative??
        self._norm = np.sqrt(np.fabs(np.linalg.det(matrix)))
        self._ndim = np.shape(matrix)[0]
        return
    '''

    @classmethod
    def evaluate(self, xx, ref=0.0, bw=1.0, weights=1.0):
        err = "`evaluate` must be overridden by the Kernel_Dist subclass!"
        raise NotImplementedError(err)

    @classmethod
    def sample(cls, ndim, cov, size):
        grid, cdf = cls._cdf_grid(0.0, 1.0)
        samps = np.random.uniform(0.0, 1.0, ndim*size)
        samps = sp.interpolate.interp1d(cdf, grid, kind='quadratic')(samps).reshape(ndim, size)
        samps = utils.add_cov(samps)
        samps = utils.add_cov(samps, cov)
        return samps

    @classmethod
    def cdf(cls, xx, ref=0.0, bw=1.0):
        zz = sp.interpolate.interp1d(*cls._cdf_grid(ref, bw), kind='cubic')(xx)
        return zz

    @classmethod
    def _cdf_grid(cls, ref, bw):
        if cls._FINITE:
            pad = (1 + _NUM_PAD)
            args = [-pad*bw, pad*bw, 2000]
        else:
            args = [-10*bw, 10*bw, 20000]
        xe, xc, dx = utils.bins(*args)

        yy = cls.evaluate(xc, ref, bw)
        csum = np.cumsum(yy*dx)
        norm = csum[-1]
        if not np.isclose(norm, 1.0, rtol=1e-4):
            err = "Failed to reach unitarity in CDF grid norm: {:.4e}!".format(norm)
            raise ValueError(err)
        # csum = csum / norm
        xc = np.concatenate([[args[0]], [args[0]], xc, [args[1]], [args[1]]], axis=0)
        csum = np.concatenate([[0.0 - _NUM_PAD], [0.0], csum, [1.0], [1.0+_NUM_PAD]], axis=0)
        cdf_grid = [xc, csum]
        return cdf_grid

    '''
    @classmethod
    def scale(self, xx, ref, bw):
        squeeze = (np.ndim(xx) < 2)
        if np.ndim(ref) < 2:
            ref = np.atleast_2d(ref).T
        if np.ndim(bw) < 2:
            bw = np.atleast_2d(bw).T
        xx = np.atleast_2d(xx)
        ndim, nvals = np.shape(xx)
        yy = (xx - ref)/bw
        return yy, ndim, nvals, squeeze
    '''

    @classmethod
    def grid(cls, edges, **kwargs):
        coords = np.meshgrid(*edges)
        shp = np.shape(coords)[1:]
        coords = np.vstack([xx.ravel() for xx in coords])
        pdf = cls.evaluate(coords, **kwargs)
        # print("coords = ", np.shape(coords), "pdf = ", np.shape(pdf), "shp = ", shp)
        pdf = pdf.reshape(shp)
        return pdf


class Gaussian(Kernel_Dist):

    _FINITE = False

    @classmethod
    def evaluate(self, yy):
        ndim, nval = np.shape(yy)
        energy = np.sum(yy * yy, axis=0) / 2.0
        norm = self.norm(ndim)
        result = np.exp(-energy) / norm
        return result

    @classmethod
    def norm(self, ndim=1):
        norm = np.power(2*np.pi, ndim/2)
        return norm

    @classmethod
    def cdf(self, yy):
        zz = sp.stats.norm.cdf(yy)
        return zz

    @classmethod
    def sample(self, size, ndim=None, squeeze=None):
        if ndim is None:
            ndim = 1
            if squeeze is None:
                squeeze = True

        if squeeze is None:
            squeeze = False

        cov = np.eye(ndim)
        samp = np.random.multivariate_normal(np.zeros(ndim), cov, size=size).T
        return samp


class Box_Asym(Kernel_Dist):

    _FINITE = True

    @classmethod
    def evaluate(self, xx, ref=0.0, bw=1.0, weights=1.0):
        yy, ndim, nvals, squeeze = self.scale(xx, ref, bw)
        norm = np.power(2*bw, ndim)
        zz = (weights / norm) * (np.max(np.fabs(yy), axis=0) < 1.0)
        if squeeze:
            zz = zz.squeeze()
        return zz

    @classmethod
    def sample(self, ndim, cov, size):
        samp = np.random.uniform(-1.0, 1.0, size=ndim*size).reshape(ndim, size)
        samp_cov = np.cov(*samp)
        samp = utils.rem_cov(samp, samp_cov)
        samp = utils.add_cov(samp, cov)
        return samp

    @classmethod
    def cdf(self, xx, ref=0.0, bw=1.0):
        yy, ndim, nval, squeeze = self.scale(xx, ref, bw)
        zz = 0.5 + np.minimum(np.maximum(yy, -1), 1)/2
        if squeeze:
            zz = zz.squeeze()
        return zz


class Parabola_Asym(Kernel_Dist):

    _FINITE = True

    @classmethod
    def evaluate(self, xx, ref=0.0, bw=1.0, weights=1.0):
        yy, ndim, nvals, squeeze = self.scale(xx, ref, bw)
        norm = (2*_nball_vol(ndim, bw)) / (ndim + 2)
        zz = np.product(np.maximum(1 - yy**2, 0.0), axis=0)
        zz = zz * weights / norm
        if squeeze:
            zz = zz.squeeze()
        return zz

    @classmethod
    def sample(self, ndim, cov, size):
        # Use the median trick to draw from the Epanechnikov distribution
        samp = np.random.uniform(-1, 1, 3*ndim*size).reshape(ndim, size, 3)
        samp = np.median(samp, axis=-1)
        # Remove intrinsic coveriance
        samp = utils.add_cov(samp)
        # Add desired coveriance
        samp = utils.add_cov(samp, cov)
        return samp

    @classmethod
    def cdf(self, xx, ref=0.0, bw=1.0):
        yy, ndim, nvals, squeeze = self.scale(xx, ref, bw)
        try:
            yy = np.minimum(np.maximum(yy, -1), 1)
        except:
            print(yy)
            raise
        zz = 0.5 + (3/4)*(yy - yy**3 / 3)
        if squeeze:
            zz = zz.squeeze()
        return zz


class Triweight(Kernel_Dist):

    _FINITE = True

    @classmethod
    def evaluate(self, xx, ref=0.0, bw=1.0, weights=1.0):
        yy, ndim, nvals, squeeze = self.scale(xx, ref, bw)
        norm = bw*32/35
        zz = np.product(np.maximum((1 - yy*yy)**3, 0.0), axis=0)
        zz = zz * weights / norm
        if squeeze:
            zz = zz.squeeze()
        return zz

    @classmethod
    def _cdf_grid(cls, ref, bw):
        if cls._FINITE:
            pad = (1 + _NUM_PAD)
            args = [-pad*bw, pad*bw, 2000]
        else:
            args = [-10*bw, 10*bw, 20000]
        xe, xc, dx = utils.bins(*args)

        yy = cls.cdf(xc, ref, bw)
        norm = yy[-1]
        if not np.isclose(norm, 1.0, rtol=1e-4):
            err = "Failed to reach unitarity in CDF grid norm: {:.4e}!".format(norm)
            raise ValueError(err)
        # csum = csum / norm
        xc = np.concatenate([[args[0]], [args[0]], xc, [args[1]], [args[1]]], axis=0)
        yy = np.concatenate([[0.0 - _NUM_PAD], [0.0], yy, [1.0], [1.0+_NUM_PAD]], axis=0)
        cdf_grid = [xc, yy]
        return cdf_grid

    @classmethod
    def cdf(self, xx, ref=0.0, bw=1.0):
        yy, ndim, nvals, squeeze = self.scale(xx, ref, bw)
        yy = np.minimum(np.maximum(yy, -1), 1)
        coeffs = [35/32, -35/32, 21/32, -5/32]
        powers = [1, 3, 5, 7]
        zz = 0.5 + np.sum([aa*np.power(yy, pp) for aa, pp in zip(coeffs, powers)], axis=0)
        if squeeze:
            zz = zz.squeeze()
        return zz


_DEFAULT_KERNEL = Gaussian

_index_list = [
    ['gaussian', Gaussian],
    ['box', Box_Asym],
    ['parabola', Parabola_Asym],
    ['epanechnikov', Parabola_Asym],
    ['triweight', Triweight],
]

_all_skip = [Parabola_Asym, Triweight]

_index = OrderedDict([(nam, val) for nam, val in _index_list])

Parabola = Parabola_Asym
Box = Box_Asym


def get_kernel_class(arg=None):
    if arg is None:
        return _DEFAULT_KERNEL

    if isinstance(arg, six.string_types):
        arg = arg.lower().strip()
        names = list(_index.keys())
        if arg not in names:
            err = "Kernel_Dist '{}' is not in the index.  Choose one of: '{}'!".format(arg, names)
            raise ValueError(err)

        return _index[arg]

    # This will raise an error if `arg` isn't a class at all
    try:
        if issubclass(arg, Kernel_Dist):
            return arg
    except:
        pass

    raise ValueError("Unrecognized Kernel_Dist type '{}'!".format(arg))


def get_all_kernels():
    kerns = []
    for kk in _index.values():
        if kk not in kerns:
            if kk in _all_skip:
                logging.warning("WARNING: skipping kernel '{}'!".format(kk))
                continue
            kerns.append(kk)
    return kerns


def _nball_vol(ndim, rad=1.0):
    vol = np.pi**(ndim/2)
    vol = (rad**ndim) * vol / sp.special.gamma((ndim/2) + 1)
    return vol
