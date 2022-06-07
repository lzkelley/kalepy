"""Kernal basis functions for KDE calculations, used by `kalepy.kde.KDE` class.

Contents:

- :class:`Kernel <kalepy.kernels.Kernel>` :
  class performing the numerical/mathematical functions of a KDE using a particular kernel-function.

- :class:`Distribution <kalepy.kernels.Distribution>` :
  base class for kernel-function functionality

- :class:`Gaussian(Distribution) <kalepy.kernels.Gaussian>` :
  class for Gaussian kernel functions

- :class:`Box_Asym(Distribution) <kalepy.kernels.Box_Asym>` :
  class for box (top-hat) kernel functions

- :class:`Parabola(Distribution) <kalepy.kernels.Parabola>` :
  class for parabolic (Epanechnikov) kernel functions

- :func:`get_distribution_class <kalepy.kernels.get_distribution_class>` :
  returns the appropriate `Distribution` subclass matching the given string specification.

- :func:`get_all_distribution_classes <kalepy.kernels.get_all_distribution_classes>` :
  returns a list of active `Distribution` subclasses (used for testing).

"""
import logging
import abc
from collections import OrderedDict

import numpy as np
import scipy as sp   # noqa
import scipy.stats   # noqa
import numba

from kalepy import utils
from kalepy import _NUM_PAD, _TRUNCATE_INFINITE_KERNELS

_NUM_CDF_INTERP_POINTS = 1e3


# def pdf(xx):
#     return (70.0/32.0) * np.power(1 - xx*xx, 3)


def _triweight_cdf(xx):
    yy = (70.0/32.0) * (((-1.0/7.0) * (xx**7)) + ((3.0/5.0) * (xx**5)) - (xx**3) + xx)
    return yy


xx = np.linspace(0.0, 1.0, int(_NUM_CDF_INTERP_POINTS))
yy = _triweight_cdf(xx)
yy[-1] = 1.0
_triweight_inverse_func = sp.interpolate.interp1d(yy, xx, kind='linear', bounds_error=True)
del xx, yy


class Kernel(object):

    def __init__(self, distribution, bandwidth, covariance, helper=False, chunk=1e5):
        bandwidth = np.atleast_2d(bandwidth)
        covariance = np.atleast_2d(covariance)
        matrix = covariance * np.square(bandwidth)

        self._helper = helper
        self._distribution = distribution()
        self._chunk = int(chunk)
        self._ndim = np.shape(matrix)[0]
        self._matrix = matrix
        self._bandwidth = bandwidth
        self._covariance = covariance
        self._matrix_inv = None
        self._norm = None
        return

    def density(self, points, data, weights=None, reflect=None, params=None):
        """Calculate the Density Function using this Kernel.

        Arguments
        ---------
        points : (D, N), 2darray of float,
            `N` points at which to evaluate the density function over `D` parameters (dimensions).
            Locations must be specified for each dimension of the data, or for each of target
            `params` dimensions of the data.

        """
        matrix_inv = self.matrix_inv
        points = np.atleast_2d(points)
        npar_pnts, num_points = np.shape(points)
        norm = self.norm

        # ----------------    Process Arguments and Sanitize
        matrix = self.matrix
        # Select subset of parameters
        if params is not None:
            params = np.atleast_1d(params)
            matrix = matrix[np.ix_(params, params)]
            # Recalculate norm & matrix-inverse
            norm = np.sqrt(np.linalg.det(matrix))
            matrix_inv = utils.matrix_invert(matrix, helper=self._helper)
            if npar_pnts != len(params):
                err = "Dimensions of `points` ({}) does not match `params` ({})!".format(
                    npar_pnts, len(params))
                raise ValueError(err)

        npar_data, num_data = np.shape(data)
        if npar_pnts != npar_data:
            err = "Dimensions of `data` ({}) does not match `points` ({})!".format(
                npar_data, npar_pnts)
            raise ValueError(err)

        if weights is None:
            weights = np.ones(num_data)
        elif (np.shape(weights) != (num_data,)):
            err = "Shape of `weights` ({}) does not match number of data points ({})!".format(
                np.shpae(weights), num_data)
            raise ValueError(err)

        # -----------------    Calculate Density

        norm *= self.distribution.norm(npar_pnts)
        try:
            whitening = sp.linalg.cholesky(matrix_inv)
        except:
            logging.error("Failed to construct cholesky on {}, {}".format(matrix_inv, matrix))
            raise
        kfunc = self.distribution._evaluate
        result = _evaluate_numba(whitening, data, points, weights, kfunc)

        if reflect is None:
            result = result / norm
            return result

        # -------------------   Perform Reflection

        if (len(reflect) != npar_data):
            err = "ERROR: shape of reflect `{}` does not match data `{}`!".format(
                np.shape(reflect), np.shape(data))
            raise ValueError(err)

        for ii, reflect_dim in enumerate(reflect):
            if reflect_dim is None:
                continue

            for loc in reflect_dim:
                if loc is None:
                    continue

                # shape (D,N) i.e. (dimensions, data-points)
                refl_data = np.array(data)
                refl_data[ii, :] = 2*loc - refl_data[ii, :]
                result += _evaluate_numba(whitening, refl_data, points, weights, kfunc)

            lo = -np.inf if (reflect_dim[0] is None) else reflect_dim[0]
            hi = +np.inf if (reflect_dim[1] is None) else reflect_dim[1]
            idx = (points[ii, :] < lo) | (hi < points[ii, :])
            result[idx] = 0.0

        result = result / norm
        return result

    def resample(self, data, weights=None, size=None, keep=None, reflect=None, squeeze=True):
        """
        """
        ndim, nval = np.shape(data)
        # If a `size` (number of resample points) isn't given, use the number of data points
        if size is None:
            size = nval

        size = int(size)

        # Check if the number of samples being drawn is near the limit imposed by truncation
        trunc_num = int(1/_TRUNCATE_INFINITE_KERNELS)
        if (size/10 > trunc_num) and (not self.FINITE):
            err = "kernel is being truncated, not accurate near {} samples!".format(trunc_num)
            logging.warning(err)

        # Make sure `reflect` matches
        if reflect is not None:
            # This is now either (D,) [and contains `None` values] or (D,2)
            reflect = _check_reflect(reflect, data, weights=weights)

        # Perform resampling
        # -------------------------------
        if reflect is None:
            samples = self._resample_clear(data, size, weights=weights, keep=keep)
        else:
            samples = self._resample_reflect(data, size, reflect, weights=weights, keep=keep)

        if (ndim == 1) and squeeze:
            samples = samples.squeeze()

        return samples

    def _resample_clear(self, data, size, weights=None, matrix=None, keep=None):
        """Resample the given data without reflection.
        """
        if matrix is None:
            matrix = self.matrix

        if (self._chunk is not None) and (self._chunk < size):
            logging.info("Chunk size: {:.2e}, requested size: {:.2e}".format(self._chunk, size))
            logging.warning("Chunking is not setup in `_resample_clear`!")

        ndim, nvals = np.shape(data)
        # Draw from the smoothing kernel, here the `bw_matrix` includes the bandwidth
        norm = self.distribution.sample(size, ndim=ndim, squeeze=False)
        norm = utils.add_cov(norm, matrix)

        if keep is not None:
            if keep is True:
                keep = np.arange(ndim)
            elif keep is False:
                keep = []
            keep = np.atleast_1d(keep)
            for pp in keep:
                norm[pp, :] = 0.0

        if weights is not None:
            weights = np.asarray(weights) / np.sum(weights)

        try:
            indices = np.random.choice(nvals, size=size, p=weights)
        except:
            logging.error(f"`numpy.random.choice` failed.")
            logging.error(f"{np.shape(data)=}, {size=}, {nvals=}, {utils.stats_str(weights)=}")
            raise
        means = data[:, indices]
        # Shift each re-drawn sample based on the kernel-samples
        samps = means + norm
        return samps

    def _resample_reflect(self, data, size, reflect, weights=None, keep=None):
        """Resample the given data using reflection.
        """
        matrix = self.matrix
        # Modify covariance-matrix for any `keep` dimensions
        matrix = utils.cov_keep_vars(matrix, keep, reflect=reflect)

        ndim, nvals = np.shape(data)

        # Actually 'reflect' (append new, mirrored points) around the given reflection points
        #   Also construct bounding box for valid data
        data, bounds, weights = self._reflect_data(data, reflect, weights=weights)

        # Remove data points outside of kernels (or truncated region)
        data, weights = self._truncate_reflections(data, bounds, weights=weights)
        if (data.size == 0):
            err = f"Empty data after reflection and truncation!"
            logging.error(err)
            raise ValueError(err)

        if (self._chunk is not None) and (self._chunk < size):
            num_chunks = int(np.ceil(size/self._chunk))
            chunk_size = int(np.ceil(size/num_chunks))
        else:
            chunk_size = size
            num_chunks = 1

        # Draw randomly from the given data points, proportionally to their weights
        samps = np.zeros((size, ndim))
        num_good = 0
        cnt = 0
        MAX = 10
        draw = chunk_size
        fracs = []
        while (num_good < size) and (cnt < MAX * num_chunks):
            # Draw candidate resample points
            #    set `keep` to None, `matrix` is already modified to account for it
            trial = self._resample_clear(data, draw, weights=weights, matrix=matrix, keep=None)
            # Find the (boolean) indices of values within target boundaries
            idx = utils.bound_indices(trial, bounds)

            # Store good values to output array
            ngd = np.count_nonzero(idx)
            fracs.append(ngd/idx.size)

            if num_good + ngd <= size:
                samps[num_good:num_good+ngd, :] = trial.T[idx, :]
            else:
                ngd = (size - num_good)
                samps[num_good:num_good+ngd, :] = trial.T[idx, :][:ngd]

            # Increment counters
            num_good += ngd
            cnt += 1
            # Next time, draw twice as many as we need
            draw = np.minimum(size - num_good, chunk_size)
            draw = (2**ndim) * draw
            draw = np.minimum(draw, int(self._chunk))

        if num_good < size:
            err = "Failed to draw '{}' samples in {} iterations!".format(size, cnt)
            logging.error("")
            logging.error(err)
            logging.error("fracs = {}\n\t{}".format(utils.stats_str(fracs), fracs))
            logging.error("Obtained {} samples".format(num_good))
            logging.error("Reflect: {}".format(reflect))
            logging.error("Bandwidths: {}".format(np.sqrt(self.matrix.diagonal().squeeze())))
            logging.error("data = ")
            for dd in data:
                logging.error("\t{}".format(utils.stats_str(dd)))

            raise RuntimeError(err)

        samps = samps.T
        return samps

    # ==== Utilities ====

    def _reflect_data(self, data, reflect, weights=None):
        """Reflect the given data about the locations specified by `reflect`.

        If input `weights` is None, then returned `weights` are None.

        """
        bounds = np.zeros((data.shape[0], 2))

        old_data = np.copy(data)
        if weights is not None:
            old_weights = np.copy(weights)

        for ii, reflect_dim in enumerate(reflect):
            if reflect_dim is None:
                bounds[ii, 0] = -np.inf
                bounds[ii, 1] = +np.inf
                continue

            for jj, loc in enumerate(reflect_dim):
                if loc is None:
                    # j=0 : -inf,  j=1: +inf
                    bounds[ii, jj] = np.inf * (2*jj - 1)
                    continue

                bounds[ii, jj] = loc
                new_data = np.array(old_data)
                new_data[ii, :] = loc - (new_data[ii, :] - loc)
                # NOTE: this returns a copy, so original `data` is *not* changed in-place
                data = np.append(data, new_data, axis=-1)
                if weights is not None:
                    weights = np.append(weights, old_weights)

        # Re-normalize the weights
        if weights is not None:
            weights = weights / np.sum(weights)

        return data, bounds, weights

    def _truncate_reflections(self, data, bounds, weights=None):
        # Determine the bounds outside of which we should truncate
        # trunc = self._get_truncation_bounds(bounds)
        trunc = bounds
        # Find the data-points outside of those bounds
        idx = utils.bound_indices(data, trunc)
        # Select only points within truncation bounds
        data = data[:, idx]
        if weights is not None:
            weights = weights[idx]
            weights /= np.sum(weights)

        return data, weights

    def _get_truncation_bounds(self, bounds):
        trunc = np.zeros_like(bounds)

        bw = self.bandwidth.diagonal()
        # If this kernel has finite-support, we only need to go out to 'bandwidth' on each side
        if self.FINITE:
            bw = bw * (1 + _NUM_PAD)
            qnts = np.array([-bw, bw]).T
        # If kernel has infinite-support, go to the quantile reaching the desired precision
        else:
            tol = _TRUNCATE_INFINITE_KERNELS
            qnts = self.distribution.ppf([tol, 1-tol])
            qnts = qnts[np.newaxis, :] * bw[:, np.newaxis]

        # Expand the reflection bounds based on the bandwidth interval
        trunc = bounds + qnts
        return trunc

    # ==== Properties ====

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def covariance(self):
        return self._covariance

    @property
    def distribution(self):
        return self._distribution

    @property
    def matrix(self):
        return self._matrix

    @property
    def matrix_inv(self):
        try:
            if self._matrix_inv is None:
                raise AttributeError
        except AttributeError:
            self._matrix_inv = utils.matrix_invert(self.matrix, helper=self._helper)
        return self._matrix_inv

    @property
    def norm(self):
        if self._norm is None:
            self._norm = np.sqrt(np.fabs(np.linalg.det(self.matrix)))
        return self._norm

    @property
    def FINITE(self):
        return self.distribution.FINITE


class _Distribution(abc.ABC):
    """

    `Distribution` positional arguments (`xx` or `yy`) must be shaped as `(D, N)`
    for 'D' dimensions and 'N' data-points.

    """

    _FINITE = None
    _SYMMETRIC = True

    # ---- Core API Wrapper Functions

    @classmethod
    def evaluate(cls, yy, ndim=None):
        """Evaluate the kernel at the position `yy` (D, N) for D-dimensions.
        """
        if np.ndim(yy) < 2:
            # raise RuntimeError("BAD SHAPE!")
            yy = np.atleast_2d(yy)
        ndim = np.shape(yy)[0]
        y2 = np.linalg.norm(yy, axis=0) ** 2
        return cls._evaluate(y2) / cls.norm(ndim)

    @classmethod
    def sample(cls, size, ndim=None, squeeze=None):
        if ndim is None:
            ndim = 1

        if squeeze is None:
            squeeze = (ndim == 1)

        samps = cls._sample(size, ndim)
        if squeeze:
            samps = samps.squeeze()

        return samps

    # ---- Abstract Methods

    @staticmethod
    @numba.njit
    @abc.abstractmethod
    def _evaluate(cls, y2):
        """Evaluate the kernel at the Euclidean (scalar) distance, y2 (N,) - NOTE: without normalization!
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def norm(ndim):
        pass

    @classmethod
    @abc.abstractmethod
    def _sample(cls, *args, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def inside(points):
        pass

    # ---- Additional Functions

    @classmethod
    def grid(cls, edges):
        coords = np.meshgrid(*edges, indexing='ij')
        shp = np.shape(coords)
        ndim, *shp = shp
        coords = np.vstack([xx.ravel() for xx in coords])
        pdf = cls.evaluate(coords, ndim)
        pdf = pdf.reshape(shp)
        return pdf

    @classmethod
    def name(cls):
        name = cls.__name__
        return name

    @property
    def FINITE(self):
        return self._FINITE

    @property
    def SYMMETRIC(self):
        return self._SYMMETRIC


class Gaussian(_Distribution):

    _FINITE = False

    @staticmethod
    @numba.njit
    def _evaluate(y2):
        return np.exp(-y2 / 2.0)

    @staticmethod
    def norm(ndim):
        norm = np.power(2*np.pi, ndim/2)
        return norm

    @classmethod
    def _sample(cls, size, ndim):
        cov = np.eye(ndim)
        samps = np.random.multivariate_normal(np.zeros(ndim), cov, size=size).T
        return samps

    @staticmethod
    def inside(cls, points):
        if np.ndim(points) == 1:
            shape = points.size
        else:
            ndim, *shape = np.shape(points)
        return np.ones(shape, dtype=bool)


class Box(_Distribution):

    _FINITE = True

    @classmethod
    def norm(self, ndim=1):
        return _nball_vol(ndim)

    @staticmethod
    @numba.njit
    def _evaluate(y2):
        # size = y2.size
        # result = np.zeros(size)
        # for ii in range(size):
        #     if (-1.0 < y2[ii] < 1.0):
        #         result[ii] = 1.0
        #     else:
        #         result[ii] = 0.0
        result = (y2 <= 1.0)
        return result

    @classmethod
    def _sample(cls, size, ndim):
        """Randomly sample from within ndim-ball, using Muller method.
        """
        if ndim == 1:
            samps = np.random.uniform(-1.0, 1.0, size=(ndim, size))
        else:
            samps = np.random.normal(0.0, 1.0, (ndim, size))
            rads = np.sum(samps**2, axis=0) ** 0.5
            norm = np.random.random((ndim, size)) ** (1.0 / ndim)
            samps = norm * samps / rads[np.newaxis, :]
        return samps

    @staticmethod
    def inside(points):
        return _inside_unit_nball(points)


class Parabola(_Distribution):
    """Symmetric parabola.
    """

    _FINITE = True

    @staticmethod
    @numba.njit
    def _evaluate(y2):
        # zz = np.zeros(y2.size)
        # for ii in range(y2.size):
        #     zz[ii] = (1.0 - y2[ii])
        #     if zz[ii] < 0.0:
        #         zz[ii] = 0.0
        zz = np.maximum(1.0 - y2, 0.0)
        return zz

    @staticmethod
    def norm(ndim):
        norm = 2.0 * _nball_vol(ndim) / (ndim + 2.0)
        return norm

    @classmethod
    def _sample(cls, size, ndim):
        # Use the median trick to draw from the Epanechnikov distribution
        samp = np.random.uniform(-1.0, 1.0, (ndim, size, 3))
        samp = np.median(samp, axis=-1)
        return samp

    @staticmethod
    def inside(points):
        return _inside_unit_nball(points)


class Triweight(_Distribution):

    _FINITE = True

    @staticmethod
    def norm(ndim):
        return _beta_kernel_norm(3, ndim)

    @staticmethod
    @numba.njit
    def _evaluate(y2):
        # zz = np.zeros(y2.size)
        # for ii in range(y2.size):
        #     zz[ii] = (1 - y2[ii])**3
        #     if zz[ii] < 0.0:
        #         zz[ii] = 0.0
        zz = np.maximum((1 - y2)**3, 0.0)
        return zz

    @classmethod
    def _sample(cls, size, ndim):
        samps = np.random.normal(0.0, 1.0, (ndim, size))
        rads = np.linalg.norm(samps, axis=0)
        norm = _triweight_inverse_func(np.random.random((ndim, size)))
        samps = norm * samps / rads[np.newaxis, :]
        return samps

    @staticmethod
    def inside(points):
        return _inside_unit_nball(points)


_DEFAULT_DISTRIBUTION = Gaussian

DISTRIBUTIONS = [
    ['gaussian', Gaussian],
    ['box', Box],
    ['parabola', Parabola],
    ['epanechnikov', Parabola],
    ['triweight', Triweight],
]

DISTRIBUTIONS = OrderedDict([(nam, val) for nam, val in DISTRIBUTIONS])

__all__ = ['Kernel', 'get_distribution_class', ]
__all__ += list(DISTRIBUTIONS.keys())


def _get_all_distribution_classes():
    dists = []
    _dists = list(DISTRIBUTIONS.values())
    # dists = np.unique(dists).tolist()
    for dd in _dists:
        if dd in dists:
            continue
        dists.append(dd)

    return dists


def get_distribution_class(arg=None):
    if arg is None:
        return _DEFAULT_DISTRIBUTION

    # This will raise an error if `arg` isn't a class at all
    try:
        if issubclass(arg, _Distribution):
            return arg
    except:
        pass

    try:
        arg = arg.lower().strip()
        return DISTRIBUTIONS[arg]
    except:
        pass

    raise ValueError("Argument '{}' is not a known distribution!".format(arg))


def _inside_unit_nball(points):
    if np.ndim(points) == 1:
        idx = (-1.0 < points < 1.0)
    else:
        points = np.linalg.norm(points, axis=0)
        idx = (points < 1.0)
    return idx


def _nball_vol(ndim, rad=1.0):
    vol = np.pi**(ndim/2)
    vol = (rad**ndim) * vol / sp.special.gamma((ndim/2) + 1)
    return vol


def _beta_kernel_norm(order, ndim):
    """Note: this is the inverse normalization (i.e. divide by this).

    See: Duong-2015 - Spherically symmetric multivariate beta family kernels
         https://doi.org/10.1016/j.spl.2015.05.012
    """
    rr = 3
    bvol = _nball_vol(ndim) / 2.0
    norm = ndim * bvol * sp.special.beta(rr + 1, ndim / 2.0)
    return norm


def _check_reflect(reflect, data, weights=None, helper=False):
    """Make sure the given `reflect` argument is valid given the data shape
    """
    if reflect is None:
        return reflect

    if reflect is False:
        return None

    # NOTE: FIX: Should this happen in the method that calls `_check_reflect`?
    # data = np.atleast_2d(data)
    # ndim, nval = np.shape(data)
    data = np.asarray(data)
    ndim, nval = data.shape
    if reflect is True:
        reflect = [True for ii in range(ndim)]

    if (len(reflect) == 2) and (ndim == 1):
        reflect = np.atleast_2d(reflect)

    if (len(reflect) != ndim):  # and not ((len(reflect) == 2) and (ndim == 1)):
        err = "`reflect` ({},) must match the data with shape ({}) parameters!".format(
            len(reflect), data.shape)
        raise ValueError(err)

    try:
        goods = [(ref is None) or (ref is True) or (len(ref) == 2) for ref in reflect]
    except TypeError as err:
        err = "Invalid `reflect` argument: Error: '{}'".format(err)
        raise ValueError(err)

    if not np.all(goods):
        err = "each row of `reflect` must be `None` or have shape (2,)!  '{}'".format(reflect)
        raise ValueError(err)

    # Perform additional diagnostics
    for ii in range(ndim):
        if (reflect[ii] is True):
            reflect[ii] = [np.min(data[ii])*(1 - _NUM_PAD), np.max(data[ii])*(1 + _NUM_PAD)]
        elif (reflect[ii] is not None) and (True in reflect[ii]):
            if reflect[ii][0] is True:
                reflect[ii][0] = np.min(data[ii])*(1 - _NUM_PAD)
            if reflect[ii][1] is True:
                reflect[ii][1] = np.max(data[ii])*(1 + _NUM_PAD)

        if np.all(np.array(reflect[ii]) != None) and (reflect[ii][0] >= reflect[ii][1]):  # noqa
            err = "Reflect is out of order:  `reflect`[{}] = {}  !".format(ii, reflect[ii])
            raise ValueError(err)

        if helper:
            # Warn if any datapoints are outside of reflection bounds
            bads = utils.bound_indices(data[ii, :], reflect[ii], outside=True)
            if np.any(bads):
                if weights is None:
                    frac = np.count_nonzero(bads) / bads.size
                else:
                    frac = np.sum(weights[bads]) / np.sum(weights)
                msg = (
                    "A fraction {:.2e} of data[{}] ".format(frac, ii) +
                    " are outside of `reflect` bounds!"
                )
                logging.warning(msg)
                msg = (
                    "`reflect[{}]` = {}; ".format(ii, reflect[ii]) +
                    "`data[{}]` = {}".format(ii, utils.stats_str(data[ii], weights=weights))
                )
                logging.warning(msg)
                logging.warning("I hope you know what you're doing.")

    return reflect


@numba.njit
def _evaluate_numba(white, data, points, weights, kfunc):
    ndim, ndata = data.shape
    _, npoints = points.shape
    zz = np.zeros(npoints)
    for ii in range(npoints):
        # Sum over kernels for each `data`-value at this sample-`point`
        for jj in range(ndata):
            y2ij = 0.0
            # Calculate Euclidean distance between whitened-data and whitened-points
            for kk in range(ndim):
                xhki = 0.0
                dhkj = 0.0
                # Whiten data and points
                for ll in range(ndim):
                    xhki += white[kk, ll] * points[ll, ii]
                    dhkj += white[kk, ll] * data[ll, jj]
                ss = (xhki - dhkj)
                y2ij += ss * ss

            # zz[ii] += np.exp(- y2ij / 2.0) * weights[jj]
            zz[ii] += kfunc(y2ij) * weights[jj]

    return zz
