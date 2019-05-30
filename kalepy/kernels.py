"""Kernal basis functions for KDE calculations.
"""
import logging
import six
from collections import OrderedDict

import numpy as np
import scipy as sp   # noqa
import scipy.stats   # noqa

from kalepy import utils
from kalepy import _QUIET, _NUM_PAD


class Kernel(object):

    def __init__(self, distribution=None, matrix=None, bandwidth=None):
        distribution = get_distribution_class(distribution)
        self._distribution = distribution()

        if matrix is None:
            if bandwidth is not None:
                matrix = np.square(bandwidth)
            else:
                matrix = 1.0
                logging.warning("No `matrix` of `bandwidth` provided, setting to [[1.0]]!")

        matrix = np.atleast_2d(matrix)
        self._ndim = np.shape(matrix)[0]
        self._matrix = matrix

        self._matrix_inv = None
        self._norm = None
        return

    def pdf(self, points, data, weights, reflect=None, **kwargs):
        pnts = np.atleast_2d(points)
        ndim, nval = np.shape(data)

        # Make sure shape/values of reflect look okay
        reflect = self._check_reflect(reflect, ndim)

        if reflect is None:
            result = self._pdf_clear(pnts, data, weights, **kwargs)
        else:
            result = self._pdf_reflect(pnts, data, weights, reflect, **kwargs)

        return result

    def _pdf_clear(self, pnts, data, weights, params=None):
        matrix_inv = self.matrix_inv
        norm = self.norm

        if params is not None:
            matrix = self.matrix
            data, matrix, norm = self._params_subset(data, matrix, params)
            # matrix_inv = np.linalg.pinv(matrix)
            matrix_inv = utils.matrix_invert(matrix, quiet=_QUIET)

        ndim, num_points = np.shape(pnts)

        whitening = sp.linalg.cholesky(matrix_inv)
        # Construct the whitened sampling points
        white_points = np.dot(whitening, pnts)

        result = np.zeros((num_points,), dtype=float)
        ndim, num_data = np.shape(data)
        # Construct the 'whitened' (independent) dataset
        white_dataset = np.dot(whitening, data)

        for ii in range(num_data):
            yy = white_points - white_dataset[:, ii, np.newaxis]
            temp = weights[ii] * self.distribution.evaluate(yy)
            result += temp.squeeze()

        result = result / norm
        return result

    def _pdf_reflect(self, pnts, data, weights, reflect):
        matrix_inv = self.matrix_inv
        norm = self.norm

        ndim, num_data = np.shape(data)
        ndim, num_points = np.shape(pnts)
        result = np.zeros((num_points,), dtype=float)

        whitening = sp.linalg.cholesky(matrix_inv)
        # Construct the 'whitened' (independent) dataset
        white_dataset = np.dot(whitening, data)
        # Construct the whitened sampling points
        white_points = np.dot(whitening, pnts)

        for ii in range(num_data):
            yy = white_points - white_dataset[:, ii, np.newaxis]
            result += weights[ii] * self.distribution.evaluate(yy)

        for ii, reflect_dim in enumerate(reflect):
            if reflect_dim is None:
                continue

            for loc in reflect_dim:
                if loc is None:
                    continue

                # shape (D,N) i.e. (dimensions, data-points)
                refl_data = np.array(data)
                refl_data[ii, :] = 2*loc - refl_data[ii, :]
                white_dataset = np.dot(whitening, refl_data)
                # Construct the whitened sampling points
                #    shape (D,M) i.e. (dimensions, sample-points)
                pnts = np.array(pnts)
                white_points = np.dot(whitening, pnts)

                if num_points >= num_data:
                    for jj in range(num_data):
                        yy = white_points - white_dataset[:, jj, np.newaxis]
                        result += weights[jj] * self.distribution.evaluate(yy)
                else:
                    for jj in range(num_points):
                        yy = white_dataset - white_points[:, jj, np.newaxis]
                        res = weights * self.distribution.evaluate(yy)
                        result[jj] += np.sum(res, axis=0)

            lo = -np.inf if reflect_dim[0] is None else reflect_dim[0]
            hi = +np.inf if reflect_dim[1] is None else reflect_dim[1]
            idx = (pnts[ii, :] < lo) | (hi < pnts[ii, :])
            result[idx] = 0.0

        result = result / norm
        return result

    def resample(self, data, weights, size=None, keep=None, reflect=None, squeeze=True):
        """
        """
        ndim, nval = np.shape(data)
        if size is None:
            # size = int(self.neff)
            size = nval

        # Make sure `reflect` matches
        if reflect is not None:
            # This is now either (D,) [and contains `None` values] or (D,2)
            reflect = self._check_reflect(reflect, ndim)

        # Have `Distribution` class perform resampling
        # ---------------------------------------------------
        if reflect is None:
            samples = self._resample_clear(data, weights, size, keep=keep)
        else:
            samples = self._resample_reflect(data, weights, size, reflect, keep=keep)

        if (ndim == 1) and squeeze:
            samples = samples.squeeze()

        return samples

    def _resample_clear(self, data, weights, size, matrix=None, keep=None):
        if matrix is None:
            matrix = self.matrix
            # matrix = self._cov_keep_vars(matrix, keep)

        ndim, nvals = np.shape(data)
        # Draw from the smoothing kernel, here the `bw_matrix` includes the bandwidth
        norm = self.distribution.sample(size, ndim=ndim)
        # norm_cov = np.cov(*norm)
        # norm = utils.rem_cov(norm, norm_cov)
        norm = utils.add_cov(norm, matrix)
        if keep is not None:
            keep = np.atleast_1d(keep)
            for pp in keep:
                norm[pp, :] = 0.0

        indices = np.random.choice(nvals, size=size, p=weights)
        means = data[:, indices]
        # Shift each re-drawn sample based on the kernel-samples
        samps = means + norm
        return samps

    def _resample_reflect(self, data, weights, size, reflect, keep=None):
        # wgts = self.weights
        # data = self.dataset
        matrix = self.matrix
        matrix = self._cov_keep_vars(matrix, keep, reflect=reflect)

        ndim, nvals = np.shape(data)
        bounds = np.zeros((ndim, 2))

        # Actually 'reflect' (append new, mirrored points) around the given reflection points
        # Also construct bounding box for valid data
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
                new_data = np.array(data)
                new_data[ii, :] = new_data[ii, :] - loc
                data = np.append(data, new_data, axis=-1)
                weights = np.append(weights, weights, axis=-1)

        weights = weights / np.sum(weights)

        # Draw randomly from the given data points, proportionally to their weights
        samps = np.zeros((size, ndim))
        num_good = 0
        cnt = 0
        MAX = 10
        draw = size
        while num_good < size and cnt < MAX:
            # Draw candidate resample points
            #    set `keep` to None, `matrix` is already modified to account for it
            trial = self._resample_clear(data, weights, draw, matrix=matrix, keep=None)
            # Find the (boolean) indices of values within target boundaries
            idx = utils.bound_indices(trial, bounds)

            # Store good values to output array
            ngd = np.count_nonzero(idx)
            if num_good + ngd <= size:
                samps[num_good:num_good+ngd, :] = trial.T[idx, :]
            else:
                ngd = (size - num_good)
                samps[num_good:num_good+ngd, :] = trial.T[idx, :][:ngd]

            # Increment counters
            num_good += ngd
            cnt += 1
            # Next time, draw twice as many as we need
            draw = 2*(size - num_good)

        if num_good < size:
            raise RuntimeError("Failed to draw '{}' samples in {} iterations!".format(size, cnt))

        samps = samps.T
        return samps

    # ==== Utilities ====

    def _check_reflect(self, reflect, ndim):
        if reflect is None:
            return reflect

        if ndim == 1 and np.ndim(reflect) == 1:
            reflect = np.atleast_2d(reflect)

        if len(reflect) != ndim:
            msg = "`reflect` ({}) must have length (D,) = ({},)!".format(
                len(reflect), ndim)
            raise ValueError(msg)
        if not np.all([(ref is None) or len(ref) == 2 for ref in reflect]):
            raise ValueError("each row of `reflect` must be `None` or shape (2,)!")

        return reflect

    @classmethod
    def _cov_keep_vars(cls, matrix, keep, reflect=None):
        matrix = np.array(matrix)
        if keep is None:
            return matrix

        keep = np.atleast_1d(keep)
        for pp in keep:
            matrix[pp, :] = 0.0
            matrix[:, pp] = 0.0
            # Make sure this isn't also a reflection axis
            if (reflect is not None) and (reflect[pp] is not None):
                err = "Cannot both 'keep' and 'reflect' about dimension '{}'".format(pp)
                raise ValueError(err)

        return matrix

    @classmethod
    def _params_subset(cls, data, matrix, params):
        if params is None:
            norm = np.sqrt(np.linalg.det(matrix))
            return data, matrix, norm

        params = np.atleast_1d(params)
        params = sorted(params)
        # Get rows corresponding to these parameters
        sub_data = data[params, :]
        # Get rows & cols corresponding to these parameters
        sub_mat = matrix[np.ix_(params, params)]
        # Recalculate norm
        norm = np.sqrt(np.linalg.det(sub_mat))
        return sub_data, sub_mat, norm

    # ==== Properties ====

    @property
    def distribution(self):
        return self._distribution

    @property
    def norm(self):
        try:
            if self._norm is None:
                raise AttributeError
        except AttributeError:
            self._norm = np.sqrt(np.fabs(np.linalg.det(self.matrix)))

        return self._norm

    @property
    def matrix(self):
        return self._matrix

    @property
    def matrix_inv(self):
        try:
            if self._matrix_inv is None:
                raise AttributeError
        except AttributeError:
            self._matrix_inv = utils.matrix_invert(self.matrix, quiet=_QUIET)
        return self._matrix_inv

    @property
    def FINITE(self):
        return self.distribution.FINITE


class Distribution(object):

    _FINITE = None

    def __init__(self):
        self._cdf_grid = None
        return

    @classmethod
    def name(cls):
        name = cls.__name__
        return name

    @classmethod
    def _parse(self, xx):
        squeeze = (np.ndim(xx) < 2)
        xx = np.atleast_2d(xx)
        ndim, nval = np.shape(xx)
        return xx, ndim, squeeze

    @classmethod
    def evaluate(self, xx):
        yy, ndim, squeeze = self._parse(xx)
        zz = self._evaluate(yy, ndim)
        if squeeze:
            zz = zz.squeeze()
        return zz

    @classmethod
    def _evaluate(self, yy, ndim):
        err = "`_evaluate` must be overridden by the Distribution subclass!"
        raise NotImplementedError(err)

    @classmethod
    def grid(cls, edges, **kwargs):
        coords = np.meshgrid(*edges)
        shp = np.shape(coords)[1:]
        coords = np.vstack([xx.ravel() for xx in coords])
        pdf = cls.evaluate(coords, **kwargs)
        pdf = pdf.reshape(shp)
        return pdf

    def sample(self, size, ndim=None, squeeze=None):
        if ndim is None:
            ndim = 1
            if squeeze is None:
                squeeze = True

        if squeeze is None:
            squeeze = False

        samps = self._sample(size, ndim)
        if squeeze:
            samps = samps.squeeze()
        return samps

    def _sample(self, size, ndim):
        grid, cdf = self.cdf_grid
        samps = np.random.uniform(0.0, 1.0, ndim*size)
        samps = sp.interpolate.interp1d(cdf, grid, kind='quadratic')(samps).reshape(ndim, size)
        # samps = utils.rem_cov(samps)
        # samps = utils.add_cov(samps, cov)
        return samps

    def cdf(self, xx):
        zz = sp.interpolate.interp1d(*self.cdf_grid, kind='cubic')(xx)
        return zz

    @property
    def cdf_grid(self):
        if self._cdf_grid is None:
            if self._FINITE:
                pad = (1 + _NUM_PAD)
                args = [-pad, pad, 2000]
            else:
                args = [-10, 10, 20000]
            xe, xc, dx = utils.bins(*args)

            yy = self.evaluate(xc)
            csum = np.cumsum(yy*dx)
            norm = csum[-1]
            if not np.isclose(norm, 1.0, rtol=1e-4):
                err = "Failed to reach unitarity in CDF grid norm: {:.4e}!".format(norm)
                raise ValueError(err)
            # csum = csum / norm
            xc = np.concatenate([[args[0]], [args[0]], xc, [args[1]], [args[1]]], axis=0)
            csum = np.concatenate([[0.0 - _NUM_PAD], [0.0], csum, [1.0], [1.0+_NUM_PAD]], axis=0)
            self._cdf_grid = [xc, csum]

        return self._cdf_grid

    @property
    def FINITE(self):
        return self._FINITE

    @classmethod
    def inside(cls, pnts):
        idx = (cls.evaluate(pnts) > 0.0)
        return idx


class Gaussian(Distribution):

    _FINITE = False

    @classmethod
    def _evaluate(self, yy, ndim):
        energy = np.sum(yy * yy, axis=0) / 2.0
        norm = self.norm(ndim)
        result = np.exp(-energy) / norm
        return result

    @classmethod
    def norm(self, ndim=1):
        norm = np.power(2*np.pi, ndim/2)
        return norm

    def cdf(self, yy):
        zz = sp.stats.norm.cdf(yy)
        return zz

    def _sample(self, size, ndim):
        cov = np.eye(ndim)
        samps = np.random.multivariate_normal(np.zeros(ndim), cov, size=size).T
        return samps

    @classmethod
    def inside(cls, pnts):
        ndim, nvals = np.shape(np.atleast_2d(pnts))
        return np.ones(nvals, dtype=bool)


class Box_Asym(Distribution):

    _FINITE = True

    @classmethod
    def _evaluate(self, yy, ndim):
        norm = np.power(2, ndim)
        zz = (np.max(np.fabs(yy), axis=0) < 1.0) / norm
        return zz

    @classmethod
    def _sample(self, size, ndim):
        samps = np.random.uniform(-1.0, 1.0, size=ndim*size).reshape(ndim, size)
        return samps

    @classmethod
    def cdf(self, xx):
        zz = 0.5 + np.minimum(np.maximum(xx, -1), 1)/2
        return zz

    @classmethod
    def inside(cls, pnts):
        pnts = np.atleast_2d(pnts)
        ndim, nvals = np.shape(pnts)
        bounds = [[-1.0, 1.0] for ii in range(ndim)]
        idx = utils.bound_indices(pnts, bounds)
        return idx


class Parabola_Asym(Distribution):

    _FINITE = True

    @classmethod
    def _evaluate(self, yy, ndim):
        norm = 2 * _nball_vol(ndim) / (ndim + 2)
        zz = np.product(np.maximum(1 - yy**2, 0.0), axis=0) / norm
        return zz

    def _sample(self, size, ndim):
        # Use the median trick to draw from the Epanechnikov distribution
        samp = np.random.uniform(-1, 1, 3*ndim*size).reshape(ndim, size, 3)
        samp = np.median(samp, axis=-1)
        return samp

    @classmethod
    def cdf(self, xx):
        xx = np.minimum(np.maximum(xx, -1), 1)
        zz = 0.5 + (3/4)*(xx - xx**3 / 3)
        return zz


class Triweight(Distribution):

    _FINITE = True

    @classmethod
    def _evaluate(self, yy, ndim):
        norm = 32.0 / 35.0
        zz = np.product(np.maximum((1 - yy*yy)**3, 0.0), axis=0)
        zz = zz / norm
        return zz

    @classmethod
    def cdf(self, xx):
        yy = np.minimum(np.maximum(xx, -1), 1)
        coeffs = [35/32, -35/32, 21/32, -5/32]
        powers = [1, 3, 5, 7]
        zz = 0.5 + np.sum([aa*np.power(yy, pp) for aa, pp in zip(coeffs, powers)], axis=0)
        return zz


_DEFAULT_DISTRIBUTION = Gaussian

_index_list = [
    ['gaussian', Gaussian],
    ['box', Box_Asym],
    ['parabola', Parabola_Asym],
    ['epanechnikov', Parabola_Asym],
    ['triweight', Triweight],
]

# _all_skip = [Parabola_Asym, Triweight]
_all_skip = []

_index = OrderedDict([(nam, val) for nam, val in _index_list])

# Parabola = Parabola_Asym
# Box = Box_Asym


def get_distribution_class(arg=None):
    if arg is None:
        return _DEFAULT_DISTRIBUTION

    if isinstance(arg, six.string_types):
        arg = arg.lower().strip()
        names = list(_index.keys())
        if arg not in names:
            err = "`Distribution` '{}' is not in the index.  Choose one of: '{}'!".format(
                arg, names)
            raise ValueError(err)

        return _index[arg]

    # This will raise an error if `arg` isn't a class at all
    try:
        if issubclass(arg, Distribution):
            return arg
    except:
        pass

    raise ValueError("Unrecognized `Distribution` type '{}'!".format(arg))


def get_all_distribution_classes():
    kerns = []
    for kk in _index.values():
        if kk not in kerns:
            if kk in _all_skip:
                logging.warning("WARNING: skipping `Distribution` '{}'!".format(kk))
                continue
            kerns.append(kk)
    return kerns


def _nball_vol(ndim, rad=1.0):
    vol = np.pi**(ndim/2)
    vol = (rad**ndim) * vol / sp.special.gamma((ndim/2) + 1)
    return vol
