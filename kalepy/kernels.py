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


class Kernel(object):

    _FINITE = None

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

    @classmethod
    def cdf(cls, xx, ref=0.0, bw=1.0):
        zz = sp.interpolate.interp1d(*cls._cdf_grid(ref, bw), kind='cubic')(xx)
        return zz

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

    @classmethod
    def evaluate(self, xx, ref=0.0, bw=1.0, weights=1.0):
        err = "`evaluate` must be overridden by the Kernel subclass!"
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
    def grid(cls, edges, **kwargs):
        coords = np.meshgrid(*edges)
        shp = np.shape(coords)[1:]
        coords = np.vstack([xx.ravel() for xx in coords])
        pdf = cls.evaluate(coords, **kwargs)
        print("coords = ", np.shape(coords), "pdf = ", np.shape(pdf), "shp = ", shp)
        pdf = pdf.reshape(shp)
        return pdf

    def pdf(self, points, data, weights, params=None):
        """
        """
        matrix_inv = self._matrix_inv
        norm = self._norm

        if params is not None:
            matrix = self._matrix
            data, matrix, norm = self._params_subset(data, matrix, params)
            matrix_inv = np.linalg.pinv(matrix)

        ndim, num_points = np.shape(points)

        whitening = sp.linalg.cholesky(matrix_inv)
        # Construct the whitened sampling points
        white_points = np.dot(whitening, points)

        result = np.zeros((num_points,), dtype=float)
        ndim, num_data = np.shape(data)
        # Construct the 'whitened' (independent) dataset
        white_dataset = np.dot(whitening, data)

        for ii in range(num_data):
            temp = self.evaluate(
                white_points, white_dataset[:, ii, np.newaxis], weights=weights[ii])
            result += temp.squeeze()

        result = result / norm
        return result

    def pdf_grid(self, edges, *args, **kwargs):
        ndim = self._ndim
        if len(edges) != ndim:
            err = "`edges` must be (D,): an array of edges for each dimension!"
            raise ValueError(err)

        coords = np.meshgrid(*edges)
        shp = np.shape(coords)[1:]
        coords = np.vstack([xx.ravel() for xx in coords])
        pdf = self.pdf(coords, *args, **kwargs)
        pdf = pdf.reshape(shp)
        return pdf

    def pdf_reflect(self, points, reflect, data, weights):
        """
        """
        matrix_inv = self._matrix_inv
        norm = self._norm

        ndim, num_data = np.shape(data)
        ndim, num_points = np.shape(points)
        result = np.zeros((num_points,), dtype=float)

        whitening = sp.linalg.cholesky(matrix_inv)
        # Construct the 'whitened' (independent) dataset
        white_dataset = np.dot(whitening, data)
        # Construct the whitened sampling points
        white_points = np.dot(whitening, points)

        for ii in range(num_data):
            result += self.evaluate(
                white_points, white_dataset[:, ii, np.newaxis], weights=weights[ii])

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
                pnts = np.array(points)
                white_points = np.dot(whitening, pnts)

                if num_points >= num_data:
                    for jj in range(num_data):
                        result += self.evaluate(
                            white_points, white_dataset[:, jj, np.newaxis], weights=weights[jj])
                else:
                    for jj in range(num_points):
                        res = self.evaluate(
                            white_dataset, white_points[:, jj, np.newaxis], weights=weights)
                        result[jj] += np.sum(res, axis=0)

            lo = -np.inf if reflect_dim[0] is None else reflect_dim[0]
            hi = +np.inf if reflect_dim[1] is None else reflect_dim[1]
            idx = (points[ii, :] < lo) | (hi < points[ii, :])
            result[idx] = 0.0

        result = result / norm
        return result

    def resample(self, size, data, weights, bw_matrix=None, keep=None):
        if bw_matrix is None:
            bw_matrix = self._matrix
        bw_matrix = self._cov_keep_vars(bw_matrix, keep)

        ndim, nvals = np.shape(data)
        # Draw from the smoothing kernel, here the `bw_matrix` includes the bandwidth
        norm = self.sample(ndim, bw_matrix, size)

        indices = np.random.choice(nvals, size=size, p=weights)
        means = data[:, indices]
        # Shift each re-drawn sample based on the kernel-samples
        samps = means + norm
        return samps

    def resample_reflect(self, size, reflect, data, weights, keep=None):
        bw_matrix = self._matrix
        bw_matrix = self._cov_keep_vars(bw_matrix, keep, reflect=reflect)

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
            #    set `keep` to None, `bw_matrix` is already modified to account for it
            trial = self.resample(draw, data=data, weights=weights, bw_matrix=bw_matrix, keep=None)
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


class Gaussian(Kernel):

    _FINITE = False

    @classmethod
    def evaluate(self, xx, ref=0.0, bw=1.0, weights=1.0):
        # ndim, nval = np.shape(xx)
        yy, ndim, nvals, squeeze = self.scale(xx, ref, bw)
        energy = np.sum(yy * yy, axis=0) / 2.0
        norm = np.power(2*np.pi*(bw**2), -ndim/2)
        result = norm * weights * np.exp(-energy)
        if squeeze:
            result = result.squeeze()
        return result

    @classmethod
    def sample(self, ndim, cov, size):
        cov = np.atleast_2d(cov)
        if np.shape(cov) != (ndim, ndim):
            err = "Shape of `cov` ({}) does not match `ndim` = {}".format(np.shape(cov), ndim)
            raise ValueError(err)
        samp = np.random.multivariate_normal(np.zeros(ndim), cov, size=size).T
        return samp

    @classmethod
    def cdf(self, xx, ref=0.0, bw=1.0):
        yy, ndim, nvals, squeeze = self.scale(xx, ref, bw)
        zz = sp.stats.norm.cdf(yy)
        if squeeze:
            zz = zz.squeeze()
        return zz


class Box_Asym(Kernel):

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
        samp = utils.add_cov(samp, samp_cov)
        samp = utils.add_cov(samp, cov)
        return samp

    @classmethod
    def cdf(self, xx, ref=0.0, bw=1.0):
        yy, ndim, nval, squeeze = self.scale(xx, ref, bw)
        zz = 0.5 + np.minimum(np.maximum(yy, -1), 1)/2
        if squeeze:
            zz = zz.squeeze()
        return zz


class Parabola_Asym(Kernel):

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


class Triweight(Kernel):

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
            err = "Kernel '{}' is not in the index.  Choose one of: '{}'!".format(arg, names)
            raise ValueError(err)

        return _index[arg]

    # This will raise an error if `arg` isn't a class at all
    try:
        if issubclass(arg, Kernel):
            return arg
    except:
        pass

    raise ValueError("Unrecognized Kernel type '{}'!".format(arg))


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
