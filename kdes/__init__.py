"""
"""
import os

_path = os.path.dirname(__file__)
_vers_path = os.path.join(_path, "VERSION.txt")
with open(_vers_path) as inn:
    _version = inn.read().strip()

__version__ = _version
__author__ = "Luke Zoltan Kelley <lzkelley@northwestern.edu>"
__copyright__ = "Copyright 2019- Luke Zoltan Kelley and contributors"
__contributors__ = []
__bibtex__ = """"""

import six
import logging
import warnings

import scipy as sp
import scipy.special  # noqa

import numpy as np

__all__ = ['KDE']

from kdes import utils  # noqa


class KDE(object):
    """

    Uses Fukunaga’s method.
    """
    _BANDWIDTH_DEFAULT = 'scott'
    _SET_OFF_DIAGONAL = True

    def __init__(self, dataset, bandwidth=None, weights=None, neff=None, quiet=False, **kwargs):
        bw_method = kwargs.get('bw_method', None)
        if bw_method is not None:
            msg = "Use `bandwidth` instead of `bw_method`"
            warnings.warn(msg, DeprecationWarning, stacklevel=3)
            if bandwidth is not None:
                raise ValueError("Both `bandwidth` and `bw_method` provided!")
            bandwidth = bw_method

        self.dataset = np.atleast_2d(dataset)
        self._ndim, self._data_size = self.dataset.shape
        if weights is None:
            weights = np.ones(self.data_size)/self.data_size

        if weights is not None:
            if np.count_nonzero(weights) == 0 or np.any(~np.isfinite(weights) | (weights < 0)):
                raise ValueError("Invalid `weights` entries, all must be finite and > 0!")
            weights = np.atleast_1d(weights).astype(float)
            weights /= np.sum(weights)
            if np.shape(weights) != (self.data_size,):
                raise ValueError("`weights` input should be shaped as (N,)!")

        self._neff = neff
        self._weights = weights
        self._compute_covariance()
        self._quiet = quiet
        self.set_bandwidth(bandwidth=bandwidth)
        return

    def pdf(self, points, reflect=None):
        points = np.atleast_2d(points)

        ndim, nv = points.shape
        if ndim != self.ndim:
            if ndim == 1 and nv == self.ndim:
                # points was passed in as a row vector
                points = np.reshape(points, (self.ndim, 1))
                nv = 1
            else:
                msg = "Mismatch between shape of `points` ({}), and `dataset` ({})".format(
                    ndim, self.ndim)
                raise ValueError(msg)

        # Make sure shape/values of reflect look okay
        reflect = self._check_reflect(reflect)

        result = np.zeros((nv,), dtype=float)

        whitening = sp.linalg.cholesky(self.bw_cov_inv)
        # Construct the 'whitened' (independent) dataset
        white_dataset = np.dot(whitening, self.dataset)
        # Construct the whitened sampling points
        white_points = np.dot(whitening, points)

        for i in range(self.data_size):
            diff = white_points - white_dataset[:, i, np.newaxis]
            energy = np.sum(diff * diff, axis=0) / 2.0
            result += self.weights[i]*np.exp(-energy)

        if reflect is not None:
            for ii, reflect_dim in enumerate(reflect):
                if reflect_dim is None:
                    continue

                for loc in reflect_dim:
                    if loc is None:
                        continue

                    # shape (D,N) i.e. (dimensions, data-points)
                    data = np.array(self.dataset)
                    data[ii, :] = data[ii, :] - loc
                    white_dataset = np.dot(whitening, data)
                    # Construct the whitened sampling points
                    #    shape (D,M) i.e. (dimensions, sample-points)
                    pnts = np.array(points)
                    pnts[ii, :] = pnts[ii, :] - loc
                    white_points = np.dot(whitening, pnts)

                    if nv >= self.data_size:
                        for jj in range(self.data_size):
                            diff = white_points + white_dataset[:, jj, np.newaxis]
                            energy = np.sum(diff * diff, axis=0) / 2.0
                            result += self.weights[jj]*np.exp(-energy)

                    else:
                        for jj in range(nv):
                            diff = white_dataset - white_points[:, jj, np.newaxis]
                            energy = np.sum(diff * diff, axis=0) / 2.0
                            result[jj] = np.sum(np.exp(-energy)*self.weights, axis=0)

                reflect_dim[0] = -np.inf if reflect_dim[0] is None else reflect_dim[0]
                reflect_dim[1] = +np.inf if reflect_dim[1] is None else reflect_dim[1]
                idx = (points[ii, :] < reflect_dim[0]) | (reflect_dim[1] < points[ii, :])
                result[idx] = 0.0

        result = result / self.bw_norm

        return result

    def resample(self, size=None, keep=None, reflect=None):
        if reflect is None:
            samples = self.resample_default(size=size, keep=keep)
        else:
            samples = self.resample_reflect(size=size, keep=keep, reflect=reflect)

        if self.ndim == 1:
            samples = samples.squeeze()

        return samples

    def resample_default(self, size=None, keep=None):
        if size is None:
            size = int(self.neff)

        bw_cov = np.array(self.bw_cov)
        if keep is not None:
            keep = np.atleast_1d(keep)
            for pp in keep:
                bw_cov[pp, :] = 0.0
                bw_cov[:, pp] = 0.0

        '''
        # Draw from the smoothing kernel, here the `cov` includes the bandwidth
        norm = np.random.multivariate_normal(np.zeros(self.ndim), bw_cov, size=size).T
        # Draw randomly from the given data points, proportionally to their weights
        indices = np.random.choice(self.data_size, size=size, p=self.weights)
        means = self.dataset[:, indices]
        # Shift each re-drawn sample based on the kernel-samples
        samps = means + norm
        '''
        samps = self._resample(self.dataset, self.weights, bw_cov, size)

        return samps

    def resample_reflect(self, size=None, keep=None, reflect=None):
        if size is None:
            size = int(self.neff)

        reflect = self._check_reflect(reflect)
        if reflect is None:
            raise ValueError("`reflect` is None!")

        bw_cov = np.array(self.bw_cov)
        if keep is not None:
            keep = np.atleast_1d(keep)
            for pp in keep:
                bw_cov[pp, :] = 0.0
                bw_cov[:, pp] = 0.0

        # shape (D,N) i.e. (dimensions, data-points)
        data = np.array(self.dataset)
        weights = np.array(self.weights)
        bounds = np.zeros((self.ndim, 2))
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
                new_data = np.array(self.dataset)
                new_data[ii, :] = new_data[ii, :] - loc
                data = np.append(data, new_data, axis=-1)
                weights = np.append(weights, self.weights, axis=-1)

        weights = weights / np.sum(weights)

        # Draw randomly from the given data points, proportionally to their weights
        samps = np.zeros((size, self.ndim))
        num_good = 0
        cnt = 0
        MAX = 10
        draw = size
        while num_good < size and cnt < MAX:
            trial = self._resample(data, weights, bw_cov, draw)
            idx = self._bound_indices(trial, bounds)

            ngd = np.count_nonzero(idx)
            if num_good + ngd <= size:
                samps[num_good:num_good+ngd, :] = trial.T[idx, :]
            else:
                ngd = (size - num_good)
                samps[num_good:num_good+ngd, :] = trial.T[idx, :][:ngd]

            num_good += ngd
            cnt += 1
            # Next time draw twice as many as we need
            draw = 2*(size - num_good)

        if num_good < size:
            raise RuntimeError("Failed to draw '{}' samples in {} iterations!".format(size, cnt))

        samps = samps.T

        return samps

    def _resample(self, data, weights, cov, size):
        ndim, nvals = np.shape(data)
        # Draw from the smoothing kernel, here the `cov` includes the bandwidth
        norm = np.random.multivariate_normal(np.zeros(ndim), cov, size=size).T

        indices = np.random.choice(nvals, size=size, p=weights)
        means = data[:, indices]
        # Shift each re-drawn sample based on the kernel-samples
        samps = means + norm
        return samps

    def _bound_indices(self, data, bounds):
        ndim, nvals = np.shape(data)
        idx = np.ones(nvals, dtype=bool)
        for ii, bnd in enumerate(bounds):
            idx = idx & (bnd[0] < data[ii, :]) & (data[ii, :] < bnd[1])
        return idx

    def scott_factor(self, *args, **kwargs):
        return np.power(self.neff, -1./(self.ndim+4))

    def silverman_factor(self, *args, **kwargs):
        return np.power(self.neff*(self.ndim+2.0)/4.0, -1./(self.ndim+4))

    def set_bandwidth(self, bandwidth=None):
        ndim = self.ndim
        _bandwidth = bandwidth
        bw_white = np.zeros((ndim, ndim))

        if len(np.atleast_1d(bandwidth)) == 1:
            _bw, bw_type = self._compute_bandwidth(bandwidth)
            if self._SET_OFF_DIAGONAL:
                bw_white[...] = _bw
            else:
                idx = np.arange(ndim)
                bw_white[idx, idx] = _bw
        else:
            if np.shape(bandwidth) == (ndim,):
                # bw_method = 'diagonal'
                for ii in range(self.ndim):
                    bw_white[ii, ii] = self._compute_bandwidth(
                        bandwidth[ii], param=(ii, ii))[0]
                bw_type = 'diagonal'
            elif np.shape(bandwidth) == (ndim, ndim):
                for ii, jj in np.ndindex(ndim, ndim):
                    bw_white[ii, jj] = self._compute_bandwidth(
                        bandwidth[ii, jj], param=(ii, jj))[0]
                bw_type = 'matrix'
            else:
                raise ValueError("`bandwidth` have shape (1,), (N,) or (N,) for `N` dimensions!")

        if np.any(np.isclose(bw_white.diagonal(), 0.0)):
            ii = np.where(np.isclose(bw_white.diagonal(), 0.0))[0]
            msg = "WARNING: diagonal '{}' of bandwidth is near zero!".format(ii)
            logging.warning(msg)

        bw_cov = self._data_cov * (bw_white ** 2)
        try:
            bw_cov_inv = np.linalg.inv(bw_cov)
        except np.linalg.LinAlgError:
            if not self._quiet:
                logging.warning("WARNING: singular `bw_cov` matrix, trying SVD...")
            bw_cov_inv = np.linalg.pinv(bw_cov)

        self.bw_white = bw_white
        self.bw_type = bw_type
        self._bandwidth = _bandwidth
        self.bw_cov = bw_cov
        self.bw_cov_inv = bw_cov_inv
        self.bw_norm = np.sqrt(np.linalg.det(2*np.pi*self.bw_cov))
        return

    def _compute_bandwidth(self, bandwidth, param=None):
        if bandwidth is None:
            bandwidth = self._BANDWIDTH_DEFAULT

        if isinstance(bandwidth, six.string_types):
            if bandwidth == 'scott':
                bw = self.scott_factor(param=param)
            elif bandwidth == 'silverman':
                bw = self.silverman_factor(param=param)
            else:
                msg = "Unrecognized bandwidth str specification '{}'!".format(bandwidth)
                raise ValueError(msg)

            bw_type = bandwidth

        elif np.isscalar(bandwidth):
            bw = bandwidth
            bw_type = 'constant scalar'

        elif callable(bandwidth):
            bw = bandwidth(self, param=param)
            bw_type = 'function'

        else:
            raise ValueError("Unrecognized `bandwidth` '{}'!".format(bandwidth))

        return bw, bw_type

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using bandwidth_func().
        """

        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_cov'):
            cov = np.cov(self.dataset, rowvar=True, bias=False, aweights=self.weights)
            self._data_cov = np.atleast_2d(cov)
        # if not hasattr(self, '_data_inv_cov'):
        #     self._data_inv_cov = np.linalg.inv(self._data_cov)

        return

    def _check_reflect(self, reflect):
        if reflect is None:
            return reflect

        if self.ndim == 1 and np.ndim(reflect) == 1:
            reflect = np.atleast_2d(reflect)

        if len(reflect) != self.ndim:
            msg = "`reflect` ({}) must have length (D,) = ({},)!".format(
                len(reflect), self.ndim)
            raise ValueError(msg)
        if not np.all([(ref is None) or len(ref) == 2 for ref in reflect]):
            raise ValueError("each row of `reflect` must be `None` or shape (2,)!")

        return reflect

    @property
    def weights(self):
        try:
            if self._weights is None:
                raise AttributeError
            return self._weights
        except AttributeError:
            self._weights = np.ones(self.data_size)/self.data_size
            return self._weights

    @property
    def neff(self):
        try:
            if self._neff is None:
                raise AttributeError
            return self._neff
        except AttributeError:
            self._neff = 1.0 / np.sum(self.weights**2)
            return self._neff

    @property
    def ndim(self):
        try:
            if self._ndim is None:
                raise AttributeError
            return self._ndim
        except AttributeError:
            self._ndim, self._data_size = np.shape(self.dataset)
            return self._ndim

    @property
    def data_size(self):
        try:
            if self._data_size is None:
                raise AttributeError
            return self._data_size
        except AttributeError:
            self._ndim, self._data_size = np.shape(self.dataset)
            return self._data_size
