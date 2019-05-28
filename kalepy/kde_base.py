"""
"""
import logging
import six

import numpy as np

from kalepy import kernels, utils


class KDE(object):
    """
    """
    _BANDWIDTH_DEFAULT = 'scott'
    _SET_OFF_DIAGONAL = True

    def __init__(self, dataset, bandwidth=None, weights=None, kernel=None, neff=None,
                 quiet=False, **kwargs):
        self.dataset = np.atleast_2d(dataset)
        self._ndim, self._data_size = self.dataset.shape
        if weights is None:
            weights = np.ones(self.data_size)/self.data_size

        if np.count_nonzero(weights) == 0 or np.any(~np.isfinite(weights) | (weights < 0)):
            raise ValueError("Invalid `weights` entries, all must be finite and > 0!")
        weights = np.atleast_1d(weights).astype(float)
        weights /= np.sum(weights)
        if np.shape(weights) != (self.data_size,):
            raise ValueError("`weights` input should be shaped as (N,)!")

        # Convert from string, class, etc to a kernel
        kernel = kernels.get_kernel_class(kernel)
        self._kernel = kernel(self)

        if bandwidth is None:
            bandwidth = self._BANDWIDTH_DEFAULT
        data_cov = np.cov(dataset, rowvar=True, bias=False, aweights=weights)
        self._data_cov = np.atleast_2d(data_cov)
        self.set_bandwidth(bandwidth)

        self._neff = neff
        self._weights = weights
        self._quiet = quiet
        return

    def pdf(self, points, reflect=None, **kwargs):
        points = np.atleast_2d(points)

        # Make sure shape/values of reflect look okay
        reflect = self._check_reflect(reflect)

        if reflect is None:
            result = self.kernel.pdf(points, **kwargs)
        else:
            result = self.kernel.pdf_reflect(points, reflect, **kwargs)

        return result

    def resample(self, size=None, keep=None, reflect=None):
        """
        """

        # Check / Prepare parameters
        # -------------------------------------------
        if size is None:
            size = int(self.neff)

        # Make sure `reflect` matches
        if reflect is not None:
            # This is now either (D,) [and contains `None` values] or (D,2)
            reflect = self._check_reflect(reflect)

        # Have `Kernel` class perform resampling
        # ---------------------------------------------------
        if reflect is None:
            samples = self._kernel.resample(size, keep=keep)
        else:
            samples = self._kernel.resample_reflect(size, reflect, keep=keep)

        if self.ndim == 1:
            samples = samples.squeeze()

        return samples

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

    @property
    def kernel(self):
        return self._kernel

    # ==== BANDWIDTH ====

    def set_bandwidth(self, bandwidth):
        ndim = self.ndim
        _input = bandwidth
        matrix_white = np.zeros((ndim, ndim))

        if len(np.atleast_1d(bandwidth)) == 1:
            _bw, method = self._compute_bandwidth(bandwidth)
            if self._SET_OFF_DIAGONAL:
                matrix_white[...] = _bw
            else:
                idx = np.arange(ndim)
                matrix_white[idx, idx] = _bw
        else:
            if np.shape(bandwidth) == (ndim,):
                # bw_method = 'diagonal'
                for ii in range(ndim):
                    matrix_white[ii, ii] = self._compute_bandwidth(
                        bandwidth[ii], param=(ii, ii))[0]
                method = 'diagonal'
            elif np.shape(bandwidth) == (ndim, ndim):
                for ii, jj in np.ndindex(ndim, ndim):
                    matrix_white[ii, jj] = self._compute_bandwidth(
                        bandwidth[ii, jj], param=(ii, jj))[0]
                method = 'matrix'
            else:
                raise ValueError("`bandwidth` have shape (1,), (N,) or (N,) for `N` dimensions!")

        if np.any(np.isclose(matrix_white.diagonal(), 0.0)):
            ii = np.where(np.isclose(matrix_white.diagonal(), 0.0))[0]
            msg = "WARNING: diagonal '{}' of bandwidth is near zero!".format(ii)
            logging.warning(msg)

        matrix = self.data_cov * (matrix_white ** 2)

        # prev: bw_white
        self._matrix_white = matrix_white
        self._method = method
        self._input = _input
        # prev: bw_cov
        self._matrix = matrix
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

            method = bandwidth

        elif np.isscalar(bandwidth):
            bw = bandwidth
            method = 'constant scalar'

        elif callable(bandwidth):
            bw = bandwidth(self, param=param)
            method = 'function'

        else:
            raise ValueError("Unrecognized `bandwidth` '{}'!".format(bandwidth))

        return bw, method

    def scott_factor(self, *args, **kwargs):
        return np.power(self.neff, -1./(self.ndim+4))

    def silverman_factor(self, *args, **kwargs):
        return np.power(self.neff*(self.ndim+2.0)/4.0, -1./(self.ndim+4))

    @property
    def data_cov(self):
        return self._data_cov

    @property
    def norm(self):
        try:
            if self._norm is None:
                raise AttributeError
        except AttributeError:
            self._norm = np.sqrt(np.linalg.det(self.matrix))

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
            self._matrix_inv = utils.matrix_invert(self.matrix, quiet=self._quiet)
        return self._matrix_inv

    @property
    def matrix_white(self):
        return self._matrix_white
