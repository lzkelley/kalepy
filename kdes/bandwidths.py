"""
"""
import logging
import six

import numpy as np


class Bandwidth(object):
    _SET_OFF_DIAGONAL = True

    def __init__(self, kde, bandwidth):
        self._kde = kde
        self._ndim = kde.ndim

        data_cov = np.cov(kde.dataset, rowvar=True, bias=False, aweights=kde.weights)
        self._data_cov = np.atleast_2d(data_cov)
        self.set_bandwidth(bandwidth)
        return

    def set_bandwidth(self, bandwidth):
        ndim = self._ndim
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

        # self.matrix_inv = matrix_inv
        # self.bw_norm = np.sqrt(np.linalg.det(2*np.pi*self.matrix))
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

    @property
    def data_cov(self):
        return self._data_cov

    @property
    def norm(self):
        try:
            if self._norm is None:
                raise AttributeError
        except AttributeError:
            self._norm = np.sqrt(np.linalg.det(2*np.pi*self.matrix))

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
            try:
                matrix_inv = np.linalg.inv(self.matrix)
            except np.linalg.LinAlgError:
                if not self._quiet:
                    logging.warning("singular `matrix`, trying SVD...")
                matrix_inv = np.linalg.pinv(self.matrix)
            self._matrix_inv = matrix_inv

        return self._matrix_inv

    @property
    def matrix_white(self):
        return self._matrix_white

    @property
    def method(self):
        return self._method

    def scott_factor(self, *args, **kwargs):
        return np.power(self._kde.neff, -1./(self._kde.ndim+4))

    def silverman_factor(self, *args, **kwargs):
        return np.power(self._kde.neff*(self._kde.ndim+2.0)/4.0, -1./(self._kde.ndim+4))
