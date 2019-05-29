"""
"""
import logging
import six

import numpy as np
# import scipy as sp

from kalepy import kernels


class KDE(object):
    """
    """
    _BANDWIDTH_DEFAULT = 'scott'
    _SET_OFF_DIAGONAL = True

    def __init__(self, dataset, bandwidth=None, weights=None, kernel=None, neff=None, **kwargs):
        self.dataset = np.atleast_2d(dataset)
        ndim, ndata = self.dataset.shape
        if weights is None:
            weights = np.ones(ndata)/ndata

        self._ndim = ndim
        self._ndata = ndata

        if np.count_nonzero(weights) == 0 or np.any(~np.isfinite(weights) | (weights < 0)):
            raise ValueError("Invalid `weights` entries, all must be finite and > 0!")
        weights = np.atleast_1d(weights).astype(float)
        weights /= np.sum(weights)
        if np.shape(weights) != (ndata,):
            raise ValueError("`weights` input should be shaped as (N,)!")

        self._weights = weights

        if neff is None:
            neff = 1.0 / np.sum(weights**2)

        self._neff = neff

        if bandwidth is None:
            bandwidth = self._BANDWIDTH_DEFAULT
        data_cov = np.cov(dataset, rowvar=True, bias=False, aweights=weights)
        self._data_cov = np.atleast_2d(data_cov)
        self.set_bandwidth(bandwidth)

        # Convert from string, class, etc to a kernel
        dist = kernels.get_distribution_class(kernel)
        self._kernel = kernels.Kernel(distribution=dist, matrix=self.matrix)

        return

    def pdf(self, pnts, *args, **kwargs):
        result = self.kernel.pdf(pnts, self.dataset, self.weights, *args, **kwargs)
        return result

    def resample(self, size=None, keep=None, reflect=None, squeeze=True):
        """
        """
        samples = self.kernel.resample(
            self.dataset, self.weights,
            size=size, keep=keep, reflect=reflect, squeeze=squeeze)
        return samples

    # ==== Properties ====

    @property
    def weights(self):
        return self._weights

    @property
    def neff(self):
        return self._neff

    @property
    def ndim(self):
        return self._ndim

    @property
    def ndata(self):
        return self._ndata

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
    def matrix(self):
        return self._matrix

    @property
    def data_cov(self):
        return self._data_cov

    '''
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
    '''
