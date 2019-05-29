"""
"""
import logging
import six

import numpy as np
import scipy as sp

from kalepy import kernels, utils
QUIET = True


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
        kdist = kernels.get_kernel_class(kernel)
        self._kernel = Kernel(kdist=kdist, matrix=self.matrix)
        # self._kernel = kernel

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


class Kernel(object):

    def __init__(self, kdist=None, matrix=None, bandwidth=None):
        kdist = kernels.get_kernel_class(kdist)
        self._kdist = kdist

        if matrix is None:
            matrix = 1.0
            logging.warning("No `matrix` provided, setting to [[1.0]]!")

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
            matrix_inv = utils.matrix_invert(matrix, quiet=QUIET)

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
            temp = weights[ii] * self.kdist.evaluate(yy)
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
            result += weights[ii] * self.kdist.evaluate(yy)

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
                        result += weights[jj] * self.kdist.evaluate(yy)
                else:
                    for jj in range(num_points):
                        yy = white_dataset - white_points[:, jj, np.newaxis]
                        res = weights * self.kdist.evaluate(yy)
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

        # Have `Kernel_Dist` class perform resampling
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
        norm = self.kdist.sample(size, ndim=ndim)
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
    def kdist(self):
        return self._kdist

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
            self._matrix_inv = utils.matrix_invert(self.matrix, quiet=QUIET)
        return self._matrix_inv
