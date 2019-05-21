"""
"""
import os

_path = os.path.dirname(__file__)
_vers_path = os.path.join(_path, "VERSION.txt")
with open(_vers_path) as inn:
    _version = inn.read().strip()

__version__ = _version
__author__ = "Luke Zoltan Kelley <lzkelley@northwestern.edu>"
__copyright__ = "Copyright 2019 - Luke Zoltan Kelley and contributors"
__contributors__ = []
__bibtex__ = """"""

# import six
# import logging
# import warnings

import scipy as sp  # noqa
import scipy.special  # noqa

import numpy as np

from kdes import utils, kernels, bandwidths   # noqa


class KDE(object):
    """

    Uses Fukunaga’s method.
    """
    _BANDWIDTH_DEFAULT = 'scott'
    _KERNEL_DEFAULT = kernels.Gaussian

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

        if kernel is None:
            kernel = self._KERNEL_DEFAULT
        self._kernel = kernel(self)

        if bandwidth is None:
            bandwidth = self._BANDWIDTH_DEFAULT
        self._bandwidth = bandwidths.Bandwidth(self, bandwidth)

        self._neff = neff
        self._weights = weights
        self._quiet = quiet
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

        if reflect is None:
            result = self.kernel.pdf(
                self.dataset, self.weights, points, self.bandwidth.matrix_inv, self.bandwidth.norm)
        else:
            result = self.kernel.pdf_reflect(
                self.dataset, self.weights, points, self.bandwidth.matrix_inv, self.bandwidth.norm, reflect=reflect)

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

        bw_cov = np.array(self.bandwidth.matrix)
        if keep is not None:
            keep = np.atleast_1d(keep)
            for pp in keep:
                bw_cov[pp, :] = 0.0
                bw_cov[:, pp] = 0.0
                # Make sure this isn't also a reflection axis
                if (reflect is not None) and (reflect[pp] is not None):
                    err = "Cannot both 'keep' and 'reflect' about dimension '{}'".format(pp)
                    raise ValueError(err)

        # Have `Kernel` class perform resampling
        # ---------------------------------------------------
        if reflect is None:
            samples = self._kernel.resample(
                self.dataset, self.weights, bw_cov, size)
        else:
            samples = self._kernel.resample_reflect(
                self.dataset, self.weights, bw_cov, size, reflect=reflect)

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

    @property
    def bandwidth(self):
        return self._bandwidth
