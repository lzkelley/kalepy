"""Kernal basis functions for KDE calculations.
"""
import logging

import numpy as np
import scipy as sp   # noqa
import scipy.stats   # noqa

from kdes import utils


class Kernel(object):

    def __init__(self, kde=None):
        self.kde = kde
        return

    def resample(self, data, weights, cov, size, **kwargs):
        if len(kwargs):
            logging.warning("Unrecognized kwargs: '{}'".format(str(list(kwargs.keys()))))

        ndim, nvals = np.shape(data)
        # Draw from the smoothing kernel, here the `cov` includes the bandwidth
        norm = np.random.multivariate_normal(np.zeros(ndim), cov, size=size).T

        indices = np.random.choice(nvals, size=size, p=weights)
        means = data[:, indices]
        # Shift each re-drawn sample based on the kernel-samples
        samps = means + norm
        return samps

    def resample_reflect(self, data, weights, cov, size, **kwargs):
        err = "reflection is not implemented for this Kernel ({})!".format(self)
        raise NotImplementedError(err)


class Gaussian(Kernel):

    def resample_reflect(self, data, weights, cov, size, reflect=None):
        # shape (D,N) i.e. (dimensions, data-points)
        ndim, nvals = np.shape(data)
        data = np.array(data)
        weights = np.array(weights)
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
            trial = self.resample(data, weights, cov, draw)
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
