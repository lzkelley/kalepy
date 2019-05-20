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

    def evaluate(self, xx, ref=0.0, hh=1.0, weights=1.0):
        err = "`evaluate` must be overridden by the Kernel subclass!"
        raise NotImplementedError(err)

    def sample(self, ndim, cov, size):
        err = "`sample` must be overridden by the Kernel subclass!"
        raise NotImplementedError(err)

    def pdf(self, data, weights, points, cov_inv, norm):
        """
        """
        ndim, num_data = np.shape(data)
        ndim, num_points = np.shape(points)
        result = np.zeros((num_points,), dtype=float)

        whitening = sp.linalg.cholesky(cov_inv)
        # Construct the 'whitened' (independent) dataset
        white_dataset = np.dot(whitening, data)
        # Construct the whitened sampling points
        white_points = np.dot(whitening, points)

        for ii in range(num_data):
            temp = self.evaluate(
                white_points, white_dataset[:, ii, np.newaxis], weights=weights[ii])
            result += temp

        result = result / norm
        return result

    def pdf_reflect(self, dataset, weights, points, cov_inv, norm, reflect=None):
        """
        """
        ndim, num_data = np.shape(dataset)
        ndim, num_points = np.shape(points)
        result = np.zeros((num_points,), dtype=float)

        whitening = sp.linalg.cholesky(cov_inv)
        # Construct the 'whitened' (independent) dataset
        white_dataset = np.dot(whitening, dataset)
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
                data = np.array(dataset)
                data[ii, :] = 2*loc - data[ii, :]
                white_dataset = np.dot(whitening, data)
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

    def resample(self, data, weights, cov, size, **kwargs):
        if len(kwargs):
            logging.warning("Unrecognized kwargs: '{}'".format(str(list(kwargs.keys()))))

        ndim, nvals = np.shape(data)
        # Draw from the smoothing kernel, here the `cov` includes the bandwidth
        norm = self.sample(ndim, cov, size)

        indices = np.random.choice(nvals, size=size, p=weights)
        means = data[:, indices]
        # Shift each re-drawn sample based on the kernel-samples
        samps = means + norm
        return samps

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


class Gaussian(Kernel):

    def evaluate(self, xx, ref=0.0, hh=1.0, weights=1.0):
        diff = (xx - ref) / hh
        energy = np.sum(diff * diff, axis=0) / 2.0
        result = weights * np.exp(-energy)
        return result

    def sample(self, ndim, cov, size):
        samp = np.random.multivariate_normal(np.zeros(ndim), cov, size=size).T
        return samp


class Box(Kernel):

    def evaluate(self, xx, ref=0.0, hh=1.0, weights=1.0):
        diff = (xx - ref) / hh
        result = weights * (np.max(np.fabs(diff), axis=0) < 1.0)
        return result

    def sample(self, ndim, cov, size):
        samp = np.random.uniform(-1.0, 1.0, size=ndim*size).reshape(ndim, size)
        # Correlate the samples appropriately
        samp = np.dot(cov, samp)
        return samp
