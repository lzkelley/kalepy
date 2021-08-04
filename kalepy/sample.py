"""Perform sampling of distributions and functions.
"""
import logging

import numpy as np

from kalepy import utils

__all__ = ['Sampler_Grid', 'sample_grid', 'sample_grid_proportional']


class Sampler_Grid:

    def __init__(self, edges, data, scalar=None):
        data = np.asarray(data)
        edges = [np.asarray(ee) for ee in edges]
        shape = data.shape
        ndim = data.ndim
        if len(edges) != ndim:
            err = "`edges` (len(edges)={}) must be a 1D array for each dimension of `data` (data.shape={})!".format(
                len(edges), data.shape)
            raise ValueError(err)

        edge_shape = np.array([ee.size for ee in edges])
        if np.all(edge_shape == shape):
            data_edge = data
            # find PDF at grid center-points
            data_cent = utils.midpoints(data_edge, log=False, axis=None)
        else:
            err = "Shape of edges ({}) inconsistent with data ({})!".format(edge_shape, shape)
            raise ValueError(err)

        # `scalar` must be shaped as either `data_cent` or `data_edge`
        #    if the latter, it will be converted to `data_cent` by averaging
        if scalar is not None:
            scalar = np.asarray(scalar)
            if np.all(scalar.shape == data_edge.shape):
                pass
            else:
                err = "Shape of `scalar` ({}) does not match `data` ({})!".format(
                    scalar.shape, data.shape)
                raise ValueError(err)

        idx, csum = self._data_to_cumulative(data_cent)

        self._edges = edges
        self._data_edge = data_edge
        self._data_cent = data_cent
        self._shape_cent = data_cent.shape
        self._ndim = ndim

        self._idx = idx
        self._csum = csum

        self._scalar = scalar
        return

    def sample(self, nsamp, interpolate=True, return_scalar=None):
        nsamp = int(nsamp)
        data_edge = self._data_edge
        scalar = self._scalar
        edges = self._edges

        if return_scalar is None:
            return_scalar = (scalar is not None)
        elif return_scalar and (scalar is None):
            return_scalar = False
            logging.warning("WARNING: no `scalar` initialized, cannot `return_scalar`!")

        # Choose random bins, proportionally to `data`, and positions within bins (uniformly distributed)
        #     `bin_numbers_flat` (N*D,) are the index numbers for bins in flattened 1D array of length N*D
        #     `intrabin_locs` (D, N) are position [0.0, 1.0] for each sample in each dimension
        bin_numbers_flat, intrabin_locs = self._random_bins(nsamp)
        # Convert from flat (1D) indices into ND indices;  (D, N) for `D` dimensions, `N` samples (`nsamp`)
        bin_numbers = np.unravel_index(bin_numbers_flat, self._shape_cent)

        # Start by finding scalar value for bin centers (i.e. bin averages)
        #    this will be updated/improved if `interpolation=True`
        if return_scalar:
            scalar_cent = utils.midpoints(scalar, log=False, axis=None)
            scalar_values = scalar_cent[bin_numbers]

        vals = np.zeros_like(intrabin_locs)
        for dim, (edge, bidx) in enumerate(zip(edges, bin_numbers)):
            # Width of bin-edges in this dimension
            wid = np.diff(edge)

            # Random location, in this dimension, for each bin. Relative position, i.e. between [0.0, 1.0]
            loc = intrabin_locs[dim]

            # random-uniform within each bin
            if (data_edge is None) or (not interpolate):
                vals[dim, :] = edge[bidx] + wid[bidx] * loc

            # random-linear proportional to bin edge-gradients (i.e. slope across bin in each dimension)
            else:
                edge = np.asarray(edge)

                # Find the gradient along this dimension (using center-values in other dimensions)
                grad = grad_along(data_edge, dim)
                # get the gradient for each sample
                grad = grad.flatten()[bin_numbers_flat]

                # interpolate edge values in this dimension
                vals[dim, :] = _intrabin_linear_interp(edge, wid, loc, bidx, grad)

            # interpolate scalar values also
            if return_scalar and interpolate:
                grad = grad_along(scalar, dim)
                grad = grad.flatten()[bin_numbers_flat]
                # shift `loc` (location within bin) to center point
                scalar_values += grad * (loc - 0.5)

        if return_scalar:
            return vals, scalar_values

        return vals

    def _random_bins(self, nsamp):
        csum = self._csum
        idx = self._idx

        # Draw random values
        #     random number for location in CDF, and additional random for position in each dimension of bin
        rand = np.random.uniform(0.0, 1.0, (1+self._ndim, nsamp))
        # np.random.shuffle(rand)    # extra-step to avoid (rare) structure in random data
        rand, *intrabin_locs = rand

        # Find which bin each random value should go in
        #    note that pre-sorting `rand` does speed up searchsorted: 'github.com/numpy/numpy/issues/10937'
        rand = np.sort(rand)
        sorted_bin_num = np.searchsorted(csum, rand) - 1

        # Convert indices from sorted-csum back to original `data_cent` ordering; (N,)
        bin_numbers_flat = idx[sorted_bin_num]
        return bin_numbers_flat, intrabin_locs

    def _data_to_cumulative(self, data_cent):
        # Convert to flat (1D) array of values
        data_cent = data_cent.flatten()
        # sort in order of probability
        idx = np.argsort(data_cent)
        csum = data_cent[idx]
        # find cumulative distribution and normalize to [0.0, 1.0]
        csum = np.cumsum(csum)
        csum = np.concatenate([[0.0], csum/csum[-1]])
        return idx, csum


# class Tracer_Outlier(Sampler_Grid):
class Outlier(Sampler_Grid):

    def __init__(self, edges, data, threshold=10.0, *args, **kwargs):
        super().__init__(edges, data, *args, **kwargs)

        data_cent = self._data_cent
        data_outs = np.copy(data_cent)

        # We're only going to stochastically sample from bins below the threshold value
        #     recalc `csum` zeroing out the values above threshold
        outs = (data_outs > threshold)
        data_outs[outs] = 0.0
        idx, csum = self._data_to_cumulative(data_outs)
        self._idx = idx
        self._csum = csum

        # We'll manually sample bins above threshold, so store those for later
        data_ins = np.copy(data_cent)
        data_ins[~outs] = 0.0

        self._threshold = threshold
        self._data_ins = data_ins
        self._data_outs = data_outs
        return

    def sample(self, nsamp=None, **kwargs):
        rv = kwargs.setdefault('return_scalar', False)
        if rv is not False:
            raise ValueError(f"Cannot use `scalar` values in `{self.__class__}`!")

        if nsamp is None:
            nsamp = self._data_outs.sum()
            print(f"sum over data_outs: {nsamp:.4e}")
            nsamp = np.random.poisson(nsamp)
            print(f"\tpoisson: {nsamp:.4e}")

        # sample outliers normally (using modified csum from `self._data_outs`)
        vals_outs = super().sample(nsamp, **kwargs)
        print(f"sample out: {utils.stats(vals_outs[0])=}")

        # sample tracer/representative points from `self._data_ins`
        data_ins = self._data_ins
        nin = np.count_nonzero(data_ins)
        ntot = nsamp + nin
        print(f"num nonzero data_ins: {nin=:.4e}, {ntot=:.4e}")
        weights = np.ones(ntot)

        bin_numbers_flat = (data_ins.flatten() > 0.0)
        bin_numbers_flat = np.arange(bin_numbers_flat.size)[bin_numbers_flat]
        bin_numbers = np.unravel_index(bin_numbers_flat, self._shape_cent)
        weights[nsamp:] = data_ins[bin_numbers]
        print(f"{utils.stats(weights)=}")
        print(f"{utils.stats(weights[nsamp:])=}")

        vals_ins = np.zeros((self._ndim, nin))
        for dim, (edge, bidx) in enumerate(zip(self._edges, bin_numbers)):
            vals_ins[dim, :] = utils.midpoints(edge, log=False)[bidx]

        print(f"sample in : {utils.stats(vals_ins[0])=}")

        vals = np.concatenate([vals_outs, vals_ins], axis=-1)
        return nsamp, vals, weights


def sample_grid(edges, dist, nsamp, scalar=None, **kwargs):
    """Draw samples following the given distribution.

    Arguments
    ---------
    edges : (D,) list/tuple of array_like,
        Edges of the (parameter space) grid.  For `D` dimensions, this is a list/tuple of D
        entries, where each entry is an array_like of scalars giving the grid-points along
        that dimension.
        e.g. if edges=([x, y], [a, b, c]) is a (2x3) dim array with coordinates
             [(x,a), (x,b), (x,c)], [(y,a), (y,b), (y,c)]
    dist : (N1,...,ND) array_like of scalar,
        Distribution values specified at either the grid edges, or grid centers.
        e.g. for the (2x3) example above, `dist` should be either (2,3) or (1, 2)
    nsamp : int
        Number of samples to draw (floats are cast to integers).
    scalar : None, or array_like of scalar
        Scalar values to associate with the given distribution.  Can be specified at either
        grid-centers or grid-edges, but the latter will be averaged down to grid-center values.

    Returns
    -------
    vals : (D, N) array of sample points,
        Sample points drawn from the given distribution in `D`, number of points `N` is that
        specified by `nsamp` param.
    [weights] : (N,) array of weights, returned if `scalar` is given
        Scalar factors for each sample point.

    """
    sampler = Sampler_Grid(edges, dist, scalar=scalar)
    return sampler.sample(nsamp, **kwargs)


def sample_grid_proportional(edges, data, portion, nsamp, **kwargs):
    scalar = data/portion
    scalar[portion == 0.0] = 0.0
    sampler = Sampler_Grid(edges, portion, scalar=scalar)
    vals, weight = sampler.sample(nsamp, **kwargs)
    return vals, weight


def grad_along(data_edge, dim):
    grad = np.diff(data_edge, axis=dim)
    nums = list(np.arange(grad.ndim))
    nums.pop(dim)
    grad = utils.midpoints(grad, log=False, axis=nums)
    return grad


def _intrabin_linear_interp(edge, wid, loc, bidx, grad, flat_tol=1e-2):
    vals = np.zeros_like(grad)

    # Find fractional gradient slope to filter out near-zero values
    grad_frac = np.fabs(grad)
    _gf_max = grad_frac.max()
    if _gf_max > 0.0:
        grad_frac /= grad_frac.max()
    # define 'flat' as below `flat_tol` threshold
    flat = (grad_frac < flat_tol)
    zer = np.ones_like(flat)

    # identify positive slope and interpolate left to right
    pos = (grad > 0.0) & ~flat
    zer[pos] = False
    vals[pos] = edge[bidx][pos] + wid[bidx][pos] * np.sqrt(loc[pos])

    # identify negative slope and interpolate right to left
    neg = (grad < 0.0) & ~flat
    zer[neg] = False
    vals[neg] = edge[bidx+1][neg] - wid[bidx][neg] * np.sqrt(loc[neg])

    # Use uniform sampling for flat cells
    vals[zer] = edge[bidx][zer] + wid[bidx][zer] * loc[zer]

    return vals
