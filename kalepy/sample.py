"""Perform sampling of distributions and functions.
"""
import logging

import numpy as np

from kalepy import utils

__all__ = [
    'Sample_Grid', 'Sample_Outliers', 'sample_grid', 'sample_outliers', 'sample_grid_proportional'
]


class Sample_Grid:

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
                grad = _grad_along(data_edge, dim)
                # get the gradient for each sample
                grad = grad.flatten()[bin_numbers_flat]

                # interpolate edge values in this dimension
                vals[dim, :] = _intrabin_linear_interp(edge, wid, loc, bidx, grad)

            # interpolate scalar values also
            if return_scalar and interpolate:
                grad = _grad_along(scalar, dim)
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


class Sample_Outliers(Sample_Grid):

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

        # if `nsamp` isn't given, assume outlier distribution values correspond to numbers
        #    and Poisson sample them
        # NOTE: `nsamp` corresponds only to the _"outliers"_ not the 'interior' points also
        if nsamp is None:
            nsamp = self._data_outs.sum()
            nsamp = np.random.poisson(nsamp)

        vals_outs = super().sample(nsamp, **kwargs)

        # sample tracer/representative points from `self._data_ins`
        data_ins = self._data_ins
        nin = np.count_nonzero(data_ins)
        ntot = nsamp + nin
        # weights needed for all points, but "outlier" points will have weigtht 1.0
        weights = np.ones(ntot)

        # Get the bin indices of all of the 'interior' bins (those where `data_ins` are nonzero)
        bin_numbers_flat = (data_ins.flatten() > 0.0)
        bin_numbers_flat = np.arange(bin_numbers_flat.size)[bin_numbers_flat]
        # Convert from 1D index to ND
        bin_numbers = np.unravel_index(bin_numbers_flat, self._shape_cent)
        # Set the weights to be the value of the bin-centers
        weights[nsamp:] = data_ins[bin_numbers]

        # Find the 'interior' bin centers and use those as tracer points for well-sampled data
        vals_ins = np.zeros((self._ndim, nin))
        for dim, (edge, bidx) in enumerate(zip(self._edges, bin_numbers)):
            vals_ins[dim, :] = utils.midpoints(edge, log=False)[bidx]

        # Combine interior-tracers and outlier-samples
        vals = np.concatenate([vals_outs, vals_ins], axis=-1)
        return nsamp, vals, weights


def sample_grid(edges, dist, nsamp, scalar=None, **sample_kwargs):
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
    sample_kwargs : additional keyword-arguments, optional
        Additional arguments passed to the `Sample_Grid.sample()` method.

    Returns
    -------
    vals : (D, N) array of sample points,
        Sample points drawn from the given distribution in `D`, number of points `N` is that
        specified by `nsamp` param.
    [weights] : (N,) array of weights, returned if `scalar` is given
        Scalar factors for each sample point.

    """

    # ---- Check/Sanitize input arguments

    if not utils.really1d(edges):
        elens = [len(ee) for ee in edges]
        if not np.all(elens == dist.shape):
            err = f"Lengths of edges ({elens}) does not match distribution shape ({dist.shape})!"
            raise ValueError(err)
        edges = np.concatenate(edges)
        if not utils.really1d(edges):
            raise ValueError("Failed to concatenate `edges` into 1D array!")

    if len(edges) != np.sum(dist.shape):
        err = f"Length of 1D edges ({len(edges)}) does not match distribution shape ({dist.shape})!"
        raise ValueError(err)

    if (scalar is not None) and (np.shape(dist) != np.shape(scalar)):
        raise ValueError(f"Shape of scalar ({scalar.shape}) does not match distribution ({dist.shape})!")

    # ---- Draw Samples

    if scalar is None:
        samples = _sample(edges, dist, nsamp, **sample_kwargs)
        rv = samples
    else:
        samples, scalars = _sample_scalar(edges, dist, scalar, nsamp, **sample_kwargs)
        rv = (samples, scalars)

    return rv


'''
def sample_grid(edges, dist, nsamp, scalar=None, **sample_kwargs):
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
    sample_kwargs : additional keyword-arguments, optional
        Additional arguments passed to the `Sample_Grid.sample()` method.

    Returns
    -------
    vals : (D, N) array of sample points,
        Sample points drawn from the given distribution in `D`, number of points `N` is that
        specified by `nsamp` param.
    [weights] : (N,) array of weights, returned if `scalar` is given
        Scalar factors for each sample point.

    """
    sampler = Sample_Grid(edges, dist, scalar=scalar)
    return sampler.sample(nsamp, **sample_kwargs)
'''


def sample_grid_proportional(edges, data, portion, nsamp, **sample_kwargs):
    scalar = data / portion
    # Avoid NaN values
    scalar[portion == 0.0] = 0.0
    sampler = Sample_Grid(edges, portion, scalar=scalar)
    vals, weight = sampler.sample(nsamp, **sample_kwargs)
    return vals, weight


def sample_outliers(edges, data, threshold, nsamp=None, **sample_kwargs):
    outliers = Sample_Outliers(edges, data, threshold=threshold)
    nsamp, vals, weights = outliers.sample(nsamp=nsamp, **sample_kwargs)
    return vals, weights


def _grad_along(data_edge, dim):
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


import numba
from numba.np.unsafe.ndarray import to_fixed_tuple


# T_INT = np.int64
# T_INT = numba.int64
T_INT = 'int32'
T_FLT = numba.float64

NUMBA = numba.njit
# NUMBA = numba.jit


@NUMBA
def _unravel_index(idx, shape_rev_prod):
    ndim = len(shape_rev_prod)
    loc = np.zeros(ndim, dtype=T_INT)
    for dd in range(ndim):
        if dd < ndim - 1:
            sz = shape_rev_prod[dd]
            loc[dd] = idx // sz
            idx = idx % sz
        else:
            loc[dd] = idx
    return loc


@NUMBA
def _ravel_index(idx, shape_rev_prod):
    ndim = len(idx)
    jj = 0
    last = ndim - 1
    for dd in range(ndim):
        jj += shape_rev_prod[last-dd] * idx[last-dd]
    return jj


@NUMBA
def _get_shape_rev_prod(shape):
    ndim = len(shape)
    last = ndim - 1
    shape_rev_prod = np.ones((ndim,), dtype=T_INT)
    for dd in range(ndim-1):
        if dd == 0:
            shape_rev_prod[last-1] = shape[last]
        else:
            shape_rev_prod[last-dd-1] = shape[last-dd] * shape_rev_prod[last-dd]
    return shape_rev_prod


@NUMBA
def _get_centers(data_edge):
    ndim = data_edge.ndim
    eshape = data_edge.shape
    cshape = np.zeros((ndim,), dtype='int32')
    for dd in range(ndim):
        cshape[dd] = eshape[dd] - 1

    cshape = to_fixed_tuple(cshape, ndim)
    cshape_rev_prod = _get_shape_rev_prod(cshape)
    eshape_rev_prod = _get_shape_rev_prod(eshape)

    corners_shape = 2 * np.ones((ndim,), dtype=T_INT)
    corners_shape = to_fixed_tuple(corners_shape, ndim)
    corners_num = 2 ** ndim

    corner = np.zeros(ndim, dtype=T_INT)
    data_cent = np.zeros(cshape)

    # ------------- find center values ----------------
    for cbin3d in np.ndindex(cshape):
        cbin1d = _ravel_index(cbin3d, cshape_rev_prod)

        tt = 0.0
        for offset in np.ndindex(corners_shape):
            for dd in range(ndim):
                corner[dd] = cbin3d[dd] + offset[dd]

            ebin1d = _ravel_index(corner, eshape_rev_prod)
            tt = tt + data_edge.ravel()[ebin1d]

        data_cent.ravel()[cbin1d] = tt / corners_num

    return data_cent


@NUMBA
def _get_centers_scalar(data_edge, scalar_edge):
    ndim = data_edge.ndim
    eshape = data_edge.shape
    cshape = np.zeros((ndim,), dtype='int32')
    for dd in range(ndim):
        cshape[dd] = eshape[dd] - 1

    cshape = to_fixed_tuple(cshape, ndim)
    cshape_rev_prod = _get_shape_rev_prod(cshape)
    eshape_rev_prod = _get_shape_rev_prod(eshape)

    corners_shape = 2 * np.ones((ndim,), dtype=T_INT)
    corners_shape = to_fixed_tuple(corners_shape, ndim)
    corners_num = 2 ** ndim

    corner = np.zeros(ndim, dtype=T_INT)
    scalar_cent = np.zeros(cshape)
    data_cent = np.zeros(cshape)

    # ------------- find center values ----------------
    for cbin3d in np.ndindex(cshape):
        cbin1d = _ravel_index(cbin3d, cshape_rev_prod)

        ss = 0.0
        tt = 0.0
        for offset in np.ndindex(corners_shape):
            for dd in range(ndim):
                corner[dd] = cbin3d[dd] + offset[dd]

            ebin1d = _ravel_index(corner, eshape_rev_prod)
            ss = ss + scalar_edge.ravel()[ebin1d]
            tt = tt + data_edge.ravel()[ebin1d]

        scalar_cent.ravel()[cbin1d] = ss / corners_num
        data_cent.ravel()[cbin1d] = tt / corners_num

    return data_cent, scalar_cent


@NUMBA
def _get_bin_grads(cbin_nd, data_edge):
    ndim = data_edge.ndim
    eshape = data_edge.shape
    cshape = np.zeros((ndim,), dtype='int32')
    for dd in range(ndim):
        cshape[dd] = eshape[dd] - 1

    eshape_rev_prod = _get_shape_rev_prod(eshape)

    corners_shape = 2 * np.ones((ndim,), dtype=T_INT)
    corners_shape = to_fixed_tuple(corners_shape, ndim)
    corn_rev_prod = _get_shape_rev_prod(corners_shape)
    corners_num = 2 ** ndim
    grads_num = 2 ** (ndim - 1)

    corner = np.zeros(ndim, dtype=T_INT)
    data_grad_vals = np.zeros((ndim,  2))
    data_grad = np.zeros(ndim)

    # calculate gradients along each dimension
    for dd in range(ndim):
        # iterate over all corners
        for off1d in np.arange(corners_num):
            offset_nd = _unravel_index(off1d, corn_rev_prod)
            # calc the location of the corner in the edge-arrays
            for dd in range(ndim):
                corner[dd] = cbin_nd[dd] + offset_nd[dd]

            # get the edge-values for this corner
            ebin1d = _ravel_index(corner, eshape_rev_prod)

            # calculate the average over every dimension, *except* this dimension
            #     store averages for left-edge vs. right-edge of this dimension
            if offset_nd[dd] == 0:
                data_grad_vals[dd, 0] += data_edge.ravel()[ebin1d]
            else:
                data_grad_vals[dd, 1] += data_edge.ravel()[ebin1d]

        # calculate and store gradient values
        data_grad[dd] = (data_grad_vals[dd, 1] - data_grad_vals[dd, 0]) / grads_num

    return data_grad


@NUMBA
def _get_bin_center_grads(cbin_nd, data_edge):
    ndim = data_edge.ndim
    eshape = data_edge.shape
    cshape = np.zeros((ndim,), dtype='int32')
    for dd in range(ndim):
        cshape[dd] = eshape[dd] - 1

    eshape_rev_prod = _get_shape_rev_prod(eshape)

    corners_shape = 2 * np.ones((ndim,), dtype=T_INT)
    corners_shape = to_fixed_tuple(corners_shape, ndim)
    corn_rev_prod = _get_shape_rev_prod(corners_shape)
    corners_num = 2 ** ndim
    grads_num = 2 ** (ndim - 1)

    corner = np.zeros(ndim, dtype=T_INT)
    data_grad_vals = np.zeros((ndim,  2))
    data_grad = np.zeros(ndim)

    data_cent = 0.0

    # calculate gradients along each dimension
    for dd in range(ndim):
        # iterate over all corners
        for off1d in np.arange(corners_num):
            offset_nd = _unravel_index(off1d, corn_rev_prod)
            # calc the location of the corner in the edge-arrays
            for dd in range(ndim):
                corner[dd] = cbin_nd[dd] + offset_nd[dd]

            # get the edge-values for this corner
            ebin1d = _ravel_index(corner, eshape_rev_prod)
            if dd == 0:
                data_cent += data_edge.ravel()[ebin1d]

            # calculate the average over every dimension, *except* this dimension
            #     store averages for left-edge vs. right-edge of this dimension
            if offset_nd[dd] == 0:
                data_grad_vals[dd, 0] += data_edge.ravel()[ebin1d]
            else:
                data_grad_vals[dd, 1] += data_edge.ravel()[ebin1d]

        # calculate and store gradient values
        data_grad[dd] = (data_grad_vals[dd, 1] - data_grad_vals[dd, 0]) / grads_num

    data_cent = data_cent / corners_num

    return data_cent, data_grad


@NUMBA
def _get_centers_grads(data_edge):
    ndim = data_edge.ndim
    eshape = data_edge.shape
    cshape = np.zeros((ndim,), dtype='int32')
    for dd in range(ndim):
        cshape[dd] = eshape[dd] - 1

    cshape = to_fixed_tuple(cshape, ndim)
    cshape_rev_prod = _get_shape_rev_prod(cshape)
    eshape_rev_prod = _get_shape_rev_prod(eshape)

    corners_shape = 2 * np.ones((ndim,), dtype=T_INT)
    corners_shape = to_fixed_tuple(corners_shape, ndim)
    corn_rev_prod = _get_shape_rev_prod(corners_shape)
    corners_num = 2 ** ndim
    grads_num = 2 ** (ndim - 1)

    corner = np.zeros(ndim, dtype=T_INT)
    data_cent = np.zeros(cshape)
    data_corners = np.zeros(corners_shape)
    data_grad_vals = np.zeros((ndim,  2))
    data_grad = np.zeros(ndim)

    # ---- Iterate over each bin in the 'centers' arrays ----
    for cbin_nd in np.ndindex(cshape):
        # convert from Nd
        cbin_1d = _ravel_index(cbin_nd, cshape_rev_prod)

        # Store the values from the corners around each center-bin
        # calculate bin-center values
        tt = 0.0
        # iterate over the offsets to reach each corner, starting from the bin center
        for offset_nd in np.ndindex(corners_shape):
            off1d = _ravel_index(offset_nd, corn_rev_prod)
            # calc the location of the corner in the edge-arrays
            for dd in range(ndim):
                corner[dd] = cbin_nd[dd] + offset_nd[dd]

            # get the edge-values for this corner
            ebin1d = _ravel_index(corner, eshape_rev_prod)
            dtemp = data_edge.ravel()[ebin1d]

            # store corner values (for gradient calculation)
            data_corners.ravel()[off1d] = dtemp

            # increment average (i.e. bin-center) calculations
            tt = tt + dtemp

        # Store bin-center values
        data_cent.ravel()[cbin_1d] = tt / corners_num

        # calculate gradients along each dimension
        for dd in range(ndim):
            # zero-out previous values
            for jj in range(2):
                data_grad_vals[dd, jj] = 0.0
            # iterate over all corners
            for off1d in np.arange(corners_num):
                off = _unravel_index(off1d, corn_rev_prod)
                # calculate the average over every dimension, *except* this dimension
                #     store averages for left-edge vs. right-edge of this dimension
                if off[dd] == 0:
                    data_grad_vals[dd, 0] += data_corners.ravel()[off1d]
                else:
                    data_grad_vals[dd, 1] += data_corners.ravel()[off1d]

            # calculate and store gradient values
            data_grad[dd] = (data_grad_vals[dd, 1] - data_grad_vals[dd, 0]) / grads_num

    return data_cent, data_grad


@NUMBA
def _get_centers_grads_scalar(data_edge, scalar_edge):
    ndim = data_edge.ndim
    eshape = data_edge.shape
    cshape = np.zeros((ndim,), dtype='int32')
    for dd in range(ndim):
        cshape[dd] = eshape[dd] - 1

    cshape = to_fixed_tuple(cshape, ndim)
    cshape_rev_prod = _get_shape_rev_prod(cshape)
    eshape_rev_prod = _get_shape_rev_prod(eshape)

    corners_shape = 2 * np.ones((ndim,), dtype=T_INT)
    corners_shape = to_fixed_tuple(corners_shape, ndim)
    corn_rev_prod = _get_shape_rev_prod(corners_shape)
    corners_num = 2 ** ndim
    grads_num = 2 ** (ndim - 1)

    corner = np.zeros(ndim, dtype=T_INT)
    scalar_cent = np.zeros(cshape)
    data_cent = np.zeros(cshape)

    scalar_corners = np.zeros(corners_shape)
    scalar_grad_vals = np.zeros((ndim,  2))
    scalar_grad = np.zeros(ndim)
    data_corners = np.zeros(corners_shape)
    data_grad_vals = np.zeros((ndim,  2))
    data_grad = np.zeros(ndim)

    # ---- Iterate over each bin in the 'centers' arrays ----
    for cbin_nd in np.ndindex(cshape):
        # convert from Nd
        cbin_1d = _ravel_index(cbin_nd, cshape_rev_prod)

        # Store the values from the corners around each center-bin
        # calculate bin-center values
        ss = 0.0
        tt = 0.0
        # iterate over the offsets to reach each corner, starting from the bin center
        for offset_nd in np.ndindex(corners_shape):
            off1d = _ravel_index(offset_nd, corn_rev_prod)
            # calc the location of the corner in the edge-arrays
            for dd in range(ndim):
                corner[dd] = cbin_nd[dd] + offset_nd[dd]

            # get the edge-values for this corner
            ebin1d = _ravel_index(corner, eshape_rev_prod)
            stemp = scalar_edge.ravel()[ebin1d]
            dtemp = data_edge.ravel()[ebin1d]

            # store corner values (for gradient calculation)
            scalar_corners.ravel()[off1d] = stemp
            data_corners.ravel()[off1d] = dtemp

            # increment average (i.e. bin-center) calculations
            ss = ss + stemp
            tt = tt + dtemp

        # Store bin-center values
        scalar_cent.ravel()[cbin_1d] = ss / corners_num
        data_cent.ravel()[cbin_1d] = tt / corners_num

        # calculate gradients along each dimension
        for dd in range(ndim):
            # zero-out previous values
            for jj in range(2):
                scalar_grad_vals[dd, jj] = 0.0
                data_grad_vals[dd, jj] = 0.0
            # iterate over all corners
            for off1d in np.arange(corners_num):
                off = _unravel_index(off1d, corn_rev_prod)
                # calculate the average over every dimension, *except* this dimension
                #     store averages for left-edge vs. right-edge of this dimension
                if off[dd] == 0:
                    scalar_grad_vals[dd, 0] += scalar_corners.ravel()[off1d]
                    data_grad_vals[dd, 0] += data_corners.ravel()[off1d]
                else:
                    scalar_grad_vals[dd, 1] += scalar_corners.ravel()[off1d]
                    data_grad_vals[dd, 1] += data_corners.ravel()[off1d]

            # calculate and store gradient values
            scalar_grad[dd] = (scalar_grad_vals[dd, 1] - scalar_grad_vals[dd, 0]) / grads_num
            data_grad[dd] = (data_grad_vals[dd, 1] - data_grad_vals[dd, 0]) / grads_num

    return data_cent, scalar_cent, data_grad, scalar_grad


# =============================    V3    ===============================


@NUMBA
def _get_edge_cumsum(data):
    shape = data.shape
    csum = np.zeros(data.ndim, dtype=T_INT)
    for dd in range(data.ndim-1):
        csum[dd+1] = csum[dd] + shape[dd]
    return csum


@NUMBA
def _get_edge_at_dim(flat_edges, dd, ii, shape_csum):
    jj = shape_csum[dd] + ii
    return flat_edges[jj]


@NUMBA
def _sample(edges, data_edge, nsamp, flat_tol=1e-2):
    """

    edges : needs to be flat, with elements matching `data_edge.shape`

    """
    ndim = data_edge.ndim

    data_cent = _get_centers(data_edge)
    idx = np.argsort(data_cent.ravel())
    csum = np.cumsum(data_cent.ravel()[idx])
    cshape = data_cent.shape
    edge_shape_csum = np.concatenate([[0, ], np.cumsum(data_edge.shape)[:-1]])

    size1d = csum.size
    cshape_rev_prod = _get_shape_rev_prod(cshape)

    data_out = np.zeros((ndim, nsamp))

    # Draw random values
    #     random number for location in CDF, and additional random for position in each dimension of bin
    rand = np.random.uniform(0.0, 1.0, (1+ndim, nsamp))
    intrabin_locs = rand[1:, :]
    rand = rand[0, :]

    last_cbin1d = -1
    cbin1d = 0
    for ii in range(nsamp):
        # Find which bin each random value should go in
        while (rand[ii] < csum[cbin1d+1]) & (cbin1d < size1d):
            # 1D bin, in sorted array
            cbin1d += 1

        # Calculate values for this bin, if it's the first time it's reached
        if cbin1d != last_cbin1d:
            # Convert from 1D array to ND array
            cbin_nd = _unravel_index(idx[cbin1d], cshape_rev_prod)
            data_grad = _get_bin_grads(cbin_nd, data_edge)

        for dd in range(ndim):
            ee = cbin_nd[dd]
            elo = _get_edge_at_dim(edges, dd, ee, edge_shape_csum)
            ehi = _get_edge_at_dim(edges, dd, ee+1, edge_shape_csum)
            # wid = edges[dd][ee+1] - edges[dd][ee]
            wid = ehi - elo
            loc = intrabin_locs[dd, ii]

            dgrad = data_grad[dd]
            # use uniform sampling for flat gradients
            temp = np.fabs(dgrad)
            norm = data_cent.ravel()[cbin1d]
            if norm != 0.0:
                temp = temp / norm
            if (temp < flat_tol):
                data_out[dd, ii] = elo + loc * wid
            # negative slope: interpolate left to right
            elif dgrad > 0.0:
                data_out[dd, ii] = elo + wid * np.sqrt(loc)
            # negative slope: interpolate right to left
            else:   # dgrad < 0.0:
                data_out[dd, ii] = ehi - wid * np.sqrt(loc)

        last_cbin1d = cbin1d

    return data_out


@NUMBA
def _sample_scalar(edges, data_edge, scalar_edge, nsamp, flat_tol=1e-2):
    ndim = data_edge.ndim
    print("ndim = ", ndim)

    data_cent = _get_centers(data_edge)
    idx = np.argsort(data_cent.ravel())
    csum = np.cumsum(data_cent.ravel()[idx])
    csum = csum / csum[-1]
    print("csum = ", csum)
    cshape = data_cent.shape
    # edge_shape_csum = np.concatenate([[0,], np.cumsum(data_edge.shape)[:-1]])
    edge_shape_csum = _get_edge_cumsum(data_edge)
    print("edge_shape_csum = ", edge_shape_csum)

    size1d = csum.size
    cshape_rev_prod = _get_shape_rev_prod(cshape)
    print("cshape_rev_prod = ", cshape_rev_prod)

    data_out = np.zeros((ndim, nsamp))
    scalar_out = np.zeros(nsamp)

    # Draw random values
    #     random number for location in CDF, and additional random for position in each dimension of bin
    rand = np.random.uniform(0.0, 1.0, (1+ndim, nsamp))
    intrabin_locs = rand[1:, :]
    rand = rand[0, :]
    rand = np.sort(rand)

    last_cbin1d = -1
    cbin1d = 0
    for ii in range(nsamp):
        print("ii = ", ii, rand[ii], csum[cbin1d], csum[cbin1d+1])
        # Find which bin each random value should go in
        while (rand[ii] < csum[cbin1d+1]) & (cbin1d < size1d):
            # 1D bin, in sorted array
            cbin1d += 1

        print("cbin1d = ", cbin1d)
        # Calculate values for this bin, if it's the first time it's reached
        if cbin1d != last_cbin1d:
            # Convert from 1D array to ND array
            cbin_nd = _unravel_index(idx[cbin1d], cshape_rev_prod)
            print("cbin_nd = ", cbin_nd)
            data_grad = _get_bin_grads(cbin_nd, data_edge)
            scalar_cent, scalar_grad = _get_bin_center_grads(cbin_nd, scalar_edge)
            # data_grad = np.zeros(ndim)
            # scalar_cent = 0
            # scalar_grad = np.zeros(ndim)

        scalar_out[ii] = scalar_cent

        '''
        for dd in range(ndim):
            ee = cbin_nd[dd]
            print("\n", dd, ee)
            elo = _get_edge_at_dim(edges, dd, ee, edge_shape_csum)
            ehi = _get_edge_at_dim(edges, dd, ee+1, edge_shape_csum)
            print(elo, ehi)
            # wid = edges[dd][ee+1] - edges[dd][ee]
            wid = ehi - elo
            loc = intrabin_locs[dd, ii]

            scalar_out[ii] += scalar_grad[dd] * (loc - 0.5)
            print(scalar_out[ii])

            dgrad = data_grad[dd]
            # use uniform sampling for flat gradients
            temp = np.fabs(dgrad)
            norm = data_cent.ravel()[cbin1d]
            if norm != 0.0:
                temp = temp / norm
            if (temp < flat_tol):
                data_out[dd, ii] = elo + loc * wid
            # negative slope: interpolate left to right
            elif dgrad > 0.0:
                data_out[dd, ii] = elo + wid * np.sqrt(loc)
            # negative slope: interpolate right to left
            else:   # dgrad < 0.0:
                data_out[dd, ii] = ehi - wid * np.sqrt(loc)

            print(data_out[dd, ii])
        '''

        last_cbin1d = cbin1d

    return data_out, scalar_out


# ===========================   V1   ==================================


'''
@NUMBA
def _sample(edges, data_edge, nsamp, flat_tol=1e-2):
    ndim = data_edge.ndim

    data_cent, data_grad = _get_centers_grads(data_edge)
    cshape = data_cent.shape

    idx = np.argsort(data_cent.ravel())
    csum = np.cumsum(data_cent.ravel()[idx])

    size1d = csum.size
    cshape_rev_prod = _get_shape_rev_prod(cshape)

    scalar_out = np.zeros(nsamp)
    data_out = np.zeros((ndim, nsamp))

    # Draw random values
    #     random number for location in CDF, and additional random for position in each dimension of bin
    rand = np.random.uniform(0.0, 1.0, (1+ndim, nsamp))
    intrabin_locs = rand[1:, :]
    rand = rand[0, :]

    last_bin1d = -1
    bin1d = 0
    for ii in range(nsamp):
        # Find which bin each random value should go in
        while (rand[ii] < csum[bin1d+1]) & (bin1d < size1d):
            # 1D bin, in sorted array
            bin1d += 1

        # Calculate values for this bin, if it's the first time it's reached
        if bin1d != last_bin1d:
            # Convert from 1D array to ND array
            bin3d = _unravel_index(idx[bin1d], cshape_rev_prod)

        for dd in range(ndim):
            ee = bin3d[dd]
            wid = edges[dd][ee+1] - edges[dd][ee]
            loc = intrabin_locs[dd, ii]

            dgrad = data_grad[dd]
            # use uniform sampling for flat gradients
            temp = np.fabs(dgrad)
            norm = data_cent.ravel()[bin1d]
            if norm != 0.0:
                temp = temp / norm
            if (temp < flat_tol):
                data_out[dd, ii] = edges[dd][ee] + loc * wid
            # negative slope: interpolate left to right
            elif dgrad > 0.0:
                data_out[dd, ii] = edges[dd][ee] + wid * np.sqrt(loc)
            # negative slope: interpolate right to left
            else:   # dgrad < 0.0:
                data_out[dd, ii] = edges[dd][ee+1] - wid * np.sqrt(loc)

        last_bin1d = bin1d

    return data_out, scalar_out


@NUMBA
def _sample_scalar(edges, data_edge, scalar_edge, nsamp, flat_tol=1e-2):
    ndim = data_edge.ndim

    data_cent, scalar_cent, data_grad, scalar_grad = _get_centers_grads_scalar(data_edge, scalar_edge)
    cshape = data_cent.shape

    idx = np.argsort(data_cent.ravel())
    csum = np.cumsum(data_cent.ravel()[idx])

    size1d = csum.size
    cshape_rev_prod = _get_shape_rev_prod(cshape)

    scalar_out = np.zeros(nsamp)
    data_out = np.zeros((ndim, nsamp))

    # Draw random values
    #     random number for location in CDF, and additional random for position in each dimension of bin
    rand = np.random.uniform(0.0, 1.0, (1+ndim, nsamp))
    intrabin_locs = rand[1:, :]
    rand = rand[0, :]

    last_bin1d = -1
    bin1d = 0
    for ii in range(nsamp):
        # Find which bin each random value should go in
        while (rand[ii] < csum[bin1d+1]) & (bin1d < size1d):
            # 1D bin, in sorted array
            bin1d += 1

        # Calculate values for this bin, if it's the first time it's reached
        if bin1d != last_bin1d:
            # Convert from 1D array to ND array
            bin3d = _unravel_index(idx[bin1d], cshape_rev_prod)

        scalar_out[ii] = scalar_cent.ravel()[bin1d]

        for dd in range(ndim):
            ee = bin3d[dd]
            wid = edges[dd][ee+1] - edges[dd][ee]

            loc = intrabin_locs[dd, ii]
            scalar_out[ii] += scalar_grad[dd] * (loc - 0.5)

            dgrad = data_grad[dd]
            # use uniform sampling for flat gradients
            temp = np.fabs(dgrad)
            norm = data_cent.ravel()[bin1d]
            if norm != 0.0:
                temp = temp / norm
            if (temp < flat_tol):
                data_out[dd, ii] = edges[dd][ee] + loc * wid
            # negative slope: interpolate left to right
            elif dgrad > 0.0:
                data_out[dd, ii] = edges[dd][ee] + wid * np.sqrt(loc)
            # negative slope: interpolate right to left
            else:   # dgrad < 0.0:
                data_out[dd, ii] = edges[dd][ee+1] - wid * np.sqrt(loc)

        last_bin1d = bin1d

    return data_out, scalar_out
'''


# ===========================   V0   ==================================


'''
@NUMBA
def _sample_scalar(corners, edges, data_edge, data_cent, csum, idx, scalar_edge, nsamp, flat_tol=1e-2):
    ndim = len(edges)
    cshape = data_cent.shape
    # cshape_arr = np.array(cshape, dtype=T_INT)
    bin3d = np.zeros((ndim,), dtype=T_INT)
    size1d = csum.size

    # shape_rev_prod = _get_shape_rev_prod(cshape_arr)
    shape_rev_prod = _get_shape_rev_prod(cshape)

    corner_shape = corners.shape
    corner_num = 2 ** ndim
    corn_rev_prod = _get_shape_rev_prod(corner_shape)

    scalar_vals = np.zeros(corner_shape)
    scalar_grad_vals = np.zeros((ndim,  2))
    scalar_grad = np.zeros(ndim)
    data_vals = np.zeros(corner_shape)
    data_grad_vals = np.zeros((ndim,  2))
    data_grad = np.zeros(ndim)

    scalar_cent = np.zeros(cshape)
    scalar_out = np.zeros(nsamp)
    data_out = np.zeros((ndim, nsamp))
    corner = np.zeros(ndim, dtype=T_INT)

    # Draw random values
    #     random number for location in CDF, and additional random for position in each dimension of bin
    rand = np.random.uniform(0.0, 1.0, (1+ndim, nsamp))
    intrabin_locs = rand[1:, :]
    rand = rand[0, :]

    # Find which bin each random value should go in
    #    note that pre-sorting `rand` does speed up searchsorted: 'github.com/numpy/numpy/issues/10937'
    rand = np.sort(rand)
    last_bin1d = -1
    bin1d = 0
    ss = 0.0
    for ii in range(nsamp):
        # Find bin for this sample
        while (rand[ii] < csum[bin1d+1]) & (bin1d < size1d):
            # 1D bin, in sorted array
            bin1d += 1

        # Calculate values for this bin, if it's the first time it's reached
        if bin1d != last_bin1d:
            # Convert from 1D array to ND array
            bin3d = _unravel_index(idx[bin1d], shape_rev_prod)

            # calculate scalar center at this bin
            ss = 0.0
            for off1d in np.arange(corner_num):
                offset = _unravel_index(off1d, corn_rev_prod)
                for dd in range(ndim):
                    corner[dd] = bin3d[dd] + offset[dd]
                # off1d = _ravel_index(offset, corn_rev_prod)
                corn1d = _ravel_index(corner, shape_rev_prod)
                scalar_vals.ravel()[off1d] = scalar_edge.ravel()[corn1d]
                data_vals.ravel()[off1d] = data_edge.ravel()[corn1d]
                ss = ss + scalar_edge.ravel()[corn1d]

            # scalar_cent[bin3d] = ss / corner_num
            scalar_cent.ravel()[bin1d] = ss / corner_num

            temp = np.zeros((2, 2, 2))

            for dd in range(ndim):
                scalar_grad_vals[dd, 0] = 0.0
                data_grad_vals[dd, 0] = 0.0
                for off1d in np.arange(corner_num):
                    off = _unravel_index(off1d, corn_rev_prod)
                    if off[dd] == 0:
                        scalar_grad_vals[dd, 0] += scalar_vals.ravel()[off1d]
                        data_grad_vals[dd, 0] += data_vals.ravel()[off1d]
                    else:
                        scalar_grad_vals[dd, 1] += scalar_vals.ravel()[off1d]
                        data_grad_vals[dd, 1] += data_vals.ravel()[off1d]

                scalar_grad[dd] = scalar_grad_vals[dd, 1] - scalar_grad_vals[dd, 0]
                data_grad[dd] = data_grad_vals[dd, 1] - data_grad_vals[dd, 0]

        scalar_out[ii] = scalar_cent.ravel()[bin1d]

        for dd in range(ndim):
            ee = bin3d[dd]
            wid = edges[dd][ee+1] - edges[dd][ee]

            loc = intrabin_locs[dd, ii]
            scalar_out[ii] += scalar_grad[dd] * (loc - 0.5)

            dgrad = data_grad[dd]
            # use uniform sampling for flat gradients
            temp = np.fabs(dgrad)
            norm = data_cent.ravel()[bin1d]
            if norm != 0.0:
                temp = temp / norm
            if (temp < flat_tol):
                data_out[dd, ii] = edges[dd][ee] + loc * wid
            # negative slope: interpolate left to right
            elif dgrad > 0.0:
                data_out[dd, ii] = edges[dd][ee] + wid * np.sqrt(loc)
            # negative slope: interpolate right to left
            else:   # dgrad < 0.0:
                data_out[dd, ii] = edges[dd][ee+1] - wid * np.sqrt(loc)

        last_bin1d = bin1d
        # break
        # } for ii

    return data_out, scalar_out


@NUMBA
def _sample(corners, edges, data_edge, data_cent, csum, idx, nsamp, flat_tol=1e-2):
    ndim = len(edges)
    cshape = data_cent.shape
    # cshape_arr = np.array(cshape, dtype=T_INT)
    bin3d = np.zeros((ndim,), dtype=T_INT)
    size1d = csum.size

    # shape_rev_prod = _get_shape_rev_prod(cshape_arr)
    shape_rev_prod = _get_shape_rev_prod(cshape)

    corner_shape = corners.shape
    corner_num = 2 ** ndim
    corn_rev_prod = _get_shape_rev_prod(corner_shape)

    data_vals = np.zeros(corner_shape)
    data_grad_vals = np.zeros((ndim,  2))
    data_grad = np.zeros(ndim)

    data_out = np.zeros((ndim, nsamp))
    # corner = np.zeros(ndim, dtype=T_INT)

    # Draw random values
    #     random number for location in CDF, and additional random for position in each dimension of bin
    rand = np.random.uniform(0.0, 1.0, (1+ndim, nsamp))
    intrabin_locs = rand[1:, :]
    rand = rand[0, :]

    # Find which bin each random value should go in
    #    note that pre-sorting `rand` does speed up searchsorted: 'github.com/numpy/numpy/issues/10937'
    rand = np.sort(rand)
    last_bin1d = -1
    bin1d = 0
    for ii in range(nsamp):
        # Find bin for this sample
        while (rand[ii] < csum[bin1d+1]) & (bin1d < size1d):
            # 1D bin, in sorted array
            bin1d += 1

        # Calculate values for this bin, if it's the first time it's reached
        if bin1d != last_bin1d:
            # Convert from 1D array to ND array
            bin3d = _unravel_index(idx[bin1d], shape_rev_prod)

            # temp = np.zeros((2, 2, 2))

            for dd in range(ndim):
                data_grad_vals[dd, 0] = 0.0
                for off1d in np.arange(corner_num):
                    off = _unravel_index(off1d, corn_rev_prod)
                    if off[dd] == 0:
                        data_grad_vals[dd, 0] += data_vals.ravel()[off1d]
                    else:
                        data_grad_vals[dd, 1] += data_vals.ravel()[off1d]

                data_grad[dd] = data_grad_vals[dd, 1] - data_grad_vals[dd, 0]

        for dd in range(ndim):
            ee = bin3d[dd]
            wid = edges[dd][ee+1] - edges[dd][ee]

            loc = intrabin_locs[dd, ii]

            dgrad = data_grad[dd]
            # use uniform sampling for flat gradients
            temp = np.fabs(dgrad)
            norm = data_cent.ravel()[bin1d]
            if norm != 0.0:
                temp = temp / norm
            if (temp < flat_tol):
                data_out[dd, ii] = edges[dd][ee] + loc * wid
            # negative slope: interpolate left to right
            elif dgrad > 0.0:
                data_out[dd, ii] = edges[dd][ee] + wid * np.sqrt(loc)
            # negative slope: interpolate right to left
            else:   # dgrad < 0.0:
                data_out[dd, ii] = edges[dd][ee+1] - wid * np.sqrt(loc)

        last_bin1d = bin1d
        # break
        # } for ii

    return data_out
'''
