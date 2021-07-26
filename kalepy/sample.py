"""Perform sampling of distributions and functions.
"""

import numpy as np

from kalepy import utils


def sample_grid(edges, data, nsamp, interpolate=True, flat_tol=1e-2):
    data = np.asarray(data)
    edges = [np.asarray(ee) for ee in edges]
    shape = data.shape
    ndim = data.ndim
    if len(edges) != ndim:
        raise ValueError(f"`edges` ({len(edges)=}) must be a 1D array for each dimension of `data` ({data.shape=})!")

    edge_shape = np.array([ee.size for ee in edges])
    if np.all(edge_shape == shape):
        data_edge = data
        # find PDF at grid center-points
        data_cent = utils.midpoints(data_edge, log=False, axis=None)
    elif np.all(edge_shape == np.array(shape) + 1):
        data_edge = None
        data_cent = data
    else:
        err = "Shape of edges ({}) inconsistent with data ({})!".format(edge_shape, shape)
        raise ValueError(err)

    shape = data_cent.shape

    # Convert to flat (1D) array of values
    data_cent = data_cent.flatten()
    # sort in order of probability
    idx = np.argsort(data_cent)
    csum = data_cent[idx]
    # find cumulative distribution and normalize to [0.0, 1.0]
    csum = np.cumsum(csum)
    csum = np.concatenate([[0.0], csum/csum[-1]])
    # csum = (csum - csum[0]) / (csum[-1] - csum[0])

    # Draw random values
    #     random number for location in CDF, and additional random for position in each dimension of bin
    rand = np.random.uniform(0.0, 1.0, (1+len(shape), nsamp))
    np.random.shuffle(rand)
    rand, *intrabin = rand

    # Find which bin each random value should go in
    #    note that pre-sorting `rand` does speed up searchsorted: 'github.com/numpy/numpy/issues/10937'
    rand = np.sort(rand)
    jj = np.searchsorted(csum, rand) - 1

    # Convert indices from sorted-csum back to original `data_cent` ordering
    ii = idx[jj]
    # Convert from flat (1D) indices into ND indices
    bin_numbers = np.unravel_index(ii, shape)

    # Convert from bin index to grid values
    vals = np.zeros_like(intrabin)
    for dim, (edge, bidx) in enumerate(zip(edges, bin_numbers)):
        # Width of bin-edges in this dimension
        wid = np.diff(edge)

        # Random location, in this dimension, for each bin
        loc = intrabin[dim]

        # random-uniform
        if (data_edge is None) or (not interpolate):
            vals[dim, :] = edge[bidx] + wid[bidx] * loc

        # random-linear
        else:
            edge = np.asarray(edge)
            grad = np.diff(data_edge, axis=dim)
            nums = list(np.arange(ndim))
            nums.pop(dim)
            grad = utils.midpoints(grad, log=False, axis=nums)
            grad = grad.flatten()[jj]
            # grad = grad.flatten()[ii]

            grad_frac = np.fabs(grad)
            grad_frac /= grad_frac.max()
            flat = (grad_frac < flat_tol)
            pos = (grad > 0.0) & ~flat
            zer = np.ones_like(pos)
            zer[pos] = False
            vals[dim, pos] = edge[bidx][pos] + wid[bidx][pos] * np.sqrt(loc[pos])
            neg = (grad < 0.0) & ~flat
            zer[neg] = False
            vals[dim, neg] = edge[bidx+1][neg] - wid[bidx][neg] * np.sqrt(loc[neg])

            vals[dim, zer] = edge[bidx][zer] + wid[bidx][zer] * loc[zer]

    return vals
