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
        # NOTE: require that `data` is edge-values (instead of center values)
        # FIX: Consider allowing center-values, but be careful about how to handle, and
        #      implications for outlier samples
        if np.all(edge_shape == shape):
            data_edge = data
            # find PDF at grid center-points
            # data_cent = utils.midpoints(data_edge, log=False, axis=None)
        else:
            err = "Shape of edges ({}) inconsistent with data ({})!".format(edge_shape, shape)
            raise ValueError(err)

        # `scalar` must be shaped as either `data_cent` or `data_edge`
        #    if the latter, it will be converted to `data_cent` by averaging
        if scalar is not None:
            scalar = np.asarray(scalar)
            if np.all(scalar.shape == edge_shape):
                pass
            else:
                err = "Shape of `scalar` ({}) does not match `data` ({})!".format(scalar.shape, data.shape)
                raise ValueError(err)

        self._edges = edges
        self._data_edge = data_edge
        self._shape_edge = tuple(edge_shape)
        self._ndim = ndim
        self._grid = None

        self._scalar_edge = scalar
        self._scalar_cent = None

        self._init_data()

        self._shape_cent = self._data_cent.shape

        return

    def _init_data(self):
        data_cent = utils.midpoints(self._data_edge, log=False, axis=None)
        idx, csum = self._data_to_cumulative(data_cent)

        raise RuntimeError(f"HAS PROBABILITY CONSERVATION BEEN FIXED YET?!")

        self._idx = idx
        self._csum = csum
        self._data_cent = data_cent
        return

    def _get_scalar_cent(self):
        if self._scalar_edge is None:
            return None

        if self._scalar_cent is None:
            self._scalar_cent = utils.midpoints(self._scalar_edge, log=False, axis=None)

        return self._scalar_cent

    def sample(self, nsamp, interpolate=True, return_scalar=None):
        nsamp = int(nsamp)
        data_edge = self._data_edge
        scalar = self._scalar_edge
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
            scalar_cent = self._get_scalar_cent()
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

    @property
    def grid(self):
        if self._grid is None:
            self._grid = np.meshgrid(*self._edges, indexing='ij')

        return self._grid


class Sample_Outliers(Sample_Grid):

    def __init__(self, edges, data, threshold=10.0, *args, **kwargs):
        super().__init__(edges, data, *args, **kwargs)

        # Note: `data_cent` has already been converted from density to mass (i.e. integrating each cell)
        #       this happened in `__init__()` ==> `_init_data()`
        #       `data_edge` is still a density (at the corners of each cell)
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

        # Find the center-of-mass of each cell (based on density corner values)
        coms = self.grid
        dens_edge = self._data_edge
        dens_cent = utils.midpoints(dens_edge, log=False, axis=None)
        coms = [utils.midpoints(dens_edge * ll, log=False, axis=None) / dens_cent for ll in coms]

        self._threshold = threshold
        self._data_ins = data_ins
        self._coms_ins = coms
        self._data_outs = data_outs
        return

    def sample(self, nsamp=None, **kwargs):
        """Outlier sample the distribution.

        Arguments
        ---------
        nsamp : int or None,
            The number of samples in the _outlier_ population only.

        """
        rv = kwargs.setdefault('return_scalar', False)
        if rv is not False:
            raise ValueError(f"Cannot use `scalar` values in `{self.__class__}`!")

        # if `nsamp` isn't given, assume outlier distribution values correspond to numbers
        #    and Poisson sample them
        # NOTE: `nsamp` corresponds only to the _"outliers"_ not the 'interior' points also
        if nsamp is None:
            nsamp = self._data_outs.sum()
            nsamp = np.random.poisson(nsamp)

        # sample outliers normally (using modified csum from `self._data_outs`)
        if nsamp > 0:
            vals_outs = super().sample(nsamp, **kwargs)
        else:
            msg = f"WARNING: outliers nsamp = {nsamp}!  outs.sum = {self._data_outs.sum():.4e}!"
            logging.warning(msg)
            vals_outs = [[] for ii in range(self._ndim)]

        # sample tracer/representative points from `self._data_ins`
        data_ins = self._data_ins
        nin = np.count_nonzero(data_ins)
        if nin < 1:
            msg = f"WARNING: in-liers nsamp = {nin}!  ins.sum() = {data_ins.sum():.4e}!"
            logging.warning(msg)

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
            vals_ins[dim, :] = self._coms_ins[dim][bin_numbers]

        # Combine interior-tracers and outlier-samples
        vals = np.concatenate([vals_outs, vals_ins], axis=-1)
        return nsamp, vals, weights

    def _init_data(self):
        """Calculate `data_cent` for sampling, integrate over each bin.

        NOTE: `idx` and `csum` are calculated directly in `Sample_Outliers.__init__()`
        """
        self._data_cent = utils.trapz_dens_to_mass(self._data_edge, self._edges, axis=None)
        # self._data_cent = utils.midpoints(data, log=False, axis=None)
        return


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
