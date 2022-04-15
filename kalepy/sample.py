"""Perform sampling of distributions and functions.
"""
# from datetime import datetime
import logging

# import numba
import numpy as np

from kalepy import utils

__all__ = [
    'Sample_Grid', 'Sample_Outliers', 'sample_grid', 'sample_outliers', 'sample_grid_proportional'
]

_DEBUG = False


class Sample_Grid:
    """Sample from a given probability distribution evaluated on a regular grid.

    The grid has probability densities (`dens`) evaluated at the grid edges, and probability masses
    (`mass`) corresponding to the centroid of each bin.  The centroids are calculated from the edge
    positions, weighted by probability density.  If `mass` is not given, it is calculated by
    integrating the densities over each bin (using the trapezoid rule).

    ## Process for drawing 'N' samples from the distributon:
    1) Using the masses of each bin, the CDF is calculated.
    2) N random values are chosen, and the CDF is inverted to find which bin they correspond to.
        - The CDF is flattened into 1D to accomodate any dimensionality of grid, and then the chosen
        bins are re-mapped to ND space.
    3) Within each bin, the position of each drawn sample is chosen proportionally to the probability
    density, based on the density-gradient within each cell.

    """

    def __init__(self, edges, dens, mass=None, scalar_dens=None, scalar_mass=None):
        """Initialize `Sample_Grid` with the given grid edges and probability distribution.

        Arguments
        ---------
        edges
            Bin edges along each dimension.
        dens : array_like of scalar
            Probability density evaluated at grid edges.
        mass : array_like of scalar  or  `None`
            Probability mass (i.e. number of samples) for each bin.  Evaluated at bin centers or
            centroids.
            If no `mass` is given, it is calculated by integrating `dens` over each bin using the
            trapezoid rule.  See: `_init_data()`.

        """
        dens = np.asarray(dens)
        shape = dens.shape
        ndim = dens.ndim
        # for 1D data, allow `edges` to be a 1D array;  convert manually to list of array
        if ndim == 1 and utils.really1d(edges):
            edges = [edges]
        edges = [np.asarray(ee) for ee in edges]
        if len(edges) != ndim:
            err = "`edges` (len(edges)={}) must be a 1D array for each dimension of `dens` (dens.shape={})!".format(
                len(edges), dens.shape)
            raise ValueError(err)

        shape_edges = np.array([ee.size for ee in edges])
        # require that `dens` is edge-values (instead of center values)
        if not np.all(shape_edges == shape):
            err = "Shape of `edges` ({}) inconsistent with `dens` ({})!".format(shape_edges, shape)
            raise ValueError(err)

        shape_bins = [sh - 1 for sh in shape_edges]
        # `scalar` must be shaped as either `data_cent` or `data_edge`
        #    if the latter, it will be converted to `data_cent` by averaging
        # if scalar is not None:
        #     scalar = np.asarray(scalar)
        #     if np.all(scalar.shape == shape_edges):
        #         pass
        #     else:
        #         err = "Shape of `scalar` ({}) does not match `data` ({})!".format(scalar.shape, dens.shape)
        #         raise ValueError(err)

        self._edges = edges
        self._dens = dens
        self._mass = mass
        self._shape_edges = tuple(shape_edges)
        self._shape_bins = tuple(shape_bins)
        self._ndim = ndim
        self._grid = None

        self._scalar_dens = scalar_dens
        self._scalar_mass = scalar_mass

        self._init_data()

        return

    def _init_data(self):
        if self._mass is None:
            self._mass = utils.trapz_dens_to_mass(self._dens, self._edges, axis=None)
        if (self._scalar_mass is None) and (self._scalar_dens is not None):
            self._scalar_mass = utils.trapz_dens_to_mass(self._scalar_dens, self._edges, axis=None)

        idx, csum = _data_to_cumulative(self._mass)
        self._idx = idx
        self._csum = csum
        return

    def sample(self, nsamp=None, interpolate=True, return_scalar=None):
        """Sample from the probability distribution.

        Arguments
        ---------
        nsamp : scalar or None
        interpolate : bool
        return_scalar : bool

        Returns
        -------
        vals : (D, N) ndarray of scalar

        """
        dens = self._dens
        scalar_dens = self._scalar_dens
        edges = self._edges

        # ---- initialize parameters
        if interpolate and (dens is None):
            logging.info("`dens` is None, cannot interpolate sampling")
            interpolate = False

        # ensure PDF `dens` is properly normalized for interpolation
        #     this normalization is based on each bin having a domain of [0.0, 1.0] during intrabin linear
        #     interpolation sampling (`_intrabin_linear_interp()`)
        if interpolate:
            dens = dens / (dens.sum() / dens.size)

        # If no number of samples are given, assume that the units of `self._mass` are number of samples, and choose
        # the total numbe of samples to be the total of this
        if nsamp is None:
            nsamp = self._mass.sum()
        nsamp = int(nsamp)

        if return_scalar is None:
            return_scalar = (scalar_dens is not None)
        elif return_scalar and (scalar_dens is None):
            return_scalar = False
            logging.warning("WARNING: no `scalar` initialized, but `return_scalar`=True!")

        # ---- Get generalized sampling locations

        # Choose random bins, proportionally to `mass`, and positions within bins (uniformly distributed)
        #     `bin_numbers_flat` (N*D,) are the index numbers for bins in flattened 1D array of length N*D
        #     `intrabin_locs` (D, N) are position [0.0, 1.0] within each bin for each sample in each dimension
        bin_numbers_flat, intrabin_locs = self._random_bins(nsamp)
        # Convert from flat (N,) indices into ND indices;  (D, N) for D dimensions, N samples (`nsamp`)
        bin_numbers = np.unravel_index(bin_numbers_flat, self._shape_bins)

        # If scalars are also being sampled: find scalar value for bin centers (i.e. bin averages)
        #     this will be updated/improved if `interpolation=True`
        if return_scalar:
            scalar_mass = self._scalar_mass
            scalar_values = scalar_mass[bin_numbers]

        # ---- Place samples in each dimension

        vals = np.zeros_like(intrabin_locs)
        for dim, (edge, bidx) in enumerate(zip(edges, bin_numbers)):
            # Width of bins in this dimension
            wid = np.diff(edge)

            # Random location, in this dimension, for each bin. Relative position, i.e. between [0.0, 1.0]
            loc = intrabin_locs[dim]

            # Uniform / no-interpolation :: random-uniform within each bin
            if (not interpolate):
                vals[dim, :] = edge[bidx] + wid[bidx] * loc

            # Interpolated :: random-linear proportional to bin gradients (i.e. slope across bin in each dimension)
            else:
                edge = np.asarray(edge)

                # Find the gradient along this dimension (using center-values in other dimensions)
                _grad = _grad_along(dens, dim)
                # get the gradient for each sample
                grad = _grad.flat[bin_numbers_flat]
                # interpolate edge values in this dimension
                vals[dim, :] = _intrabin_linear_interp(edge, wid, loc, bidx, grad)

            # interpolate scalar values also
            if return_scalar and interpolate:
                grad = _grad_along(scalar_dens, dim)
                grad = grad.flat[bin_numbers_flat]
                # shift `loc` (location within bin) to center point
                scalar_values += grad * (loc - 0.5)

        if return_scalar:
            return vals, scalar_values

        return vals

    def _random_bins(self, nsamp: int):
        """Choose bins and intrabin locations proportionally to the distribution.

        Choose bins proportionally to the probability mass (bin centroids), and intra-bin locations
        uniformly within the bin.  Intrabin locations will later be converted to proportional to
        the probability density (at bin edges).

        Arguments
        ---------
        nsamp : int,
            Number of samples to draw.

        Returns
        -------
        bin_numbers_flat :
        intrabin_locs :

        """
        csum = self._csum
        idx = self._idx

        # Draw random values
        #     random number for location in CDF (to determine which bin each value belongs in),
        #     and additional random for position in each dimension of bin
        sh = (1+self._ndim, nsamp)
        rand = np.random.uniform(0.0, 1.0, sh)
        # np.random.shuffle(rand)    # extra-step to avoid (rare/unlikely) structure in "random" data

        # `rand` shape: (N,) for N samples
        # `intrabin_locs` shape: (D, N) for D dimensions of data and N samples
        rand, *intrabin_locs = rand

        # Find which bin each random value should go in
        #    note that pre-sorting `rand` does speed up searchsorted: 'github.com/numpy/numpy/issues/10937'
        rand = np.sort(rand)
        sorted_bin_num = np.searchsorted(csum, rand) - 1

        # Convert indices from sorted-csum back to original `data_cent` ordering; (N,)
        bin_numbers_flat = idx[sorted_bin_num]
        return bin_numbers_flat, intrabin_locs

    @property
    def grid(self):
        if self._grid is None:
            self._grid = np.meshgrid(*self._edges, indexing='ij')

        return self._grid


class Sample_Outliers(Sample_Grid):
    """Sample outliers from a given probability distribution evaluated on a regular grid.

    "Outliers" are points in areas of low probability mass, which are drawn randomly.
    "Inliers"  are bins with high probability mass, which are assumed to be well represented by the
    centroid of those bins.  The `threshold` parameter determines the dividing point between low
    and high probability masses.

    The grid has probability densities (`dens`) evaluated at the grid edges, and probability masses
    (`mass`) corresponding to the centroid of each bin.  The centroids are calculated from the edge
    positions, weighted by probability density.  If `mass` is not given, it is calculated by
    integrating the densities over each bin (using the trapezoid rule).

    ## Process for drawing 'N' samples from the distributon:
    1) Bins with 'low' probability density (i.e. `mass` < `threshold`) are sampled in the same way
    as the super-class `Sample_Grid`.  These values are given a `weight` of 1.0.
    2) Bins with 'high' probability density (`mass` > `threshold`), are all used (i.e. with no
    stochasticity), where the location of sample points is the bin centroid (i.e. grid points
    weighted by probability density), and the `weight` is the total bin mass.

    """

    def __init__(self, edges, dens, threshold=10.0, **kwargs):
        super().__init__(edges, dens, **kwargs)

        # Note: `dens` has already been converted from density to mass (i.e. integrating each cell)
        #       this happened in `Sample_Grid.__init__()` ==> `Sample_Outliers._init_data()`
        #       `data_edge` is still a density (at the corners of each cell)
        mass_outs = np.copy(self._mass)

        # We're only going to stochastically sample from bins below the threshold value
        #     recalc `csum` zeroing out the values above threshold
        outs = (mass_outs > threshold)
        # print(f"Outside: {np.count_nonzero(outs)/outs.size:.4f}")
        # print(f"Inside : {np.count_nonzero(~outs)/outs.size:.4f}")
        mass_outs[outs] = 0.0
        idx, csum = _data_to_cumulative(mass_outs, prefilter=False)
        self._idx = idx
        self._csum = csum

        # We'll manually sample bins above threshold, so store those for later
        mass_ins = np.copy(self._mass)
        mass_ins[~outs] = 0.0

        # Find the center-of-mass of each cell (based on density corner values)
        coms = utils.centroids(self.grid, self._dens)
        # coms = self.grid
        # dens_edge = self._dens
        # dens_cent = utils.midpoints(dens_edge, log=False, axis=None)
        # coms = [utils.midpoints(dens_edge * ll, log=False, axis=None) / dens_cent for ll in coms]

        self._threshold = threshold
        self._mass_ins = mass_ins
        self._coms_ins = coms
        self._mass_outs = mass_outs
        return

    def _init_data(self):
        """Override `Sample_Grid._init_data()` to avoid calculating `idx` and `csum`, not needed yet
        """
        if self._mass is None:
            self._mass = utils.trapz_dens_to_mass(self._dens, self._edges, axis=None)
        if (self._scalar_mass is None) and (self._scalar_dens is not None):
            self._scalar_mass = utils.trapz_dens_to_mass(self._scalar_dens, self._edges, axis=None)
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
            nsamp = self._mass_outs.sum()
            nsamp = np.random.poisson(nsamp)

        # sample outliers normally (using modified csum from `self._data_outs`)
        if nsamp > 0:
            vals_outs = super().sample(nsamp, **kwargs)
        else:
            msg = f"WARNING: outliers nsamp = {nsamp}!  outs.sum = {self._mass_outs.sum():.4e}!"
            logging.warning(msg)
            vals_outs = [[] for ii in range(self._ndim)]

        # sample tracer/representative points from `self._data_ins`
        mass_ins = self._mass_ins
        nin = np.count_nonzero(mass_ins)
        if nin < 1:
            msg = f"WARNING: in-liers nsamp = {nin}!  ins.sum() = {mass_ins.sum():.4e}!"
            logging.warning(msg)

        ntot = nsamp + nin
        # weights needed for all points, but "outlier" points will have weigtht 1.0
        weights = np.ones(ntot)

        # Get the bin indices of all of the 'interior' bins (those where `mass_ins` are nonzero)
        bin_numbers_flat = (mass_ins.flat > 0.0)
        bin_numbers_flat = np.arange(bin_numbers_flat.size)[bin_numbers_flat]
        # Convert from 1D index to ND
        bin_numbers = np.unravel_index(bin_numbers_flat, self._shape_bins)
        # Set the weights to be the value of the bin-centers
        weights[nsamp:] = mass_ins[bin_numbers]

        # Find the 'interior' bin centroids and use those as tracer points for well-sampled data
        vals_ins = np.zeros((self._ndim, nin))
        for dim, (edge, bidx) in enumerate(zip(self._edges, bin_numbers)):
            vals_ins[dim, :] = self._coms_ins[dim][bin_numbers]

        # Combine interior-tracers and outlier-samples
        vals = np.concatenate([vals_outs, vals_ins], axis=-1)
        return nsamp, vals, weights


def sample_grid(edges, dens, nsamp=None, mass=None, scalar_dens=None, scalar_mass=None, **sample_kwargs):
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
    nsamp : int or None
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
    squeeze = (np.ndim(dens) == 1)
    sampler = Sample_Grid(edges, dens, mass=mass, scalar_dens=scalar_dens, scalar_mass=scalar_mass)
    samples = sampler.sample(nsamp=nsamp, **sample_kwargs)
    if squeeze:
        samples = samples.squeeze()
    return samples


def sample_grid_proportional(edges, dens, portion, nsamp, mass=None, **sample_kwargs):
    scalar_dens = dens / portion
    # Avoid NaN values
    scalar_dens[portion == 0.0] = 0.0
    sampler = Sample_Grid(edges, portion, scalar_dens=scalar_dens)
    vals, weight = sampler.sample(nsamp, **sample_kwargs)
    return vals, weight


def sample_outliers(edges, data, threshold, nsamp=None, mass=None, **sample_kwargs):
    """Sample a PDF randomly in low-density regions, and with weighted points at high-densities.

    Selects (semi-)random samples from the given PDF.  In high-density regions, bin centroids are
    used as representative points and recieve a corresponding (large) weight.  Low-density regions
    are sampled proportionally with actual (weight = one) points.

    Parameters
    ----------
    edges : list/tuple of array_like
        An iterable containing the grid edges for each dimension of the space.
    data : ndarray
        Array giving the PDF to sample.
    threshold : float
        Threshold mass below which true-samples should be drawn.  Representative (centroid) values
        will be chosen for bins above this threshold.
    nsamp : int, optional
        Number of samples to draw.
    mass : ndarray, optional
        Probability mass function determining the number of samples to draw in each bin.

    Returns
    -------
    vals
    weights

    """
    outliers = Sample_Outliers(edges, data, threshold=threshold, mass=mass)
    nsamp, vals, weights = outliers.sample(nsamp=nsamp, **sample_kwargs)
    return vals, weights


def _grad_along(data_edge, dim):
    grad = np.diff(data_edge, axis=dim)
    nums = list(np.arange(grad.ndim))
    nums.pop(dim)
    grad = utils.midpoints(grad, log=False, axis=nums)
    return grad


def _intrabin_linear_interp(edge, wid, loc, bidx, grad):
    """Perform linear interpolation within each bin, based on gradient information, for a particular dimension.

    Use the gradient across each bin to sample proportionally to a linear PDF within that bin.  Here the 'gradient' is
    actually just the delta-PDF value (i.e. ``y2 - y1``), which **must be calculated from a normalized (unitary) PDF**.
    The form of the PDF across each bin is assumed to be linear, and the CDF is inverted to convert from the random
    uniform positions (given by `loc`) to random-linear positions.

    Arguments
    ---------
        edge : (X+1,) ndarray of scalar,
            Location of grid edges in this dimension.  For a number `X` of bins, there are `X+1` edges.
        wid : (X,) ndarray of scalar,
            Width of grid bins in this dimension (``wid == np.diff(edge)``).
        loc : (S,) ndarray of scalar,
            Location of sample within each bin (i.e. [0.0, 1.0]) for each sample.
        bidx : (S,) ndarray of int,
            Bin index (i.e. [0, X]) for each sample.
        grad : (S,) ndarray of scalar,
            Gradient across the bin, in this dimension, for each sample.

    Returns
    -------
        vals : (S,) ndarray of float
            Sample locations in this dimension.

    """

    # Get the bin-width for each sample (i.e. the width of the bin that each sample is in)
    bw = wid[bidx]
    vals = np.zeros_like(grad)

    # sel = np.fabs(grad) > 1.0e-12
    sel = np.fabs(grad) > 1.0e-16
    # When the gradient is roughly flat, values maintain uniform random distribution
    vals[~sel] = loc[~sel]

    # Assume our distribution is parametrized as a linear PDF `y = y1 + grad * x`
    #     and assume our domain is [0.0, 1.0] on x
    # calculate `y1` to ensure the CDF is unitary
    y1 = 1.0 - grad[sel] / 2.0
    # invert the CDF (``F(x) = y1*x + grad * x^2 / 2``) to sample from PDF (still on [0.0, to 1.0])
    vals[sel] = 2 * loc[sel] * grad[sel] + y1 ** 2
    vals[sel] = (np.sqrt(vals[sel]) - y1) / grad[sel]
    # convert [0.0, 1.0] domain to the location and width of each bin
    vals = edge[bidx] + bw * vals

    # Make sure all values are within bounds of their bins
    if _DEBUG:
        bl = (vals < edge[bidx]) & ~np.isclose(vals, edge[bidx])
        br = (vals > edge[bidx+1]) & ~np.isclose(vals, edge[bidx+1])
        bads = bl | br
        if np.any(bads):
            logging.error(f"BAD!  {np.count_nonzero(bads)}/{bads.size}")
            logging.error(f"{vals[bads]=}")
            logging.error(f"{edge[bidx][bads]=}")
            logging.error(f"{loc[bads]=}")
            logging.error(f"{grad[bads]=}")
            logging.error(f"{wid[bidx][bads]=}")
            raise

    return vals


def _data_to_cumulative(mass, prefilter=False):
    # Convert to flat (1D) array of values
    mass = mass.flat

    # sort in order of probability
    if prefilter:
        logging.warning("WARNING: `prefilter`=True has not been tested in `_data_to_cumulative()`!!")
        csum = np.zeros_like(mass)
        mass = mass[mass > 0.0]
        mnum = mass.size
        beg = csum.size - mnum
        # print(f"{mnum=}, {beg=}")
        idx = np.argsort(mass)
        csum[beg:] = mass[idx]
        idx = np.concatenate([np.arange(beg), idx + beg])
    else:
        idx = np.argsort(mass)
        csum = mass[idx]

    # find cumulative distribution and normalize to [0.0, 1.0]
    csum = np.cumsum(csum)
    temp = 1.0 if csum[-1] == 0.0 else csum[-1]
    csum = np.concatenate([[0.0], csum/temp])
    return idx, csum
