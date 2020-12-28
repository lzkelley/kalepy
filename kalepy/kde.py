"""
"""
import logging
import six
import copy

import numpy as np
import scipy as sp

from kalepy import kernels, utils, _NUM_PAD
from kalepy import _BANDWIDTH_DEFAULT


class KDE(object):
    """Core class and primary API for using `kalepy`, by constructin a KDE based on given data.

    The `KDE` class acts as an API to the underlying `kernel` structures and methods.  From the
    passed data, a 'bandwidth' is calculated and/or set (using optional specifications using the
    `bandwidth` argument).  A `kernel` is constructed (using optional specifications in the
    `kernel` argument) which performs the calculations of the kernel density estimation.


    Notes
    -----

    *Reflection*

        Reflective boundary conditions can be used to better reconstruct a PDF that is known to
        have finite support (i.e. boundaries outside of which the PDF should be zero).

        The `pdf` and `resample` methods accept the keyword-argument (kwarg) `reflect` to specify
        that a reflecting boundary should be used.

        reflect : (D,) array_like, None (default)
            Locations at which reflecting boundary conditions should be imposed.
            For each dimension `D`, a pair of boundary locations (for: lower, upper) must be
            specified, or `None`.  `None` can also be given to specify no boundary at that
            location.

            If a pair of boundaries are given, then the
            first value corresponds to the lower boundary, and the second value to the upper
            boundary, in that dimension.  If there should only be a single lower or upper
            boundary, then `None` should be passed as the other boundary value.

        For example, `reflect=[None, [-1.0, 1.0], [0.0, None]]`, specifies that the 0th dimension
        has no boundaries, the 1st dimension has boundaries at both -1.0 and 1.0, and the 2nd
        dimension has a lower boundary at 0.0, and no upper boundary.

    *Projection / Marginalization*

        The PDF can be calculated for only particular parameters/dimensions.
        The `pdf` method accepts the keyword-argument (kwarg) `params` to specify particular
        parameters over which to calculate the PDF (i.e. the other parameters are projected over).

        params : int, array_like of int, None (default)
            Only calculate the PDF for certain parameters (dimensions).

            If `None`, then calculate PDF along all dimensions.
            If `params` is specified, then the target evaluation points `pnts`, must only
            contain the corresponding dimensions.

        For example, if the `dataset` has shape (4, 100), but `pdf` is called with `params=(1, 2)`,
        then the `pnts` array should have shape `(2, M)` where the two provides dimensions
        correspond to the 1st and 2nd variables of the `dataset`.

    TO-DO: add notes on `keep` parameter

    *Dynamic Range*

        When the elements of the covariace matrix between data variables differs by numerous
        orders of magnitude, the KDE values (especially marginalized values) can become spurious.
        One solution is to use a diagonal covariance matrix by initializing the KDE instance with
        `diagonal=True`.  An alternative is to transform the input data in such a way that each
        variable's dynamic range becomes similar (e.g. taking the log of the values).  A warning
        is given if the covariance matrix has a large dynamic very-large dynamic range, but no
        error is raised.

    Examples
    --------
    Construct semi-random data:

    >>> import numpy as np
    >>> np.random.seed(1234)
    >>> data = np.random.normal(0.0, 1.0, 1000)

    Construct `KDE` instance using this data, and the default bandwidth and kernels.

    >>> import kalepy as kale
    >>> kde = kale.KDE(data)

    Compare original PDF and the data to the reconstructed PDF from the KDE:

    >>> xx = np.linspace(-3, 3, 400)
    >>> pdf_tru = np.exp(-xx*xx/2) / np.sqrt(2*np.pi)
    >>> xx, pdf_kde = kde.density(xx, probability=True)

    >>> import matplotlib.pyplot as plt
    >>> ll = plt.plot(xx, pdf_tru, 'k--', label='Normal PDF')
    >>> _, bins, _ = plt.hist(data, bins=14, density=True, \
                              color='0.5', rwidth=0.9, alpha=0.5, label='Data')
    >>> ll = plt.plot(xx, pdf_kde, 'r-', label='KDE')
    >>> ll = plt.legend()

    Compare the KDE reconstructed PDF to the "true" PDF, make sure the chi-squared is consistent:

    >>> dof = xx.size - 1
    >>> x2 = np.sum(np.square(pdf_kde - pdf_tru)/pdf_tru**2)
    >>> x2 = x2 / dof
    >>> x2 < 0.1
    True
    >>> print("Chi-Squared: {:.1e}".format(x2))
    Chi-Squared: 1.7e-02

    Draw new samples from the data and make sure they are consistent with the original data:

    >>> import scipy as sp
    >>> samp = kde.resample()
    >>> ll = plt.hist(samp, bins=bins, density=True, color='r', alpha=0.5, rwidth=0.5, \
                      label='Samples')
    >>> ks, pv = sp.stats.ks_2samp(data, samp)
    >>> pv > 0.05
    True

    """

    _EDGE_REFINEMENT = np.sqrt(10.0)

    def __init__(self, dataset, bandwidth=None, weights=None, kernel=None,
                 extrema=None, points=None, reflect=None,
                 neff=None, diagonal=False, helper=True, bw_rescale=None, **kwargs):
        """Initialize the `KDE` class with the given dataset and optional specifications.

        Arguments
        ---------
        dataset : array_like (N,) or (D,N,)
            Dataset from which to construct the kernel-density-estimate.
            For multivariate data with `D` variables and `N` values, the data must be shaped (D,N).
            For univariate (D=1) data, this can be a single array with shape (N,).

        bandwidth : str, float, array of float, None  [optional]
            Specification for the bandwidth, or the method by which the bandwidth should be
            determined.  If a `str` is given, it must match one of the standard bandwidth
            determination methods.  If a `float` is given, it is used as the bandwidth in each
            dimension.  If an array of `float`s are given, then each value will be used as the
            bandwidth for the corresponding data dimension.

        weights : array_like (N,), None  [optional]
            Weights corresponding to each `dataset` point.  Must match the number of points `N` in
            the `dataset`.
            If `None`, weights are uniformly set to 1.0 for each value.

        kernel : str, Distribution, None  [optional]
            The distribution function that should be used for the kernel.  This can be a `str`
            specification that must match one of the existing distribution functions, or this can
            be a `Distribution` subclass itself that overrides the `_evaluate` method.

        neff : int, None  [optional]
            An effective number of datapoints.  This is used in the plugin bandwidth determination
            methods.
            If `None`, `neff` is calculated from the `weights` array.  If `weights` are all
            uniform, then `neff` equals the number of datapoints `N`.

        diagonal : bool,
            Whether the bandwidth/covariance matrix should be set as a diagonal matrix
            (i.e. without covariances between parameters).
            NOTE: see `KDE` docstrings, "Dynamic Range".

        """
        self._helper = helper
        self._squeeze = (np.ndim(dataset) == 1)
        self._dataset = np.atleast_2d(dataset)
        ndim, ndata = self.dataset.shape
        self._ndim = ndim
        self._ndata = ndata
        self._diagonal = diagonal
        self._reflect = reflect
        # The first time `points` are used, they need to be 'checked' for consistency
        self._check_points_flag = True
        self._points = points
        if ndata < 3:
            print(self._dataset)
            err = "ERROR: too few data points!  Dataset shape: ({}, {})".format(ndim, ndata)
            raise ValueError(err)

        # Set `weights`
        # --------------------------------
        weights_uniform = True
        if weights is not None:
            if np.shape(weights) != (ndata,):
                raise ValueError("`weights` input should be shaped as (N,)!")

            if np.count_nonzero(weights) == 0 or np.any(~np.isfinite(weights) | (weights < 0)):
                raise ValueError("Invalid `weights` entries, all must be finite and > 0!")

            weights = np.asarray(weights).astype(float)
            weights_uniform = False

        if neff is None:
            if weights_uniform:
                neff = ndata
            else:
                neff = np.sum(weights)**2 / np.sum(weights**2)

        self._weights = weights
        self._weights_uniform = weights_uniform    # currently unused
        self._neff = neff

        # Set covariance, bandwidth, distribution and kernel
        # -----------------------------------------------------------
        covariance = np.cov(dataset, rowvar=True, bias=False, aweights=weights)
        self._covariance = np.atleast_2d(covariance)

        if bandwidth is None:
            bandwidth = _BANDWIDTH_DEFAULT

        self._set_bandwidth(bandwidth, bw_rescale)

        # Convert from string, class, etc to a kernel
        dist = kernels.get_distribution_class(kernel)
        self._kernel = kernels.Kernel(
            distribution=dist, bandwidth=self._bandwidth, covariance=self._covariance,
            helper=helper, **kwargs)

        # Get Distribution Extrema
        # ------------------------------------
        # Determine the effective minima / maxima that should be used; KDE generally has support
        #   outside of the data values themselves.

        # If the Kernel is finite, then there is only support out to `bandwidth` beyond datapoints
        if self.kernel.FINITE:
            out = (1.0 + _NUM_PAD)
        # If infinite kernel, how many standard-deviations can we expect values to lie at
        else:
            out = sp.stats.norm.ppf(1.0 - 1.0/neff)
            # Extra to be double sure...
            out *= 1.2

        reflect = kernels._check_reflect(reflect, self.dataset)

        # Find the effective-extrema in each dimension, to be used if `extrema` is not specified
        _bandwidth = np.sqrt(self.kernel.matrix.diagonal())
        eff_extrema = [
            [np.min(dd) - bw*out, np.max(dd) + bw*out]
            for bw, dd in zip(_bandwidth, self.dataset)
        ]

        if (extrema is None) and (reflect is not None):
            extrema = copy.deepcopy(reflect)

        # `eff_extrema` is, by design, outside of data limits, so don't `warn` about limits
        extrema = utils._parse_extrema(eff_extrema, extrema, warn=False)
        self._extrema = extrema

        # Finish Intialization
        # -------------------------------
        self._cdf_grid = None
        self._cdf_func = None

        self._finalize()
        return

    def density(self, points=None, reflect=None, params=None, grid=False, probability=False):
        """Evaluate the KDE distribution at the given data-points.

        This method acts as an API to the `Kernel.pdf` method for this instance's `kernel`.


        Arguments
        ---------
        points : ([D,]M,) array_like of float, or (D,) set of array_like point specifications
            The locations at which the PDF should be evaluated.  The number of dimensions `D` must
            match that of the `dataset` that initialized this class' instance.
            NOTE: If the `params` kwarg (see below) is given, then only those dimensions of the
            target parameters should be specified in `points`.

            The meaning of `points` depends on the value of the `grid` argument:

            * `grid=True`  : `points` must be a set of (D,) array_like objects which each give the
              evaluation points for the corresponding dimension to produce a grid of values.
              For example, for a 2D dataset,
              `points=([0.1, 0.2, 0.3], [1, 2])`,
              would produce a grid of points with shape (3, 2):
              `[[0.1, 1], [0.1, 2]], [[0.2, 1], [0.2, 2]], [[0.3, 1], [0.3, 2]]`,
              and the returned values would be an array of the same shape (3, 2).

            * `grid=False` : `points` must be an array_like (D,M) describing the position of `M`
              sample points in each of `D` dimensions.
              For example, for a 3D dataset:
              `points=([0.1, 0.2], [1.0, 2.0], [10, 20])`,
              describes 2 sample points at the 3D locations, `(0.1, 1.0, 10)` and `(0.2, 2.0, 20)`,
              and the returned values would be an array of shape (2,).

        reflect : (D,) array_like, None (default)
            Locations at which reflecting boundary conditions should be imposed.
            For each dimension `D`, a pair of boundary locations (for: lower, upper) must be
            specified, or `None`.  `None` can also be given to specify no boundary at that
            location.  See class docstrings:`Reflection` for more information.

        params : int, array_like of int, None (default)
            Only calculate the PDF for certain parameters (dimensions).
            See class docstrings:`Projection` for more information.

        grid : bool,
            Evaluate the KDE distribution at a grid of points specified by `points`.
            See `points` argument description above.

        probability : bool, normalize the results to sum to unity


        Returns
        -------
        points : array_like of scalar
            Locations at which the PDF is evaluated.
        vals : array_like of scalar
            PDF evaluated at the given points

        """
        ndim = self.ndim
        data = self.dataset
        if reflect is None:
            reflect = self._reflect

        squeeze = False
        if params is not None:
            if (ndim == 1):
                if params == 0:
                    params = None
                else:
                    err = "Cannot specify `params` ('{}') > 0 for 1D data!".format(params)
                    raise ValueError(err)

            if params is not None:
                squeeze = np.isscalar(params)
                params = np.atleast_1d(params)
                if reflect is not None:
                    if len(reflect) == ndim:
                        reflect = [reflect[pp] for pp in params]
                    elif len(reflect) == 2 and len(params) == 1:
                        pass
                    elif len(reflect) != len(params):
                        err = (
                            "length of `reflect` ({}) ".format(len(reflect)),
                            "does not match `params` ({})!".format(len(params))
                        )
                        raise ValueError(err)

                data = data[params, :]

        if points is None:
            points = self.points
            if params is not None:
                points = [points[pp] for pp in params]
                grid = (len(points) > 1)
            else:
                grid = (self.ndim > 1)
        elif utils.really1d(points):
            points = np.atleast_2d(points)
            squeeze = True

        if grid:
            _points = points
            points = utils.meshgrid(*points)
            shape = np.shape(points[0])
            points = [pp.flatten() for pp in points]

        values = self.kernel.density(points, data, self.weights, reflect=reflect, params=params)

        if probability:
            if self.weights is None:
                values = values / self.ndata
            else:
                values = values / np.sum(self.weights)

        if grid:
            values = values.reshape(shape)
            points = _points

        if squeeze:
            points = points[0]
            values = values.squeeze()

        return points, values

    def pdf(self, *args, **kwargs):
        kwargs['probability'] = True
        return self.density(*args, **kwargs)

    def cdf(self, pnts, params=None, reflect=None):
        """Cumulative Distribution Function based on KDE smoothed data.

        Arguments
        ---------
        pnts : ([D,]N,) array_like of scalar
            Target evaluation points

        Returns
        -------
        cdf : (N,) ndarray of scalar
            CDF Values at the target points

        """
        if params is not None:
            raise NotImplementedError("`params` is not yet implemented for CDF!")

        if reflect is not None:
            raise NotImplementedError("`reflect` is not yet implemented for CDF!")

        if self._cdf_func is None:
            points = self.points

            # Calculate PDF at grid locations
            pdf = self.pdf(points, grid=True)[1]
            # Convert to CDF using trapezoid rule
            cdf = utils.cumtrapz(pdf, points)
            # Normalize to the maximum value
            cdf /= cdf.max()

            ndim = np.ndim(cdf)
            # points = np.atleast_2d(points)
            if ndim == 1 and np.ndim(points) == 1:
                points = np.atleast_2d(points)

            self._cdf_grid = (points, cdf)
            self._cdf_func = sp.interpolate.RegularGridInterpolator(
                *self._cdf_grid, bounds_error=False, fill_value=None)

        # `scipy.interplate.RegularGridInterpolator` expects shape (N,D,) -- so transpose
        pnts = np.asarray(pnts).T
        cdf = self._cdf_func(pnts)
        return cdf

    def cdf_grid(self, edges, **kwargs):
        """

        NOTE: optimize: there are likely much faster methods than broadcasting and flattening,
                        use a different method to calculate cdf on a grid.
        """
        ndim = self.ndim
        if len(edges) != ndim:
            err = "`edges` must be (D,)=({},): an arraylike of edges for each dim/param!"
            err = err.format(ndim)
            raise ValueError(err)

        coords = np.meshgrid(*edges, indexing='ij')
        shp = np.shape(coords)[1:]
        coords = np.vstack([xx.ravel() for xx in coords])
        cdf = self.cdf(coords, **kwargs)
        cdf = cdf.reshape(shp)
        return cdf

    def resample(self, size=None, keep=None, reflect=None, squeeze=True):
        """Draw new values from the kernel-density-estimate calculated PDF.

        The KDE calculates a PDF from the given dataset.  This method draws new, semi-random data
        points from that PDF.

        Arguments
        ---------
        size : int, None (default)
            The number of new data points to draw.  If `None`, then the number of `datapoints` is
            used.

        keep : int, array_like of int, None (default)
            Parameters/dimensions where the original data-values should be drawn from, instead of
            from the reconstructed PDF.
            TODO: add more information.

        reflect : (D,) array_like, None (default)
            Locations at which reflecting boundary conditions should be imposed.
            For each dimension `D`, a pair of boundary locations (for: lower, upper) must be
            specified, or `None`.  `None` can also be given to specify no boundary at that
            location.

        squeeze : bool, (default: True)
            If the number of dimensions `D` is one, then return an array of shape (L,) instead of
            (1, L).

        Returns
        -------
        samples : ([D,]L) ndarray of float
            Newly drawn samples from the PDF, where the number of points `L` is determined by the
            `size` argument.
            If `squeeze` is True (default), and the number of dimensions in the original dataset
            `D` is one, then the returned array will have shape (L,).

        """
        if reflect is False:
            reflect = None
        elif reflect is None:
            reflect = self._reflect

        samples = self.kernel.resample(
            self.dataset, self.weights,
            size=size, keep=keep, reflect=reflect, squeeze=squeeze)
        return samples

    # def ppf(self, cfrac):
    #     return self.kernel.ppf(cfrac)

    def _finalize(self, log_ratio_tol=5.0):
        WARN_ALL_BELOW = -10
        EXTR_ABOVE = -20

        mat = self.kernel.matrix
        ndim = self._ndim
        # Diagonal elements of the matrix
        diag = mat.diagonal()

        # Off-diagonal matrix elements
        ui = np.triu_indices(ndim, 1)
        li = np.tril_indices(ndim, -1)
        offd = np.append(mat[ui], mat[li])

        warn = False

        # if np.any(np.isclose(diag, 0.0)):
        #     err = "Diagonal matrix elements zero!  {}".format(diag)
        #     logging.warning(err)

        # if np.any(offd != 0.0):
        if not np.all(np.isclose(offd, 0.0)):
            d_vals = np.log10(np.fabs(diag[diag != 0.0]))
            o_vals = np.log10(np.fabs(offd[offd != 0.0]))
            if np.all(d_vals <= WARN_ALL_BELOW) and np.all(o_vals <= WARN_ALL_BELOW):
                logging.warning("Covariance matrix:\n" + str(mat))
                logging.warning("All matrix elements are less than 1e{}!".format(WARN_ALL_BELOW))
                warn = True

            d_vals = d_vals[d_vals > EXTR_ABOVE]
            o_vals = o_vals[o_vals > EXTR_ABOVE]
            if len(d_vals) > 0 and len(o_vals) > 0:
                d_extr = utils.minmax(d_vals)
                o_extr = utils.minmax(o_vals)
                ratio = np.fabs(d_extr[:, np.newaxis] - o_extr[np.newaxis, :])
                if np.any(ratio >= log_ratio_tol):
                    logging.warning("Covariance matrix:\n" + str(mat))
                    msg = "(log) Ratio of covariance elements ({}) exceeds tolerance ({})!".format(
                        ratio, log_ratio_tol)
                    logging.warning(msg)
                    warn = True

        if warn:
            msg = "Recommend rescaling input data, or using a diagonal covariance matrix"
            logging.warning(msg)

        return

    # ==== Properties ====

    @property
    def bandwidth(self):
        return self._bandwidth

    @property
    def covariance(self):
        return self._covariance

    @property
    def dataset(self):
        return self._dataset

    @property
    def points(self):
        # The values of `self._points` set during initialization can be general specifications
        #   for bin-edges instead of the bin-edges themselves.  So they need to be "checked" the
        #   first time
        if (self._points is not None) and (not self._check_points_flag):
            return self._points

        # `extrema` is already set to include preset `reflect` values and takes into account
        #   whether the kernel is finite or not (see `KDE.__init__`)
        extrema = self.extrema
        points = utils.parse_edges(
            self.dataset, edges=self._points, extrema=extrema, weights=self.weights,
            nmin=3, nmax=200, pad=0, refine=self._EDGE_REFINEMENT)

        # If input `data` to KDE is given as 1D array, then give 1D points (instead of `(1, E)`)
        if self._squeeze:
            points = np.squeeze(points)

        self._points = points
        self._check_points_flag = False
        return self._points

    @property
    def extrema(self):
        return self._extrema

    @property
    def kernel(self):
        return self._kernel

    @property
    def ndata(self):
        return self._ndata

    @property
    def ndim(self):
        return self._ndim

    @property
    def neff(self):
        return self._neff

    @property
    def weights(self):
        return self._weights

    # ==== BANDWIDTH ====

    def _set_bandwidth(self, bw_input, bw_rescale):
        ndim = self.ndim
        bandwidth = np.zeros((ndim, ndim))

        if len(np.atleast_1d(bw_input)) == 1:
            _bw, method = self._compute_bandwidth(bw_input)
            if not self._diagonal:
                bandwidth[...] = _bw
            else:
                idx = np.arange(ndim)
                bandwidth[idx, idx] = _bw
        else:
            if np.shape(bw_input) == (ndim,):
                # bw_method = 'diagonal'
                for ii in range(ndim):
                    bandwidth[ii, ii], *_ = self._compute_bandwidth(
                        bw_input[ii], param=(ii, ii))
                method = 'diagonal'
            elif np.shape(bw_input) == (ndim, ndim):
                for ii, jj in np.ndindex(ndim, ndim):
                    bandwidth[ii, jj], *_ = self._compute_bandwidth(
                        bw_input[ii, jj], param=(ii, jj))
                method = 'matrix'
            else:
                err = "`bandwidth` must have shape (1,), (N,) or (N,N,) for `N` dimensions!"
                raise ValueError(err)

        if self._helper and np.any(np.isclose(bandwidth.diagonal(), 0.0)):
            ii = np.where(np.isclose(bandwidth.diagonal(), 0.0))[0]
            msg = "WARNING: diagonal '{}' of bandwidth is near zero!".format(ii)
            logging.warning(msg)

        # Rescale the bandwidth matrix
        if bw_rescale is not None:
            bwr = np.atleast_2d(bw_rescale)
            bandwidth = bwr * bandwidth
            if self._helper:
                logging.info("Rescaling `bw_white_matrix` by '{}'".format(
                    bwr.squeeze().flatten()))

        # bw_matrix = self._data_cov * (bw_white_matrix ** 2)
        self._bandwidth = bandwidth

        # prev: bw_white
        # self._bw_white_matrix = bw_white_matrix
        self._method = method
        # self._bw_input = bw_input
        # prev: bw_cov
        # self._bw_matrix = bw_matrix
        return

    def _compute_bandwidth(self, bandwidth, param=None):
        if isinstance(bandwidth, six.string_types):
            if bandwidth == 'scott':
                bw = self._scott_factor(param=param)
            elif bandwidth == 'silverman':
                bw = self._silverman_factor(param=param)
            else:
                msg = "Unrecognized bandwidth str specification '{}'!".format(bandwidth)
                raise ValueError(msg)

            method = bandwidth

        elif np.isscalar(bandwidth):
            bw = bandwidth
            method = 'constant scalar'

        else:
            raise ValueError("Unrecognized `bandwidth` '{}'!".format(bandwidth))

        '''
        elif callable(bandwidth):
            bw = bandwidth(self, param=param)
            method = 'function'
            bw_cov = 1.0
        '''

        return bw, method

    def _scott_factor(self, *args, **kwargs):
        return np.power(self.neff, -1./(self.ndim+4))

    def _silverman_factor(self, *args, **kwargs):
        return np.power(self.neff*(self.ndim+2.0)/4.0, -1./(self.ndim+4))
