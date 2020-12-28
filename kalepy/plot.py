"""kalepy's plotting submodule

This submodule containts the `Corner` class, and all plotting methods.  The `Corner` class, and
additional API functions are imported into the base package namespace of `kalepy`, e.g.
`kalepy.Corner` and `kalepy.carpet` access the `kalepy.plot.Corner` and `kalepy.plot.carpet` methods
respectively.

Additional options and customization:

The core plotting routines, such as `draw_hist1d`, `draw_hist2d`, `draw_contour2d`, etc include a
fairly large number of keyword arguments for customization.  The top level API methods, such as
`corner()` or `Corner.plot_data()` often do not provide access to all of those arguments, but
additional customization is possible by using the drawing methods directly, and optionally
subclassing the `Corner` class to provide additional or different functionality.

API Contents
------------
- Corner : class for corner/triangle/pair plots.
- corner : method which constructs a `Corner` instance and plots 1D and 2D distributions.

- dist1d : plot a 1D distribution with numerous possible elements (e.g. histogram, carpet, etc)
- dist2d : plot a 2D distribution with numerous possible elements (e.g. histogram, contours, etc)

- carpet : draw a 1D scatter-like plot to depict semi-quantitative information about a distribution.
- contour : draw a 2D contour plot. A wrapper of additional functionality around `plt.contour`
- confidence : draw 1D confidence intervals using shaded bands.
- hist1d : draw a 1D histogram
- hist2d : draw a 2D histogram.  A wrapper of additional functionality around `plt.pcolormesh`

"""

import logging

import numpy as np
import matplotlib as mpl
import matplotlib.patheffects  # noqa
import matplotlib.pyplot as plt

import kalepy as kale
from kalepy import utils
# from kalepy import KDE

_DEF_SIGMAS = [0.5, 1.0, 1.5, 2.0]
_PAD = 1

# Define functions for module level import
__all__ = [
    "carpet", "contour", "confidence", "Corner", "corner", "dist1d", "dist2d", "hist1d", "hist2d",
]


class Corner:
    """Class for creating 'corner' / 'pair' plots of multidimensional covariances.

    The `Corner` class acts as a constructor for a `matplotlib` figure and axes, and coordinates the
    plotting of 1D and 2D distributions.  The `kalepy.plot.dist1d()` and `kalepy.plot.dist2d()`
    methods are used for plotting the distributions.  The class methods provide wrappers, and
    default setting for those methods.  The `Corner.plot` method is the standard plotting method
    with default parameters chosen for plotting a single, multidimensional dataset.  For
    overplotting numerous datasets, the `Corner.clean` or `Corner.plot_data` methods are better.

    API Methods
    -----------
    - `plot` : the standard plotting method which, by default, includes both KDE and data elements.
    - `clean` : minimal plots with only the KDE generated PDF in 1D and contours in 2D, by default.
    - `hist` : minimal plots with only the data based 1D and 2D histograms, by default.
    - `plot_kde` : plot elements with only KDE based info: the `clean` settings with a little more.
    - `plot_data` : plot elements without using KDE info.

    Examples
    --------
    Load some predefined 3D data, and generate a default corner plot:

    >>> import kalepy as kale
    >>> data = kale.utils._random_data_3d_03()
    >>> corner = kale.corner(data)

    Load two different datasets, and overplot them using a `kalepy.Corner` instance.

    >>> data1 = kale.utils._random_data_3d_03(par=[0.0, 0.5], cov=0.05)
    >>> data2 = kale.utils._random_data_3d_03(par=[1.0, 0.25], cov=0.5)
    >>> corner = kale.Corner(3)   # construct '3' dimensional corner-plot (i.e. 3x3 axes)
    >>> corner.clean(data1)
    >>> corner.clean(data2)

    """

    # How much the axes-limits should be extended beyond the range of plotted data/distributions
    #    the units are fractions of the data-range, i.e. '0.1' would mean 10% beyond data range.
    _LIMITS_STRETCH = 0.1

    def __init__(self, kde_data, weights=None, labels=None, limits=None, rotate=True, **kwfig):
        """Initialize Corner instance and construct figure and axes based on the given arguments.

        Arguments
        ---------
        kde_data : object, one of the following
            * int D, the number of parameters/dimensions to construct a DxD corner plot.
            * instance of `kalepy.kde.KDE`, providing the data and KDE to be plotted.
            * array_like scalar (D,N) of data with `D` parameters and `N` data points.

        weights : array_like scalar (N,) or None
            The weights for each data point.
            NOTE: only applicable when `kde_data` is a (D,N) dataset.

        labels : array_like string (N,) of names for each parameters.

        limits : None, or (D,2) of scalar
            Specification for the limits of each axes (for each of `D` parameters):
            * None : the limits are determined automatically,
            * (D,2) : limits for each axis

        rotate : bool,
            Whether or not the bottom-right-most axes should be rotated.

        **kwfig : keyword-arguments passed to `_figax()` for constructing figure and axes.
            See `kalepy.plot._figax()` for specifications.

        """

        # --- Parse the given `kde_data` and store parameters accordingly --
        if np.isscalar(kde_data):
            if not isinstance(kde_data, int):
                err = ("If `kde_data` is a scalar, it must be an integer "
                       "specifying the number of parameters!")
                raise ValueError(err)
            size = kde_data
            kde = None
            data = None
        else:
            kde, data, weights = _parse_kde_data(kde_data, weights=weights)
            size = kde._ndim

        # -- Construct figure and axes
        fig, axes = _figax(size, **kwfig)
        self.fig = fig
        self.axes = axes

        last = size - 1
        if labels is None:
            labels = [''] * size

        for (ii, jj), ax in np.ndenumerate(axes):
            # Set upper-right plots to invisible
            if jj > ii:
                ax.set_visible(False)
                continue
            ax.grid(True)

            # Bottom row
            if ii == last:
                ax.set_xlabel(labels[jj])
            # Non-bottom row
            else:
                ax.set_xlabel('')
                for tlab in ax.xaxis.get_ticklabels():
                    tlab.set_visible(False)

            # First column
            if jj == 0:
                # Not-first rows
                if ii != 0:
                    ax.set_ylabel(labels[ii])
            # Not-first columns
            else:
                ax.set_ylabel('')
                for tlab in ax.yaxis.get_ticklabels():
                    tlab.set_visible(False)

            # Diagonals
            if ii == jj:
                # not top-left
                if ii != 0:
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.set_ticks_position('right')

            # Off-Diagonals
            else:
                pass

        # If axes limits are given, set axes to them
        if limits is not None:
            limit_flag = False
            _set_corner_axes_extrema(self.axes, limits, rotate)
        # Otherwise, prepare to calculate limits during plotting
        else:
            limits = [None] * size
            limit_flag = True

        # --- Store key parameters
        self.size = size
        self._kde = kde
        self._data = data
        self._weights = weights
        self._limits = limits
        self._limit_flag = limit_flag
        self._rotate = rotate

        return

    def clean(self, kde_data=None, weights=None, dist1d={}, dist2d={}, **kwargs):
        """Wrapper for `plot_kde` that sets parameters for minimalism: PDF and contours only.

        Arguments
        ---------
        kde_data : `kalepy.KDE` instance, (D,N) array_like of scalars, or None
            * instance of `kalepy.kde.KDE`, providing the data and KDE to be plotted.
            * array_like scalar (D,N) of data with `D` parameters and `N` data points.
            * `None` : use the KDE/data stored during class initialization.
                       raises `ValueError` if no KDE/data was provided

        weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
            only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

        dist1d : dict of keyword-arguments passed to the `kale.plot.dist1d` method.

        dist2d : dict of keyword-arguments passed to the `kale.plot.dist2d` method.

        **kwargs : additiona keyword-arguments passed directly to `Corner.plot_kde`.

        """
        if kde_data is None:
            # If either data or a KDE was given on initialization, `self._kde` will be set
            kde_data = self._kde
            if kde_data is None:
                err = "kde or data required either during initialization or here!"
                raise ValueError(err)

        # If data is given, convert to KDE as needed
        kde, data, weights = _parse_kde_data(kde_data, weights=weights)

        # Set default 1D parameters
        dist1d.setdefault('density', True)
        dist1d.setdefault('confidence', False)
        dist1d.setdefault('carpet', False)
        dist1d.setdefault('hist', False)

        # Set default 2D parameters
        dist2d.setdefault('hist', False)
        dist2d.setdefault('contour', True)
        dist2d.setdefault('scatter', False)
        dist2d.setdefault('mask_dense', False)
        dist2d.setdefault('mask_below', False)

        # Plot
        rv = self.plot_kde(kde, dist1d=dist1d, dist2d=dist2d, **kwargs)
        return rv

    def hist(self, kde_data=None, weights=None, dist1d={}, dist2d={}, **kwargs):
        """Wrapper for `plot_data` that sets parameters to only plot 1D and 2D histograms of data.

        Arguments
        ---------
        kde_data : `kalepy.KDE` instance, (D,N) array_like of scalars, or None
            * instance of `kalepy.kde.KDE`, providing the data and KDE to be plotted.
            * array_like scalar (D,N) of data with `D` parameters and `N` data points.
            * `None` : use the KDE/data stored during class initialization.
                       raises `ValueError` if no KDE/data was provided

        weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
            only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

        dist1d : dict of keyword-arguments passed to the `kale.plot.dist1d` method.

        dist2d : dict of keyword-arguments passed to the `kale.plot.dist2d` method.

        **kwargs : additiona keyword-arguments passed directly to `Corner.plot_kde`.

        """
        if kde_data is None:
            # If either data or a KDE was given on initialization, `self._kde` will be set
            kde_data = self._kde
            if kde_data is None:
                err = "kde or data required either during initialization or here!"
                raise ValueError(err)

        # If KDE is given, retrieve the dataset from it
        kde, data, weights = _parse_kde_data(kde_data, weights=weights)

        # Set default 1D parameters
        dist1d.setdefault('density', False)
        dist1d.setdefault('confidence', False)
        dist1d.setdefault('carpet', False)
        dist1d.setdefault('hist', True)

        # Set default 2D parameters
        dist2d.setdefault('hist', True)
        dist2d.setdefault('contour', False)
        dist2d.setdefault('scatter', False)
        dist2d.setdefault('mask_dense', False)
        dist2d.setdefault('mask_below', False)

        # Plot
        rv = self.plot_data(data, dist1d=dist1d, dist2d=dist2d, **kwargs)
        return rv

    def plot(self, kde_data=None, edges=None, weights=None, quantiles=None,
             limit=None, color=None, cmap=None, dist1d={}, dist2d={}):
        """Plot with standard settings for plotting a single, multidimensional dataset or KDE.

        This function coordinates the drawing of a corner plot that ultimately uses the
        `kalepy.plot.dist1d` and `kalepy.plot.dist2d` methods to draw parameter distributions
        using an instance of `kalepy.kde.KDE`.


        Arguments
        ---------
        kde_data : `kalepy.KDE` instance, (D,N) array_like of scalars, or `None`
            * instance of `kalepy.kde.KDE`, providing the data and KDE to be plotted.
            * array_like scalar (D,N) of data with `D` parameters and `N` data points.
            * `None` : use the KDE/data stored during class initialization.
                       raises `ValueError` if no KDE/data was provided

        edges : object specifying historgam edge locations; or `None`
            * int : the number of bins for all dimensions, locations calculated automatically
            * (D,) array_like of int : the number of bins for each of `D` dimensions
            * (D,) of array_like : the bin-edge locations for each of `D` dimensions, e.g.
                                   ([0, 1, 2], [0.0, 0.1, 0.2, 0.3],) would describe two bins for
                                   the 0th dimension, and 3 bins for the 1st dimension.
            * (X,) array_like of scalar : the bin-edge locations to be used for all dimensions
            * `None` : the number and locations of bins are calculated automatically for each dim

        weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
            only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

        quantiles : `None` or array_like of scalar values in [0.0, 1.0] denoting the fractions of
            data to demarkate with contours and confidence bands.

        limit : bool or `None`, whether the axes limits should be reset based on the plotted data.
            If `None`, then the limits will be readjusted unless `limits` were provided on class
            initialization.

        color : matplotlib color specification (i.e. named color, hex or rgb) or `None`.
            If `None`:
                * `cmap` is given, then the color will be set to the `cmap` midpoint.
                * `cmap` is not given, then the color will be determined by the next value of the
                    default matplotlib color-cycle, and `cmap` will be set to a matching colormap.
            This parameter effects the color of 1D: histograms, confidence intervals, and carpet;
            2D: scatter points.

        cmap : matplotlib colormap specification, or `None`
            * All valid matplotlib specifications can be used, e.g. named value (like 'Reds' or
            'viridis') or a `matplotlib.colors.Colormap` instance.
            * If `None` then a colormap is constructed based on the value of `color` (see above).

        dist1d : dict of keyword-arguments passed to the `kale.plot.dist1d` method.

        dist2d : dict of keyword-arguments passed to the `kale.plot.dist2d` method.

        """

        if kde_data is None:
            # If either data or a KDE was given on initialization, `self._kde` will be set
            kde_data = self._kde
            if kde_data is None:
                err = "kde or data required either during initialization or here!"
                raise ValueError(err)

        # If data is given, construct KDE
        kde, data, weights = _parse_kde_data(kde_data, weights=weights)

        # Set default 1D parameters
        dist1d.setdefault('density', True)
        dist1d.setdefault('confidence', True)
        dist1d.setdefault('carpet', True)
        dist1d.setdefault('hist', False)

        # Set default 2D parameters
        dist2d.setdefault('hist', True)
        dist2d.setdefault('scatter', True)
        dist2d.setdefault('contour', True)
        dist2d.setdefault('mask_dense', True)
        dist2d.setdefault('mask_below', True)

        # Plot
        rv = self.plot_kde(
            kde, edges=edges, quantiles=quantiles, limit=limit, color=color, cmap=cmap,
            dist1d=dist1d, dist2d=dist2d
        )
        return rv

    def plot_kde(self, kde=None, edges=None, weights=None, quantiles=None, limit=None,
                 color=None, cmap=None, dist1d={}, dist2d={}):
        """Plot with default settings to emphasize the KDE derived distributions.

        This function coordinates the drawing of a corner plot that ultimately uses the
        `kalepy.plot.dist1d` and `kalepy.plot.dist2d` methods to draw parameter distributions
        using an instance of `kalepy.kde.KDE`.


        Arguments
        ---------
        kde : `kalepy.KDE` instance, (D,N) array_like of scalars, or None
            * instance of `kalepy.kde.KDE`, providing the data and KDE to be plotted.
            * array_like scalar (D,N) of data with `D` parameters and `N` data points.
            * `None` : use the KDE/data stored during class initialization.
                       raises `ValueError` if no KDE/data was provided

        edges : object specifying historgam edge locations; or None
            * int : the number of bins for all dimensions, locations calculated automatically
            * (D,) array_like of int : the number of bins for each of `D` dimensions
            * (D,) of array_like : the bin-edge locations for each of `D` dimensions, e.g.
                                   ([0, 1, 2], [0.0, 0.1, 0.2, 0.3],) would describe two bins for
                                   the 0th dimension, and 3 bins for the 1st dimension.
            * (X,) array_like of scalar : the bin-edge locations to be used for all dimensions
            * `None` : the number and locations of bins are calculated automatically for each dim

        weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
            only-if the given `kde` argument is a (D,N) array_like of scalar data from which a
            `KDE` instance is created.

        quantiles : `None` or array_like of scalar values in [0.0, 1.0] denoting the fractions of
            data to demarkate with contours and confidence bands.

        limit : bool or `None`, whether the axes limits should be reset based on the plotted data.
            If `None`, then the limits will be readjusted unless `limits` were provided on class
            initialization.

        color : matplotlib color specification (i.e. named color, hex or rgb) or `None`.
            If `None`:
                * `cmap` is given, then the color will be set to the `cmap` midpoint.
                * `cmap` is not given, then the color will be determined by the next value of the
                    default matplotlib color-cycle, and `cmap` will be set to a matching colormap.
            This parameter effects the color of 1D: histograms, confidence intervals, and carpet;
            2D: scatter points.

        cmap : matplotlib colormap specification, or `None`
            * All valid matplotlib specifications can be used, e.g. named value (like 'Reds' or
            'viridis') or a `matplotlib.colors.Colormap` instance.
            * If `None` then a colormap is constructed based on the value of `color` (see above).

        dist1d : dict of keyword-arguments passed to the `kale.plot.dist1d` method.

        dist2d : dict of keyword-arguments passed to the `kale.plot.dist2d` method.

        """
        if kde is None:
            # If either data or a KDE was given on initialization, `self._kde` will be set
            kde = self._kde
            if kde is None:
                err = "kde or data required either during initialization or here!"
                raise ValueError(err)

        # If data is given, construct KDE
        kde, data, weights = _parse_kde_data(kde, weights=weights)

        # ---- Sanitize
        axes = self.axes
        size = kde.ndim
        shp = np.shape(axes)
        if (shp[0] != shp[1]) or (shp[0] != size):
            err = "`axes` (shape: {}) does not match data dimension {}!".format(shp, size)
            raise ValueError(err)

        # ---- Set parameters
        last = size - 1
        rotate = self._rotate

        if limit is None:
            limit = self._limit_flag

        edges = utils.parse_edges(kde.dataset, edges=edges)
        quantiles, _ = _default_quantiles(quantiles=quantiles)

        # Set default color or cmap as needed
        color, cmap = _parse_color_cmap(ax=axes[0][0], color=color, cmap=cmap)

        #
        # Draw / Plot KDE
        # ----------------------------------

        # ---- Draw 1D
        limits = [None] * size    # variable to store the limits of the plotted data
        for jj, ax in enumerate(axes.diagonal()):
            rot = (rotate and (jj == last))
            self._kde1d(
                ax, edges[jj], kde, param=jj, quantiles=quantiles, rotate=rot,
                color=color, **dist1d
            )
            limits[jj] = utils.minmax(edges[jj], stretch=self._LIMITS_STRETCH)

        # ---- Draw 2D
        for (ii, jj), ax in np.ndenumerate(axes):
            if jj >= ii:
                continue
            self._kde2d(
                ax, [edges[jj], edges[ii]], kde, params=[jj, ii], quantiles=quantiles,
                color=color, cmap=cmap, **dist2d
            )

        # If we are dynamically setting the axes limits
        if limit:
            # Update stored limits
            for ii in range(self.size):
                self._limits[ii] = utils.minmax(limits[ii], prev=self._limits[ii])

            # Set axes to limits
            _set_corner_axes_extrema(self.axes, self._limits, self._rotate)

        return

    def plot_data(self, data=None, edges=None, weights=None, quantiles=None, limit=None,
                  color=None, cmap=None, dist1d={}, dist2d={}):
        """Plot with default settings to emphasize the given data (not KDE derived properties).

        This function coordinates the drawing of a corner plot that ultimately uses the
        `kalepy.plot.dist1d` and `kalepy.plot.dist2d` methods to draw parameter distributions
        using an instance of `kalepy.kde.KDE`.

        Arguments
        ---------
        data : (D,N) array_like of scalars, `kalepy.KDE` instance, or None
            * array_like scalar (D,N) of data with `D` parameters and `N` data points.
            * `None` : use the KDE/data stored during class initialization.
                       raises `ValueError` if no KDE/data was provided
            * instance of `kalepy.kde.KDE`, providing the data and KDE to be plotted.
            NOTE: if a `KDE` instance is given, or one was stored during initilization, then the
                  dataset is extracted from the instance.

        edges : object specifying historgam edge locations; or None
            * int : the number of bins for all dimensions, locations calculated automatically
            * (D,) array_like of int : the number of bins for each of `D` dimensions
            * (D,) of array_like : the bin-edge locations for each of `D` dimensions, e.g.
                                   ([0, 1, 2], [0.0, 0.1, 0.2, 0.3],) would describe two bins for
                                   the 0th dimension, and 3 bins for the 1st dimension.
            * (X,) array_like of scalar : the bin-edge locations to be used for all dimensions
            * `None` : the number and locations of bins are calculated automatically for each dim

        weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
            only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

        quantiles : `None` or array_like of scalar values in [0.0, 1.0] denoting the fractions of
            data to demarkate with contours and confidence bands.

        limit : bool or `None`, whether the axes limits should be reset based on the plotted data.
            If `None`, then the limits will be readjusted unless `limits` were provided on class
            initialization.

        color : matplotlib color specification (i.e. named color, hex or rgb) or `None`.
            If `None`:
                * `cmap` is given, then the color will be set to the `cmap` midpoint.
                * `cmap` is not given, then the color will be determined by the next value of the
                    default matplotlib color-cycle, and `cmap` will be set to a matching colormap.
            This parameter effects the color of 1D: histograms, confidence intervals, and carpet;
            2D: scatter points.

        cmap : matplotlib colormap specification, or `None`
            * All valid matplotlib specifications can be used, e.g. named value (like 'Reds' or
            'viridis') or a `matplotlib.colors.Colormap` instance.
            * If `None` then a colormap is constructed based on the value of `color` (see above).

        dist1d : dict of keyword-arguments passed to the `kale.plot.dist1d` method.

        dist2d : dict of keyword-arguments passed to the `kale.plot.dist2d` method.

        """

        if data is None:
            # If either data or a KDE was given on initialization, `self._data` will be set
            data = self._data
            if data is None:
                err = "kde or data required either during initialization or here!"
                raise ValueError(err)

        # If a KDE is given, extract the dataset
        kde, data, weights = _parse_kde_data(data, weights=weights)

        # ---- Sanitize
        if np.ndim(data) != 2:
            err = "`data` (shape: {}) must be 2D with shape (parameters, data-points)!".format(
                np.shape(data))
            raise ValueError(err)

        axes = self.axes
        size = np.shape(data)[0]
        shp = np.shape(axes)
        if (np.ndim(axes) != 2) or (shp[0] != shp[1]) or (shp[0] != size):
            raise ValueError("`axes` (shape: {}) does not match data dimension {}!".format(shp, size))

        # ---- Set parameters
        last = size - 1
        rotate = self._rotate

        if limit is None:
            limit = self._limit_flag

        # Set default color or cmap as needed
        color, cmap = _parse_color_cmap(ax=axes[0][0], color=color, cmap=cmap)

        edges = utils.parse_edges(data, edges=edges)
        quantiles, _ = _default_quantiles(quantiles=quantiles)

        #
        # Draw / Plot Data
        # ----------------------------------

        # ---- Draw 1D Histograms & Carpets
        limits = [None] * size      # variable to store the data extrema
        for jj, ax in enumerate(axes.diagonal()):
            rot = (rotate and (jj == last))
            self._data1d(
                ax, edges[jj], data[jj], weights=weights, quantiles=quantiles, rotate=rot,
                color=color, **dist1d
            )
            limits[jj] = utils.minmax(data[jj], stretch=self._LIMITS_STRETCH)

        # ---- Draw 2D Histograms and Contours
        for (ii, jj), ax in np.ndenumerate(axes):
            if jj >= ii:
                continue
            self._data2d(
                ax, [edges[jj], edges[ii]], [data[jj], data[ii]], weights=weights,
                color=color, cmap=cmap, quantiles=quantiles, **dist2d
            )

        # If we are setting the axes limits dynamically
        if limit:
            # Update any stored values
            for ii in range(self.size):
                self._limits[ii] = utils.minmax(limits[ii], prev=self._limits[ii])

            # Set axes to limits
            _set_corner_axes_extrema(self.axes, self._limits, self._rotate)

        return

    def _data1d(self, ax, edge, data, color=None, **dist1d):
        """Wrapper for `kalepy.plot.dist1d` that sets default parameters appropriate for 1D data.
        """
        # Set default parameters
        dist1d.setdefault('density', False)
        dist1d.setdefault('confidence', False)
        dist1d.setdefault('carpet', True)
        dist1d.setdefault('hist', True)
        # This is identical to `kalepy.plot.dist1d` (just used for naming convenience)
        rv = _dist1d(data, ax=ax, edges=edge, color=color, **dist1d)
        return rv

    def _data2d(self, ax, edges, data, cmap=None, **dist2d):
        """Wrapper for `kalepy.plot.dist2d` that sets default parameters appropriate for 2D data.
        """
        # Set default parameters
        dist2d.setdefault('hist', True)
        dist2d.setdefault('contour', False)
        dist2d.setdefault('scatter', True)
        dist2d.setdefault('mask_dense', True)
        dist2d.setdefault('mask_below', True)
        # This is identical to `kalepy.plot.dist2d` (just used for naming convenience)
        rv = _dist2d(data, ax=ax, edges=edges, cmap=cmap, **dist2d)
        return rv

    def _kde1d(self, ax, edge, kde, param, color=None, **dist1d):
        """Wrapper for `kalepy.plot.dist1d` that sets parameters appropriate for KDE distributions.
        """
        # Set default parameters
        dist1d.setdefault('density', True)
        dist1d.setdefault('confidence', True)
        dist1d.setdefault('carpet', False)
        dist1d.setdefault('hist', False)
        # This is identical to `kalepy.plot.dist1d` (just used for naming convenience)
        rv = _dist1d(kde, ax=ax, edges=edge, color=color, param=param, **dist1d)
        return rv

    def _kde2d(self, ax, edges, kde, params, cmap=None, **dist2d):
        """Wrapper for `kalepy.plot.dist2d` that sets parameters appropriate for KDE distributions.
        """
        # Set default parameters
        dist2d.setdefault('hist', False)
        dist2d.setdefault('contour', True)
        dist2d.setdefault('scatter', False)
        dist2d.setdefault('mask_dense', True)
        dist2d.setdefault('mask_below', True)
        # This is identical to `kalepy.plot.dist2d` (just used for naming convenience)
        rv = _dist2d(kde, ax=ax, edges=edges, cmap=cmap, params=params, **dist2d)
        return rv

    '''
    def legend(self, handles=None, labels=None, index=None,
               loc=None, fancybox=False, borderaxespad=0, **kwargs):
        """
        """
        fig = self.fig

        # Set Bounding Box Location
        # ------------------------------------
        bbox = kwargs.pop('bbox', None)
        bbox = kwargs.pop('bbox_to_anchor', bbox)
        if bbox is None:
            if index is None:
                size = self.size
                if size in [2, 3]:
                    index = (0, -1)
                    loc = 'lower left'
                elif size == 1:
                    index = (0, 0)
                    loc = 'upper right'
                elif size % 2 == 0:
                    index = size // 2
                    index = (1, index)
                    loc = 'upper right'
                else:
                    index = (size // 2) + 1
                    loc = 'lower left'
                    index = (size-index-1, index)

            bbox = self.axes[index].get_position()
            bbox = (bbox.x0, bbox.y0)
            kwargs['bbox_to_anchor'] = bbox
            kwargs.setdefault('bbox_transform', fig.transFigure)

        # Set other defaults
        leg = fig.legend(handles, labels, fancybox=fancybox,
                         borderaxespad=borderaxespad, loc=loc, **kwargs)
        return leg
    '''


def corner(kde_data, labels=None, kwcorner={}, **kwplot):
    """Simple wrapper function to construct a `Corner` instance and plot the given data.

    See `kalepy.plot.Corner` and `kalepy.plot.Corner.plot` for more information.

    Arguments
    ---------
    kde_data : `kalepy.KDE` instance, or (D,N) array_like of scalars
        * instance of `kalepy.kde.KDE`, providing the data and KDE to be plotted.
            In this case the `param` argument selects which dimension/parameter is plotted if
            numerous are included in the `KDE`.
        * array_like scalar (D,N) of data with `D` parameters and `N` data points.

    labels : `None` or (D,) array_like of string, names of each parameter being plotted.

    kwcorner : dict, keyword-arguments passed to `Corner` constructor.

    **kwplot : additional keyword-arguments passed to `Corner.plot` method.

    """
    corner = Corner(kde_data, labels=labels, **kwcorner)
    corner.plot(**kwplot)
    return corner


'''
def plot_clean(kde_data, labels=None, kwcorner={}, **kwplot):
    corner = Corner(kde_data, labels=labels, **kwcorner)
    corner.clean(**kwplot)
    return corner


def plot_data(data, labels=None, kwcorner={}, **kwplot):
    corner = Corner(data, labels=labels, **kwcorner)
    corner.plot_data(**kwplot)
    return corner


def plot_hist(kde_data, labels=None, kwcorner={}, **kwplot):
    corner = Corner(kde_data, labels=labels, **kwcorner)
    corner.hist(**kwplot)
    return corner


def plot_kde(kde, labels=None, kwcorner={}, **kwplot):
    corner = Corner(kde, labels=labels, **kwcorner)
    corner.plot_kde(**kwplot)
    return corner
'''


# ======  Additional API Methods  ======
# ======================================


def carpet(xx, weights=None, ax=None, ystd=None, yave=None, shift=0.0,
           fancy=False, random='normal', rotate=False, **kwargs):
    """Draw a 'carpet plot' that shows semi-quantitatively the distribution of points.

    The given data (`xx`) is plotted as scatter points, where the abscissa (typically x-values) are
    the actual locations of the data and the ordinate are generated randomly.  The size and
    transparency of points are chosen based on the number of points.  If `weights` are given, it
    the size of the data points are chosen proportionally.

    NOTE: the `shift` argument determines the reference ordinate-value of the distribution, this is
          particularly useful when numerous datasets are being overplotted.


    Arguments
    ---------
    xx : (N,) array_like of scalar, the data values to be plotted

    weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
        only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

    ax : `None` or `matplotlib.axis.Axis`, if `None` the `plt.gca()` is used

    ystd : scalar or `None`, a measure of the dispersion in the ordinate scatter of values
        If `None` then an appropriate value is guessed based `yave` or the axis limits

    yave : scalar or `None`, the baseline at which the ordinate values are generated,
        This is very similar to the `shift` argument, determining the ordinate-offset, but in the
        case that `ystd` is not given but `yave` is given, then the `yave` value determines `ystd`.

    shift : scalar,
        A systematic ordinate shift of all data-points, particularly useful when multiple datasets
        are being plotted, such that one carpet plot can be offset from the other(s).

    fancy : bool,
        *Experimental* resizing of data-points to visually emphasize outliers.

    random : str, one of ['normal', 'uniform'],
        How the ordinate values are randomly generated: either a uniform or normal (i.e. Gaussian).

    rotate : bool, if True switch the x and y values such that x becomes the ordinate.

    kwargs : additional keyword-arguments passed to `matplotlib.axes.Axes.scatter()`


    """

    xx = np.asarray(xx)
    if ax is None:
        ax = plt.gca()

    # Dispersion (yaxis) of the fuzz values
    if ystd is None:
        if yave is None:
            get_lim_func = ax.get_xlim if rotate else ax.get_ylim
            ystd = get_lim_func()[1] * 0.02
        else:
            ystd = np.fabs(yave) / 5.0

    # Baseline on the yaxis at which the fuzz is plotted
    if yave is None:
        yave = -5.0 * ystd

    # Convert weights to a linear scaling for opacity and size
    if weights is None:
        ww = 1.0
    else:
        if utils.iqrange(weights, log=True) > 1:
            weights = np.log10(weights)
        ww = weights / np.median(weights)

    # General random y-values for the fuzz
    if random.lower() == 'normal':
        yy = np.random.normal(yave, ystd, size=xx.size)
    elif random.lower() == 'uniform':
        yy = np.random.uniform(yave-ystd, yave+ystd, size=xx.size)
    else:
        raise ValueError("Unrecognized `random` = '{}'!".format(random))

    # Choose an appropriate opacity
    alpha = kwargs.pop('alpha', None)
    if alpha is None:
        alpha = _scatter_alpha(xx)

    # Choose sizes proportional to their deviation (to make outliers more visible)
    size = 300 * ww / np.sqrt(xx.size)
    size = np.clip(size, 5, 100)

    # Try to make point sizes proportional to 'outlier'-ness... EXPERIMENTAL
    if fancy:
        # Estimate the deviation of each point from the median
        dev = np.fabs(xx - np.median(xx)) / np.std(xx)
        # Extend deviation based on weighting
        dev *= ww
        # Rescale the y-values based on their deviation from median
        yy = (yy - yave) / (np.sqrt(dev) + 1) + yave
        # Choose sizes proportional to their deviation (to make outliers more visible)
        size = (size / 1.5) * (1.5 + dev)

    # Set parameters
    color = kwargs.pop('color', _get_next_color(ax))
    kwargs.setdefault('facecolor', color)
    kwargs.setdefault('edgecolor', 'none')
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('alpha', alpha)
    kwargs.setdefault('s', kwargs.pop('size', size))

    extr = utils.minmax(yy)
    # trans = [ax.transData, ax.transAxes]
    if shift is not None:
        yy += shift

    # Switch x and y
    if rotate:
        temp = xx
        xx = yy
        yy = temp
        # trans = trans[::-1]

    # plot
    rv = ax.scatter(xx, yy, **kwargs), extr
    return rv


def confidence(data, ax=None, weights=None, quantiles=[0.5, 0.9],
               median=True, rotate=False, **kwargs):
    """Plot 1D Confidence intervals at the given quantiles.

    For each quantile `q`, a shaded range is plotted that includes a fration `q` of data values
    around the median.  Ultimately either `plt.axhspan` or `plt.axvspan` is used for drawing.


    Parameters
    ----------
    data : (N,) array_like of scalar, the data values around which to calculate confidence intervals

    ax : `None` or `matplotlib.axes.Axes` instance, if `None` then `plt.gca()` is used.

    weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
        only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

    quantiles : array_like of scalar values in [0.0, 1.0] denoting the fractions of data to mark.

    median : bool, mark the location of the median value.

    rotate : bool, if true switch the x and y coordinates (i.e. rotate plot 90deg clockwise).

    **kwargs : additional keyword-arguments passed to `plt.axhspan` or `plt.axvspan`.


    """

    if ax is None:
        ax = plt.gca()

    color = kwargs.pop('color', _get_next_color(ax))
    kwargs['facecolor'] = color
    kwargs['edgecolor'] = 'none'
    kwargs.setdefault('alpha', 0.1)

    # Calculate Cumulative Distribution Function
    if weights is None:
        data = np.sort(data)
        cdf = np.arange(data.size) / (data.size - 1)
    else:
        idx = np.argsort(data)
        data = data[idx]
        weights = weights[idx]
        cdf = np.cumsum(weights) / np.sum(weights)

    # Get both the lower (left) and upper (right) values of quantiles
    quantiles = np.asarray(quantiles) / 2
    qnts = np.append(0.5 - quantiles, 0.5 + quantiles)
    # Reshape to (Q, 2)
    locs = np.interp(qnts, cdf, data).reshape(2, len(quantiles)).T

    # Draw median line
    if median:
        mm = np.interp(0.5, cdf, data)
        line_func = ax.axhline if rotate else ax.axvline
        line_func(mm, ls='--', color=color, alpha=0.25)

    # Draw confidence bands
    for lo, hi in locs:
        span_func = ax.axhspan if rotate else ax.axvspan
        handle = span_func(lo, hi, **kwargs)

    return handle


def contour(data, edges=None, ax=None, weights=None,
            color=None, cmap=None, quantiles=None, smooth=1.0, upsample=2, pad=1, **kwargs):
    """Calculate and draw 2D contours.

    This is a wrapper for `draw_contour`, which in turn wraps `plt.contour`.  This function
    constructs bin-edges and calculates the histogram from which the contours are calculated.


    Arguments
    ---------
    data : (2, N) array_like of scalars,
        The data from which contours should be calculated.

    edges : object specifying historgam edge locations; or `None`
        * int : the number of bins for both dimensions, locations calculated automatically
        * (2,) array_like of int : the number of bins for each dimension.
        * (2,) of array_like : the bin-edge locations for each dimension, e.g.
                               ([0, 1, 2], [0.0, 0.1, 0.2, 0.3],) would describe two bins for
                               the 0th dimension, and 3 bins for the 1st dimension: i.e. 6 total.
        * (X,) array_like of scalar : the bin-edge locations to be used for both dimensions.
        * `None` : the number and locations of bins are calculated automatically.

    ax : `matplotlib.axes.Axes` instance, or `None`; if `None` then `plt.gca()` is used.

    weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
        only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

    color : matplotlib color specification (i.e. named color, hex or rgb) or `None`.
        If `None`:
            * `cmap` is given, then the color will be set to the `cmap` midpoint.
            * `cmap` is not given, then the color will be determined by the next value of the
                default matplotlib color-cycle, and `cmap` will be set to a matching colormap.
        This parameter effects the color of 1D: histograms, confidence intervals, and carpet;
        2D: scatter points.

    cmap : matplotlib colormap specification, or `None`
        * All valid matplotlib specifications can be used, e.g. named value (like 'Reds' or
        'viridis') or a `matplotlib.colors.Colormap` instance.
        * If `None` then a colormap is constructed based on the value of `color` (see above).

    quantiles : `None` or array_like of scalar values in [0.0, 1.0] denoting the fractions of
        data to demarkate with contours and confidence bands.

    smooth : scalar or `None`/`False`,
        if scalar: The width, in histogram bins, of a gaussian smoothing filter
        if `None` or `False`: no smoothing.

    upsample : int or `None`/`False`,
        if int: the factor by which to upsample the histogram by interpolation.
        if `None` or `False`: no upsampling

    pad : int, True, or `None`/`False`,
        if int: the number of edge bins added to the histogram to close contours hitting the edges
        if true: the default padding size is used
        if `None` or `False`: no padding is used.

    **kwargs : additiona keyword-arguments passed to `kalepy.plot.draw_contour2d()`.

    """

    if ax is None:
        ax = plt.gca()

    # Set color and cmap as needed
    color, cmap = _parse_color_cmap(ax=ax, color=color, cmap=cmap)

    # Calculate histogram
    edges = utils.parse_edges(data, edges=edges)
    hist, *_ = np.histogram2d(*data, bins=edges, weights=weights, density=True)
    # Plot
    rv = draw_contour2d(
        ax, edges, hist,
        cmap=cmap, quantiles=quantiles, smooth=smooth, upsample=upsample, pad=pad, **kwargs
    )
    return rv


def dist1d(kde_data, ax=None, edges=None, weights=None, probability=True, param=0, rotate=False,
           density=None, confidence=False, hist=None, carpet=True, color=None, quantiles=None):
    """Draw 1D data distributions with numerous possible components.

    The components of the plot are controlled by the arguments:
    * `density` : a KDE distribution curve,
    * `confidence` : 1D confidence bands calculated from a KDE,
    * `hist` : 1D histogram from the provided data,
    * `carpet` : 'carpet plot' (see `kalepy.plot.carpet()`) showing the data as a scatter-like plot.


    Arguments
    ---------
    kde_data : `kalepy.KDE` instance, (D,N) array_like of scalars, or `None`
        * instance of `kalepy.kde.KDE`, providing the data and KDE to be plotted.
            In this case the `param` argument selects which dimension/parameter is plotted if
            numerous are included in the `KDE`.
        * array_like scalar (D,N) of data with `D` parameters and `N` data points.

    ax : `matplotlib.axes.Axes` instance, or `None`; if `None` then `plt.gca()` is used.

    edges : object specifying historgam edge locations; or `None`
        * int : the number of bins, locations calculated automatically
        * array_like : the bin-edge locations
        * `None` : the number and locations of bins are calculated automatically

    weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
        only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

    probability : bool,
        Whether distributions (`hist` and `density`) are normalized such that the sum is unity.

    param : int,
        If a `KDE` instance is provided as the `kde_data` argument, and it includes multiple
        dimensions/parameters of data, then this argument determines which parameter is plotted.

    rotate : bool, if true switch the x and y coordinates (i.e. rotate plot 90deg clockwise).

    density : bool or `None`, whether the density KDE distribution is plotted or not.
        If `None` then this is set based on what is passed as the `kde_data`.

    confidence : bool, whether confidence intervals are plotted based on the KDE distribution,
        intervals are placed according to the `quantiles` argument.

    hist : bool or `None`, whether a histogram is plotted from the given data.
        If `None`, then the value is chosen based on the given `kde_data` argument.

    carpet : bool, whether or not a 'carpet plot' is shown from the given data.

    color : matplotlib color specification (i.e. named color, hex or rgb) or `None`.
        If `None` then the color will be determined by the next value of the default matplotlib
        color-cycle.

    quantiles : array_like of scalar values in [0.0, 1.0] denoting the fractions of data to mark.

    """

    # ---- Set parameters
    if isinstance(kde_data, kale.KDE):
        if weights is not None:
            raise ValueError("`weights` of given `KDE` instance cannot be overridden!")
        kde = kde_data
        data = kde.dataset
        if np.ndim(data) > 1:
            data = data[param]

        weights = None if kde._weights_uniform else kde.weights
    else:
        data = kde_data
        kde = None

    if ax is None:
        ax = plt.gca()

    # set default color to next from axes' color-cycle
    if color is None:
        color = _get_next_color(ax)

    # set default: plot KDE-density curve if KDE is given (data not given explicitly)
    if density is None:
        density = (kde is not None)

    # Default: plot histogram if data is given (KDE is *not* given)
    if hist is None:
        hist = (kde is None)

    # ---- Draw Components

    # Draw PDF from KDE
    handle = None     # variable to store a plotting 'handle' from one of the plotted objects
    if density:
        if kde is None:
            try:
                kde = kale.KDE(data, weights=weights)
            except:
                logging.error("Failed to construct KDE from given data!")
                raise

        # If histogram is also being plotted (as a solid line) use a dashed line
        ls = '--' if hist else '-'

        # Calculate KDE density distribution for the given parameter
        points, pdf = kde.density(probability=probability, params=param)
        # Plot
        if rotate:
            handle, = ax.plot(pdf, points, color=color, ls=ls)
        else:
            handle, = ax.plot(points, pdf, color=color, ls=ls)

    # Draw Histogram
    if hist:
        _, _, hh = hist1d(
            data, ax=ax, edges=edges, weights=weights, color=color,
            density=True, probability=probability, joints=True, rotate=rotate
        )
        if handle is None:
            handle = hh

    # Draw Contours and Median Line
    if confidence:
        hh = _confidence(data, ax=ax, color=color, quantiles=quantiles, rotate=rotate)
        if handle is None:
            handle = hh

    # Draw Carpet Plot
    if carpet:
        hh = _carpet(data, weights=weights, ax=ax, color=color, rotate=rotate)
        if handle is None:
            handle = hh

    return handle


def dist2d(kde_data, ax=None, edges=None, weights=None, params=[0, 1],
           quantiles=None, color=None, cmap=None, smooth=None, upsample=None, pad=True,
           median=True, scatter=True, contour=True, hist=True, mask_dense=None, mask_below=True):
    """Draw 2D data distributions with numerous possible components.

    The components of the plot are controlled by the arguments:
    * `median` : the median values of each coordinate in a 'cross-hairs' style,
    * `scatter` : 2D scatter points of the raw data,
    * `contour` : 2D contour plot from the KDE,
    * `hist` : 2D histogram of the raw data.

    These components are modified by:
    * `mask_dense` : mask over scatter points within the outer-most contour interval,
    * `mask_below` : mask out (ignore) histogram bins below a certain value.


    Arguments
    ---------
    kde_data : `kalepy.KDE` instance, or (D,N) array_like of scalars
        * instance of `kalepy.kde.KDE`, providing the data and KDE to be plotted.
            In this case the `param` argument selects which dimension/parameter is plotted if
            numerous are included in the `KDE`.
        * array_like scalar (D,N) of data with `D` parameters and `N` data points.

    ax : `matplotlib.axes.Axes` instance, or `None`; if `None` then `plt.gca()` is used.

    edges : object specifying historgam edge locations; or `None`
        * int : the number of bins for both dimensions, locations calculated automatically
        * (2,) array_like of int : the number of bins for each dimension.
        * (2,) of array_like : the bin-edge locations for each dimension, e.g.
                               ([0, 1, 2], [0.0, 0.1, 0.2, 0.3],) would describe two bins for
                               the 0th dimension, and 3 bins for the 1st dimension: i.e. 6 total.
        * (X,) array_like of scalar : the bin-edge locations to be used for both dimensions.
        * `None` : the number and locations of bins are calculated automatically.

    weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
        only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

    param : (2,) array_like of int,
        If a `KDE` instance is provided as the `kde_data` argument, and it includes multiple
        dimensions/parameters of data, then this argument determines which parameters are plotted.

    quantiles : array_like of scalar values in [0.0, 1.0] denoting the fractions of data to contour.

    color : matplotlib color specification (i.e. named color, hex or rgb) or `None`.
        If `None`:
            * `cmap` is given, then the color will be set to the `cmap` midpoint.
            * `cmap` is not given, then the color will be determined by the next value of the
                default matplotlib color-cycle, and `cmap` will be set to a matching colormap.
        This parameter effects the color of 1D: histograms, confidence intervals, and carpet;
        2D: scatter points.

    cmap : matplotlib colormap specification, or `None`
        * All valid matplotlib specifications can be used, e.g. named value (like 'Reds' or
        'viridis') or a `matplotlib.colors.Colormap` instance.
        * If `None` then a colormap is constructed based on the value of `color` (see above).

    smooth : scalar or `None`/`False`, smoothing of plotted contours (*only*)
        if scalar: The width, in histogram bins, of a gaussian smoothing filter
        if `None` or `False`: no smoothing.

    upsample : int or `None`/`False`, upsampling of plotted contours (*only*)
        if int: the factor by which to upsample the histogram by interpolation.
        if `None` or `False`: no upsampling

    pad : int, True, or `None`/`False`,
        if int: the number of edge bins added to the histogram to close contours hitting the edges
        if true: the default padding size is used
        if `None` or `False`: no padding is used.

    median : bool, mark the location of the median values in both dimensions (cross-hairs style).

    scatter : bool, whether to plot scatter points of the data points.
        The `mask_dense` parameter determines if some of these points are masked over.

    contour : bool, whether or not contours are plotted at the given `quantiles`.

    hist : bool, whether a 2D histogram is plotted from the given data.

    mask_dense : bool, whether to mask over high-density scatter points (within the lowest contour).

    mask_below : bool or scalar; whether, or the value below which, hist bins should be excluded.
        If True : exclude histogram bins with less than the average weight of a data point.
            If `weights` are not given, this means exclude empty histogram bins.
        If False : do not exclude any bins (i.e. include all bins).
        If scalar : exclude histogram bins with values below the given value.


    Notes
    -----
    - There is no `probability` argument because the normalization of the 2D distributions currently
      has no effect.

    """

    # ---- Process parameters

    if isinstance(kde_data, kale.KDE):
        if weights is not None:
            raise ValueError("`weights` of given `KDE` instance cannot be overridden!")
        kde = kde_data
        data = kde.dataset
        ndim = np.shape(data)[0]
        if ndim > 2:
            if len(params) != 2:
                raise ValueError("`dist2d` requires two chosen `params` (dimensions)!")
            data = np.vstack([data[ii] for ii in params])
        weights = None if kde._weights_uniform else kde.weights
    else:
        try:
            data = kde_data
            kde = kale.KDE(data, weights=weights)
        except:
            logging.error("Failed to construct KDE from given data!")
            raise

    if ax is None:
        ax = plt.gca()

    # Set default color or cmap as needed
    color, cmap = _parse_color_cmap(ax=ax, color=color, cmap=cmap)

    # Default: if either hist or contour is being plotted, mask over high-density scatter points
    if mask_dense is None:
        mask_dense = scatter and (hist or contour)

    # Calculate histogram (used for hist and contours)
    edges = utils.parse_edges(data, edges=edges)
    hh, *_ = np.histogram2d(*data, bins=edges, weights=weights, density=True)

    _, levels, quantiles = _dfm_levels(hh, quantiles=quantiles)
    if mask_below is True:
        # no weights : Mask out empty bins
        if weights is None:
            mask_below = 0.9 / len(data[0])
        # weights : Mask out bins with less than average weight
        else:
            mask_below = len(data[0]) / np.sum(weights)

    # ---- Draw components
    # ------------------------------------

    # ---- Draw Scatter Points
    if scatter:
        draw_scatter(ax, *data, color=color, zorder=5)

    # ---- Draw Median Lines (cross-hairs style)
    if median:
        for dd, func in zip(data, [ax.axvline, ax.axhline]):
            # Calculate value
            if weights is None:
                med = np.median(dd)
            else:
                med = utils.quantiles(dd, percs=0.5, weights=weights)

            # Load path_effects
            outline = _get_outline_effects()
            # Draw
            func(med, color=color, ls='-', alpha=0.25, lw=1.0, zorder=40, path_effects=outline)

    # ---- Draw 2D Histogram
    # We may need edges and histogram for `mask_dense` later; store them from hist2d or contour2d
    _ee = None
    _hh = None
    if hist:
        _ee, _hh, _ = draw_hist2d(
            ax, edges, hh, mask_below=mask_below, cmap=cmap, zorder=10
        )
        # Convert from edges to centers, then to meshgrid (if we need it)
        if mask_dense:
            _ee = [utils.midpoints(ee, axis=-1) for ee in _ee]
            _ee = np.meshgrid(*_ee, indexing='ij')

    # ---- Draw Contours
    if contour:
        contour_cmap = cmap.reversed()
        # Narrow the range of contour colors relative to full `cmap`
        dd = 0.7 / 2
        nq = len(quantiles)
        if nq < 4:
            dd = nq*0.08
        contour_cmap = _cut_colormap(contour_cmap, 0.5 - dd, 0.5 + dd)
        # Calculate PDF
        points, pdf = kde.density(params=params)
        # Plot
        _ee, _hh, _ = draw_contour2d(
            ax, points, pdf, quantiles=quantiles, smooth=smooth, upsample=upsample, pad=pad,
            cmap=contour_cmap, zorder=20,
        )

    # Mask dense scatter-points
    if mask_dense:
        # Load the histogram or PDF
        hh = _hh if (_hh is not None) else hh
        # Load the bin edges
        if _ee is not None:
            ee = _ee
        # Convert to mesh-grid of centerpoints if needed
        else:
            ee = [utils.midpoints(ee, axis=-1) for ee in edges]
            ee = np.meshgrid(*ee, indexing='ij')

        # NOTE: levels need to be recalculated here!
        _, levels, quantiles = _dfm_levels(hh, quantiles=quantiles)
        span = [levels.min(), hh.max()]
        # Set mask as white-to-white
        mask_cmap = [(1, 1, 1), (1, 1, 1)]
        mask_cmap = mpl.colors.LinearSegmentedColormap.from_list("mask", mask_cmap, N=2)
        # Draw
        ax.contourf(*ee, hh, span, cmap=mask_cmap, antialiased=True, zorder=9)

    return


def hist1d(data, edges=None, ax=None, weights=None, density=False, probability=False,
           renormalize=False, joints=True, positive=True, rotate=False, **kwargs):
    """Calculate and draw a 1D histogram.

    This is a thin wrapper around the `kalepy.plot.draw_hist1d()` method which draws a histogram
    that has already been computed (e.g. with `kalepy.utils.histogram` or `numpy.histogram`).

    Arguments
    ---------
    data : (N,) array_like of scalar, data to be histogrammed.

    edges : object specifying historgam edge locations; or `None`
        * int : the number of bins, locations calculated automatically
        * array_like : the bin-edge locations
        * `None` : the number and locations of bins are calculated automatically

    ax : `matplotlib.axes.Axes` instance, or `None`; if `None` then `plt.gca()` is used.

    weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
        only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

    density : bool or `None`, whether the density KDE distribution is plotted or not.
        If `None` then this is set based on what is passed as the `kde_data`.

    probability : bool,
        Whether distributions (`hist` and `density`) are normalized such that the sum is unity.
        NOTE: this can be overridden by the `renormalize` argument.

    renormalize : bool or scalar, whether or to what value to renormalize the histrogram maximum.
        If True : renormalize the maximum histogram value to unity.
        If False : do not renormalize.
        If scalar : renormalize the histogram maximum to this value.

    joints : bool, plot the vertical connectors ('joints') between histogram bins; if False, only
        horizontal lines are plotted for each bin.

    positive : bool, only plot bins with positive values.

    rotate : bool, if true switch the x and y coordinates (i.e. rotate plot 90deg clockwise).

    **kwargs : additional keyword-arguments passed to `kalepy.plot.draw_hist1d()`.
        Any arguments not caught by `draw_hist1d()` are eventually passed to `plt.plot()` method.


    To-Do
    -----
    - Add `scipy.binned_statistic` functionality for arbitrary statistics beyond histgramming.

    """

    if ax is None:
        ax = plt.gca()

    # Calculate histogram
    hist, edges = utils.histogram(
        data, bins=edges, weights=weights, density=density, probability=probability
    )

    # Draw
    rv = draw_hist1d(
        ax, edges, hist,
        renormalize=renormalize, joints=joints, positive=positive, rotate=rotate,
        **kwargs
    )
    return hist, edges, rv


def hist2d(data, edges=None, ax=None, weights=None, mask_below=False, **kwargs):
    """Calculate and draw a 2D histogram.

    This is a thin wrapper around the `kalepy.plot.draw_hist2d()` method which draws a 2D histogram
    that has already been computed (e.g. with `numpy.histogram2d`).

    Arguments
    ---------
    data : (2, N) array_like of scalar, data to be histogrammed.

    edges : object specifying historgam edge locations; or `None`
        * int : the number of bins for both dimensions, locations calculated automatically
        * (2,) array_like of int : the number of bins for each dimension.
        * (2,) of array_like : the bin-edge locations for each dimension, e.g.
                               ([0, 1, 2], [0.0, 0.1, 0.2, 0.3],) would describe two bins for
                               the 0th dimension, and 3 bins for the 1st dimension: i.e. 6 total.
        * (X,) array_like of scalar : the bin-edge locations to be used for both dimensions.
        * `None` : the number and locations of bins are calculated automatically.

    ax : `matplotlib.axes.Axes` instance, or `None`; if `None` then `plt.gca()` is used.

    weights : `None` or (N,) array_like of scalar, the weighting of each data-point if and
        only-if the given `kde_data` argument is a (D,N) array_like of scalar data.

    mask_below : bool or scalar; whether, or the value below which, hist bins should be excluded.
        If True : exclude histogram bins with less than the average weight of a data point.
            If `weights` are not given, this means exclude empty histogram bins.
        If False : do not exclude any bins (i.e. include all bins).
        If scalar : exclude histogram bins with values below the given value.

    **kwargs : additional keyword-arguments passed to `kalepy.plot.draw_hist2d()`.
        Any arguments not caught by `draw_hist1d()` are eventually passed to `plt.pcolormesh()`.


    To-Do
    -----
    - Add `scipy.binned_statistic` functionality for arbitrary statistics beyond histgramming.

    """

    if ax is None:
        ax = plt.gca()

    if mask_below is True:
        mask_below = 0.9 / len(data[0])

    # Calculate histogram
    edges = utils.parse_edges(data, edges=edges)
    hist, *_ = np.histogram2d(*data, bins=edges, weights=weights, density=True)
    # Draw
    rv = draw_hist2d(ax, edges, hist, mask_below=mask_below, **kwargs)
    return rv


# ======  Drawing Methods  =====
# ==============================


def draw_hist1d(ax, edges, hist, renormalize=False, nonzero=False, positive=False,
                joints=True, rotate=False, **kwargs):

    # Construct plot points to manually create a step-plot
    xval = np.hstack([[edges[jj], edges[jj+1]] for jj in range(len(edges)-1)])
    yval = np.hstack([[hh, hh] for hh in hist])

    if not joints:
        size = len(xval)
        half = size // 2

        outs = []
        for zz in [xval, yval]:
            zz = np.atleast_2d(zz).T
            zz = np.reshape(zz, [half, 2], order='C')
            zz = np.pad(zz, [[0, 0], [0, 1]], constant_values=np.nan)
            zz = zz.reshape(size + half)
            outs.append(zz)

        xval, yval = outs

    # Select nonzero values
    if nonzero:
        xval = np.ma.masked_where(yval == 0.0, xval)
        yval = np.ma.masked_where(yval == 0.0, yval)

    # Select positive values
    if positive:
        xval = np.ma.masked_where(yval < 0.0, xval)
        yval = np.ma.masked_where(yval < 0.0, yval)

    if rotate:
        temp = np.array(xval)
        xval = yval
        yval = temp

    # Plot Histogram
    if renormalize not in [False, None]:
        if renormalize is True:
            renormalize = 1.0
        yval = yval / yval[np.isfinite(yval)].max()
        yval *= renormalize

    line, = ax.plot(xval, yval, **kwargs)

    return line


def draw_hist2d(ax, edges, hist, mask_below=None, **kwargs):
    if mask_below not in [False, None]:
        hist = np.ma.masked_less_equal(hist, mask_below)
    kwargs.setdefault('shading', 'auto')
    # kwargs.setdefault('edgecolors', 'face')
    # NOTE: this avoids edge artifacts when alpha is not unity!
    kwargs.setdefault('edgecolors', [1.0, 1.0, 1.0, 0.0])
    kwargs.setdefault('linewidth', 0.01)
    # Plot
    rv = ax.pcolormesh(*edges, hist.T, **kwargs)
    return edges, hist, rv


def draw_contour2d(ax, edges, hist, quantiles=None, smooth=None, upsample=None, pad=True,
                   outline=True, **kwargs):

    LW = 1.5

    # ---- (Pre-)Process histogram and bin edges

    # Pad Histogram for Smoother Contour Edges
    if pad not in [False, None]:
        if pad is True:
            pad = _PAD
        edges, hist = _pad_hist(edges, hist, pad)

    # Convert from bin edges to centers as needed
    xx, yy = _match_edges_to_hist(edges, hist)

    # Construct grid from center values
    xx, yy = np.meshgrid(xx, yy, indexing='ij')

    # Perform upsampling
    if (upsample not in [None, False]):
        import scipy as sp
        if upsample is True:
            upsample = 2
        xx = sp.ndimage.zoom(xx, upsample)
        yy = sp.ndimage.zoom(yy, upsample)
        hist = sp.ndimage.zoom(hist, upsample)
    # perform smoothing
    if (smooth not in [None, False]):
        import scipy as sp
        if upsample is not None:
            smooth *= upsample
        hist = sp.ndimage.filters.gaussian_filter(hist, smooth)

    # Update edges based on pre-processing
    edges = [xx, yy]

    # ---- Setup parameters
    _, levels, quantiles = _dfm_levels(hist, quantiles=quantiles)
    alpha = kwargs.setdefault('alpha', 0.8)
    lw = kwargs.pop('linewidths', kwargs.pop('lw', LW))
    kwargs.setdefault('linestyles', kwargs.pop('ls', '-'))
    kwargs.setdefault('zorder', 10)

    # ---- Draw contours
    cont = ax.contour(xx, yy, hist, levels=levels, linewidths=lw, **kwargs)

    # ---- Add Outline path effect to contours
    if (outline is True):
        # If multiple linewidths were specified, add outlines individually
        if (not np.isscalar(lw)) and (len(cont.collections) == len(lw)):
            for line, _lw in zip(cont.collections, lw):
                outline = _get_outline_effects(2*_lw, alpha=1 - np.sqrt(1 - alpha))
                plt.setp(line, path_effects=outline)
        # Add uniform outlines to all contour lines
        elif np.isscalar(lw):
            outline = _get_outline_effects(2*lw, alpha=1 - np.sqrt(1 - alpha))
            plt.setp(cont.collections, path_effects=outline)
        # Otherwise error
        else:
            err = (
                "kalepy.plot.draw_contour2d() :: ",
                "Disregarding unexpected `lw`/`linewidths` argument: '{}'".format(lw)
            )
            logging.warning(err)
            outline = _get_outline_effects(2*LW, alpha=1 - np.sqrt(1 - alpha))
            plt.setp(cont.collections, path_effects=outline)

    elif (outline is not False):
        raise ValueError("`outline` must be either 'True' or 'False'!")

    return edges, hist, cont


def draw_scatter(ax, xx, yy, alpha=None, s=4, **kwargs):
    # color = kwargs.pop('color', kwargs.pop('c', None))
    # fc = kwargs.pop('facecolor', kwargs.pop('fc', None))
    # if fc is None:
    #     fc = ax._get_lines.get_next_color()
    # kwargs.setdefault('facecolor', color)
    # kwargs.setdefault('edgecolor', 'none')
    if alpha is None:
        alpha = _scatter_alpha(xx)
    kwargs.setdefault('alpha', alpha)
    kwargs.setdefault('s', s)
    return ax.scatter(xx, yy, **kwargs)


# ====  Utility Methods  ====
# ===========================


def _confidence(*args, **kwargs):
    """Wrapper for `confidence`, allows for reusing the variable name in function calls."""
    return confidence(*args, **kwargs)


def _dist1d(*args, **kwargs):
    """Wrapper for `dist1d`, allows for reusing the variable name in function calls."""
    return dist1d(*args, **kwargs)


def _dist2d(*args, **kwargs):
    """Wrapper for `dist2d`, allows for reusing the variable name in function calls."""
    return dist2d(*args, **kwargs)


def _carpet(*args, **kwargs):
    """Wrapper for `carpet`, allows for reusing the variable name in function calls."""
    return carpet(*args, **kwargs)


def _set_corner_axes_extrema(axes, extrema, rotate, pdf=None):
    """Set all of the axes in a corner plot to the given extrema (limits).
    """
    npar = len(axes)
    last = npar - 1
    if not np.all([sh == npar for sh in np.shape(axes)]):
        raise ValueError("`axes` (shape: {}) must be square!".format(np.shape(axes)))

    if len(extrema) == 2 and npar != 2:
        extrema = [extrema] * npar

    if len(extrema) != npar:
        err = "Length of `extrema` (shape: {}) does not match axes shape ({}^2)!".format(
            np.shape(extrema), npar)
        raise ValueError(err)

    if (pdf is not None) and (len(pdf) != 2 or not utils.really1d(pdf)):
        raise ValueError("`pdf` (shape: {}) must be length 2!".format(np.shape(pdf)))

    for (ii, jj), ax in np.ndenumerate(axes):
        if jj > ii:
            ax.set_visible(False)
            continue

        # Diagonals
        # ----------------------
        if ii == jj:
            rot = (rotate and (jj == last))
            set_lim_func = ax.set_ylim if rot else ax.set_xlim
            set_lim_func(extrema[jj])

        # Off-Diagonals
        # ----------------------
        else:
            ax.set_xlim(extrema[jj])
            ax.set_ylim(extrema[ii])

    return extrema


def _figax(size, grid=True, left=None, bottom=None, right=None, top=None, hspace=None, wspace=None,
           **kwfig):
    """Construct a matplotlib figure and axes with the given parameters.
    """
    _def_figsize = np.clip(4 * size, 6, 20)
    _def_figsize = [_def_figsize for ii in range(2)]

    figsize = kwfig.pop('figsize', _def_figsize)
    if not np.iterable(figsize):
        figsize = [figsize, figsize]

    if hspace is None:
        hspace = 0.1
    if wspace is None:
        wspace = 0.1

    fig, axes = plt.subplots(figsize=figsize, squeeze=False, ncols=size, nrows=size, **kwfig)

    plt.subplots_adjust(
        left=left, bottom=bottom, right=right, top=top, hspace=hspace, wspace=wspace)
    if grid is True:
        grid = dict(alpha=0.2, color='0.5', lw=0.5)
    elif grid is False:
        grid = None

    for idx, ax in np.ndenumerate(axes):
        if grid is not None:
            ax.grid(True, **grid)

    return fig, axes


def _match_edges_to_hist(edges, hist):
    """Convert from bin-edges to bin-centers as needed for 2D histograms.
    """
    esh = tuple([len(ee) for ee in edges])
    esh_p1 = tuple([len(ee) - 1 for ee in edges])
    # If the shape of edges matches the hist, then we're good
    if np.shape(hist) == esh:
        pass
    # If `edges` have one more element each, then convert from edges to centers
    elif np.shape(hist) == esh_p1:
        edges = [utils.midpoints(ee, axis=-1) for ee in edges]
    else:
        err = (
            "Shape of hist [{}=(X,Y)] ".format(np.shape(hist)),
            "does not match edges ({})!".format(esh)
        )
        raise ValueError(err)

    return edges


def _parse_kde_data(kde_data, weights=None):
    """Convert from either a KDE or data (array) to both.
    """
    if isinstance(kde_data, kale.KDE):
        if weights is not None:
            raise ValueError("`weights` must be used from given `KDE` instance!")
        kde = kde_data
        data = kde.dataset
        weights = None if kde._weights_uniform else kde.weights
    # If the raw data is given, construct a KDE from it
    else:
        try:
            data = kde_data
            kde = kale.KDE(kde_data, weights=weights)
        except:
            err = "Failed to construct `KDE` instance from given data!"
            logging.error(err)
            raise

    if not isinstance(kde, kale.KDE):
        raise RuntimeError("kalepy.plot._parse_kde_data() :: failed to produce `KDE` instance!")

    return kde, data, weights


def _parse_color_cmap(ax=None, color=None, cmap=None):
    """Set `color` and `cmap` values appropriately.
    """
    if (color is None) and (cmap is None):
        if ax is None:
            ax = plt.gca()
        color = _get_next_color(ax)
        cmap = _color_to_cmap(color)
    elif (color is None):
        cmap = plt.get_cmap(cmap)
        color = cmap(0.5)
    else:
        cmap = _color_to_cmap(color)

    return color, cmap


def _dfm_levels(data, quantiles=None, sigmas=None):
    quantiles, sigmas = _default_quantiles(quantiles=quantiles, sigmas=sigmas)

    # Compute the density levels.
    data = np.asarray(data).flatten()
    inds = np.argsort(data)[::-1]
    data = data[inds]
    sm = np.cumsum(data)
    sm /= sm[-1]
    levels = np.empty(len(quantiles))
    for i, v0 in enumerate(quantiles):
        try:
            levels[i] = data[sm <= v0][-1]
        except:
            levels[i] = data[0]

    levels.sort()

    # -- Remove Bad Levels
    # bad = (np.diff(levels) == 0)
    # bad = np.pad(bad, [1, 0], constant_values=False)
    # levels = np.delete(levels, np.where(bad)[0])
    # if np.any(bad):
    #     _levels = quantiles
    #     quantiles = np.array(quantiles)[~bad]
    #     logging.warning("Removed bad levels: '{}' ==> '{}'".format(_levels, quantiles))

    return sigmas, levels, quantiles


def _default_quantiles(quantiles=None, sigmas=None):
    """Set default quantile values.
    """
    if quantiles is None:
        if sigmas is None:
            sigmas = _DEF_SIGMAS
        # Convert from standard-deviations to CDF values
        quantiles = 1.0 - np.exp(-0.5 * np.square(sigmas))
    elif sigmas is None:
        quantiles = np.asarray(quantiles)
        sigmas = np.sqrt(-2.0 * np.log(1.0 - quantiles))

    return quantiles, sigmas


def _pad_hist(edges, hist, pad):
    """Pad the given histogram to allow contours near/at the edges to close.
    """
    hh = np.pad(hist, pad, mode='constant', constant_values=hist.min())
    tf = np.arange(1, pad+1)  # [+1, +2]
    tr = - tf[::-1]    # [-2, -1]
    edges = [
        [ee[0] + tr * np.diff(ee[:2]), ee, ee[-1] + tf * np.diff(ee[-2:])]
        for ee in edges
    ]
    edges = [np.concatenate(ee) for ee in edges]
    return edges, hh


def _scatter_alpha(xx, norm=10.0):
    """Choose a transparency for the given number of scatter points.
    """
    alpha = norm / np.sqrt(len(xx))
    # NOTE: array values dont work for alpha parameters (added to `colors`)
    # if alpha is None:
    #     aa = 10 / np.sqrt(xx.size)
    #     alpha = aa
    #     # alpha = aa * ww
    #     # alpha = np.clip(alpha, aa/10, aa*10)
    #     # alpha = np.clip(alpha, 1e-4, 1e-1)

    return alpha


def _get_outline_effects(lw=2.0, fg='0.75', alpha=0.8):
    outline = ([
        mpl.patheffects.Stroke(linewidth=lw, foreground=fg, alpha=alpha),
        mpl.patheffects.Normal()
    ])
    return outline


def _get_next_color(ax):
    return ax._get_lines.get_next_color()


def _color_to_cmap(col, pow=0.333, sat=0.25, val=0.25, white=1.0, black=0.0):
    """Construct a matplotlib colormap based on the given color.
    """
    rgb = mpl.colors.to_rgb(col)

    # ---- Increase 'value' and 'saturation' of color
    # Convert to HSV
    hsv = mpl.colors.rgb_to_hsv(rgb)
    # Increase '[v]alue'
    par = 2
    hsv[par] = np.interp(val, [0.0, 1.0], [hsv[par], 1.0])
    # Increase '[s]aturation'
    par = 1
    hsv[par] = np.interp(sat, [0.0, 1.0], [hsv[par], 1.0])
    # Convert back to RGB
    rgb = mpl.colors.hsv_to_rgb(hsv)

    # ---- Create edge colors near-white and near-black
    # find distance to white and black
    dw = np.linalg.norm(np.diff(np.vstack([rgb, np.ones_like(rgb)]), axis=0)) / np.sqrt(3)
    db = np.linalg.norm(np.diff(np.vstack([rgb, np.zeros_like(rgb)]), axis=0)) / np.sqrt(3)
    # shift edges towards white and black proportionally to distance
    lo = [np.interp(dw**pow, [0.0, 1.0], [ll, white]) for ll in rgb]
    hi = [np.interp(db**pow, [0.0, 1.0], [ll, black]) for ll in rgb]

    # ---- Construct colormap
    my_colors = [lo, rgb, hi]
    cmap = mpl.colors.LinearSegmentedColormap.from_list("mycmap", my_colors)
    return cmap


def _cut_colormap(cmap, min=0.0, max=1.0, n=10):
    """Truncate the given colormap with the given minimum and maximum values.
    """
    name = 'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=min, b=max)
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        name, cmap(np.linspace(min, max, n)))
    return new_cmap


def nbshow():
    return utils.run_if_notebook(plt.show, otherwise=lambda: plt.close('all'))


'''
def _get_corner_axes_extrema(axes, rotate, extrema=None, pdf=None):
    npar = len(axes)
    last = npar - 1
    if not np.all([sh == npar for sh in np.shape(axes)]):
        raise ValueError("`axes` (shape: {}) must be square!".format(np.shape(axes)))

    if extrema is None:
        extrema = npar * [None]

    for (ii, jj), ax in np.ndenumerate(axes):
        if jj > ii:
            continue

        if ii == jj:
            pdf_func = ax.get_xlim if (rotate and (ii == last)) else ax.get_ylim
            oth_func = ax.get_ylim if (rotate and (ii == last)) else ax.get_xlim
            pdf = utils.minmax(pdf_func(), prev=pdf)
            extrema[jj] = utils.minmax(oth_func(), prev=extrema[jj])

        else:
            extrema[jj] = utils.minmax(ax.get_xlim(), prev=extrema[jj])
            extrema[ii] = utils.minmax(ax.get_ylim(), prev=extrema[ii])

    return extrema, pdf
'''

'''
def _get_smap(args=[0.0, 1.0], cmap=None, log=False, norm=None, under='w', over='w'):
    args = np.asarray(args)

    if not isinstance(cmap, mpl.colors.Colormap):
        if cmap is None:
            cmap = 'viridis'
        if isinstance(cmap, six.string_types):
            cmap = plt.get_cmap(cmap)

    import copy
    cmap = copy.copy(cmap)
    if under is not None:
        cmap.set_under(under)
    if over is not None:
        cmap.set_over(over)

    vmin, vmax = utils.minmax(args, positive=log)
    if vmin == vmax:
        raise ValueError("`smap` extrema are identical: {}, {}!".format(vmin, vmax))

    if log:
        norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    # Create scalar-mappable
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    # Bug-Fix something something
    smap._A = []
    smap._log = log

    return smap


def _parse_smap(smap, color, cmap=None, defaults=dict(log=False)):
    uniform = False
    if isinstance(smap, mpl.cm.ScalarMappable):
        # If `smap` was created with `kalepy.plot._get_smap()` than it should have this attribute
        try:
            smap_is_log = smap._log
        # Otherwise assume it's linear
        # NOTE: this might be wrong.  Better way to check?
        except AttributeError:
            smap_is_log = False

        return smap, smap_is_log, uniform

    if smap is None:
        smap = {}

    if not isinstance(smap, dict):
        raise ValueError("`smap` must either be a dict or ScalarMappable!")

    for kk, vv in defaults.items():
        smap.setdefault(kk, vv)

    smap_is_log = smap['log']
    if cmap is None:
        cmap = _COLOR_CMAP.get(color[0].lower(), None)
        if cmap is None:
            cmap = 'Greys'
            uniform = True

    smap.setdefault('cmap', cmap)

    return smap, smap_is_log, uniform
'''


'''
def align_axes_loc(tw, ax, ymax=None, ymin=None, loc=0.0):
    if ((ymax is None) and (ymin is None)) or ((ymin is not None) and (ymax is not None)):
        raise ValueError("Either `ymax` or `ymin`, and not both, must be provided!")

    ylim = np.array(ax.get_ylim())
    # beg = np.array(tw.get_ylim())

    hit = np.diff(ylim)[0]
    frac_up = (loc - ylim[0]) / hit
    frac_dn = 1 - frac_up

    new_ylim = [0.0, 0.0]
    if ymax is not None:
        new_ylim[1] = ymax
        new_hit = (ymax - loc) / frac_dn
        new_ylim[0] = ymax - new_hit

    if ymin is not None:
        new_ylim[0] = ymin
        new_hit = (loc - ymin) / frac_up
        new_ylim[1] = ymax - new_hit

    tw.set_ylim(new_ylim)
    return new_ylim
'''

'''
def _draw_colorbar_contours(cbar, levels, colors=None, smap=None):
    ax = cbar.ax

    if colors is None:
        if smap is None:
            colors = ['0.5' for ll in levels]
        else:
            colors = [smap.to_rgba(ll) for ll in levels]
            # colors = [_invert_color(cc) for cc in colors]

    orient = cbar.orientation
    if orient.startswith('v'):
        line_func = ax.axhline
    elif orient.startswith('h'):
        line_func = ax.axvline
    else:
        raise RuntimeError("UNKNOWN ORIENTATION '{}'!".format(orient))

    for ll, cc, bg in zip(levels, colors, colors[::-1]):
        effects = ([
            mpl.patheffects.Stroke(linewidth=4.0, foreground=bg, alpha=0.5),
            mpl.patheffects.Normal()
        ])
        line_func(ll, 0.0, 1.0, color=cc, path_effects=effects, lw=2.0)

    return
'''

'''
def _invert_color(col):
    rgba = mpl.colors.to_rgba(col)
    alpha = rgba[-1]
    col = 1.0 - np.array(rgba[:-1])
    col = tuple(col.tolist() + [alpha])
    return col
'''

'''
if colorbar:
    if fig is None:
        fig = plt.gcf()

    # bbox = ax.get_position()
    # cbax = fig.add_axes([bbox.x1+PAD, bbox.y0, 0.03, bbox.height])

    # if size in [2, 3]:
    bbox = axes[0, -1].get_position()
    left = bbox.x0
    width = bbox.width
    top = bbox.y1
    height = 0.04
    # elif size in [4, 5]
    #     bbox = axes[0, -2].get_position()
    #     left = bbox.x0
    #     width = bbox.width

    cbax = fig.add_axes([left, top - height, width, height])
    cbar = plt.colorbar(smap, orientation='horizontal', cax=cbax)
    _draw_colorbar_contours(cbar, pdf_levels, smap=smap)
'''
