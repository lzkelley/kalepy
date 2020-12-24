"""
"""

# import warnings
import logging
# import os
import six

import numpy as np
import scipy as sp
import scipy.stats  # noqa
import matplotlib as mpl
import matplotlib.patheffects  # noqa
import matplotlib.pyplot as plt

import kalepy as kale
from kalepy import utils
from kalepy import KDE

_STRETCH = 0.2
_COLOR_CMAP = {
    'k': 'Greys',
    'b': 'Blues',
    'r': 'Reds',
    'g': 'Greens',
    'o': 'Oranges',
    'p': 'Purples',
}
# _DEF_SIGMAS = np.arange(0.5, 2.1, 0.5)
_DEF_SIGMAS = [0.5, 1.0, 1.5, 2.0]
_MASK_CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

_OUTLINE = ([
    mpl.patheffects.Stroke(linewidth=2.0, foreground='white', alpha=0.85),
    mpl.patheffects.Normal()
])


class Corner:

    _LW = 2.0
    _LIMITS_STRETCH = 0.1

    def __init__(self, kde_data, weights=None, labels=None, limits=None, rotate=None, **kwfig):
        """Construct a figure and axes for creating a 'corner' plot.

        Arguments
        ---------
        kde_data : object, one of the following
            * instance of `kalepy.kde.KDE`, including the data to be plotted,
            * array_like scalar (D,N) of data with `D` parameters and `N` data points,
            * int D, the number of parameters to construct a DxD corner plot.

        weights : array_like scalar (N,) or None
            The weights for each data point.
            NOTE: only applicable when `kde_data` is a (D,N) dataset.

        labels : array_like string (N,) of labels/names for each parameters.

        limits : None or (2,) or (D,) of [None or (2,)]
            Specification for the limits of each axes (for each of `D` parameters):
            * None : the limits are determined automatically,
            * (2,) : limits to be applied to all axes,
            * (3,) : limits for each axis, where each entry can be either 'None' or (2,)

        **kwfig : keyword-arguments passed to `_figax()` for constructing figure and axes.
            See `kalepy.plot._figax()` for specifications.

        """

        # --- Parse the given `kde_data` and store parameters accordingly --
        kde = None
        data = None
        # If a KDE instance is given, take all parameters from there
        if isinstance(kde_data, kale.KDE):
            if weights is not None:
                raise ValueError("`weights` must be used from given `KDE` instance!")
            kde = kde_data
            data = kde.dataset
            weights = None if kde._weights_uniform else kde.weights
            size = len(data)
        # If an integer is given, it's the number of parameters/variables
        elif np.isscalar(kde_data):
            if not isinstance(kde_data, int):
                err = ("If `kde_data` is a scalar, it must be an integer "
                       "specifying the number of parameters!")
                raise ValueError(err)
            size = kde_data
        # If the raw data is given, construct a KDE from it
        else:
            try:
                kde = KDE(kde_data, weights=weights)
            except:
                err = "Failed to construct `KDE` instance from given data!"
                logging.error(err)
                raise

            data = kde_data
            size = len(data)

        if limits is None:
            limits = [None] * size

        if rotate is None:
            # rotate = (size == 2)
            rotate = (size in [2, 3])

        # -- Construct figure and axes using `_figax()`
        fig, axes = _figax(size, **kwfig)
        self.fig = fig
        self.axes = axes

        # -- Setup axes
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
                if ii != 0:
                    ax.yaxis.set_label_position('right')
                    ax.yaxis.set_ticks_position('right')

            # Off-Diagonals
            else:
                pass

        # --- Store key parameters
        self.size = size
        self._data = data
        self._weights = weights
        self._kde = kde
        self._limits = limits
        self._rotate = rotate

        return

    def plot(self, kde=None, data=None, kwkde=None, kwdata=None, **kwargs):
        axes = self.axes
        npar = self.size

        rotate = kwargs.setdefault('rotate', npar == 2)

        if (kde is None) or (kde is True):
            kde = self._kde
        elif (kde is False):
            kde = None

        if (data is None) or (data is True):
            if kde is not None:
                data = kde.dataset
            elif self._kde is not None:
                data = self._kde.dataset

        elif (data is False):
            data = None

        if (data is None) and (kde is None):
            raise ValueError("Neither `kde` nor `data` is being plotted!")

        if kwkde is None:
            kwkde = kwargs.copy()
        if kwdata is None:
            kwdata = kwargs.copy()

        '''
        kwkde.setdefault('hist2d', False)
        kwkde.setdefault('rotate', rotate)
        kwdata.setdefault('contour', False)
        kwdata.setdefault('rotate', rotate)

        for kk, vv in kwargs.items():
            kwkde.setdefault(kk, vv)
            kwdata.setdefault(kk, vv)
        '''

        extrema = None
        pdf = None
        if kde is not None:
            corner_kde(axes, kde, **kwkde)
            extrema, pdf = _get_corner_axes_extrema(axes, rotate)
        if data is not None:
            self._plot_data(axes, data, **kwdata)
            extrema, pdf = _get_corner_axes_extrema(axes, rotate, extrema=extrema, pdf=pdf)

        _set_corner_axes_extrema(axes, extrema, rotate, pdf=None)

        return

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

    def hist(self, **kw):
        return self.plot_data(scatter=False, carpet=False, dist1d=dict(contour=False), **kw)

    def plot_data(self, data=None, edges=None, weights=None, **kwdata):
        if data is None:
            data = self._data
        if weights is None:
            weights = self._weights

        limits = self._plot_data(self.axes, data, weights=weights, rotate=self._rotate, **kwdata)

        for ii in range(self.size):
            if self._limits[ii] is None:
                self._limits[ii] = utils.minmax(limits[ii], prev=self._limits[ii])

        _set_corner_axes_extrema(self.axes, self._limits, self._rotate)
        return

    @classmethod
    def _plot_data(cls, axes, data, weights=None, edges=None, quantiles=None, sigmas=None,
                   rotate=True, scatter=True, hist=True, carpet=True,
                   dist1d={}, dist2d={}, **kwargs):

        if np.ndim(data) != 2:
            err = "`data` (shape: {}) must be 2D with shape (parameters, data-points)!".format(
                np.shape(data))
            raise ValueError(err)

        size = np.shape(data)[0]
        shp = np.shape(axes)
        if (np.ndim(axes) != 2) or (shp[0] != shp[1]) or (shp[0] != size):
            raise ValueError("`axes` (shape: {}) does not match data dimension {}!".format(shp, size))

        last = size - 1
        if rotate is None:
            rotate = (size == 2)

        kwargs.setdefault('color', 'k')
        kwargs.setdefault('lw', cls._LW)
        kwargs.setdefault('alpha', 0.8)

        dist1d = _none_dict(dist1d, 'dist1d', kwargs)
        dist2d = _none_dict(dist2d, 'dist2d', kwargs)

        edges = utils.parse_edges(data, edges=edges)
        # extr = [utils.minmax(dd) for dd in data]
        # smap = None
        # cmap = None
        # smap, smap_is_log, uniform_color = _parse_smap(smap, kwargs['color'], cmap=cmap)
        if weights is not None:
            logging.warning("WARNING: `weights` not being accounted for in contour locations")
        sigmas, levels, quantiles = _dfm_levels(data, quantiles=quantiles, sigmas=sigmas)

        #
        # Draw / Plot Data
        # ===========================

        # Draw 1D Histograms & Carpets
        # -----------------------------------------
        limits = [None] * size
        for jj, ax in enumerate(axes.diagonal()):
            rot = (rotate and (jj == last))
            cls.dist1d_data(
                ax, edges[jj], data[jj], weights=weights, quantiles=quantiles,
                rotate=rot, carpet=carpet, **dist1d
            )
            limits[jj] = utils.minmax(data[jj], stretch=cls._LIMITS_STRETCH)

        # Draw 2D Histograms and Contours
        # -----------------------------------------
        for (ii, jj), ax in np.ndenumerate(axes):
            if jj >= ii:
                continue

            cls.dist2d_data(
                ax, [edges[jj], edges[ii]], data=[data[jj], data[ii]], weights=weights,
                quantiles=quantiles, scatter=scatter, hist=hist, **dist2d
            )

        return limits

    @classmethod
    def dist1d_data(cls, ax, edges, data, weights=None, quantiles=None, median=True,
                    rotate=False, contour=True, hist=True, carpet=True, **kwargs):

        color = kwargs.setdefault('color', 'k')
        kwargs.setdefault('lw', 1.0)
        kwargs.setdefault('alpha', 0.8)

        hist = _none_dict(hist, 'hist', kwargs)
        carpet = _none_dict(carpet, 'carpet', kwargs)
        contour = _none_dict(contour, 'contour', dict(quantiles=quantiles, median=median))

        # Draw Histogram
        # --------------------------
        if (hist is not None):
            cls._hist1d(ax, edges, data, weights=weights, rotate=rotate, **hist)

        # Draw Contours and Median Line
        # ------------------------------------
        if (contour is not None):
            cls._contour1d(ax, data, rotate=rotate, color=color, **contour)

        # Draw Carpet Plot
        # ------------------------------------
        if (carpet is not None):
            _draw_carpet(data, weights=weights, ax=ax, rotate=rotate, **carpet)

        return

    @classmethod
    def dist2d_data(cls, ax, edges, data, weights=None, quantiles=None,
                    scatter=True, hist=True, contour=True,
                    median=True, pad=True, mask_dense=None, mask_sparse=True, **kwargs):

        color = kwargs.setdefault('color', 'k')
        kwargs.setdefault('lw', 2.0)
        kwargs.setdefault('alpha', 0.5)

        hist = _none_dict(hist, 'hist', kwargs)
        scatter = _none_dict(scatter, 'scatter', kwargs)
        contour = _none_dict(contour, 'contour', kwargs)

        if mask_dense is None:
            mask_dense = (hist is not None) and (scatter is not None)

        # Draw Scatter Points
        # -------------------------------
        if scatter is not None:
            cls._scatter2d(ax, *data, **scatter)

        # Draw Median Lines (Target Style)
        # -----------------------------------------
        if median:
            for dd, func in zip(data, [ax.axvline, ax.axhline]):
                if weights is None:
                    med = np.median(dd)
                else:
                    med = utils.quantiles(dd, percs=0.5, weights=weights)

                func(med, color=color, ls='-', alpha=0.25, lw=1.0, path_effects=_OUTLINE)

        hh, *_ = np.histogram2d(*data, bins=edges, weights=weights, density=True)
        # Pad Histogram for Smoother Contour Edges
        if pad:
            hh = np.pad(hh, 2, mode='constant', constant_values=hh.min())
            tf = np.arange(1, 3)  # [+1, +2]
            tr = - tf[::-1]    # [-2, -1]
            edges = [
                [ee[0] + tr * np.diff(ee[:2]), ee, ee[-1] + tf * np.diff(ee[-2:])]
                for ee in edges
            ]
            edges = [np.concatenate(ee) for ee in edges]

        xc, yc = [utils.midpoints(ee, axis=-1) for ee in edges]
        xc, yc = np.meshgrid(xc, yc, indexing='ij')
        _, levels, quantiles = _dfm_levels(hh, quantiles=quantiles)

        # Mask dense scatter-points
        if mask_dense:
            span = [levels.min(), hh.max()]
            ax.contourf(xc, yc, hh, span, cmap=_MASK_CMAP, antialiased=True)

        # Draw 2D Histogram
        # -------------------------------
        cont_col_rev = False
        if hist is not None:
            cont_col_rev = True
            if mask_sparse is True:
                mask_sparse = levels.min()
            cls._hist2d(ax, edges, hh, mask_below=mask_sparse, **hist)

        # Draw Contours
        # --------------------------------
        if (contour is not None):
            cls._contour2d(ax, xc, yc, hh, quantiles, reverse=cont_col_rev, **contour)

        return

    @classmethod
    def _hist1d(cls, ax, edges, hist, weights=None, renormalize=False,
                joints=True, nonzero=False, positive=False, rotate=False, **kwargs):

        # hist, edges = np.histogram(data, bins=edges, weights=weights, density=True)

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
        if renormalize:
            yval = yval / yval[np.isfinite(yval)].max()

        line, = ax.plot(xval, yval, **kwargs)

        return line

    @classmethod
    def _hist2d(cls, ax, edges, hist, mask_below=None, color=None, smap=None, **kwargs):
        cmap = kwargs.pop('cmap', None)
        smap, smap_is_log, uniform_color = _parse_smap(smap, color, cmap=cmap)
        if not isinstance(smap, mpl.cm.ScalarMappable):
            smap = _get_smap(hist, **smap)

        kwargs.setdefault('cmap', smap.cmap)
        kwargs.setdefault('norm', smap.norm)
        kwargs.setdefault('shading', 'auto')

        if (mask_below is not None) and (mask_below is not False):
            hist = np.ma.masked_less_equal(hist, mask_below)
        return ax.pcolormesh(*edges, hist.T, **kwargs)

    @classmethod
    def _contour1d(cls, ax, data, quantiles, weights=None, median=True, rotate=False,
                   color='k', alpha=0.1, **kwargs):
        kw = udict(dict(facecolor=color, alpha=alpha, edgecolor='none'), kwargs)

        # Calculate Cumulative Distribution Function
        if weights is None:
            data = np.sort(data)
            cdf = np.arange(data.size) / (data.size - 1)
        else:
            idx = np.argsort(data)
            data = data[idx]
            weights = weights[idx]
            cdf = np.cumsum(weights) / np.sum(weights)

        # Convert from standard-deviations to percentiles
        # percs = sp.stats.norm.cdf(sigmas)
        # Get both the lower (left) and upper (right) values
        qnts = np.append(1 - quantiles, quantiles)
        # Reshape to (sigmas, 2)
        locs = np.interp(qnts, cdf, data).reshape(2, len(quantiles)).T

        if median:
            mm = np.interp(0.5, cdf, data)
            line_func = ax.axhline if rotate else ax.axvline
            line_func(mm, ls='--', color=color, alpha=0.25)

        for lo, hi in locs:
            span_func = ax.axhspan if rotate else ax.axvspan
            handle = span_func(lo, hi, **kw)

        return handle

    @classmethod
    def _contour2d(cls, ax, xx, yy, hist, quantiles, smooth=None, upsample=2,
                   outline=True, colors=None, color=None, reverse=False, **kwargs):

        if (upsample is not None):
            xx = sp.ndimage.zoom(xx, upsample)
            yy = sp.ndimage.zoom(yy, upsample)
            hist = sp.ndimage.zoom(hist, upsample)
        if (smooth is not None):
            if upsample is not None:
                smooth *= upsample
            hist = sp.ndimage.filters.gaussian_filter(hist, smooth)

        _, levels, quantiles = _dfm_levels(hist, quantiles=quantiles)

        alpha = kwargs.setdefault('alpha', 0.8)
        if colors is None:
            # if color is not None:
            #     colors = color
            # else:
            smap, smap_is_log, uniform_color = _parse_smap(None, color, cmap=None)
            if not isinstance(smap, mpl.cm.ScalarMappable):
                smap = _get_smap(hist, **smap)

            colors = smap.to_rgba(levels)
            if reverse:
                colors = colors[::-1]

            # Set alpha (transparency)
            colors[:, 3] = alpha

        kwargs['colors'] = colors
        kwargs.setdefault('linewidths', kwargs.pop('lw', cls._LW))
        # kwargs['linewidths'] = cls._LW
        kwargs.setdefault('linestyles', kwargs.pop('ls', '-'))
        kwargs.setdefault('zorder', 10)
        cont = ax.contour(xx, yy, hist, levels=levels, **kwargs)
        if (outline is True):
            outline = _get_outline_effects(2*cls._LW, alpha=1 - np.sqrt(1 - alpha))
            plt.setp(cont.collections, path_effects=outline)
        elif (outline is not False):
            raise ValueError("`outline` must be either 'True' or 'False'!")

        return

    @classmethod
    def _scatter2d(cls, ax, xx, yy, color='k', alpha=0.1, s=4, **kwargs):
        kwargs.setdefault('facecolor', color)
        kwargs.setdefault('edgecolor', 'none')
        kwargs.setdefault('alpha', alpha)
        kwargs.setdefault('s', s)
        return ax.scatter(xx, yy, **kwargs)


def _figax(size, grid=True, left=None, bottom=None, right=None, top=None, hspace=None, wspace=None,
           **kwfig):

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


def corner(kde_data, labels=None, kwcorner={}, kwplot={}):
    corner = Corner(kde_data, labels=labels, **kwcorner)
    corner.plot(**kwplot)
    return corner


# ======  API KDEs Methods  ======
# ================================


def corner_kde(axes, kde, edges=None, reflect=None, sigmas=None, levels=None, rotate=None,
               median=True, hist2d=True, contour=None, contour1d=True, contour2d=True,
               smap=None, cmap=None, renormalize=False, **kwargs):

    shp = np.shape(axes)
    if (np.ndim(axes) != 2) or (shp[0] != shp[1]):
        raise ValueError("`axes` (shape: {}) must be an NxN arrays!".format(shp))

    size = shp[0]
    last = size - 1
    # if reflect is None:
    #     reflect = [None] * size

    if rotate is None:
        rotate == (size == 2)

    if edges is None:
        edges = kde.points

    if contour is not None:
        contour1d = contour
        contour2d = contour

    color = kwargs.setdefault('color', 'k')
    kwargs.setdefault('lw', 2.0)
    kwargs.setdefault('alpha', 0.8)

    # pdf = kde.pdf_grid(edges, reflect=reflect)
    extr = [utils.minmax(ee, stretch=0.1) for ee in edges]
    _, smap_is_log, _ = _parse_smap(smap, color, cmap=cmap)

    #
    # Calculate Distributions
    # ================================
    #

    pdf1d = np.full(size, None, dtype=object)
    pdf2d = np.full(shp, None, dtype=object)
    extr_hist2d = None
    for (ii, jj), ax in np.ndenumerate(axes):
        if jj > ii:
            continue

        # Diagonals
        # ----------------------
        if ii == jj:
            _, pdf1d[jj] = kde.density(edges[jj], params=jj, reflect=reflect)

        # Off-Diagonals
        # ----------------------
        else:
            _, pdf2d[jj, ii] = kde.density([edges[jj], edges[ii]], params=[jj, ii],
                                           reflect=reflect, grid=True)
            extr_hist2d = utils.minmax(pdf2d[jj, ii], prev=extr_hist2d, positive=smap_is_log)

    _set_corner_axes_extrema(axes, extr, rotate)

    #
    # Draw / Plot Data
    # ===========================
    #

    # Draw 1D Histograms & Carpets
    # -----------------------------------------
    for jj, ax in enumerate(axes.diagonal()):
        rot = (rotate and (jj == last))
        handle1d = dist1d_kde(
            ax, kde, pdf=pdf1d[jj], param=jj, reflect=reflect, rotate=rot,
            sigmas=sigmas, median=median, contour=contour1d, renormalize=renormalize, **kwargs)

    # Draw 2D Histograms and Contours
    # -----------------------------------------
    # _smap = _get_smap(extr_hist2d, **smap)

    for (ii, jj), ax in np.ndenumerate(axes):
        if jj >= ii:
            continue

        # _smap = _get_smap(pdf2d[jj, ii], **smap)
        handle2d = dist2d_kde(
            ax, kde, params=(jj, ii), pdf=pdf2d[jj, ii], reflect=reflect,
            sigmas=sigmas, smap=smap, cmap=cmap,
            median=median, hist2d=hist2d, contour=contour2d, **kwargs)

    return handle1d, handle2d


def dist1d_kde(ax, kde, param=None, pdf=None, reflect=None, edges=None, sigmas=True,
               contour=True, median=True, rotate=False, renormalize=False, **kwargs):

    if (param is None):
        if kde.ndim > 1:
            raise ValueError("`kde` has {} dimensions, `param` required!".format(kde.ndim))

        param = 0

    edges = kde.points
    if (not utils.really1d(edges)) or (param > 0):
        edges = edges[param]

    if pdf is None:
        _, pdf = kde.density(edges, reflect=reflect, params=param)

    if renormalize:
        pdf = pdf / pdf.max()

    color = kwargs.setdefault('color', 'k')
    kwargs.setdefault('lw', 2.0)
    kwargs.setdefault('alpha', 0.8)
    contour = _none_dict(contour, 'contour', kwargs)

    vals = [edges, pdf]
    if rotate:
        vals = vals[::-1]
    handle_density = ax.plot(*vals, **kwargs)

    handle_contour = None
    if (contour is not None) or median:
        sigmas = _get_def_sigmas(sigmas, contour=contour, median=median)
        # Convert from standard-deviations to percentiles
        percs = sp.stats.norm.cdf(sigmas)
        # NOTE: currently `kde.cdf` does not work with `params`... once it does, use that!
        cdf = utils.cumtrapz(pdf, edges)
        # Normalize to the maximum value
        cdf /= cdf.max()
        # Get both the lower (left) and upper (right) values
        percs = np.append(1 - percs, percs)
        # Reshape to (sigmas, 2)
        locs = np.interp(percs, cdf, edges).reshape(2, len(sigmas)).T

        # handle_contour = _draw_contours_1d(ax, locs, rotate=rotate, **contour)
        handle_contour = Corner._contour1d(ax, locs, rotate=rotate, color=color)

    handles = dict()
    handles['density'] = handle_density
    handles['contour'] = handle_contour

    return handles


def dist2d_kde(ax, kde, params=None, pdf=None, reflect=None, smap=None, cmap=None,
               hist2d=True, contour=True, sigmas=None, median=True, **kwargs):

    if (params is None):
        if kde.ndim > 2:
            raise ValueError("`kde` has {} dimensions, `params` required!".format(kde.ndim))

        params = (0, 1)

    kwargs.setdefault('color', 'k')
    kwargs.setdefault('lw', 2.0)
    kwargs.setdefault('alpha', 0.8)
    hist2d = _none_dict(hist2d, 'hist2d', kwargs)
    contour = _none_dict(contour, 'contour', kwargs)

    edges = kde.points
    edges = [edges[pp] for pp in params]

    if pdf is None:
        _, pdf = kde.density(edges, params=params, reflect=reflect, grid=True)
    xx, yy = np.meshgrid(*edges, indexing='ij')

    sigmas, pdf_levels, _levels = _dfm_levels(pdf, sigmas=sigmas)

    # smap, smap_is_log, uniform_color = _parse_smap(smap, color, cmap=cmap)
    # if not isinstance(smap, mpl.cm.ScalarMappable):
    #     smap = _get_smap(pdf, **smap)

    # Draw 2D Histogram
    # -------------------------------
    handle_hist = None
    if hist2d is not None:
        if smap is not None:
            hist2d.setdefault('cmap', smap.cmap)
            hist2d.setdefault('norm', smap.norm)
        handle_hist = _draw_hist2d(ax, *edges, pdf, **hist2d)

    # Draw Median Lines (Target Style)
    # -----------------------------------------
    if median:
        # effects = ([
        #     mpl.patheffects.Stroke(linewidth=2.0, foreground='0.75', alpha=0.75),
        #     mpl.patheffects.Normal()
        # ])

        for ii, func in enumerate([ax.axvline, ax.axhline]):
            pp = params[ii]
            ee = edges[ii]
            _, cdf = kde.density(ee, reflect=reflect, params=pp)
            cdf = utils.cumtrapz(cdf, ee)
            cdf /= cdf.max()
            cdf = np.interp(0.5, cdf, ee)
            func(cdf, color=kwargs['color'], ls='-', alpha=0.5, lw=1.0, path_effects=_OUTLINE)

    # Draw Contours
    # --------------------------------
    handle_contour = None
    if contour is not None:
        handle_contour = _draw_contours_2d(ax, xx, yy, pdf, smap, sigmas=sigmas, **contour)

    handles = dict()
    handles['hist'] = handle_hist
    handles['contour'] = handle_contour

    return handles


def _get_def_sigmas(sigmas, contour=True, median=True):
    if (sigmas is False) or (contour is False):
        sigmas = []
    elif (sigmas is True) or (sigmas is None):
        sigmas = _DEF_SIGMAS

    if median:
        sigmas = np.append([0.0], sigmas)

    return sigmas


# ======  Drawing Methods  =====
# ==============================


def carpet(xx, weights=None, ax=None, ystd=None, yave=None, shift=0.0,
           fancy=False, random='normal', rotate=False, **kwargs):
    """Draw a carpet plot on the given axis in the 'fuzz' style.

    Arguments
    ---------
    xx : values to plot
    ax : matplotlib.axis.Axis
    kwargs : key-value pairs
        Passed to `matplotlib.axes.Axes.scatter()`

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
    # NOTE: array values dont work for alpha parameters (added to `colors`)
    if alpha is None:
        aa = 10 / np.sqrt(xx.size)
        alpha = aa
        # alpha = aa * ww
        # alpha = np.clip(alpha, aa/10, aa*10)
        # alpha = np.clip(alpha, 1e-4, 1e-1)

    # Choose sizes proportional to their deviation (to make outliers more visible)
    size = 300 * ww / np.sqrt(xx.size)
    size = np.clip(size, 5, 100)

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
    color = kwargs.pop('color', 'red')
    # NOTE: array values dont work for facecolor parameters
    # colors = np.array(mpl.colors.to_rgba(color))[np.newaxis, :] * np.ones(xx.size)[:, np.newaxis]
    # colors[:, -1] = alpha

    kwargs.setdefault('facecolor', color)
    kwargs.setdefault('edgecolor', 'none')
    kwargs.setdefault('marker', '.')
    kwargs.setdefault('alpha', alpha)
    kwargs.setdefault('s', size)

    extr = utils.minmax(yy)
    trans = [ax.transData, ax.transAxes]
    if shift is not None:
        yy += shift

    if rotate:
        temp = xx
        xx = yy
        yy = temp
        trans = trans[::-1]

    return ax.scatter(xx, yy, **kwargs), extr


def _set_corner_axes_extrema(axes, extrema, rotate, pdf=None):
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


def hist(data, bins=None, ax=None, weights=None, density=True, probability=False, **kwargs):
    if ax is None:
        ax = plt.gca()

    hist, edges = utils.histogram(data, bins=bins, weights=weights,
                                  density=density, probability=probability)

    return hist, edges, Corner._hist1d(ax, edges, hist, **kwargs)


# ====  Utility Methods  ====
# ===========================


def nbshow():
    return utils.run_if_notebook(plt.show, otherwise=lambda: plt.close('all'))


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


def _none_dict(val, name, defaults={}):
    """

    False/None  ===>  None
    True/dict   ===>  dict
    otherwise   ===>  error

    """
    if (val is False) or (val is None):
        return None

    if val is True:
        val = dict()

    if not isinstance(val, dict):
        raise ValueError("Unrecognized type '{}' for `{}`!".format(type(val), name))

    for kk, vv in defaults.items():
        val.setdefault(kk, vv)

    return val


def _dfm_levels(data, quantiles=None, sigmas=None):
    if quantiles is None:
        if sigmas is None:
            sigmas = _DEF_SIGMAS
        # Convert from standard-deviations to CDF values
        quantiles = 1.0 - np.exp(-0.5 * np.square(sigmas))
    elif sigmas is None:
        sigmas = np.sqrt(-2.0 * np.log(1.0 - quantiles))

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


def udict(*args, copy_vals=True):
    """Create a new, 'updated' dictionary with updates from the given dictionaries
    """
    import copy
    rv = dict()
    for aa in args:
        if copy_vals:
            rv.update({kk: copy.copy(vv) for kk, vv in aa.items()})
        else:
            rv.update(aa)
    return rv


def _draw_carpet(*args, **kwargs):
    return carpet(*args, **kwargs)


def _get_outline_effects(lw=2.0, fg='0.75', alpha=0.8):
    outline = ([
        mpl.patheffects.Stroke(linewidth=lw, foreground=fg, alpha=alpha),
        mpl.patheffects.Normal()
    ])
    return outline


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
