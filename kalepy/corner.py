"""
"""

import warnings

import numpy as np
import scipy as sp
import scipy.stats  # noqa
import matplotlib as mpl
import matplotlib.pyplot as plt

from kalepy import plot, utils

_STRETCH = 0.2
_COLOR_CMAP = {
    'k': 'Greys',
    'b': 'Blues',
    'r': 'Reds',
}
_DEF_SIGMAS = np.arange(0.5, 2.1, 0.5)


class Corner:

    def __init__(self, size=None, data=None, kde=None, labels=None, **figax_kwargs):
        if data is None:
            data = kde.dataset
        if size is None:
            size = len(data)

        last = size - 1
        figax_kwargs.setdefault('figsize', [6, 6])
        figax_kwargs.setdefault('hspace', 0.1)
        figax_kwargs.setdefault('wspace', 0.1)
        fig, axes = figax(nrows=size, ncols=size, **figax_kwargs)
        self.fig = fig
        self.axes = axes
        if labels is None:
            labels = [''] * size

        for (ii, jj), ax in np.ndenumerate(axes):
            if jj > ii:
                ax.set_visible(False)
                continue
            ax.grid(True)

            if ii == last:
                ax.set_xlabel(labels[jj])
            else:
                ax.set_xlabel('')
                for tlab in ax.xaxis.get_ticklabels():
                    tlab.set_visible(False)

            if jj == 0:
                ax.set_ylabel(labels[ii])
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

            self.size = size
            self._data = data
            self._kde = kde

        return

    def data(self, axes=None, data=None, **kwargs):
        if axes is None:
            axes = self.axes
        if data is None:
            data = self._data

        return corner_data(axes, data, fig=self.fig, **kwargs)


def figax(figsize=[12, 6], nrows=1, ncols=1, scale='linear',
          xlabel='', xlim=None, xscale=None,
          ylabel='', ylim=None, yscale=None,
          left=None, bottom=None, right=None, top=None, hspace=None, wspace=None,
          grid=True, squeeze=True, **kwargs):

    if scale is not None:
        xscale = scale
        yscale = scale

    fig, axes = plt.subplots(figsize=figsize, squeeze=False, ncols=ncols, nrows=nrows,
                             **kwargs)

    plt.subplots_adjust(
        left=left, bottom=bottom, right=right, top=top, hspace=hspace, wspace=wspace)

    if ylim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(ylim) == (2,):
            ylim = np.array(ylim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols,)

    ylim = np.broadcast_to(ylim, shape)

    if xlim is not None:
        shape = (nrows, ncols, 2)
        if np.shape(xlim) == (2,):
            xlim = np.array(xlim)[np.newaxis, np.newaxis, :]
    else:
        shape = (nrows, ncols)

    xlim = np.broadcast_to(xlim, shape)

    _, xscale, xlabel = np.broadcast_arrays(axes, xscale, xlabel)
    _, yscale, ylabel = np.broadcast_arrays(axes, yscale, ylabel)

    for idx, ax in np.ndenumerate(axes):
        ax.set(xscale=xscale[idx], xlabel=xlabel[idx],
               yscale=yscale[idx], ylabel=ylabel[idx])
        if xlim[idx] is not None:
            ax.set_xlim(xlim[idx])
        if ylim[idx] is not None:
            ax.set_ylim(ylim[idx])

        if grid is not None:
            if grid in [True, False]:
                ax.grid(grid)
            else:
                ax.grid(True, **grid)

    if squeeze:
        axes = np.squeeze(axes)
        if np.ndim(axes) == 0:
            axes = axes[()]

    return fig, axes


def corner_data(axes, data, edges=None, levels=None, hist=None, pad=True, rotate=True,
                mask_dense=True, mask_sparse=True, median=True,
                hist1d=True, hist2d=True, scatter=True, carpet=True, contour=True,
                smap=None, color='k', cmap=None, fig=None):

    shp = np.shape(axes)
    if (np.ndim(axes) != 2) or (shp[0] != shp[1]):
        raise ValueError("`axes` (shape: {}) must be an NxN arrays!".format(shp))

    size = shp[0]
    last = size - 1
    if np.ndim(data) != 2 or np.shape(data)[0] != size:
        err = "`data` (shape: {}) must be 2D with shape (parameters, data-points)!".format(
            np.shape(data))
        raise ValueError(err)

    if hist is not None:
        hist1d = hist
        hist2d = hist

    if (hist1d is not None) or (hist2d is not None):
        edges = _parse_edges(edges, data)

    extr = [utils.minmax(dd) for dd in data]

    if carpet and not hist1d:
        warnings.warn("Cannot `carpet` without `hist1d`!")
        carpet = False

    if smap is None:
        smap = {}

    if not isinstance(smap, dict) and not isinstance(smap, mpl.cm.ScalarMappable):
        raise ValueError("`smap` must either be a dict or ScalarMappable!")

    smap_is_log = smap.setdefault('log', False)
    if cmap is None:
        cmap = _COLOR_CMAP.get(color, 'Greys')

    smap.setdefault('cmap', cmap)

    # Convert to `None` for False, and `dict` for True
    hist1d = _none_dict(hist1d, 'hist1d', dict(color=color))
    hist2d = _none_dict(hist2d, 'hist2d')
    carpet = _none_dict(carpet, 'carpet', dict(color=color))
    scatter = _none_dict(scatter, 'scatter', dict(color=color))
    contour = _none_dict(contour, 'contour')

    if mask_dense:
        mask_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            "white_cmap", [(1, 1, 1), (1, 1, 1)], N=2)

    data_hist1d = np.empty(size, dtype=object)
    data_hist2d = np.empty(shp, dtype=object)
    extr_hist2d = None
    extr_hist1d = None
    for (ii, jj), ax in np.ndenumerate(axes):
        if jj > ii:
            continue

        # Diagonals
        # ----------------------
        xx = data[jj]
        if ii == jj:
            if hist1d is not None:
                data_hist1d[jj], _ = np.histogram(xx, bins=edges[jj], density=True)
                extr_hist1d = utils.minmax(data_hist1d[jj], stretch=_STRETCH, prev=extr_hist1d)

            rot = (rotate and (jj == last))
            if not rot:
                ax.set_xlim(extr[jj])
            else:
                ax.set_ylim(extr[jj])

        # Off-Diagonals
        # ----------------------
        else:
            yy = data[ii]
            if scatter is not None:
                _draw_scatter(ax, xx, yy, **scatter)
            if (hist2d is not None) or (contour is not None):
                bins = [edges[jj], edges[ii]]
                # NOTE/WARNING: `density` should be an option here!!
                data_hist2d[jj, ii], *_ = np.histogram2d(xx, yy, bins=bins, density=False)
                extr_hist2d = utils.minmax(
                    data_hist2d[jj, ii], prev=extr_hist2d, positive=smap_is_log)

            if median is not None:
                effects = ([
                    mpl.patheffects.Stroke(linewidth=2.0, foreground='0.75', alpha=0.75),
                    mpl.patheffects.Normal()
                ])

                for dd, func in zip([xx, yy], [ax.axvline, ax.axhline]):
                    med = np.median(dd)
                    func(med, color=color, ls='-', alpha=0.5, lw=1.0, path_effects=effects)

            ax.set_xlim(extr[jj])
            ax.set_ylim(extr[ii])

    # Draw 1D Histograms & Carpets
    # ===========================================
    if hist1d is not None:
        for jj, ax in enumerate(axes.diagonal()):
            rot = (rotate and (jj == last))
            _draw_hist1d(ax, edges[jj], hist=data_hist1d[jj], data=data[jj],
                         rotate=rot, span_max=extr_hist1d[1], **hist1d)

        # Draw Carpets
        if carpet is not None:
            ystd = extr_hist1d[1] * 0.02
            for jj, ax in enumerate(axes.diagonal()):
                rot = (rotate and (jj == last))
                ystd = carpet.pop('ystd', ystd)
                _, _extr = plot.draw_carpet_fuzz(data[jj], ax=ax, ystd=ystd, rotate=rot, **carpet)
                extr_hist1d = utils.minmax(_extr, stretch=_STRETCH, prev=extr_hist1d)

        for jj, ax in enumerate(axes.diagonal()):
            rot = (rotate and (jj == last))
            if rot:
                ax.set_xlim(extr_hist1d)
            else:
                ax.set_ylim(extr_hist1d)

    # Draw 2D Histograms and Contours
    # ============================================
    if (hist2d is not None) or (contour is not None):
        smap = plot.smap(extr_hist2d, **smap)

        for (ii, jj), ax in np.ndenumerate(axes):
            if jj >= ii:
                continue

            dh2 = data_hist2d[jj, ii]
            e2d = [edges[jj], edges[ii]]

            if pad:
                dh2 = np.pad(dh2, 2, mode='edge')
                tf = np.arange(2)  # [+1, +2]
                tr = - tf[::-1]    # [-2, -1]
                e2d = [
                    [ee[0] + tr * np.diff(ee[:2]), ee, ee[-1] + tf * np.diff(ee[-2:])]
                    for ee in e2d
                ]
                e2d = [np.concatenate(ee) for ee in e2d]

            xe, ye = e2d
            xc, yc = [utils.midpoints(ee, axis=-1) for ee in [xe, ye]]
            xc, yc = np.meshgrid(xc, yc, indexing='ij')

            # Mask dense scatter-points
            if mask_dense and (scatter is not None):
                pdf_levels, _levels = _dfm_levels(dh2, levels=levels)
                ax.contourf(xc, yc, dh2, [pdf_levels.min(), dh2.max()],
                            cmap=mask_cmap, antialiased=False)

            if hist2d is not None:
                hist2d.setdefault('cmap', smap.cmap)
                hist2d.setdefault('norm', smap.norm)
                if mask_sparse is True:
                    mask_sparse = pdf_levels.min()
                _draw_hist2d(ax, xe, ye, dh2, mask_below=mask_sparse, **hist2d)

            if contour is not None:
                _draw_contours(ax, xc, yc, dh2, smap, **contour)

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

    return


def _draw_scatter(ax, xx, yy, color='k', alpha=0.1, s=4, **kwargs):
    kwargs.setdefault('color', color)
    # kwargs.setdefault('alpha', 0.03)
    kwargs.setdefault('alpha', alpha)
    kwargs.setdefault('s', s)
    # kwargs.setdefault('zorder', 10)
    return ax.scatter(xx, yy, **kwargs)


def _draw_hist2d(ax, xx, yy, data, mask_below=None, **kwargs):
    if (mask_below is not None) and (mask_below is not False):
        data = np.ma.masked_less_equal(data, mask_below)
    ax.pcolormesh(xx, yy, data.T, **kwargs)
    return


def _draw_contours(ax, xx, yy, hist_data, smap,
                   linewidths=1.0, alpha=0.8, colors=None, zorder=20,
                   background=True, levels=None, **kwargs):
    pdf_levels, levels = _dfm_levels(hist_data, levels=levels)
    lw = kwargs.pop('lw', None)
    bg_alpha = alpha  # /2
    if lw is not None:
        linewidths = lw
    kwargs['levels'] = pdf_levels
    background = _none_dict(background, 'background', defaults=kwargs)

    if colors is None:
        colors_bg = smap.to_rgba(pdf_levels)
        colors = colors_bg[::-1]

        # colors_bg = [0.8] * 3 + [bg_alpha]
        # colors = smap.to_rgba(pdf_levels)

        # colors_bg = smap.to_rgba(pdf_levels)
        # colors = np.array([_invert_color(cc) for cc in colors_bg])

        # Set alpha (transparency)
        colors[:, 3] = alpha
        colors_bg[:, 3] = bg_alpha
    else:
        colors_bg = ['0.5'] * 3 + [bg_alpha]

    background.setdefault('colors', colors_bg)
    background.setdefault('linewidths', 2*linewidths)
    background.setdefault('zorder', zorder - 1)

    kwargs['colors'] = colors
    kwargs['linewidths'] = linewidths
    kwargs['zorder'] = zorder

    if background:
        ax.contour(xx, yy, hist_data, **background)

    ax.contour(xx, yy, hist_data, **kwargs)

    return


def _draw_hist1d(ax, edges, hist=None, data=None, joints=False, median=True, sigmas=True,
                 nonzero=False, positive=False, extend=None, span_max=None,
                 rotate=False, **kwargs):

    if hist is None:
        hist, _ = np.histogram(data, bins=edges, density=True)

    # yerr_fmt = '+'
    color = kwargs.setdefault('color', 'k')

    # Extend bin edges if needed
    if len(edges) != len(hist)+1:
        raise RuntimeError("``edges`` must have length hist+1!")

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
        line_func = ax.axhline
    else:
        line_func = ax.axvline

    # Plot Histogram
    line, = ax.plot(xval, yval, **kwargs)

    if ((sigmas is not None) and (sigmas is not False)) or median:
        if sigmas is True:
            sigmas = _DEF_SIGMAS
        elif (sigmas is None) or (sigmas is False):
            sigmas = []

        if median:
            sigmas = np.append([0.0], sigmas)

        # Calculate Cumulative Distribution Function
        data = np.sort(data)
        cdf = np.arange(data.size) / (data.size - 1)
        # Convert from standard-deviations to percentiles
        percs = sp.stats.norm.cdf(sigmas)
        # Get both the lower (left) and upper (right) values
        percs = np.append(1 - percs, percs)
        # Reshape to (sigmas, 2)
        locs = np.interp(percs, cdf, data).reshape(2, len(sigmas)).T

        if span_max is None:
            span_max = ax.get_ylim()[1]
        pc = []
        for ss, (lo, hi) in zip(sigmas, locs):
            # If median is included (sigma=0, cdf=0.5), lo will equal hi
            if lo == hi:
                line_func(np.median(data), ls='--', color=color, alpha=0.5)

            center = [lo, 0.0]
            span = [hi - lo, span_max]
            if rotate:
                center = center[::-1]
                span = span[::-1]
            rect = mpl.patches.Rectangle(center, *span)
            pc.append(rect)

        # Create patch collection with specified colour/alpha
        pc = mpl.collections.PatchCollection(
            pc, facecolor=color, alpha=0.1, edgecolor='none')

        # Add collection to axes
        ax.add_collection(pc)

    return line


def _parse_edges(edges, data, num=None):
    shp, pnts = np.shape(data)
    is_scalar = (edges is None) or np.isscalar(edges)
    is_nbins = ((np.shape(edges) == (shp,)) and utils.really1d(edges))
    if is_scalar or is_nbins:
        # If a single value is given (`None` or an integer number of bins) convert to list
        if is_scalar:
            if edges is None:
                edges = np.min([40, np.sqrt(pnts)]) if num is None else num

            edges = [edges for ii in range(shp)]

        edges = [utils.spacing(data[ii], 'lin', edges[ii]) for ii in range(shp)]

    if not np.all([np.ndim(hh) == 1 and np.size(hh) > 1 for hh in edges]):
        raise ValueError("`hist` is invalid for {} parameters!".format(shp))

    return edges


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


def _dfm_levels(hist, levels=None):
    if levels is None:
        # sigmas = [1, 2, 3]
        sigmas = _DEF_SIGMAS
        # Convert from standard-deviations to CDF values
        # levels = 1.0 - 2 * (1.0 - sp.stats.norm.cdf(sigmas))
        levels = 1.0 - np.exp(-0.5 * np.square(sigmas))

    # Compute the density levels.
    hflat = hist.flatten()
    inds = np.argsort(hflat)[::-1]
    hflat = hflat[inds]
    sm = np.cumsum(hflat)
    sm /= sm[-1]
    pdf_levels = np.empty(len(levels))
    for i, v0 in enumerate(levels):
        try:
            pdf_levels[i] = hflat[sm <= v0][-1]
        except:
            pdf_levels[i] = hflat[0]

    pdf_levels.sort()
    bad = (np.diff(pdf_levels) == 0)
    bad = np.pad(bad, [1, 0], constant_values=False)

    # -- Remove Bad Levels
    pdf_levels = np.delete(pdf_levels, np.where(bad)[0])
    if np.any(bad):
        _levels = levels
        levels = np.array(levels)[~bad]
        print("Removed bad levels: '{}' ==> '{}'".format(_levels, levels))

    # -- Adjust Bad Levels:
    # if np.any(bad) and not quiet:
    #     logging.warning("Too few points to create valid contours")
    # while np.any(bad):
    #     V[np.where(bad)[0][0]] *= 1.0 - 1e-4
    #     m = np.diff(V) == 0
    pdf_levels.sort()
    return pdf_levels, levels


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


def _invert_color(col):
    rgba = mpl.colors.to_rgba(col)
    alpha = rgba[-1]
    col = 1.0 - np.array(rgba[:-1])
    col = tuple(col.tolist() + [alpha])
    return col
