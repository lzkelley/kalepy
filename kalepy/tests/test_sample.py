"""

Can be run with:
    $ nosetests tests/test_sample.py

"""

import numpy as np
# import scipy as sp
from numpy.testing import run_module_suite
from nose import tools

import kalepy as kale
from kalepy import sample


class Test_Sampler_Grid:

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)
        return

    def test_shapes_valid(self):
        # Number of dimensions for distribution
        for ndim in range(1, 5):
            # construct random shape with `ndim` dimensions
            shape_edges = np.random.randint(2, 6, ndim)
            print(f"{ndim=}, {shape_edges=}")
            # choose random extr for each dimension of edges
            extr = [np.sort(np.random.uniform(-100, 100, 2)) for ii in range(ndim)]
            # calculate edges with linspacing
            edges = [np.linspace(*ex, nn) for ex, nn in zip(extr, shape_edges)]
            # choose a number of samples
            nsamp = np.random.randint(100, 1000)

            # calculate shape of bin centers
            # shape_cents = [sh - 1 for sh in shape_edges]
            # for shape in [shape_edges, shape_cents]:
            for shape in [shape_edges]:
                # construct a distribution
                data = np.random.uniform(0.0, 1.0, shape)
                # try both values of interpolation flag
                for interp in [True, False]:
                    print(f"{shape=}, {interp=}")
                    sample.sample_grid(edges, data, nsamp, interpolate=interp)

        return

    def test_shapes_invalid(self):

        # different combinations of edge-shapes and data-shapes that are inconsistent
        #    they should each lead to an error being raised
        shapes = [
            [(4, 5, 3), (4, 5, 3, 4)],
            [(4, 5, 3), (4, 5, 4)],
            [(4, 5), (3, 6)],
            [(4, 5), (2, 4, 5)],
        ]

        for aa, bb in shapes:
            edges = [np.linspace(0.0, 1.0, ai) for ai in aa]
            data = np.random.uniform(100.0, 200.0, bb)
            # try both values of interpolation flag
            for interp in [True, False]:
                with tools.assert_raises(ValueError):
                    sample.sample_grid(edges, data, 100)

        return


class Test_Sample_Outliers:

    _NUM = 1234

    @classmethod
    def setup_class(cls):
        np.random.seed(9865232)
        return

    def _func(self, xx):
        zz = np.power(xx, +1.5) * np.exp(-xx)
        return zz

    def _get_outliers(self, nsamp, threshold, nsamp_by_mass, poisson_inside):
        nbins = 100
        extr = [1e-2, 1e1]

        xx = kale.utils.spacing(extr, scale='log', num=nbins)
        yy = self._func(xx)
        Y = np.cumsum(yy)
        norm = self._NUM / Y[-1]
        yy *= norm
        Y *= norm
        dydx = np.diff(Y) / np.diff(xx)
        xc = kale.utils.midpoints(xx)

        ss, ww = kale.sample_outliers(
            xc, dydx, threshold,
            nsamp_by_mass=nsamp_by_mass, poisson_inside=poisson_inside, nsamp=nsamp
        )
        wdist, _ = np.histogram(ss, bins=xx, weights=ww)
        return xx, yy, ss, ww, xc, wdist

    def test_nsamp_by_mass_yes(self):
        nsamp_by_mass = True
        poisson_inside = False
        threshold = 10.0
        NSAMP = int(2.0 * self._NUM)   # make sure this isn't actually equatl to `_NUM`
        xx, yy, ss, ww, xc, wdist = self._get_outliers(NSAMP, threshold, nsamp_by_mass, poisson_inside)

        assert ss.size == ww.size, f"Mismatch between size of weights ({ww.size}) and samples ({ss.size})!"

        # for `nsamp_by_mass == True`, the actual number of samples generally is not `nsamp`
        assert ss.size != NSAMP, f"Number of samples ({ss.size}) should not equal NSAMP ({NSAMP})!"

        # for `nsamp_by_mass == True`, the total weight of samples must be nearly equal to `nsamp`
        diff = np.fabs(np.sum(ww) - NSAMP)
        assert diff <= 1.0, "Sum of weights ({np.sum(ww)}) is inconsistent with `NSAMP` ({NSAMP})!"

        # for `nsamp_by_mass == True`, lowest weights should be unity
        extr = kale.utils.minmax(ww)
        assert (extr[0] == 1.0) & (extr[1] > 1.0), f"weights extrema are not bounded correctly!  ({extr})"

        yc = kale.utils.midpoints(yy)
        idx = (yc > 1.2*threshold)   # multiple by a buffer factor to avoid the edges of the region
        diff = (wdist[idx] - yc[idx]) / yc[idx]
        diff = np.fabs(diff)
        print("diff = ", diff)
        print("diff = ", kale.utils.stats_str(diff, format='.2e'))
        percs = np.percentile(diff, [25, 50, 75])
        assert percs[0] < 1e-3, f"25% error is too innaccurate!  {percs[0]}"
        assert percs[1] < 1e-3, f"Median is too innaccurate!  {percs[0]}"
        # Because this portion of the curve is exponential, it's not fit as well by the centroids
        # so allow for a larger error here
        assert percs[2] < 0.5, f"75% error is too innaccurate!  {percs[0]}"

    def test_nsamp_by_mass_no(self):
        nsamp_by_mass = False
        poisson_inside = False
        threshold = 10.0
        NSAMP = int(2.0 * self._NUM)   # make sure this isn't actually equatl to `_NUM`
        xx, yy, ss, ww, xc, wdist = self._get_outliers(NSAMP, threshold, nsamp_by_mass, poisson_inside)

        assert ss.size == ww.size, f"Mismatch between size of weights ({ww.size}) and samples ({ss.size})!"

        # for `nsamp_by_mass == False`, the actual number of samples must be exactly `nsamp`
        assert ss.size == NSAMP, f"Number of samples ({ss.size}) should equal NSAMP ({NSAMP})!"

        # for `nsamp_by_mass == False`, the total weight of samples should generally not equal `nsamp`
        diff = np.fabs(np.sum(ww) - NSAMP)
        assert diff > 1.0, f"Sum of weights ({np.sum(ww)}) is inconsistent with `NSAMP` ({NSAMP})!"

        # for `nsamp_by_mass == False`, there should be weights both above and below zero
        extr = kale.utils.minmax(ww)
        assert (extr[0] < 1.0) & (extr[1] > 1.0), f"weights extrema do not bound unity!  ({extr})"

        yc = kale.utils.midpoints(yy)
        idx = (yc > 1.2*threshold)   # multiple by a buffer factor to avoid the edges of the region
        diff = (wdist[idx] - yc[idx]) / yc[idx]
        diff = np.fabs(diff)
        print("diff = ", diff)
        print("diff = ", kale.utils.stats_str(diff, format='.2e'))
        percs = np.percentile(diff, [25, 50, 75])
        assert percs[0] < 1e-3, f"25% error is too innaccurate!  {percs[0]}"
        assert percs[1] < 1e-3, f"Median is too innaccurate!  {percs[0]}"
        # Because this portion of the curve is exponential, it's not fit as well by the centroids
        # so allow for a larger error here
        assert percs[2] < 0.5, f"75% error is too innaccurate!  {percs[0]}"


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
