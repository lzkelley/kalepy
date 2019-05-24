"""

Can be run with:
    $ nosetests tests/test_kde.py

"""

import numpy as np
import scipy as sp
import scipy.stats  # noqa
from numpy.testing import run_module_suite
from nose.tools import assert_true, assert_false

import kalepy as kale
from kalepy import utils


class Test_KDE_PDF(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    def compare_scipy_1d(self, kernel):
        print("\n|Test_KDE_PDF:test_compare_scipy_1d()|")
        NUM = 100
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bins = utils.spacing([-1, 14.0], 'lin', 40)
        grid = utils.spacing(bins, 'lin', 3000)

        methods = ['scott', 0.04, 0.2, 0.8]
        classes = [lambda xx, bw: sp.stats.gaussian_kde(xx, bw_method=bw),
                   lambda xx, bw: kale.KDE(xx, bandwidth=bw, kernel=kernel)]
        for mm in methods:
            kde_list = [cc(aa, mm).pdf(grid) for cc in classes]
            print("method: {}".format(mm))
            print("\t" + utils.stats_str(kde_list[0]))
            print("\t" + utils.stats_str(kde_list[1]))
            assert_true(np.allclose(kde_list[0], kde_list[1]))

        return

    def compare_scipy_2d(self, kernel):
        print("\n|Test_KDE_PDF:test_compare_scipy_2d()|")

        NUM = 1000
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bb = np.random.normal(3.0, 0.02, NUM) + aa/100

        data = [aa, bb]
        edges = [utils.spacing(dd, 'lin', 30, stretch=0.5) for dd in data]
        cents = [utils.midpoints(ee, 'lin') for ee in edges]

        xe, ye = np.meshgrid(*edges)
        xc, yc = np.meshgrid(*cents)
        grid = np.vstack([xc.ravel(), yc.ravel()])

        methods = ['scott', 0.04, 0.2, 0.8]
        # classes = [sp.stats.gaussian_kde, kale.KDE]
        classes = [lambda xx, bw: sp.stats.gaussian_kde(xx, bw_method=bw),
                   lambda xx, bw: kale.KDE(xx, bandwidth=bw, kernel=kernel)]
        for mm in methods:
            kdes_list = [cc(data, mm).pdf(grid).reshape(xc.shape).T
                         for cc in classes]
            assert_true(np.allclose(kdes_list[0], kdes_list[1]))

        return

    def reflect_1d(self, kernel):
        print("\n|Test_KDE_PDF:reflect_1d()|")

        np.random.seed(124)
        NUM = 1000
        EXTR = [0.0, 2.0]
        aa = np.random.uniform(*EXTR, NUM)

        egrid = utils.spacing(aa, 'lin', 2000, stretch=0.5)
        cgrid = utils.midpoints(egrid, 'lin')
        delta = np.diff(egrid)

        boundaries = [None, EXTR]
        for bnd in boundaries:
            kde = kale.KDE(aa, kernel=kernel)
            pdf = kde.pdf(cgrid, reflect=bnd)

            # If the kernel's support is infinite, then all points outside of boundaries should be
            # nonzero; if it's finite-supported, then only some of them (near edges) will be
            outside_test_func = np.all if kernel._FINITE == 'infinite' else np.any

            # Make sure unitarity is preserved
            tot = np.sum(pdf*delta)
            print("Boundary '{}', total = {:.4e}".format(bnd, tot))
            assert_true(np.isclose(tot, 1.0, rtol=1e-3))

            ratio_extr = np.max(pdf)/np.min(pdf[pdf > 0])
            # No reflection, then non-zero PDF everywhere, and large ratio of extrema
            if bnd is None:
                assert_true(outside_test_func(pdf[cgrid < EXTR[0]] > 0.0))
                assert_true(outside_test_func(pdf[cgrid > EXTR[1]] > 0.0))
                assert_true(ratio_extr > 10.0)
            # No lower-reflection, nonzero values below 0.0
            elif bnd[0] is None:
                assert_true(outside_test_func(pdf[cgrid < EXTR[0]] > 0.0))
                assert_true(np.all(pdf[cgrid > EXTR[1]] == 0.0))
            # No upper-reflection, nonzero values above 2.0
            elif bnd[1] is None:
                assert_true(np.all(pdf[cgrid < EXTR[0]] == 0.0))
                assert_true(outside_test_func(pdf[cgrid > EXTR[1]] > 0.0))
            else:
                assert_true(np.all(pdf[cgrid < EXTR[0]] == 0.0))
                assert_true(np.all(pdf[cgrid > EXTR[1]] == 0.0))
                assert_true(ratio_extr < 2.0)

        return

    def reflect_2d(self, kernel):
        print("\n|Test_KDE_PDF:test_reflect_2d()|")
        np.random.seed(124)
        NUM = 1000
        xx = np.random.uniform(0.0, 2.0, NUM)
        yy = np.random.normal(1.0, 1.0, NUM)
        yy = yy[yy < 2.0]
        yy = np.concatenate([yy, np.random.choice(yy, NUM-yy.size)])

        data = [xx, yy]
        edges = [kale.utils.spacing(aa, 'lin', 30) for aa in [xx, yy]]
        egrid = [kale.utils.spacing(ee, 'lin', 100, stretch=0.5) for ee in edges]
        cgrid = [kale.utils.midpoints(ee, 'lin') for ee in egrid]
        width = [np.diff(ee) for ee in egrid]

        xc, yc = np.meshgrid(*cgrid)

        grid = np.vstack([xc.ravel(), yc.ravel()])

        hist, *_ = np.histogram2d(*data, bins=egrid, density=True)

        kde = kale.KDE(data, kernel=kernel)
        inside_test_func = np.all if kernel._FINITE == 'infinite' else np.any

        reflections = [
            [[0.0, 2.0], [None, 2.0]],
            [[0.0, 2.0], None],
            [None, [None, 2.0]],
            None
        ]
        for jj, reflect in enumerate(reflections):
            pdf_1d = kde.pdf(grid, reflect=reflect)
            pdf = pdf_1d.reshape(hist.shape)

            inside = np.ones_like(pdf_1d, dtype=bool)
            if reflect is None:
                outside = np.zeros_like(pdf_1d, dtype=bool)
            else:
                outside = np.ones_like(pdf_1d, dtype=bool)
                for ii, ref in enumerate(reflect):
                    if ref is None:
                        ref = [-np.inf, np.inf]
                    if ref[0] is None:
                        ref[0] = -np.inf
                    if ref[1] is None:
                        ref[1] = np.inf
                    inside = inside & (ref[0] < grid[ii]) & (grid[ii] < ref[1])
                    outside = outside & ((grid[ii] < ref[0]) | (ref[1] < grid[ii]))

            assert_true(inside_test_func(pdf_1d[inside] > 0.0))
            assert_true(np.allclose(pdf_1d[outside], 0.0))

            area = width[0][:, np.newaxis] * width[1][np.newaxis, :]
            prob_tot = np.sum(pdf * area)
            print(jj, reflect, "prob_tot = {:.4e}".format(prob_tot))
            assert_true(np.isclose(prob_tot, 1.0, rtol=3e-2))

        return

    def pdf_params_fixed_bandwidth(self, kernel):
        print("\n|Test_KDE_PDF:pdf_params_fixed_bandwidth()|")
        np.random.seed(124)

        NUM = 1000
        bandwidth = 0.02

        sigma = [2.5, 1.5]
        corr = 0.9

        s2 = np.square(sigma)
        cc = corr*sigma[0]*sigma[1]
        cov = [[s2[0], cc], [cc, s2[1]]]
        cov = np.array(cov)

        data = np.random.multivariate_normal([1.0, 2.0], cov, NUM).T

        sigma = [2.5, 0.5]
        corr = 0.0

        s2 = np.square(sigma)
        cc = corr*sigma[0]*sigma[1]
        cov = [[s2[0], cc], [cc, s2[1]]]
        cov = np.array(cov)
        more = np.random.multivariate_normal([1.0, 6.0], cov, NUM).T
        data = np.concatenate([data, more], axis=-1)

        kde = kale.KDE(data, bandwidth=bandwidth, kernel=kernel)

        edges = [kale.utils.spacing(dd, 'lin', 200, stretch=0.1) for dd in data]
        cents = [kale.utils.midpoints(ee, 'lin') for ee in edges]
        widths = [np.diff(ee) for ee in edges]
        # area = widths[0][:, np.newaxis] * widths[1][np.newaxis, :]

        xe, ye = np.meshgrid(*edges)
        xc, yc = np.meshgrid(*cents)
        # grid = np.vstack([xc.ravel(), yc.ravel()])

        hist, *_ = np.histogram2d(*data, bins=edges, density=True)

        for par in range(2):
            xx = cents[par]
            pdf_2d = kde.pdf(xx, params=par)
            kde_1d = kale.KDE(data[par, :], bandwidth=bandwidth, kernel=kernel)
            pdf_1d = kde_1d.pdf(xx)
            # print("matrix : ", kde.bandwidth.matrix, kde_1d.bandwidth.matrix)
            assert_true(np.allclose(pdf_2d, pdf_1d, rtol=1e-3))

            for pdf, ls, lw in zip([pdf_2d, pdf_1d], ['-', '--'], [1.5, 3.0]):

                tot = np.sum(pdf*widths[par])
                print("tot = {:.4e}".format(tot))
                assert_true(np.isclose(tot, 1.0, rtol=2e-2))
                vals = [xx, pdf]
                if par == 1:
                    vals = vals[::-1]

        return


class Test_KDE_PDF_Gaussian(Test_KDE_PDF):

    def test_compare_scipy_1d(self):
        print("\n|Test_KDE_PDF:test_compare_scipy_1d()|")
        self.compare_scipy_1d(kale.kernels.Gaussian)
        return

    def test_compare_scipy_2d(self):
        print("\n|Test_KDE_PDF:test_compare_scipy_2d()|")
        self.compare_scipy_2d(kale.kernels.Gaussian)
        return

    def test_reflect_1d(self):
        print("\n|Test_KDE_PDF:test_reflect_1d()|")
        self.reflect_1d(kale.kernels.Gaussian)
        return

    def test_reflect_2d(self):
        print("\n|Test_KDE_PDF:test_reflect_2d()|")
        self.reflect_2d(kale.kernels.Gaussian)
        return

    def test_pdf_params_fixed_bandwidth(self):
        print("\n|Test_KDE_PDF_Gaussian:test_pdf_params_fixed_bandwidth()|")
        self.pdf_params_fixed_bandwidth(kale.kernels.Gaussian)
        return


class Test_KDE_PDF_Box(Test_KDE_PDF):

    def test_reflect_1d(self):
        print("\n|Test_KDE_PDF:test_reflect_1d()|")
        self.reflect_1d(kale.kernels.Box)
        return

    def test_reflect_2d(self):
        print("\n|Test_KDE_PDF:test_reflect_2d()|")
        self.reflect_2d(kale.kernels.Box)
        return

    def test_pdf_params_fixed_bandwidth(self):
        print("\n|Test_KDE_PDF_Box:test_pdf_params_fixed_bandwidth()|")
        self.pdf_params_fixed_bandwidth(kale.kernels.Box)
        return


class Test_KDE_Resample(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    def test_different_bws(self):
        print("\n|Test_KDE_Resample:test_different_bws()|")
        np.random.seed(9235)
        NUM = 1000
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bb = np.random.normal(3.0, 0.02, NUM) + aa/100

        data = [aa, bb]
        edges = [utils.spacing(dd, 'lin', 100, stretch=1.0) for dd in data]
        cents = [utils.midpoints(ee, 'lin') for ee in edges]

        xe, ye = np.meshgrid(*edges)
        xc, yc = np.meshgrid(*cents)

        bws = [0.5, 2.0]
        kde2d = kale.KDE(data, bandwidth=bws)
        kde1d = [kale.KDE(dd, bandwidth=ss) for dd, ss in zip(data, bws)]

        for ii in range(2):
            samp_1d = kde1d[ii].resample(NUM).squeeze()
            samp_2d = kde2d.resample(NUM)[ii]

            # Make sure the two distributions resemble eachother
            ks, pv = sp.stats.ks_2samp(samp_1d, samp_2d)
            # Calibrated to the above seed-value of `9235`
            assert_true(pv > 0.5)

        return

    def test_resample_keep_params_1(self):
        print("\n|Test_KDE_Resample:test_resample_keep_params_1()|")
        np.random.seed(9235)
        NUM = 1000

        # Construct some random data
        # ------------------------------------
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(1.0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bb = np.random.normal(3.0, 0.02, NUM) + aa/100

        data = [aa, bb]

        norm = 2.3

        # Add an array of uniform values at location `ii`, make sure they are preserved in resample
        for ii in range(3):
            test = np.array(data)
            test = np.insert(test, ii, norm*np.ones_like(test[0]), axis=0)

            # Construct KDE
            kde3d = kale.KDE(test)

            # Resample from KDE preserving the uniform data
            samples = kde3d.resample(NUM, keep=ii)
            # Make sure the uniform values are still the same
            param_samp = samples[ii]
            assert_true(np.allclose(param_samp, norm))

            # Make sure the other two parameters are consistent (KS-test) with input data
            samples = np.delete(samples, ii, axis=0)
            for jj in range(2):
                stuff = [samples[jj], data[jj]]
                ks, pv = sp.stats.ks_2samp(*stuff)
                # msg = "{} {} :: {:.2e} {:.2e}".format(ii, jj, ks, pv)
                # print(msg)
                assert_true(pv > 0.2)

        return

    def test_resample_keep_params_2(self):
        print("\n|Test_KDE_Resample:test_resample_keep_params_2()|")

        # Construct random data
        # -------------------------------
        np.random.seed(2235)
        NUM = 300
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(1.0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bb = np.random.normal(3.0, 0.02, NUM) + aa/100

        data = [aa, bb]

        norms = [2.3, -3.4]
        # Choose two locations to insert new, uniform variables
        for ii in range(3):
            jj = ii
            # Make sure the locations are different
            while jj == ii:
                jj = np.random.choice(3)

            # Insert uniform arrays
            lo = np.min([ii, jj])
            hi = np.max([ii, jj])
            test = np.array(data)
            test = np.insert(test, lo, norms[0]*np.ones_like(test[0]), axis=0)
            test = np.insert(test, hi, norms[1]*np.ones_like(test[0]), axis=0)

            # Construct KDE and draw new samples preserving the inserted variables
            kde4d = kale.KDE(test)
            samples = kde4d.resample(NUM, keep=(lo, hi))
            # Make sure the target variables are preserved
            for kk, ll in enumerate([lo, hi]):
                param_samps = samples[ll]
                # print(norms[kk], zmath.stats_str(param_samps))
                assert_true(np.allclose(param_samps, norms[kk]))

            # Make sure the resamples data is all consistent with input
            for jj in range(4):
                stuff = [samples[jj], test[jj]]
                ks, pv = sp.stats.ks_2samp(*stuff)
                # msg = "{} {} :: {:.2e} {:.2e}".format(ii, jj, ks, pv)
                assert_true(pv > 0.1)

        return

    def test_reflect_1d(self):
        print("\n|Test_KDE_Resample:test_reflect_1d()|")

        np.random.seed(1245)
        NUM = 1000

        extr = [0.0, 2.0]
        data = np.random.uniform(*extr, NUM)

        edges = kale.utils.spacing(extr, 'lin', 30)
        edges = np.concatenate([
            edges[0] - np.arange(1, 5)[::-1]*np.diff(edges)[0],
            edges,
            edges[-1] + np.arange(1, 5)*np.diff(edges)[-1],
        ])

        kde = kale.KDE(data)

        reflections = [None, extr]
        for ii, reflect in enumerate(reflections):
            samp = kde.resample(NUM, reflect=reflect)

            some_outside = np.any((samp < extr[0]) + (extr[1] < samp))
            if reflect is None:
                # There should be some samples outside
                assert_true(some_outside)
            else:
                # There should not be any samples outside
                assert_false(some_outside)
                ks, pv = sp.stats.ks_2samp(data, samp)
                # Check new sample is consistent
                assert_true(pv > 0.4)

        return

    def test_reflect_2d(self):
        print("\n|Test_KDE_Resample:test_reflect_2d()|")

        seed = np.random.randint(int(1e4))
        seed = 8067
        print(seed)
        np.random.seed(seed)
        NUM = 2000
        xx = np.random.uniform(0.0, 2.0, NUM)
        yy = np.random.normal(1.0, 1.5, NUM)
        yy = yy[yy < 2.0]
        yy = np.concatenate([yy, np.random.choice(yy, NUM-yy.size)])

        data = [xx, yy]
        edges = [kale.utils.spacing(aa, 'lin', 30) for aa in [xx, yy]]
        egrid = [kale.utils.spacing(ee, 'lin', 100, stretch=0.5) for ee in edges]
        cgrid = [kale.utils.midpoints(ee, 'lin') for ee in egrid]
        # width = [np.diff(ee) for ee in egrid]

        xc, yc = np.meshgrid(*cgrid)

        # grid = np.vstack([xc.ravel(), yc.ravel()])

        hist, *_ = np.histogram2d(*data, bins=egrid, density=True)

        kde = kale.KDE(data)

        reflections = [
            [[0.0, 2.0], [None, 2.0]],
            [[0.0, 2.0], None],
            [None, [None, 2.0]],
            None
        ]
        for jj, reflect in enumerate(reflections):
            samps_ref = kde.resample(reflect=reflect)
            samps_nrm = kde.resample()

            if reflect is None:
                continue

            for ii, ref in enumerate(reflect):
                if ref is None:
                    continue
                if ref[0] is None:
                    ref[0] = -np.inf
                if ref[1] is None:
                    ref[1] = np.inf

                print(jj, ii, ref)
                for kk, zz in enumerate([samps_nrm[ii], samps_ref[ii]]):
                    inside = (ref[0] < zz) & (zz < ref[1])
                    outside = ((zz < ref[0]) | (ref[1] < zz))

                    print("\tin : ", kk, np.all(inside), np.any(inside))
                    print("\tout: ", kk, np.all(outside), np.any(outside))

                    if kk == 0:
                        assert_false(np.all(inside))
                        assert_true(np.any(outside))
                    else:
                        assert_true(np.all(inside))
                        assert_false(np.any(outside))

        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
