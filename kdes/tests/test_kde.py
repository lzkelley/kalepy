"""

Can be run with:
    $ nosetests tests/test_kde.py

"""

import numpy as np
import scipy as sp
from numpy.testing import run_module_suite
from nose.tools import assert_true


class Test_KDE(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    def test_compare_scipy_1d(self):
        print("\Test_KDE:test_compare_scipy_1d()")
        NUM = 100
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bins = zmath.spacing([-1, 14.0], 'lin', 40)
        grid = zmath.spacing(bins, 'lin', 3000)

        methods = ['scott', 0.04, 0.2, 0.8]
        classes = [sp.stats.gaussian_kde, zmath.kde.KDE]
        setters = ['bw_method', 'bandwidth']
        for mm in methods:
            kdes = [cc(aa, **{ss: mm}).pdf(grid)
                    for cc, ss in zip(classes, setters)]
            assert_true(np.allclose(kdes[0], kdes[1]))

        return

    def test_compare_scipy_2d(self):
        print("\n|Test_KDE:test_compare_scipy_2d()|")

        NUM = 1000
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bb = np.random.normal(3.0, 0.02, NUM) + aa/100

        data = [aa, bb]
        edges = [zmath.spacing(dd, 'lin', 30, stretch=0.5) for dd in data]
        cents = [zmath.midpoints(ee, 'lin') for ee in edges]

        xe, ye = np.meshgrid(*edges)
        xc, yc = np.meshgrid(*cents)
        grid = np.vstack([xc.ravel(), yc.ravel()])

        methods = ['scott', 0.04, 0.2, 0.8]
        classes = [sp.stats.gaussian_kde, zmath.kde.KDE]
        setters = ['bw_method', 'bandwidth']
        for mm in methods:
            kdes = [cc(data, **{ss: mm}).pdf(grid).reshape(xc.shape).T
                    for cc, ss in zip(classes, setters)]
            assert_true(np.allclose(kdes[0], kdes[1]))

        return

    def test_different_bws(self):
        print("\n|Test_KDE:test_different_bws()|")
        np.random.seed(9235)
        NUM = 1000
        a1 = np.random.normal(6.0, 1.0, NUM//2)
        a2 = np.random.lognormal(0, 0.5, size=NUM//2)
        aa = np.concatenate([a1, a2])

        bb = np.random.normal(3.0, 0.02, NUM) + aa/100

        data = [aa, bb]
        edges = [zmath.spacing(dd, 'lin', 100, stretch=1.0) for dd in data]
        cents = [zmath.midpoints(ee, 'lin') for ee in edges]

        xe, ye = np.meshgrid(*edges)
        xc, yc = np.meshgrid(*cents)

        bws = [0.5, 2.0]
        kde2d = zmath.kde.KDE(data, bandwidth=bws)
        kde1d = [zmath.kde.KDE(dd, bandwidth=ss) for dd, ss in zip(data, bws)]

        for ii in range(2):
            samp_1d = kde1d[ii].resample(NUM).squeeze()
            samp_2d = kde2d.resample(NUM)[ii]

            # Make sure the two distributions resemble eachother
            ks, pv = sp.stats.ks_2samp(samp_1d, samp_2d)
            # Calibrated to the above seed-value of `9235`
            assert_true(pv > 0.5)

        return

    def test_resample_keep_params_1(self):
        print("\n|Test_KDE:test_resample_keep_params_1()|")
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
            kde3d = zmath.kde.KDE(test)

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
        print("\n|Test_KDE:test_resample_keep_params_2()|")

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
            kde4d = zmath.kde.KDE(test)
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
        print("\n|Test_KDE:test_reflect_1d()|")

        np.random.seed(124)
        NUM = 1000
        EXTR = [0.0, 2.0]
        aa = np.random.uniform(*EXTR, NUM)

        egrid = zmath.spacing(aa, 'lin', 1000, stretch=0.5)
        cgrid = zmath.midpoints(egrid, 'lin')
        delta = np.diff(egrid)

        boundaries = [None, EXTR]
        for bnd in boundaries:
            kde = zmath.kde.KDE(aa)
            pdf = kde.pdf(cgrid, reflect=bnd)

            # Make sure unitarity is preserved
            tot = np.sum(pdf*delta)
            assert_true(np.isclose(tot, 1.0, rtol=1e-3))

            ratio_extr = np.max(pdf)/np.min(pdf[pdf > 0])
            # No reflection, then non-zero PDF everywhere, and large ratio of extrema
            if bnd is None:
                assert_true(np.all(pdf[cgrid < EXTR[0]] > 0.0))
                assert_true(np.all(pdf[cgrid > EXTR[1]] > 0.0))
                assert_true(ratio_extr > 10.0)
            # No lower-reflection, nonzero values below 0.0
            elif bnd[0] is None:
                assert_true(np.all(pdf[cgrid < EXTR[0]] > 0.0))
                assert_true(np.all(pdf[cgrid > EXTR[1]] == 0.0))
            # No upper-reflection, nonzero values above 2.0
            elif bnd[1] is None:
                assert_true(np.all(pdf[cgrid < EXTR[0]] == 0.0))
                assert_true(np.all(pdf[cgrid > EXTR[1]] > 0.0))
            else:
                assert_true(np.all(pdf[cgrid < EXTR[0]] == 0.0))
                assert_true(np.all(pdf[cgrid > EXTR[1]] == 0.0))
                assert_true(ratio_extr < 2.0)

        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
