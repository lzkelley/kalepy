"""

Can be run with:
    $ nosetests tests/test_kde.py

"""

import numpy as np
import scipy as sp
import scipy.stats  # noqa
from numpy.testing import run_module_suite
from nose import tools

import kalepy as kale
from kalepy import utils

GOOD_DISTRIBUTION_NAMES = kale.kernels.DISTRIBUTIONS.keys()
BAD_DISTRIBUTION_NAMES = ['triangle', 'spaceship', '', 0.5]


# class Test_Distribution_Base(utils.Test_Base):
class Test_Distribution_Base:

    def test_callable(self):
        from kalepy.kernels import Distribution
        Distribution()
        return

    def test_not_implemented(self):
        from kalepy.kernels import Distribution

        xx = np.random.uniform(-1, 1, 10)

        # Class Methods
        # ----------------------
        class_funcs = [
            ["_evaluate", [xx, 1], {}],
            ["evaluate", [xx], {}],
            ["grid", [xx], {}],
        ]

        for ff, ar, kw in class_funcs:
            print("Function: '{}', args: '{}', kwargs: '{}'".format(ff, ar, kw))
            with tools.assert_raises(NotImplementedError):
                getattr(Distribution, ff)(*ar, **kw)
            with tools.assert_raises(NotImplementedError):
                getattr(Distribution(), ff)(*ar, **kw)

        # Instance Methods
        # ---------------------------
        inst_funcs = [
            ["sample", [10], {}],
            ["_sample", [11, 2], {}],
            ["cdf", [xx], {}],
            ["cdf_grid", [], {}],
        ]

        for ff, ar, kw in inst_funcs:
            print("Function: '{}', args: '{}', kwargs: '{}'".format(ff, ar, kw))
            with tools.assert_raises(NotImplementedError):
                getattr(Distribution(), ff)(*ar, **kw)

        return

    def test_parse(self):
        from kalepy.kernels import Distribution

        ndim_max = 5
        for ndim in range(ndim_max):
            # Use `ndim` to make sure that a scalar (non-array) is valid, but this still counts
            # as a "one-dimensional" value (i.e. a single-variate distribution), so set ndim=1
            if ndim == 0:
                xx = 0.5
                ndim = 1
            else:
                nvals = np.random.randint(2, 10)
                # Squeeze this array to test that (N,) will be expanded to (1, N)
                xx = np.random.uniform(-10.0, 10.0, nvals*ndim).reshape(ndim, nvals).squeeze()

            yy, _ndim, squeeze = Distribution._parse(xx)
            # Make sure number of dimensions are accurate
            utils.alltrue(_ndim == ndim)
            # Squeeze should be true for less than 2D
            utils.alltrue(squeeze == (ndim < 2))
            # Make sure values are the same, but 2d
            utils.allclose(yy, xx)
            utils.alltrue(np.ndim(yy) == 2)

        return

    def test_name_finite(self):
        from kalepy.kernels import Distribution

        dist = Distribution()
        print("Name: '{}', Finite: '{}'".format(dist.name(), dist.FINITE))

        tools.assert_true(dist.FINITE is None)
        tools.assert_true(dist.name() == "Distribution")

        # Try simple subclasses
        # ---------------------------
        words = ['Hello', 'Something', 'blubber', 'monKey23']
        finitude = [True, False, False, True]
        for ww, ff in zip(words, finitude):
            temp_class = type(ww, (Distribution,), dict(_FINITE=ff))
            temp = temp_class()
            # Make sure the name of the class is accurately returned
            tools.assert_true(temp.name() == ww)

            # Make sure the '_FINITE' property is accurately returned
            tools.assert_true(temp.FINITE is ff)

        return


# class Test_Distribution_Generic(utils.Test_Base):
class Test_Distribution_Generic:

    @classmethod
    def _test_evaluate(self, kernel):
        print(kernel)

        hh = 1.0
        edges = np.linspace(-4*hh, 4*hh, 10000)
        cents = kale.utils.midpoints(edges, 'lin')

        yy = kernel.evaluate(cents)
        # Make sure kernel is callable
        # tools.assert_true(np.allclose(yy, kernel().evaluate(cents)))

        # Make sure kernel is normalized
        tot = np.trapz(yy, cents)
        msg = "Kernel is {fail:} unitary"
        utils.allclose(tot, 1.0, rtol=1e-3, msg=msg)

        # Make sure kernels have expected support
        tools.assert_true(np.all(yy >= 0.0))
        if kernel._FINITE:
            outside = (cents < -hh) | (hh < cents)
            inside = (-hh < cents) & (cents < hh)
        else:
            outside = []
            inside = np.ones_like(yy, dtype=bool)

        utils.allclose(yy[outside], 0.0, rtol=1e-4, atol=1e-4)
        utils.alltrue(yy[inside] > 0.0)

        return

    @classmethod
    def _test_grid_at_ndim(cls, kern, ndim, num=1e5):
        if ndim > 4:
            raise ValueError("`ndim` = {} is too memory intensive!")

        pad = 2.0 if kern._FINITE else 4.0
        bw = 1.0
        extr = [-pad*bw, pad*bw]
        num = np.power(num, 1/ndim)
        num = int(num)
        edges = np.zeros((ndim, num+1))
        cents = np.zeros((ndim, num))
        diffs = np.zeros_like(cents)
        for ii in range(ndim):
            edges[ii, :], cents[ii, :], diffs[ii, :] = kale.utils.bins(*extr, num+1)

        pdf_edges = kern.grid(edges)
        tot = np.array(pdf_edges)
        for ii in range(ndim):
            tot = np.trapz(tot, x=edges[-1-ii])

        print("{} :: nd={}, bw={:.2f} : tot={:.4e}".format(
            kern.__name__, ndim, bw, tot))

        dpow = -4 + ndim
        delta = 2*np.power(10.0, np.minimum(dpow, -1))

        tools.assert_almost_equal(tot, 1.0, delta=delta)

        return

    @classmethod
    def _test_evaluate_nd(cls, kernel):
        for kern in GOOD_DISTRIBUTION_NAMES:
            kern = kale.kernels.get_distribution_class(kern)
            for ndim in range(1, 4):
                cls._test_grid_at_ndim(kern, ndim)
        return

    @classmethod
    def _test_sample(self, kernel):
        kern = kernel()

        NUM = int(1e6)
        bw = 1.0
        pad = 4.0
        xe, xc, dx = kale.utils.bins(-pad*bw, pad*bw, 100)
        samp = kern.sample(NUM)

        hist, _ = np.histogram(samp, xe, density=True)
        pdf = kern.evaluate(xc)

        cum_pdf = utils.trapz_dens_to_mass(pdf, xc)
        cum_pdf = np.cumsum(cum_pdf)

        cum_pdf = np.append([0.0], cum_pdf)
        cdf = kern.cdf(xc)

        # Compare 'analytic' PDF/CDF with distribution of samples
        # CDF tend not to match as well, so use larger tolerance
        for aa, bb, name, tol in zip([hist, cum_pdf], [pdf, cdf], ['pdf', 'cdf'], [1e-2, 1e-1]):
            idx = (aa > 0.0) & (bb > 0.0)
            dof = np.count_nonzero(idx) - 1
            x2 = np.sum(np.square(aa[idx] - bb[idx])/bb[idx]**2)
            x2 = x2 / dof

            print("Distribution: {} :: {} : x2/dof = {:.4e}".format(kern.name(), name, x2))
            print("\t" + kale.utils.array_str(aa[idx]))
            print("\t" + kale.utils.array_str(bb[idx]))
            utils.alltrue(x2 < tol)

        return

    @classmethod
    def _test_sample_at_ndim(self, kernel, ndim, num=1e6, conf=0.99):
        kern = kernel()
        np.random.seed(9876)
        print("\nkernel: {}, dim: {}".format(kern.name(), ndim))
        num = int(np.power(num, 1/ndim))

        pad = 4.0
        # xe, xc, dx = kale.utils.bins(-pad, pad, 10)
        samp = kern.sample(num, ndim, squeeze=False)

        hconf = 0.5 * (1 - conf)

        xx = np.linspace(-pad, pad, 1000)
        fracs = [0.1, 0.5, 0.9]
        for ii in range(ndim):
            cdf = kern.cdf(xx)
            for ff in fracs:
                # Find the location at which this fraction of the distribution should be included
                loc = np.interp(ff, cdf, xx)
                # The expectation value for the number of samples below this location
                exp_num = ff * num
                # Find the 98% confidence region for the number of values below this location
                bounds = sp.stats.poisson.ppf([hconf, 1 - hconf], exp_num)
                # Count the number of samples up to this location
                cnt = np.count_nonzero(samp[ii, :] < loc)
                msg = (
                    "f={:.3f}, loc={:.5e}, exp={:.5e}".format(ff, loc, exp_num)
                    + " ==> bounds: {:.5e}, {:.5e} ==> cnt: {:.5e}".format(*bounds, cnt)
                )
                print(msg)
                # Make sure that the count is within the confidence interval
                tools.assert_true((bounds[0] < cnt) & (cnt < bounds[1]))

        return


class Test_Distribution(Test_Distribution_Generic):

    def test_evaluate(self):
        for kernel in GOOD_DISTRIBUTION_NAMES:
            kernel = kale.kernels.get_distribution_class(kernel)
            self._test_evaluate(kernel)

        return

    def test_evaluate_nd(self):

        for kernel in GOOD_DISTRIBUTION_NAMES:
            kernel = kale.kernels.get_distribution_class(kernel)
            print("Testing '{}'".format(kernel))
            self._test_evaluate_nd(kernel)

        return

    def test_kernels_sample(self):
        for kernel in GOOD_DISTRIBUTION_NAMES:
            kernel = kale.kernels.get_distribution_class(kernel)
            Test_Distribution_Generic._test_sample(kernel)

        return

    def test_kernels_sample_nd(self):
        for kernel in GOOD_DISTRIBUTION_NAMES:
            kernel = kale.kernels.get_distribution_class(kernel)
            for ndim in range(1, 4):
                Test_Distribution_Generic._test_sample_at_ndim(kernel, ndim)

        return


def test_get_distribution_class():
    print("\n|test_kernels.py:test_get_distribution_class()|")
    for name in GOOD_DISTRIBUTION_NAMES:
        print("Name: '{}'".format(name))
        kale.kernels.get_distribution_class(name)()

    for name in kale.kernels.DISTRIBUTIONS.keys():
        kale.kernels.get_distribution_class(name)()

    for name in BAD_DISTRIBUTION_NAMES:
        with tools.assert_raises(ValueError):
            kale.kernels.get_distribution_class(name)

    # Test defaults
    kale.kernels.get_distribution_class(None)()
    kale.kernels.get_distribution_class()()

    # Test custom kernel
    class Good_Kernel(kale.kernels.Distribution):
        pass

    kale.kernels.get_distribution_class(Good_Kernel)()

    # Test bad custom kernel
    class Bad_Kernel(object):
        pass

    with tools.assert_raises(ValueError):
        kale.kernels.get_distribution_class(Bad_Kernel)

    return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
