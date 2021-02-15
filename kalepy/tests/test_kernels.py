"""

Can be run with:
    $ nosetests tests/test_kde.py

"""

import numpy as np
# import scipy as sp
import scipy.stats  # noqa
from numpy.testing import run_module_suite
from nose import tools

import kalepy as kale
from kalepy import utils, kernels


class Test_Kernels_Generic(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    @classmethod
    def _test_evaluate(self, kernel):
        print("\n|Test_Kernels_Generic:_test_evaluate()|")
        print(kernel)

        # bandwidth should always be scaled to 1.0
        # yy, ndim, nval, squeeze = kernel.scale(hh, 0.0, hh)
        # tools.assert_true(np.isclose(yy, 1.0))
        # tools.assert_true(ndim == 1)
        # tools.assert_true(nval == 1)

        hh = 1.0
        edges = np.linspace(-10*hh, 10*hh, 10000)
        cents = kale.utils.midpoints(edges, 'lin')
        # width = np.diff(edges)
        yy = kernel.evaluate(cents[np.newaxis, :], 1).squeeze()
        # Make sure kernel is callable
        # tools.assert_true(np.allclose(yy, kernel().evaluate(cents)))

        # Make sure kernel is normalized
        # tot = np.sum(yy*width)
        tot = np.trapz(yy, cents)
        # print("\t\ttot = {:.4e}".format(tot))
        tools.assert_almost_equal(tot, 1.0, delta=1e-3)

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
    def kernel_at_dim(cls, kern, ndim, num=1e6):
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

        err = "kernel_at_dim({}, {}, {})\t".format(kern, ndim, num)
        err += "Integrated PDF {:.4e} is not within {:.2e} -- not unitary!".format(tot, delta)
        tools.assert_almost_equal(tot, 1.0, delta=delta, msg=err)

        return

    @classmethod
    def _test_evaluate_nd(cls, kernel):
        # print("\n|Test_Kernels_Generic:_test_evaluate_nd()|")
        # print(kernel)

        # kernels = kale.kernels.DISTRIBUTIONS.values()

        num_dims = [1, 2, 3, 4]

        # for kern in kernels:
        # print("kernel: ", kernel)
        for ndim in num_dims:
            # print("\tndim: ", ndim)
            cls.kernel_at_dim(kernel, ndim)

        return

    @classmethod
    def _test_resample(self, kern):

        def resample_at_bandwidth(bw):
            NUM = int(1e6)
            xe, xc, dx = kale.utils.bins(-2*bw, 2*bw, 40)
            samp = kern.sample(1, np.atleast_2d(bw), NUM).squeeze()

            hist, _ = np.histogram(samp, xe, density=True)
            pdf = kern.evaluate(xc, 0.0, bw)

            hist_cum = np.cumsum(hist*dx)
            hist_cum = np.append([0.0], hist_cum)
            cdf = kern.cdf(xe, bw=bw)

            for aa, bb, name in zip([hist, hist_cum], [pdf, cdf], ['pdf', 'cdf']):
                idx = (aa > 0.0) & (bb > 0.0)
                dof = np.count_nonzero(idx) - 1
                x2 = np.sum(np.square(aa[idx] - bb[idx])/bb[idx]**2)
                x2 = x2 / dof
                print("Distribution: {}, bw: {:.2e} :: {} : x2/dof = {:.4e}".format(
                    kern.__name__, bw, name, x2))
                print("\t" + kale.utils.array_str(aa[idx]))
                print("\t" + kale.utils.array_str(bb[idx]))
                tools.assert_true(x2 < 1e-2)

            return

        bandwidths = [0.5, 1.0, 2.0, 3.0]
        for jj, bw in enumerate(bandwidths):
            resample_at_bandwidth(bw)

        return


def test_kernels_evaluate():
    print("\n|test_kernels.py:test_kernels_evaluate()|")

    for kernel in kale.kernels.DISTRIBUTIONS.values():
        print("Testing '{}'".format(kernel))
        Test_Kernels_Generic._test_evaluate(kernel)

    return


def test_kernels_evaluate_nd():
    # print("\n|test_kernels.py:test_kernels_evaluate_nd()|")
    for kernel in kale.kernels.DISTRIBUTIONS.values():
        # print("Testing '{}'".format(kernel))
        Test_Kernels_Generic._test_evaluate_nd(kernel)

    return


def _test_check_reflect(ndim, weights_flag):
    reflects_good = [
        None,
        [-1.0, 1.0],
        [-1.0, None],
        [None, 1.0],
        [None, None],
    ]

    reflects_bad = [
        [None],
        10.0,
        [12.0],
        [None, 1.0, 2.0],
        [2.0, 1.0],
    ]

    ndata = 7

    # Generate data
    data = []
    for ii in range(ndim):
        data.append(np.random.uniform(-1.0, 1.0, ndata))

    weights = np.random.uniform(0.0, 1.0, ndata) if weights_flag else None
    nref = len(reflects_good)
    shape = [nref] * ndim
    for inds in np.ndindex(*shape):
        reflect = [reflects_good[ii] for ii in inds]
        print("---------")
        print("inds = {}, reflect = {}".format(inds, reflect))
        # Make sure good values do not raise errors, and return the appropriate object
        ref = kernels._check_reflect(reflect, data, weights=weights)
        if len(ref) != ndim:
            err = "Returned reflect has length {}, but ndim = {}!".format(len(ref), ndim)
            raise ValueError(err)

        # Make sure bad values raise errors
        for jj, bad in enumerate(reflects_bad):
            jj = jj % ndim
            reflect[jj] = bad
            tools.assert_raises(ValueError, kernels._check_reflect, *(reflect, data, weights))

    return


def test_check_reflect():

    for ndim in range(1, 4):
        for wf in [False, True]:
            _test_check_reflect(ndim, wf)

    return


def test_check_reflect_boolean():
    np.random.seed(12456)
    ndim = 3

    data = np.random.uniform(-1.0, 1.0, (ndim, 100))
    data[1] *= 2.0

    ref = kernels._check_reflect(None, data)
    tools.assert_true(ref is None)

    ref = kernels._check_reflect(False, data)
    tools.assert_true(ref is None)

    extrema = [utils.minmax(dd) for dd in data]
    extrema = np.asarray(extrema)
    extrema[:, 0] *= (1 - kale._NUM_PAD)
    extrema[:, 1] *= (1 + kale._NUM_PAD)

    # If `True` is Given, values should be set to extrema
    reflect = kernels._check_reflect(True, data)
    tools.assert_true(np.allclose(reflect, extrema, atol=1e-10))

    for ii in range(ndim):
        # True given for a single dimension
        reflect = [None for ii in range(ndim)]
        reflect[ii] = True
        ref = kernels._check_reflect(reflect, data)

        tools.assert_true(np.allclose(ref[ii], extrema[ii], atol=1e-10))
        tools.assert_true(np.all([ref[jj] is None for jj in range(ndim) if jj != ii]))

        # True given as lower or upper bound for dimension
        reflect = [None for ii in range(ndim)]
        # val = np.random.uniform(-10.0, 10.0)
        val = 12.34

        # Fixed upper-value (must be larger than lower-value!)
        reflect[ii] = [True, val]
        ref = kernels._check_reflect(reflect, data)
        tools.assert_true(np.isclose(ref[ii][0], extrema[ii][0], atol=1e-10))
        tools.assert_true(ref[ii][1] == val)
        tools.assert_true(np.all([ref[jj] is None for jj in range(ndim) if jj != ii]))

        # Fixed lower-value (must be less than upper-value!)
        reflect[ii] = [-val, True]
        ref = kernels._check_reflect(reflect, data)
        tools.assert_true(np.isclose(ref[ii][1], extrema[ii][1], atol=1e-10))
        tools.assert_true(ref[ii][0] == -val)
        tools.assert_true(np.all([ref[jj] is None for jj in range(ndim) if jj != ii]))

    return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
