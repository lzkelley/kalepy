"""

"""

import numpy as np
import pytest

import kalepy as kale
import kalepy.utils

GOOD_DISTRIBUTION_NAMES = kale.kernels.DISTRIBUTIONS.keys()
BAD_DISTRIBUTION_NAMES = ['triangle', 'spaceship', '', 0.5]


class Test_Distribution_Generic:

    @classmethod
    def _test_evaluate(self, kernel):
        print(kernel)

        hh = 1.0
        edges = np.linspace(-4*hh, 4*hh, 10000)
        cents = kale.utils.midpoints(edges)

        yy = kernel.evaluate(cents[np.newaxis, :], 1).squeeze()
        # Make sure kernel is callable
        # assert (np.allclose(yy, kernel().evaluate(cents)))

        # Make sure kernel is normalized
        tot = np.trapz(yy, cents)
        msg = "Kernel is not unitary"
        assert np.allclose(tot, 1.0, rtol=1e-3), msg

        # Make sure kernels have expected support
        assert np.all(yy >= 0.0)
        if kernel._FINITE:
            outside = (cents < -hh) | (hh < cents)
            inside = (-hh < cents) & (cents < hh)
        else:
            outside = []
            inside = np.ones_like(yy, dtype=bool)

        assert np.allclose(yy[outside], 0.0, rtol=1e-4, atol=1e-4)
        assert np.all(yy[inside] > 0.0)

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

        assert np.isclose(tot, 1.0, rtol=delta)

        return

    @classmethod
    def _test_evaluate_nd(cls, kernel):
        for kern in GOOD_DISTRIBUTION_NAMES:
            kern = kale.kernels.get_distribution_class(kern)
            for ndim in range(1, 4):
                cls._test_grid_at_ndim(kern, ndim)
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


def test_get_distribution_class():
    print("\n|test_kernels.py:test_get_distribution_class()|")
    for name in GOOD_DISTRIBUTION_NAMES:
        print("Name: '{}'".format(name))
        kale.kernels.get_distribution_class(name)()

    for name in kale.kernels.DISTRIBUTIONS.keys():
        kale.kernels.get_distribution_class(name)()

    for name in BAD_DISTRIBUTION_NAMES:
        # with tools.assert_raises(ValueError):
        with pytest.raises(ValueError):
            kale.kernels.get_distribution_class(name)

    # Test defaults
    kale.kernels.get_distribution_class(None)()
    kale.kernels.get_distribution_class()()

    return
