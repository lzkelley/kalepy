"""

Can be run with:
    $ nosetests tests/test_kde.py

"""

import numpy as np
# import scipy as sp
import scipy.stats  # noqa
from numpy.testing import run_module_suite
from nose import tools

import kdes
from kdes import utils

GOOD_KERNEL_NAMES = ['gaussian', 'box', 'parabola', 'epanechnikov']
BAD_KERNEL_NAMES = ['triangle', 'spaceship', '', 0.5]


class Test_Kernels_Base(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    def test_not_implemented(self):
        print("\n|Test_Kernels_Base:test_not_implemented()|")

        kern = kdes.kernels.Kernel()
        with tools.assert_raises(NotImplementedError):
            kern.sample(2, [[1.0, 0.0], [0.0, 1.0]], 10)

        with tools.assert_raises(NotImplementedError):
            kern.evaluate(np.linspace(0.0, 1.0, 10))

        with tools.assert_raises(NotImplementedError):
            kern(np.linspace(0.0, 1.0, 10))

        return

    def test_cov_keep_vars(self):
        print("\n|Test_Kernels_Base:test_cov_keep_vars()|")
        np.random.seed(3422)

        npars = [1, 2, 3, 4, 5]
        for num in npars:
            matrix = np.random.uniform(size=num*num).reshape(num, num)
            reflect = [None]*num

            for ii in range(num):
                if ii == 0:
                    keep = None
                else:
                    keep = np.random.choice(num, ii, replace=False)

                mat = kdes.kernels.Kernel._cov_keep_vars(matrix, keep)
                if keep is None:
                    tools.assert_true(np.allclose(matrix, mat))
                else:
                    for jj in keep:
                        # Make sure the keep (co)variance is all zero
                        tools.assert_true(np.allclose(mat[jj, :], 0.0))
                        tools.assert_true(np.allclose(mat[:, jj], 0.0))

                        # Make sure `reflect` consistency checks work
                        # These should also work
                        kdes.kernels.Kernel._cov_keep_vars(matrix, keep, reflect=None)
                        kdes.kernels.Kernel._cov_keep_vars(matrix, keep, reflect=reflect)
                        # Make sure reflection in `keep` parameter fails
                        fail = [rr for rr in reflect]
                        fail[jj] = [1.0, 2.0]
                        with tools.assert_raises(ValueError):
                            kdes.kernels.Kernel._cov_keep_vars(matrix, keep, reflect=fail)
                        # Make sure reflection in any other parameters succeeds
                        not_keep = list(set(range(num)) - set(keep))
                        succeed = [rr for rr in reflect]
                        for ss in not_keep:
                            succeed[ss] = [1.0, 2.0]
                        kdes.kernels.Kernel._cov_keep_vars(matrix, keep, reflect=succeed)

        return

    def test_params_subset(self):
        print("\n|Test_Kernels_Base:test_params_subset()|")
        np.random.seed(2342)
        npars = [1, 2, 3, 4, 5]
        nvals = 100
        # Try a range of dimensionalities (i.e. number of parameters)
        for num in npars:
            data = np.random.uniform(size=num*nvals).reshape(num, nvals)
            matrix = np.atleast_2d(np.cov(data, rowvar=True, bias=False))
            # Take subsets in each possible number of parameters
            for ii in range(num):
                # Randomly select which parameters to choose
                if ii == 0:
                    params = None
                    iter_params = np.arange(num)
                else:
                    params = sorted(np.random.choice(num, ii, replace=False))
                    iter_params = params

                sub_data, sub_mat, sub_norm = kdes.kernels.Kernel._params_subset(
                    data, matrix, params)

                # Compare each parameter
                for jj, kk in enumerate(iter_params):
                    tools.assert_true(np.allclose(data[kk, :], sub_data[jj, :]))

                test_mat = np.atleast_2d(matrix[np.ix_(iter_params, iter_params)])
                tools.assert_true(np.allclose(test_mat, sub_mat))
                test_norm = np.sqrt(np.linalg.det(test_mat))
                tools.assert_true(np.isclose(test_norm, sub_norm))

        return


class Test_Kernels_Generic(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    @classmethod
    def _test_evaluate(self, kernel):
        print("\n|Test_Kernels_Generic:_test_evaluate()|")
        print(kernel)
        bws = [0.1, 1.0, 2.0, 10.0]
        for ii, hh in enumerate(bws):
            print("\t", ii, hh)
            # bandwidth should always be scaled to 1.0
            yy, ndim, nval, squeeze = kernel.scale(hh, 0.0, hh)
            tools.assert_true(np.isclose(yy, 1.0))
            tools.assert_true(ndim == 1)
            tools.assert_true(nval == 1)

            edges = np.linspace(-10*hh, 10*hh, 10000)
            cents = kdes.utils.midpoints(edges, 'lin')
            width = np.diff(edges)
            yy = kernel.evaluate(cents, bw=hh)
            # Make sure kernel is callable
            tools.assert_true(np.allclose(yy, kernel()(cents, bw=hh)))

            # Make sure kernel is normalized
            tot = np.sum(yy*width)
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


def test_get_kernel_class():
    print("\n|test_kernels.py:test_get_kernel_class()|")
    for name in GOOD_KERNEL_NAMES:
        print("Name: '{}'".format(name))
        kern = kdes.kernels.get_kernel_class(name)
        kern()

    for name in kdes.kernels._index.keys():
        kern = kdes.kernels.get_kernel_class(name)
        kern()

    for name in BAD_KERNEL_NAMES:
        with tools.assert_raises(ValueError):
            kern = kdes.kernels.get_kernel_class(name)

    # Test defaults
    kern = kdes.kernels.get_kernel_class(None)
    kern()
    kern = kdes.kernels.get_kernel_class()
    kern()

    # Test custom kernel
    class Good_Kernel(kdes.kernels.Kernel):
        pass

    kern = kdes.kernels.get_kernel_class(Good_Kernel)

    # Test bad custom kernel
    class Bad_Kernel(object):
        pass

    with tools.assert_raises(ValueError):
        kern = kdes.kernels.get_kernel_class(Bad_Kernel)

    return


def test_kernels_evaluate():
    print("\n|test_kernels.py:test_kernels_evaluate()|")

    for kernel in kdes.kernels._get_all_kernels():
        print("Testing '{}'".format(kernel))
        Test_Kernels_Generic._test_evaluate(kernel)

    return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
