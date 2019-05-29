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
from kalepy import utils

# GOOD_KERNEL_NAMES = ['gaussian', 'box', 'parabola', 'epanechnikov']
GOOD_KERNEL_NAMES = [val[0] for val in kale.kernels._index_list]
BAD_KERNEL_NAMES = ['triangle', 'spaceship', '', 0.5]


class Test_Kernels_Base(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    def test_not_implemented(self):
        print("\n|Test_Kernels_Base:test_not_implemented()|")

        kern = kale.kernels.Distribution()
        with tools.assert_raises(NotImplementedError):
            kern.sample(2, [[1.0, 0.0], [0.0, 1.0]], 10)

        with tools.assert_raises(NotImplementedError):
            kern.evaluate(np.linspace(0.0, 1.0, 10))

        return

    '''
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

                mat = kale.kernels.Distribution._cov_keep_vars(matrix, keep)
                if keep is None:
                    tools.assert_true(np.allclose(matrix, mat))
                else:
                    for jj in keep:
                        # Make sure the keep (co)variance is all zero
                        tools.assert_true(np.allclose(mat[jj, :], 0.0))
                        tools.assert_true(np.allclose(mat[:, jj], 0.0))

                        # Make sure `reflect` consistency checks work
                        # These should also work
                        kale.kernels.Distribution._cov_keep_vars(matrix, keep, reflect=None)
                        kale.kernels.Distribution._cov_keep_vars(matrix, keep, reflect=reflect)
                        # Make sure reflection in `keep` parameter fails
                        fail = [rr for rr in reflect]
                        fail[jj] = [1.0, 2.0]
                        with tools.assert_raises(ValueError):
                            kale.kernels.Distribution._cov_keep_vars(matrix, keep, reflect=fail)
                        # Make sure reflection in any other parameters succeeds
                        not_keep = list(set(range(num)) - set(keep))
                        succeed = [rr for rr in reflect]
                        for ss in not_keep:
                            succeed[ss] = [1.0, 2.0]
                        kale.kernels.Distribution._cov_keep_vars(matrix, keep, reflect=succeed)

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

                sub_data, sub_mat, sub_norm = kale.kernels.Distribution._params_subset(
                    data, matrix, params)

                # Compare each parameter
                for jj, kk in enumerate(iter_params):
                    tools.assert_true(np.allclose(data[kk, :], sub_data[jj, :]))

                test_mat = np.atleast_2d(matrix[np.ix_(iter_params, iter_params)])
                tools.assert_true(np.allclose(test_mat, sub_mat))
                test_norm = np.sqrt(np.linalg.det(test_mat))
                tools.assert_true(np.isclose(test_norm, sub_norm))

        return
    '''


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
            cents = kale.utils.midpoints(edges, 'lin')
            width = np.diff(edges)
            yy = kernel.evaluate(cents, bw=hh)
            # Make sure kernel is callable
            tools.assert_true(np.allclose(yy, kernel().evaluate(cents, bw=hh)))

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

    @classmethod
    def _test_evaluate_nd(self, kernel):
        print("\n|Test_Kernels_Generic:_test_evaluate_nd()|")
        print(kernel)

        def kernel_at_dim_bw(kern, ndim, bw, num=1e6):
            if ndim > 3:
                raise ValueError("`ndim` = {} is too memory intensive!")
            pad = 2.0 if kern._FINITE else 4.0
            extr = [-pad*bw, pad*bw]
            num = np.power(num, 1/ndim)
            num = int(num)
            edges = np.zeros((ndim, num+1))
            cents = np.zeros((ndim, num))
            diffs = np.zeros_like(cents)
            for ii in range(ndim):
                edges[ii, :], cents[ii, :], diffs[ii, :] = kale.utils.bins(*extr, num+1)

            pdf_edges = kern.grid(edges, ref=np.zeros(ndim), bw=bw)
            tot = np.array(pdf_edges)
            for ii in range(ndim):
                tot = np.trapz(tot, x=edges[-1-ii])

            print("{} :: nd={}, bw={:.2f} : tot={:.4e}".format(
                kern.__name__, ndim, bw, tot))

            dpow = -4 + ndim
            delta = 2*np.power(10.0, np.minimum(dpow, -1))

            tools.assert_almost_equal(tot, 1.0, delta=delta)

            return

        kernels = kale.kernels.get_all_distribution_classes()

        num_dims = [1, 2, 3]
        bandwidths = [0.5, 1.0, 2.0, 4.0]

        for kern in kernels:
            print("\nkern: ", kern)
            for ndim in num_dims:
                print("\nndim: ", ndim)
                for bw in bandwidths:
                    kernel_at_dim_bw(kern, ndim, bw)

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


def test_get_distribution_class():
    print("\n|test_kernels.py:test_get_distribution_class()|")
    for name in GOOD_KERNEL_NAMES:
        print("Name: '{}'".format(name))
        kale.kernels.get_distribution_class(name)()

    for name in kale.kernels._index.keys():
        kale.kernels.get_distribution_class(name)()

    for name in BAD_KERNEL_NAMES:
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


def test_kernels_evaluate():
    print("\n|test_kernels.py:test_kernels_evaluate()|")

    for kernel in kale.kernels.get_all_distribution_classes():
        print("Testing '{}'".format(kernel))
        Test_Kernels_Generic._test_evaluate(kernel)

    return


def test_kernels_evaluate_nd():
    print("\n|test_kernels.py:test_kernels_evaluate_nd()|")

    for kernel in kale.kernels.get_all_distribution_classes():
        print("Testing '{}'".format(kernel))
        Test_Kernels_Generic._test_evaluate_nd(kernel)

    return


'''
def test_kernels_resample():
    print("\n|test_kernels.py:test_kernels_resample()|")

    for kernel in kale.kernels.get_all_distribution_classes():
        print("Testing '{}'".format(kernel))
        Test_Kernels_Generic._test_resample(kernel)

    return
'''

# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
