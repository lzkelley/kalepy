"""

Can be run with:
    $ nosetests tests/test_kde.py

"""

import numpy as np
import scipy as sp
import scipy.stats  # noqa
from numpy.testing import run_module_suite
from nose.tools import assert_true, assert_false, assert_raises

import kdes
from kdes import utils


class Test_Kernels_Base(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)

    def test_not_implemented(self):
        print("\n|Test_Kernels_Base:test_not_implemented()|")

        kern = kdes.kernels.Kernel()
        with assert_raises(NotImplementedError):
            kern.sample(2, [[1.0, 0.0], [0.0, 1.0]], 10)

        with assert_raises(NotImplementedError):
            kern.evaluate(np.linspace(0.0, 1.0, 10))

        with assert_raises(NotImplementedError):
            kern(np.linspace(0.0, 1.0, 10))

        return

    def test_params_subset(self):
        print("\n|Test_Kernels_Base:test_params_subset()|")

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
                    assert_true(np.allclose(data[kk, :], sub_data[jj, :]))

                test_mat = np.atleast_2d(matrix[np.ix_(iter_params, iter_params)])
                assert_true(np.allclose(test_mat, sub_mat))
                test_norm = np.sqrt(np.linalg.det(test_mat))
                assert_true(np.isclose(test_norm, sub_norm))

        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
