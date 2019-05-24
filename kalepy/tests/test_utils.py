"""

Can be run with:
    $ nosetests tests/test_utils.py

"""

import numpy as np
# import scipy as sp
from numpy.testing import run_module_suite
from nose.tools import assert_true

from kalepy import utils


class Test_Midpoints(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(9865)
        return

    def test_midpoints_axes(self):
        print("\n|Test_Utils:test_midpoints_axes()|")
        # NUM = 100

        shp = (12, 14, 16)
        test = np.ones(shp)
        for ii in range(test.ndim):
            vals = utils.midpoints(test, 'lin', axis=ii)
            new_shape = np.array(shp)
            new_shape[ii] -= 1
            assert_true(np.all(vals.shape == new_shape))
            assert_true(np.all(vals == 1.0))

            vals = utils.midpoints(test, 'log', axis=ii)
            new_shape = np.array(shp)
            new_shape[ii] -= 1
            assert_true(np.all(vals.shape == new_shape))
            assert_true(np.all(vals == 1.0))

        test = np.arange(10)
        vals = utils.midpoints(test, 'lin')
        true = 0.5 * (test[:-1] + test[1:])
        assert_true(np.allclose(vals, true))
        return

    def test_midpoints_lin(self):
        print("\n|Test_Utils:test_midpoints_lin()|")
        test = [
            [0, 1, 2, 3],
            [2, 3, 4, 5]
        ]

        truth = [
            [1, 2, 3, 4],
            [
                [0.5, 1.5, 2.5],
                [2.5, 3.5, 4.5]
            ]
        ]

        for ii, tr in enumerate(truth):
            vals = utils.midpoints(test, 'lin', axis=ii)
            assert_true(np.all(np.shape(tr) == np.shape(vals)))
            assert_true(np.all(tr == vals))

        shp = (4, 5)
        test = np.random.uniform(-1.0, 1.0, np.product(shp)).reshape(shp)
        for ii in range(2):
            vals = utils.midpoints(test, 'lin', axis=ii)

            temp = np.moveaxis(test, ii, 0)
            true = temp[:-1, :] + 0.5*np.diff(temp, axis=0)
            true = np.moveaxis(true, 0, ii)
            assert_true(np.all(np.shape(true) == np.shape(vals)))
            assert_true(np.allclose(true, vals))

        return

    def test_midpoints_log(self):
        print("\n|Test_Utils:test_midpoints_log()|")
        test = [
            [1e0, 1e1, 1e2, 1e3],
            [1e2, 1e3, 1e4, 1e5]
        ]

        aa = np.sqrt(10.0)

        truth = [
            [1e1, 1e2, 1e3, 1e4],
            [
                [aa*1e0, aa*1e1, aa*1e2],
                [aa*1e2, aa*1e3, aa*1e4]
            ]
        ]

        for ii, tr in enumerate(truth):
            vals = utils.midpoints(test, 'log', axis=ii)
            assert_true(np.all(np.shape(tr) == np.shape(vals)))
            assert_true(np.allclose(tr, vals))

        shp = (4, 5)
        test_log = np.random.uniform(-2.0, 2.0, np.product(shp)).reshape(shp)
        test_lin = 10**test_log
        for ii in range(2):
            # Make sure `midpoints` gives consistent results itself
            vals_log = utils.midpoints(test_log, 'lin', axis=ii)
            vals_lin = utils.midpoints(test_lin, 'log', axis=ii)
            assert_true(np.all(np.shape(vals_log) == np.shape(vals_lin)))
            assert_true(np.allclose(10**vals_log, vals_lin))

            # Compare log-midpoint to known values
            temp = np.moveaxis(test_lin, ii, 0)
            temp = np.log10(temp)
            true = temp[:-1, :] + 0.5*np.diff(temp, axis=0)
            true = np.moveaxis(true, 0, ii)
            true = 10**true
            assert_true(np.all(np.shape(true) == np.shape(vals_lin)))
            assert_true(np.allclose(true, vals_lin))

        return

    def test_midpoints_off_center(self):
        print("\n|Test_Utils:test_midpoints_off_center()|")
        pass


class Test_Spacing(object):

    @classmethod
    def setup_class(cls):
        np.random.seed(1234)
        return

    def test_lin(self):
        print("\n|Test_Spacing:test_lin()|")
        aa = [64.15474369, 30.23993491, 18.74843086, 90.36893423, 81.49347391,
              21.66373546, 26.36243961,  9.54536041, 33.48985127, 87.77429238]
        bb = [9.54536041, 18.5257575, 27.5061546, 36.48655169, 45.46694878,
              54.44734587, 63.42774296, 72.40814005, 81.38853714, 90.36893423]

        test = utils.spacing(aa, 'lin', np.size(bb))
        assert_true(np.allclose(bb, test))
        return

    def test_log(self):
        print("\n|Test_Spacing:test_log()|")
        aa = [0.56979885,  0.06782166, 38.00982397,  0.76822742,  0.24328732,
              18.22846225,  7.22905804,  0.5140395,  0.97960639, 14.57931413]
        bb = [0.06782166,  0.13701255,  0.27679121,  0.55917048,  1.12962989,
              2.28206553,  4.61020298,  9.31347996, 18.81498695, 38.00982397]

        test = utils.spacing(aa, 'log', np.size(bb))
        assert_true(np.allclose(bb, test))
        return


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
