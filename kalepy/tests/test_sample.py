"""

Can be run with:
    $ nosetests tests/test_sample.py

"""

import numpy as np
# import scipy as sp
from numpy.testing import run_module_suite
from nose import tools

# import kalepy as kale
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
            # calculate shape of bin centers
            shape_cents = [sh - 1 for sh in shape_edges]
            # choose a number of samples
            nsamp = np.random.randint(100, 1000)

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


# Run all methods as if with `nosetests ...`
if __name__ == "__main__":
    run_module_suite()
