"""DEVELOPMENT: submodule for running speed-tests for optimization checking.
"""
import os
import logging
from datetime import datetime
import numpy as np
import kalepy as kale

fname = "~/kalepy-sampling_test-data.npz"


def main():
    global fname
    fname = os.path.realpath(os.path.expanduser(fname))

    # print(f"Loading from '{fname=}")
    data = np.load(fname, allow_pickle=True)
    log_edges = data['log_edges']
    dens = data['dens']
    mass = data['mass']
    # print(f"loaded {dens.shape=}, {mass.shape=}")

    beg = datetime.now()
    vals, weights = kale.sample_outliers(log_edges, dens, 10.0, mass=mass)
    end = datetime.now()
    print(f"Finished after {(end-beg)}")

    return


if __name__ == "__main__":
    print(f"\n{__file__}\n")
    logging.getLogger(__file__).setLevel(logging.INFO)
    import cProfile
    import pstats
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    print("\n\n")

    # stats = pstats.Stats(profiler).sort_stats('ncalls')
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.print_stats(30)
    stats.dump_stats('cprofile-stats')
