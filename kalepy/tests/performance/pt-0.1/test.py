"""
"""
import os
import numpy as np
import kalepy as kale

FNAME = "pt-0.1.npz"


def load(fname):
    data = np.load(fname)['data'][...]
    # print(f"Loaded {data.shape=}. {data.size=}")
    return data


def test(data, smaller=10):
    if smaller is not None:
        data = [dd[slice(None, None, smaller)] for dd in data]

    data = np.asarray(data)
    kde = kale.KDE(data)
    corner = kale.Corner(data.shape[0])
    corner.clean(kde)

    samp = kde.resample()
    corner.plot_data(samp)
    return


def main():
    path = os.path.abspath(__file__)
    path = os.path.dirname(path)
    path = os.path.join(path, FNAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File '{path}' does not exist!")

    data = load(path)
    test(data)
    return


if __name__ == "__main__":
    main()
