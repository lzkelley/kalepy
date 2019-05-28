"""
"""
import os

_path = os.path.dirname(__file__)
_vers_path = os.path.join(_path, "VERSION.txt")
with open(_vers_path) as inn:
    _version = inn.read().strip()

__version__ = _version
__author__ = "Luke Zoltan Kelley <lzkelley@northwestern.edu>"
__copyright__ = "Copyright 2019 - Luke Zoltan Kelley and contributors"
__contributors__ = []
__bibtex__ = """"""

from kalepy import kernels, bandwidths   # noqa
from kalepy.kde_base import KDE  # noqa
from kalepy.kernels import get_kernel_class, get_all_kernels  # noqa
