"""
"""

import os

_path = os.path.dirname(os.path.abspath(__file__))
_vers_path = os.path.join(_path, "VERSION.txt")
with open(_vers_path) as inn:
    _version = inn.read().strip()

__version__ = _version
__author__ = "Luke Zoltan Kelley <lzkelley@northwestern.edu>"
__copyright__ = "Copyright 2019 - Luke Zoltan Kelley and contributors"
__contributors__ = []
__bibtex__ = """"""

# Numerical padding parameter (e.g. to avoid edge issues, etc)
_NUM_PAD = 1e-8
# Zero-out the PDF of kernels with infinite-support beyond this probability
_TRUNCATE_INFINITE_KERNELS = 1e-8
# Default bandwidth calculation method
_BANDWIDTH_DEFAULT = 'scott'

_PATH_NB_OUT = os.path.join(_path, 'notebooks', 'output')


from kalepy import kernels   # noqa
from kalepy.kernels import *  # noqa
from kalepy import utils   # noqa
from kalepy.utils import *  # noqa

from kalepy.kde_base import KDE  # noqa

__all__ = []
__all__.extend(kernels.__all__)
__all__.extend(utils.__all__)
