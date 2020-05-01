# kalepy: Kernel Density Estimation and Sampling

[![Build Status](https://travis-ci.org/lzkelley/kalepy.svg?branch=master)](https://travis-ci.org/lzkelley/kalepy)
[![codecov](https://codecov.io/gh/lzkelley/kalepy/branch/master/graph/badge.svg)](https://codecov.io/gh/lzkelley/kalepy)


![kalepy animated logo](https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/logo_anim_small.gif)

This package performs KDE operations on multidimensional data to: **1) calculate estimated PDFs** (probability distribution functions), and **2) resample new data** from those PDFs.

## Installation

#### from pypi (i.e. via pip)

```bash
pip install kalepy
```

#### from source (e.g. for development)

```bash
git clone https://github.com/lzkelley/kalepy.git
pip install -e kalepy/
```

In this case the package can easily be updated by changing into the source directory, pulling, and rebuilding:

```bash
cd kalepy
git pull
pip install -e .
# Optional: run unit tests (using the `nosetests` package)
nosetests
```
