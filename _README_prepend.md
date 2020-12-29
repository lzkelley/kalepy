# kalepy: Kernel Density Estimation and Sampling

[![Build Status](https://travis-ci.org/lzkelley/kalepy.svg?branch=master)](https://travis-ci.org/lzkelley/kalepy)
[![codecov](https://codecov.io/gh/lzkelley/kalepy/branch/master/graph/badge.svg)](https://codecov.io/gh/lzkelley/kalepy)
[![Documentation Status](https://readthedocs.org/projects/kalepy/badge/?version=latest)](https://kalepy.readthedocs.io/en/latest/?badge=latest)

![kalepy animated logo](https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/logo_anim_small.gif)

This package performs KDE operations on multidimensional data to: **1) calculate estimated PDFs** (probability distribution functions), and **2) resample new data** from those PDFs.

## Documentation

A number of examples (also used for continuous integration testing) are included in [the package notebooks](https://github.com/lzkelley/kalepy/tree/master/notebooks).  Some background information and references are included in [the JOSS paper]().

Full documentation is available on [kalepy.readthedocs.io](https://kalepy.readthedocs.io/en/latest/).

## README Contents

- [Installation](#Installation)
- Quickstart
    - [Basic Usage](#Basic-Usage)
    - [Fancy Usage](#Fancy-Usage)
- [Development & Contributions](#Development-&-Contributions)
- [Attribution (citation)](#Attribution)


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
