# KDES: Kernel Density Estimation and Sampling

[![Build Status](https://travis-ci.org/lzkelley/kdes.svg?branch=master)](https://travis-ci.org/lzkelley/kdes)
[![codecov](https://codecov.io/gh/lzkelley/kdes/branch/master/graph/badge.svg)](https://codecov.io/gh/lzkelley/kdes)
:: master

[![Build Status](https://travis-ci.org/lzkelley/kdes.svg?branch=dev)](https://travis-ci.org/lzkelley/kdes)
:: dev

This package performs KDE operations on multidimensional data to: **1) calculate estimated PDFs** (probability distribution functions), and **2) to resample new data** from those PDFs.

## Installation

#### from source

```bash
git clone https://github.com/lzkelley/kdes.git
pip install -e kdes/
```


## Examples

Calculating a PDF and resampling with and without reflection:

#### 1D

![1D Samples with Reflection](docs/media/kde_1d_reflect.png)

#### 2D

![2D Samples with Reflection](docs/media/kde_2d_reflect.png)
