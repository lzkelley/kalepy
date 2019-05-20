.. kdes documentation master file, created by
   sphinx-quickstart on Mon May 20 14:31:16 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to kdes's documentation!
================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:


Theory
======

Introduction
------------

A **Kernel Density Estimate (KDE)** is a generalization of a histogram, with two key differences.  First, the 'bins' are no longer restricted to boxes (which are zero within the bin-width of a data-point, and zero elsewhere), and are instead smoothing **'kernels'** that can take on an arbitrary functional form (they could still be a box, but Gaussian, and other shapes are also common).  Second, the location of the bins are no longer arbitrary and fixed, but instead are centered on the datapoints.

Consider a set of :math:`n` data points :math:`X_i = \left{X_1, X_2, ..., X_n}`, drawn from a **Probability Distribution Function (PDF)** :math:`f(x)`.  Each data-point may be associated with a 'weight' :math:`w_i \,:\, \sum_{i=1}^n w_i = 1`, which could be associated with a measurement uncertainty, selection biases, a number of repetitions, etc.  A KDE can be used to calculate a reconstruction/estimate of the PDF, as:

.. math::

   \hat{f}(x) = \frac{1}{n} \sum_{i=1}^n w_i \, K\left(x \right).

Here, :math:`K(x)` is the **Kernel function**, which can often be written as :math:`K\left(\frac{x-X_i}{h}\right)` or :math:`K_h\left(x-X_i\right)`, where :math:`h` is the **bandwidth**.  In general, the bandwidth itself may be variable (e.g. :math:`h=h(x)` or :math:`h=h_i`).  The Kernel *must be normalized* such that :math:`\int_{-\infty}^{+\infty} K(x) dx = 1` (note that deopending on how the argument of the Kernel function is written, different authors sometimes include a pre-factor of :math:`1/h` to handle the jacobian, i.e. :math:`K(x) \rightarrow \frac{1}{h}K\left(\frac{x-X_i}{h}\right)`, to preserve the unitarity condition).  The Kernel is *usually* chosen to be a symmetric function, i.e. :math:`K(x) = K(-x)`, but this needn't be the case (and some solutions to boundary-problems, discussed below, employ assymetric kernels).

Many different kernel functions have been explored in the literation, with the Gaussian kernel being especially common,

.. math::
   
   K(x) = (2\pi h)^{-1/2} \exp\left[-\frac{1}{2}\left(\frac{x-X_i}{h}\right)^2\right].
   
Because the Gaussian kernel is non-zero for all :math:`x` (i.e. it has infinite support), calculations using it can be computationally intensive.  Kernel's with finite-support are also useful, the simplest example being the box:

.. math::
   
   K(x) &= \frac{1}{2h}, \,\,\, \frac{|x-X_i|}{h} < 1 \\
        &= \,\, 0.
   
Note that :math:`x` can be a vector in an arbitrary number of dimensions :math:`d`, in which case the kernel must be multivariate (i.e. multidimensional), in which case the bandwidth, generally, must be a :math:`d \times d` matrix :math:`H`,

.. math::

   K(y) = \left(2\pi\right)^{-d/2} |H|^{-1/2} \, \exp\left(y^T H^{-1} y \right), \,\,\, \rm{s.t.} \,\,\, y = x - X_i


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
