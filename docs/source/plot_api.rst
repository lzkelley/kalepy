
.. code:: ipython3

    import kalepy as kale
    import numpy as np
    import matplotlib.pyplot as plt

Top Level Functions
-------------------

kalepy.corner() and the kalepy.Corner class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the full documentation, see:

* `kalepy.plot.corner <kalepy_plot.html#kalepy.plot.corner>`_
* `kalepy.plot.Corner <kalepy_plot.html#kalepy.plot.Corner>`_
* `kalepy.plot.Corner.plot <kalepy_plot.html#kalepy.plot.Corner.plot>`_

Plot some three-dimensional data called ``data3`` with shape (3, N) with
``N`` data points.

.. code:: ipython3

    kale.corner(data3);




.. parsed-literal::

    (<kalepy.plot.Corner at 0x7ff9b6f08b50>,
     <matplotlib.lines.Line2D at 0x7ff9ba7ac550>)




.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_plot_files/demo_plot_9_1.png


Extensive modifications are possible with passed arguments, for example:

.. code:: ipython3

    # 1D plot settings: turn on histograms, and modify the confidence-interval quantiles
    dist1d = dict(hist=True, quantiles=[0.5, 0.9])
    # 2D plot settings: turn off the histograms, and turn on scatter
    dist2d = dict(hist=False, scatter=True)
    
    kale.corner(data3, labels=['a', 'b', 'c'], color='purple',
                dist1d=dist1d, dist2d=dist2d);




.. parsed-literal::

    (<kalepy.plot.Corner at 0x7ff9bb17a250>,
     <matplotlib.lines.Line2D at 0x7ff9bb8f9460>)




.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_plot_files/demo_plot_11_1.png


The ``kalepy.corner`` method is a wrapper that builds a
``kalepy.Corner`` instance, and then plots the given data. For
additional flexibility, the ``kalepy.Corner`` class can be used
directly. This is particularly useful for plotting multiple
distributions, or using preconfigured plotting styles.

.. code:: ipython3

    # Construct a `Corner` instance for 3 dimensional data, modify the figure size
    corner = kale.Corner(3, figsize=[9, 9])
    
    # Plot two different datasets using the `clean` plotting style
    corner.clean(data3a)
    corner.clean(data3b);




.. parsed-literal::

    <matplotlib.lines.Line2D at 0x7ff9bab90790>




.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_plot_files/demo_plot_13_1.png


kalepy.dist1d and kalepy.dist2d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Corner`` class ultimately calls the functions ``dist1d`` and
``dist2d`` to do the actual plotting of each figure panel. These
functions can also be used directly.

For the full documentation, see:

* `kalepy.plot.dist1d <kalepy_plot.html#kalepy.plot.dist1d>`_
* `kalepy.plot.dist2d <kalepy_plot.html#kalepy.plot.dist2d>`_


.. code:: ipython3

    # Plot a 1D dataset, shape: (N,) for `N` data points
    kale.dist1d(data1);




.. parsed-literal::

    <matplotlib.lines.Line2D at 0x7ff9bbb7c0d0>




.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_plot_files/demo_plot_17_1.png


.. code:: ipython3

    # Plot a 2D dataset, shape: (2, N) for `N` data points
    kale.dist2d(data2, hist=False);



.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_plot_files/demo_plot_18_0.png


These functions can also be called on a ``kalepy.KDE`` instance, which
is particularly useful for utilizing the advanced KDE functionality like
reflection.

.. code:: ipython3

    # Construct a random dataset, and truncate it on the left at 1.0
    import numpy as np
    data = np.random.lognormal(sigma=0.5, size=int(3e3))
    data = data[data >= 1.0]
    
    # Construct a KDE, and include reflection (only on the lower/left side)
    kde_reflect = kale.KDE(data, reflect=[1.0, None])
    # plot, and include confidence intervals
    hr = kale.dist1d(kde_reflect, confidence=True);



.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_plot_files/demo_plot_20_0.png


.. code:: ipython3

    # Load a predefined 2D, 'random' dataset that includes boundaries on both dimensions
    data = kale.utils._random_data_2d_03(num=1e3)
    # Initialize figure
    fig, axes = plt.subplots(figsize=[10, 5], ncols=2)
    
    # Construct a KDE included reflection
    kde = kale.KDE(data, reflect=[[0, None], [None, 1]])
    
    # plot using KDE's included reflection parameters
    kale.dist2d(kde, ax=axes[0]);
    
    # plot data without reflection
    kale.dist2d(data, ax=axes[1], cmap='Reds')
    
    titles = ['reflection', 'no reflection']
    for ax, title in zip(axes, titles):
        ax.set(xlim=[-0.5, 2.5], ylim=[-0.2, 1.2], title=title)



.. image:: https://raw.githubusercontent.com/lzkelley/kalepy/dev/docs/media/demo_plot_files/demo_plot_21_0.png

