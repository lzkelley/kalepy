"""
"""

import os
import shutil

import numpy as np
import tqdm
import matplotlib.pyplot as plt

import kalepy as kale

NUM = int(1e4)
np.random.seed(123456)
num_1 = NUM//2
num_2 = NUM - num_1
a1 = np.random.normal(4.0, 1.0, num_1)
loc = 0.3
a2 = np.random.lognormal(loc, 0.5, size=num_2)
data = np.concatenate([a1, a2])
np.random.seed(1234)
np.random.shuffle(data)

grid = np.linspace(-0.2, 8.0, 1000)
# edges = np.linspace(-0.2, 8.0, 10)

true_pdf = (num_1/NUM)*np.power(2*np.pi, -1/2) * np.exp(-(grid-4.0)**2/2)
true_pdf += (num_2/NUM)*np.power(2*np.pi, -1/2)/(0.5*grid) * np.exp(-(np.log(grid)-loc)**2/0.5)


kde_full = kale.KDE(data, kernel='Parabola', bandwidth=0.3)
pdf_full = kde_full.pdf(grid)
np.random.seed(4321)
samp_full = kde_full.resample(NUM)
edges_full = np.linspace(-0.2, 8.0, 40)

output_path = os.path.join(os.path.curdir, "anim", "")
if os.path.exists(output_path) and os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

fname_base = os.path.join(output_path, "anim_{:05d}.png")

fig, axes = plt.subplots(figsize=[10, 4], ncols=2, sharey=True)
plt.subplots_adjust(left=0.04, bottom=0.08, right=0.98, top=0.92, wspace=0.05)

vals = [np.arange(2, 25),
        np.arange(25, 50, 2),
        np.arange(50, 100, 5),
        np.arange(100, 1000, 10),
        np.arange(1000, NUM+1, 100)]

vals = np.concatenate(vals, axis=0)


def repeat(fname, times):
    ref = None
    for tt in range(times+1):
        parts = fname.split('.')
        parts[-2] += "_{:02d}".format(tt)
        temp = ".".join(parts)
        if tt == 0:
            shutil.move(fname, temp)
            ref = temp
        else:
            shutil.copy(ref, temp)

    return


titles = ['KDE', 'Resample']

# for ii, NN in enumerate(tqdm.tqdm_notebook(vals)):
for ii, NN in enumerate(vals):
    last = (ii == len(vals)-1)
    for jj, ax in enumerate(axes):
        ax.cla()
        ax.grid(alpha=0.2)
        ax.set_title(titles[jj])
        ax.set_ylim([-0.04, 0.5])
        ax.plot(grid, true_pdf, 'k--', alpha=0.2)

    num_edge = np.min([40, 5*int(np.sqrt(NN))])
    edges = np.linspace(-0.2, 8.0, num_edge)

    ax = axes[0]
    samp = data[:NN]

    kde = kale.KDE(samp, kernel='Parabola')
    pdf = kde.pdf(grid)
    bw = np.sqrt(kde.bandwidth.matrix[0, 0])

    ax.plot(grid, pdf, 'b-', lw=2.0)
    ax.plot(samp, -0.02*np.ones_like(samp), 'o', color='blue', alpha=0.1)

    ax.hist(samp, edges, rwidth=0.9, alpha=0.2, facecolor='dodgerblue', edgecolor='blue', density=True)

    label = "$N: {:5d}$\n$B: {:5d}$\n$h: {:.2f}$".format(NN, num_edge-1, bw)
    ax.text(0.98, 0.98, label, ha='right', va='top', transform=ax.transAxes)

    if NN < 20:
        for ss in samp:
            ps = kale.kernels.Parabola.evaluate(grid, ref=ss, bw=bw)/NN
            idx = (ps > 0)
            ax.plot(grid[idx], ps[idx], 'b--', alpha=0.5, lw=0.5)

    ax = axes[1]
    samp = samp_full[:NN]

    ax.plot(grid, pdf_full, 'k-', lw=2.0, alpha=0.5)
    ax.hist(data, edges_full, rwidth=0.9, alpha=0.2, facecolor='0.5', edgecolor='k', density=True)
    ax.hist(samp, edges, rwidth=0.9, alpha=0.2, facecolor='firebrick', edgecolor='red', density=True)
    ax.plot(samp, -0.02*np.ones_like(samp), 'o', color='red', alpha=0.1)

    fname = fname_base.format(NN)
    fig.savefig(fname, dpi=100, rasterize=True)

#     if ii > 33:
#         break
#     continue

    if ii < 13:
        repeat(fname, 4)
    elif ii < 23:
        repeat(fname, 2)
    elif last:
        repeat(fname, 10)

plt.show()
