"""
This module provides utilities for plotting with `matplotlib.pyplot`.
"""
from collections import defaultdict
from pathlib import Path
from typing import List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def scatter_na(x, y, c, color_label: str = None):
    """
    Plot a scatter plot. Points corresponding to missing values in the `c` array will be plotted with red color,
    while non-missing values are colorized based on `c`.
    """
    na_mask = pd.isna(c)
    vmin = np.percentile(c[~na_mask], 1)
    vmax = np.percentile(c[~na_mask], 99)

    plt.scatter(x[na_mask], y[na_mask], c="r", s=1, zorder=0)
    scatter = plt.scatter(x[~na_mask], y[~na_mask], c=c[~na_mask], vmin=vmin, vmax=vmax, s=1, zorder=1)
    plt.colorbar(scatter, label=color_label)


def subplots(nrows: int, ncols: int, xscale: int = 3, yscale: int = 3):
    """
    Create a figure and a set of subplots, adjusting the figure size based on the number of rows and columns.
    """
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * xscale, nrows * yscale), dpi=300)
    return fig, axs


# TODO: make it possible to provide additional colors; add a warning
def get_surface_type_cmap(surface_types: List[str]):
    """
    Get color map corresponding to provided surface types.
    """
    # Define surface type indexes and corresponding colors
    missing_surface_color = "black"
    surface_to_color = defaultdict(lambda: missing_surface_color, {
        "Ocean": "blue",
        "Sea-Ice": "cyan",
        "High vegetation": "darkgreen",
        "Medium vegetation": "green",
        "Low vegetation": "yellowgreen",
        "Sparse vegetation": "lightgreen",
        "Desert": "tan",
        "Elevated snow cover": "mediumpurple",
        "High snow cover": "rebeccapurple",
        "Moderate snow cover": "darkorchid",
        "Light snow cover": "mediumslateblue",
        "Standing Water": "navy",
        "Ocean or water Coast": "deepskyblue",
        "Mixed land/ocean or water coast": "cornflowerblue",
        "Land coast": "gold",
        "Sea-ice edge": "slateblue",
        "Mountain rain": "orange",
        "Mountain snow": "lavender",
    })
    # Create a custom colormap
    cmap = mcolors.ListedColormap([surface_to_color[surface_type] for surface_type in surface_types])
    bounds = range(0, len(surface_types))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def generate_colors(n: int, colormap: str = "tab10"):
    cmap = plt.get_cmap(colormap)
    return [cmap(i / max(1, n - 1)) for i in range(n)]


def plot_histogram2d(data, weight, ax, vmin=None, vmax=None, cmap="rocket_r", norm=None, alpha=1.0, bins=10, title=None,
                     cbar=False):
    hist, xedges, yedges = np.histogram2d(data[:, 0], data[:, 1], range=range, weights=weight, bins=bins)
    sns.heatmap(hist, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, annot=False, ax=ax, alpha=alpha, cbar=cbar)
    ax.set_title(title)


def plot_histograms2d(datas, weights, titles, path: Path, colors=None, cmaps=None, bins=10,
                      use_log_norm=True, alpha: int | float | List = 1.0):
    if cmaps is None:
        if colors is None:
            colors = generate_colors(len(datas))
        cmaps = [mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", c]) for c in colors]

    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf

    for data in datas:
        xmin = min(xmin, data[:, 0].min())
        xmax = max(xmax, data[:, 0].max())
        ymin = min(ymin, data[:, 1].min())
        ymax = max(ymax, data[:, 1].max())

    range = ((xmin, xmax), (ymin, ymax))

    # Build ticks and labels
    n_ticks = min(bins, 10)

    x_ticks = np.linspace(0, bins - 1, n_ticks)
    y_ticks = np.linspace(0, bins - 1, n_ticks)

    x_tick_labels = np.round(np.linspace(xmin, xmax, n_ticks), 2)
    y_tick_labels = np.round(np.linspace(ymin, ymax, n_ticks), 2)

    hists = []

    for data, weight in zip(datas, weights):
        hist = np.histogram2d(data[:, 0], data[:, 1], range=range, weights=weight, bins=bins)[0]
        hist = hist.T
        hists.append(hist)

    # Plot histograms separately
    nrows = min(len(datas), 3)
    ncols = (len(datas) + nrows - 1) // nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))

    for i, hist in enumerate(hists):
        vmin = hist.min()
        vmax = hist.max()

        if use_log_norm:
            vmin = max(1, vmin)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

        if ncols == 1:
            if nrows == 1:
                ax: plt.Axes = axes
            else:
                ax: plt.Axes = axes[i]
        else:
            ax: plt.Axes = axes[i // ncols][i % ncols]

        sns.heatmap(hist, vmin=vmin, vmax=vmax, cmap=cmaps[i], norm=norm, annot=False, ax=ax)
        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_yticks(y_ticks, y_tick_labels)
        ax.set_title(titles[i])

    fig.tight_layout()
    fig.savefig(path)
    fig.show()

    if len(datas) == 1:
        return

    # Plot histograms combined
    vmin_combined = np.inf
    vmax_combined = -np.inf

    for hist in hists:
        vmin_combined = min(vmin_combined, hist.min())
        vmax_combined = max(vmax_combined, hist.max())

    if use_log_norm:
        vmin_combined = max(1, vmin_combined)
        norm_combined = mcolors.LogNorm(vmin=1, vmax=vmax_combined)
    else:
        norm_combined = None
    fig_combined, ax_combined = plt.subplots(figsize=(8, 6))

    for i, hist in enumerate(hists):
        if isinstance(alpha, int) or isinstance(alpha, float):
            a = alpha
        else:
            a = alpha[i]

        # plot histograms combined
        sns.heatmap(hist, vmin=vmin_combined, vmax=vmax_combined, cmap=cmaps[i], norm=norm_combined,
                    annot=False, ax=ax_combined, alpha=a, cbar=False)
        ax_combined.set_xticks(x_ticks, x_tick_labels)
        ax_combined.set_yticks(y_ticks, y_tick_labels)

    fig_combined.savefig(path.parent / f"{path.stem}_combined.png")
    fig_combined.show()
