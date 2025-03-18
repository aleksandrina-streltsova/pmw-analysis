"""
This module provides utilities for plotting with `matplotlib.pyplot`.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
