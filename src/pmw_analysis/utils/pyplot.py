"""
This module provides utilities for plotting with `matplotlib.pyplot`.
"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


def scatter_na(x, y, c, color_label: str = None):
    """
    Plot a scatter plot. Points corresponding to missing values in the `c` array will be plotted with red color,
    while non-missing values are colored based on `c`.
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
    """
    Generate n colors from a colormap. By default, "tab10" colormap is used.
    """
    cmap = plt.get_cmap(colormap)
    return [cmap(i / max(1, n - 1)) for i in range(n)]


@dataclass
class HistogramData:
    """
    Represents a data structure for histogram configuration and data storage.
    """
    data: np.ndarray | pl.DataFrame | pd.DataFrame
    weight: np.ndarray | pl.Series | pd.Series
    title: str
    alpha: float
    cmap: str | mcolors.Colormap | None
    color: str | None


def plot_histograms2d(hist_datas: List[HistogramData], path: Path, bins: int = 10,
                      use_log_norm=True, use_shared_norm=True):
    """
    Plot multiple 2D histograms from datasets and a combined visualization where
    2D histograms are overlayed.
    """
    generated_colors = generate_colors(len(hist_datas))

    for i, hist_data in enumerate(hist_datas):
        if hist_data.cmap is None:
            color = hist_data.color if hist_data.color is not None else generated_colors.pop()
            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", color])
            hist_data.cmap = cmap

    hist_range = _calculate_histogram_range(hist_datas)

    # Build ticks and labels
    n_ticks = min(bins, 10)

    x_ticks = np.linspace(0, bins - 1, n_ticks)
    y_ticks = np.linspace(0, bins - 1, n_ticks)

    x_tick_labels = np.round(np.linspace(hist_range[0][0], hist_range[0][1], n_ticks), 2)
    y_tick_labels = np.round(np.linspace(hist_range[1][0], hist_range[1][1], n_ticks), 2)

    hists = []

    for hist_data in hist_datas:
        hist = np.histogram2d(hist_data.data[:, 0], hist_data.data[:, 1],
                              range=hist_range, weights=hist_data.weight, bins=bins)[0]
        hist = hist.T
        hists.append(hist)

    # Plot histograms separately
    k = 3 if len(hist_datas) % 3 == 0 else 2

    nrows = min(len(hist_datas), k)
    ncols = (len(hist_datas) + nrows - 1) // nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows))

    for i, (hist_data, hist) in enumerate(zip(hist_datas, hists)):
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

        sns.heatmap(hist, vmin=vmin, vmax=vmax, cmap=hist_data.cmap, norm=norm, annot=False, ax=ax)
        ax.set_xticks(x_ticks, x_tick_labels)
        ax.set_yticks(y_ticks, y_tick_labels)
        ax.set_title(hist_data.title)

    fig.tight_layout()
    fig.savefig(path)
    fig.show()

    if len(hist_datas) == 1:
        return

    # Plot histograms combined
    vmin_combined = np.inf
    vmax_combined = -np.inf

    for hist in hists:
        vmin_combined = min(vmin_combined, hist.min())
        vmax_combined = max(vmax_combined, hist.max())

    norm_combined = _get_norm(vmin_combined, vmax_combined, use_log_norm)
    fig_combined, ax_combined = plt.subplots(figsize=(8, 6))

    for i, (hist_data, hist) in enumerate(zip(hist_datas, hists)):
        # plot histograms combined
        norm = norm_combined if use_shared_norm else _get_norm(hist.min(), hist.max(), use_log_norm)
        sns.heatmap(hist, cmap=hist_data.cmap, norm=norm, mask=hist==0,
                    annot=False, ax=ax_combined, alpha=hist_data.alpha, cbar=False)
        ax_combined.set_xticks(x_ticks, x_tick_labels)
        ax_combined.set_yticks(y_ticks, y_tick_labels)

    fig_combined.savefig(path.parent / f"{path.stem}_combined.png")
    fig_combined.show()


def finalize_axis(axes: plt.Axes,
                  title: Optional[str],
                  x_label: Optional[str],
                  y_label: Optional[str],
                  ):
    """
    Finalize plot by setting title and labels.
    """
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    axes.legend()


def _calculate_histogram_range(hist_datas: List[HistogramData]):
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf

    for hist_data in hist_datas:
        x_min = min(x_min, hist_data.data[:, 0].min())
        x_max = max(x_max, hist_data.data[:, 0].max())
        y_min = min(y_min, hist_data.data[:, 1].min())
        y_max = max(y_max, hist_data.data[:, 1].max())

    return (x_min, x_max), (y_min, y_max)


def _get_norm(vmin, vmax, use_log_norm):
    if use_log_norm:
        vmin = max(1, vmin)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return norm