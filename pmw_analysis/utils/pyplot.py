"""
This module provides utilities for plotting with `matplotlib.pyplot`.
"""
from collections import defaultdict
from typing import List

import matplotlib.colors as mcolors
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
        "Elevated snow cover": "white",
        "High snow cover": "lightgray",
        "Moderate snow cover": "gray",
        "Light snow cover": "darkgray",
        "Standing Water": "navy",
        "Ocean or water Coast": "deepskyblue",
        "Mixed land/ocean or water coast": "cornflowerblue",
        "Land coast": "gold",
        "Sea-ice edge": "slateblue",
        "Mountain rain": "orange",
        "Mountain snow": "snow",
    })
    # Create a custom colormap
    cmap = mcolors.ListedColormap([surface_to_color[surface_type] for surface_type in surface_types])
    bounds = range(0, len(surface_types))
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm
