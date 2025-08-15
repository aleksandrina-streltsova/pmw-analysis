"""
This module contains copy-pasted functions from GPM library and should be removed.
"""
import numpy as np
import pandas as pd
import polars as pl


def compute_2d_histogram(df, x, y, var=None, x_bins=10, y_bins=10, x_labels=None, y_labels=None, prefix_name=True):
    """Compute bivariate statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    x : str
        Column name for x-axis binning (will be rounded to integers)
    y : str
        Column name for y-axis binning
    var : str, optional
        Column name for which statistics will be computed.
        If None, only counts are computed.
    x_bins : int or array-like
        Number of bins or bin edges for x
    y_bins : int or array-like
        Number of bins or bin edges for y
    x_labels : array-like, optional
        Labels for x bins. If None, uses bin centers
    y_labels : array-like, optional
        Labels for y bins. If None, uses bin centers

    Returns
    -------
    xarray.Dataset
        Dataset with dimensions corresponding to binned variables and
        data variables for each statistic
    """
    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    # Copy data
    df = df.copy()

    # If no var specified, create dummy variable
    var_specified = True
    if var is None:
        var = "dummy"
        df["dummy"] = np.ones(df[x].shape)
        var_specified = False

    # Handle x-axis binning
    if isinstance(x_bins, int):
        x_bins = np.linspace(df[x].min(), df[x].max(), x_bins + 1)

    # Handle y-axis binning
    if isinstance(y_bins, int):
        y_bins = np.linspace(df[y].min(), df[y].max(), y_bins + 1)

    # Drop rows where any of the key columns have NaN
    df = df.dropna(subset=[x, y, var])

    if len(df) == 0:
        raise ValueError("No valid data points after removing NaN values")

    # Create binned columns with explicit handling of out-of-bounds values
    df[f"{x}_binned"] = pd.cut(df[x], bins=x_bins, include_lowest=True)
    df[f"{y}_binned"] = pd.cut(df[y], bins=y_bins, include_lowest=True)

    # Create complete IntervalIndex for both dimensions
    x_intervals = df[f"{x}_binned"].cat.categories
    y_intervals = df[f"{y}_binned"].cat.categories

    # Prepare prefix
    prefix = f"{var}_" if prefix_name and var_specified else ""

    # Define statistics to compute
    if var_specified:
        list_stats = [
            ("count", "count"),
            (f"{prefix}median", "median"),
            (f"{prefix}std", "std"),
            (f"{prefix}min", "min"),
            (f"{prefix}max", "max"),
        ]
    else:
        list_stats = [("count", "count")]

    # Compute statistics
    df_stats = df.groupby([f"{x}_binned", f"{y}_binned"])[var].agg(list_stats)

    # Create MultiIndex with all possible combinations
    full_index = pd.MultiIndex.from_product([x_intervals, y_intervals], names=[f"{x}_binned", f"{y}_binned"])

    # Reindex to include all interval combinations
    df_stats = df_stats.reindex(full_index)

    # Determine coordinates
    x_centers = x_intervals.mid
    y_centers = y_intervals.mid

    # Use provided labels if available
    x_coords = x_labels if x_labels is not None else x_centers
    y_coords = y_labels if y_labels is not None else y_centers

    # Reset index and set new coordinates
    df_stats = df_stats.reset_index()
    df_stats[f"{x}"] = pd.Categorical(df_stats[f"{x}_binned"].map(dict(zip(x_intervals, x_coords, strict=False))))
    df_stats[f"{y}"] = pd.Categorical(df_stats[f"{y}_binned"].map(dict(zip(y_intervals, y_coords, strict=False))))

    # Set new MultiIndex with coordinates
    df_stats = df_stats.set_index([f"{x}", f"{y}"])
    df_stats = df_stats.drop(columns=[f"{x}_binned", f"{y}_binned"])

    # Convert to dataset
    ds = df_stats.to_xarray()

    # Transpose arrays
    ds = ds.transpose(y, x)
    return ds
