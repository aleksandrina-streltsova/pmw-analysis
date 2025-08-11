"""
This module contains utilities for analyzing quantized transformed data.
"""
import pathlib
from typing import List, Callable, Tuple

import configargparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.colors import LogNorm
from tqdm import tqdm

from pmw_analysis.constants import (
    DIR_PMW_ANALYSIS,
    COLUMN_COUNT, COLUMN_ACCUM_UNIQUE, COLUMN_ACCUM_ALL, COLUMN_OCCURRENCE_TIME,
    ST_GROUP_SNOW, ST_GROUP_OCEAN, ST_GROUP_VEGETATION,
    VARIABLE_SURFACE_TYPE_INDEX,
    TC_COLUMNS, ST_COLUMNS, COLUMN_OCCURRENCE, ArgEDA, ArgTransform, DIR_IMAGES, DIR_HISTS,
)
from pmw_analysis.copypaste.utils.cli import EnumAction
from pmw_analysis.quantization.dataframe_polars import expand_occurrence_column
from pmw_analysis.processing.filter import filter_by_surface_type
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.io import combine_paths, file_to_dir
from pmw_analysis.utils.polars import get_column_ranges, take_k_sorted
from pmw_analysis.utils.pyplot import finalize_axis


def plot_point_accumulation(path: pathlib.Path, var: str | None):
    """
    Plot the accumulation of unique points over time for a given dataset.

    Parameters
    ----------
        path : pathlib.Path
            Path to the input parquet file containing the dataset.
        var : str | None
            Optional variable name to perform additional analysis.

    """
    df = pl.read_parquet(path)
    columns = [COLUMN_OCCURRENCE_TIME, COLUMN_COUNT] + ([] if var is None else [var, f"{var}_count"])
    df = expand_occurrence_column(df).select(columns)
    df = df.with_columns(pl.col(COLUMN_OCCURRENCE_TIME).dt.round("1d"))
    df = df.with_columns(pl.col(COLUMN_COUNT).cast(pl.UInt64))
    df_count = _cum_sum_over_time(df)

    fig, axes = _plot_count_over_time(df_count)
    fig_var = _plot_var_over_time(df, var, axes)


    images_dir = combine_paths(path_base=DIR_IMAGES, path_rel=file_to_dir(path),
                               path_rel_base=DIR_PMW_ANALYSIS) / "over_time"
    images_dir.mkdir(parents=True, exist_ok=True)

    for fig, filename in [(fig, f"count_over_time{"" if var is None else f"_{var}"}.png"),
                          (fig_var, f"var_over_time{"" if var is None else f"_{var}"}.png")]:
        fig.tight_layout()
        fig.savefig(images_dir / filename)


def _plot_count_over_time(df_count) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, col_accum, title_prefix in zip(axes, [COLUMN_ACCUM_ALL, COLUMN_ACCUM_UNIQUE],
                                           ["Signatures first", "Unique signatures"]):
        ax.plot(df_count[COLUMN_OCCURRENCE_TIME], df_count[col_accum], color="b", label="All")
        ax.set_yscale("log")
        finalize_axis(ax, title=f"{title_prefix} seen before this time", x_label="Time", y_label="Cumulative count")

    return fig, axes


def _plot_var_over_time(df: pl.DataFrame, var: str, axes: np.ndarray[plt.Axes]) -> plt.Figure:
    df_var_count_desc, df_var_mean_desc, df_var_not_null = _calculate_reverse_cumsum_for_variable(df, var)
    count_total_all = df_var_not_null[COLUMN_COUNT].sum()
    count_total_unique = len(df_var_not_null)

    for i, (col_accum, count_total) in enumerate(zip([COLUMN_ACCUM_ALL, COLUMN_ACCUM_UNIQUE],
                                                     [count_total_all, count_total_unique])):
        axes[i].plot(df_var_count_desc[COLUMN_OCCURRENCE_TIME], count_total - df_var_count_desc[col_accum],
                     color="g", label=f"Not-null '{var}'")

    fig_var, (axes_var, axes_count) = plt.subplots(2, 2, figsize=(15, 10))
    for i, col_accum in enumerate([COLUMN_ACCUM_ALL, COLUMN_ACCUM_UNIQUE]):
        axes_var[i].plot(df_var_mean_desc[COLUMN_OCCURRENCE_TIME], df_var_mean_desc[col_accum], color="r", label=var)
        finalize_axis(axes_var[i], title=f"Mean of '{var}' (from this time to end)", x_label="Time", y_label=var)

        axes_count[i].plot(df_var_mean_desc[COLUMN_OCCURRENCE_TIME], df_var_count_desc[col_accum],
                           color="g", label=f"Not-null '{var}'")
        finalize_axis(axes_count[i], title="Cumulative count (from this time to end)", x_label="Time", y_label="Count")
        axes_count[i].set_yscale("log")

    return fig_var


def analyze(path: pathlib.Path, var: str | None, transform: Callable, k: int | None):
    """
    Generate pairplots of features for a given dataset.

    Parameters
    ----------
        path : pathlib.Path
            Path to the input parquet file containing the dataset.
        var : str | None
            Optional variable name to color pairplots by. If None, pairplots are colored by counts of signatures.

    """
    df_merged = pl.read_parquet(path)
    if COLUMN_COUNT not in df_merged.columns:
        df_merged = df_merged.with_columns(pl.lit(1).alias(COLUMN_COUNT))

    feature_columns = transform(TC_COLUMNS)
    feature_columns = [col for col in feature_columns if col != var]

    columns = (
            feature_columns +
            [VARIABLE_SURFACE_TYPE_INDEX, COLUMN_COUNT, COLUMN_OCCURRENCE] +
            ([] if var is None else [var])
    )
    df = df_merged[[col for col in columns if col in df_merged.columns]]
    if k is not None:
        df_k = take_k_sorted(df, COLUMN_OCCURRENCE, k, COLUMN_COUNT, descending=True)
    else:
        df_k = None
    # df = df[:10000]  # for testing

    n_bins = []
    for tc_col in feature_columns:
        value_count = df[[tc_col, COLUMN_COUNT]].group_by(tc_col).sum()
        null_count = value_count.filter(pl.col(tc_col).is_null())[COLUMN_COUNT]
        if len(null_count) > 0:
            null_count = null_count.item()
        else:
            null_count = 0
        unique_value_count = len(value_count.filter(pl.col(tc_col).is_not_null()))

        print(f"{tc_col} has {unique_value_count} unique non-null values.")
        print(f"{tc_col} has {null_count} missing values.")

        n_bins.append(min(unique_value_count, 50))

    n_bins = np.array(n_bins)

    feature_ranges = get_column_ranges(df, feature_columns)

    if VARIABLE_SURFACE_TYPE_INDEX in df_merged.columns:
        groups = [
            (None, None, None),
            ("Ocean (Group)", ST_GROUP_OCEAN, "navy"),
            ("Vegetation (Group)", ST_GROUP_VEGETATION, "darkgreen"),
            ("Snow (Group)", ST_GROUP_SNOW, "rebeccapurple"),
        ]
    else:
        groups = [(None, None, None)]

    for group in groups:
        _analyze_surface_type_group(df, df_k, feature_columns, group, var, feature_ranges, n_bins, path)


def _sorted_not_null_unique_values(df: pl.DataFrame, col: str) -> pl.Series:
    return df[col].unique().drop_nulls().sort()


def _cum_sum_over_time(df: pl.DataFrame, var: str | None = None, descending: bool = False) -> pl.DataFrame:
    if var is None:
        df = df.group_by(COLUMN_OCCURRENCE_TIME).agg(
            pl.len().alias(COLUMN_ACCUM_UNIQUE),
            pl.col(COLUMN_COUNT).cast(pl.UInt64).sum().alias(COLUMN_ACCUM_ALL)
        )
    else:
        df = df.group_by(COLUMN_OCCURRENCE_TIME).agg(
            pl.col(var).sum().alias(COLUMN_ACCUM_UNIQUE),
            pl.col(var).mul(pl.col(COLUMN_COUNT)).sum().alias(COLUMN_ACCUM_ALL)
        )

    df = df.sort(COLUMN_OCCURRENCE_TIME, descending=descending)
    df = df.with_columns(
        pl.col(COLUMN_ACCUM_UNIQUE).cum_sum(),
        pl.col(COLUMN_ACCUM_ALL).cum_sum(),
    )
    if descending:
        df = df.reverse()
    return df


def _calculate_reverse_cumsum_for_variable(df, var):
    var_count = f"{var}_{COLUMN_COUNT}"

    df_var_not_null = df.filter(pl.col(var_count).ne(0).and_(pl.col(var).is_not_nan()))  # TODO: fix
    df_var_not_null = df_var_not_null.drop(COLUMN_COUNT).rename({var_count: COLUMN_COUNT})
    df_var_not_null = df_var_not_null.with_columns(pl.col(COLUMN_COUNT).cast(pl.UInt64))

    df_var_sum_desc = _cum_sum_over_time(df_var_not_null, var=var, descending=True)
    df_var_count_desc = _cum_sum_over_time(df_var_not_null, descending=True)

    df_var_mean_desc = df_var_sum_desc.with_columns(
        df_var_sum_desc[COLUMN_ACCUM_ALL] / df_var_count_desc[COLUMN_ACCUM_ALL],
        df_var_sum_desc[COLUMN_ACCUM_UNIQUE] / df_var_count_desc[COLUMN_ACCUM_UNIQUE],
    )
    return df_var_count_desc, df_var_mean_desc, df_var_not_null


def _analyze_surface_type_group(df, df_k, feature_columns, group, var, feature_ranges, n_bins, path):
    group_name, surface_types, color = group

    if group_name is None:
        flag_values = None
        cmap = "rocket_r"
    else:
        flag_values = [idx + 1 for idx, st in enumerate(ST_COLUMNS) if st in surface_types]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", color])

    df_to_use = df
    df_to_use_k = df_k

    if flag_values is not None:
        df_to_use = filter_by_surface_type(df_to_use, flag_values)

        if df_to_use_k is not None:
            df_to_use_k = filter_by_surface_type(df_to_use_k, flag_values)

    if var is not None:
        cmap = "YlGnBu"

    hists_mtx = _calculate_pairplots_concat_matrix(df_to_use, feature_columns, feature_ranges, n_bins, var)
    if df_to_use_k is not None:
        hists_mtx_k = _calculate_pairplots_concat_matrix(df_to_use_k, feature_columns, feature_ranges, n_bins, var)
    else:
        hists_mtx_k = None

    images_dir = combine_paths(path_base=DIR_IMAGES, path_rel=file_to_dir(path), path_rel_base=DIR_PMW_ANALYSIS)
    images_dir.mkdir(parents=True, exist_ok=True)

    hists_dir = combine_paths(path_base=DIR_HISTS, path_rel=file_to_dir(path), path_rel_base=DIR_PMW_ANALYSIS)
    hists_dir.mkdir(parents=True, exist_ok=True)

    hist_path = hists_dir / f"{group_name}.npy"
    np.save(hist_path, hists_mtx)

    if var is None:
        vmin = max(hists_mtx.min(), 1)
        vmax = max(hists_mtx.max(), 1)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    fig, ax = plt.subplots(1, 1, figsize=(4 * len(feature_columns) + 4, 4 * len(feature_columns)))
    _plot_heatmap_with_varying_cell_sizes(hists_mtx, hists_mtx_k, n_bins, cmap, norm, ax, feature_ranges)

    _set_histograms2d_label(fig, ax, feature_columns)
    _set_histograms2d_label(fig, ax, feature_columns[::-1], is_y=True)

    title_suffix = "" if var is None else f" ({var})"
    fig_suffix = "" if df_k is None else "_k"
    fig_suffix += "" if var is None else f"_{var}"

    if group_name is not None:
        fig.suptitle(group_name + title_suffix, fontsize=30)
        fig.savefig(images_dir / f"count_{group_name.replace("/", "_")}_{len(feature_columns)}{fig_suffix}.png")
    else:
        fig.suptitle("All Surfaces" + title_suffix, fontsize=30)
        fig.savefig(images_dir / f"count_{group_name}_{len(feature_columns)}{fig_suffix}.png")


def _plot_heatmap_with_varying_cell_sizes(hists_mtx: np.ndarray, hists_mtx_k: np.ndarray | None, n_bins: np.ndarray,
                                          cmap, norm, ax: plt.Axes, feature_ranges: np.ndarray):
    xs = np.ones(n_bins.sum())
    n_bins_cumsum = np.cumsum(n_bins)
    n_bins_cumsum = np.insert(n_bins_cumsum, 0, 0)
    lines = n_bins.min() * np.arange(len(n_bins) + 1)
    for n, offset in zip(n_bins, n_bins_cumsum):
        xs[offset:offset + n] *= n_bins.min() / n
    xs = np.cumsum(xs)
    bounds = (xs[:-1] + xs[1:]) / 2
    bounds = np.concatenate([[2 * bounds[0] - bounds[1]], bounds, [2 * bounds[-1] - bounds[-2]]])

    c = ax.pcolormesh(bounds, (bounds[-1] - bounds)[::-1], hists_mtx[::-1], cmap=cmap, norm=norm)
    if hists_mtx_k is not None:
        cmap_k = "viridis"
        ax.pcolormesh(bounds, (bounds[-1] - bounds)[::-1], hists_mtx_k[::-1],
                      cmap=cmap_k, alpha=hists_mtx_k[::-1] > 0)

    ax.vlines(lines[1:-1], ymin=lines[0], ymax=lines[-1], colors='black')
    ax.hlines(lines[1:-1], xmin=lines[0], xmax=lines[-1], colors='black')

    n_ticks = 15
    x_tick_labels = np.concatenate([np.round(np.linspace(min_val, max_val, n_ticks + 1)[1:], decimals=2)
                                    for min_val, max_val in feature_ranges])
    y_tick_labels = np.concatenate([np.round(np.linspace(min_val, max_val, n_ticks + 1)[1:], decimals=2)[::-1]
                                    for min_val, max_val in feature_ranges])
    x_ticks = np.ones(n_ticks * len(n_bins)) * n_bins.min() / n_ticks

    x_ticks = np.cumsum(x_ticks)
    y_ticks = x_ticks[::-1]

    ax.set_xticks(x_ticks, x_tick_labels, fontsize=10, rotation=90)
    ax.set_yticks(y_ticks, y_tick_labels, fontsize=10, rotation=0)

    ax.set_xlim((bounds[0], bounds[-1]))
    ax.set_ylim((bounds[0], bounds[-1]))

    cb = plt.colorbar(c, ax=ax)
    cb.ax.tick_params(labelsize=15)


def _plot_hist(value_count_pd: pd.DataFrame, tc_col, group_name, images_dir: pathlib.Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    # TODO: width
    ax.bar(value_count_pd[tc_col], value_count_pd[COLUMN_COUNT], width=0.05)
    ax.set_xlabel(tc_col)
    ax.set_ylabel(COLUMN_COUNT)
    ax.set_yscale("log")
    ax.set_title(f"{tc_col} {group_name}")
    fig.tight_layout()
    fig.savefig(images_dir / f"{tc_col}_{group_name}.png")


def _calculate_pairplots_concat_matrix(df: pl.DataFrame, feature_columns: List[str],
                                       feature_ranges: np.ndarray, n_bins: np.ndarray,
                                       var: str | None):
    n_bins_cumsum = np.cumsum(n_bins)
    n_bins_cumsum = np.insert(n_bins_cumsum, 0, 0)

    hists_mtx = np.zeros((n_bins_cumsum[-1], n_bins_cumsum[-1]))
    for idx1, tc_col1 in tqdm(enumerate(feature_columns)):
        for idx2, tc_col2 in enumerate(feature_columns[:idx1 + 1]):
            cols = [tc_col1] if idx1 == idx2 else [tc_col1, tc_col2]

            counts = df.select(cols + [COLUMN_COUNT]).group_by(cols, maintain_order=False).sum()
            hist_range = (feature_ranges[idx1], feature_ranges[idx2])
            hist = np.histogram2d(counts[tc_col1], counts[tc_col2],
                                  range=hist_range, weights=counts[COLUMN_COUNT],
                                  bins=(n_bins[idx1], n_bins[idx2]))[0]

            if var is not None:
                counts_var = df.with_columns(pl.col(var).mul(COLUMN_COUNT).alias("tmp")).select(
                    cols + ["tmp"]).group_by(cols, maintain_order=False).sum()
                hist_var = np.histogram2d(counts_var[tc_col1], counts_var[tc_col2],
                                          range=hist_range, weights=counts_var["tmp"],
                                          bins=(n_bins[idx1], n_bins[idx2]))[0]
                hist = hist_var / np.where(np.isclose(hist, 0), 1, hist)
            hist = hist[::-1]

            range1 = (n_bins_cumsum[idx1], n_bins_cumsum[idx1 + 1])
            range2 = (n_bins_cumsum[idx2], n_bins_cumsum[idx2 + 1])

            hists_mtx[range1[0]: range1[1], range2[0]: range2[1]] = hist
            if idx1 == idx2:
                continue
            hists_mtx[range2[0]: range2[1], range1[0]: range1[1]] = np.flipud(np.transpose(hist))[:, ::-1]

    return hists_mtx


def _set_histograms2d_label(fig: plt.Figure, ax: plt.Axes, columns: List[str], is_y=False):
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    length_px = bbox.height if is_y else bbox.width

    char_width_per_pt = 0.6
    max_title_width_pt = length_px / char_width_per_pt

    k = 3
    approx_title_len = k * len("".join(columns))

    font_size = max_title_width_pt / approx_title_len

    n_lens = np.ones(len(columns)) * approx_title_len / len(columns)
    n_lens = np.round(n_lens).astype(int)

    label = ""
    for i, column in enumerate(columns):
        n_spaces = n_lens[i] - (k - 1) * len(column) // k
        label += " " * (n_spaces // 2) + column + " " * (n_spaces - n_spaces // 2)

    if is_y:
        ax.set_ylabel(label, fontsize=font_size)
    else:
        ax.set_xlabel(label, fontsize=font_size)


def main():
    parser = configargparse.ArgumentParser(config_arg_is_required=True, args_for_setting_config_path=["--config"],
                                           description="Analyse quantized PMW features")
    parser.add_argument("--analysis", default=ArgEDA.ACCUM, type=ArgEDA, action=EnumAction, required=True,
                        help="Analysis to perform")
    parser.add_argument("--transform", default=ArgTransform.DEFAULT, type=ArgTransform, action=EnumAction,
                        help="Type of transformation performed on data")
    parser.add_argument("--path", help="Transformed data path if it is different from the default one")
    parser.add_argument("--var", help="Variable to use in analysis")
    parser.add_argument("--k", type=int, help="Number of newest signatures to plot in red")

    args = parser.parse_args()

    if args.path is not None:
        path = pathlib.Path(args.path)
    else:
        path = pathlib.Path(DIR_PMW_ANALYSIS) / args.transform.value / "final.parquet"
    transform = get_transformation_function(args.transform)

    match args.analysis:
        case ArgEDA.ACCUM:
            plot_point_accumulation(path, args.var)
        case ArgEDA.PAIRPLOT:
            analyze(path, args.var, transform, args.k)
        case _:
            raise ValueError(f"{args.analysis.value} is not supported.")


if __name__ == '__main__':
    main()
