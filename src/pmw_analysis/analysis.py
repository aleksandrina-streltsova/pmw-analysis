"""
This module contains utilities for analyzing quantized transformed data.
"""
import argparse
import pathlib
from typing import Optional, List

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from matplotlib.colors import LogNorm
from tqdm import tqdm

from pmw_analysis.constants import (
    PMW_ANALYSIS_DIR,
    COLUMN_COUNT, COLUMN_ACCUM_UNIQUE, COLUMN_ACCUM_ALL, COLUMN_OCCURRENCE_TIME,
    ST_GROUP_SNOW, ST_GROUP_OCEAN, ST_GROUP_VEGETATION,
    VARIABLE_SURFACE_TYPE_INDEX,
    TC_COLUMNS, ST_COLUMNS,
)
from pmw_analysis.quantization.dataframe_polars import filter_surface_type, expand_occurrence_column
from pmw_analysis.utils.pyplot import finalize_axis


def plot_point_accumulation(path: pathlib.Path, var: Optional[str]):
    """
    Plot the accumulation of unique points over time for a given dataset.

    Parameters
    ----------
        path : pathlib.Path
            Path to the input parquet file containing the dataset.
        var : Optional[str]
            Optional variable name to perform additional analysis.

    """
    df = pl.read_parquet(path)
    columns = [COLUMN_OCCURRENCE_TIME, COLUMN_COUNT] + ([] if var is None else [var, f"{var}_count"])
    df = expand_occurrence_column(df).select(columns)
    df = df.with_columns(pl.col(COLUMN_OCCURRENCE_TIME).dt.round("1d"))
    df = df.with_columns(pl.col(COLUMN_COUNT).cast(pl.UInt64))
    df_count = _cum_sum_over_time(df)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, col_accum, title_prefix in zip(axes, [COLUMN_ACCUM_ALL, COLUMN_ACCUM_UNIQUE],
                                           ["Signatures first", "Unique signatures"]):
        ax.plot(df_count[COLUMN_OCCURRENCE_TIME], df_count[col_accum], color="b", label="All")
        ax.set_yscale("log")
        finalize_axis(ax, title=f"{title_prefix} seen before this time", x_label="Time", y_label="Cumulative count")
    figs_with_filenames = [(fig, f"count_over_time{"" if var is None else f"_{var}"}.png")]

    if var is not None:
        df_var_count_desc, df_var_mean_desc, df_var_not_null = _calculate_reverse_cumsum_for_variable(df, var)
        count_total_all = df_var_not_null[COLUMN_COUNT].sum()
        count_total_unique = len(df_var_not_null)

        for i, (col_accum, count_total) in enumerate(zip([COLUMN_ACCUM_ALL, COLUMN_ACCUM_UNIQUE],
                                                         [count_total_all, count_total_unique])):
            axes[i].plot(df_var_count_desc[COLUMN_OCCURRENCE_TIME],
                         count_total - df_var_count_desc[col_accum],
                         color="g", label=f"Not-null '{var}'")

        fig_var, (axes_var, axes_count) = plt.subplots(2, 2, figsize=(15, 10))
        for i, col_accum in enumerate([COLUMN_ACCUM_ALL, COLUMN_ACCUM_UNIQUE]):
            axes_var[i].plot(df_var_mean_desc[COLUMN_OCCURRENCE_TIME], df_var_mean_desc[col_accum],
                             color="r", label=var)
            finalize_axis(axes_var[i], title=f"Mean of '{var}' (from this time to end)", x_label="Time", y_label=var)

            axes_count[i].plot(df_var_mean_desc[COLUMN_OCCURRENCE_TIME], df_var_count_desc[col_accum],
                               color="g", label=f"Not-null '{var}'")
            finalize_axis(axes_count[i],
                          title="Cumulative count (from this time to end)", x_label="Time", y_label="Count")
            axes_count[i].set_yscale("log")
        figs_with_filenames.append((fig_var, f"{var}_over_time.png"))

    images_path = pathlib.Path("images") / path.parent.name / "over_time"
    images_path.mkdir(parents=True, exist_ok=True)

    for fig, filename in figs_with_filenames:
        fig.tight_layout()
        fig.savefig(images_path / filename)


def analyze(path: pathlib.Path, var: Optional[str]):
    """
    Generate pairplots of features for a given dataset.

    Parameters
    ----------
        path : pathlib.Path
            Path to the input parquet file containing the dataset.
        var : Optional[str]
            Optional variable name to color pairplots by. If None, pairplots are colored by counts of signatures.

    """
    df_merged = pl.read_parquet(path)

    feature_columns = TC_COLUMNS

    df = df_merged[feature_columns + ["surfaceTypeIndex", "count"]]
    # df = df[:10000]  # for testing

    n_bins = []
    for tc_col in feature_columns:
        value_count = df[[tc_col, COLUMN_COUNT]].group_by(tc_col).sum()
        null_count = value_count.filter(pl.col(tc_col).is_null())[COLUMN_COUNT][0]
        unique_value_count = len(value_count.filter(pl.col(tc_col).is_not_null()))

        print(f"{tc_col} has {unique_value_count} unique non-null values.")
        print(f"{tc_col} has {null_count} missing values.")

        n_bins.append(unique_value_count)

    n_bins = np.array(n_bins)

    min_vals = np.array(df.select([pl.col(col).min() for col in feature_columns]).row(0))
    max_vals = np.array(df.select([pl.col(col).max() for col in feature_columns]).row(0))

    images_path = pathlib.Path("images") / path.parent.name
    images_path.mkdir(exist_ok=True)

    groups = [
        (None, None, None),
        ("Ocean (Group)", ST_GROUP_OCEAN, "navy"),
        ("Vegetation (Group)", ST_GROUP_VEGETATION, "darkgreen"),
        ("Snow (Group)", ST_GROUP_SNOW, "rebeccapurple"),
    ]
    for group in groups:
        _analyze_surface_type_group(df, feature_columns, group, var, min_vals, max_vals, n_bins, images_path)


def _sorted_not_null_unique_values(df: pl.DataFrame, col: str) -> pl.Series:
    return df[col].unique().drop_nulls().sort()


def _cum_sum_over_time(df: pl.DataFrame, var: Optional[str] = None, descending: bool = False) -> pl.DataFrame:
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


def _analyze_surface_type_group(df, feature_columns, group, var, min_vals, max_vals, n_bins, images_path):
    group_name, surface_types, color = group

    if group_name is None:
        flag_values = None
        cmap = "rocket_r"
    else:
        flag_values = [idx + 1 for idx, st in enumerate(ST_COLUMNS) if st in surface_types]
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", color])

    df_to_use = df

    if flag_values is not None:
        df_to_use = df_to_use[feature_columns + [VARIABLE_SURFACE_TYPE_INDEX]]
        df_to_use = filter_surface_type(df_to_use, flag_values)

    for tc_col in feature_columns:
        value_count = df_to_use[[tc_col, COLUMN_COUNT]].group_by(tc_col).sum()
        value_count = value_count.filter(pl.col(tc_col).is_not_null())
        _plot_hist(value_count.to_pandas(), tc_col, group_name, images_path)

    if var is not None:
        cmap = "viridis"

    hists_mtx = _calculate_pairplots_concat_matrix(df_to_use, min_vals, max_vals, n_bins, var)

    hists_path = pathlib.Path("hists") / images_path.name
    hists_path.mkdir(parents=True, exist_ok=True)

    hist_path = hists_path / f"{group_name}.npy"
    np.save(hist_path, hists_mtx)

    if var is None:
        vmin = max(hists_mtx.min(), 1)
        vmax = max(hists_mtx.max(), 1)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        vmin = None
        vmax = np.percentile(hists_mtx, 99)  # TODO
        norm = None

    fig, ax = plt.subplots(1, 1, figsize=(3 * len(feature_columns) + 3, 3 * len(feature_columns)))
    sns.heatmap(hists_mtx, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, annot=False, ax=ax)

    x_ticks, y_ticks, x_tick_labels, y_tick_labels = _get_ticks_for_pairplots_concat(n_bins, min_vals, max_vals)
    ax.set_xticks(x_ticks, x_tick_labels, fontsize=7, rotation=90)
    ax.set_yticks(y_ticks, y_tick_labels, fontsize=7, rotation=0)

    x_label, x_label_fontsize = _get_histograms2d_label(fig, ax, feature_columns, n_bins)
    y_label, y_label_fontsize = _get_histograms2d_label(fig, ax, feature_columns[::-1], n_bins, use_height=True)
    ax.set_xlabel(x_label, fontsize=x_label_fontsize)
    ax.set_ylabel(y_label, fontsize=y_label_fontsize)

    title_suffix = "" if var is None else f" ({var})"

    if group_name is not None:
        fig.suptitle(group_name + title_suffix, fontsize=30)
        fig.savefig(images_path / f"count_{group_name.replace("/", "_")}_{len(feature_columns)}.png")
    else:
        fig.suptitle("All Surfaces" + title_suffix, fontsize=30)
        fig.savefig(images_path / f"count_{group_name}_{len(feature_columns)}.png")


def _plot_hist(value_count_pd: pd.DataFrame, tc_col, group_name, images_path: pathlib.Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    # TODO: width
    ax.bar(value_count_pd[tc_col], value_count_pd[COLUMN_COUNT], width=0.05)
    ax.set_xlabel(tc_col)
    ax.set_ylabel(COLUMN_COUNT)
    ax.set_yscale("log")
    ax.set_title(f"{tc_col} {group_name}")
    fig.tight_layout()
    fig.savefig(images_path / f"{tc_col}_{group_name}.png")


def _calculate_pairplots_concat_matrix(df: pl.DataFrame, min_vals: np.ndarray, max_vals: np.ndarray, n_bins: np.ndarray,
                                       var: Optional[str]):
    n_bins_cumsum = np.cumsum(n_bins)
    n_bins_cumsum = np.insert(n_bins_cumsum, 0, 0)

    hists_mtx = np.zeros((n_bins_cumsum[-1], n_bins_cumsum[-1]))
    for idx1, tc_col1 in tqdm(enumerate(TC_COLUMNS)):
        for idx2, tc_col2 in enumerate(TC_COLUMNS[:idx1 + 1]):
            cols = [tc_col1] if idx1 == idx2 else [tc_col1, tc_col2]

            counts = df.select(cols + [COLUMN_COUNT]).group_by(cols, maintain_order=False).sum()
            hist_range = ((min_vals[idx1], max_vals[idx1]), (min_vals[idx2], max_vals[idx2]))
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
            hists_mtx[range2[0]: range2[1], range1[0]: range1[1]] = hist.T

    return hists_mtx


def _get_ticks_for_pairplots_concat(n_bins: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray, ):
    x_ticks = np.linspace(0, n_bins.sum() - 1, n_bins.sum())
    y_ticks = np.linspace(0, n_bins.sum() - 1, n_bins.sum())
    x_tick_labels = np.concatenate([np.round(np.linspace(min_val, max_val, n_bin), decimals=1)
                                    for min_val, max_val, n_bin
                                    in zip(min_vals, max_vals, n_bins)])
    y_tick_labels = np.concatenate([np.round(np.linspace(min_val, max_val, n_bin), decimals=1)[::-1]
                                    for min_val, max_val, n_bin
                                    in zip(min_vals, max_vals, n_bins)])

    return x_ticks, y_ticks, x_tick_labels, y_tick_labels


def _get_histograms2d_label(fig: plt.Figure, ax: plt.Axes, columns: List[str], n_bins: List[int], use_height=False):
    fig.canvas.draw()
    bbox = ax.get_window_extent()
    length_px = bbox.height if use_height else bbox.width

    char_width_per_pt = 0.6
    max_title_width_pt = length_px / char_width_per_pt

    approx_title_len = 2 * len("".join(columns))

    font_size = max_title_width_pt / approx_title_len
    n_bins = np.array(n_bins)

    n_lens = n_bins / n_bins.sum() * approx_title_len
    n_lens = np.round(n_lens).astype(int)

    label = ""
    for i, column in enumerate(columns):
        n_spaces = n_lens[i] - len(column)
        label += " " * (n_spaces // 2) + column + " " * (n_spaces - n_spaces // 2)

    return label, font_size


def main():
    parser = argparse.ArgumentParser(description="Analyse quantized PMW features")
    subparsers = parser.add_subparsers(dest="analysis", required=True, help="Analysis to perform")
    parser.add_argument("--transform", "-t", default="default", choices=["default", "pd", "ratio"],
                        help="Type of transformation performed on data")

    pairplot_parser = subparsers.add_parser("pairplot", help="")
    pairplot_parser.add_argument("--var", "-v", help="Variable to color by in pair plots")

    accum_parser = subparsers.add_parser("accum", help="")
    accum_parser.add_argument("--var", "-v", help="Variable to ")

    args = parser.parse_args()

    path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform / "final.parquet"

    if args.analysis == "accum":
        plot_point_accumulation(path, args.var)
    elif args.analysis == "pairplot":
        analyze(path, args.var)


if __name__ == '__main__':
    main()
