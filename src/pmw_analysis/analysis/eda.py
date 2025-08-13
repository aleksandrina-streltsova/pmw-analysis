"""
This module contains utilities for analyzing quantized transformed data.
"""
import dataclasses
import logging
import pathlib
import pickle
from typing import List, Callable, Tuple, Any

import configargparse
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
from matplotlib.colors import LogNorm, Normalize
from tqdm import tqdm

from pmw_analysis.constants import (
    DIR_PMW_ANALYSIS,
    COLUMN_COUNT, COLUMN_ACCUM_UNIQUE, COLUMN_ACCUM_ALL,
    ST_GROUP_SNOW, ST_GROUP_OCEAN, ST_GROUP_VEGETATION, ST_GROUP_EDGES, ST_GROUP_MISC,
    VARIABLE_SURFACE_TYPE_INDEX,
    TC_COLUMNS, ST_COLUMNS, ArgEDA, ArgTransform, DIR_IMAGES, DIR_HISTS, Stats, COLUMN_OCCURRENCE, COLUMN_TIME,
)
from pmw_analysis.copypaste.utils.cli import EnumAction
from pmw_analysis.processing.filter import filter_by_flag_values
from pmw_analysis.quantization.dataframe_polars import expand_occurrence_column, get_agg_column
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.io import combine_paths, file_to_dir
from pmw_analysis.utils.polars import get_column_ranges, take_k_sorted
from pmw_analysis.utils.pyplot import finalize_axis


def plot_point_accumulation(path: pathlib.Path, occurrence_stat: Stats, var: str | None):
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
    column_occurrence_time = f"{get_agg_column(COLUMN_OCCURRENCE, occurrence_stat)}_{COLUMN_TIME}"
    columns = [column_occurrence_time, COLUMN_COUNT] + ([] if var is None else [var, f"{var}_count"])
    df = expand_occurrence_column(df).select(columns)
    df = df.with_columns(pl.col(column_occurrence_time).dt.round("1d"))
    df = df.with_columns(pl.col(COLUMN_COUNT).cast(pl.UInt64))
    df_count = _cum_sum_over_time(df, column_occurrence_time)

    fig, axes = _plot_count_over_time(df_count, column_occurrence_time)
    fig_var = _plot_var_over_time(df, column_occurrence_time, var, axes)

    images_dir = combine_paths(path_base=DIR_IMAGES, path_rel=file_to_dir(path),
                               path_rel_base=DIR_PMW_ANALYSIS) / "over_time"
    images_dir.mkdir(parents=True, exist_ok=True)

    for fig, file_name in [(fig, f"count_over_time{"" if var is None else f"_{var}"}.png"),
                           (fig_var, f"var_over_time{"" if var is None else f"_{var}"}.png")]:
        fig.tight_layout()
        fig.savefig(images_dir / file_name)


def _plot_count_over_time(df_count, column_time) -> Tuple[plt.Figure, np.ndarray[plt.Axes]]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, col_accum, title_prefix in zip(axes, [COLUMN_ACCUM_ALL, COLUMN_ACCUM_UNIQUE],
                                           ["Signatures first", "Unique signatures"]):
        ax.plot(df_count[column_time], df_count[col_accum], color="b", label="All")
        ax.set_yscale("log")
        finalize_axis(ax, title=f"{title_prefix} seen before this time", x_label="Time", y_label="Cumulative count")

    return fig, axes


def _plot_var_over_time(df: pl.DataFrame, column_time: str, var: str, axes: np.ndarray[plt.Axes]) -> plt.Figure:
    df_var_count_desc, df_var_mean_desc, df_var_not_null = _calculate_reverse_cumsum_for_variable(df, column_time, var)
    count_total_all = df_var_not_null[COLUMN_COUNT].sum()
    count_total_unique = len(df_var_not_null)

    for i, (col_accum, count_total) in enumerate(zip([COLUMN_ACCUM_ALL, COLUMN_ACCUM_UNIQUE],
                                                     [count_total_all, count_total_unique])):
        axes[i].plot(df_var_count_desc[column_time], count_total - df_var_count_desc[col_accum],
                     color="g", label=f"Not-null '{var}'")

    fig_var, (axes_var, axes_count) = plt.subplots(2, 2, figsize=(15, 10))
    for i, col_accum in enumerate([COLUMN_ACCUM_ALL, COLUMN_ACCUM_UNIQUE]):
        axes_var[i].plot(df_var_mean_desc[column_time], df_var_mean_desc[col_accum], color="r", label=var)
        finalize_axis(axes_var[i], title=f"Mean of '{var}' (from this time to end)", x_label="Time", y_label=var)

        axes_count[i].plot(df_var_mean_desc[column_time], df_var_count_desc[col_accum],
                           color="g", label=f"Not-null '{var}'")
        finalize_axis(axes_count[i], title="Cumulative count (from this time to end)", x_label="Time", y_label="Count")
        axes_count[i].set_yscale("log")

    return fig_var


@dataclasses.dataclass
class HistogramsMatrixParameters:
    n_bins: np.ndarray
    feature_ranges: np.ndarray
    var_norm: Any | None = None


def analyze(path: pathlib.Path, occurrence_stat: Stats, var: str | None, transform: Callable, k: int | None,
            ref_params_dir_path: pathlib.Path | None):
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

    column_occurrence_time = f"{get_agg_column(COLUMN_OCCURRENCE, occurrence_stat)}_{COLUMN_TIME}"
    columns = (
            feature_columns +
            [VARIABLE_SURFACE_TYPE_INDEX, COLUMN_COUNT, column_occurrence_time] +
            ([] if var is None else [var])
    )
    df = df_merged[[col for col in columns if col in df_merged.columns]]
    if k is not None:
        df_transients = take_k_sorted(df, column_occurrence_time, k, COLUMN_COUNT, descending=True)
    else:
        df_transients = None
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
                     ("Edges (Group)", ST_GROUP_EDGES, None),
                     ("Misc (Group)", ST_GROUP_MISC, None),
                 ] + [(st.replace("/", "_"), st, None) for st in ST_COLUMNS]
    else:
        groups = [(None, None, None)]

    for group in groups:
        _analyze_surface_type_group(df, df_transients, feature_columns, group, var, feature_ranges, n_bins,
                                    path, ref_params_dir_path)


def _sorted_not_null_unique_values(df: pl.DataFrame, col: str) -> pl.Series:
    return df[col].unique().drop_nulls().sort()


def _cum_sum_over_time(df: pl.DataFrame, column_time: str, var: str | None = None,
                       descending: bool = False) -> pl.DataFrame:
    if var is None:
        df = df.group_by(column_time).agg(
            pl.len().alias(COLUMN_ACCUM_UNIQUE),
            pl.col(COLUMN_COUNT).cast(pl.UInt64).sum().alias(COLUMN_ACCUM_ALL)
        )
    else:
        df = df.group_by(column_time).agg(
            pl.col(var).sum().alias(COLUMN_ACCUM_UNIQUE),
            pl.col(var).mul(pl.col(COLUMN_COUNT)).sum().alias(COLUMN_ACCUM_ALL)
        )

    df = df.sort(column_time, descending=descending)
    df = df.with_columns(
        pl.col(COLUMN_ACCUM_UNIQUE).cum_sum(),
        pl.col(COLUMN_ACCUM_ALL).cum_sum(),
    )
    if descending:
        df = df.reverse()
    return df


def _calculate_reverse_cumsum_for_variable(df, column_time, var):
    var_count = f"{var}_{COLUMN_COUNT}"

    df_var_not_null = df.filter(pl.col(var_count).ne(0).and_(pl.col(var).is_not_nan()))  # TODO: fix
    df_var_not_null = df_var_not_null.drop(COLUMN_COUNT).rename({var_count: COLUMN_COUNT})
    df_var_not_null = df_var_not_null.with_columns(pl.col(COLUMN_COUNT).cast(pl.UInt64))

    df_var_sum_desc = _cum_sum_over_time(df_var_not_null, column_time, var=var, descending=True)
    df_var_count_desc = _cum_sum_over_time(df_var_not_null, column_time, descending=True)

    df_var_mean_desc = df_var_sum_desc.with_columns(
        df_var_sum_desc[COLUMN_ACCUM_ALL] / df_var_count_desc[COLUMN_ACCUM_ALL],
        df_var_sum_desc[COLUMN_ACCUM_UNIQUE] / df_var_count_desc[COLUMN_ACCUM_UNIQUE],
    )
    return df_var_count_desc, df_var_mean_desc, df_var_not_null


def _analyze_surface_type_group(df, df_transients, feature_columns, group, var, feature_ranges, n_bins,
                                path, ref_params_dir_path):
    group_name, surface_types, color = group

    if group_name is None:
        flag_values = None
    else:
        flag_values = [idx + 1 for idx, st in enumerate(ST_COLUMNS) if st in surface_types]

    if color is None:
        cmap = plt.get_cmap("rocket_r")
    else:
        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["lightyellow", color])

    df_to_use = df
    df_to_use_transients = df_transients

    if flag_values is not None:
        df_to_use = filter_by_flag_values(df_to_use, VARIABLE_SURFACE_TYPE_INDEX, flag_values)
        if df_to_use.is_empty():
            logging.info(f"No data for group: {group_name}. Continuing without plotting.")
            return

        if df_to_use_transients is not None:
            df_to_use_transients = filter_by_flag_values(df_to_use_transients, VARIABLE_SURFACE_TYPE_INDEX, flag_values)

    if var is not None:
        cmap = plt.get_cmap("YlGnBu")

    cmap.set_bad(alpha=0)

    if ref_params_dir_path is not None:
        ref_params_path = ref_params_dir_path / f"{group_name}_{var}.pkl"
        if not ref_params_path.exists():
            logging.info(f"Reference parameters file not found for {group_name}. Plotting without reference.")
            ref_params = None
        else:
            ref_params = pickle.load(open(ref_params_path, "rb"))
    else:
        ref_params = None

    params = ref_params if ref_params is not None else HistogramsMatrixParameters(n_bins, feature_ranges)
    hists_mtx, alpha_mtx = _calculate_pairplots_concat_matrix(df_to_use, feature_columns, params, var)
    if df_to_use_transients is not None:
        hists_mtx_transients, _ = _calculate_pairplots_concat_matrix(df_to_use_transients, feature_columns, params,
                                                                     var=var)
    else:
        hists_mtx_transients = None

    hists_mtx_min = np.nanquantile(hists_mtx, 0.01)
    hists_mtx_max = np.nanquantile(hists_mtx, 0.99)

    if var is None:
        norm = LogNorm(vmin=hists_mtx_min, vmax=hists_mtx_max)
    elif ref_params is None:
        norm = Normalize(vmin=hists_mtx_min, vmax=hists_mtx_max)
    else:
        norm = ref_params.var_norm

    if ref_params is None:
        params_used = HistogramsMatrixParameters(n_bins, feature_ranges, norm)
    else:
        params_used = ref_params

    images_dir = combine_paths(path_base=DIR_IMAGES, path_rel=file_to_dir(path), path_rel_base=DIR_PMW_ANALYSIS)
    images_dir.mkdir(parents=True, exist_ok=True)

    hists_dir = combine_paths(path_base=DIR_HISTS, path_rel=file_to_dir(path), path_rel_base=DIR_PMW_ANALYSIS)
    hists_dir.mkdir(parents=True, exist_ok=True)

    hist_file_name = f"{group_name}_{var}.npy"
    hists_matx_params_file_name = f"{group_name}_{var}.pkl"
    np.save(hists_dir / hist_file_name, hists_mtx)
    pickle.dump(params_used, open(hists_dir / hists_matx_params_file_name, "wb"))

    fig, ax = plt.subplots(1, 1, figsize=(4 * len(feature_columns) + 4, 4 * len(feature_columns)))
    _plot_heatmap_with_varying_cell_sizes(hists_mtx, hists_mtx_transients, alpha_mtx, params_used, cmap, norm, ax)

    _set_histograms2d_label(fig, ax, feature_columns)
    _set_histograms2d_label(fig, ax, feature_columns[::-1], is_y=True)

    title_suffix = "" if var is None else f" ({var})"
    fig_suffix = "" if df_transients is None else "_newest"
    fig_suffix += "" if var is None else f"_{var}"

    if group_name is not None:
        fig.suptitle(group_name + title_suffix, fontsize=30)
        fig.savefig(images_dir / f"count_{group_name.replace("/", "_")}_{len(feature_columns)}{fig_suffix}.png")
    else:
        fig.suptitle("All Surfaces" + title_suffix, fontsize=30)
        fig.savefig(images_dir / f"count_{group_name}_{len(feature_columns)}{fig_suffix}.png")


def _plot_heatmap_with_varying_cell_sizes(hists_mtx: np.ndarray, hists_mtx_transients: np.ndarray | None,
                                          alpha_mtx: np.ndarray,
                                          params: HistogramsMatrixParameters,
                                          cmap, norm, ax: plt.Axes):
    n_bins, feature_ranges = params.n_bins, params.feature_ranges
    xs = np.ones(n_bins.sum())
    n_bins_cumsum = np.cumsum(n_bins)
    n_bins_cumsum = np.insert(n_bins_cumsum, 0, 0)
    lines = n_bins.min() * np.arange(len(n_bins) + 1)
    for n, offset in zip(n_bins, n_bins_cumsum):
        xs[offset:offset + n] *= n_bins.min() / n
    xs = np.cumsum(xs)
    bounds = (xs[:-1] + xs[1:]) / 2
    bounds = np.concatenate([[2 * bounds[0] - bounds[1]], bounds, [2 * bounds[-1] - bounds[-2]]])

    c = ax.pcolormesh(bounds, (bounds[-1] - bounds)[::-1], hists_mtx[::-1], cmap=cmap, norm=norm, alpha=alpha_mtx[::-1])
    if hists_mtx_transients is not None:
        cmap_transients = "viridis"
        ax.pcolormesh(bounds, (bounds[-1] - bounds)[::-1], hists_mtx_transients[::-1], cmap=cmap_transients)

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
                                       params: HistogramsMatrixParameters, var: str | None,
                                       ) -> Tuple[np.ndarray, np.ndarray]:
    n_bins_cumsum = np.cumsum(params.n_bins)
    n_bins_cumsum = np.insert(n_bins_cumsum, 0, 0)

    hists_mtx = np.zeros((n_bins_cumsum[-1], n_bins_cumsum[-1]))
    alpha_mtx = np.zeros((n_bins_cumsum[-1], n_bins_cumsum[-1]))
    for idx1, tc_col1 in tqdm(enumerate(feature_columns)):
        for idx2, tc_col2 in enumerate(feature_columns[:idx1 + 1]):
            bins = (params.n_bins[idx1], params.n_bins[idx2])
            cols = [tc_col1] if idx1 == idx2 else [tc_col1, tc_col2]

            counts = df.select(cols + [COLUMN_COUNT]).group_by(cols, maintain_order=False).sum()
            hist_range = (params.feature_ranges[idx1], params.feature_ranges[idx2])
            hist = np.histogram2d(counts[tc_col1], counts[tc_col2],
                                  range=hist_range, weights=counts[COLUMN_COUNT], bins=bins)[0]
            hist[hist == 0] = np.nan

            alpha = np.power(np.log(hist), 0.5)
            alpha = alpha / np.nanmax(alpha)
            alpha[np.isnan(alpha)] = 0

            if var is not None:
                counts_var = df.with_columns(pl.col(var).mul(COLUMN_COUNT).alias("tmp")).select(
                    cols + ["tmp"]).group_by(cols, maintain_order=False).sum()
                hist_var = np.histogram2d(counts_var[tc_col1], counts_var[tc_col2],
                                          range=hist_range, weights=counts_var["tmp"],
                                          bins=bins)[0]
                hist = hist_var / hist
            hist = hist[::-1]
            alpha = alpha[::-1]

            range1 = (n_bins_cumsum[idx1], n_bins_cumsum[idx1 + 1])
            range2 = (n_bins_cumsum[idx2], n_bins_cumsum[idx2 + 1])

            hists_mtx[range1[0]: range1[1], range2[0]: range2[1]] = hist
            alpha_mtx[range1[0]: range1[1], range2[0]: range2[1]] = alpha
            if idx1 == idx2:
                continue
            hists_mtx[range2[0]: range2[1], range1[0]: range1[1]] = np.flipud(np.transpose(hist))[:, ::-1]
            alpha_mtx[range2[0]: range2[1], range1[0]: range1[1]] = np.flipud(np.transpose(alpha))[:, ::-1]

    return hists_mtx, alpha_mtx


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
    parser.add_argument("--path", type=pathlib.Path,
                        help="Transformed data path if it is different from the default one")
    parser.add_argument("--occurrence-stat", default=Stats.MIN, type=Stats, action=EnumAction,
                        help="Statistic used on occurrence column. "
                             "Used in accumulation analysis and in pairplot analysis if k is specified")
    parser.add_argument("--var", help="Variable to use in analysis")
    parser.add_argument("--k", type=int, help="Number of newest signatures to plot in red")
    # TODO: naming is inconsistent: `ref_params_path` vs `path_ref_params`
    parser.add_argument("--ref-params-path", type=pathlib.Path,
                        help="Path to reference parameters to use for creating pairplots")

    args = parser.parse_args()

    if args.path is not None:
        path = pathlib.Path(args.path)
    else:
        path = pathlib.Path(DIR_PMW_ANALYSIS) / args.transform.value / "final.parquet"
    transform = get_transformation_function(args.transform)

    match args.analysis:
        case ArgEDA.ACCUM:
            plot_point_accumulation(path, args.occurrence_stat, args.var)
        case ArgEDA.PAIRPLOT:
            analyze(path, args.occurrence_stat, args.var, transform, args.k, args.ref_params_path)
        case _:
            raise ValueError(f"{args.analysis.value} is not supported.")


if __name__ == '__main__':
    main()
