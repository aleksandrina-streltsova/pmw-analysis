"""
Example of analyzing signatures that have appeared for the first time later than others.
"""
import datetime
import logging
import math
import pathlib
from typing import Any

import gpm.utils.geospatial
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks
from skimage.feature import peak_local_max

from pmw_analysis.analysis.spatial_visualization import plot_variables_on_map
from pmw_analysis.constants import COLUMN_LON, COLUMN_LAT, FILE_DF_FINAL_NEWEST, \
    DIR_IMAGES, ArgSurfaceType, FILE_DF_FINAL_OLDEST, COLUMN_TIME, COLUMN_COUNT
from pmw_analysis.constants import DIR_PMW_ANALYSIS, TC_COLUMNS, ArgTransform
from pmw_analysis.processing.filter import filter_by_signature_occurrences_count, get_filter_expr_from_value_range
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.io import combine_paths, file_to_dir
from pmw_analysis.utils.logging import timing
from pmw_analysis.utils.pyplot import plot_histogram, plot_histogram2d


def main():
    arg_transform = ArgTransform.V4
    arg_surface_type = ArgSurfaceType.LAND
    df_dir_path = pathlib.Path(DIR_PMW_ANALYSIS) / arg_transform.value / arg_surface_type.value

    transform = get_transformation_function(arg_transform)
    quant_columns = transform(TC_COLUMNS)

    # df_all = pl.read_parquet(df_dir_path / FILE_DF_FINAL)
    # print(f"{df_all.select(pl.col(COLUMN_COUNT).cast(pl.UInt64).sum()).item()} -> {df_all.height}")

    df_newest_path = df_dir_path / FILE_DF_FINAL_NEWEST
    df_oldest_path = df_dir_path / FILE_DF_FINAL_OLDEST

    logging.basicConfig(level=logging.INFO)

    for i, df_path in enumerate([df_newest_path, df_oldest_path]):
        # Read data and create an images directory
        df = pl.read_parquet(df_path)
        images_dir = combine_paths(path_base=DIR_IMAGES, path_rel=file_to_dir(df_path), path_rel_base=DIR_PMW_ANALYSIS)
        images_dir.mkdir(parents=True, exist_ok=True)

        # Only consider signatures that appeared at least a specified number of times
        m_occurrences = 4
        df_m = filter_by_signature_occurrences_count(df, m_occurrences, quant_columns)

        bin_sizes_time = {COLUMN_TIME: datetime.timedelta(days=1)}
        bin_ranges_time = {}

        peaks_time = _detect_peaks(df_m[[COLUMN_TIME]], bin_sizes_time, bin_ranges_time, images_dir)

        bin_sizes_lon_lat = {COLUMN_LON: 1, COLUMN_LAT: 1}
        bin_ranges_lon_lat = {COLUMN_LON: (-180, 180), COLUMN_LAT: (-75, 75)}

        peaks_lon_lat = _detect_peaks(df_m[[COLUMN_LON, COLUMN_LAT]], bin_sizes_lon_lat, bin_ranges_lon_lat, images_dir)

        extent_greenland = gpm.utils.geospatial.get_country_extent("Greenland")
        filter_expr_greenland = get_filter_expr_from_value_range([COLUMN_LON, COLUMN_LAT],
                                                                 np.array(extent_greenland).reshape((2, 2)))
        zones = df_m.select(pl.when(filter_expr_greenland).then(pl.lit(1)).otherwise(pl.lit(0))).to_series()

        peaks_time_10 = take_k_peaks_by_zones(peaks_time, 10, zones)
        peaks_lon_lat_10 = take_k_peaks_by_zones(peaks_lon_lat, 10, zones)

        df_m = df_m.with_columns(peaks_time_10.to_series().alias("peaks_time"))
        df_m = df_m.with_columns(peaks_lon_lat_10.to_series().alias("peaks_lon_lat"))

        columns_to_plot = [COLUMN_LON, COLUMN_LAT, COLUMN_COUNT]
        columns_to_plot += ["peaks_time", "peaks_lon_lat"]

        with timing("Plotting on map"):
            m_occurrences_text = "" if m_occurrences == 1 else f"; Signature occurred at least {m_occurrences} times."
            # peaks_suffix = "" if not flag_peaks else "_peaks"
            # greenland_suffix = "" if flag_with_greenland else "_no_greenland"
            plot_variables_on_map(df_m[columns_to_plot], arg_transform,
                                  images_dir=images_dir,
                                  title_text_suffix=m_occurrences_text,
                                  file_name_suffix=f"_{m_occurrences}",
                                  # file_name_suffix=f"_{m_occurrences}{peaks_suffix}{greenland_suffix}_{value_range[0]}_{value_range[1]}",
                                  extent=None,
                                  threshold_cat=20)
        df_m.write_parquet(df_path.parent / f"{df_path.stem}_peaks.parquet")

    #
    #
    # column_occurrence_class = "occurrence_class"
    # df_with_peaks = expand_occurrence_column(df_all)
    # df_with_peaks = df_with_peaks.with_columns(
    #     pl.when(pl.col("peaks").is_not_null()).then(pl.col("peaks")).otherwise(pl.col("peaks1"))).drop(pl.col("peaks1"))
    # df_with_peaks = df_with_peaks.with_columns(
    #     pl.when(pl.col(f"{COLUMN_OCCURRENCE}_{Stats.MIN.value}_{COLUMN_TIME}").dt.year() >= 2022).then(
    #         pl.lit("newest")).otherwise(pl.lit("regular")).alias(column_occurrence_class)
    # )
    # df_with_peaks = df_with_peaks.with_columns(
    #     pl.when(pl.col(f"{COLUMN_OCCURRENCE}_{Stats.MAX.value}_{COLUMN_TIME}").dt.year() < 2020).then(
    #         pl.lit("oldest")).otherwise(pl.col(column_occurrence_class)).alias(column_occurrence_class)
    # )
    # df_with_peaks.write_parquet(df_dir_path / "final_extra.parquet")

    # unc_dict =  transform(get_uncertainties_dict(TC_COLUMNS))
    # for col in quant_columns:
    #     print(f"{col}:\n\t{2 * unc_dict[col]:.2f}")
    # print()
    # print()
    # print("land with no filtering:\n\t6.5 billions → 5.4")
    # print("land with filtering:\n\tmillions 5 billions → 5 millions")


def _detect_peaks(df: pl.DataFrame,
                  bin_sizes: dict[str, Any], bin_ranges: dict[str, Any],
                  images_dir: pathlib.Path) -> pl.Series:
    assert df.shape[1] <= 2, "Peak detection using more than two variables is not supported"

    if any(dtype == pl.Datetime for dtype in df.dtypes):
        df = df.clone()
        bin_sizes = bin_sizes.copy()

        for col in df.columns:
            if df[col].dtype == pl.Datetime:
                df = df.with_columns(pl.col(col).dt.timestamp("ms") // 1000)
                bin_sizes[col] = bin_sizes[col].total_seconds()

                if col in bin_ranges:
                    # TODO: fix
                    raise ValueError("Custom range is not supported for Datetime column yet")

    hist, bin_edges_list = _build_hist(df, bin_sizes, bin_ranges)

    peak_indices = _get_peak_indices_from_hist(df, hist, bin_edges_list, images_dir)

    peaks = _build_peaks_series(df, peak_indices, bin_edges_list)

    return peaks


def _build_hist(df: pl.DataFrame,
                bin_sizes: dict[str, Any], bin_ranges: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray]]:
    bin_edges_list = []

    for col in df.columns:
        if col in bin_ranges:
            col_min, col_max = bin_ranges[col]

            n_bins = int(math.ceil((col_max - col_min) / bin_sizes[col]))
            bin_edges = col_min + np.arange(n_bins) * bin_sizes[col]
            bin_edges = np.append(bin_edges, col_max)
        else:
            col_min = df[col].min()
            col_max = df[col].max()

            n_bins = int(math.ceil((col_max - col_min) / bin_sizes[col]))
            bin_edges = col_min + np.arange(n_bins + 1) * bin_sizes[col]

        bin_edges_list.append(bin_edges)

    if len(df.columns) == 1:
        hist, _ = np.histogram(df, bins=bin_edges_list[0])

    elif len(df.columns) == 2:
        hist, _, _ = np.histogram2d(df[df.columns[0]], df[df.columns[1]], bins=bin_edges_list)
    else:
        raise ValueError("Building histogram using more than two variables is not supported")

    return hist, bin_edges_list


def _get_peak_indices_from_hist(df: pl.DataFrame, hist: np.ndarray, bin_edges_list: list[np.ndarray],
                                images_dir: pathlib.Path) -> np.ndarray:
    if len(hist.shape) == 1:
        bin_edges = bin_edges_list[0]

        peak_indices, _ = find_peaks(hist)
        # Plot histogram and threshold
        plt.figure(figsize=(20, 5))
        # TODO: title, xlabel, ylabel, hlines label should be passed
        plot_histogram(hist, bin_edges, title="Transient signatures count", x_label="Datetime", y_label="Count")
        plt.scatter((bin_edges[peak_indices] + bin_edges[peak_indices + 1]) / 2, hist[peak_indices],
                    marker="x", color="red", label="Detected peaks")
        plt.legend()
        plt.tight_layout()
        plt.savefig(images_dir / f"hist_peaks_{'_'.join(df.columns)}.png")
        plt.show()

        return peak_indices

    if len(hist.shape) == 2:
        bin_edges_x = bin_edges_list[0]
        bin_edges_y = bin_edges_list[1]

        peak_indices = peak_local_max(hist, min_distance=1)
        # Plot 2D-histogram and detected local maxima
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
        # TODO: title, xlabel, ylabel, hlines label should be passed
        use_log_norm = True
        x_ticks = np.arange(len(bin_edges_x))[::len(bin_edges_x) // 40]
        y_ticks = np.arange(len(bin_edges_y))[::len(bin_edges_y) // 20]
        plot_histogram2d(ax, hist.T[::-1], use_log_norm,
                         x_ticks, bin_edges_x[x_ticks],
                         y_ticks, bin_edges_y[::-1][y_ticks],
                         title="Transient signatures count", x_label="Longitude", y_label="Latitude", cmap="viridis")
        plt.scatter(peak_indices[:, 0] + 1 / 2, len(bin_edges_y) - 1 - (peak_indices[:, 1] + 1 / 2),
                    marker=".", color="red", label="Detected peaks")
        plt.legend()
        plt.tight_layout()
        plt.savefig(images_dir / f"hist_peaks_{'_'.join(df.columns)}.png")
        plt.show()

        return peak_indices

    raise ValueError("Peak detection using more than two variables is not supported")


def _build_peaks_series(df: pl.DataFrame, peak_indices: np.ndarray, bin_edges_list: list[np.ndarray]) -> pl.Series:
    peaks = pl.DataFrame(pl.Series(values=np.full(df.shape[0], None), name="peaks", dtype=pl.Int64))

    if len(peak_indices.shape) == 1:
        col = df.columns[0]
        bin_edges = bin_edges_list[0]

        for i, peak_idx in enumerate(peak_indices):
            lower_bound = bin_edges[peak_idx]
            upper_bound = bin_edges[peak_idx + 1]
            peaks = peaks.with_columns(pl.when(df[col].is_between(lower_bound, upper_bound))
                                       .then(pl.lit(i))
                                       .otherwise(pl.col("peaks"))
                                       .alias("peaks"))
        return peaks

    if len(peak_indices.shape) == 2:
        col_x = df.columns[0]
        col_y = df.columns[1]
        bin_edges_x = bin_edges_list[0]
        bin_edges_y = bin_edges_list[1]

        for i, (peak_idx_x, peak_idx_y) in enumerate(peak_indices):
            predicate_x = df[col_x].is_between(bin_edges_x[peak_idx_x], bin_edges_x[peak_idx_x + 1])
            predicate_y = df[col_y].is_between(bin_edges_y[peak_idx_y], bin_edges_y[peak_idx_y + 1])
            peaks = peaks.with_columns(pl.when([predicate_x, predicate_y])
                                       .then(pl.lit(i))
                                       .otherwise(pl.col("peaks"))
                                       .alias("peaks"))

        return peaks

    raise ValueError("Peak detection using more than two variables is not supported")


def _take_k_peaks(peaks: pl.DataFrame, k: int) -> pl.DataFrame:
    assert len(peaks.columns) == 1
    col = peaks.columns[0]

    value_counts = peaks.to_series().value_counts(sort=True).filter(pl.col(col).is_not_null())

    peaks_k_labels = value_counts[col][:k]
    peaks_k = peaks.with_columns(pl.when(pl.col(col).is_in(peaks_k_labels))
                                 .then(pl.col("peaks"))
                                 .otherwise(pl.lit(None)))

    return peaks_k


def take_k_peaks_by_zones(peaks: pl.DataFrame, k: int, zones: pl.Series):
    peaks_k = peaks.clone()
    for zone, _ in zones.value_counts().iter_rows():
        peaks_zone = peaks.with_columns(pl.when(zones == zone).then(pl.col("peaks")).otherwise(None).alias("peaks"))
        peaks_zone_k = _take_k_peaks(peaks_zone, k)
        peaks_k = peaks_k.with_columns(pl.when(zones == zone)
                                       .then(peaks_zone_k["peaks"])
                                       .otherwise(peaks_k["peaks"])
                                       .alias("peaks"))

    return peaks_k


if __name__ == '__main__':
    main()
