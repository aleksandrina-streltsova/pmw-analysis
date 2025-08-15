"""
Example of analyzing signatures that have appeared for the first time later than others.
"""
import datetime
import pathlib

import gpm.utils.geospatial
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks

from pmw_analysis.analysis.spatial_visualization import plot_variables_on_map
from pmw_analysis.constants import COLUMN_LON, COLUMN_LAT, FILE_DF_FINAL_NEWEST, \
    DIR_IMAGES, ArgSurfaceType, VARIABLE_SURFACE_TYPE_INDEX, FILE_DF_FINAL_OLDEST, COLUMN_TIME
from pmw_analysis.constants import DIR_PMW_ANALYSIS, TC_COLUMNS, ArgTransform
from pmw_analysis.processing.filter import filter_by_signature_occurrences_count
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.io import combine_paths, file_to_dir


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

    flag_peaks = False
    flag_with_greenland = True

    for df_path in [df_newest_path, df_oldest_path]:
        df = pl.read_parquet(df_path)

        if not flag_with_greenland:
            extent = gpm.utils.geospatial.get_country_extent("Greenland")
            filter_lon_expr = pl.col(COLUMN_LON).is_between(extent.xmin, extent.xmax)
            filter_lat_expr = pl.col(COLUMN_LAT).is_between(extent.ymin, extent.ymax)
            df = df.filter((filter_lon_expr.and_(filter_lat_expr)).not_())

        images_dir = combine_paths(path_base=DIR_IMAGES, path_rel=file_to_dir(df_path), path_rel_base=DIR_PMW_ANALYSIS)
        images_dir.mkdir(parents=True, exist_ok=True)

        m_occurrences = 4

        df_quant_m, quant_columns_with_suffix = filter_by_signature_occurrences_count(df, m_occurrences, quant_columns)
        df_m = df.join(df_quant_m, on=quant_columns_with_suffix, how="inner")

        if flag_peaks:
            df_m = _get_observations_around_peak_periods(df_m, images_dir)

        columns_to_plot = [COLUMN_LON, COLUMN_LAT, "L1CqualityFlag", "qualityFlag", VARIABLE_SURFACE_TYPE_INDEX]
        columns_to_plot += quant_columns
        if flag_peaks:
            columns_to_plot += ["peaks"]

        df_m = df_m[columns_to_plot]
        # df_m = df_m[[COLUMN_LON, COLUMN_LAT, "peaks"]]

        m_occurrences_text = "" if m_occurrences == 1 else f"; Signature occurred at least {m_occurrences} times."
        peaks_suffix = "" if not flag_peaks else "_peaks"
        greenland_suffix = "" if flag_with_greenland else "_no_greenland"
        plot_variables_on_map(df_m, arg_transform,
                              images_dir=images_dir,
                              title_text_suffix=m_occurrences_text,
                              file_name_suffix=f"_{m_occurrences}{peaks_suffix}{greenland_suffix}")


def _get_observations_around_peak_periods(df, images_dir):
    full_period_in_days = (df[COLUMN_TIME].max() - df[COLUMN_TIME].min()).days
    n_days_per_bin = 4
    n_bins = int(full_period_in_days // n_days_per_bin)

    hist, bin_edges = np.histogram(df.select(pl.col(COLUMN_TIME).dt.timestamp("ms")), bins=n_bins)

    peaks_distance = n_bins / 20
    peaks = find_peaks(hist, prominence=np.std(hist), distance=peaks_distance)[0]

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centers_peaks = bin_centers[peaks]
    datetimes_peaks = [datetime.datetime.fromtimestamp(ts / 1000) for ts in bin_centers_peaks]

    plt.hist(df[COLUMN_TIME], bins=n_bins)
    plt.vlines(x=datetimes_peaks, ymin=0, ymax=hist.max(), color="red")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(images_dir / "hist_peaks.png")
    plt.show()

    column_peaks = "peaks"
    timedelta = datetime.timedelta(days=n_days_per_bin)
    df = df.with_columns([pl.Series(column_peaks, [None] * df.height, dtype=datetime.datetime)])

    for i, peak_datetime in enumerate(datetimes_peaks):
        df = df.with_columns(
            pl
            .when((pl.col(COLUMN_TIME) - peak_datetime).abs() <= timedelta)
            .then(pl.lit(peak_datetime))
            .otherwise(pl.col(column_peaks))
            .alias(column_peaks)
        )
    df = df.filter(pl.col(column_peaks).is_not_null())

    return df


if __name__ == '__main__':
    main()
