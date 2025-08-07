"""
Example of detecting and plotting seasonal cycles in time series data.
"""
import pathlib
from typing import Dict, List

import gpm.bucket
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

from pmw_analysis.constants import (
    COLUMN_TIME,
    COLUMN_TIME_FRACTION,
    COLUMN_OCCURRENCE_TIME,
    ATTR_NAME,
    FLAG_SAVEFIG,
    DIR_IMAGES,
    DIR_BUCKET, TC_COLUMNS, AGG_OFF_COLUMNS, COLUMN_LAT, COLUMN_LON, DIR_PMW_ANALYSIS,
)
from pmw_analysis.analysis.cycle_detection import detect_cycle, plot_cycle, plot_periodogram
from pmw_analysis.quantization.dataframe_pandas import read_time_series_and_drop_nan
from pmw_analysis.quantization.dataframe_polars import quantize_pmw_features, get_uncertainties_dict, \
    expand_occurrence_column, _round
from pmw_analysis.quantization.script import get_range_dict
from pmw_analysis.retrievals.retrieval_1b_c_pmw import retrieve_PD
from pmw_analysis.utils.pandas import timestamp_to_fraction
from pmw_analysis.utils.pyplot import scatter_na

DISTANCE = 20000
YEAR = 365.242


# TODO: use ids
def _is_in_quantized(ts: pd.DataFrame, df_quantized: pl.DataFrame,
                     quant_columns: List[str],
                     uncertainty_dict: Dict[str, float],
                     range_dict: Dict[str, float],
                     ) -> pd.Series:
    quant_steps = [uncertainty_dict[col] for col in quant_columns]
    quant_ranges = [(range_dict[f"{col}_min"], range_dict[f"{col}_max"]) for col in quant_columns]

    ts_pl = pl.from_pandas(ts[quant_columns])
    ts_pl_rounded = _round(ts_pl.select(quant_columns), quant_columns, quant_steps, quant_ranges)

    df_quantized_struct = pl.struct(df_quantized.select(quant_columns))

    is_in = ts_pl_rounded.select(pl.struct(quant_columns).is_in(df_quantized_struct.implode()).alias("is_in"))
    return is_in.to_pandas()["is_in"]


def main():
    plt.rcParams["image.cmap"] = "viridis"

    # point_hunga_tonga = (-175.38206882739425, -20.53562301896748)
    # point_ch = (8.30863808059546, 47.0545280638769)
    # name_hunga_tonga = "Hunga Tonga island"
    # name_ch = "Switzerland"

    point_greenland = (-47.125, 61.625)
    name_greenland = "Greenland"

    images_dir = pathlib.Path(DIR_IMAGES) / "time_series" / "greenland"
    images_dir.mkdir(parents=True, exist_ok=True)

    quant_cols = TC_COLUMNS
    unc_dict = {col: 10.0 * unc for col, unc in get_uncertainties_dict(TC_COLUMNS).items()}
    range_dict = get_range_dict()

    # 0. Check signatures that have appeared for the first time later than others
    for point, name in [(point_greenland, name_greenland)]:
        ts_pl = gpm.bucket.read(bucket_dir=DIR_BUCKET, point=point, distance=DISTANCE)

        ts_pl_quantized = quantize_pmw_features(ts_pl, quant_cols, unc_dict, range_dict, AGG_OFF_COLUMNS)
        ts_pl_quantized = expand_occurrence_column(ts_pl_quantized)

        ts_pl_exploded = ts_pl_quantized.sort(COLUMN_OCCURRENCE_TIME, descending=True).explode(AGG_OFF_COLUMNS)

        df_unique = pl.read_parquet(pathlib.Path(DIR_PMW_ANALYSIS) / "unique.parquet")
        df_unique = expand_occurrence_column(df_unique)

        df_inner = df_unique.join(ts_pl_exploded, on=quant_cols, how="inner", nulls_equal=True)
        assert df_inner.filter(pl.col(f"{COLUMN_OCCURRENCE_TIME}") >
                               pl.col(f"{COLUMN_OCCURRENCE_TIME}_right")).height == 0

    # 1. Read and process time series.
    ts_list = []
    ts_unique_k_list = []
    l2_cols = ["probabilityOfPrecip"]

    aggregate_flag = False
    agg_n_days = YEAR / 360

    feature_cols = TC_COLUMNS + ["temp2mIndex"]
    for point, name in [(point_greenland, name_greenland)]:
        ts = read_time_series_and_drop_nan(DIR_BUCKET, point, DISTANCE, name, feature_cols)
        ts = ts[np.unique([COLUMN_TIME, COLUMN_LON, COLUMN_LAT, *quant_cols, *feature_cols, *l2_cols])]
        ts[COLUMN_TIME_FRACTION] = ts["time"].apply(timestamp_to_fraction)

        if aggregate_flag:
            ts = ts.groupby(by=[ts[COLUMN_TIME_FRACTION] - ts[COLUMN_TIME_FRACTION] % agg_n_days],
                            as_index=False).agg("mean")
        else:
            # 1.1. Leave only the signatures that have appeared for the first time later than others
            df_unique_k = pl.read_parquet(pathlib.Path(DIR_PMW_ANALYSIS) / "unique_k.parquet").select(quant_cols)

            is_in = _is_in_quantized(ts, df_unique_k, quant_cols, unc_dict, range_dict)
            ts_unique_k = ts.reset_index(drop=True)[is_in]
            ts_unique_k_list.append(ts_unique_k)

        ts = retrieve_PD(ts)
        ts_list.append(ts)

    freqs_pd = ["10", "19", "37", "89", "165"]
    feature_cols += [f"PD_{freq}" for freq in freqs_pd]

    # 1.2. Get unique point time coordinates.
    # ts_unique_k[ts_unique_k["Tc_19H"] == ts_unique_k["Tc_19H"].min()][[COLUMN_LON, COLUMN_LAT, COLUMN_TIME]]
    # dt = datetime.datetime(year=2021, month=10, day=19, hour=4, minute=6, second=27)
    # ts.iloc[(ts["time"] - dt).abs().argmin()]
    # ts["time_dist"] = (ts[COLUMN_TIME] - dt).abs()
    # ts.sort_values(by="time_dist")[:40]

    # 2. Plot time series colorized by precipitation flags.
    for color_col in l2_cols:
        n = len(feature_cols)
        _, axes = plt.subplots(n, len(ts_list), figsize=(6 * len(ts_list), 2 * n), dpi=300)
        if len(ts_list) == 1:
            if n == 1:
                axes = np.array([axes])
            axes = np.expand_dims(axes, axis=1)

        for idx_col, ts in enumerate(ts_list):
            name = ts.attrs[ATTR_NAME]
            for idx_row, feature_col in enumerate(feature_cols):
                # TODO: think about removing this assertion
                assert ts[feature_col].isna().sum() == 0

                plt.sca(axes[idx_row][idx_col])
                scatter_na(x=ts[COLUMN_TIME], y=ts[feature_col], c=ts[color_col], color_label=color_col)

                ts_unique = ts_unique_k_list[idx_col]
                plt.scatter(ts_unique[COLUMN_TIME], ts_unique[feature_col], c="orange", s=1, zorder=1)

                plt.plot(ts[COLUMN_TIME], ts[feature_col], linestyle="--", linewidth=0.5, alpha=0.2)
                plt.ylabel("[K]")
                plt.title(f"{feature_col} ({name})")
        plt.tight_layout()
        if FLAG_SAVEFIG:
            if aggregate_flag:
                plt.savefig(images_dir / f"{color_col}_{int(agg_n_days)}.png")
            else:
                plt.savefig(images_dir / f"{color_col}.png")
        plt.show()

    # 3. Calculate correlation matrix.
    for ts in ts_list:
        name = ts.attrs[ATTR_NAME]
        corr_mtx = ts[feature_cols].corr().abs()
        sns.heatmap(corr_mtx, annot=True)
        if FLAG_SAVEFIG:
            plt.savefig(pathlib.Path(DIR_IMAGES) / f"{name}_corr.png")
        plt.show()

    # 4. Detect cycles.
    for idx, ts in enumerate(ts_list):
        ts_list[idx] = detect_cycle(ts, feature_cols)

    for freq in freqs_pd:
        cols = [col for col in feature_cols if freq in col]
        plot_cycle(ts_list, cols, suffix="all")
        plot_periodogram(ts_list, cols, suffix="all")

    # 5. Divide the time series into periods, keeping only the median value for each period, and then detect cycles.
    def _get_medians(ts_noisy: pd.DataFrame, column: str, time_period: float) -> pd.DataFrame:
        ts_noisy = ts_noisy.copy()
        ts_noisy["period"] = ts_noisy[COLUMN_TIME_FRACTION] // time_period

        # Find the row closest to the median value in each period.
        def median_row(sub_ts):
            median = sub_ts[column].median()
            return sub_ts.iloc[(sub_ts[column] - median).abs().argmin()]

        median_rows = ts_noisy.groupby("period").apply(median_row, include_groups=False).reset_index(drop=True)
        median_rows.attrs[ATTR_NAME] = ts_noisy.attrs[ATTR_NAME]
        print(f"{len(median_rows)}/{len(ts_noisy)} rows")
        return median_rows

    time_period = 20
    for freq in freqs_pd:
        cols = [col for col in feature_cols if freq in col]
        ts_medians_list = []
        for idx, ts in enumerate(ts_list):
            ts_medians = _get_medians(ts, f"Tc_{freq}H", time_period)
            ts_medians = detect_cycle(ts_medians, cols)
            ts_medians_list.append(ts_medians)

        plot_cycle(ts_medians_list, cols, suffix=f"median_{time_period}")
        plot_periodogram(ts_medians_list, cols, suffix=f"median_{time_period}")
