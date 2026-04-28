"""
Script for performing quantization on data from bucket.
"""
import datetime
import logging
import multiprocessing
import pathlib
import pickle
from collections import defaultdict
from typing import Dict, Callable, List, Sequence, Tuple

import configargparse
import gpm
import numpy as np
import polars as pl
from gpm.bucket import LonLatPartitioning
from gpm.bucket.io import get_bucket_spatial_partitioning
from quantization.aggregation import AggregationPlan, AggCircularMean, AggMax, AggMean, AggMin, AggOff, AggValueCounts
from quantization.api import quantize as quantization_quantize, merge_quantized as quantization_merge_quantized
from quantization.config import QuantizationConfig
from quantization.constants import COLUMN_COUNT as COLUMN_COUNT_QUANTIZATION, COLUMN_SUFFIX_QUANT as COLUMN_SUFFIX_QUANTIZATION
from quantization.quant_columns import FixedStepQuantColumnConfig
from tqdm import tqdm

from pmw_analysis.constants import DIR_BUCKET, DIR_PMW_ANALYSIS, COLUMN_LON, COLUMN_LAT, TC_COLUMNS, COLUMN_COUNT, \
    COLUMN_TIME, FLAG_DEBUG, COLUMN_GPM_ID, COLUMN_GPM_CROSS_TRACK_ID, COLUMN_LON_BIN, COLUMN_LAT_BIN, \
    FILE_DF_FINAL, FILE_DF_FINAL_NEWEST, \
    ArgQuantizationStep, ArgTransform, ArgQuantizationL2L3Columns, VARIABLE_SURFACE_TYPE_INDEX, COLUMN_L1C_QUALITY_FLAG, \
    DIR_NO_SUN_GLINT, ArgSurfaceType, COLUMN_BREAKPOINT, COLUMN_CATEGORY, \
    FILE_DF_FINAL_WITHOUT_NEWEST, COLUMN_SUN_GLINT_ANGLE_HF, COLUMN_SUN_GLINT_ANGLE_LF, SUN_GLINT_PRESENCE_RANGE, \
    QUALITY_FLAG_NON_NORMAL_STATUS_MODE, COLUMN_TEMP_2M_INDEX, Stats, FILE_DF_FINAL_WITHOUT_OLDEST, \
    FILE_DF_FINAL_OLDEST, FILE_DF_FINAL_RAREST, FILE_DF_FINAL_WITHOUT_RAREST
from pmw_analysis.copypaste.utils.cli import EnumAction
from pmw_analysis.processing.filter import filter_by_flag_values, filter_by_value_range
from pmw_analysis.quantization.dataframe_polars import get_uncertainties_dict
from pmw_analysis.quantization.transforms import get_pd_col, get_ratio_col, get_diff_col, _get_pd_expr, \
    _get_ratio_expr, _get_diff_expr, _add_pd_unc, _add_ratio_unc, _add_diff_unc, get_transformation_function
from pmw_analysis.retrievals.retrieval_1b_c_pmw import retrieve_possible_sun_glint
from pmw_analysis.utils.io import rmtree
from pmw_analysis.utils.logging import disable_logging, timing, get_memory_usage
from pmw_analysis.utils.polars import take_k_sorted, weighted_quantiles

MERGE_MEMORY_USAGE_FACTOR = 50
MEMORY_USAGE_LIMIT = 800

UNCERTAINTY_FACTOR_MAX = 20

FLAG_TEST = False
N_DFS_TEST = 3

X_STEP = 10
Y_STEP = 4
X_STEP_TEST = 1
Y_STEP_TEST = 1

_PERIODIC_COLUMN_TO_RANGE = {
    "sunLocalTime": (0.0, 24.0),
    COLUMN_LON: (-180.0, 180.0),
    "day": (1.0, 366.0),
}

_FLAG_COLUMN_TO_VALUES = {
    "Quality_LF": [-99, 0, 1, 2, 3, None],
    "SCorientation": [-9999, -8000, 0, 180, None],
    "Quality_HF": [-99, 0, 1, 2, 3, None],
    COLUMN_L1C_QUALITY_FLAG: [-10, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, None],
    "airmassLiftIndex": [0, 1, 2, 3, None],
    "pixelStatus": [-99, 0, 1, 2, 3, 4, 5, None],
    "qualityFlag": [-99, 0, 1, 2, 3, None],
    VARIABLE_SURFACE_TYPE_INDEX: [-9999] + list(range(1, 19)) + [None],
}




def _get_quantized_columns(quant_columns: Sequence[str]) -> list[str]:
    return [f"{col}{COLUMN_SUFFIX_QUANTIZATION}" for col in quant_columns]


def _get_time_min_column() -> str:
    return f"{COLUMN_TIME}__{Stats.MIN.value}"


def _get_time_max_column() -> str:
    return f"{COLUMN_TIME}__{Stats.MAX.value}"

def _calculate_bounds(x_step: int = X_STEP, y_step: int = Y_STEP) -> Tuple[np.ndarray, np.ndarray]:
    if FLAG_TEST:
        x_step = X_STEP_TEST
        y_step = Y_STEP_TEST

    p: LonLatPartitioning = get_bucket_spatial_partitioning(DIR_BUCKET)
    x_bounds = p.x_bounds.tolist()
    y_bounds = list(filter(lambda b: abs(b) <= 70, p.y_bounds))

    x_include_final = (len(x_bounds) - 1) % x_step != 0
    y_include_final = (len(y_bounds) - 1) % y_step != 0

    x_bounds = x_bounds[::x_step] + ([x_bounds[-1]] if x_include_final else [])
    y_bounds = y_bounds[::y_step] + ([y_bounds[-1]] if y_include_final else [])

    return np.array(x_bounds), np.array(y_bounds)
def get_transients(path: pathlib.Path, transform: Callable, k: int | None, timedelta: datetime.timedelta | None):
    df_id = pl.read_parquet(path / FILE_DF_FINAL)
    column_time_first = _get_time_min_column()
    column_time_last = _get_time_max_column()

    # 1. Get observations before quantization
    if timedelta is not None:
        start_time = df_id.select(pl.col(column_time_first).min()).item()
        end_time = df_id.select(pl.col(column_time_last).max()).item()

        start_time_newest = end_time - timedelta
        end_time_oldest = start_time + timedelta

        df_id_newest = df_id.filter(pl.col(column_time_first) >= start_time_newest)
        df_id_oldest = df_id.filter(pl.col(column_time_last) < end_time_oldest)
    else:
        df_id_newest = take_k_sorted(df_id, column_time_first, k, COLUMN_COUNT_QUANTIZATION, descending=True)
        df_id_oldest = take_k_sorted(df_id, column_time_last, k, COLUMN_COUNT_QUANTIZATION, descending=False)

    df_id_without_newest = df_id.join(df_id_newest, on=column_time_first, how="anti")
    df_id_without_newest.write_parquet(path / FILE_DF_FINAL_WITHOUT_NEWEST)

    df_id_without_oldest = df_id.join(df_id_oldest, on=column_time_last, how="anti")
    df_id_without_oldest.write_parquet(path / FILE_DF_FINAL_WITHOUT_OLDEST)

    dir_no_sun_glint = path / DIR_NO_SUN_GLINT
    dir_no_sun_glint.mkdir(parents=True, exist_ok=True)

    df_list = _get_bucket_data_for_ids([df_id_newest, df_id_oldest], transform)
    file_name_list = [FILE_DF_FINAL_NEWEST, FILE_DF_FINAL_OLDEST]

    for i, (df, file_name) in enumerate(zip(df_list, file_name_list)):
        # 2. Add a column to mark sun glint presence
        df, sun_glint_column = retrieve_possible_sun_glint(df)
        df.write_parquet(path / file_name)

        # 3. Store observations excluding the ones affected by sun glint
        df_no_sun_glint = df.filter(~pl.col(sun_glint_column)).drop(sun_glint_column)
        df_no_sun_glint.write_parquet(dir_no_sun_glint / file_name)


def get_rarest(path: pathlib.Path, transform: Callable, max_count: int):
    quant_columns = transform(TC_COLUMNS)
    quantized_columns = _get_quantized_columns(quant_columns)

    df_id = pl.read_parquet(path / FILE_DF_FINAL)
    df_id_rarest = df_id.filter(pl.col(COLUMN_COUNT_QUANTIZATION) <= max_count)

    df_id_without_rarest = df_id.join(df_id_rarest, on=quantized_columns, how="anti")
    df_id_without_rarest.write_parquet(path / FILE_DF_FINAL_WITHOUT_RAREST)

    dir_no_sun_glint = path / DIR_NO_SUN_GLINT
    dir_no_sun_glint.mkdir(parents=True, exist_ok=True)

    df = _get_bucket_data_for_ids([df_id_rarest], transform)[0]

    file_name = FILE_DF_FINAL_RAREST

    # TODO: fix copy-pasting from `get_transients`
    df, sun_glint_column = retrieve_possible_sun_glint(df)
    df.write_parquet(path / file_name)

    df_no_sun_glint = df.filter(~pl.col(sun_glint_column)).drop(sun_glint_column)
    df_no_sun_glint.write_parquet(dir_no_sun_glint / file_name)


def _get_bucket_data_for_ids(df_id_list: List[pl.DataFrame], transform: Callable) -> List[pl.DataFrame]:
    id_columns = [COLUMN_GPM_ID, COLUMN_GPM_CROSS_TRACK_ID]
    quant_columns = transform(TC_COLUMNS)
    quantized_columns = _get_quantized_columns(quant_columns)

    p: LonLatPartitioning = get_bucket_spatial_partitioning(DIR_BUCKET)
    x_bounds, y_bounds = _calculate_bounds(x_step=1, y_step=1)
    x_centroids = (x_bounds[:-1] + x_bounds[1:]) / 2
    y_centroids = (y_bounds[:-1] + y_bounds[1:]) / 2

    df_id_agg_list = []
    for df_id in df_id_list:
        df_id = df_id.select([COLUMN_LON, COLUMN_LAT] + id_columns + quantized_columns)
        df_id = df_id.explode([COLUMN_LON, COLUMN_LAT] + id_columns)

        df_id = p.add_labels(df_id, x=COLUMN_LON, y=COLUMN_LAT)
        df_id = p.add_centroids(df_id, x=COLUMN_LON, y=COLUMN_LAT,
                                x_coord=COLUMN_LON_BIN, y_coord=COLUMN_LAT_BIN)

        df_id_grouped = df_id.group_by(p.levels)
        df_id_agg = df_id_grouped.agg([col for col in df_id.columns if col not in p.levels]).sort(p.levels)
        df_id_agg_list.append(df_id_agg)

    dfs_bin_list = [[] for _ in range(len(df_id_agg_list))]

    progress_bar = tqdm(total=(len(x_bounds) - 1) * (len(y_bounds) - 1))
    for x_min, x_max, x_c in zip(p.x_bounds[:-1], x_bounds[1:], x_centroids):
        for y_min, y_max, y_c in zip(y_bounds[:-1], y_bounds[1:], y_centroids):

            if FLAG_TEST and len(dfs_bin_list[0]) >= N_DFS_TEST:
                progress_bar.update(1)
                break

            # Read observations from bucket
            extent = [x_min, x_max, y_min, y_max]
            df_bin = gpm.bucket.read(bucket_dir=DIR_BUCKET,
                                     extent=extent,
                                     backend="polars")

            for i, df_id_agg in enumerate(df_id_agg_list):
                df_id_bin = df_id_agg.filter(pl.col(COLUMN_LON_BIN) == x_c, pl.col(COLUMN_LAT_BIN) == y_c)
                df_id_bin = df_id_bin.drop(p.levels)
                df_id_bin = df_id_bin.explode(df_id_bin.columns)

                if df_id_bin.is_empty():
                    continue

                df_id_bin = df_id_bin.join(df_bin, on=id_columns, how="inner")
                dfs_bin_list[i].append(df_id_bin)
            progress_bar.update(1)

    df_list = []
    for i, dfs_bin in enumerate(dfs_bin_list):
        if len(dfs_bin) == 0:
            schema = (
                    {col: pl.Float32 for col in quantized_columns} |
                    {COLUMN_LON: pl.Float32, COLUMN_LAT: pl.Float32, COLUMN_TIME: pl.Datetime} |
                    {VARIABLE_SURFACE_TYPE_INDEX: pl.UInt32, COLUMN_L1C_QUALITY_FLAG: pl.Int32}
            )
            df = pl.DataFrame(schema=schema)
        else:
            df = pl.concat(dfs_bin)

        df = transform(df, drop=False)
        df_list.append(df)

    return df_list


def main():
    logging.basicConfig(level=logging.INFO)

    parser = configargparse.ArgumentParser(config_arg_is_required=True, args_for_setting_config_path=["--config"],
                                           description="Run rarest/transients extraction")

    parser.add_argument("--step", type=ArgQuantizationStep, action=EnumAction,
                        help="Quantization pipeline's step to perform")
    parser.add_argument("--transform", default=ArgTransform.DEFAULT, type=ArgTransform, action=EnumAction,
                        help="Type of transformation to perform on data")
    parser.add_argument("--dir", default=DIR_PMW_ANALYSIS,
                        help="Path to the directory where quantized data is stored")
    parser.add_argument("--surface-type", default=ArgSurfaceType.ALL,
                        type=ArgSurfaceType, action=EnumAction,
                        help="Surface type that was used during quantization")
    parser.add_argument("--month", type=int, help="Month subdirectory of quantized data")
    parser.add_argument("--year", type=int, help="Year subdirectory of quantized data")
    parser.add_argument("--transients-k", type=int,
                        help="The number of transient signatures to use when acquiring observations")
    parser.add_argument("--transients-timedelta-days", type=int,
                        help="The time period as a timedelta for acquiring observations based on transient signatures")
    parser.add_argument("--rarest-max-count", type=int,
                        help="The maximum allowed count for signature to be considered one of the rarest")
    args = parser.parse_args()

    if args.step not in (ArgQuantizationStep.TRANSIENTS, ArgQuantizationStep.RAREST):
        raise ValueError("Only --step transients and --step rarest are supported by this script.")

    assert not (args.year is None and args.month is not None)

    path = pathlib.Path(args.dir) / args.transform.value / args.surface_type.value
    if args.year is not None:
        path = path / str(args.year)
    if args.month is not None:
        path = path / str(args.month)

    transform = get_transformation_function(args.transform)

    match args.step:
        case ArgQuantizationStep.TRANSIENTS:
            if args.transients_k is None and args.transients_timedelta_days is None:
                raise ValueError("Either --transients-k or --transients-timedelta-days must be specified.")
            if args.transients_k is not None and args.transients_timedelta_days is not None:
                raise ValueError("Only one of --transients-k or --transients-timedelta-days can be specified.")

            transients_timedelta = None
            if args.transients_timedelta_days is not None:
                transients_timedelta = datetime.timedelta(days=args.transients_timedelta_days)

            get_transients(path, transform, args.transients_k, transients_timedelta)
        case ArgQuantizationStep.RAREST:
            if args.rarest_max_count is None:
                raise ValueError("--rarest-max-count must be specified for --step rarest.")
            get_rarest(path, transform, args.rarest_max_count)
        case _:
            raise ValueError(f"{args.step.value} is not supported.")


if __name__ == '__main__':
    main()
