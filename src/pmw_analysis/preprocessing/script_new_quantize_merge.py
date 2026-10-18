"""
Script for performing quantization on data from bucket.
"""
import logging
import pathlib
import pickle
from datetime import datetime
from typing import Dict, Callable, Iterable

import configargparse
import gpm
import gpm.bucket
import polars as pl
from pmw_analysis.constants import DIR_BUCKET, DIR_PMW_ANALYSIS, COLUMN_LON, COLUMN_LAT, TC_COLUMNS, \
    COLUMN_TIME, COLUMNS_TO_DROP, \
    ArgQuantizationStep, ArgTransform, VARIABLE_SURFACE_TYPE_INDEX, COLUMN_L1C_QUALITY_FLAG, \
    ArgSurfaceType, COLUMN_SUN_GLINT_ANGLE_HF, COLUMN_SUN_GLINT_ANGLE_LF, SUN_GLINT_PRESENCE_RANGE, \
    QUALITY_FLAG_NON_NORMAL_STATUS_MODE, COLUMN_TEMP_2M_INDEX
from pmw_analysis.copypaste.utils.cli import EnumAction
from pmw_analysis.preprocessing.aggregation_defaults import build_default_per_column_aggregations
from pmw_analysis.preprocessing.dataframe_polars import get_uncertainties_dict
from pmw_analysis.preprocessing.transforms import get_transformation_function
from pmw_analysis.processing.filter import filter_by_value_range
from pmw_analysis.utils.cli import parse_bool
from pmw_analysis.utils.logging import timing
from pmw_analysis.utils.polars import weighted_quantiles
from quantization.aggregation import AggregationPlan, AggMax, AggMean, AggMin
from quantization.api import collect_frame_statistics, quantize_streaming, merge_streaming, \
    finalize_quant_column_configs
from quantization.config import QuantizationConfig
from quantization.constants import COLUMN_SUFFIX_QUANT
from quantization.quant_columns import FixedStepQuantColumnConfig, UncertaintyQuantColumnConfig
from quantization.types import Frame, FrameStatistics

FULL_EXTENT = [-180, 180, -90, 90]
CHUNK_SIZE_QUANTIZE = 100_000_000
CHUNK_SIZE_MERGE = 10_000_000
MAX_N_FINAL = 10_000_000
FACTORS = [1.0, 2.0, 4.0, 8.0]
CLIP_QUANTILE = 1e-6
SUN_GLINT_ANGLE_BELOW_HORIZON = -88
FILE_FRAME_STATISTICS = "frame_statistics.pkl"
FILE_QUANTIZATION_CONFIG = "quantization_config.pkl"


def _prepare_for_quantization(frame: Frame) -> Frame:
    frame = frame.fill_nan(None)
    frame = _add_below_horizon_columns(frame)
    frame = _replace_special_missing_values_with_null(frame)
    frame = frame.with_columns(pl.col(COLUMN_TIME).dt.ordinal_day().alias("day"))
    frame = frame.drop(COLUMNS_TO_DROP, strict=False)

    return frame

def _add_below_horizon_columns(frame: Frame) -> Frame:
    # Sun below horizon (-88) becomes null in the angle; this flag keeps it, its mean is the fraction below horizon
    schema = frame.collect_schema()
    return frame.with_columns([
        pl.col(col).eq(SUN_GLINT_ANGLE_BELOW_HORIZON).alias(f"{col}_below_horizon")
        for col in [COLUMN_SUN_GLINT_ANGLE_LF, COLUMN_SUN_GLINT_ANGLE_HF]
        if col in schema
    ])


def _replace_special_missing_values_with_null(frame: Frame) -> Frame:
    schema = frame.collect_schema()  # {col: dtype}

    special_values = {
        COLUMN_SUN_GLINT_ANGLE_LF: [-99, -9999, SUN_GLINT_ANGLE_BELOW_HORIZON],
        COLUMN_SUN_GLINT_ANGLE_HF: [-99, -9999, SUN_GLINT_ANGLE_BELOW_HORIZON],
        COLUMN_LON: [-9999],  # lon ranges from -180 to 180 and can be equal to -99
    }

    exprs = [
        pl.when(pl.col(col).is_in(special_values.get(col, [-99, -9999])))
          .then(None)
          .otherwise(pl.col(col))
          .alias(col)
        for col, dtype in schema.items()
        if dtype.is_numeric()
    ]

    return frame.with_columns(exprs)


def _get_agg_plan(columns: Iterable[str]) -> AggregationPlan:
    columns_set = set(columns)

    per_column = build_default_per_column_aggregations()
    per_column = {key: value for key, value in per_column.items() if key in columns_set}
    if COLUMN_TIME in columns_set:
        per_column[COLUMN_TIME] = [AggMin(), AggMax()]

    return AggregationPlan(
        per_column=per_column,
        default_non_quant=AggMean(),
        default_quant=None,
    )


def _create_quantization_config(columns: Iterable[str], quant_columns: Iterable[str], output_dir: pathlib.Path,
                                frame: Frame,
                                frame_statistics: FrameStatistics,
                                uncertainty_dict: Dict[str, float],
                                max_n_final: int,
                                clip: bool = False) -> QuantizationConfig:
    unfinalized_quant_column_configs = [
        UncertaintyQuantColumnConfig(
            col,
            uncertainty_dict[col],
            clip=_get_clip_range_from_frame_statistics(frame_statistics, col) if clip else None,
        )
        for col in quant_columns
    ]

    quant_column_configs = finalize_quant_column_configs(
        frame=frame,
        configs=unfinalized_quant_column_configs,
        max_n_final=max_n_final,
        frame_stats=frame_statistics,
    )
    for col_config in quant_column_configs:
        logging.info("Column %s: step=%s, clip=%s", col_config.name, col_config.step, col_config.clip)

    return QuantizationConfig(
        quant_column_configs=quant_column_configs,
        agg_plan=_get_agg_plan(columns),
        output_dir=output_dir,
    )


def _save_quantization_config(path: pathlib.Path, config: QuantizationConfig):
    with open(path / FILE_QUANTIZATION_CONFIG, "wb") as file:
        pickle.dump(
            {
                "quant_column_configs": list(config.quant_column_configs),
                "agg_plan": config.agg_plan,
            },
            file,
        )


def _load_quantization_config(path: pathlib.Path) -> QuantizationConfig:
    path_config = path / FILE_QUANTIZATION_CONFIG
    if not path_config.exists():
        raise ValueError(
            f"Quantization config was not found at {path_config}. "
            f"Run --step {ArgQuantizationStep.QUANTIZE.value} first."
        )

    with open(path_config, "rb") as file:
        payload = pickle.load(file)

    return QuantizationConfig(
        quant_column_configs=payload["quant_column_configs"],
        agg_plan=payload["agg_plan"],
        output_dir=path,
    )


def quantize(path: pathlib.Path, transform: Callable, filter_rows: Callable, clip: bool,
             l2_l3_columns: list[str] | None, extent: list[float], max_n_final: int, chunk_size: int):
    """
    Quantize bucket data in one streaming pass.
    """
    path_final = path / "final.parquet"
    if path_final.exists():
        return

    path_frame_statistics = path / FILE_FRAME_STATISTICS
    if not path_frame_statistics.exists():
        raise ValueError(
            f"Frame statistics were not found at {path_frame_statistics}. "
            f"Run --step {ArgQuantizationStep.STATISTICS.value} first."
        )
    unc_dict = transform(get_uncertainties_dict(TC_COLUMNS))
    quant_columns = transform(TC_COLUMNS)
    with open(path_frame_statistics, "rb") as file:
        frame_statistics: FrameStatistics = pickle.load(file)

    # 1. Lazy reading from bucket
    # All columns if no L2/L3 columns are given
    columns = None if l2_l3_columns is None else \
        set(TC_COLUMNS + [COLUMN_LON, COLUMN_LAT, COLUMN_TIME] + l2_l3_columns)
    frame: Frame = gpm.bucket.read(
        bucket_dir=DIR_BUCKET,
        columns=columns,
        extent=extent,
        backend="polars_lazy",
    )
    # Filters and transform are applied to each collected chunk: in the lazy frame, Polars reads far ahead
    # of the processing (about 120 GB on the full bucket)
    preprocess = lambda chunk: _prepare_for_quantization(transform(filter_rows(chunk)))

    # 2. Quantizing
    if (path / FILE_QUANTIZATION_CONFIG).exists():
        # Resuming an interrupted run: quantize the remaining chunks with the config of the existing ones
        config = _load_quantization_config(path)
    else:
        frame_prepared = preprocess(frame)
        config = _create_quantization_config(
            columns=frame_prepared.collect_schema().names(),
            quant_columns=quant_columns,
            output_dir=path,
            frame=frame_prepared,
            frame_statistics=frame_statistics,
            uncertainty_dict=unc_dict,
            max_n_final=max_n_final,
            clip=clip,
        )
        _save_quantization_config(path, config)

    # Chunks already quantized by an interrupted run are skipped (but still read)
    quantize_streaming(
        frame=frame,
        config=config,
        chunk_size=chunk_size,
        preprocess=preprocess,
        skip_existing=True,
    )


def collect_statistics(path: pathlib.Path, transform: Callable, filter_rows: Callable, extent: list[float],
                       factors: list[float], chunk_size: int):
    path_frame_statistics = path / FILE_FRAME_STATISTICS
    if path_frame_statistics.exists():
        return

    quant_columns = transform(TC_COLUMNS)
    uncertainty_dict = transform(get_uncertainties_dict(TC_COLUMNS))

    # 1. Reading from bucket
    # Only quantization columns are relevant for frame statistics.
    # Keep filter-related columns for pre-statistics filtering.
    columns = set(TC_COLUMNS + [COLUMN_LON, COLUMN_LAT, COLUMN_TIME]) | {
        VARIABLE_SURFACE_TYPE_INDEX,
        COLUMN_L1C_QUALITY_FLAG,
        COLUMN_TEMP_2M_INDEX,
        COLUMN_SUN_GLINT_ANGLE_LF,
        COLUMN_SUN_GLINT_ANGLE_HF,
    }
    frame: Frame = gpm.bucket.read(
        bucket_dir=DIR_BUCKET,
        columns=columns,
        extent=extent,
        backend="polars_lazy",
    )

    # 2. Collecting frame statistics
    # Filters and transform are applied to each collected chunk (see quantize).
    # Statistics only use the quant columns: other columns would be read and averaged in every trial quantization
    frame_statistics = collect_frame_statistics(
        frame=frame,
        quant_columns=quant_columns,
        uncertainties=uncertainty_dict,
        factors=factors,
        available_memory_gb=None,
        chunk_size=chunk_size,
        preprocess=lambda chunk: _prepare_for_quantization(transform(filter_rows(chunk))).select(quant_columns),
    )
    logging.info("Estimated number of signatures per factor: %s", frame_statistics.factor2size)

    with open(path_frame_statistics, "wb") as file:
        pickle.dump(frame_statistics, file)


def merge(path: pathlib.Path, chunk_size: int):
    """
    Merge quantized fractions of data from bucket.
    """
    if (path / "final.parquet").exists():
        return

    partial_paths = sorted([p for p in path.glob("quantized_*.parquet") if p.is_file()])

    if len(partial_paths) == 0:
        raise ValueError(f"No quantized parquet files found in {path}.")

    config = _load_quantization_config(path)
    result = merge_streaming(
        frames_quant=[pl.scan_parquet(p) for p in partial_paths],
        config=config,
        chunk_size=chunk_size,
    )
    result.write_parquet(path / "final.parquet")


def _filter_rows(frame: Frame, filter_by_quality: bool, arg_surface_type: ArgSurfaceType,
                 month: int | None = None, year: int | None = None) -> Frame:
    if year is not None:
        frame = frame.filter(pl.col(COLUMN_TIME).dt.year() == year)
    if month is not None:
        frame = frame.filter(pl.col(COLUMN_TIME).dt.month() == month)

    if arg_surface_type == ArgSurfaceType.ALL:
        frame = frame
    else:
        frame = frame.filter(pl.col(VARIABLE_SURFACE_TYPE_INDEX).is_in(arg_surface_type.indexes()))

    if filter_by_quality:
        frame = filter_by_value_range(frame, COLUMN_SUN_GLINT_ANGLE_LF, SUN_GLINT_PRESENCE_RANGE, filter_out=True)
        frame = filter_by_value_range(frame, COLUMN_SUN_GLINT_ANGLE_HF, SUN_GLINT_PRESENCE_RANGE, filter_out=True)

        flag_values = [
            QUALITY_FLAG_NON_NORMAL_STATUS_MODE,
        ]
        frame = frame.filter(pl.col(COLUMN_L1C_QUALITY_FLAG).is_in(flag_values).not_())

    return frame


def _get_clip_range_from_frame_statistics(frame_statistics: FrameStatistics, column: str) -> tuple[float, float]:
    # Count-weighted quantiles of the ECDF bins; nulls excluded (its `quantile` column counts them, sorted first)
    ecdf = frame_statistics.column2ecdf[column].filter(pl.col(column).is_not_null())
    q_min, q_max = weighted_quantiles(ecdf, [CLIP_QUANTILE, 1 - CLIP_QUANTILE], column, "count")
    return q_min, q_max


def main():
    log_dir = pathlib.Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    logging.basicConfig(filename=log_path, level=logging.INFO)
    print(f"Logging to {log_path}")

    parser = configargparse.ArgumentParser(config_arg_is_required=True, args_for_setting_config_path=["--config"],
                                           description="Run quantization (quantize/merge only)")

    parser.add_argument("--step", type=ArgQuantizationStep, action=EnumAction,
                        help="Quantization pipeline's step to perform")
    parser.add_argument("--transform", default=ArgTransform.DEFAULT, type=ArgTransform, action=EnumAction,
                        help="Type of transformation to perform on data")
    parser.add_argument("--dir", default=DIR_PMW_ANALYSIS,
                        help="Path to the directory to store quantized data in")
    parser.add_argument("--month", type=int, help="Month of the data to quantize")
    parser.add_argument("--year", type=int, help="Year of the data to quantize")
    parser.add_argument("--l2-l3-columns", nargs="+",
                        help="L2 and L3 columns to read and aggregate besides Tc, lon, lat, and time; "
                             "all columns if not set")
    parser.add_argument("--surface-type", default=ArgSurfaceType.ALL,
                        type=ArgSurfaceType, action=EnumAction,
                        help="Surface type to process during quantization")
    parser.add_argument("--clip", type=parse_bool, default=False,
                        help="If true, quantized columns are clipped to their 1e-6 and 1 - 1e-6 quantiles "
                             "from frame statistics")
    parser.add_argument("--filter-by-quality", type=parse_bool, default=True,
                        help="If true, data is filtered by sun glint angle and quality flags")
    parser.add_argument("--lat-limit", type=float,
                        help="If set, only data with |lat| <= lat-limit is read")
    parser.add_argument("--max-n-final", type=int, default=MAX_N_FINAL,
                        help="Maximum estimated number of signatures when choosing the uncertainty factor")
    parser.add_argument("--factors", type=float, nargs="+", default=FACTORS,
                        help="Uncertainty factors to estimate signature counts for")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE_QUANTIZE,
                        help="Rows per chunk in the statistics and quantize steps")
    parser.add_argument("--chunk-size-merge", type=int, default=CHUNK_SIZE_MERGE,
                        help="Maximum rows merged at once in the merge step")
    args = parser.parse_args()

    extent = FULL_EXTENT if args.lat_limit is None else [-180, 180, -args.lat_limit, args.lat_limit]

    path = pathlib.Path(args.dir) / "new" / args.transform.value / args.surface_type.value
    path.mkdir(parents=True, exist_ok=True)

    transform = get_transformation_function(args.transform)
    filter_rows = lambda frame: _filter_rows(
        frame=frame,
        filter_by_quality=args.filter_by_quality,
        arg_surface_type=args.surface_type,
        month=args.month,
        year=args.year,
    )

    if args.year is not None:
        path = path / str(args.year)
    if args.month is not None:
        path = path / str(args.month)
    path.mkdir(parents=True, exist_ok=True)

    with timing(f"Step {args.step}"):
        match args.step:
            case ArgQuantizationStep.STATISTICS:
                collect_statistics(path, transform, filter_rows, extent, args.factors, args.chunk_size)
            case ArgQuantizationStep.QUANTIZE:
                quantize(path, transform, filter_rows, args.clip, args.l2_l3_columns, extent, args.max_n_final,
                         args.chunk_size)
            case ArgQuantizationStep.MERGE:
                merge(path, args.chunk_size_merge)
            case _:
                raise ValueError(f"{args.step.value} is not supported.")


if __name__ == '__main__':
    main()
