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
    ArgQuantizationStep, ArgTransform, ArgQuantizationL2L3Columns, VARIABLE_SURFACE_TYPE_INDEX, COLUMN_L1C_QUALITY_FLAG, \
    ArgSurfaceType, COLUMN_SUN_GLINT_ANGLE_HF, COLUMN_SUN_GLINT_ANGLE_LF, SUN_GLINT_PRESENCE_RANGE, \
    QUALITY_FLAG_NON_NORMAL_STATUS_MODE, COLUMN_TEMP_2M_INDEX
from pmw_analysis.copypaste.utils.cli import EnumAction
from pmw_analysis.preprocessing.aggregation_defaults import build_default_per_column_aggregations
from pmw_analysis.preprocessing.dataframe_polars import get_uncertainties_dict
from pmw_analysis.preprocessing.transforms import get_transformation_function
from pmw_analysis.processing.filter import filter_by_value_range
from quantization.aggregation import AggregationPlan, AggMax, AggMean, AggMin
from quantization.api import collect_frame_statistics, quantize_streaming, merge_streaming, \
    finalize_quant_column_configs
from quantization.config import QuantizationConfig
from quantization.constants import COLUMN_SUFFIX_QUANT
from quantization.quant_columns import FixedStepQuantColumnConfig, UncertaintyQuantColumnConfig
from quantization.types import Frame, FrameStatistics

FULL_EXTENT = [-180, 180, -90, 90]
QUANTIZATION_CHUNK_SIZE = 200_000_000
MAX_N_FINAL = 2 * 10 ** 10
FILE_FRAME_STATISTICS = "frame_statistics.pkl"
FILE_QUANTIZATION_CONFIG = "quantization_config.pkl"


def _prepare_for_quantization(frame: Frame) -> Frame:
    frame = frame.fill_nan(None)
    frame = _replace_special_missing_values_with_null(frame)
    frame = frame.with_columns(pl.col(COLUMN_TIME).dt.ordinal_day().alias("day"))
    frame = frame.drop(COLUMNS_TO_DROP, strict=False)

    return frame

def _replace_special_missing_values_with_null(frame: Frame) -> Frame:
    schema = frame.collect_schema()  # {col: dtype}

    special_values = {
        COLUMN_SUN_GLINT_ANGLE_LF: [-99, -9999, -88],
        COLUMN_SUN_GLINT_ANGLE_HF: [-99, -9999, -88],
    }

    int_types = {pl.Int8, pl.Int16, pl.Int32, pl.Int64}

    exprs = [
        pl.when(pl.col(col).is_in(special_values.get(col, [-99, -9999])))
          .then(None)
          .otherwise(pl.col(col))
          .alias(col)
        for col, dtype in schema.items()
        if col != COLUMN_LON and dtype in int_types
    ]

    if schema.get(COLUMN_LON) in int_types:
        exprs.append(pl.col(COLUMN_LON).replace(-9999, None))

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
        max_n_final=MAX_N_FINAL,
        frame_stats=frame_statistics,
    )

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


def _get_columns_for_l2_l3(
    l2_l3_columns: ArgQuantizationL2L3Columns,
    required_columns: set[str],
) -> set[str] | None:
    match l2_l3_columns:
        case ArgQuantizationL2L3Columns.NONE:
            return required_columns
        case ArgQuantizationL2L3Columns.ANALYSIS_MINIMUM:
            return required_columns | {
                VARIABLE_SURFACE_TYPE_INDEX,
                COLUMN_L1C_QUALITY_FLAG,
                COLUMN_TEMP_2M_INDEX,
                COLUMN_SUN_GLINT_ANGLE_LF,
                COLUMN_SUN_GLINT_ANGLE_HF,
            }
        case ArgQuantizationL2L3Columns.ALL:
            return None


def quantize(path: pathlib.Path, transform: Callable, filter_rows: Callable, clip: bool,
             l2_l3_columns: ArgQuantizationL2L3Columns):
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
    required_columns = set(TC_COLUMNS + [COLUMN_LON, COLUMN_LAT, COLUMN_TIME])
    columns = _get_columns_for_l2_l3(l2_l3_columns, required_columns)

    frame: Frame = gpm.bucket.read(
        bucket_dir=DIR_BUCKET,
        columns=columns,
        extent=FULL_EXTENT,
        backend="polars_lazy",
    )

    frame = transform(filter_rows(frame))

    # 2. Quantizing
    frame_prepared = _prepare_for_quantization(frame)
    config = _create_quantization_config(
        columns=frame_prepared.collect_schema().names(),
        quant_columns=quant_columns,
        output_dir=path,
        frame=frame_prepared,
        frame_statistics=frame_statistics,
        uncertainty_dict=unc_dict,
        clip=clip,
    )
    _save_quantization_config(path, config)

    quantize_streaming(
        frame=frame_prepared,
        config=config,
        chunk_size=QUANTIZATION_CHUNK_SIZE,
    )


def collect_statistics(path: pathlib.Path, transform: Callable, filter_rows: Callable):
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
        extent=FULL_EXTENT,
        backend="polars_lazy",
    )

    frame = transform(filter_rows(frame))
    frame_prepared = _prepare_for_quantization(frame)

    # 2. Collecting frame statistics
    frame_statistics = collect_frame_statistics(
        frame=frame_prepared,
        quant_columns=quant_columns,
        uncertainties=uncertainty_dict,
        factors=[1.0, 2.0, 4.0, 8.0],
        available_memory_gb=None,
        chunk_size=QUANTIZATION_CHUNK_SIZE,
    )

    with open(path_frame_statistics, "wb") as file:
        pickle.dump(frame_statistics, file)


def merge(path: pathlib.Path):
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
        chunk_size=QUANTIZATION_CHUNK_SIZE,
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


def _get_clip_range_from_frame_statistics(frame_statistics: FrameStatistics, column: str) -> tuple[float, float] | None:
    ecdf = frame_statistics.column2ecdf.get(column)

    if ecdf is not None and isinstance(ecdf, pl.Series):
        values = ecdf.drop_nulls()
        if len(values) != 0:
            return values.min(), values.max()

    return None


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
    parser.add_argument("--l2-l3-columns", default=ArgQuantizationL2L3Columns.ALL,
                        type=ArgQuantizationL2L3Columns, action=EnumAction,
                        help="L2 and L3 columns to process during quantization")
    parser.add_argument("--surface-type", default=ArgSurfaceType.ALL,
                        type=ArgSurfaceType, action=EnumAction,
                        help="Surface type to process during quantization")
    parser.add_argument("--clip", type=bool, default=False,
                        help="If true, quantized columns are clipped to min/max inferred from frame statistics")
    parser.add_argument("--filter-by-quality", type=bool, default=True,
                        help="If true, data is filtered by sun glint angle and quality flags")
    args = parser.parse_args()

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

    match args.step:
        case ArgQuantizationStep.STATISTICS:
            collect_statistics(path, transform, filter_rows)
        case ArgQuantizationStep.QUANTIZE:
            quantize(path, transform, filter_rows, args.clip, args.l2_l3_columns)
        case ArgQuantizationStep.MERGE:
            merge(path)
        case _:
            raise ValueError(f"{args.step.value} is not supported.")


if __name__ == '__main__':
    main()
