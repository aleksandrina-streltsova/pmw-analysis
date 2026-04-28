"""
This module contains methods for quantization of data using Polars data frames.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Sequence

import pandas as pd
import polars as pl
from tqdm import tqdm

from pmw_analysis.constants import COLUMN_COUNT, STRUCT_FIELD_COUNT, COLUMN_TIME, COLUMN_LON, COLUMN_LAT, \
    COLUMN_OCCURRENCE, FLAG_DEBUG, AGG_OFF_LIMIT, COLUMN_SUN_GLINT_ANGLE_LF, COLUMN_SUN_GLINT_ANGLE_HF, Stats


#### COLUMNS ####
def _get_columns_to_drop(columns: Sequence[str]) -> Sequence[str]:
    columns_to_drop = [
        'gpm_id',
        'gpm_granule_id',
        'gpm_cross_track_id',
        'gpm_along_track_id',
        'orbit_mode',
        'time',
    ]
    return [col for col in columns if col in columns_to_drop]


def _get_tc_columns(columns: Sequence[str]) -> Sequence[str]:
    return [col for col in columns if col.startswith("Tc")]


def _get_flag_columns(columns: Sequence[str]) -> Sequence[str]:
    flag_columns = [
        'Quality_LF',  # 0 .. 3, -99
        'SCorientation',  # 0 180, -8000, -9999
        'Quality_HF',  # 0 .. 3, -99
        'L1CqualityFlag',  # -10, -7 .. 4
        'airmassLiftIndex',  # 0 .. 3
        'pixelStatus',  # 0 .. 5, -99
        'qualityFlag',  # 0 .. 3, -99
        'surfaceTypeIndex',  # 1 .. 18, -9999
    ]
    return [col for col in columns if col in flag_columns]


# TODO: depends on info
def _get_periodic_columns(columns: Sequence[str]) -> Sequence[str]:
    periodic_columns = [
        'sunLocalTime',  # mean of [0, 12], mean of [12, 24]
        'lon',  # mean of [-180, 0], mean of [0, 180]
        'day',  # month: mean of [1, 183], mean of [183, 365]
    ]
    return [col for col in columns if col in periodic_columns]


def _get_special_columns(columns: Sequence[str]) -> Sequence[str]:
    special_columns = [
        COLUMN_SUN_GLINT_ANGLE_LF,  # mean, [0, 127], -88, -99
        COLUMN_SUN_GLINT_ANGLE_HF,  # mean, [0, 127] -88, -99
    ]
    return [col for col in columns if col in special_columns]


def _get_periodic_dict() -> Dict[str, float]:
    return {
        'sunLocalTime': 12.,
        'lon': 0.,
        'day': 366. / 2,
    }


def _get_special_dict() -> Dict[str, Tuple[float, str]]:
    return {
        COLUMN_SUN_GLINT_ANGLE_LF: (-88, "below_horizon"),
        COLUMN_SUN_GLINT_ANGLE_HF: (-88, "below_horizon"),
    }


#### PROCESSING MISSING VALUES ####
def _replace_special_missing_values_with_null(lf: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    columns = lf.collect_schema().names()
    lf_result = lf.with_columns([
                                    pl.col(col).replace({-99: None, -9999: None})
                                    for col in columns
                                    if col != "lon"  # lon ranges from -180 to 180 and can be equal to -99
                                ] + [pl.col("lon").replace(-9999, None)])

    return lf_result


#### ROUNDING ####
_FREQUENCY_GHZ = "Frequency (GHz)"
_EXPECTED_NEDT = "Expected NEDT"
_EXPECTED_CALIBRATION_UNCERTAINTY = "Expected Calibration Uncert."
_EXPECTED_TOTAL_UNCERTAINTY = "Expected Total Uncert."


def _get_gmi_characteristics() -> pd.DataFrame:
    data = {
        _FREQUENCY_GHZ: ["10.65", "18.7", "23.8", "36.64", "89.0", "166.0", "183.31±3", "183.31±7"],
        "Polarization": ["V/H", "V/H", "V", "V/H", "V/H", "V/H", "V", "V"],
        "NEDT/Reqmt (K)": [0.96, 0.84, 1.05, 0.65, 0.57, 1.5, 1.5, 1.5],
        _EXPECTED_NEDT: [0.96, 0.82, 0.82, 0.56, 0.40, 0.81, 0.87, 0.81],
        "Expected Beam Efficiency (%)": [91.4, 92.0, 92.5, 96.6, 95.6, 91.9, 91.7, 91.7],
        _EXPECTED_CALIBRATION_UNCERTAINTY: [1.04, 1.08, 1.26, 1.20, 1.19, 1.20, 1.20, 1.20],
        "Resolution (km)": ["32.1 x 19.4", "18.1 x 10.9", "16.0 x 9.7", "15.6 x 9.4", "7.2 x 4.4", "6.3 x 4.1",
                            "5.8 x 3.8", "5.8 x 3.8"]
    }
    df = pd.DataFrame(data)
    df[_EXPECTED_TOTAL_UNCERTAINTY] = (df[_EXPECTED_NEDT] ** 2 + df[_EXPECTED_CALIBRATION_UNCERTAINTY] ** 2) ** 0.5

    df.set_index(_FREQUENCY_GHZ, inplace=True)
    return df


def _get_frequency(key: str):
    match key:
        case "10V" | "10H":
            freq = "10.65"
        case "19V" | "19H":
            freq = "18.7"
        case "23V":
            freq = "23.8"
        case "37V" | "37H":
            freq = "36.64"
        case "89V" | "89H":
            freq = "89.0"
        case "165V" | "165H":
            freq = "166.0"
        case "183V3":
            freq = "183.31±3"
        case "183V7":
            freq = "183.31±7"
        case _:
            raise KeyError(f"Unknown key {key}")

    return freq


def get_uncertainties_dict(tc_columns: Sequence[str]) -> Dict[str, float]:
    """
    Create a dictionary of expected uncertainties for each Tc column.
    """
    gmi_characteristics = _get_gmi_characteristics()
    uncertainties = {
        col: gmi_characteristics.loc[_get_frequency(col.removeprefix("Tc_"))][_EXPECTED_TOTAL_UNCERTAINTY]
        for col in tc_columns
    }
    return uncertainties


def _round(lf: pl.DataFrame | pl.LazyFrame,
           quant_columns: Sequence[str],
           quant_steps: Sequence[float | int],
           quant_ranges: Sequence[Tuple[float | int, float | int]] | None) -> pl.DataFrame | pl.LazyFrame:
    if quant_ranges is not None:
        lf = lf.with_columns([pl.col(c).clip(lower, upper) for c, (lower, upper) in zip(quant_columns, quant_ranges)])

    lf_result = lf.with_columns([
        (pl.col(col) / step).round() * step
        for col, step in zip(quant_columns, quant_steps)
    ])

    return lf_result


#### QUANTIZATION ####
@dataclass
class DataFrameQuantizationInfo:
    """
    Data class containing information required for performing quantization algorithm on provided Polars data frame.

    Attributes
    ----------
    quant_columns : Sequence[str]
        Columns whose values will be quantized (rounded to nearest step) and used for grouping.

    quant_steps :  Sequence[float | int]
        Step sizes used for quantizing each column in `quant_columns`. Must match the length of `quant_columns`.

    quant_ranges : Sequence[Tuple[float | int, float | int]], optional
        Value ranges used for clipping (limiting) `quant_columns`. Must match the length of `quant_columns`.

    flag_columns : Sequence[str], optional
        Columns with discrete classification or quality flags (e.g., cloud status, surface type).
        These are aggregated by computing value counts, producing a list of structs per group.

    periodic_columns : Sequence[str], optional
        Columns with periodic or cyclic values (e.g., longitude, local solar time, day of year).
        These are aggregated using conditional means split at a predefined midpoint:
            - Values less than or equal to the midpoint (e.g., before noon) are averaged separately from
              values greater than the midpoint (e.g., after noon).
        This allows better representation of mean behavior across cycles.

    special_columns : Sequence[str], optional
        Columns with special sentinel values indicating non-physical states (e.g., -88 = "below horizon").
        These are aggregated by computing:
          - Mean of valid (non-sentinel) values.
          - Count of sentinel values.
          - Count of valid values.

    agg_min_columns : Sequence[str], optional
        Columns where only the minimum value is retained for each group.
        This is typically used for timestamp fields (e.g., earliest observation in group).

    agg_max_columns : Sequence[str], optional
        Columns where only the maximum value is retained for each group.
        This is typically used for timestamp fields (e.g., latest observation in group).

    agg_off_columns : Sequence[str], optional
        Columns whose values are stored in lists, without aggregation.

    periodic_dict : Dict[str, float], optional
        Midpoints for each periodic column, used to separate the value space during aggregation.
        Example: {'sunLocalTime': 12.0, 'lon': 0.0, 'day': 183.0}

    special_dict : Dict[str, Tuple[float, str]], optional
        Dictionary specifying the sentinel value and its label for each `special_column`.
        Example: {'sunGlintAngle_LF': (-88, "below_horizon")}

    agg_off_limit : int
        Number of rows to include in `agg_off_columns` before dropping them.

    """
    quant_columns: Sequence[str]
    quant_steps: Sequence[float | int] = None
    quant_ranges: Sequence[Tuple[float | int, float | int]] = None
    flag_columns: Sequence[str] = ()
    periodic_columns: Sequence[str] = ()
    special_columns: Sequence[str] = ()
    # TODO: allow providing a dictionary of columns with corresponding operations
    agg_min_columns: Sequence[str] = ()
    agg_max_columns: Sequence[str] = ()
    agg_off_columns: Sequence[str] = ()
    periodic_dict: Dict[str, float] = None
    special_dict: Dict[str, Tuple[float, str]] = None
    agg_off_limit: int = AGG_OFF_LIMIT

    @staticmethod
    def create(columns: Sequence[str], quant_columns: Sequence[str], quant_steps: Sequence[float | int] = None,
               quant_ranges: Sequence[Tuple[float | int, float | int]] = None,
               agg_off_columns: Sequence[str] = (),
               periodic_dict: Dict[str, float] = None,
               special_dict: Dict[str, Tuple[float, str]] = None,
               agg_off_limit: int = AGG_OFF_LIMIT,
               ) -> "DataFrameQuantizationInfo":
        """
        Create a DataFrameQuantizationInfo object from a list of columns.
        """
        agg_on_columns = [col for col in columns if col not in agg_off_columns]

        return DataFrameQuantizationInfo(
            quant_columns=quant_columns,
            quant_steps=quant_steps,
            quant_ranges=quant_ranges,
            flag_columns=_get_flag_columns(agg_on_columns),
            periodic_columns=_get_periodic_columns(agg_on_columns),
            special_columns=_get_special_columns(agg_on_columns),
            agg_min_columns=[COLUMN_OCCURRENCE],
            agg_max_columns=[COLUMN_OCCURRENCE],
            agg_off_columns=agg_off_columns,
            periodic_dict=periodic_dict,
            special_dict=special_dict,
            agg_off_limit=agg_off_limit,
        )

    def get_agg_mean_columns(self, columns: Sequence[str]) -> Sequence[str]:
        """
        Return the columns whose values are aggregated using averaging function.
        """
        agg_min_columns = [get_agg_column(col, Stats.MIN) for col in self.agg_min_columns]
        agg_max_columns = [get_agg_column(col, Stats.MAX) for col in self.agg_max_columns]

        return [col for col in columns if
                col not in self.quant_columns and
                col not in self.flag_columns and
                col not in self.periodic_columns and
                col not in self.special_columns and
                col not in self.agg_min_columns and col not in agg_min_columns and
                col not in self.agg_max_columns and col not in agg_max_columns and
                col not in self.agg_off_columns]


def _aggregate(lf: pl.DataFrame | pl.LazyFrame, info: DataFrameQuantizationInfo) -> pl.DataFrame | pl.LazyFrame:
    columns = lf.collect_schema().names()
    agg_mean_cols = info.get_agg_mean_columns(columns)

    # estimated_size = df.estimated_size() / (1024 * 1024)
    # logging.info("Estimated dataframe size: %.2f MB", estimated_size)

    lf_result = (
        lf
        .group_by(info.quant_columns)
        .agg(
            *[pl.col(flag_col).value_counts(name=STRUCT_FIELD_COUNT) for flag_col in info.flag_columns],

            *[expr
              for col in info.periodic_columns
              for expr in
              [pl.when(pl.col(col) <= info.periodic_dict[col]).then(pl.col(col)).mean().alias(f"{col}_lt"),
               pl.when(pl.col(col) <= info.periodic_dict[col]).then(pl.col(col)).count().alias(f"{col}_lt_count"),
               pl.when(pl.col(col) > info.periodic_dict[col]).then(pl.col(col)).mean().alias(f"{col}_gt"),
               pl.when(pl.col(col) > info.periodic_dict[col]).then(pl.col(col)).count().alias(f"{col}_gt_count")]],

            *[expr
              for col in info.special_columns
              for expr in [pl.col(col).filter(pl.col(col).ne(info.special_dict[col][0])).mean(),
                           pl.col(col).filter(pl.col(col).eq(info.special_dict[col][0])).count().alias(
                               f"{col}_{info.special_dict[col][1]}_count"),
                           pl.col(col).filter(
                               pl.col(col).ne(info.special_dict[col][0]) & pl.col(col).is_not_null()).count().alias(
                               f"{col}_count")]],

            *[expr
              for mean_col in agg_mean_cols
              for expr in
              [pl.col(mean_col).mean(), (pl.len() - pl.col(mean_col).null_count()).alias(f"{mean_col}_count")]],

            pl.col(info.agg_off_columns).head(info.agg_off_limit),

            *[pl.col(min_col).min().alias(get_agg_column(min_col, Stats.MIN)) for min_col in info.agg_min_columns],
            *[pl.col(max_col).max().alias(get_agg_column(max_col, Stats.MAX)) for max_col in info.agg_max_columns],
            pl.len().alias(COLUMN_COUNT),
        )
    )

    if FLAG_DEBUG:
        assert (lf_result.select(pl.col(COLUMN_COUNT).sum()).collect(engine="streaming").item() ==
                lf.select(pl.len()).collect(engine="streaming").item())
    return lf_result


#### PREPROCESSING PIPELINE ####
def quantize_pmw_features(lf: pl.DataFrame | pl.LazyFrame, quant_columns: Sequence[str],
                          uncertainty_dict: Dict[str, float],
                          range_dict: Dict[str, float],
                          agg_off_columns: Sequence[str] = (),
                          agg_off_limit: int = AGG_OFF_LIMIT,
                          ) -> pl.DataFrame | pl.LazyFrame:
    """
    Quantize PMW feature columns and performs group-wise aggregation on data stored in Polars format.
    """
    # 1. Replace NaN, -99, and -9999 by nulls.
    lf = lf.fill_nan(None)
    lf = _replace_special_missing_values_with_null(lf)

    # 2. Extract ordinal day from time column.
    lf = lf.with_columns(
        pl.col(COLUMN_TIME).dt.ordinal_day().alias("day")
    )

    # 3. Create a column with occurrence info (`time`, `lat`, and `lon`).
    lf = create_occurrence_column(lf)

    # 4. Drop id columns. Remove `time` column, since we have `sunLocalTime`, `day`, and `occurrence` columns now.
    lf = lf.drop(_get_columns_to_drop([col for col in lf.collect_schema().names() if col not in agg_off_columns]))
    lf_agg_columns = [col for col in lf.collect_schema().names() if col not in agg_off_columns]

    # 5. Quantize Tc columns and group duplicate signatures.
    quant_steps = [uncertainty_dict[col] for col in quant_columns]
    quant_ranges = [(range_dict[f"{col}_min"], range_dict[f"{col}_max"]) for col in quant_columns]

    info = DataFrameQuantizationInfo.create(lf_agg_columns, quant_columns, quant_steps, quant_ranges,
                                            agg_off_columns=agg_off_columns,
                                            periodic_dict=_get_periodic_dict(),
                                            special_dict=_get_special_dict(),
                                            agg_off_limit=agg_off_limit)

    return quantize_features(lf, info)


def quantize_features(lf: pl.DataFrame | pl.LazyFrame, info: DataFrameQuantizationInfo) -> pl.DataFrame | pl.LazyFrame:
    """
    Quantize specified feature columns and performs group-wise aggregation on data stored in Polars format.

    This function performs two main operations:
      1. Rounds numerical features (`quant_columns`).
      2. Aggregates data based on the quantized values, computing descriptive statistics over other types of columns.

    Parameters
    ----------
    lf : pl.DataFrame | pl.LazyFrame
        The input data stored in Polars format containing features to quantize and aggregate.

    info : DataFrameQuantizationInfo
        Input information required for performing quantization algorithm.

    Returns
    -------
        pl.DataFrame | pl.LazyFrame
            Aggregated data where rows with similar feature profiles are grouped and summarized.
    """

    lf_result = _round(lf, info.quant_columns, info.quant_steps, info.quant_ranges)
    lf_result = _aggregate(lf_result, info)

    # logging.info("[quantize] query explanation:\n%s", lf.explain(streaming=True))

    if FLAG_DEBUG:
        row_count_before = lf.select(pl.len()).collect(engine="streaming").item()
        row_count_after = lf_result.select(pl.len()).collect(engine="streaming").item()
        assert row_count_before == lf_result.select(pl.col(COLUMN_COUNT).sum()).collect(engine="streaming").item()
        logging.info("Quantized features into %d/%d", row_count_after, row_count_before)

    return lf_result


#### MERGING PREPROCESSED DATA ####
def _get_struct_list_type(flag_column: str) -> pl.List:
    return pl.List(pl.Struct([
        pl.Field(flag_column, pl.Int16),
        pl.Field(STRUCT_FIELD_COUNT, pl.Int64)
    ]))


def _aggregate_structs(lf: pl.DataFrame | pl.LazyFrame, quant_columns: Sequence[str],
                       flag_column: str) -> pl.DataFrame | pl.LazyFrame:
    # TODO: the commented approach doesn't work with lazy frame, seemingly because of ".with_row_index()"
    # lf = lf.with_row_index() # track original rows
    # lf_result = (
    #     lf
    #     .select(["index", flag_column])
    #     .explode(flag_column)  # explode the List[Struct]
    #     .unnest(flag_column)  # turns into columns (flag_column, STRUCT_FIELD_COUNT)
    #     .group_by(["index", flag_column])
    #     .agg(pl.col(STRUCT_FIELD_COUNT).sum())
    #     .with_columns(pl.struct([flag_column, STRUCT_FIELD_COUNT]).alias(f"{flag_column}_tmp"))  # build back struct
    #     .group_by("index")
    #     .agg(pl.col(f"{flag_column}_tmp"))
    #     .join(lf.drop(flag_column), on="index")
    #     .rename({f"{flag_column}_tmp": flag_column})
    #     .drop("index")
    # )
    lf_result = (
        lf
        .select(list(quant_columns) + [flag_column])
        .with_row_index()  # track original rows
        .explode(flag_column)  # explode the List[Struct]
        .unnest(flag_column)  # turns into columns (flag_column, STRUCT_FIELD_COUNT)
        .group_by(["index", flag_column])
        .agg([pl.col(STRUCT_FIELD_COUNT).sum()] + [pl.col(quant_col).first() for quant_col in quant_columns])
        .with_columns(pl.struct([flag_column, STRUCT_FIELD_COUNT]).alias(f"{flag_column}_tmp"))  # build back struct
        .group_by("index")
        .agg([pl.col(f"{flag_column}_tmp")] + [pl.col(quant_col).first() for quant_col in quant_columns])
        .drop("index")
        .join(lf.drop(flag_column), on=quant_columns, nulls_equal=True)
        .rename({f"{flag_column}_tmp": flag_column})
    )

    return lf_result


def merge_quantized_pmw_features(lfs: Sequence[pl.DataFrame | pl.LazyFrame],
                                 quant_columns: Sequence[str],
                                 agg_off_columns: Sequence[str] = (),
                                 agg_off_limit: int = AGG_OFF_LIMIT,
                                 ) -> pl.DataFrame | pl.LazyFrame:
    """
    Merge quantized PMW features from a collection of Polars data structures using specified quantization columns.
    """
    assert len(lfs) > 0, "No data to merge was provided."
    columns = [col.removesuffix("_lt").removesuffix("_gt") for col in lfs[0].collect_schema().names() if
               not col.endswith("count") and not col.endswith("_gt")]
    info = DataFrameQuantizationInfo.create(columns, quant_columns,
                                            agg_off_columns=agg_off_columns,
                                            agg_off_limit=agg_off_limit)

    return merge_quantized_features(lfs, info)


def merge_quantized_features(lfs: Sequence[pl.DataFrame | pl.LazyFrame],
                             info: DataFrameQuantizationInfo) -> pl.DataFrame | pl.LazyFrame:
    """
    Merge quantized features from a collection of Polars data structures using specified quantization configurations.
    """

    # TODO: FIX
    lfs = [lf for lf in lfs if not lf.is_empty()]
    lf = pl.concat(lfs, how="diagonal")
    if lf.is_empty():
        return lf

    columns = [col.removesuffix("_lt").removesuffix("_gt") for col in lf.collect_schema().names() if
               not col.endswith("count") and not col.endswith("_gt")]
    agg_mean_cols = info.get_agg_mean_columns(columns)

    special_dict = _get_special_dict()

    def aggregate_mean(col):
        return (pl.col(col) * pl.col(f"{col}_count")).sum() / pl.col(f"{col}_count").sum()

    # estimated_size = df.estimated_size() / (1024 * 1024)
    # logging.info("Estimated dataframe size: %.2f MB", estimated_size)

    lf_result = (
        lf
        .group_by(info.quant_columns)
        .agg(*[pl.col(flag_col).flatten() for flag_col in info.flag_columns],
             *[expr
               for p_col in info.periodic_columns
               for expr in [aggregate_mean(f"{p_col}_lt"), pl.col(f"{p_col}_lt_count").sum(),
                            aggregate_mean(f"{p_col}_gt"), pl.col(f"{p_col}_gt_count").sum()]],
             *[expr
               for s_col in info.special_columns
               for expr in [aggregate_mean(s_col),
                            pl.col(f"{s_col}_{special_dict[s_col][1]}_count").sum(),
                            pl.col(f"{s_col}_count").sum()]],

             *[expr
               for mean_col in agg_mean_cols
               for expr in [aggregate_mean(mean_col),
                            pl.col(f"{mean_col}_count").sum()]],
             *[pl.col(get_agg_column(min_col, Stats.MIN)).min() for min_col in info.agg_min_columns],
             *[pl.col(get_agg_column(max_col, Stats.MAX)).max() for max_col in info.agg_max_columns],
             pl.col(info.agg_off_columns).list.explode().head(info.agg_off_limit),
             pl.col(COLUMN_COUNT).sum(),
             )
    )

    for flag_col in tqdm(info.flag_columns):
        lf_result = _aggregate_structs(lf_result, info.quant_columns, flag_col)

    # logging.info("query explanation:\n%s", lf_result.explain(streaming=True))

    if FLAG_DEBUG:
        row_count_before = lf.select(pl.len()).collect(engine="streaming").item()
        row_count_after = lf_result.select(pl.len()).collect(engine="streaming").item()
        logging.info("Quantized features into %d/%d", row_count_after, row_count_before)

    # assert df_result[COLUMN_COUNT].sum() == df[COLUMN_COUNT].sum()
    return lf_result


###############################################################################


def get_agg_column(column: str, func: Stats) -> str:
    return f"{column}_{func.value}"


def create_occurrence_column(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Create a new column representing occurrences
    based on a combination of timestamp, rounded longitude, and latitude values.
    """
    if COLUMN_OCCURRENCE in df.collect_schema().names():
        return df

    df = df.with_columns(
        (pl.col(COLUMN_TIME).cast(str) + "|" +
         pl.col(COLUMN_LON).round(1).cast(str) + "|" +
         pl.col(COLUMN_LAT).round(1).cast(str)).alias(COLUMN_OCCURRENCE)
    )
    return df


def expand_occurrence_column(df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """
    Expand all occurrence columns into three, containing time, longitude, and latitude.
    """
    occurrence_columns = [COLUMN_OCCURRENCE] + [get_agg_column(COLUMN_OCCURRENCE, stat) for stat in Stats]
    for column in occurrence_columns:
        if column not in df.collect_schema().names():
            continue
        df = df.with_columns(
            pl.col(column).str.split("|").list.get(0).str.to_datetime("%Y-%m-%d %H:%M:%S.%9f")
            .alias(f"{column}_{COLUMN_TIME}"),
            pl.col(column).str.split("|").list.get(1).str.to_decimal().alias(f"{column}_{COLUMN_LON}"),
            pl.col(column).str.split("|").list.get(2).str.to_decimal().alias(f"{column}_{COLUMN_LAT}"),
        )
    return df
