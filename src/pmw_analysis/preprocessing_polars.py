import logging
from typing import List, Dict, Tuple, Iterable

import pandas as pd
import polars as pl
from tqdm import tqdm

from pmw_analysis.constants import COLUMN_COUNT, STRUCT_FIELD_COUNT, COLUMN_TIME
from pmw_analysis.utils.performance import get_memory_usage


#### COLUMNS ####
def _get_columns_to_drop(columns: Iterable[str]) -> List[str]:
    columns_to_drop = [
        'gpm_id',
        'gpm_granule_id',
        'gpm_cross_track_id',
        'gpm_along_track_id',
        'orbit_mode',
    ]
    return [col for col in columns if col in columns_to_drop]


def _get_tc_columns(columns: Iterable[str]) -> List[str]:
    return [col for col in columns if col.startswith("Tc")]


def _get_flag_columns(columns: Iterable[str]) -> List[str]:
    flag_columns = [
        'Quality_LF',  # 0 .. 3, -99
        'SCorientation',  # 0 180, -8000, -9999
        'Quality_HF',  # 0 .. 3, -99
        'L1CqualityFlag',  # -10, -7 .. 0
        'airmassLiftIndex',  # 0 .. 3
        'pixelStatus',  # 0 .. 5, -99
        'qualityFlag',  # 0 .. 3, -99
        'surfaceTypeIndex',  # 1 .. 18, -9999
    ]
    return [col for col in columns if col in flag_columns]


# TODO: depends on info
def _get_periodic_columns(columns: Iterable[str]) -> List[str]:
    periodic_columns = [
        'sunLocalTime',  # mean of [0, 12], mean of [12, 24]
        'lon',  # mean of [-180, 0], mean of [0, 180]
        'day',  # month: mean of [1, 183], mean of [183, 365]
    ]
    return [col for col in columns if col in periodic_columns]


def _get_special_columns(columns: Iterable[str]) -> List[str]:
    special_columns = [
        'sunGlintAngle_LF',  # mean, [0, 127], -88, -99
        'sunGlintAngle_HF',  # mean, [0, 127] -88, -99
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
        'sunGlintAngle_LF': (-88, "below_horizon"),
        'sunGlintAngle_HF': (-88, "below_horizon"),
    }


#### PROCESSING MISSING VALUES ####
def _replace_special_missing_values_with_null(df: pl.DataFrame) -> pl.DataFrame:
    columns = df.columns
    df_result = df.with_columns([
                                    pl.col(col).replace({-99: None, -9999: None})
                                    for col in columns
                                    if col != "lon"  # lon ranges from -180 to 180 and can be equal to -99
                                ] + [df["lon"].replace(-9999, None)])

    return df_result


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
            return "10.65"
        case "19V" | "19H":
            return "18.7"
        case "23V":
            return "23.8"
        case "37V" | "37H":
            return "36.64"
        case "89V" | "89H":
            return "89.0"
        case "165V" | "165H":
            return "166.0"
        case "183V3":
            return "183.31±3"
        case "183V7":
            return "183.31±7"
        case _:
            raise Exception(f"Unknown key {key}")


def _get_uncertainties_dict(tc_columns: List[str]) -> Dict[str, float]:
    gmi_characteristics = _get_gmi_characteristics()
    uncertainties = {
        col: gmi_characteristics.loc[_get_frequency(col.removeprefix("Tc_"))][_EXPECTED_TOTAL_UNCERTAINTY]
        for col in tc_columns
    }
    return uncertainties


def _round(df: pl.DataFrame, uncertainties_dict: Dict[str, float]) -> pl.DataFrame:
    tc_cols = _get_tc_columns(df.columns)

    df_result = df.with_columns([
        (pl.col(col) / uncertainties_dict[col]).round() * uncertainties_dict[col]
        for col in tc_cols
    ])

    return df_result


#### SEGMENTING INTO BINS ####
def _aggregate(df: pl.DataFrame) -> pl.DataFrame:
    columns = set(df.columns)

    tc_cols = _get_tc_columns(columns)
    flag_cols = _get_flag_columns(columns)
    periodic_cols = _get_periodic_columns(columns)
    special_cols = _get_special_columns(columns)
    other_cols = [col for col in columns if
                  col not in tc_cols + flag_cols + periodic_cols + special_cols + [COLUMN_TIME]]

    periodic_dict = _get_periodic_dict()
    special_dict = _get_special_dict()

    estimated_size = df.estimated_size() / (1024 * 1024)
    logging.info(f"Estimated dataframe size: {estimated_size:.2f} MB")

    memory_usage_before = get_memory_usage()

    df_result = (
        df
        .group_by(tc_cols)
        .agg(
            *[pl.col(flag_col).value_counts(name=STRUCT_FIELD_COUNT) for flag_col in flag_cols],

            *[expr
              for p_col in periodic_cols
              for expr in
              [pl.when(pl.col(p_col) <= periodic_dict[p_col]).then(pl.col(p_col)).mean().alias(f"{p_col}_lt"),
               pl.when(pl.col(p_col) <= periodic_dict[p_col]).then(pl.col(p_col)).count().alias(f"{p_col}_lt_count"),
               pl.when(pl.col(p_col) > periodic_dict[p_col]).then(pl.col(p_col)).mean().alias(f"{p_col}_gt"),
               pl.when(pl.col(p_col) > periodic_dict[p_col]).then(pl.col(p_col)).count().alias(f"{p_col}_gt_count")]],

            *[expr
              for s_col in special_cols
              for expr in [pl.col(s_col).filter(pl.col(s_col).ne(special_dict[s_col][0])).mean(),
                           pl.col(s_col).filter(pl.col(s_col).eq(special_dict[s_col][0])).count().alias(
                               f"{s_col}_{special_dict[s_col][1]}_count"),
                           pl.col(s_col).filter(
                               pl.col(s_col).ne(special_dict[s_col][0]) & pl.col(s_col).is_not_null()).count().alias(
                               f"{s_col}_count")]],

            *[expr
              for other_col in other_cols
              for expr in
              [pl.col(other_col).mean(), (pl.len() - pl.col(other_col).null_count()).alias(f"{other_col}_count")]],

            *[pl.col(COLUMN_TIME).min(), pl.len().alias(COLUMN_COUNT)],
        )
    )

    memory_usage_after = get_memory_usage()
    logging.info(f"Memory used by aggregation: {memory_usage_after - memory_usage_before:.2f} MB")

    return df_result


#### PREPROCESSING PIPELINE ####
def segment_features_into_bins(df: pl.DataFrame) -> pl.DataFrame:
    row_count_before = len(df)
    columns = df.columns

    # 0. Drop id columns.
    df = df.drop(_get_columns_to_drop(columns))

    # 1. Replace NaN, -99, and -9999 by nulls.
    df = df.fill_nan(None)
    df = _replace_special_missing_values_with_null(df)

    # 2. Extract ordinal day from time column.
    df = df.with_columns(
        pl.col(COLUMN_TIME).dt.ordinal_day().alias("day")
    )

    # 3. Round Tc values.
    df = _round(df, _get_uncertainties_dict(_get_tc_columns(columns)))

    # 4. Aggregate by rounded Tc values.
    df = _aggregate(df)

    row_count_after = len(df)
    logging.info(f"Segmented features into {row_count_after}/{row_count_before} bins")

    return df


#### MERGING PREPROCESSED DATA ####


def _get_struct_list_type(flag_column: str) -> pl.List:
    return pl.List(pl.Struct([
        pl.Field(flag_column, pl.Int16),
        pl.Field(STRUCT_FIELD_COUNT, pl.Int64)
    ]))


def _aggregate_structs(df: pl.DataFrame, flag_column: str) -> pl.DataFrame:
    df_result = (
        df
        .with_row_index()  # track original rows
        .explode(flag_column)  # explode the List[Struct]
        .unnest(flag_column)  # turns into columns 'k', 'v'
        .group_by(["index", flag_column])
        .agg(pl.col(STRUCT_FIELD_COUNT).sum())
        .with_columns(pl.struct([flag_column, STRUCT_FIELD_COUNT]).alias(f"{flag_column}_tmp"))  # build back struct
        .group_by("index")
        .agg(pl.col(f"{flag_column}_tmp"))
        .join(df.drop(flag_column).with_row_index(), on="index")
        .rename({f"{flag_column}_tmp": flag_column})
        .drop("index")
    )

    return df_result


# TODO: replace List with Iterable in other places
def merge_segmented_features(dfs: Iterable[pl.DataFrame]) -> pl.DataFrame:
    df = pl.concat(dfs)
    row_count_before = len(df)

    columns = [col.removesuffix("_lt").removesuffix("_gt") for col in df.columns if
               not col.endswith("count") and not col.endswith("_gt")]

    tc_cols = _get_tc_columns(columns)
    flag_cols = _get_flag_columns(columns)
    periodic_cols = _get_periodic_columns(columns)
    special_cols = _get_special_columns(columns)
    other_cols = [col for col in columns if
                  col not in tc_cols + flag_cols + periodic_cols + special_cols + [COLUMN_TIME]]

    special_dict = _get_special_dict()

    def aggregate_mean(col):
        return (pl.col(col) * pl.col(f"{col}_count")).sum() / pl.col(f"{col}_count").sum()

    estimated_size = df.estimated_size() / (1024 * 1024)
    logging.info(f"Estimated dataframe size: {estimated_size:.2f} MB")

    memory_usage_before = get_memory_usage()

    df_result = (
        df
        .group_by(tc_cols)
        .agg(*[pl.col(flag_col).flatten() for flag_col in flag_cols],
             *[expr
               for p_col in periodic_cols
               for expr in [aggregate_mean(f"{p_col}_lt"), pl.col(f"{p_col}_lt_count").sum(),
                            aggregate_mean(f"{p_col}_gt"), pl.col(f"{p_col}_gt_count").sum()]],
             *[expr
               for s_col in special_cols
               for expr in [aggregate_mean(s_col),
                            pl.col(f"{s_col}_{special_dict[s_col][1]}_count").sum(),
                            pl.col(f"{s_col}_count").sum()]],

             *[expr
               for other_col in other_cols
               for expr in [aggregate_mean(other_col),
                            pl.col(f"{other_col}_count").sum()]],
             pl.col(COLUMN_TIME).min(),
             pl.col(COLUMN_COUNT).sum(),
             )
    )

    memory_usage_after = get_memory_usage()
    logging.info(f"Memory used by aggregation: {memory_usage_after - memory_usage_before:.2f} MB")

    for flag_col in tqdm(flag_cols):
        df_result = _aggregate_structs(df_result, flag_col)

    row_count_after = len(df_result)
    logging.info(f"Segmented features into {row_count_after}/{row_count_before} bins")

    return df_result

