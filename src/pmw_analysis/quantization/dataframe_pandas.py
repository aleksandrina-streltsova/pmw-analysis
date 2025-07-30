"""
This module contains feature extraction from xarray DataSet to DataFrame.
"""
import warnings
from typing import List, Tuple

import gpm.bucket
import pandas as pd

from pmw_analysis.constants import COLUMN_TIME


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col != COLUMN_TIME]


# TODO: enable using custom bins and return calculated bins
def segment_features_into_bins(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Use `pandas.qcut` to discretize features and retain only unique rows.
    """
    feature_cols = _get_feature_columns(df)
    df = remove_missing_values(df, subset=feature_cols)

    for feature_col in feature_cols:
        # TODO: how many bins can we afford to keep?
        n_bins = 30
        segmented, _ = pd.qcut(df[feature_col], n_bins, retbins=True)
        df[feature_col] = segmented.astype(object).apply(lambda interval: interval.mid)

    df_agg = df.groupby(feature_cols).agg(
        count=(COLUMN_TIME, 'size'),
        first_occurrence=(COLUMN_TIME, 'min'),
    ).reset_index()

    return df_agg, feature_cols


def read_time_series_and_drop_nan(bucket_dir: str,
                                  point: Tuple[float, float],
                                  distance: float,
                                  name: str | None = None,
                                  feature_columns: List[str] | None = None) -> pd.DataFrame:
    """
    Read a geographic bucket and remove missing values.
    """
    ts_pl = gpm.bucket.read(bucket_dir=bucket_dir,
                            point=point,
                            distance=distance,
                            parallel="auto",  # "row_groups", "columns"
                            )

    ts_pd: pd.DataFrame = ts_pl.to_pandas().copy().sort_values(by="time")
    if feature_columns is not None:
        # TODO: should rows with NaNs be processed differently?
        missing_columns = set(feature_columns).difference(ts_pd.columns)
        if len(missing_columns) > 0:
            warnings.warn(f"The following columns were not found in the dataframe: {missing_columns}."
                          f" Available columns are: {ts_pd.columns}.")
            feature_columns = [col for col in feature_columns if col not in missing_columns]

    ts_pd = remove_missing_values(ts_pd, subset=feature_columns)

    if name is not None:
        ts_pd.attrs["name"] = name

    return ts_pd


def remove_missing_values(df: pd.DataFrame, subset: List[str] | None = None) -> pd.DataFrame:
    """
    Remove missing values from a DataFrame and print the number of rows removed.
    """
    count_before_filtering = len(df)
    df = df.dropna(subset=subset)
    count_after_filtering = len(df)
    print(f"Filtered {count_before_filtering - count_after_filtering}/{count_before_filtering} points")

    return df
