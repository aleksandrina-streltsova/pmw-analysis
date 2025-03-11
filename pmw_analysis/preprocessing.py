"""
This module contains feature extraction from xarray DataSet to DataFrame.
"""
import warnings
from typing import List, Tuple, Optional

import gpm.bucket
import pandas as pd
import xarray as xr

from pmw_analysis.constants import TIME_COLUMN


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col != TIME_COLUMN]


# TODO: enable using custom bins and return calculated bins
def dataset_to_dataframe(ds: xr.Dataset) -> Tuple[pd.DataFrame, List[str]]:
    """
    Convert xarray Dataset to DataFrame leaving only features.
    """
    df = ds.reset_coords(names=TIME_COLUMN).reset_coords(drop=True).to_dataframe()
    # TODO: why does PD_165 have NaN? should rows with NaNs be processed differently?
    df = df.dropna()

    feature_cols = _get_feature_columns(df)
    for feature_col in feature_cols:
        # TODO: how many bins can we afford to keep?
        n_bins = 50
        df[feature_col] = pd.cut(df[feature_col], n_bins).astype(object).apply(lambda interval: interval.mid)

    df_agg = df.groupby(feature_cols).agg(
        count=(TIME_COLUMN, 'size'),
        first_occurrence=(TIME_COLUMN, 'min'),
    ).reset_index()

    return df_agg, feature_cols


def read_time_series_and_drop_nan(bucket_dir: str,
                                  point: Tuple[float, float],
                                  distance: float,
                                  name: Optional[str] = None,
                                  feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
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

    count_before_filtering = len(ts_pd)
    ts_pd = ts_pd.dropna(subset=feature_columns)
    count_after_filtering = len(ts_pd)
    print(f"Filtered {count_before_filtering - count_after_filtering}/{count_after_filtering} points")

    if name is not None:
        ts_pd.attrs["name"] = name

    return ts_pd
