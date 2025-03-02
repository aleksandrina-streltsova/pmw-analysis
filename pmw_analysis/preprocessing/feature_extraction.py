"""This module contains feature extraction from xarray DataSet to DataFrame."""
from typing import List, Tuple

import pandas as pd
import xarray as xr

_TIME_COLUMN = "time"


def _get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col != _TIME_COLUMN]


# TODO: enable using custom bins and return calculated bins
def dataset_to_dataframe(ds: xr.Dataset) -> Tuple[pd.DataFrame, List[str]]:
    """Convert xarray Dataset to DataFrame leaving only features."""
    df = ds.reset_coords(names=_TIME_COLUMN).reset_coords(drop=True).to_dataframe()
    # TODO: why does PD_165 have NaN? should rows with NaNs be processed differently?
    df = df.dropna()

    feature_cols = _get_feature_columns(df)
    for feature_col in feature_cols:
        # TODO: how many bins can we afford to keep?
        n_bins = 50
        df[feature_col] = pd.cut(df[feature_col], n_bins).astype(object).apply(lambda interval: interval.mid)

    df_agg = df.groupby(feature_cols).agg(
        count=(_TIME_COLUMN, 'size'),
        first_occurrence=(_TIME_COLUMN, 'min'),
    ).reset_index()

    return df_agg, feature_cols
