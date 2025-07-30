"""
This module provides utilities for working with data stored in Polars format.
"""
from typing import List, Sequence

import numpy as np
import polars as pl


def weighted_quantiles(value_count: pl.DataFrame, quantiles: List[float], value_col: str, count_col: str) \
        -> List[float]:
    """
    Calculate weighted quantiles from a value-count DataFrame.
    """
    value_count = value_count.sort(value_col).with_columns(
        pl.col(count_col).truediv(pl.col(count_col).cast(pl.UInt64).sum()).cum_sum().alias("cumprob"),
    )

    results = []

    for q in quantiles:
        upper_idx = value_count.with_row_index().filter(pl.col("cumprob") >= q)["index"][0]

        lower_idx = max(0, upper_idx - 1)
        upper_idx = lower_idx + 1

        x0 = value_count["cumprob"][lower_idx]
        x1 = value_count["cumprob"][upper_idx]
        y0 = value_count[value_col][lower_idx]
        y1 = value_count[value_col][upper_idx]

        interp_value = y0 + (q - x0) * (y1 - y0) / (x1 - x0)
        results.append(interp_value)

    return results


def get_column_ranges(df: pl.DataFrame, columns: Sequence[str] | None = None) -> np.ndarray:
    """
    Retrieve the minimum and maximum values for each column in a DataFrame and return them as a range.
    """
    columns = columns if columns is not None else df.columns
    return np.stack([
        df.select(pl.col(columns).min()).to_numpy().flatten(),
        df.select(pl.col(columns).max()).to_numpy().flatten()
    ], axis=1)


def take_k_sorted(df: pl.DataFrame, by: str | Sequence[str], k: int, count: str, descending: bool):
    """
    Take the top k rows from a sorted DataFrame based on cumulative count.
    """
    column_count_cumsum = f"{count}_cumsum"

    df = df.sort(by, descending=descending)
    df = df.with_columns(pl.col(count).cast(pl.UInt64).cum_sum().alias(column_count_cumsum))

    df_k = df.filter(pl.col(column_count_cumsum) <= k)
    return df_k
