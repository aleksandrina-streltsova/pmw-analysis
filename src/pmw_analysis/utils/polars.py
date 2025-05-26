from typing import List

import polars as pl


def weighted_quantiles(value_count: pl.DataFrame, quantiles: List[float], value_col: str, count_col: str) \
        -> List[float]:
    """
    Calculate weighted quantiles from a value-count DataFrame.
    """
    value_count = value_count.sort(value_col).with_columns(
        pl.col(count_col).cum_sum().truediv(pl.col(count_col).sum()).alias("cumprob"),
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
