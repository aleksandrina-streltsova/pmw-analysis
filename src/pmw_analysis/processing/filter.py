from typing import Sequence, Tuple, List, Any

import numpy as np
import polars as pl

from pmw_analysis.constants import COLUMN_SUFFIX_QUANT, COLUMN_COUNT, STRUCT_FIELD_COUNT


def filter_by_signature_occurrences_count(df: pl.DataFrame,
                                          m_occurrences: int,
                                          quant_columns: Sequence[str],
                                          ) -> pl.DataFrame:
    """
    Filter the input DataFrame by occurrences of signatures.
    Calculate the number of occurrences of each unique combination of specified quant columns and
    filter based on a minimum occurrence threshold.
    """
    quant_columns_suffixed = [f"{col}{COLUMN_SUFFIX_QUANT}" for col in quant_columns]

    df_quant_m = df.select(quant_columns_suffixed).group_by(quant_columns_suffixed).agg(pl.len().alias(COLUMN_COUNT))
    df_quant_m = df_quant_m.filter(pl.col(COLUMN_COUNT) >= m_occurrences)

    df_m = df.join(df_quant_m, on=quant_columns_suffixed, how="inner")

    return df_m


def filter_by_flag_values(df, flag_column: str, flag_value: Any | List[Any],
                          filter_out: bool = False, nulls_equal: bool = False) -> pl.DataFrame:
    """
    Filter rows in data frame leaving only those with the specified flag value.
    """
    if isinstance(flag_value, List):
        flag_values = set(flag_value)
    else:
        flag_values = [flag_value]

    if df[flag_column].dtype == pl.List:
        filter_expr = pl.element().struct.field(flag_column).is_in(flag_values, nulls_equal=nulls_equal)
        if filter_out:
            filter_expr = filter_expr.not_()

        df_result = df.with_columns(
            pl.col(flag_column).list.eval(
                pl.element().filter(filter_expr)
            ).list.first().struct.field(STRUCT_FIELD_COUNT).alias(COLUMN_COUNT)
        ).filter(pl.col(COLUMN_COUNT) > 0)
    else:
        filter_expr = pl.col(flag_column).is_in(flag_values, nulls_equal=nulls_equal)
        if filter_out:
            filter_expr = filter_expr.not_()

        df_result = df.filter(filter_expr)
    return df_result


def filter_by_value_range(df, column: str | list[str], value_range: Tuple | list[Tuple] | np.ndarray,
                          filter_out: bool = False) -> pl.DataFrame:
    """
    Filter rows in data frame leaving only those with values within the specified range.
    If multiple columns are specified, the ranges are applied to each column individually.

    Parameters
    ----------
    filter_out : bool, optional
        If True, filter out rows that satisfy the condition, by default, False
    """
    filter_expr = get_filter_expr_from_value_range(column, value_range, filter_out)

    return df.filter(filter_expr)


def get_filter_expr_from_value_range(column: str | list[str], value_range: Tuple | list[Tuple] | np.ndarray,
                                     filter_out: bool = False) -> pl.Expr:
    if isinstance(column, str):
        column = [column]
        value_range = [value_range]

    filter_expr = pl.lit(True)

    for col, rng in zip(column, value_range):
        filter_expr = filter_expr.and_(pl.col(col).is_between(*rng, closed="left"))

    if filter_out:
        filter_expr = filter_expr.not_()

    return filter_expr