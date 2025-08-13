from typing import Sequence, Tuple, List

import polars as pl

from pmw_analysis.constants import COLUMN_SUFFIX_QUANT, COLUMN_COUNT, STRUCT_FIELD_COUNT


def filter_by_signature_occurrences_count(df: pl.DataFrame,
                                          m_occurrences: int,
                                          quant_columns: Sequence[str],
                                          ) -> Tuple[pl.DataFrame, Sequence[str]]:
    """
    Filter the input DataFrame by occurrences of signatures.
    Calculate the number of occurrences of each unique combination of specified quant columns and
    filter based on a minimum occurrence threshold.
    """
    quant_columns_suffixed = [f"{col}{COLUMN_SUFFIX_QUANT}" for col in quant_columns]

    df_quant_m = df.select(quant_columns_suffixed).group_by(quant_columns_suffixed).agg(pl.len().alias(COLUMN_COUNT))
    df_quant_m = df_quant_m.filter(pl.col(COLUMN_COUNT) >= m_occurrences)

    return df_quant_m, quant_columns_suffixed


def filter_by_flag_values(df, flag_column: str, flag_value: int | List[int], filter_out: bool = False) -> pl.DataFrame:
    """
    Filter rows in data frame leaving only those with the specified flag value.
    """
    if isinstance(flag_value, int):
        flag_values = [flag_value]
    else:
        flag_values = set(flag_value)

    if df[flag_column].dtype == pl.List:
        filter_expr = pl.element().struct.field(flag_column).is_in(flag_values)
        if filter_out:
            filter_expr = filter_expr.not_()

        df_result = df.with_columns(
            pl.col(flag_column).list.eval(
                pl.element().filter(filter_expr)
            ).list.first().struct.field(STRUCT_FIELD_COUNT).alias(COLUMN_COUNT)
        ).filter(pl.col(COLUMN_COUNT) > 0)
    else:
        filter_expr = pl.col(flag_column).is_in(flag_values)
        if filter_out:
            filter_expr = filter_expr.not_()

        df_result = df.filter(filter_expr)
    return df_result


def filter_by_value_range(df, column: str, value_range: Tuple, filter_out: bool = False):
    """
    Filter rows in data frame leaving only those with values within the specified range.
    """
    filter_expr = pl.col(column).is_between(*value_range, closed="left")
    if filter_out:
        filter_expr = filter_expr.not_()

    return df.filter(filter_expr)
