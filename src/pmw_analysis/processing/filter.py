from typing import Sequence, Tuple, List

import polars as pl

from pmw_analysis.constants import COLUMN_SUFFIX_QUANT, COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX, STRUCT_FIELD_COUNT


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


def filter_by_surface_type(df, flag_value: int | List[int]) -> pl.DataFrame:
    """
    Filter values in data frame leaving only those with the specified surface type.
    """
    if isinstance(flag_value, int):
        flag_values = [flag_value]
    else:
        flag_values = set(flag_value)

    if isinstance(df[VARIABLE_SURFACE_TYPE_INDEX], pl.datatypes.List):
        df_result = df.with_columns(
            pl.col(VARIABLE_SURFACE_TYPE_INDEX).list.eval(
                pl.element().filter(pl.element().struct.field(VARIABLE_SURFACE_TYPE_INDEX).is_in(flag_values))
            ).list.first().struct.field(STRUCT_FIELD_COUNT).alias(COLUMN_COUNT)
        )
    else:
        df_result = df.filter(pl.col(VARIABLE_SURFACE_TYPE_INDEX).is_in(flag_values))
    return df_result
