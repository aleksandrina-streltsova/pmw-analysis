from typing import Sequence, Tuple

import polars as pl

from pmw_analysis.constants import COLUMN_SUFFIX_QUANT, COLUMN_COUNT


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