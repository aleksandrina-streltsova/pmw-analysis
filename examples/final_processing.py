import pathlib

import polars as pl

from pmw_analysis.constants import PMW_ANALYSIS_DIR
from pmw_analysis.quantization.dataframe_polars import create_occurrence_column
from pmw_analysis.retrievals.retrieval_1b_c_pmw import retrieve_possible_sun_glint


def main():
    df_k = pl.read_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "final_k.parquet")
    df_k = create_occurrence_column(df_k)
    df_k, sun_glint_column = retrieve_possible_sun_glint(df_k)
    df_k.write_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "final_k.parquet")

    dir_no_sun_glint = pathlib.Path(PMW_ANALYSIS_DIR) / "no_sun_glint"
    dir_no_sun_glint.mkdir(parents=True, exist_ok=True)
    df_k_no_sun_glint = df_k.filter(~pl.col(sun_glint_column)).drop(sun_glint_column)
    df_k_no_sun_glint.write_parquet(dir_no_sun_glint / "final_k.parquet")


if __name__ == '__main__':
    main()