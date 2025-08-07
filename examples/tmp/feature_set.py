"""
Script for running KMeans++ on quantized transformed data.
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from feature_engine.selection import DropCorrelatedFeatures

from pmw_analysis.constants import COLUMN_COUNT, TC_COLUMNS, DIR_PMW_ANALYSIS, VARIABLE_SURFACE_TYPE_INDEX

def main():
    df_path = pathlib.Path(DIR_PMW_ANALYSIS) / "partial" / "final.parquet"
    df_merged: pl.DataFrame = pl.read_parquet(df_path)

    feature_columns = [
        "Tc_19V", "PD_19",
        "Tc_23V",
        "Tc_37V", "PD_37",
        "Tc_89V", "PD_89",
    ]
    tc_denom = "Tc_19H"
    feature_columns += [f"ratio_{freq}" for freq in [19, 37, 89]]
    # feature_columns += [f"ratio_{freq}" for freq in [10, 19, 37, 89, 165]]
    feature_columns += [f"{tc_col}_{tc_denom}" for tc_col in TC_COLUMNS
                        if tc_col != tc_denom and "10" not in tc_col and "165" not in tc_col and "183" not in tc_col]

    fig, ax = plt.subplots(figsize=(12, 10))
    df = df_merged[feature_columns + [COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX]]
    df = df.drop_nans()

    corr_mtx = df[feature_columns].corr()
    corr_mtx = corr_mtx.select([pl.col(col).abs() for col in corr_mtx.columns])
    sns.heatmap(corr_mtx, annot=True, ax=ax, xticklabels=feature_columns, yticklabels=feature_columns)
    plt.title("Correlation matrix", fontsize=20)
    plt.tight_layout()
    plt.savefig(pathlib.Path("images") / "corr_mtx.png")
    plt.show()

    corr_mtx_pd = corr_mtx.to_pandas()
    upper = corr_mtx_pd.where(np.triu(np.ones(corr_mtx_pd.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    df_filtered1 = df.drop(to_drop)
    print(df_filtered1.columns)

    drop_corr = DropCorrelatedFeatures(threshold=0.95)
    df_filtered2 = drop_corr.fit_transform(df[feature_columns].to_pandas())
    print(df_filtered2.columns)


if __name__ == "__main__":
    main()
