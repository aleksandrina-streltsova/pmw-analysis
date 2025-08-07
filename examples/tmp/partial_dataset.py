import pathlib
from copy import copy

import gpm
from gpm.bucket import LonLatPartitioning
import matplotlib.pyplot as plt
import polars as pl
import numpy as np

from pmw_analysis.constants import DIR_BUCKET, TC_COLUMNS, COLUMN_LON, COLUMN_LAT, DIR_PMW_ANALYSIS, COLUMN_COUNT, \
    COLUMN_TIME, VARIABLE_SURFACE_TYPE_INDEX, ArgTransform
from pmw_analysis.quantization.dataframe_polars import quantize_pmw_features, get_uncertainties_dict
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.polars import weighted_quantiles


def main():
    extents = [
        [42, 44, 39, 41], # Sochi, Russia
        [59, 61, 4, 6], # Bergen, Norway
        [17, 19, -17, -15], # Nouakchott, Mauritania
        [59, 61, -46, -44], # Nanortalik, Greenland
    ]
    dfs_partial = []
    for extent in extents:
        df_partial: pl.DataFrame = gpm.bucket.read(bucket_dir=DIR_BUCKET,
                                                   columns=[COLUMN_TIME, COLUMN_LON, COLUMN_LAT, VARIABLE_SURFACE_TYPE_INDEX] + TC_COLUMNS,
                                                   extent=extent)
        dfs_partial.append(df_partial)
    df_partial = pl.concat(dfs_partial)

    unc_dict = {col: 1.0 * unc for col, unc in get_uncertainties_dict(TC_COLUMNS).items()}
    df_partial = quantize_pmw_features(df_partial, unc_dict, TC_COLUMNS)
    df_default = pl.read_parquet(pathlib.Path(DIR_PMW_ANALYSIS) / "default" / "final.parquet")

    feature_columns = copy(TC_COLUMNS)
    dfs = [df_default, df_partial]
    for freq in [10, 19, 37, 89, 165]:
        pd_col = f"PD_{freq}"
        ratio_col = f"ratio_{freq}"
        for i, df in enumerate(dfs):
            dfs[i] = df.with_columns(pl.col(f"Tc_{freq}V").sub(pl.col(f"Tc_{freq}H")).alias(pd_col),
                                     pl.col(f"Tc_{freq}V").truediv(pl.col(f"Tc_{freq}H")).alias(ratio_col))
        feature_columns += [pd_col, ratio_col]

    tc_denom = "Tc_19H"
    for tc_col in TC_COLUMNS:
        if tc_col == tc_denom:
            continue

        ratio_col = f"{tc_col}_{tc_denom}"
        for i, df in enumerate(dfs):
            dfs[i] = df.with_columns(pl.col(tc_col).truediv(pl.col(tc_denom)).alias(ratio_col))
        feature_columns.append(ratio_col)

    df_path = pathlib.Path(DIR_PMW_ANALYSIS) / "partial" / "final.parquet"
    df_path.parent.mkdir(exist_ok=True)
    dfs[1].write_parquet(df_path)

    transform_arg = ArgTransform.V2
    transform = get_transformation_function(transform_arg)
    df_path = pathlib.Path(DIR_PMW_ANALYSIS) / transform_arg / "final.parquet"
    df_transformed = pl.read_parquet(df_path)

    feature_columns = transform(TC_COLUMNS)
    df = df_transformed
    dfs = [df]
    # Histograms
    for tc_col in feature_columns:
        n_cols = 2
        n_rows = len(dfs)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        x_min = None
        x_max = None

        for i, y_scale in enumerate(["linear", "log"]):
            for ax, df, title in zip([axes[i]], dfs, [transform_arg]):
                value_count = df[[tc_col, COLUMN_COUNT]].group_by(tc_col).sum()
                value_count = value_count.filter(pl.col(tc_col).is_not_null())
                bins = len(value_count)

                if x_min is None:
                    x_min = value_count[tc_col].min()
                    x_max = value_count[tc_col].max()

                # q_min, q_max = weighted_quantiles(value_count, [1e-6, 1 - 1e-6], tc_col, COLUMN_COUNT)

                ax.hist(value_count[tc_col], weights=value_count[COLUMN_COUNT], bins=bins)
                # ax.set_xlabel(tc_col)
                # ax.set_ylabel(COLUMN_COUNT)
                ax.set_yscale(y_scale)
                ax.set_title(y_scale)
                # ax.set_title(f"{tc_col} {title}")
                ax.set_xlim(x_min, x_max)
                # y_min = 0
                # y_max = value_count[COLUMN_COUNT].max()
                # ax.vlines([q_min, q_max], ymin=y_min, ymax=y_max, colors="r")
            fig.suptitle(_fix_column(tc_col))
            fig.supxlabel(_get_xlabel(tc_col))
            fig.supylabel(COLUMN_COUNT)
            fig.tight_layout()
            fig.savefig(pathlib.Path("images") / f"{transform_arg}_hist_{tc_col}.png")
            fig.show()


def _fix_column(col: str):
    if col.find("Tc_") != col.rfind("Tc_"):
        i = col.rfind("Tc_")
        return col[:i-1] + "/" + col[i:]
    return  col


def _get_xlabel(col: str):
    if "PD" in col:
        return "Polarization Difference [K]"
    elif col.find("Tc_") != col.rfind("Tc_"):
        return "Temperature Brightness Ratio [1]"
    return "Temperature Brightness [K]"


if __name__ == '__main__':
    main()