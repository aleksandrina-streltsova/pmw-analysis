"""
Build the subset as in the subset part of examples/tmp/partial_dataset.py,
and count the observations read per box and surface type.
"""
import argparse
import pathlib

import gpm
import polars as pl

from pmw_analysis.constants import DIR_BUCKET, TC_COLUMNS, COLUMN_LON, COLUMN_LAT, COLUMN_TIME, COLUMN_COUNT, \
    VARIABLE_SURFACE_TYPE_INDEX, ST_COLUMNS, EXTENTS_SUBSET
from pmw_analysis.preprocessing.dataframe_polars import quantize_pmw_features, get_uncertainties_dict
from pmw_analysis.preprocessing.script import get_range_dict

DIR_OUT = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b1")


def main(dir_out: pathlib.Path, extents: dict):
    dir_out.mkdir(exist_ok=True)
    dfs_partial = []
    dfs_composition = []
    for box, extent in extents.items():
        df_partial: pl.DataFrame = gpm.bucket.read(bucket_dir=DIR_BUCKET,
                                                   columns=[COLUMN_TIME, COLUMN_LON, COLUMN_LAT, VARIABLE_SURFACE_TYPE_INDEX] + TC_COLUMNS,
                                                   extent=extent)
        dfs_partial.append(df_partial)
        dfs_composition.append(
            df_partial.group_by(VARIABLE_SURFACE_TYPE_INDEX).len("n_obs").with_columns(pl.lit(box).alias("box"))
        )
        print(f"{box}: {len(df_partial)} observations")
    df_partial = pl.concat(dfs_partial)

    df_st = pl.DataFrame({VARIABLE_SURFACE_TYPE_INDEX: range(1, len(ST_COLUMNS) + 1), "surface_type": ST_COLUMNS})
    df_composition = pl.concat(dfs_composition)
    df_st = df_st.cast({VARIABLE_SURFACE_TYPE_INDEX: df_composition.schema[VARIABLE_SURFACE_TYPE_INDEX]})
    df_composition = (
        df_composition.join(df_st, on=VARIABLE_SURFACE_TYPE_INDEX, how="left")
        .select("box", VARIABLE_SURFACE_TYPE_INDEX, "surface_type", "n_obs")
        .sort("box", VARIABLE_SURFACE_TYPE_INDEX)
    )
    df_composition.write_csv(dir_out / "composition.csv")

    unc_dict = {col: 1.0 * unc for col, unc in get_uncertainties_dict(TC_COLUMNS).items()}
    df_partial = quantize_pmw_features(df_partial, TC_COLUMNS, unc_dict, get_range_dict(False))

    for freq in [10, 19, 37, 89, 165]:
        df_partial = df_partial.with_columns(pl.col(f"Tc_{freq}V").sub(pl.col(f"Tc_{freq}H")).alias(f"PD_{freq}"),
                                             pl.col(f"Tc_{freq}V").truediv(pl.col(f"Tc_{freq}H")).alias(f"ratio_{freq}"))

    tc_denom = "Tc_19H"
    for tc_col in TC_COLUMNS:
        if tc_col == tc_denom:
            continue
        df_partial = df_partial.with_columns(pl.col(tc_col).truediv(pl.col(tc_denom)).alias(f"{tc_col}_{tc_denom}"))

    df_path = dir_out / "partial" / "final.parquet"
    df_path.parent.mkdir(exist_ok=True)
    df_partial.write_parquet(df_path)
    print(f"subset: {len(df_partial)} signatures, {df_partial[COLUMN_COUNT].sum()} observations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the subset from EXTENTS_SUBSET")
    parser.add_argument("--spring-boxes", action="store_true",
                        help="Use only the first four (spring 2025) boxes and write to b1/ instead of b1/eight_boxes/")
    args = parser.parse_args()

    if args.spring_boxes:
        main(DIR_OUT, dict(list(EXTENTS_SUBSET.items())[:4]))
    else:
        main(DIR_OUT / "eight_boxes", EXTENTS_SUBSET)
