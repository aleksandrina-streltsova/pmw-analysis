"""
Count, per evaluation box, the rows metric.py reads, the rows with missing or special-missing surfaceTypeIndex
in the raw bucket data, and the rows metric.py labels -1 (a null v5 feature after its null conversion).
"""
import gpm
import gpm.bucket
import polars as pl

from pmw_analysis.constants import DIR_BUCKET, EXTENTS_SUBSET, TC_COLUMNS, VARIABLE_SURFACE_TYPE_INDEX, ST_COLUMNS, \
    ArgTransform, COLUMN_LON, COLUMN_LAT
from pmw_analysis.preprocessing.dataframe_polars import _replace_special_missing_values_with_null
from pmw_analysis.preprocessing.transforms import get_transformation_function


def main():
    transform = get_transformation_function(ArgTransform.V5)
    feature_columns = transform(TC_COLUMNS)
    st = pl.col(VARIABLE_SURFACE_TYPE_INDEX)
    rows = []
    for idx, (box, extent) in enumerate(EXTENTS_SUBSET.items()):
        df = gpm.bucket.read(bucket_dir=DIR_BUCKET, columns=TC_COLUMNS + [COLUMN_LON, COLUMN_LAT, VARIABLE_SURFACE_TYPE_INDEX],
                             extent=extent)
        # As metric.py: NaN and special missing values to null, then the v5 transform
        df_features = transform(_replace_special_missing_values_with_null(df.fill_nan(None)))
        rows.append(df.select(
            pl.lit(idx).alias("box"),
            pl.lit(box).alias("name"),
            pl.len().alias("rows"),
            st.is_null().sum().alias("st_null"),
            st.is_nan().sum().alias("st_nan"),
            (st == -99).sum().alias("st_minus_99"),
            (st == -9999).sum().alias("st_minus_9999"),
            (~st.is_in([float(i) for i in range(1, len(ST_COLUMNS) + 1)] + [-99.0, -9999.0]) & st.is_not_nan()).sum()
            .alias("st_other"),
        ).with_columns(
            pl.lit(df_features.select(pl.any_horizontal(pl.col(feature_columns).is_null()).sum()).item())
            .alias("null_feature"),
        ))
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(pl.concat(rows))


if __name__ == "__main__":
    main()
