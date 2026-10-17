import polars as pl
import gpm.bucket

from pmw_analysis.constants import DIR_BUCKET, TC_COLUMNS, COLUMN_LON, COLUMN_LAT, COLUMN_TIME
from pmw_analysis.quantization.dataframe_polars import get_uncertainties_dict, _round


def main():
    extent = [-180, 180, -90, 90]
    df = gpm.bucket.read(bucket_dir=DIR_BUCKET,
                         extent=extent,
                         columns=TC_COLUMNS + [COLUMN_LON, COLUMN_LAT, COLUMN_TIME],
                         backend="polars_lazy")
    quant_columns = TC_COLUMNS
    uncertainty_dict = get_uncertainties_dict(TC_COLUMNS)
    quant_steps = [10 * uncertainty_dict[col] for col in quant_columns]

    df_rounded: pl.LazyFrame = _round(df.select(TC_COLUMNS), quant_columns, quant_steps, quant_ranges=None)

    approx_n = df_rounded.select(pl.struct(quant_columns).hash().approx_n_unique()).collect()
    print(f"Approximately {approx_n} unique signatures.")


if __name__ == '__main__':
    main()