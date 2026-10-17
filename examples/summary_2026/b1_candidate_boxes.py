"""
Count observations per surfaceTypeIndex in candidate boxes for extending the four-box subset.
"""
import pathlib

import gpm
import gpm.bucket
import polars as pl

from pmw_analysis.constants import DIR_BUCKET, COLUMN_LON, COLUMN_LAT, VARIABLE_SURFACE_TYPE_INDEX, ST_COLUMNS

DIR_OUT = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b1")

# [lon_min, lon_max, lat_min, lat_max]; all within |lat| <= 70, like the full dataset
CANDIDATES = {
    "Congo Basin": [21, 23, -2, 0],
    "Amazon (Tefe)": [-66, -64, -6, -4],
    "Zambia (miombo)": [26, 28, -14, -12],
    "Kansas (Great Plains)": [-100, -98, 38, 40],
    "Ukraine (steppe)": [34, 36, 48, 50],
    "Sahel (Niger)": [2, 4, 14, 16],
    "Central Australia": [132, 134, -24, -22],
    "Saskatchewan": [-106, -104, 52, 54],
    "Yakutia": [128, 130, 62, 64],
    "Greenland ice sheet": [-44, -42, 64, 66],
    "Tibetan Plateau": [88, 90, 32, 34],
    "Hudson Bay": [-86, -84, 58, 60],
    "Nepal Himalaya": [84, 86, 27, 29],
}


def main():
    dfs = []
    for box, extent in CANDIDATES.items():
        df = gpm.bucket.read(bucket_dir=DIR_BUCKET, columns=[COLUMN_LON, COLUMN_LAT, VARIABLE_SURFACE_TYPE_INDEX],
                             extent=extent)
        dfs.append(df.group_by(VARIABLE_SURFACE_TYPE_INDEX).len("n_obs").with_columns(pl.lit(box).alias("box")))
        print(f"{box}: {len(df)} observations")
    df_counts = pl.concat(dfs)

    df_st = pl.DataFrame({VARIABLE_SURFACE_TYPE_INDEX: range(1, len(ST_COLUMNS) + 1), "surface_type": ST_COLUMNS})
    df_st = df_st.cast({VARIABLE_SURFACE_TYPE_INDEX: df_counts.schema[VARIABLE_SURFACE_TYPE_INDEX]})
    df_counts = (
        df_counts.join(df_st, on=VARIABLE_SURFACE_TYPE_INDEX, how="left")
        .select("box", VARIABLE_SURFACE_TYPE_INDEX, "surface_type", "n_obs")
        .sort("box", VARIABLE_SURFACE_TYPE_INDEX)
    )
    df_counts.write_csv(DIR_OUT / "candidate_boxes_composition.csv")


if __name__ == "__main__":
    main()
