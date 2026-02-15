import pathlib
import gpm.bucket
from pmw_analysis.constants import COLUMN_TIME, COLUMN_LON, COLUMN_LAT, COLUMN_GPM_ID, COLUMN_GPM_CROSS_TRACK_ID, TC_COLUMNS

def main():
    bucket_dpr_dry_surface = pathlib.Path("/ltenas8/data/GPM_Buckets/DPR_DrySurface")
    bucket_dpr_dry_surface = pathlib.Path("/ltenas8/data/GPM_Buckets/DPR_RainySurface")

    point = (-29.442882537841797, -60.98043441772461)
    DISTANCE = 10000

    ts = gpm.bucket.read(bucket_dir=bucket_dpr_dry_surface, point=point, distance=DISTANCE, backend="polars",
                         columns=[COLUMN_LON, COLUMN_LAT, COLUMN_TIME, "Ka_Band"])


if __name__ == '__main__':
    main()