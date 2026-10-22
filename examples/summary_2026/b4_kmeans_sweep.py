"""
KMeans++ on the B2 dataset for several numbers of clusters and weightings, scored with B-Cubed on the eight boxes.
Each weighting is applied to the scaler, the PCA and KMeans, as count weighting is in train_spring.py.
"""
import pathlib
import sys
import time

import gpm
import gpm.bucket
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from b4_diagnostics import bcubed, ST_GROUPS  # noqa: E402
from train_spring import PATH_DATASET, load_signatures  # noqa: E402

from pmw_analysis.analysis.clustering import ClusterModel, _pca_fit_transform  # noqa: E402
from pmw_analysis.constants import DIR_BUCKET, EXTENTS_SUBSET, TC_COLUMNS, VARIABLE_SURFACE_TYPE_INDEX, ST_COLUMNS, \
    COLUMN_LON, COLUMN_LAT, ArgTransform  # noqa: E402
from pmw_analysis.preprocessing.dataframe_polars import _replace_special_missing_values_with_null  # noqa: E402
from pmw_analysis.preprocessing.transforms import get_transformation_function  # noqa: E402
from quantization.constants import COLUMN_COUNT  # noqa: E402

DIR_OUT = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b4/kmeans_sweep")
N_CLUSTERS = [4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40]
WEIGHTINGS = {
    "count": lambda count: count.cast(pl.Float64),
    "log_count": lambda count: count.cast(pl.Float64).log1p(),
    "none": lambda count: None,
}


def read_boxes(transform, feature_columns: list[str]) -> pl.DataFrame:
    """
    The eight boxes as metric.py reads them: nulls for missing values, the transform, and a mask of null features.
    """
    dfs = [gpm.bucket.read(bucket_dir=DIR_BUCKET, columns=TC_COLUMNS + [COLUMN_LON, COLUMN_LAT, VARIABLE_SURFACE_TYPE_INDEX],
                           extent=extent).with_columns(pl.lit(idx).alias("box"))
           for idx, extent in enumerate(EXTENTS_SUBSET.values())]
    df = transform(_replace_special_missing_values_with_null(pl.concat(dfs).fill_nan(None)))
    st_to_group = {float(idx + 1): group for group, classes in ST_GROUPS.items()
                   for idx, st in enumerate(ST_COLUMNS) if st in classes}
    return df.select(feature_columns + ["box", VARIABLE_SURFACE_TYPE_INDEX]).with_columns(
        pl.any_horizontal(pl.col(feature_columns).is_null()).alias("has_null"),
        pl.col(VARIABLE_SURFACE_TYPE_INDEX).replace_strict(st_to_group, return_dtype=pl.String).alias("st_group"),
    )


def score(df_boxes: pl.DataFrame, model: ClusterModel, feature_columns: list[str]) -> dict:
    # As metric.py: rows with a null feature get cluster -1
    labels = -1 * np.ones(df_boxes.height, dtype=int)
    has_null = df_boxes["has_null"].to_numpy()
    labels[~has_null] = model.predict(df_boxes.filter(~pl.col("has_null")).select(feature_columns))
    df = df_boxes.with_columns(pl.Series("cluster", labels))
    p, r, f1 = bcubed(df, "cluster", VARIABLE_SURFACE_TYPE_INDEX)
    f1_box = np.mean([bcubed(df_box, "cluster", VARIABLE_SURFACE_TYPE_INDEX)[2]
                      for _, df_box in df.group_by("box")])
    return {"P": p, "R": r, "F1": f1, "F1_groups": bcubed(df, "cluster", "st_group")[2], "F1_box_mean": f1_box,
            "largest_cluster_share": df["cluster"].value_counts()["count"].max() / df.height}


def main():
    DIR_OUT.mkdir(exist_ok=True)
    transform = get_transformation_function(ArgTransform.V5)
    feature_columns = transform(TC_COLUMNS)
    df = load_signatures(PATH_DATASET, feature_columns)
    df_boxes = read_boxes(transform, feature_columns)

    rows = []
    for weighting, get_weight in WEIGHTINGS.items():
        weight = get_weight(df[COLUMN_COUNT])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[feature_columns], sample_weight=weight)
        features_reduced, reducer = _pca_fit_transform(features_scaled, weight, n_components=None)
        for n_clusters in N_CLUSTERS:
            start = time.time()
            clusterer = KMeans(n_clusters=n_clusters, random_state=0)
            clusterer.fit(features_reduced, sample_weight=weight)
            row = {"weighting": weighting, "n_clusters": n_clusters,
                   **score(df_boxes, ClusterModel(scaler, reducer, clusterer), feature_columns),
                   "seconds": time.time() - start}
            rows.append(row)
            print(row, flush=True)

    results = pl.DataFrame(rows)
    results.write_csv(DIR_OUT / "results.csv")
    with pl.Config(tbl_rows=50, tbl_cols=-1, tbl_width_chars=200):
        print(results)


if __name__ == "__main__":
    main()
