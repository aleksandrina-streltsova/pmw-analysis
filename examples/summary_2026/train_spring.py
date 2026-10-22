"""
Train PCA + KMeans++ and UMAP + HDBSCAN on the B2 dataset with the spring settings (clusterize() at 404f236).
Run with CUDA_VISIBLE_DEVICES="": otherwise _umap_fit_transform uses TorchDR, which ignores the kNN graph.
"""
import logging
import pathlib

import hdbscan
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pmw_analysis.analysis.clustering import ClusterModel, CLusterIndexModel, _pca_fit_transform, \
    _umap_fit_transform
from pmw_analysis.constants import TC_COLUMNS, ArgTransform
from pmw_analysis.preprocessing.transforms import get_transformation_function
from pmw_analysis.utils.logging import timing
from quantization.constants import COLUMN_COUNT, COLUMN_SUFFIX_QUANT

PATH_DATASET = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b2/factor2/new/v5/all/final.parquet")
DIR_OUT = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b3")
COLUMN_CUM_PROB = "cum_prob"
RARE_SHARE = 0.05


def load_signatures(path: pathlib.Path, feature_columns: list[str]) -> pl.DataFrame:
    """
    Read the features and counts of the signatures, sorted by count, with the cumulative share of observations.
    """
    df = pl.read_parquet(path, columns=[f"{col}{COLUMN_SUFFIX_QUANT}" for col in feature_columns] + [COLUMN_COUNT])
    # v5's column names, so that the models accept the v5 transform of raw observations
    df = df.rename({f"{col}{COLUMN_SUFFIX_QUANT}": col for col in feature_columns})

    n_signatures = df.height
    # As spring's drop_nans, which under polars 1.24 also dropped nulls
    df = df.drop_nulls(feature_columns)
    logging.info("%d/%d signatures without null features", df.height, n_signatures)

    # 64-bit, as the sum of the UInt32 counts overflows
    count = pl.col(COLUMN_COUNT).cast(pl.UInt64)
    df = df.sort(COLUMN_COUNT)
    return df.with_columns((count.cum_sum() / count.sum()).alias(COLUMN_CUM_PROB))


def train_pca_kmeans(df: pl.DataFrame, feature_columns: list[str], path_model: pathlib.Path):
    """
    Weighted PCA (all components) and KMeans++ (k = 4), weighted by count, on all signatures.
    """
    weight = df[COLUMN_COUNT].cast(pl.Float64)
    with timing("Scaling features (PCA + KMeans++)"):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df[feature_columns], sample_weight=weight)
    with timing("Weighted PCA"):
        features_reduced, reducer = _pca_fit_transform(features_scaled, weight, n_components=None)
    with timing("KMeans++"):
        clusterer = KMeans(n_clusters=4)
        clusterer.fit(features_reduced, sample_weight=weight)

    ClusterModel(scaler, reducer, clusterer).save(path_model)
    logging.info("Saved %s", path_model)


def train_umap_hdbscan(df: pl.DataFrame, feature_columns: list[str], path_model: pathlib.Path,
                       knn_dir_path: pathlib.Path):
    """
    UMAP (2-D, 200 neighbours, min_dist 0.95, 500 epochs) and HDBSCAN (min_cluster_size 100)
    on the signatures left after dropping the rarest ones making up 5 % of observations.
    """
    df_train = df.filter(pl.col(COLUMN_CUM_PROB) > RARE_SHARE)
    logging.info("%d/%d rows after filtering", df_train.height, df.height)

    weight = df_train[COLUMN_COUNT].cast(pl.Float64)
    with timing("Scaling features (UMAP + HDBSCAN)"):
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(df_train[feature_columns], sample_weight=weight)
    with timing("UMAP"):
        features_reduced, reducer = _umap_fit_transform(features_scaled, n_components=2, max_iter=500,
                                                        n_neighbors=200, min_dist=0.95,
                                                        knn_dir_path=knn_dir_path)
    with timing("HDBSCAN"):
        clusterer_base = hdbscan.HDBSCAN(min_cluster_size=100, prediction_data=True)
        clusterer_base.fit(features_reduced)
        labels_train = hdbscan.approximate_predict(clusterer_base, features_reduced)[0]
        clusterer = CLusterIndexModel(features_reduced, labels_train)
    logging.info("HDBSCAN: %d clusters, %d/%d training signatures are noise",
                 labels_train.max() + 1, (labels_train == -1).sum(), len(labels_train))

    ClusterModel(scaler, reducer, clusterer).save(path_model)
    logging.info("Saved %s", path_model)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    DIR_OUT.mkdir(exist_ok=True)

    feature_columns = get_transformation_function(ArgTransform.V5)(TC_COLUMNS)
    df = load_signatures(PATH_DATASET, feature_columns)

    train_pca_kmeans(df, feature_columns, DIR_OUT / "pca_kmeans.pkl")
    train_umap_hdbscan(df, feature_columns, DIR_OUT / "umap_hdbscan.pkl", knn_dir_path=DIR_OUT)


if __name__ == "__main__":
    main()
