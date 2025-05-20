"""
Script for running KMeans++ on quantized transformed data.
"""
import argparse
import pathlib

import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pmw_analysis.constants import COLUMN_COUNT, TC_COLUMNS, PMW_ANALYSIS_DIR
from pmw_analysis.dr.pca import pca_fit_transform
from pmw_analysis.utils.pyplot import plot_histograms2d, HistogramData

N_BINS = 200

def kmeans(df_path: pathlib.Path, use_weights: bool):
    """
    Perform KMeans++ clusterization on the specified dataset and
    plot the first two principal components of features, colored by obtained clusters.

    Parameters
    ----------
    df_path : pathlib.Path
        The path to the input parquet file containing the dataset to process.
    use_weights: bool
         If True, point counts are used to weigh points when running KMeans++ and PCA.
    """
    df_merged: pl.DataFrame = pl.read_parquet(df_path)

    df = df_merged[TC_COLUMNS + [COLUMN_COUNT]]
    df = df.drop_nans()

    weight = df[COLUMN_COUNT] if use_weights else pl.ones(len(df), eager=True)
    features = df[TC_COLUMNS]

    fs_scaled = StandardScaler().fit_transform(features, sample_weight=weight)

    n_clusters = 3

    reducer = KMeans(n_clusters=n_clusters)
    labels = reducer.fit_predict(fs_scaled)

    fs_reduced, _ = pca_fit_transform(fs_scaled, weight if use_weights else None)

    dir_path = pathlib.Path("images") / df_path.parent.name
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"kmeans_log{"_w" if use_weights else ""}.png"

    hist_datas = [
        HistogramData(data=fs_reduced[labels == cluster],
                      weight=df[COLUMN_COUNT].filter(labels == cluster),
                      title=f"Cluster {cluster}",
                      alpha=0.5, cmap=None, color=None)
        for cluster in range(n_clusters)
    ]
    plot_histograms2d(hist_datas, path=file_path, bins=N_BINS, use_log_norm=True)


def main():
    parser = argparse.ArgumentParser(description="Run KMeans++ on data and plot results using PCA for visualization")

    parser.add_argument("-w", action="store_true", help="Use weighted PCA")
    parser.add_argument("--transform", "-t", default="default", choices=["default", "pd", "ratio"],
                        help="Type of transformation performed on data")

    args = parser.parse_args()
    path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform / "final.parquet"
    kmeans(path, args.w)


if __name__ == "__main__":
    main()
