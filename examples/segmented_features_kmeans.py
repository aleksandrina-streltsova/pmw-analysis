import argparse
import pathlib

import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pmw_analysis.constants import COLUMN_COUNT, TC_COLUMNS, PMW_ANALYSIS_DIR
from pmw_analysis.decomposition import WPCA
from pmw_analysis.utils.pyplot import plot_histograms2d

N_BINS = 200

def kmeans(df_path: pathlib.Path, use_weights: bool):
    df_merged: pl.DataFrame = pl.read_parquet(df_path)

    df = df_merged[TC_COLUMNS + [COLUMN_COUNT]]
    df = df.drop_nans()

    weight = df[COLUMN_COUNT] if use_weights else pl.ones(len(df), eager=True)
    features = df[TC_COLUMNS]

    fs_scaled = StandardScaler().fit_transform(features, sample_weight=weight)

    n_clusters = 3

    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(fs_scaled)

    if use_weights:
        wpca = WPCA(n_components=2)
        fs_reduced = wpca.fit_transform(fs_scaled, sample_weight=weight.to_numpy())
    else:
        pca = PCA(n_components=2)
        fs_reduced = pca.fit_transform(fs_scaled)

    datas = [fs_reduced[labels == cluster] for cluster in range(n_clusters)]
    weights = [df[COLUMN_COUNT].filter(labels == cluster) for cluster in range(n_clusters)]
    titles = [f"Cluster {cluster}" for cluster in range(n_clusters)]

    dir_path = pathlib.Path("images") / df_path.parent.name
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"kmeans_log{"_w" if use_weights else ""}.png"

    plot_histograms2d(datas, weights, titles, file_path, bins=N_BINS, use_log_norm=True, alpha=0.5)


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
