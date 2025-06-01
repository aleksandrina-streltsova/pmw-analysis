"""
Script for running clusterization on quantized transformed data.
"""
import argparse
import pathlib
from typing import Callable

import faiss
import hdbscan
import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pmw_analysis.constants import COLUMN_COUNT, PMW_ANALYSIS_DIR, ST_COLUMNS, ST_GROUP_OCEAN, \
    ST_GROUP_SNOW, ST_GROUP_VEGETATION, VARIABLE_SURFACE_TYPE_INDEX, COLUMN_OCCURRENCE
from pmw_analysis.dr.pca import pca_fit_transform
from pmw_analysis.dr.umap import umap_fit_transform
from pmw_analysis.quantization.dataframe_polars import filter_surface_type, merge_quantized_pmw_features
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.pyplot import plot_histograms2d, HistogramData

N_BINS = 200

PCA = "pca"
UMAP = "umap"
KMEANS = "kmeans"
HDBSCAN = "hdbscan"

def clusterize(df_path: pathlib.Path, reduction: str, clusterization: str, transform: Callable):
    """
    Perform clusterization on the specified dataset.
    """
    use_weights = True
    df_path = pathlib.Path(PMW_ANALYSIS_DIR) / "partial" / "final.parquet"
    reduction = "pca"
    clusterization = "kmeans"
    #transform = get_transformation_function("v1")

    df_merged: pl.DataFrame = pl.read_parquet(df_path)
    #feature_columns = transform(TC_COLUMNS)
    #feature_columns = ['Tc_19V', 'Tc_89V', 'PD_89', 'Tc_37H_Tc_19H']
    feature_columns = ['PD_19', 'Tc_23V', 'Tc_89V', 'PD_89', 'Tc_37H_Tc_19H']

    df = df_merged[feature_columns + [COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX, COLUMN_OCCURRENCE]]
    df = merge_quantized_pmw_features([df], quant_columns=feature_columns)
    df = df.drop_nans()

    column_cum_prob = "cum_prob"
    df = df.with_columns(pl.col(COLUMN_COUNT).cast(pl.Float64))
    df = df.sort(COLUMN_COUNT, descending=False)
    df = df.with_columns((pl.col(COLUMN_COUNT).cum_sum() / pl.col(COLUMN_COUNT).sum()).alias(column_cum_prob))

    if reduction == "umap" or clusterization == "hdbscan":
        df_train = df.filter(pl.col(column_cum_prob) > 0.6)
    else:
        df_train = df

    weight_train = df_train[COLUMN_COUNT] if use_weights else pl.ones(len(df_train), eager=True)
    features_train = df_train[feature_columns]

    #### Scaling features ####
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train, sample_weight=weight_train)

    #### Reducing dimensionality ####
    if reduction == PCA:
        features_train_reduced, reducer = pca_fit_transform(features_train_scaled, weight_train if use_weights else None,
                                                            n_components=None)
    elif reduction == UMAP:
        features_train_reduced, reducer = umap_fit_transform(features_train_scaled,
                                                             n_components=2, max_iter=500,
                                                             n_neighbors=40, min_dist=0.6)
    else:
        raise ValueError(f"{reduction} is not supported")

    #### Clustering ####
    if clusterization == KMEANS:
        n_clusters = 4
        clusterer = KMeans(n_clusters=n_clusters)
        clusterer.fit(features_train_reduced, sample_weight=weight_train)
    elif clusterization == HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=100, prediction_data=True)
        clusterer.fit(features_train_reduced)
        labels_train, _ = hdbscan.approximate_predict(clusterer, features_train_reduced)
        n_clusters = labels_train.max() + 1
    else:
        raise ValueError(f"{clusterization} is not supported")

    #### Applying on the whole dataset ####
    features_scaled = scaler.transform(df[feature_columns])
    features_reduced = reducer.transform(features_scaled)
    if clusterization == KMEANS:
        labels = clusterer.predict(features_reduced)
    else:
        non_noisy_data = features_train_reduced[labels_train != -1]
        non_noisy_labels = labels_train[labels_train != -1]

        index = faiss.IndexFlatL2(non_noisy_data.shape[1])
        index.add(non_noisy_data)

        indices = index.search(features_reduced, k=1)[1].flatten()
        labels = non_noisy_labels[indices]

    #### Plotting ####
    use_log_norm = False

    dir_path = pathlib.Path("images") / df_path.parent.name
    dir_path.mkdir(parents=True, exist_ok=True)

    # density plot
    file_path = dir_path / f'{reduction}_{clusterization}_count.png'
    hist_data_count = [
        HistogramData(data=features_reduced, weight=df[COLUMN_COUNT], title=f"All", alpha=1.0,
                      cmap="rocket_r", color=None)
    ]
    plot_histograms2d(hist_data_count,path=file_path, bins=N_BINS, use_log_norm=use_log_norm)

    # clusterization results
    file_path = dir_path / f'{reduction}_{clusterization}.png'
    hist_datas = [
        HistogramData(data=features_reduced[labels == cluster],
                      weight=df[COLUMN_COUNT].filter(labels == cluster),
                      title=f"Cluster {cluster}",
                      alpha=0.5, cmap=None, color=None)
        for cluster in range(n_clusters)
    ]
    plot_histograms2d(hist_datas, path=file_path, bins=N_BINS, use_log_norm=use_log_norm, use_shared_norm=False)

    # reference data
    df = df.with_columns(
        pl.Series("x", features_reduced[:, 0]),
        pl.Series("y", features_reduced[:, 1]),
    )
    file_path = dir_path / f'{reduction}_{clusterization}_ref.png'
    hist_datas_ref = []
    groups = [
        ("Ocean (Group)", ST_GROUP_OCEAN, "navy"),
        ("Vegetation (Group)", ST_GROUP_VEGETATION, "darkgreen"),
        ("Snow (Group)", ST_GROUP_SNOW, "magenta"),
    ]
    for group in tqdm(groups):
        name, surface_types, color = group
        flag_values = [idx_st + 1 for idx_st, st in enumerate(ST_COLUMNS) if st in surface_types]

        df_to_use = filter_surface_type(df, flag_values)
        df_to_use = df_to_use.filter(df_to_use[COLUMN_COUNT].is_not_null())

        hist_datas_ref.append(HistogramData(data=df_to_use[["x", "y"]], weight=df_to_use[COLUMN_COUNT], title=name,
                                        alpha=0.5, cmap=None, color=color))
    plot_histograms2d(hist_datas_ref, file_path, bins=200, use_log_norm=use_log_norm, use_shared_norm=False)


def main():
    parser = argparse.ArgumentParser(description="Run KMeans++ on data and plot results using PCA for visualization")

    parser.add_argument("--transform", "-t", default="default", choices=["default", "pd", "ratio", "partial"],
                        help="Type of transformation performed on data")
    parser.add_argument("-dr", "--reduction", choices=["pca", "umap"])
    parser.add_argument("-c", "--clusterization", choices=["kmeans", "hdbscan"])

    args = parser.parse_args()
    transform = get_transformation_function(args.transform)
    path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform / "final.parquet"
    clusterize(path, args.reduction, args.clusterization, transform)


if __name__ == "__main__":
    main()
