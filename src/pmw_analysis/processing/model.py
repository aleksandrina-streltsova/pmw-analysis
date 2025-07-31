"""
Script for running clusterization on quantized transformed data.
"""
import logging
import pathlib
import pickle
from typing import Callable, Tuple, Any

import configargparse
import hdbscan
import joblib
import polars as pl
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchdr import UMAP as TorchdrUMAP
from torchdr.utils import faiss
from tqdm import tqdm
from umap import UMAP
from umap.umap_ import nearest_neighbors

from pmw_analysis.constants import COLUMN_COUNT, PMW_ANALYSIS_DIR, ST_COLUMNS, ST_GROUP_VEGETATION, \
    ST_GROUP_OCEAN, ST_GROUP_SNOW
from pmw_analysis.constants import VARIABLE_SURFACE_TYPE_INDEX, COLUMN_OCCURRENCE, TC_COLUMNS
from pmw_analysis.copypaste.wpca import WPCA
from pmw_analysis.quantization.dataframe_polars import filter_surface_type
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.logging import timing
from pmw_analysis.utils.pyplot import plot_histograms2d, HistogramData

N_BINS = 200

DR_PCA = "pca"
DR_UMAP = "umap"
CLUSTER_KMEANS = "kmeans"
CLUSTER_HDBSCAN = "hdbscan"


class ClusterModel:
    """
    A machine learning pipeline which combines a scaler, dimensionality reducer, and a clustering model.
    """

    def __init__(self, scaler, reducer, clusterer):
        self.scaler = scaler
        self.reducer = reducer
        self.clusterer = clusterer

    def predict(self, features):
        """
        Perform clusterization pipeline on the input data.
        """
        features_scaled = self.scaler.transform(features)
        features_reduced = self.reducer.transform(features_scaled)
        labels = self.clusterer.predict(features_reduced)
        return labels

    def save(self, path):
        """
        Save the model to the specified path.
        """
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        """
        Load the model from the specified path.
        """
        return joblib.load(path)


class CLusterIndexModel:
    """
    A clustering model which clusters data using nearest neighbors search with a precomputed index.
    """

    def __init__(self, index_train, labels_train):
        self.index_train = index_train
        self.labels_train = labels_train

    def predict(self, features_reduced):
        """
        Perform clusterization on the input data using nearest neighbor search.
        """
        indices = self.index_train.search(features_reduced, k=1)[1].flatten()
        labels = self.labels_train[indices]
        return labels


class DimensionalityReductionIndexModel:
    """
    A dimensionality reduction model which reduces data using nearest neighbors search with a precomputed index.
    """

    def __init__(self, index_train, embeddings_train):
        self.index_train = index_train
        self.embeddings_train = embeddings_train

    def transform(self, features_reduced):
        """
        Perform dimensionality reduction on the input data using nearest neighbor search.
        """
        indices = self.index_train.search(features_reduced, k=1)[1].flatten()
        embeddings = self.embeddings_train[indices]
        return embeddings


def _umap_fit_transform(features: pl.DataFrame,
                        n_components: int, max_iter: int,
                        n_neighbors: int, min_dist: float,
                        knn_path: pathlib.Path) -> Tuple[Any, Any]:
    if not knn_path.exists():
        mnist_knn = nearest_neighbors(features,
                                      n_neighbors=200,
                                      metric="euclidean",
                                      metric_kwds=None,
                                      angular=False,
                                      random_state=566)
        with open(knn_path, "wb") as knn_file:
            pickle.dump(mnist_knn, knn_file)
    else:
        with open(knn_path, "rb") as knn_file:
            mnist_knn = pickle.load(knn_file)

    if torch.cuda.is_available():
        kwargs_torchdr = {
            "n_components": n_components,
            "max_iter": max_iter,
            "verbose": True,
            "backend": "faiss",
            "device": "cuda",
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
        }
        reducer_base = TorchdrUMAP(**kwargs_torchdr)
    else:
        kwargs_umap = {
            "n_components": n_components,
            "n_epochs": max_iter,
            "verbose": True,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "precomputed_knn": mnist_knn,
        }
        reducer_base = UMAP(**kwargs_umap)
    features_reduced = reducer_base.fit_transform(features)

    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features.shape[0], features)

    reducer = DimensionalityReductionIndexModel(index, features_reduced)
    return features_reduced, reducer


# TODO: remove other reduction/clusterization scripts
def _pca_fit_transform(data, weight, n_components=2):
    """
    Perform dimensionality reduction on the input data using PCA or Weighted PCA.
    """
    if weight is not None:
        reducer = WPCA(n_components)
        fs_reduced = reducer.fit_transform(data, sample_weight=weight.to_numpy())
    else:
        reducer = PCA(n_components)
        fs_reduced = reducer.fit_transform(data)
    return fs_reduced, reducer


def clusterize(df_path: pathlib.Path, reduction: str, clusterization: str, transform: Callable):
    """
    Perform clusterization on the specified dataset.
    """
    # args_transform = "v2"
    #
    # transform = get_transformation_function(args_transform)
    # df_path = pathlib.Path(PMW_ANALYSIS_DIR) / args_transform / "final.parquet"
    #
    use_weights = True

    df_merged: pl.DataFrame = pl.read_parquet(df_path)
    feature_columns = transform(TC_COLUMNS)

    df = df_merged[feature_columns + [COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX, COLUMN_OCCURRENCE]]
    # df = merge_quantized_pmw_features([df], quant_columns=feature_columns)
    df = df.drop_nans(feature_columns)

    column_cum_prob = "cum_prob"
    df = df.with_columns(pl.col(COLUMN_COUNT).cast(pl.Float64))
    df = df.sort(COLUMN_COUNT, descending=False)
    df = df.with_columns((pl.col(COLUMN_COUNT).cum_sum() / pl.col(COLUMN_COUNT).sum()).alias(column_cum_prob))

    # reduction = "umap"
    # clusterization = "kmeans"

    if reduction == "umap" or clusterization == "hdbscan":
        df_train = df.filter(pl.col(column_cum_prob) > 0.05)
        logging.info("%d/%d rows after filtering", len(df_train), len(df))
    else:
        df_train = df

    weight_train = df_train[COLUMN_COUNT] if use_weights else pl.ones(len(df_train), eager=True)
    features_train = df_train[feature_columns]

    with timing("Scaling features (train)"):
        scaler = StandardScaler()
        features_train_scaled = scaler.fit_transform(features_train, sample_weight=weight_train)

    with timing("Reducing dimensionality (train)"):
        if reduction == DR_PCA:
            features_train_reduced, reducer = _pca_fit_transform(features_train_scaled,
                                                                 weight_train if use_weights else None,
                                                                 n_components=None)
        elif reduction == DR_UMAP:
            knn_path = pathlib.Path(df_path.parent / f"knn_{len(df_train)}.pkl")
            features_train_reduced, reducer = _umap_fit_transform(features_train_scaled,
                                                                  n_components=2, max_iter=500,
                                                                  n_neighbors=200, min_dist=0.95,
                                                                  knn_path=knn_path)
        else:
            raise ValueError(f"{reduction} is not supported")

    with timing("Clustering (train)"):
        if clusterization == CLUSTER_KMEANS:
            n_clusters = 4
            clusterer = KMeans(n_clusters=n_clusters)
            clusterer.fit(features_train_reduced, sample_weight=weight_train)
        elif clusterization == CLUSTER_HDBSCAN:
            clusterer_base = hdbscan.HDBSCAN(min_cluster_size=100, prediction_data=True)
            clusterer_base.fit(features_train_reduced)

            labels_train = hdbscan.approximate_predict(clusterer_base, features_train_reduced)[0]
            non_noisy_data = features_train_reduced[labels_train != -1]
            non_noisy_labels = labels_train[labels_train != -1]

            index = faiss.IndexFlatL2(non_noisy_data.shape[1])
            index.add(non_noisy_data.shape[0], non_noisy_data)

            clusterer = CLusterIndexModel(index, non_noisy_labels)
            n_clusters = labels_train.max() + 1
        else:
            raise ValueError(f"{clusterization} is not supported")

    final_model = ClusterModel(scaler, reducer, clusterer)
    final_model.save(df_path.parent / f"{reduction}_{clusterization}.pkl")

    with timing("Scaling features"):
        features_scaled = scaler.transform(df[feature_columns])
    with timing("Reducing dimensionality"):
        features_reduced = reducer.transform(features_scaled)
    with timing("Clustering"):
        labels = clusterer.predict(features_reduced)

    #### Plotting ####
    use_log_norm = False

    dir_path = pathlib.Path("images") / df_path.parent.name
    dir_path.mkdir(parents=True, exist_ok=True)

    # density plot
    file_path = dir_path / f'{reduction}_{clusterization}_count.png'
    hist_data_count = [
        HistogramData(data=features_reduced, weight=df[COLUMN_COUNT], title="All surfaces", alpha=1.0,
                      cmap="rocket_r", color=None, x_label="Component 1", y_label="Component 2")
    ]
    plot_histograms2d(hist_data_count, path=file_path, title=reduction.upper(),
                      bins=N_BINS, use_log_norm=use_log_norm)

    # clusterization results
    file_path = dir_path / f'{reduction}_{clusterization}.png'
    hist_datas = [
        HistogramData(data=features_reduced[labels == cluster],
                      weight=df[COLUMN_COUNT].filter(labels == cluster),
                      title=f"Cluster {cluster}",
                      alpha=0.8, cmap=None, color=None, x_label="Component 1", y_label="Component 2")
        for cluster in range(n_clusters)
    ]
    clusterization_title = "KMeans++" if clusterization == "kmeans" else "HDBSCAN"
    plot_histograms2d(hist_datas, path=file_path, title=clusterization_title, bins=N_BINS,
                      use_log_norm=use_log_norm, use_shared_norm=False)

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
                                            alpha=0.8, cmap=None, color=color,
                                            x_label="Component 1", y_label="Component 2"))
    plot_histograms2d(hist_datas_ref, file_path, title="Reference", bins=N_BINS,
                      use_log_norm=use_log_norm, use_shared_norm=False)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = configargparse.ArgumentParser(config_arg_is_required=True, args_for_setting_config_path=["--config"],
                                           description="Run clusterization and visualize results using DR")

    parser.add_argument("--transform", default="default",
                        choices=["default", "pd", "ratio", "partial", "v1", "v2", "v3"],
                        help="Type of transformation performed on data")
    parser.add_argument("--reduction", choices=["pca", "umap"])
    parser.add_argument("--clusterization", choices=["kmeans", "hdbscan"])

    args = parser.parse_args()
    transform = get_transformation_function(args.transform)
    path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform / "final.parquet"
    clusterize(path, args.reduction, args.clusterization, transform)


if __name__ == "__main__":
    main()
