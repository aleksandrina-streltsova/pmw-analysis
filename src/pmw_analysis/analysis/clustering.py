"""
Script for clustering quantized transformed data.
"""
import itertools
import logging
import pathlib
import pickle
from datetime import datetime
from typing import Callable, Tuple, Any

import configargparse
import hdbscan
import joblib
import polars as pl
import pycolorbar
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchdr import UMAP as TorchdrUMAP
from torchdr.utils import faiss
from tqdm import tqdm
from umap import UMAP
from umap.umap_ import nearest_neighbors

from pmw_analysis.constants import COLUMN_COUNT, DIR_PMW_ANALYSIS, ST_COLUMNS, ST_GROUP_VEGETATION, \
    ST_GROUP_OCEAN, ST_GROUP_SNOW, ArgTransform, ArgDimensionalityReduction, ArgClustering, DIR_IMAGES, ArgSurfaceType, \
    FILE_DF_FINAL, DIR_MODEL, ST_GROUP_EDGES, ST_GROUP_MISC
from pmw_analysis.constants import VARIABLE_SURFACE_TYPE_INDEX, TC_COLUMNS
from pmw_analysis.copypaste.utils.cli import EnumAction
from pmw_analysis.copypaste.wpca import WPCA
from pmw_analysis.processing.filter import filter_by_flag_values
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.io import combine_paths, file_to_dir
from pmw_analysis.utils.logging import timing
from pmw_analysis.utils.pyplot import plot_histograms2d, HistogramData

N_BINS = 200


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
        Perform clustering pipeline on the input data.
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

    def __init__(self, features_train, labels_train):
        if isinstance(features_train, pl.DataFrame):
            features_train = features_train.to_numpy()

        non_noisy_data = features_train[labels_train != -1]
        non_noisy_labels = labels_train[labels_train != -1]

        index_train = faiss.IndexFlatL2(non_noisy_data.shape[1])
        index_train.add(non_noisy_data)

        self.index_train = index_train
        self.labels_train = non_noisy_labels

    def predict(self, features_reduced):
        """
        Perform clustering on the input data using nearest neighbor search.
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
                        knn_dir_path: pathlib.Path) -> Tuple[Any, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info("Reducing dimensionality with UMAP on a device: %s", device)

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
        knn_path = knn_dir_path / f"knn_{len(features)}.pkl"
        if not knn_path.exists():
            knn = nearest_neighbors(features,
                                    n_neighbors=200,
                                    metric="euclidean",
                                    metric_kwds=None,
                                    angular=False,
                                    random_state=566,
                                    verbose=True)
            with open(knn_path, "wb") as knn_file:
                pickle.dump(knn, knn_file)
        else:
            with open(knn_path, "rb") as knn_file:
                knn = pickle.load(knn_file)
        kwargs_umap = {
            "n_components": n_components,
            "n_epochs": max_iter,
            "verbose": True,
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "precomputed_knn": knn,
        }
        reducer_base = UMAP(**kwargs_umap)
    features_reduced = reducer_base.fit_transform(features)

    index = faiss.IndexFlatL2(features.shape[1])
    index.add(features)

    reducer = DimensionalityReductionIndexModel(index, features_reduced)
    return features_reduced, reducer


# TODO: remove other reduction/clustering scripts
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


def clusterize(df_path: pathlib.Path,
               reduction: ArgDimensionalityReduction, clustering: ArgClustering,
               transform: Callable):
    """
    Perform clustering on the specified dataset.
    """
    # args_transform = "v2"
    #
    # transform = get_transformation_function(args_transform)
    # df_path = pathlib.Path(PMW_ANALYSIS_DIR) / args_transform / "final.parquet"
    arg_transform = ArgTransform.V4
    arg_surface_type = ArgSurfaceType.LAND
    df_dir_path = pathlib.Path(DIR_PMW_ANALYSIS) / arg_transform.value / arg_surface_type.value

    transform = get_transformation_function(arg_transform)
    df_path = df_dir_path / "final_extra.parquet"
    clustering = ArgClustering.KMEANS
    reduction = ArgDimensionalityReduction.UMAP

    model_dir_path = combine_paths(path_base=DIR_MODEL, path_rel=df_path, path_rel_base=DIR_PMW_ANALYSIS)
    model_dir_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.INFO)

    use_weights = True

    df_merged: pl.DataFrame = pl.read_parquet(df_path)
    feature_columns = transform(TC_COLUMNS)

    df = df_merged
    # df = merge_quantized_pmw_features([df], quant_columns=feature_columns)
    df = df.drop_nans(feature_columns)
    df = df.with_columns(pl.col("peaks").dt.timestamp("ms").alias("peaks_timestamp"))

    column_cum_prob = "cum_prob"
    df = df.with_columns(pl.col(COLUMN_COUNT).cast(pl.Float64))
    df = df.sort(COLUMN_COUNT, descending=False)
    df = df.with_columns((pl.col(COLUMN_COUNT).cum_sum() / pl.col(COLUMN_COUNT).sum()).alias(column_cum_prob))
    # reduction = ArgDimensionalityReduction.UMAP
    # clustering = ArgClustering.KMEANS

    if reduction == ArgDimensionalityReduction.UMAP or clustering == ArgClustering.HDBSCAN:
        df_train = df.filter(pl.col(column_cum_prob) > 0.05)
        logging.info("%d/%d rows after filtering", len(df_train), len(df))
    else:
        df_train = df[::10]

    for i, params in enumerate(itertools.product([50, 100, 150, 200], [0.0, 0.5, 0.95], [500])):
        if i < 1:
            continue
        n_neighbors, min_dist, max_iter = params

        weight_train = df_train[COLUMN_COUNT] if use_weights else pl.ones(len(df_train), eager=True)
        features_train = df_train[feature_columns]

        with timing("Scaling features (train)"):
            scaler = StandardScaler()
            features_train_scaled = scaler.fit_transform(features_train, sample_weight=weight_train)

        with timing("Reducing dimensionality (train)"):
            match reduction:
                case ArgDimensionalityReduction.PCA:
                    features_train_reduced, reducer = _pca_fit_transform(features_train_scaled,
                                                                         weight_train if use_weights else None,
                                                                         n_components=None)
                case ArgDimensionalityReduction.UMAP:
                    features_train_reduced, reducer = _umap_fit_transform(features_train_scaled,
                                                                          n_components=2, max_iter=max_iter,
                                                                          n_neighbors=n_neighbors, min_dist=min_dist,

                                                                          knn_dir_path=model_dir_path)
                case _:
                    raise ValueError(f"{reduction.value} is not supported.")

        with timing("Clustering (train)"):
            match clustering:
                case ArgClustering.KMEANS:
                    n_clusters = 1
                    clusterer = KMeans(n_clusters=n_clusters)
                    clusterer.fit(features_train_reduced, sample_weight=weight_train)
                case ArgClustering.HDBSCAN:
                    clusterer_base = hdbscan.HDBSCAN(min_cluster_size=100, prediction_data=True)
                    clusterer_base.fit(features_train_reduced)

                    labels_train = hdbscan.approximate_predict(clusterer_base, features_train_reduced)[0]

                    clusterer = CLusterIndexModel(features_train_reduced, labels_train)
                    n_clusters = labels_train.max() + 1
                case _:
                    raise ValueError(f"{clustering.value} is not supported.")

        file_suffix = f"_{n_neighbors}_{min_dist}_{max_iter}_{len(df_train)}" \
            if reduction == ArgDimensionalityReduction.UMAP else ""
        title_suffix = f" (n_neighbors = {n_neighbors}, min_dist = {min_dist}, max_iter = {max_iter})" \
            if reduction == ArgDimensionalityReduction.UMAP else ""

        final_model = ClusterModel(scaler, reducer, clusterer)
        final_model.save(model_dir_path / f"{reduction.value}_{clustering.value}{file_suffix}.pkl")

        with timing("Scaling features"):
            # features_scaled = features_train_scaled
            features_scaled = scaler.transform(df[feature_columns])
        with timing("Reducing dimensionality"):
            # features_reduced = features_train_reduced
            features_reduced = reducer.transform(features_scaled)
        with timing("Clustering"):
            # labels = np.ones(len(features_scaled))
            labels = clusterer.predict(features_reduced)

        #### Plotting ####
        use_log_norm = True

        images_dir = combine_paths(path_base=DIR_IMAGES, path_rel=file_to_dir(df_path), path_rel_base=DIR_PMW_ANALYSIS)
        images_dir.mkdir(parents=True, exist_ok=True)

        # density plot
        file_path = images_dir / f'count_{reduction.value}_{clustering.value}{file_suffix}.png'
        hist_data_count = [
            HistogramData(data=features_reduced, weight=df[COLUMN_COUNT], title=f"All surfaces{title_suffix}", alpha=1.0,
                          cmap="rocket_r", color=None, x_label="Component 1", y_label="Component 2")
        ]
        plot_histograms2d(hist_data_count, path=file_path, title=reduction.value.upper(),
                          bins=N_BINS, use_log_norm=use_log_norm)


        # clustering results
        # file_path = images_dir / f'{reduction.value}_{clustering.value}{file_suffix}.png'
        # hist_datas = [
        #     HistogramData(data=features_reduced[labels == cluster],
        #                   weight=df[COLUMN_COUNT].filter(labels == cluster),
        #                   title=f"Cluster {cluster}",
        #                   alpha=0.8, cmap=None, color=None, x_label="Component 1", y_label="Component 2")
        #     for cluster in range(n_clusters)
        # ]
        # clustering_title = "KMeans++" if clustering == ArgClustering.KMEANS else "HDBSCAN"
        # plot_histograms2d(hist_datas, path=file_path, title=clustering_title, bins=N_BINS,
        #                   use_log_norm=use_log_norm, use_shared_norm=False)

        # reference data
        df = df.with_columns(
            pl.Series("x", features_reduced[:, 0]),
            pl.Series("y", features_reduced[:, 1]),
        )
        file_path = images_dir / f'{reduction.value}_{clustering.value}_ref{file_suffix}.png'
        hist_datas_ref = []
        groups = [
            # ("Ocean (Group)", ST_GROUP_OCEAN, "navy"),
            ("Vegetation (Group)", ST_GROUP_VEGETATION, "darkgreen"),
            ("Snow (Group)", ST_GROUP_SNOW, "magenta"),
            ("Edges (Group)", ST_GROUP_EDGES, None),
            ("Misc (Group)", ST_GROUP_MISC, None),
        ]
        for group in tqdm(groups):
            name, surface_types, color = group
            flag_values = [idx_st + 1 for idx_st, st in enumerate(ST_COLUMNS) if st in surface_types]

            df_to_use = filter_by_flag_values(df, VARIABLE_SURFACE_TYPE_INDEX, flag_values)
            df_to_use = df_to_use.filter(df_to_use[COLUMN_COUNT].is_not_null())

            hist_datas_ref.append(HistogramData(data=df_to_use[["x", "y"]], weight=df_to_use[COLUMN_COUNT], title=name,
                                                alpha=0.8, cmap=None, color=color,
                                                x_label="Component 1", y_label="Component 2"))
        plot_histograms2d(hist_datas_ref, file_path, title="Reference", bins=N_BINS,
                          use_log_norm=use_log_norm, use_shared_norm=False)

        # reference data (surface type)
        file_path = images_dir / f'{reduction.value}_{clustering.value}_ref_st{file_suffix}.png'
        hist_datas_ref = []
        st_cmap = pycolorbar.get_cmap("surfaceTypeIndexPalette")
        for idx, column_st in tqdm(enumerate(ST_COLUMNS)):
            df_to_use = filter_by_flag_values(df, VARIABLE_SURFACE_TYPE_INDEX, [idx + 1])
            if len(df_to_use) == 0:
                continue
            color = st_cmap.colors[idx]

            hist_datas_ref.append(HistogramData(data=df_to_use[["x", "y"]], weight=df_to_use[COLUMN_COUNT], title=column_st,
                                                alpha=0.8, cmap=None, color=color,
                                                x_label="Component 1", y_label="Component 2"))
        plot_histograms2d(hist_datas_ref, file_path, title="Reference", bins=N_BINS,
                          use_log_norm=use_log_norm, use_shared_norm=False)

        # occurrence class + peaks
        for column in ["occurrence_class", "peaks_timestamp"]:
            file_path = images_dir / f'{reduction.value}_{clustering.value}_{column}{file_suffix}.png'
            hist_datas_ref = []

            unique_values = np.sort(df[column].value_counts()[column].to_numpy())
            for idx, unique_value in tqdm(enumerate(unique_values)):
                if not isinstance(unique_value, str) and np.isnan(unique_value):
                    flag_value = None
                else:
                    flag_value = unique_value
                df_to_use = filter_by_flag_values(df, column, flag_value, nulls_equal=True)
                if len(df_to_use) == 0:
                    continue
                title = unique_value if column != "peaks_timestamp" or np.isnan(unique_value) \
                    else datetime.fromtimestamp(unique_value / 1000)
                hist_datas_ref.append(
                    HistogramData(data=df_to_use[["x", "y"]], weight=df_to_use[COLUMN_COUNT], title=title,
                                  alpha=0.8, cmap=None, color=None,
                                  x_label="Component 1", y_label="Component 2"))
            plot_histograms2d(hist_datas_ref, file_path, title=f"Reference: {column}", bins=N_BINS,
                              use_log_norm=use_log_norm, use_shared_norm=False)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = configargparse.ArgumentParser(config_arg_is_required=True, args_for_setting_config_path=["--config"],
                                           description="Run clustering and visualize results using DR")

    parser.add_argument("--transform", default=ArgTransform.DEFAULT,
                        type=ArgTransform, action=EnumAction,
                        help="Type of transformation performed on data")
    parser.add_argument("--reduction", type=ArgDimensionalityReduction, action=EnumAction)
    parser.add_argument("--clustering", type=ArgClustering, action=EnumAction)

    args = parser.parse_args()
    transform = get_transformation_function(args.transform)
    path = pathlib.Path(DIR_PMW_ANALYSIS) / args.transform.value / "final.parquet"
    clusterize(path, args.reduction, args.clustering, transform)


if __name__ == "__main__":
    main()
