"""
Option 4 for UMAP + HDBSCAN: the fine HDBSCAN clusters (B3's setting) on the option-3 UMAP embeddings, merged to 6 by
average linkage on unweighted cluster centroids, as b4_opt4_merge.py does for KMeans. Centroids in the scaled feature
space, and for comparison in the embedding. Extra UMAP runs of (15, 0.0) show the run-to-run spread.
"""
import pathlib
import sys
import time

import faiss
import hdbscan
import numpy as np
import polars as pl
from sklearn.cluster import AgglomerativeClustering
from umap import UMAP

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from b4_options_common import DIR_B4, FEATURE_COLUMNS, PATH_DATASET, PATH_UMAP_MODEL, write_results  # noqa: E402
from b4_opt3_umap import UMAP_GRID  # noqa: E402
from surface_groups import group_rows, load_signatures_with_counts, score_groups  # noqa: E402
from train_spring import RARE_SHARE  # noqa: E402

from pmw_analysis.analysis.clustering import ClusterModel, CLusterIndexModel, \
    DimensionalityReductionIndexModel  # noqa: E402

N_MERGED = 6
EXTRA_RUNS = [(15, 0.0, run) for run in [1, 2]]


def merge_scores(embedding_train, features_train, embedding_all, rows, signatures, extra: dict) -> list[dict]:
    clusterer_base = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=100, prediction_data=True).fit(embedding_train)
    labels_train = hdbscan.approximate_predict(clusterer_base, embedding_train)[0]
    labels = CLusterIndexModel(embedding_train, labels_train).predict(embedding_all)
    clusters = np.arange(labels_train.max() + 1)
    results = []
    for space, points in [("features", features_train), ("embedding", embedding_train)]:
        centroids = np.stack([points[labels_train == c].mean(axis=0) for c in clusters])
        merge = AgglomerativeClustering(n_clusters=N_MERGED, linkage="average").fit_predict(centroids)
        results.append({**extra, "fine_clusters": len(clusters), "centroid_space": space, "k": N_MERGED,
                        **score_groups(rows, signatures, merge[labels])})
    return results


def main(dir_umap=DIR_B4 / "opt3_umap", dir_out=DIR_B4 / "opt4_umap_merge"):
    dir_out.mkdir(exist_ok=True)
    df = load_signatures_with_counts(PATH_DATASET, FEATURE_COLUMNS)
    rows = group_rows(df)
    train = df.filter(pl.col("cum_prob") > RARE_SHARE)
    scaler = ClusterModel.load(PATH_UMAP_MODEL).scaler
    features_train = np.ascontiguousarray(scaler.transform(train.select(FEATURE_COLUMNS)), dtype="float32")
    features_all = scaler.transform(df.select(FEATURE_COLUMNS))
    index = faiss.IndexFlatL2(features_train.shape[1])
    index.add(features_train)

    results = []
    runs = [(n_neighbors, min_dist, 0) for n_neighbors, min_dist in UMAP_GRID] + EXTRA_RUNS
    for n_neighbors, min_dist, run in runs:
        start = time.time()
        if run == 0:  # the option-3 embedding
            embedding_train = np.load(dir_umap / f"embedding_{n_neighbors}_{min_dist}.npy")
        else:
            embedding_train = UMAP(n_components=2, n_epochs=500, n_neighbors=n_neighbors,
                                   min_dist=min_dist).fit_transform(features_train)
            np.save(dir_out / f"embedding_{n_neighbors}_{min_dist}_run{run}.npy", embedding_train)
        embedding_all = DimensionalityReductionIndexModel(index, embedding_train).transform(features_all)
        extra = {"n_neighbors": n_neighbors, "min_dist": min_dist, "umap_run": run}
        for row in merge_scores(embedding_train, features_train, embedding_all, rows, df["signature"], extra):
            results.append({**row, "seconds": time.time() - start})
            print(results[-1], flush=True)
        write_results(results, dir_out / "results.csv")


if __name__ == "__main__":
    main()
