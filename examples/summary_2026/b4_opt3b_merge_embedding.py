"""
Option 3b: fewer clusters on the option-3 UMAP embeddings, whose HDBSCAN clusters split the groups into many islands:
HDBSCAN with larger min_cluster_size or cluster_selection_epsilon, and KMeans on the 2-D embedding under count, sqrt
and no weights. Every signature gets the label of its nearest training point; scored against the surface-type groups.
"""
import pathlib
import sys
import time

import faiss
import hdbscan
import numpy as np
import polars as pl
from sklearn.cluster import KMeans

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from b4_options_common import DIR_B4, FEATURE_COLUMNS, PATH_DATASET, PATH_UMAP_MODEL, hdbscan_grid, \
    write_results  # noqa: E402
from b4_opt3_umap import UMAP_GRID  # noqa: E402
from surface_groups import group_rows, load_signatures_with_counts, score_groups  # noqa: E402
from train_spring import RARE_SHARE  # noqa: E402

from pmw_analysis.analysis.clustering import ClusterModel, CLusterIndexModel, \
    DimensionalityReductionIndexModel  # noqa: E402
from quantization.constants import COLUMN_COUNT  # noqa: E402

# Gaps between the fine clusters of the (15, 0.0) embedding: median 3.7, max 4.9 (embedding spans about 30 x 40)
HDBSCAN_GRID = [("eom", size, 100) for size in [5000, 10000, 20000]] + [("eom", 100, 100, eps) for eps in [4.0, 8.0, 16.0]]
KMEANS_K = [6, 8, 10]
WEIGHTINGS = {"count": lambda c: c, "sqrt": np.sqrt, "none": lambda c: None}


def majority_group_merge(rows: pl.DataFrame, signatures: pl.Series, labels: np.ndarray) -> np.ndarray:
    # Each cluster relabelled with its majority group: uses the reference, so a diagnostic of the clusters, not a result
    df = rows.join(pl.DataFrame({"signature": signatures, "cluster": labels}), on="signature")
    majority = (df.filter(pl.col("group").is_not_null()).group_by("cluster", "group").agg(pl.col("n").cast(pl.Float64).sum())
                .sort("cluster", "n", descending=[False, True]).group_by("cluster", maintain_order=True).first())
    mapping = dict(zip(majority["cluster"].to_list(), majority["group"].to_list()))
    return np.array([mapping[label] for label in labels])


def main(umap_grid=UMAP_GRID, dir_umap=DIR_B4 / "opt3_umap", dir_out=DIR_B4 / "opt3b_merge_embedding"):
    dir_out.mkdir(exist_ok=True)
    df = load_signatures_with_counts(PATH_DATASET, FEATURE_COLUMNS)
    rows = group_rows(df)
    # The option-3 training set and features (the stable sort makes it the same rows in the same order)
    train = df.filter(pl.col("cum_prob") > RARE_SHARE)
    scaler = ClusterModel.load(PATH_UMAP_MODEL).scaler
    features_train = np.ascontiguousarray(scaler.transform(train.select(FEATURE_COLUMNS)), dtype="float32")
    features_all = scaler.transform(df.select(FEATURE_COLUMNS))
    index = faiss.IndexFlatL2(features_train.shape[1])
    index.add(features_train)
    count_train = train[COLUMN_COUNT].cast(pl.Float64).to_numpy()

    results = []
    for n_neighbors, min_dist in umap_grid:
        embedding_train = np.load(dir_umap / f"embedding_{n_neighbors}_{min_dist}.npy")
        assert len(embedding_train) == train.height
        embedding_all = DimensionalityReductionIndexModel(index, embedding_train).transform(features_all)
        extra = {"n_neighbors": n_neighbors, "min_dist": min_dist}
        hdbscan_grid(embedding_train, embedding_all, rows, df["signature"], HDBSCAN_GRID, extra={**extra, "clusterer": "hdbscan"},
                     results=results, path_results=dir_out / "results.csv")
        # Fine HDBSCAN clusters (B3's setting) merged by majority group
        start = time.time()
        clusterer_base = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=100, prediction_data=True).fit(embedding_train)
        labels_train = hdbscan.approximate_predict(clusterer_base, embedding_train)[0]
        labels = CLusterIndexModel(embedding_train, labels_train).predict(embedding_all)
        results.append({**extra, "clusterer": "hdbscan + majority-group merge (uses reference)", "method": "eom",
                        "min_cluster_size": 100, "min_samples": 100, "epsilon": 0.0, "clusters": int(labels_train.max() + 1),
                        "noise_train": float((labels_train == -1).mean()),
                        **score_groups(rows, df["signature"], majority_group_merge(rows, df["signature"], labels)),
                        "seconds": time.time() - start})
        print(results[-1], flush=True)
        for name, get_weight in WEIGHTINGS.items():
            for k in KMEANS_K:
                start = time.time()
                labels_train = KMeans(n_clusters=k, random_state=0).fit(
                    embedding_train, sample_weight=get_weight(count_train)).labels_
                labels = CLusterIndexModel(embedding_train, labels_train).predict(embedding_all)
                results.append({**extra, "clusterer": "kmeans", "method": f"kmeans {name}", "min_cluster_size": None,
                                "min_samples": None, "epsilon": None, "clusters": k, "noise_train": 0.0,
                                **score_groups(rows, df["signature"], labels), "seconds": time.time() - start})
                print(results[-1], flush=True)
                write_results(results, dir_out / "results.csv")


if __name__ == "__main__":
    main()
