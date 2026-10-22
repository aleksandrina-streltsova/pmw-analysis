"""
Option 3: UMAP with smaller n_neighbors and min_dist on the B3 training set (5 % cut, B3's scaler), then the HDBSCAN
grid, scored against the surface-type groups on all signatures. umap-learn computes its own kNN graph here.
"""
import pathlib
import sys
import time

import faiss
import numpy as np
import polars as pl
from umap import UMAP

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from b4_options_common import DIR_B4, FEATURE_COLUMNS, HDBSCAN_GRID, PATH_DATASET, PATH_UMAP_MODEL, \
    hdbscan_grid  # noqa: E402
from surface_groups import group_rows, load_signatures_with_counts  # noqa: E402
from train_spring import RARE_SHARE  # noqa: E402

from pmw_analysis.analysis.clustering import ClusterModel, DimensionalityReductionIndexModel  # noqa: E402

# (n_neighbors, min_dist)
UMAP_GRID = [(15, 0.0), (15, 0.1), (50, 0.0), (50, 0.1)]


def main(umap_grid=UMAP_GRID, hdbscan_settings=HDBSCAN_GRID, n_epochs=500, n_train=None, dir_out=DIR_B4 / "opt3_umap"):
    dir_out.mkdir(exist_ok=True)
    df = load_signatures_with_counts(PATH_DATASET, FEATURE_COLUMNS)
    rows = group_rows(df)
    train = df.filter(pl.col("cum_prob") > RARE_SHARE)
    if n_train is not None:  # smoke test only
        train = train.sample(n_train, seed=0)
    scaler = ClusterModel.load(PATH_UMAP_MODEL).scaler
    features_train = np.ascontiguousarray(scaler.transform(train.select(FEATURE_COLUMNS)), dtype="float32")
    features_all = scaler.transform(df.select(FEATURE_COLUMNS))
    index = faiss.IndexFlatL2(features_train.shape[1])
    index.add(features_train)

    results = []
    for n_neighbors, min_dist in umap_grid:
        start = time.time()
        embedding_train = UMAP(n_components=2, n_epochs=n_epochs, n_neighbors=n_neighbors,
                               min_dist=min_dist).fit_transform(features_train)
        seconds_umap = time.time() - start
        np.save(dir_out / f"embedding_{n_neighbors}_{min_dist}.npy", embedding_train)
        # Every signature gets the embedding of its nearest training signature, as in B3
        embedding_all = DimensionalityReductionIndexModel(index, embedding_train).transform(features_all)
        hdbscan_grid(embedding_train, embedding_all, rows, df["signature"], hdbscan_settings,
                     extra={"n_neighbors": n_neighbors, "min_dist": min_dist, "seconds_umap": seconds_umap},
                     results=results, path_results=dir_out / "results.csv")
    train.select("signature").write_parquet(dir_out / "train_signatures.parquet")


if __name__ == "__main__":
    main()
