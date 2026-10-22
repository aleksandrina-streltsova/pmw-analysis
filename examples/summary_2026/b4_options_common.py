"""
Shared parts of the B4 option scripts (b4_opt*.py): data, HDBSCAN grid on a 2-D embedding, incremental result files.
"""
import pathlib
import time

import hdbscan
import polars as pl

from surface_groups import score_groups

from pmw_analysis.analysis.clustering import CLusterIndexModel
from pmw_analysis.constants import TC_COLUMNS, ArgTransform
from pmw_analysis.preprocessing.transforms import get_transformation_function

PATH_DATASET = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b2/factor2/new/v5/all/final.parquet")
PATH_UMAP_MODEL = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b3/umap_hdbscan.pkl")
DIR_B4 = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b4")
FEATURE_COLUMNS = get_transformation_function(ArgTransform.V5)(TC_COLUMNS)
# (cluster_selection_method, min_cluster_size, min_samples[, cluster_selection_epsilon]); B3 is ("eom", 100, 100)
HDBSCAN_GRID = [(method, size, samples) for method, sizes in [("eom", [100, 500, 1000, 2000]),
                                                               ("leaf", [500, 1000, 2000, 5000])]
                for size in sizes for samples in [10, 100]]


def write_results(results: list[dict], path: pathlib.Path):
    # Rewritten after every row, so an interrupted run keeps what it has
    pl.DataFrame(results).write_csv(path)


def hdbscan_grid(embedding_train, embedding_all, rows: pl.DataFrame, signatures: pl.Series, grid: list[tuple],
                 extra: dict, results: list[dict], path_results: pathlib.Path):
    """
    For each HDBSCAN setting: fit on the training embedding, label the training points with approximate_predict as
    train_spring.py does, give every signature the label of its nearest non-noise training point, and score.
    """
    for method, size, samples, *epsilon in grid:
        epsilon = epsilon[0] if epsilon else 0.0
        start = time.time()
        clusterer_base = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=samples, cluster_selection_method=method,
                                         cluster_selection_epsilon=epsilon, prediction_data=True).fit(embedding_train)
        labels_train = hdbscan.approximate_predict(clusterer_base, embedding_train)[0]
        labels = CLusterIndexModel(embedding_train, labels_train).predict(embedding_all)
        results.append({**extra, "method": method, "min_cluster_size": size, "min_samples": samples, "epsilon": epsilon,
                        "clusters": int(labels_train.max() + 1), "noise_train": float((labels_train == -1).mean()),
                        **score_groups(rows, signatures, labels), "seconds": time.time() - start})
        print(results[-1], flush=True)
        write_results(results, path_results)
