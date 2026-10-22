"""
Option 1: HDBSCAN settings on the B3 UMAP embedding, scored against the surface-type groups on all signatures.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from b4_options_common import DIR_B4, FEATURE_COLUMNS, HDBSCAN_GRID, PATH_DATASET, PATH_UMAP_MODEL, \
    hdbscan_grid  # noqa: E402
from surface_groups import group_rows, load_signatures_with_counts  # noqa: E402

from pmw_analysis.analysis.clustering import ClusterModel  # noqa: E402


def main(grid=HDBSCAN_GRID, dir_out=DIR_B4 / "opt1_hdbscan"):
    dir_out.mkdir(exist_ok=True)
    df = load_signatures_with_counts(PATH_DATASET, FEATURE_COLUMNS)
    rows = group_rows(df)
    model = ClusterModel.load(PATH_UMAP_MODEL)
    # Embedding of every signature: that of its nearest training signature, as the B3 model's reducer gives it
    embedding_all = model.reducer.transform(model.scaler.transform(df.select(FEATURE_COLUMNS)))
    hdbscan_grid(model.reducer.embeddings_train, embedding_all, rows, df["signature"], grid,
                 extra={"embedding": "B3 (n_neighbors 200, min_dist 0.95)"}, results=[],
                 path_results=dir_out / "results.csv")


if __name__ == "__main__":
    main()
