"""
Option 2: KMeans++ with k = 5-8 under count, sqrt, log1p and no weights (applied to the scaler, the PCA and KMeans),
on all signatures, scored against the surface-type groups.
"""
import pathlib
import sys
import time

import polars as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from b4_options_common import DIR_B4, FEATURE_COLUMNS, PATH_DATASET, write_results  # noqa: E402
from surface_groups import group_rows, load_signatures_with_counts, score_groups  # noqa: E402

from pmw_analysis.analysis.clustering import _pca_fit_transform  # noqa: E402
from quantization.constants import COLUMN_COUNT  # noqa: E402

WEIGHTINGS = {
    "count": lambda count: count,
    "sqrt": lambda count: count.sqrt(),
    "log1p": lambda count: count.log1p(),
    "none": lambda count: None,
}
# (k, seed): seed 0 for every k, two more seeds at k = 6 to see the run-to-run spread
GRID = [(k, 0) for k in [5, 6, 7, 8]] + [(6, 1), (6, 2)]


def fit_reduce(df: pl.DataFrame, weight: pl.Series | None):
    # As train_spring.py: weighted scaler, then weighted PCA with all components
    features_scaled = StandardScaler().fit_transform(df.select(FEATURE_COLUMNS), sample_weight=weight)
    return _pca_fit_transform(features_scaled, weight, n_components=None)[0]


def main(weightings=WEIGHTINGS, grid=GRID, dir_out=DIR_B4 / "opt2_kmeans"):
    dir_out.mkdir(exist_ok=True)
    df = load_signatures_with_counts(PATH_DATASET, FEATURE_COLUMNS)
    rows = group_rows(df)
    count = df[COLUMN_COUNT].cast(pl.Float64)
    results = []
    for name, get_weight in weightings.items():
        weight = get_weight(count)
        features_reduced = fit_reduce(df, weight)
        for k, seed in grid:
            start = time.time()
            clusterer = KMeans(n_clusters=k, random_state=seed).fit(features_reduced, sample_weight=weight)
            results.append({"weighting": name, "k": k, "seed": seed,
                            **score_groups(rows, df["signature"], clusterer.labels_), "seconds": time.time() - start})
            print(results[-1], flush=True)
            write_results(results, dir_out / "results.csv")


if __name__ == "__main__":
    main()
