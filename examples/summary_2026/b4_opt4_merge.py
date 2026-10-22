"""
Option 4: KMeans++ with many clusters (k = 20, 30, 40) under count, sqrt and no weights, then the centroids merged to
6 or 8 clusters by agglomerative clustering (unweighted centroids), scored against the surface-type groups.
The fine clustering's precision bounds that of any merge.
"""
import pathlib
import sys
import time

import polars as pl
from sklearn.cluster import AgglomerativeClustering, KMeans

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from b4_opt2_kmeans import WEIGHTINGS, fit_reduce  # noqa: E402
from b4_options_common import DIR_B4, FEATURE_COLUMNS, PATH_DATASET, write_results  # noqa: E402
from surface_groups import group_rows, load_signatures_with_counts, score_groups  # noqa: E402

from quantization.constants import COLUMN_COUNT  # noqa: E402

FINE_K = [20, 30, 40]
MERGED_K = [6, 8]
LINKAGES = ["average", "complete", "ward"]


def main(weightings=("count", "sqrt", "none"), fine_k=FINE_K, merged_k=MERGED_K, linkages=LINKAGES,
         dir_out=DIR_B4 / "opt4_merge"):
    dir_out.mkdir(exist_ok=True)
    df = load_signatures_with_counts(PATH_DATASET, FEATURE_COLUMNS)
    rows = group_rows(df)
    count = df[COLUMN_COUNT].cast(pl.Float64)
    results = []
    for name in weightings:
        weight = WEIGHTINGS[name](count)
        features_reduced = fit_reduce(df, weight)
        for k in fine_k:
            start = time.time()
            clusterer = KMeans(n_clusters=k, random_state=0).fit(features_reduced, sample_weight=weight)
            results.append({"weighting": name, "fine_k": k, "linkage": "none (fine)", "k": k,
                            **score_groups(rows, df["signature"], clusterer.labels_), "seconds": time.time() - start})
            print(results[-1], flush=True)
            for linkage in linkages:
                for k_merged in merged_k:
                    merge = AgglomerativeClustering(n_clusters=k_merged, linkage=linkage).fit_predict(
                        clusterer.cluster_centers_)
                    results.append({"weighting": name, "fine_k": k, "linkage": linkage, "k": k_merged,
                                    **score_groups(rows, df["signature"], merge[clusterer.labels_]),
                                    "seconds": time.time() - start})
                    print(results[-1], flush=True)
            write_results(results, dir_out / "results.csv")


if __name__ == "__main__":
    main()
