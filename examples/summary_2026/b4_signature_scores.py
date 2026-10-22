"""
B-Cubed on all signatures of the B2 dataset, from the surface-type counts each signature aggregates: every observation
gets its signature's cluster. Cluster = signature gives the upper bound on precision of any clustering of signatures.
"""
import pathlib
import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from b4_diagnostics import ST_GROUPS  # noqa: E402

from pmw_analysis.analysis.clustering import ClusterModel  # noqa: E402
from pmw_analysis.constants import TC_COLUMNS, ST_COLUMNS, VARIABLE_SURFACE_TYPE_INDEX, ArgTransform  # noqa: E402
from pmw_analysis.preprocessing.transforms import get_transformation_function  # noqa: E402
from quantization.constants import COLUMN_COUNT, COLUMN_SUFFIX_QUANT  # noqa: E402

PATH_DATASET = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b2/factor2/new/v5/all/final.parquet")
DIR_MODELS = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b3")
DIR_OUT = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b4/signature_scores")
COLUMN_VALUE_COUNTS = f"{VARIABLE_SURFACE_TYPE_INDEX}__valuecounts"
LAND = [st for st in ST_COLUMNS if st not in ["Ocean", "Sea-Ice"]]


def bcubed(df: pl.DataFrame, cluster_col: str, reference_col: str) -> tuple[float, float, float]:
    """
    B-Cubed from rows of (cluster, reference class, number of observations).
    """
    # Cast before summing: the sum of UInt32 counts overflows
    s = df.group_by(cluster_col, reference_col).agg(pl.col("n").cast(pl.Float64).sum()).with_columns(
        pl.col("n").sum().over(cluster_col).alias("n_cluster"), pl.col("n").sum().over(reference_col).alias("n_ref"))
    n = s["n"].sum()
    precision = (s["n"] ** 2 / s["n_cluster"]).sum() / n
    recall = (s["n"] ** 2 / s["n_ref"]).sum() / n
    return precision, recall, 2 * precision * recall / (precision + recall)


def main():
    DIR_OUT.mkdir(exist_ok=True)
    feature_columns = get_transformation_function(ArgTransform.V5)(TC_COLUMNS)
    df = pl.read_parquet(PATH_DATASET, columns=[f"{col}{COLUMN_SUFFIX_QUANT}" for col in feature_columns]
                         + [COLUMN_COUNT, COLUMN_VALUE_COUNTS])
    df = df.rename({f"{col}{COLUMN_SUFFIX_QUANT}": col for col in feature_columns}).with_row_index("signature")
    has_null = df.select(pl.any_horizontal(pl.col(feature_columns).is_null())).to_series().to_numpy()

    clusterings = {"one cluster": np.zeros(df.height, dtype=int), "cluster = signature": df["signature"].to_numpy()}
    for model_name in ["pca_kmeans", "umap_hdbscan"]:
        # As metric.py: signatures with a null feature get cluster -1
        labels = -1 * np.ones(df.height, dtype=int)
        labels[~has_null] = ClusterModel.load(DIR_MODELS / f"{model_name}.pkl").predict(
            df.filter(~pl.Series(has_null)).select(feature_columns))
        clusterings[model_name] = labels

    # One row per (signature, surface type) with its number of observations
    st_to_group = {float(idx + 1): group for group, classes in ST_GROUPS.items()
                   for idx, st in enumerate(ST_COLUMNS) if st in classes}
    land = [float(idx + 1) for idx, st in enumerate(ST_COLUMNS) if st in LAND]
    rows = (df.select("signature", COLUMN_VALUE_COUNTS).explode(COLUMN_VALUE_COUNTS, empty_as_null=True)
            .unnest(COLUMN_VALUE_COUNTS).rename({"struct_count": "n"}).filter(pl.col("n") > 0)
            .with_columns(pl.col(VARIABLE_SURFACE_TYPE_INDEX).cast(pl.Float64))
            .with_columns(pl.col(VARIABLE_SURFACE_TYPE_INDEX).replace_strict(st_to_group, return_dtype=pl.String)
                          .alias("st_group")))
    assert rows["n"].cast(pl.UInt64).sum() == df[COLUMN_COUNT].cast(pl.UInt64).sum()

    results = []
    for name, labels in clusterings.items():
        df_scored = rows.join(pl.DataFrame({"signature": df["signature"], "cluster": labels}), on="signature")
        df_land = df_scored.filter(pl.col(VARIABLE_SURFACE_TYPE_INDEX).is_in(land))
        results.append({"clustering": name, "clusters": len(np.unique(labels)),
                        **dict(zip(["P", "R", "F1"], bcubed(df_scored, "cluster", VARIABLE_SURFACE_TYPE_INDEX))),
                        "F1_groups": bcubed(df_scored, "cluster", "st_group")[2],
                        **dict(zip(["P_land", "R_land", "F1_land"],
                                   bcubed(df_land, "cluster", VARIABLE_SURFACE_TYPE_INDEX)))})
        print(results[-1], flush=True)

    results = pl.DataFrame(results)
    results.write_csv(DIR_OUT / "results.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(results)


if __name__ == "__main__":
    main()
