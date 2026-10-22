"""
B-Cubed of the B3 models against the surface-type groups of surface_groups.py, on the B2 signatures without null
features, for all observations and for land (the observations of every type except Ocean and Sea-Ice, scored among
themselves). Reference rows: one cluster, cluster = signature (precision bound), the most frequent group (target).
"""
import pathlib
import sys

import numpy as np
import polars as pl

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from surface_groups import COLUMN_VALUE_COUNTS, GROUPS, bcubed, unify_surface_types  # noqa: E402

from pmw_analysis.analysis.clustering import ClusterModel  # noqa: E402
from pmw_analysis.constants import TC_COLUMNS, ST_COLUMNS, VARIABLE_SURFACE_TYPE_INDEX, ArgTransform  # noqa: E402
from pmw_analysis.preprocessing.transforms import get_transformation_function  # noqa: E402
from quantization.constants import COLUMN_COUNT, COLUMN_SUFFIX_QUANT, STRUCT_FIELD_COUNT  # noqa: E402

PATH_DATASET = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b2/factor2/new/v5/all/final.parquet")
DIR_MODELS = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b3")
DIR_OUT = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b4/group_scores")
SUFFIX = "_unified"
NOT_LAND = [ST_COLUMNS.index("Ocean") + 1, ST_COLUMNS.index("Sea-Ice") + 1]


def main():
    DIR_OUT.mkdir(exist_ok=True)
    feature_columns = get_transformation_function(ArgTransform.V5)(TC_COLUMNS)
    df = pl.read_parquet(PATH_DATASET, columns=[f"{col}{COLUMN_SUFFIX_QUANT}" for col in feature_columns]
                         + [COLUMN_COUNT, COLUMN_VALUE_COUNTS])
    df = df.rename({f"{col}{COLUMN_SUFFIX_QUANT}": col for col in feature_columns})
    df = df.drop_nulls(feature_columns).with_row_index("signature")
    df = unify_surface_types(df, groups=GROUPS, suffix=SUFFIX)

    # Most frequent group of each signature, as in the notebook
    counts_unified = pl.col(f"{COLUMN_VALUE_COUNTS}{SUFFIX}")
    most_frequent = df.select(
        counts_unified.list.get(counts_unified.list.eval(pl.element().struct.field(STRUCT_FIELD_COUNT)).list.arg_max())
        .struct.field(VARIABLE_SURFACE_TYPE_INDEX)
    ).to_series()
    clusterings = {"one cluster": np.zeros(df.height, dtype=int), "cluster = signature": df["signature"].to_numpy(),
                   "most frequent group": most_frequent.to_numpy()}
    for model_name in ["pca_kmeans", "umap_hdbscan"]:
        clusterings[model_name] = ClusterModel.load(DIR_MODELS / f"{model_name}.pkl").predict(df.select(feature_columns))

    # One row per (signature, surface type) with its number of observations and the type's group
    mapping = {st: group[0] for group in GROUPS for st in group}
    rows = (df.select("signature", COLUMN_VALUE_COUNTS).explode(COLUMN_VALUE_COUNTS, empty_as_null=True)
            .unnest(COLUMN_VALUE_COUNTS).rename({STRUCT_FIELD_COUNT: "n"}).filter(pl.col("n") > 0)
            .with_columns(pl.col(VARIABLE_SURFACE_TYPE_INDEX).replace(mapping).alias("group")))
    assert rows["n"].cast(pl.UInt64).sum() == df[COLUMN_COUNT].cast(pl.UInt64).sum()
    in_range = pl.col(VARIABLE_SURFACE_TYPE_INDEX).is_between(1, len(ST_COLUMNS)).fill_null(False)
    n_other = rows.filter(~in_range)["n"].cast(pl.UInt64).sum()
    print(f"{df.height} signatures; observations with a null or out-of-range surface type: {n_other}")

    results = []
    for name, labels in clusterings.items():
        df_scored = rows.join(pl.DataFrame({"signature": df["signature"], "cluster": labels}), on="signature")
        df_land = df_scored.filter(~pl.col(VARIABLE_SURFACE_TYPE_INDEX).is_in(NOT_LAND))
        results.append({"clustering": name, "clusters": len(np.unique(labels)),
                        **dict(zip(["P", "R", "F1"], bcubed(df_scored, "cluster", "group"))),
                        **dict(zip(["P_land", "R_land", "F1_land"], bcubed(df_land, "cluster", "group"))),
                        "F1_types": bcubed(df_scored, "cluster", VARIABLE_SURFACE_TYPE_INDEX)[2]})
        print(results[-1], flush=True)

    results = pl.DataFrame(results)
    results.write_csv(DIR_OUT / "results.csv")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(results)


if __name__ == "__main__":
    main()
