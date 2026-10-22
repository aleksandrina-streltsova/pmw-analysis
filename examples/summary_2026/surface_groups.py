"""
Surface-type groups for scoring clusterings, and observation-weighted B-Cubed from signature value counts.
unify_surface_types is copied from notebook.ipynb; bcubed from b4_signature_scores.py.
"""
from collections.abc import Sequence

import polars as pl

from pmw_analysis.constants import ST_COLUMNS, VARIABLE_SURFACE_TYPE_INDEX
from quantization.constants import STRUCT_FIELD_COUNT

COLUMN_VALUE_COUNTS = f"{VARIABLE_SURFACE_TYPE_INDEX}__valuecounts"

# Each group is labelled with the id of its first type; Sea-Ice (2) and Mixed coast (14) stay on their own
GROUPS_BY_NAME = [
    ["Ocean", "Ocean or water Coast", "Sea-ice edge"],
    ["High vegetation", "Medium vegetation", "Low vegetation", "Mountain rain", "Standing Water", "Land coast"],
    ["Sparse vegetation", "Desert"],
    ["Elevated snow cover", "High snow cover", "Moderate snow cover", "Light snow cover", "Mountain snow"],
]
# Surface type ids are 1-based positions in ST_COLUMNS
GROUPS = [[ST_COLUMNS.index(name) + 1 for name in group] for group in GROUPS_BY_NAME]


def unify_surface_types(df: pl.DataFrame, groups: Sequence[Sequence[int]] = (), suffix: str = "_unified") -> pl.DataFrame:
    """
    Add the surface-type value counts with each group of surface types counted as one type, labelled by the group's
    first index, as column f"{COLUMN_VALUE_COUNTS}{suffix}". Surface types in no group are kept as they are.
    """
    mapping = {st: group[0] for group in groups for st in group}
    column_out = f"{COLUMN_VALUE_COUNTS}{suffix}"
    counts = (
        df.select(pl.int_range(pl.len()).alias("row"), COLUMN_VALUE_COUNTS)
        .explode(COLUMN_VALUE_COUNTS, empty_as_null=True)
        .unnest(COLUMN_VALUE_COUNTS)
        .with_columns(pl.col(VARIABLE_SURFACE_TYPE_INDEX).replace(mapping))
        .group_by("row", VARIABLE_SURFACE_TYPE_INDEX)
        .agg(pl.col(STRUCT_FIELD_COUNT).sum())
        .sort("row", VARIABLE_SURFACE_TYPE_INDEX, nulls_last=True)
        .group_by("row", maintain_order=True)
        .agg(pl.struct(VARIABLE_SURFACE_TYPE_INDEX, STRUCT_FIELD_COUNT).alias(column_out))
    )
    assert counts.height == df.height
    return df.with_columns(counts[column_out])


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


NOT_LAND = [ST_COLUMNS.index("Ocean") + 1, ST_COLUMNS.index("Sea-Ice") + 1]


def load_signatures_with_counts(path, feature_columns: list[str]) -> pl.DataFrame:
    """
    The signatures without null features, with v5's column names, sorted by count (stably) with the cumulative share
    of observations "cum_prob" as in train_spring.py, their value counts, and a "signature" index.
    """
    from quantization.constants import COLUMN_COUNT, COLUMN_SUFFIX_QUANT
    df = pl.read_parquet(path, columns=[f"{col}{COLUMN_SUFFIX_QUANT}" for col in feature_columns]
                         + [COLUMN_COUNT, COLUMN_VALUE_COUNTS])
    df = df.rename({f"{col}{COLUMN_SUFFIX_QUANT}": col for col in feature_columns}).drop_nulls(feature_columns)
    count = pl.col(COLUMN_COUNT).cast(pl.UInt64)
    df = df.sort(COLUMN_COUNT, maintain_order=True).with_columns((count.cum_sum() / count.sum()).alias("cum_prob"))
    return df.with_row_index("signature")


def group_rows(df: pl.DataFrame, groups: Sequence[Sequence[int]] = GROUPS) -> pl.DataFrame:
    """
    One row per (signature, surface type) with its number of observations "n" and the type's group.
    """
    mapping = {st: group[0] for group in groups for st in group}
    return (df.select("signature", COLUMN_VALUE_COUNTS).explode(COLUMN_VALUE_COUNTS, empty_as_null=True)
            .unnest(COLUMN_VALUE_COUNTS).rename({STRUCT_FIELD_COUNT: "n"}).filter(pl.col("n") > 0)
            .with_columns(pl.col(VARIABLE_SURFACE_TYPE_INDEX).replace(mapping).alias("group")))


def score_groups(rows: pl.DataFrame, signatures: pl.Series, labels) -> dict:
    """
    B-Cubed against the groups for all observations and for land (every type except Ocean and Sea-Ice, among themselves).
    """
    df = rows.join(pl.DataFrame({"signature": signatures, "cluster": labels}), on="signature")
    p, r, f1 = bcubed(df, "cluster", "group")
    p_land, r_land, f1_land = bcubed(df.filter(~pl.col(VARIABLE_SURFACE_TYPE_INDEX).is_in(NOT_LAND)), "cluster", "group")
    return {"P": p, "R": r, "F1": f1, "P_land": p_land, "R_land": r_land, "F1_land": f1_land}
