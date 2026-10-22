"""
What do the B3 clusters follow? From the B4 labels: B-Cubed against surface-type groups and per box, the share of
the cluster information explained by surface type, box, month, L2 variables and 2-m temperature, and cluster profiles.
"""
import pathlib

import gpm
import gpm.bucket
import numpy as np
import polars as pl

from pmw_analysis.constants import DIR_BUCKET, EXTENTS_SUBSET, ST_COLUMNS, VARIABLE_SURFACE_TYPE_INDEX, COLUMN_LON, \
    COLUMN_LAT, COLUMN_TIME, COLUMN_TEMP_2M_INDEX, COLUMN_CLUSTER, ST_GROUP_OCEAN, ST_GROUP_VEGETATION, ST_GROUP_SNOW, \
    ST_GROUP_EDGES, ST_GROUP_MISC
from pmw_analysis.preprocessing.dataframe_polars import _replace_special_missing_values_with_null

DIR_B4 = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b4")
MODELS = ["pca_kmeans", "umap_hdbscan"]
ST_GROUPS = {"Ocean": ST_GROUP_OCEAN, "Vegetation": ST_GROUP_VEGETATION, "Snow": ST_GROUP_SNOW,
             "Edges": ST_GROUP_EDGES, "Misc": ST_GROUP_MISC}
L2_BINNED = ["totalColumnWaterVaporIndex", "cloudWaterPath", "probabilityOfPrecip", COLUMN_TEMP_2M_INDEX]
SEASONS = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM", 6: "JJA", 7: "JJA", 8: "JJA",
           9: "SON", 10: "SON", 11: "SON"}


def read_temp2m(df_labels: pl.DataFrame) -> pl.Series:
    """
    Read temp2mIndex for the boxes in metric.py's order and check that the rows match the labels.
    """
    dfs = [gpm.bucket.read(bucket_dir=DIR_BUCKET, columns=[COLUMN_LON, COLUMN_LAT, COLUMN_TIME, COLUMN_TEMP_2M_INDEX],
                           extent=extent) for extent in EXTENTS_SUBSET.values()]
    df = _replace_special_missing_values_with_null(pl.concat(dfs).fill_nan(None))
    assert df.select(COLUMN_LON, COLUMN_LAT, COLUMN_TIME).equals(df_labels.select(COLUMN_LON, COLUMN_LAT, COLUMN_TIME))
    return df[COLUMN_TEMP_2M_INDEX]


def add_variables(df: pl.DataFrame) -> pl.DataFrame:
    st_to_group = {float(idx + 1): group for group, classes in ST_GROUPS.items()
                   for idx, st in enumerate(ST_COLUMNS) if st in classes}
    df = df.with_columns(
        pl.col(VARIABLE_SURFACE_TYPE_INDEX).replace_strict(st_to_group, return_dtype=pl.String).alias("st_group"),
        pl.col(COLUMN_TIME).dt.month().alias("month"),
        pl.col(COLUMN_TIME).dt.month().replace_strict(SEASONS, return_dtype=pl.String).alias("season"),
        (pl.col("box") * 100 + pl.col(COLUMN_TIME).dt.month()).alias("box_month"),
    )
    # Deciles of the continuous variables; nulls stay a level of their own
    return df.with_columns([pl.col(var).qcut(10, allow_duplicates=True).alias(f"{var}_decile") for var in L2_BINNED])


def entropy(df: pl.DataFrame, columns: list[str]) -> float:
    p = df.group_by(columns).len("n")["n"].cast(pl.Float64)
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def bcubed(df: pl.DataFrame, cluster_col: str, reference_col: str) -> tuple[float, float, float]:
    # As metric.py's calculate_b_cubed_f1, from the contingency counts
    s = df.group_by(cluster_col, reference_col).len("n").with_columns(pl.col("n").cast(pl.Float64)).with_columns(
        pl.col("n").sum().over(cluster_col).alias("n_cluster"), pl.col("n").sum().over(reference_col).alias("n_ref"))
    n = s["n"].sum()
    precision = (s["n"] ** 2 / s["n_cluster"]).sum() / n
    recall = (s["n"] ** 2 / s["n_ref"]).sum() / n
    return precision, recall, 2 * precision * recall / (precision + recall)


def information(df: pl.DataFrame, variables: list[str]) -> pl.DataFrame:
    """
    U = I(cluster; V) / H(cluster), and within boxes I(cluster; V | box) / H(cluster | box).
    """
    h_c, h_b, h_cb = entropy(df, [COLUMN_CLUSTER]), entropy(df, ["box"]), entropy(df, [COLUMN_CLUSTER, "box"])
    rows = []
    for var in variables:
        h_v, h_cv = entropy(df, [var]), entropy(df, [COLUMN_CLUSTER, var])
        u = (h_c + h_v - h_cv) / h_c
        u_box = None
        if var not in ("box", "box_month"):
            h_vb, h_cvb = entropy(df, [var, "box"]), entropy(df, [COLUMN_CLUSTER, var, "box"])
            u_box = (h_cb + h_vb - h_cvb - h_b) / (h_cb - h_b)
        rows.append({"variable": var, "levels": df[var].n_unique(), "U": u, "U_within_box": u_box})
    return pl.DataFrame(rows)


def cluster_profiles(df: pl.DataFrame) -> pl.DataFrame:
    n = df.height
    top = (df.group_by(COLUMN_CLUSTER, VARIABLE_SURFACE_TYPE_INDEX).len("n")
           .with_columns((pl.col("n") / pl.col("n").sum().over(COLUMN_CLUSTER)).alias("share"))
           .sort(COLUMN_CLUSTER, "n", descending=[False, True]).group_by(COLUMN_CLUSTER, maintain_order=True).head(3)
           .with_columns(pl.format("{} {}", pl.col(VARIABLE_SURFACE_TYPE_INDEX)
                                   .map_elements(lambda v: ST_COLUMNS[int(v) - 1], return_dtype=pl.String),
                                   (pl.col("share") * 100).round(0)).alias("label"))
           .group_by(COLUMN_CLUSTER, maintain_order=True).agg(pl.col("label").str.join(", ").alias("top_classes")))
    seasons = (df.group_by(COLUMN_CLUSTER, "season").len("n")
               .with_columns((pl.col("n") / pl.col("n").sum().over(COLUMN_CLUSTER)).round(3))
               .pivot(on="season", index=COLUMN_CLUSTER, values="n").select(COLUMN_CLUSTER, "DJF", "MAM", "JJA", "SON"))
    stats = df.group_by(COLUMN_CLUSTER).agg(
        (pl.len() / n).round(4).alias("share"),
        pl.col("box").mode().first().alias("main_box"),
        (pl.col("box") == pl.col("box").mode().first()).mean().round(3).alias("main_box_share"),
        pl.col(COLUMN_TEMP_2M_INDEX).median().alias("temp2m_median"),
        pl.col("totalColumnWaterVaporIndex").median().alias("tcwv_median"),
        (pl.col("probabilityOfPrecip") >= 50).mean().round(3).alias("precip_ge50"),
        pl.col("cloudWaterPath").mean().round(3).alias("cwp_mean"),
    )
    return stats.join(top, on=COLUMN_CLUSTER).join(seasons, on=COLUMN_CLUSTER).sort(COLUMN_CLUSTER)


def main():
    dir_out = DIR_B4 / "diagnostics"
    dir_out.mkdir(exist_ok=True)
    pl.Config(tbl_rows=50, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=80)

    columns = ["box", COLUMN_LON, COLUMN_LAT, COLUMN_TIME, VARIABLE_SURFACE_TYPE_INDEX, "cloudWaterPath",
               "probabilityOfPrecip", "totalColumnWaterVaporIndex"]
    df_base = pl.read_parquet(DIR_B4 / MODELS[0] / f"labels_{MODELS[0]}.parquet", columns=columns)
    df_base = add_variables(df_base.with_columns(read_temp2m(df_base)))

    # B-Cubed against surface types and groups, with trivial baselines
    df_base = df_base.with_columns(pl.lit(0).alias("one"))
    rows = [{"clustering": name, "reference": ref, **dict(zip(["P", "R", "F1"], bcubed(df_base, col, ref)))}
            for name, col in [("one cluster", "one"), ("cluster = box", "box")]
            for ref in [VARIABLE_SURFACE_TYPE_INDEX, "st_group"]]
    per_box, infos, profiles = [], [], {}
    for model in MODELS:
        clusters = pl.read_parquet(DIR_B4 / model / f"labels_{model}.parquet", columns=[COLUMN_CLUSTER])
        df = df_base.with_columns(clusters[COLUMN_CLUSTER])
        rows += [{"clustering": model, "reference": ref, **dict(zip(["P", "R", "F1"], bcubed(df, COLUMN_CLUSTER, ref)))}
                 for ref in [VARIABLE_SURFACE_TYPE_INDEX, "st_group"]]
        for (box,), df_box in df.group_by("box", maintain_order=True):
            p, r, f1 = bcubed(df_box, COLUMN_CLUSTER, VARIABLE_SURFACE_TYPE_INDEX)
            p0, r0, f0 = bcubed(df_box, "one", VARIABLE_SURFACE_TYPE_INDEX)
            per_box.append({"model": model, "box": box, "name": list(EXTENTS_SUBSET)[box],
                            "classes": df_box[VARIABLE_SURFACE_TYPE_INDEX].n_unique(),
                            "clusters": df_box[COLUMN_CLUSTER].n_unique(), "P": p, "R": r, "F1": f1,
                            "one_cluster_P": p0, "one_cluster_F1": f0})
        infos.append(information(df, [VARIABLE_SURFACE_TYPE_INDEX, "st_group", "box", "month", "season", "box_month"]
                                 + [f"{var}_decile" for var in L2_BINNED]).with_columns(pl.lit(model).alias("model")))
        profiles[model] = cluster_profiles(df)

    outputs = {"bcubed_groups": pl.DataFrame(rows), "bcubed_per_box": pl.DataFrame(per_box),
               "information": pl.concat(infos)}
    outputs.update({f"cluster_profiles_{model}": profile for model, profile in profiles.items()})
    for name, table in outputs.items():
        table.write_csv(dir_out / f"{name}.csv")
        print(f"== {name}\n{table}")


if __name__ == "__main__":
    main()
