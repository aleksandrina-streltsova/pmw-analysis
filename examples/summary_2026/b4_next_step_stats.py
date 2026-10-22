"""
Statistics of the B2 signatures behind the next-step options, with the surface-type groups of surface_groups.py:
group shares of the training weight and of the KMeans objective per weighting, nearest-group-centroid scores,
neighbourhood group purity in the UMAP training set, and where the 5 % cut's rare signatures are mapped.
"""
import pathlib
import sys

import faiss
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from surface_groups import COLUMN_VALUE_COUNTS, GROUPS, bcubed  # noqa: E402
from train_spring import RARE_SHARE  # noqa: E402

from pmw_analysis.analysis.clustering import ClusterModel  # noqa: E402
from pmw_analysis.constants import TC_COLUMNS, ST_COLUMNS, VARIABLE_SURFACE_TYPE_INDEX, ArgTransform  # noqa: E402
from pmw_analysis.preprocessing.transforms import get_transformation_function  # noqa: E402
from quantization.constants import COLUMN_COUNT, COLUMN_SUFFIX_QUANT, STRUCT_FIELD_COUNT  # noqa: E402

PATH_DATASET = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b2/factor2/new/v5/all/final.parquet")
PATH_UMAP_MODEL = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b3/umap_hdbscan.pkl")
DIR_OUT = pathlib.Path("/ltenas8/data/PMW_Analysis/summary_2026/b4/next_step_stats")
GROUP_NAMES = {1: "Ocean+coast", 2: "Sea-Ice", 3: "Vegetation", 6: "Arid", 8: "Snow", 14: "Mixed coast"}
NOT_LAND = [ST_COLUMNS.index("Ocean") + 1, ST_COLUMNS.index("Sea-Ice") + 1]
WEIGHTINGS = {"count": lambda c: c, "sqrt": np.sqrt, "log1p": np.log1p, "none": np.ones_like}
NEIGHBOURS = [15, 50, 200]
RARE_CUTS = [0.05, 0.01, 0.005, 0.001]


def load():
    feature_columns = get_transformation_function(ArgTransform.V5)(TC_COLUMNS)
    df = pl.read_parquet(PATH_DATASET, columns=[f"{col}{COLUMN_SUFFIX_QUANT}" for col in feature_columns]
                         + [COLUMN_COUNT, COLUMN_VALUE_COUNTS])
    df = df.rename({f"{col}{COLUMN_SUFFIX_QUANT}": col for col in feature_columns})
    count = pl.col(COLUMN_COUNT).cast(pl.UInt64)
    # As train_spring.py: no null features, sorted by count, cumulative share of observations
    df = df.drop_nulls(feature_columns).sort(COLUMN_COUNT).with_columns((count.cum_sum() / count.sum()).alias("cum"))
    df = df.with_row_index("signature")

    mapping = {st: group[0] for group in GROUPS for st in group}
    rows = (df.select("signature", COLUMN_VALUE_COUNTS).explode(COLUMN_VALUE_COUNTS, empty_as_null=True)
            .unnest(COLUMN_VALUE_COUNTS).rename({STRUCT_FIELD_COUNT: "n"}).filter(pl.col("n") > 0)
            .with_columns(pl.col(VARIABLE_SURFACE_TYPE_INDEX).replace(mapping).alias("group")))
    # Majority group of each signature (null surface types left out)
    majority = (rows.filter(pl.col("group").is_not_null()).group_by("signature", "group").agg(pl.col("n").sum())
                .sort("signature", "n", descending=[False, True]).group_by("signature", maintain_order=True).first()
                .select("signature", pl.col("group").alias("majority")))
    df = df.join(majority, on="signature", how="left")
    return df, rows, feature_columns


def score(rows: pl.DataFrame, signatures: pl.Series, labels: np.ndarray) -> dict:
    df = rows.join(pl.DataFrame({"signature": signatures, "cluster": labels}), on="signature")
    p, r, f1 = bcubed(df, "cluster", "group")
    p_land, r_land, f1_land = bcubed(df.filter(~pl.col(VARIABLE_SURFACE_TYPE_INDEX).is_in(NOT_LAND)), "cluster", "group")
    return {"P": p, "R": r, "F1": f1, "P_land": p_land, "R_land": r_land, "F1_land": f1_land}


def weighting_stats(df, rows, feature_columns):
    """
    Per weighting: each group's share of the weight (observations split by their groups) and of the within-group
    sum of squares in the scaled feature space, and the score of assigning every signature to the nearest group centroid.
    """
    count = df[COLUMN_COUNT].cast(pl.Float64).to_numpy()
    majority = df["majority"].to_numpy()
    shares, scores = [], []
    for name, get_weight in WEIGHTINGS.items():
        weight = get_weight(count)
        x = StandardScaler().fit_transform(df[feature_columns].to_numpy(), sample_weight=weight)
        # Weight of each signature split by the group composition of its observations
        w = rows.join(df.select("signature", pl.Series("w", weight), pl.col(COLUMN_COUNT).cast(pl.Float64)), on="signature")
        w_group = (w.group_by("group").agg((pl.col("w") * pl.col("n") / pl.col(COLUMN_COUNT)).sum().alias("w"))
                   .with_columns(pl.col("w") / pl.col("w").sum()))
        centroids, sse = {}, {}
        for group in GROUP_NAMES:
            mask = majority == group
            centroids[group] = np.average(x[mask], axis=0, weights=weight[mask])
            sse[group] = float((weight[mask] * ((x[mask] - centroids[group]) ** 2).sum(axis=1)).sum())
        total_sse = sum(sse.values())
        for group, group_name in GROUP_NAMES.items():
            shares.append({"weighting": name, "group": group_name,
                           "weight_share": w_group.filter(pl.col("group") == group)["w"].item(),
                           "sse_share": sse[group] / total_sse})
        groups = np.array(list(centroids))
        nearest = groups[np.argmin(((x[:, None, :] - np.stack(list(centroids.values()))[None]) ** 2).sum(axis=2), axis=1)]
        scores.append({"weighting": name, **score(rows, df["signature"], nearest)})
    return pl.DataFrame(shares), pl.DataFrame(scores)


def neighbourhood_stats(df, feature_columns):
    """
    In the UMAP training set (5 % cut), with the scaler fitted on it under each weighting (count is B3's): the share
    of each signature's nearest neighbours with its majority group; and with B3's scaling, whether the rare
    signatures' nearest training signature has the same group.
    """
    train, rare = df.filter(pl.col("cum") > RARE_SHARE), df.filter(pl.col("cum") <= RARE_SHARE)
    group_train = train["majority"].to_numpy()
    count_train = train[COLUMN_COUNT].cast(pl.Float64).to_numpy()
    purity = []
    for name, get_weight in WEIGHTINGS.items():
        scaler = StandardScaler().fit(train.select(feature_columns).to_numpy(), sample_weight=get_weight(count_train))
        x_train = np.ascontiguousarray(scaler.transform(train.select(feature_columns).to_numpy()), dtype="float32")
        index = faiss.IndexFlatL2(x_train.shape[1])
        index.add(x_train)
        neighbours = index.search(x_train, max(NEIGHBOURS) + 1)[1][:, 1:]  # the first neighbour is the signature itself
        same = group_train[neighbours] == group_train[:, None]
        for k in NEIGHBOURS:
            share = same[:, :k].mean(axis=1)
            purity.append({"scaler_weighting": name, "group": "all", "k": k, "purity": share.mean(),
                           "purity_count_weighted": np.average(share, weights=count_train)})
            for group, group_name in GROUP_NAMES.items():
                mask = group_train == group
                purity.append({"scaler_weighting": name, "group": group_name, "k": k, "purity": share[mask].mean(),
                               "purity_count_weighted": np.average(share[mask], weights=count_train[mask])})

    scaler = ClusterModel.load(PATH_UMAP_MODEL).scaler
    index = faiss.IndexFlatL2(len(feature_columns))
    index.add(np.ascontiguousarray(scaler.transform(train.select(feature_columns)), dtype="float32"))
    nearest = index.search(np.ascontiguousarray(scaler.transform(rare.select(feature_columns)), dtype="float32"), 1)[1][:, 0]
    count_rare = rare[COLUMN_COUNT].cast(pl.Float64).to_numpy()
    agree = group_train[nearest] == rare["majority"].to_numpy()
    rare_stats = {"rare_signatures": rare.height, "rare_obs_share": count_rare.sum() / df[COLUMN_COUNT].cast(pl.Float64).sum(),
                  "same_group_signatures": agree.mean(), "same_group_obs": np.average(agree, weights=count_rare)}
    cuts = [{"cut": cut, "training_signatures": df.filter(pl.col("cum") > cut).height} for cut in RARE_CUTS]
    return pl.DataFrame(purity), pl.DataFrame([rare_stats]), pl.DataFrame(cuts)


def main():
    DIR_OUT.mkdir(exist_ok=True)
    df, rows, feature_columns = load()
    outputs = dict(zip(["weighting_shares", "nearest_group_centroid"], weighting_stats(df, rows, feature_columns)))
    outputs.update(zip(["neighbour_purity", "rare_mapping", "rare_cuts"], neighbourhood_stats(df, feature_columns)))
    with pl.Config(tbl_rows=60, tbl_cols=-1, tbl_width_chars=200):
        for name, table in outputs.items():
            table.write_csv(DIR_OUT / f"{name}.csv")
            print(f"== {name}\n{table}", flush=True)


if __name__ == "__main__":
    main()
