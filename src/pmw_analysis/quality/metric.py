"""
This module contains functions for calculating metrics for clustering evaluation.
"""
import pathlib

import configargparse
import gpm
import gpm.bucket
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from pmw_analysis.constants import BUCKET_DIR, COLUMN_CLUSTER, VARIABLE_SURFACE_TYPE_INDEX, ST_COLUMNS, TC_COLUMNS, \
    PMW_ANALYSIS_DIR, ArgTransform, ArgDimensionalityReduction, ArgClustering
from pmw_analysis.analysis.clustering import ClusterModel
from pmw_analysis.copypaste.utils.cli import EnumAction
from pmw_analysis.quantization.dataframe_polars import _replace_special_missing_values_with_null
from pmw_analysis.quantization.script import get_transformation_function

L2_VARIABLES = ['cloudWaterPath', 'convectivePrecipitation', 'frozenPrecipitation',
                'probabilityOfPrecip', 'totalColumnWaterVaporIndex']


def calculate_b_cubed_f1(df: pl.DataFrame, cluster_col: str, reference_col: str, model_path: pathlib.Path) -> float:
    """
    Calculate B-Cubed F1 score for clustering evaluation.

    Attributes
    ----------
        df : pl.DataFrame
            Polars DataFrame containing the data.

        cluster_col : str
            Name of column with cluster assignments.

        reference_col : str
            Name of column with reference/true cluster assignments.

    Returns
    ----------
        float
            B-Cubed F1 score (float between 0 and 1)
    """
    # Calculate precision and recall for each item
    item_scores = df.select([
        pl.all(),
        # Count of same cluster items that are also in same reference cluster (for precision)
        pl.col(cluster_col).count().over(cluster_col).alias("cluster_size"),
        pl.col(reference_col).count().over(cluster_col, reference_col).alias("correct_in_cluster"),

        # Count of same reference items that are also in same predicted cluster (for recall)
        pl.col(reference_col).count().over(reference_col).alias("ref_cluster_size"),
        pl.col(cluster_col).count().over(cluster_col, reference_col).alias("correct_in_ref"),
    ]).with_columns([
        (pl.col("correct_in_cluster") / pl.col("cluster_size")).alias("precision"),
        (pl.col("correct_in_ref") / pl.col("ref_cluster_size")).alias("recall"),
        # (2 * pl.col("correct_in_ref") / (pl.col("ref_cluster_size") + pl.col("cluster_size"))).alias("f1"),
    ])

    df = df.fill_nan(None)
    df = _replace_special_missing_values_with_null(df)
    df_agr = df.group_by(cluster_col).mean()
    df_agr = df_agr.sort(cluster_col)[1:][[cluster_col] + L2_VARIABLES]

    fig, axes = plt.subplots(nrows=1, ncols=len(L2_VARIABLES),
                             figsize=(len(VARIABLE_SURFACE_TYPE_INDEX), 0.25 * len(df_agr) + 1), sharey=True)
    for idx, var in enumerate(L2_VARIABLES):
        ax = axes[idx]
        df_var = df_agr[[cluster_col, var]].to_pandas().set_index(cluster_col)
        sns.heatmap(df_var, annot=True, cmap="bwr", ax=ax)
        ax.set_ylabel(None)
    fig.supylabel("Cluster")
    fig.suptitle("Aggregated L2-data")

    plt.tight_layout()
    plt.savefig(pathlib.Path("images") / f"l2_{model_path.name.removesuffix(".pkl")}.png")
    plt.show()

    contingency_table = (
        item_scores
        .group_by([cluster_col, reference_col])
        .agg([
            pl.len().alias("count"),
            pl.col("precision").mean().alias("avg_precision"),
            pl.col("recall").mean().alias("avg_recall"),
        ])
        .sort(cluster_col, reference_col)
    )

    for value in ["avg_precision", "avg_recall"]:
        pivot_table = (
            contingency_table
            .pivot(
                index=reference_col,
                on=cluster_col,
                values=[value],
                aggregate_function="first",
            )
        )
        pivot_table = pivot_table.sort(reference_col)
        pivot_table = pivot_table.with_columns(
            pl.col(reference_col).map_elements(lambda flag: ST_COLUMNS[int(flag) - 1], return_dtype=str)
        )
        pivot_table = pivot_table.to_pandas().set_index(VARIABLE_SURFACE_TYPE_INDEX)

        plt.figure(figsize=(len(pivot_table.columns) + 2, 0.25 * len(pivot_table)))
        sns.heatmap(pivot_table, annot=True, cmap="Blues", vmin=0, vmax=1)  # Precision
        plt.xlabel("Cluster")
        plt.ylabel("Surface Type Index")
        plt.title(f"{value.removeprefix("avg_").capitalize()} Contingency Table")

        plt.tight_layout()
        plt.savefig(pathlib.Path("images") / f"cont_table_{value}_{model_path.name.removesuffix(".pkl")}.png")
        plt.show()

    # Calculate average precision and recall
    avg_precision = item_scores["precision"].mean()
    avg_recall = item_scores["recall"].mean()
    print(f"B-Cubed Precision: {avg_precision:.4f}")
    print(f"B-Cubed Recall: {avg_recall:.4f}")

    # return item_scores["f1"].mean()
    # Calculate F1 score
    if (avg_precision + avg_recall) == 0:
        return 0.0
    avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
    print(f"B-Cubed F1: {avg_f1:.4f}")
    return avg_f1


def main(model_path, transform):
    # reduction = ArgDimensionalityReduction.PCA
    # clustering = ArgClustering.KMEANS
    # model_path = pathlib.Path(PMW_ANALYSIS_DIR) / "v2" / f"{reduction}_{clustering}.pkl"
    # transform = get_transformation_function("v2")

    extents = [
        [42, 44, 39, 41],  # Sochi, Russia
        [59, 61, 4, 6],  # Bergen, Norway
        [17, 19, -17, -15],  # Nouakchott, Mauritania
        [59, 61, -46, -44],  # Nanortalik, Greenland
    ]
    dfs_partial = []

    for extent in extents:
        df_partial: pl.DataFrame = gpm.bucket.read(bucket_dir=BUCKET_DIR,
                                                   columns=None,
                                                   extent=extent)
        dfs_partial.append(df_partial)
    df = pl.concat(dfs_partial)

    feature_columns = transform(TC_COLUMNS)
    df = transform(df)
    has_nan = df.select(pl.any_horizontal(pl.col(feature_columns).is_nan())).to_series()
    model = ClusterModel.load(model_path)

    labels = -1 * np.ones(len(df), dtype=int)
    labels[(~has_nan).to_numpy().astype(bool)] = model.predict(df[feature_columns].filter(~has_nan))

    labels = pl.Series(COLUMN_CLUSTER, labels)
    df = df.insert_column(-1, labels.alias(COLUMN_CLUSTER))
    calculate_b_cubed_f1(df, COLUMN_CLUSTER, VARIABLE_SURFACE_TYPE_INDEX, model_path)


if __name__ == '__main__':
    parser = configargparse.ArgumentParser(config_arg_is_required=True, args_for_setting_config_path=["--config"],
                                           description="Calculate F1-score")

    parser.add_argument("--transform", default=ArgTransform.DEFAULT,
                        type=ArgTransform, action=EnumAction,
                        help="Type of transformation performed on data")
    parser.add_argument("--reduction", type=ArgDimensionalityReduction, action=EnumAction)
    parser.add_argument("--clustering", type=ArgClustering, action=EnumAction)

    args = parser.parse_args()
    main(
        model_path=pathlib.Path(PMW_ANALYSIS_DIR) / args.transform.value / f"{args.reduction}_{args.clustering}.pkl",
        transform=get_transformation_function(args.transform)
    )
