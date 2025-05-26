"""
Script for running GPU implementation of UMAP on quantized transformed data.
"""
import argparse
import pathlib

import polars as pl
import torch
from sklearn.preprocessing import StandardScaler
from torchdr import UMAP as TorchdrUMAP
from tqdm import tqdm

from pmw_analysis.constants import COLUMN_COUNT, PMW_ANALYSIS_DIR, ST_COLUMNS, ST_GROUP_VEGETATION, \
    ST_GROUP_OCEAN, ST_GROUP_SNOW
from pmw_analysis.quantization.dataframe_polars import filter_surface_type
from pmw_analysis.utils.pyplot import plot_histograms2d, HistogramData


def umap(df_path: pathlib.Path, min_dist, n_neighbors):
    """
    Performs UMAP (Uniform Manifold Approximation and Projection) dimensionality reduction
    on the specified dataset and plots embeddings in 2D space.

    Parameters
    ----------
    df_path : pathlib.Path
        The path to the input parquet file containing the dataset to process.
    min_dist : float
        The minimum distance parameter for UMAP.
    n_neighbors : int
        The number of nearest neighbors parameter for UMAP.
    """
    use_log_norm = True

    df_merged = pl.read_parquet(df_path)
    feature_columns = [col for col in df_merged.columns if col.startswith('Tc_')]
    df_merged = df_merged.drop_nans(subset=feature_columns)

    df = df_merged[:1000000]

    max_iter = 500

    features_scaled = df[feature_columns]
    features_scaled = StandardScaler().fit_transform(features_scaled)

    kwargs_torchdr = {
        "max_iter": max_iter,
        "verbose": True,
        "backend": "faiss",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
    }

    torchdr_umap = TorchdrUMAP(**kwargs_torchdr)
    embedding = torchdr_umap.fit_transform(features_scaled)

    df = df.with_columns(
        pl.Series("x", embedding[:, 0]),
        pl.Series("y", embedding[:, 1]),
    )

    groups = [
        ("Ocean (Group)", ST_GROUP_OCEAN, "navy"),
        ("Vegetation (Group)", ST_GROUP_VEGETATION, "darkgreen"),
        ("Snow (Group)", ST_GROUP_SNOW, "magenta"),
    ]
    hist_datas = []

    for group in tqdm(groups):
        name, surface_types, color = group
        flag_values = [idx_st + 1 for idx_st, st in enumerate(ST_COLUMNS) if st in surface_types]

        df_to_use = filter_surface_type(df, flag_values)
        df_to_use = df_to_use.filter(df_to_use[COLUMN_COUNT].is_not_null())

        hist_datas.append(HistogramData(data=df_to_use[["x", "y"]], weight=df_to_use[COLUMN_COUNT], title=name,
                                        alpha=0.5, cmap=None, color=color))

    dir_path = pathlib.Path("images") / df_path.parent.name
    dir_path.mkdir(parents=True, exist_ok=True)

    suffix = f"{min_dist}_{n_neighbors}" + ("_log" if use_log_norm else "")
    file_path = dir_path / f"umap{suffix}.png"

    plot_histograms2d(hist_datas, file_path, bins=200, use_log_norm=use_log_norm)


def main():
    parser = argparse.ArgumentParser(description="Run UMAP on data")

    parser.add_argument("--transform", "-t", default="default", choices=["default", "pd", "ratio"],
                        help="Type of transformation performed on data")
    parser.add_argument("--min-dist", "-d", default=0.8, type=float,
                        help="Minimum distance between points in the embedding space")
    parser.add_argument("--n-neighbors", "-n", default=40, type=int,
                        help="Number of nearest neighbors")

    args = parser.parse_args()
    path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform / "final.parquet"
    umap(path, args.n_neighbors, args.min_dist)


if __name__ == "__main__":
    main()
