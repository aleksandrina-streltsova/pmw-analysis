"""
Script for running PCA on quantized transformed data.
"""
import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pmw_analysis.constants import COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX, TC_COLUMNS, ST_COLUMNS, PMW_ANALYSIS_DIR
from pmw_analysis.copypaste.wpca import WPCA
from pmw_analysis.quantization.polars import filter_surface_type
from pmw_analysis.utils.pyplot import get_surface_type_cmap, plot_histograms2d, HistogramData

N_BINS = 200


def pca(df_path: pathlib.Path, use_weights: bool, use_log_norm: bool):
    """
    Perform PCA on the specified dataset and
    plot the first two principal components of features for each surface type.

    Parameters
    ----------
    df_path : pathlib.Path
        The path to the input parquet file containing the dataset to process.
    use_weights: bool
         If True, point counts are used to weigh points when running KMeans++ and PCA.
    use_log_norm: bool
         If True, principal components are colored with logarithmic colormap normalization.
    """
    df_merged: pl.DataFrame = pl.read_parquet(df_path)

    df_count = df_merged[COLUMN_COUNT].log()
    hist: pl.DataFrame = df_count.hist(bin_count=50)

    plt.stairs(hist["count"][1:], np.exp(hist["breakpoint"]), fill=True)
    plt.xscale("log")
    plt.show()

    df = df_merged[TC_COLUMNS + [VARIABLE_SURFACE_TYPE_INDEX, COLUMN_COUNT]]
    # TODO: should we process NaNs differently?
    df = df.drop_nans()

    weight = df[COLUMN_COUNT] if use_weights else pl.ones(len(df), eager=True)
    features = df[TC_COLUMNS]

    scaler = StandardScaler()
    fs_scaled = scaler.fit_transform(features, sample_weight=weight)
    fs_reduced, reducer = pca_fit_transform(fs_scaled, weight if use_weights else None)

    dir_path = pathlib.Path("images") / df_path.parent.name
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"pca{"_log" if use_log_norm else ""}{"_w" if use_weights else ""}.png"
    file_path_count = dir_path / f"pca_count{"_log" if use_log_norm else ""}{"_w" if use_weights else ""}.png"

    hist_data = HistogramData(data=df[TC_COLUMNS], weight=df[COLUMN_COUNT], title="All surfaces", alpha=1.0,
                              cmap = "rocket_r", color=None)

    plot_histograms2d([hist_data], file_path_count, bins=N_BINS, use_log_norm=use_log_norm)

    cmap_st, _ = get_surface_type_cmap(['NaN'] + ST_COLUMNS)

    hist_datas = []
    counts = np.zeros(len(ST_COLUMNS))

    for idx_st in tqdm(range(len(ST_COLUMNS))):
        flag_value = idx_st + 1

        df_to_use = filter_surface_type(df, flag_value)
        df_to_use = df_to_use.filter(df_to_use[COLUMN_COUNT].is_not_null())

        features = df_to_use[TC_COLUMNS]
        fs_reduced = reducer.transform(scaler.transform(features))

        counts[idx_st] = df_to_use[COLUMN_COUNT].cast(pl.Int64).sum()

        hist_datas.append(HistogramData(data=fs_reduced, weight=df_to_use[COLUMN_COUNT], title=ST_COLUMNS[idx_st],
                                        alpha=1.0, cmap=cmap_st.colors[idx_st + 1], color=None))

    alphas = np.log(counts)
    alphas = alphas * 0.8 / alphas.max()
    for i, hist_data in enumerate(hist_datas):
        hist_datas.alpha = alphas[i]

    plot_histograms2d(hist_datas, path=file_path, bins=N_BINS, use_log_norm=use_log_norm)


def pca_fit_transform(data, weight, n_components=2):
    """
    Perform dimensionality reduction on the input data using PCA or Weighted PCA.
    """
    if weight is not None:
        reducer = WPCA(n_components)
        fs_reduced = reducer.fit_transform(data, sample_weight=weight.to_numpy())
    else:
        reducer = PCA(n_components)
        fs_reduced = reducer.fit_transform(data)
    return fs_reduced, reducer


def main():
    parser = argparse.ArgumentParser(description="Run KMeans++ on data and plot results using PCA for visualization")

    parser.add_argument("-w", action="store_true", help="Use weighted PCA")
    parser.add_argument("-l", action="store_true", help="Use Log Norm for visualization")
    parser.add_argument("--transform", "-t", default="default", choices=["default", "pd", "ratio"],
                        help="Type of transformation performed on data")

    args = parser.parse_args()
    path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform / "final.parquet"

    pca(path, args.w, args.l)


if __name__ == "__main__":
    main()
