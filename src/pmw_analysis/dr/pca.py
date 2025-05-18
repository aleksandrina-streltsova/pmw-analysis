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
from pmw_analysis.utils.pyplot import get_surface_type_cmap, plot_histograms2d

N_BINS = 200

def pca(df_path: pathlib.Path, use_weights: bool, use_log_norm: bool):
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
    if use_weights:
        pca = WPCA(n_components=2)
        fs_reduced = pca.fit_transform(fs_scaled, sample_weight=weight.to_numpy())
    else:
        pca = PCA(n_components=2)
        fs_reduced = pca.fit_transform(fs_scaled)

    dir_path = pathlib.Path("images") / df_path.parent.name
    dir_path.mkdir(parents=True, exist_ok=True)
    file_path = dir_path / f"pca{"_log" if use_log_norm else ""}{"_w" if use_weights else ""}.png"
    file_path_count = dir_path / f"pca_count{"_log" if use_log_norm else ""}{"_w" if use_weights else ""}.png"

    plot_histograms2d([fs_reduced], [df[COLUMN_COUNT]], ["All surfaces"], file_path_count,
                      bins=N_BINS, cmaps=["rocket_r"], use_log_norm=use_log_norm)

    cmap_st, _ = get_surface_type_cmap(['NaN'] + ST_COLUMNS)

    datas = []
    weights = []
    counts = np.zeros(len(ST_COLUMNS))

    for idx_st in tqdm(range(len(ST_COLUMNS))):
        flag_value = idx_st + 1

        df_to_use = filter_surface_type(df, flag_value)
        df_to_use = df_to_use.filter(df_to_use[COLUMN_COUNT].is_not_null())

        features = df_to_use[TC_COLUMNS]
        fs_reduced = pca.transform(scaler.transform(features))

        counts[idx_st] = df_to_use[COLUMN_COUNT].cast(pl.Int64).sum()

        datas.append(fs_reduced)
        weights.append(df_to_use[COLUMN_COUNT])

    colors = [cmap_st.colors[idx_st + 1] for idx_st in range(len(ST_COLUMNS))]
    alphas = np.log(counts)
    alphas = alphas * 0.8 / alphas.max()
    titles = ST_COLUMNS

    plot_histograms2d(datas, weights, titles, file_path,
                      colors=colors, bins=N_BINS, use_log_norm=use_log_norm, alpha=alphas)


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