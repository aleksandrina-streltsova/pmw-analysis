"""
Example of analyzing signatures that have appeared for the first time later than others.
"""
import pathlib

import faiss
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyproj
import xarray as xr
from gpm.bucket import LonLatPartitioning
from gpm.dataset.crs import set_dataset_crs
from tqdm import tqdm

from pmw_analysis.constants import PMW_ANALYSIS_DIR, COLUMN_COUNT, TC_COLUMNS, COLUMN_OCCURRENCE, \
    AGG_OFF_COLUMNS, SAVEFIG_FLAG
from pmw_analysis.utils.polars import take_k_sorted
from pmw_analysis.quantization.dataframe_polars import merge_quantized_pmw_features, \
    get_uncertainties_dict
from pmw_analysis.quantization.script import get_transformation_function

K = 100000


def _calculate_nn_distances(df_all: pl.DataFrame, df_k: pl.DataFrame):
    transform = get_transformation_function("default")
    quant_columns = transform(TC_COLUMNS)
    unc_dict = {col: 10 * unc for col, unc in transform(get_uncertainties_dict(TC_COLUMNS)).items()}

    def make_discrete(df: pl.DataFrame):
        df_result = df.select(quant_columns)
        df_result = df_result.with_columns([(pl.col(quant_col) / unc_dict[quant_col]).round().cast(pl.UInt16)
                                            for quant_col in quant_columns])
        return df_result

    array_all = make_discrete(df_all).to_numpy()
    array_k = make_discrete(df_k).to_numpy()
    distances = np.zeros(len(array_k))

    index = faiss.IndexFlatL2(array_all.shape[1])
    index.add(len(array_all), array_all)

    # divide query into m parts to estimate total time more easily
    m = 100
    size = (len(array_k) + m - 1) // m
    for i in tqdm(range(m)):
        l = i * size
        r = min((i + 1) * size, len(array_k))
        distances[l: r] = np.sqrt(index.search(r - l, array_k[l: r], k=2)[0][:, 1])

    return distances


def main():
    quant_columns = TC_COLUMNS

    images_dir = pathlib.Path("images/k_unique")
    images_dir.mkdir(exist_ok=True, parents=True)

    lfs = []
    dfs = []
    for i in range(6):
        lf_path = pathlib.Path(f"/ltenas8/data/PMW_Analysis/tmp/default/level0/level1/level2/level3/{i}.parquet")
        lf = pl.scan_parquet(lf_path)
        lfs.append(lf)
        lf = lf.drop(AGG_OFF_COLUMNS)
        df = lf.collect()
        dfs.append(df)

    df_quantized = merge_quantized_pmw_features(dfs, quant_columns)
    df_quantized_k = take_k_sorted(df_quantized, COLUMN_OCCURRENCE, K, COLUMN_COUNT, descending=True)

    # 1. Analyze quantized features' ranges
    df_quantized_min = df_quantized.select(pl.min(quant_columns))
    df_quantized_max = df_quantized.select(pl.max(quant_columns))

    n_rows_with_extrema = df_quantized_k.filter(pl.any_horizontal([
        (pl.col(quant_col) == df_quantized_min[quant_col].item())
        .or_(pl.col(quant_col) == df_quantized_max[quant_col].item())
        for quant_col in quant_columns
    ]))[COLUMN_COUNT].sum()

    print(f"{n_rows_with_extrema} / {df_quantized_k[COLUMN_COUNT].sum()} "
          f"observations contain extrema of some quantized features")

    # 2. Check the distance to the closest neighbor
    distances = _calculate_nn_distances(df_quantized, df_quantized_k)
    np.save(pathlib.Path(PMW_ANALYSIS_DIR) / "unique_k_nn_distances.npy", distances)
    distances = np.load(pathlib.Path(PMW_ANALYSIS_DIR) / "unique_k_nn_distances.npy")

    print(f"{(distances == 1).sum()} / {len(distances)} = {100 * (distances == 1).sum() / len(distances):.2f}% "
          f"observations have their closest neighbor at distance 1")
    plt.hist(distances ** 2, bins=16)
    plt.yscale("log")
    plt.grid()
    if SAVEFIG_FLAG:
        plt.savefig(images_dir / "hist_nn_distances.png")
    plt.show()

    df_struct = pl.struct(df_quantized_k.select(quant_columns))

    dfs_filtered = []
    for lf in tqdm(lfs):
        lf_filtered = lf.filter(pl.struct(quant_columns).is_in(df_struct.implode()))
        df_filtered = lf_filtered.collect()

        dfs_filtered.append(df_filtered)

    df = merge_quantized_pmw_features(dfs_filtered, quant_columns, AGG_OFF_COLUMNS)

    df_quantized.write_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "unique.parquet")
    df.write_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "unique_k.parquet")

    df_quantized = pl.read_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "unique.parquet")
    df = pl.read_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "unique_k.parquet")
    df.select(["lon", "lat", "qualityFlag"]).row(distances.argmax())

    ###################
    # lf = pl.scan_parqueet(pathlib.Path(PMW_ANALYSIS_DIR) / "tmp" / "default" / "2018" / "1" / "final.parquet")
    # lf = lf.with_columns(pl.col("time").list.min().alias("time_min"))
    # lf = lf.sort("time_min", descending=True)
    #
    #
    # lf = lf.with_columns(pl.col(COLUMN_COUNT).cum_sum().alias(column_count_cumsum))
    # df = lf.filter(pl.col(column_count_cumsum) <= K).collect(engine="streaming")
    ###################

    df_exploded = df.explode(AGG_OFF_COLUMNS)

    # extent = [-73, -11, 59, 83] # Greenland
    extent = [-180, 180, -70, 70]

    partitioning = LonLatPartitioning(size=0.5, extent=extent)

    df_exploded = partitioning.add_labels(df_exploded, x="lon", y="lat")
    df_exploded = partitioning.add_centroids(df_exploded, x="lon", y="lat", x_coord="lon_bin", y_coord="lat_bin")

    list_variables = TC_COLUMNS
    list_expressions = (
            [pl.col(variable).mean().name.prefix("mean_") for variable in list_variables] +
            [pl.col(list_variables[0]).count().alias("count")]
    )

    grouped_df = df_exploded.group_by(partitioning.levels)
    df_agg = grouped_df.agg(*list_expressions)

    ds = partitioning.to_xarray(df_agg, spatial_coords=("lon_bin", "lat_bin"))
    ds = ds.rename({"lon_bin": "longitude", "lat_bin": "latitude"})

    fig_kwargs = {"figsize": (15, 7), "dpi": 300}
    cbar_kwargs = {"extend": "both",
                   "extendfrac": 0.05,
                   "label": "Brightness Temperature [K]"}

    crs = pyproj.CRS.from_epsg(4326)
    ds: xr.Dataset = set_dataset_crs(ds, crs=crs)

    for var in ds.data_vars:
        p = ds[var].gpm.plot_map(x="longitude", y="latitude", cmap="Spectral_r",
                                 fig_kwargs=fig_kwargs,
                                 cbar_kwargs=cbar_kwargs)
        p.axes.set_title(var)
        p.set_extent(extent)
        if SAVEFIG_FLAG:
            plt.savefig(images_dir / f"{var}.png")
        plt.show()


if __name__ == '__main__':
    main()
