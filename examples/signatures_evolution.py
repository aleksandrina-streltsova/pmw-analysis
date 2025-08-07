"""
Example of analyzing signatures that have appeared for the first time later than others.
"""
import pathlib

import faiss
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pycolorbar
import pycolorbar.norm
import pyproj
import xarray as xr
from gpm.bucket import LonLatPartitioning
from gpm.dataset.crs import set_dataset_crs
from gpm.visualization.plot import _sanitize_cartopy_plot_kwargs
from pycolorbar import get_plot_kwargs
from tqdm import tqdm

from pmw_analysis.constants import PMW_ANALYSIS_DIR, COLUMN_COUNT, TC_COLUMNS, COLUMN_OCCURRENCE, \
    AGG_OFF_COLUMNS, SAVEFIG_FLAG, VARIABLE_SURFACE_TYPE_INDEX, COLUMN_SUFFIX_QUANT
from pmw_analysis.quantization.dataframe_polars import merge_quantized_pmw_features, \
    get_uncertainties_dict
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.polars import take_k_sorted

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
    index.add(array_all)

    # divide query into m parts to estimate total time more easily
    m = 100
    size = (len(array_k) + m - 1) // m
    for i in tqdm(range(m)):
        l = i * size
        r = min((i + 1) * size, len(array_k))
        distances[l: r] = np.sqrt(index.search(r - l, array_k[l: r], k=2)[0][:, 1])

    return distances


def main():
    df_dir_path = pathlib.Path(PMW_ANALYSIS_DIR) / "no_sun_glint"
    quant_columns = TC_COLUMNS

    images_dir = pathlib.Path("images") / "map" /  df_dir_path.name
    images_dir.mkdir(parents=True, exist_ok=True)

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

    # TODO: replace `unique` with `final` or `newest` for consistency
    df_quantized = pl.read_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "unique.parquet")
    df_k = pl.read_parquet(df_dir_path / "final_k.parquet")
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

    m_occurrences = 1
    quant_columns_with_suffix = [f"{col}{COLUMN_SUFFIX_QUANT}" for col in quant_columns]
    df_k_quant_m = df_k.select(quant_columns_with_suffix).group_by(quant_columns_with_suffix).agg(pl.len().alias(COLUMN_COUNT))
    df_k_quant_m = df_k_quant_m.filter(pl.col(COLUMN_COUNT) >= m_occurrences)

    df_k_m = df_k.join(df_k_quant_m, on=quant_columns_with_suffix, how="inner")

    feature_columns = TC_COLUMNS
    quality_flag_columns = [
        # 'Quality_LF',
        # 'Quality_HF',
        'L1CqualityFlag',
        'qualityFlag',
    ]
    sun_glint_angle_columns = ["sunGlintAngle_LF", "sunGlintAngle_HF"]

    # extent = [-73, -11, 59, 83] # Greenland
    extent = [-180, 180, -70, 70]

    partitioning = LonLatPartitioning(size=0.5, extent=extent)

    df_with_bin_k_m = partitioning.add_labels(df_k_m, x="lon", y="lat")
    df_with_bin_k_m = partitioning.add_centroids(df_with_bin_k_m, x="lon", y="lat", x_coord="lon_bin", y_coord="lat_bin")

    get_mode_col = lambda col: f"{col}_mode"

    expressions = (
            [pl.col(col).mean() for col in feature_columns] +
            [pl.col(feature_columns[0]).count().alias(COLUMN_COUNT)] +
            [pl.col(quality_flag_columns)] +
            [pl.col(flag_col).mode().first().alias(get_mode_col(flag_col)) for flag_col in quality_flag_columns] +
            [pl.col(VARIABLE_SURFACE_TYPE_INDEX).mode().first()] +
            [pl.col(col).eq(-88).sum() / pl.len() for col in sun_glint_angle_columns]
    )

    texts = (
            {col: "Mean of bin values" for col in feature_columns} |
            {COLUMN_COUNT: "Count of bin values"} |
            {VARIABLE_SURFACE_TYPE_INDEX: "Mode of bin values"} |
            {col: "0 if 0 is present in bin, otherwise, mode" for col in quality_flag_columns} |
            {col: "Share of values equal to -88 in bin" for col in sun_glint_angle_columns}
    )

    grouped_df = df_with_bin_k_m.group_by(partitioning.levels)
    df_agg = grouped_df.agg(*expressions)

    df_agg = df_agg.with_columns([
        pl.when(pl.col(flag_col).list.contains(0)).then(0).otherwise(pl.col(get_mode_col(flag_col))).alias(flag_col)
        for flag_col in quality_flag_columns
    ]).drop([get_mode_col(flag_col) for flag_col in quality_flag_columns])

    ds = partitioning.to_xarray(df_agg, spatial_coords=("lon_bin", "lat_bin"))
    ds = ds.rename({"lon_bin": "longitude", "lat_bin": "latitude"})

    fig_kwargs = {"figsize": (17, 7), "dpi": 300}

    crs = pyproj.CRS.from_epsg(4326)
    ds: xr.Dataset = set_dataset_crs(ds, crs=crs)

    m_occurrences_text = "" if m_occurrences == 1 else f"; Signature occurred at least {m_occurrences} times."
    for i, var in enumerate(ds.data_vars):
        # if i >= 2:
        #     break
        plot_kwargs, cbar_kwargs = _get_plot_kwargs_for_variable(ds, var)

        p = ds[var].gpm.plot_map(x="longitude", y="latitude",
                                 **plot_kwargs,
                                 fig_kwargs=fig_kwargs,
                                 cbar_kwargs=cbar_kwargs)
        p.axes.set_title(f"{var}\n{texts[var]}{m_occurrences_text}")
        p.set_extent(extent)

        if SAVEFIG_FLAG:
            plt.savefig(images_dir / f"{var}_{m_occurrences}.png")
        plt.show()


def _get_plot_kwargs_for_variable(ds, var):
    if var in TC_COLUMNS:
        plot_kwargs, cbar_kwargs = get_plot_kwargs("brightness_temperature")
        # TODO: why is `alpha_bad = 0.5` if all the plots are with `alpha_bad = 0.0`?
        return _sanitize_cartopy_plot_kwargs(plot_kwargs), cbar_kwargs

    if var in pycolorbar.colorbars.names:
        return get_plot_kwargs(var)

    plot_kwargs = {}
    cbar_kwargs = None

    unique_values = np.unique(ds[var].values)
    unique_values = unique_values[~np.isnan(unique_values)]
    count_unique_values = len(unique_values)

    if count_unique_values < 10:
        if np.all(np.isclose(unique_values % 1, 0)):
            unique_values = unique_values.astype(int)
        if count_unique_values > 1:
            norm = pycolorbar.norm.CategoryNorm({i: str(v) for i, v in enumerate(unique_values)})
            plot_kwargs = {"norm": norm}

    return plot_kwargs, cbar_kwargs


if __name__ == '__main__':
    main()
