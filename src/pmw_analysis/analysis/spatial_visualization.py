import pathlib

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

from pmw_analysis.constants import TC_COLUMNS, COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX, FLAG_SAVEFIG, COLUMN_LON, \
    COLUMN_LAT, COLUMN_LON_BIN, COLUMN_LAT_BIN, ArgTransform, COLUMN_SUN_GLINT_ANGLE_HF, COLUMN_SUN_GLINT_ANGLE_LF
from pmw_analysis.quantization.script import get_transformation_function


def plot_variables_on_map(df: pl.DataFrame, transform_arg: ArgTransform, images_dir: pathlib.Path,
                          title_text_suffix: str = "", file_name_suffix: str = ""):
    transform = get_transformation_function(transform_arg)

    feature_columns = transform(TC_COLUMNS)
    quality_flag_columns = [
        # 'Quality_LF',
        # 'Quality_HF',
        'L1CqualityFlag',
        'qualityFlag',
    ]
    sun_glint_angle_columns = [COLUMN_SUN_GLINT_ANGLE_LF, COLUMN_SUN_GLINT_ANGLE_HF]
    known_columns = (feature_columns + quality_flag_columns + sun_glint_angle_columns +
                     [COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX])
    columns_to_plot = [col for col in df.columns if col not in [COLUMN_LON, COLUMN_LAT]]
    # extent = [-73, -11, 59, 83] # Greenland
    extent = [-180, 180, -70, 70]

    partitioning = LonLatPartitioning(size=0.5, extent=extent)

    df_with_bin = partitioning.add_labels(df, x=COLUMN_LON, y=COLUMN_LAT)
    df_with_bin = partitioning.add_centroids(df_with_bin,
                                             x=COLUMN_LON, y=COLUMN_LAT,
                                             x_coord=COLUMN_LON_BIN, y_coord=COLUMN_LAT_BIN)

    get_mode_col = lambda col: f"{col}_mode"

    expressions = (
            {col: [pl.col(col).mean()] for col in feature_columns} |
            {col: [pl.col(col), pl.col(col).mode().first().alias(get_mode_col(col))] for col in quality_flag_columns} |
            {COLUMN_COUNT: [pl.col(feature_columns[0]).count().alias(COLUMN_COUNT)]} |
            {VARIABLE_SURFACE_TYPE_INDEX: [pl.col(VARIABLE_SURFACE_TYPE_INDEX).mode().first()]} |
            {col: [pl.col(col).eq(-88).sum() / pl.len()] for col in sun_glint_angle_columns}
    )

    texts = (
            {col: "Mean of bin values" for col in feature_columns} |
            {COLUMN_COUNT: "Count of bin values"} |
            {VARIABLE_SURFACE_TYPE_INDEX: "Mode of bin values"} |
            {col: "0 if 0 is present in bin, otherwise, mode" for col in quality_flag_columns} |
            {col: "Share of values equal to -88 in bin" for col in sun_glint_angle_columns}
    )

    for col in columns_to_plot:
        if col in known_columns:
            continue
        value_counts = df[col].value_counts().filter(pl.col(col).is_not_null())
        if value_counts.height < 10:
            expressions[col] = [pl.col(col).mode().first()]
            texts[col] = "Mode of bin values"
        else:
            expressions[col] = [pl.col(col).mean()]
            texts[col] = "Mean of bin values"

    grouped_df = df_with_bin.group_by(partitioning.levels)
    df_agg = grouped_df.agg([expr for col in columns_to_plot for expr in expressions[col]])

    df_agg = df_agg.with_columns([
        pl.when(pl.col(flag_col).list.contains(0)).then(0).otherwise(pl.col(get_mode_col(flag_col))).alias(flag_col)
        for flag_col in quality_flag_columns if flag_col in columns_to_plot
    ]).drop([get_mode_col(flag_col) for flag_col in quality_flag_columns if flag_col in columns_to_plot])

    ds = partitioning.to_xarray(df_agg, spatial_coords=("lon_bin", "lat_bin"))
    ds = ds.rename({"lon_bin": "longitude", "lat_bin": "latitude"})

    fig_kwargs = {"figsize": (17, 7), "dpi": 300}

    crs = pyproj.CRS.from_epsg(4326)
    ds: xr.Dataset = set_dataset_crs(ds, crs=crs)

    for i, var in enumerate(ds.data_vars):
        # if i >= 2:
        #     break
        plot_kwargs, cbar_kwargs = _get_plot_kwargs_for_variable(ds, var)

        p = ds[var].gpm.plot_map(x="longitude", y="latitude",
                                 **plot_kwargs,
                                 fig_kwargs=fig_kwargs,
                                 cbar_kwargs=cbar_kwargs)
        p.axes.set_title(f"{var}\n{texts[var]}{title_text_suffix}")
        p.set_extent(extent)

        if FLAG_SAVEFIG:
            plt.savefig(images_dir / f"{var}{file_name_suffix}.png")
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
            if count_unique_values > 1:
                norm = pycolorbar.norm.CategoryNorm({int(v): str(v) for _, v in enumerate(unique_values)})
                plot_kwargs = {"norm": norm}

    return plot_kwargs, cbar_kwargs
