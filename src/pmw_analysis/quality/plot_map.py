import argparse
import pathlib
from collections import defaultdict
from typing import Callable, List

import gpm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import polars as pl
import xarray as xr
from tqdm import tqdm

from pmw_analysis.constants import PMW_ANALYSIS_DIR, TC_COLUMNS, PRODUCT_1C_GMI_R, PRODUCT_TYPE_RS, VERSION, \
    STORAGE_GES_DISC, VARIABLE_TC, COLUMN_CLUSTER, PRODUCT_2A_GMI, VARIABLE_SURFACE_TYPE_INDEX, ST_GROUP_OCEAN, \
    ST_GROUP_VEGETATION, ST_GROUP_SNOW, ST_COLUMNS
from pmw_analysis.dr.clusterization import ClusterModel
from pmw_analysis.quantization.script import get_transformation_function


# TODO: rewrite xr <-> pl
def _xr_to_pl(da: xr.DataArray) -> pl.DataFrame:
    da_flat = da.stack(points=("cross_track", "along_track"))

    df = da_flat.to_dataframe(name='Tc').reset_index(level="pmw_frequency")
    df = df.pivot(index=["cross_track", "along_track"], columns="pmw_frequency", values="Tc")
    df = pl.from_pandas(df, include_index=True)
    df = df.rename({col: f"Tc_{col}" for col in df.columns if f"Tc_{col}" in TC_COLUMNS})

    return df


def _pl_to_xr(df: pd.DataFrame, coords: dict):
    cross_track = df["cross_track"]
    along_track = df["along_track"]

    da = xr.DataArray(
        data=df[COLUMN_CLUSTER].reshape((cross_track.max() + 1, along_track.max() + 1)),
        dims=("cross_track", "along_track"),
        coords=coords,
        name=COLUMN_CLUSTER
    )
    return da


def _get_da_example() -> List[xr.DataArray]:
    start_time = "2021-08-15 15:00:00"
    end_time = "2021-08-15 16:00:00"

    das = []
    for variable, product in [(VARIABLE_TC, PRODUCT_1C_GMI_R), (VARIABLE_SURFACE_TYPE_INDEX, PRODUCT_2A_GMI)]:
        product = product
        product_type = PRODUCT_TYPE_RS

        gpm.download(product, start_time, end_time, product_type, VERSION, STORAGE_GES_DISC)

        dt = gpm.open_datatree(product, start_time, end_time, variable, product_type=product_type, chunks={})
        ds = dt.gpm.regrid_pmw_l1(scan_mode_reference="S1")
        da: xr.DataArray = ds[variable]

        das.append(da)

    return das


def plot_map(model_path: pathlib.Path, transform: Callable):
    da, da_ref = _get_da_example()
    df = _xr_to_pl(da)

    feature_columns = transform(TC_COLUMNS)
    df = transform(df)
    mask_nan =  df.select(pl.any_horizontal(pl.col(feature_columns).is_nan().alias("has_nan")))
    model = ClusterModel.load(model_path)

    labels = model.predict(df[feature_columns].filter(~mask_nan["has_nan"]))
    df = df.with_columns(pl.lit(-1).alias(COLUMN_CLUSTER))
    df = df.with_columns(pl.when(mask_nan).then(pl.col(COLUMN_CLUSTER)).otherwise(labels).name.keep())

    # TODO: pass coords properly
    coords = {"lon": da.lon, "lat": da.lat, "along_track": da.along_track, "cross_track": da.cross_track}
    da_labels = _pl_to_xr(df, coords)

    cbar_kwargs = {"orientation": "horizontal"}
    fig_kwargs = {"figsize": (20, 10)}
    da_labels.gpm.plot_map(
        fig_kwargs=fig_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_swath_lines=False,
        cmap="Set1",
    )
    plt.show()

    # TODO: extract method
    groups = [
        ("Ocean (Group)", ST_GROUP_OCEAN, "cornflowerblue"),
        ("Vegetation (Group)", ST_GROUP_VEGETATION, "mediumseagreen"),
        ("Snow (Group)", ST_GROUP_SNOW, "orchid"),
    ]

    missing_surface_color = "black"
    flag_to_color = defaultdict(lambda: missing_surface_color)
    for group in tqdm(groups):
        _, surface_types, color = group
        for idx_st, st in enumerate(ST_COLUMNS):
            if st not in surface_types:
                continue
            flag_to_color[idx_st + 1] = color
    cmap_ref = mcolors.ListedColormap([flag_to_color[flag_value] for flag_value in range(1, len(ST_COLUMNS) + 1)])

    cbar_kwargs = {"orientation": "horizontal"}
    fig_kwargs = {"figsize": (20, 10)}
    da_ref.gpm.plot_map(
        fig_kwargs=fig_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_swath_lines=False,
        cmap=cmap_ref,
    )
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot clusterization results on map")

    parser.add_argument("--transform", "-t", default="default",
                        choices=["default", "pd", "ratio", "partial", "v1", "v2"],
                        help="Type of transformation performed on data")
    parser.add_argument("-dr", "--reduction", choices=["pca", "umap"])
    parser.add_argument("-c", "--clusterization", choices=["kmeans", "hdbscan"])

    args = parser.parse_args()
    model_path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform / f"{args.reduction}_{args.clusterization}.pkl"
    transform = get_transformation_function(args.transform)

    plot_map(model_path, transform)


if __name__ == '__main__':
    main()