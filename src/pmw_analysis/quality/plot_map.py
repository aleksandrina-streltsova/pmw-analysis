"""
This module contains functions for plotting clustering results on map.
"""
import pathlib
from typing import List, Callable

import configargparse
import gpm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import xarray as xr

from pmw_analysis.constants import PRODUCT_1C_GMI_R, PRODUCT_TYPE_RS, VERSION, \
    STORAGE_GES_DISC, VARIABLE_TC, COLUMN_CLUSTER, PRODUCT_2A_GMI, TC_COLUMNS, VARIABLE_SURFACE_TYPE_INDEX, ST_COLUMNS, \
    PMW_ANALYSIS_DIR, ArgTransform, ArgDimensionalityReduction, ArgClustering
from pmw_analysis.analysis.clustering import ClusterModel
from pmw_analysis.copypaste.utils.cli import EnumAction
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.pyplot import get_surface_type_cmap


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
    start_time = "2021-02-15 14:00:00"
    end_time = "2021-02-17 14:00:00"

    das = []
    for variable, product in [(VARIABLE_TC, PRODUCT_1C_GMI_R), (VARIABLE_SURFACE_TYPE_INDEX, PRODUCT_2A_GMI)]:
        product_type = PRODUCT_TYPE_RS

        gpm.download(product, start_time, end_time, product_type, VERSION, STORAGE_GES_DISC)

        dt = gpm.open_datatree(product, start_time, end_time, variable, product_type=product_type, chunks={})
        ds = dt.gpm.regrid_pmw_l1(scan_mode_reference="S1")
        da: xr.DataArray = ds[variable]

        das.append(da)

    return das


def plot_map(model_path: pathlib.Path, transform: Callable):
    """
    Plot clustering results and reference classes on map.
    """
    da, da_ref = _get_da_example()
    df = _xr_to_pl(da)

    feature_columns = transform(TC_COLUMNS)
    df = transform(df)
    mask_nan = df.select(pl.any_horizontal(pl.col(feature_columns).is_nan().alias("has_nan")))
    model = ClusterModel.load(model_path)

    labels = -1 * np.ones(len(df), dtype=int)
    labels[(~mask_nan["has_nan"]).to_numpy().astype(bool)] = model.predict(
        df[feature_columns].filter(~mask_nan["has_nan"]))

    labels = pl.Series(COLUMN_CLUSTER, labels)
    df = df.insert_column(-1, labels.alias(COLUMN_CLUSTER))

    # TODO: pass coords properly
    coords = {"lon": da.lon, "lat": da.lat, "along_track": da.along_track, "cross_track": da.cross_track}
    da_labels = _pl_to_xr(df, coords)

    #### Plotting results ####
    dir_path = pathlib.Path("images") / model_path.parent.name
    dir_path.mkdir(parents=True, exist_ok=True)

    if labels.max() + 1 < 5:
        cmap = mcolors.ListedColormap(["black", "moccasin", "orchid", "lightcoral", "lightcyan", "cornflowerblue"])
        # cmap = mcolors.ListedColormap(["cornflowerblue", "orchid", "lightcoral", "lightcyan", "moccasin"])
    else:
        cmap = "Set3"

    cbar_kwargs = {"orientation": "horizontal"}
    fig_kwargs = {"figsize": (20, 10)}
    da_labels.gpm.plot_map(
        fig_kwargs=fig_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_swath_lines=False,
        cmap=cmap,
    )

    suffix = "_winter"
    file_path = dir_path / f"map_{model_path.name.removesuffix(".pkl")}{suffix}.png"
    plt.savefig(file_path)
    plt.show()

    # TODO: extract method
    # groups = [
    #     ("Ocean (Group)", ST_GROUP_OCEAN, "cornflowerblue"),
    #     ("Vegetation (Group)", ST_GROUP_VEGETATION, "mediumseagreen"),
    #     ("Snow (Group)", ST_GROUP_SNOW, "orchid"),
    #     ("Edges (Group)", ST_GROUP_EDGES, "lightcoral"),
    #     ("Misc (Group)", ST_GROUP_MISC, "moccasin"),
    # ]
    #
    # missing_surface_color = "black"
    # flag_to_color = defaultdict(lambda: missing_surface_color)
    # for group in tqdm(groups):
    #     _, surface_types, color = group
    #     for idx_st, st in enumerate(ST_COLUMNS):
    #         if st not in surface_types:
    #             continue
    #         flag_to_color[idx_st + 1] = color
    # cmap_ref = mcolors.ListedColormap([flag_to_color[flag_value] for flag_value in range(1, len(ST_COLUMNS) + 1)])

    cmap_ref, _ = get_surface_type_cmap(ST_COLUMNS)
    cbar_kwargs = {"orientation": "horizontal"}
    fig_kwargs = {"figsize": (20, 12)}
    plot = da_ref.gpm.plot_map(
        fig_kwargs=fig_kwargs,
        cbar_kwargs=cbar_kwargs,
        add_swath_lines=False,
        cmap=cmap_ref,
    )
    plot.colorbar.ax.tick_params(labelrotation=90)
    file_path = dir_path / f"map_ref{suffix}.png"
    plt.savefig(file_path)
    plt.show()


def main():
    parser = configargparse.ArgumentParser(config_arg_is_required=True, args_for_setting_config_path=["--config"],
                                           description="Plot clustering results on map")

    parser.add_argument("--transform", default=ArgTransform.DEFAULT,
                        type=ArgTransform, action=EnumAction,
                        help="Type of transformation performed on data")
    parser.add_argument("--reduction", type=ArgDimensionalityReduction, action=EnumAction)
    parser.add_argument("--clustering", type=ArgClustering, action=EnumAction)

    args = parser.parse_args()
    model_path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform.value / f"{args.reduction}_{args.clustering}.pkl"
    transform = get_transformation_function(args.transform)

    plot_map(model_path, transform)


if __name__ == '__main__':
    main()
