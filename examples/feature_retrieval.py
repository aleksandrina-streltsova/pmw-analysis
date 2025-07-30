"""
Example of feature retrieval.
"""
import pathlib

import gpm
import xarray as xr
from gpm.retrievals.retrieval_1b_c_pmw import retrieve_PD
from gpm.utils.pmw import get_pmw_channel
from matplotlib import pyplot as plt

from pmw_analysis.constants import (
    DIM_ALONG_TRACK, COLUMN_TIME, SAVEFIG_FLAG, SAVEFIG_DIR, PRODUCT_1C_GMI_R,
    PRODUCT_TYPE_RS, VERSION, STORAGE_GES_DISC, VARIABLE_TC, DIM_PMW, TC_COLUMNS,
)
from pmw_analysis.quantization.dataframe_pandas import segment_features_into_bins

# Define analysis time period
START_TIME = "2023-10-31 11:20:00"
END_TIME = "2023-10-31 11:35:00"

# Define product
PRODUCT = PRODUCT_1C_GMI_R
PRODUCT_TYPE = PRODUCT_TYPE_RS


def main():
    images_dir = pathlib.Path(SAVEFIG_DIR) / "plot_map"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Get available products
    gpm.available_products(product_levels="1C", start_time=START_TIME, end_time=END_TIME)

    # Download data
    gpm.download(PRODUCT, START_TIME, END_TIME, PRODUCT_TYPE, VERSION, STORAGE_GES_DISC)

    # Open data
    # TODO: chunks?
    dt = gpm.open_datatree(PRODUCT, START_TIME, END_TIME, VARIABLE_TC, product_type=PRODUCT_TYPE, chunks={})
    ds_full = dt.gpm.regrid_pmw_l1(scan_mode_reference="S1")

    # TODO: why does it work differently?
    # ds = gpm.open_dataset(product, start_time, end_time, variable, product_type=product_type, chunks={})

    # Load data over region of interest
    # extent = [-180, -170, -25, -15]
    extent = [-32, -28, -63, -59]
    point = (-29.442882537841797, -60.98043441772461)
    suffix = "max_distance"
    list_isel_dict = ds_full.gpm.get_crop_slices_by_extent(extent)
    ds = xr.concat([ds_full.isel(isel_dict) for isel_dict in list_isel_dict], dim=DIM_ALONG_TRACK)
    ds = ds.compute()

    da = ds[VARIABLE_TC]

    # Retrieve dataset of brightness temperature
    ds_tc = da.gpm.unstack_dimension(dim=DIM_PMW, suffix="_")

    # Compute polarization difference
    ds_pd = retrieve_PD(ds)  # ds.gpm.retrieve("polarization_difference") # ds.gpm.retrieve("PD")

    # # Compute frequency difference
    # ds_fd = retrieve_FD(ds)
    #
    # # Compute polarization ratio
    # ds_pr = retrieve_PR(ds)  # ds.gpm.retrieve("polarization_ratio") # ds.gpm.retrieve("PR")
    #
    # # Compute polarization correct temperature
    # # - Remove surface water signature on land brightness temperature
    # ds_pct = retrieve_PCT(ds)  # ds.gpm.retrieve("polarization_corrected_temperature")   # ds.gpm.retrieve("PCT")
    #
    # # Retrieve RGB composites
    # ds_rgb = retrieve_rgb_composites(ds)  # ds.gpm.retrieve("rgb_composites")

    # Compute frequency ratio
    # TODO: extract function
    tc_denom = "Tc_19H"
    freq_denom = tc_denom.removeprefix("Tc_")
    pmw_denom = get_pmw_channel(ds, name=freq_denom)

    dict_ratio = {name: (get_pmw_channel(ds, name=name.removeprefix("Tc_")) / pmw_denom).rename(f"{name}/{tc_denom}")
                  for name in TC_COLUMNS if name != tc_denom}
    ds_ratio = xr.merge(dict_ratio.values(), compat="minimal")

    freqs_pd = ["19", "37", "89", "165"]
    tc_columns_to_plot = TC_COLUMNS + ["Tc_183V3", "Tc_183V7"]
    pd_columns_to_plot = [f"PD_{freq}" for freq in freqs_pd]
    ratio_columns_to_plot = [f"Tc_{freq}H/Tc_19H" for freq in freqs_pd if freq != "19"]

    tc_range = (75, 275)
    pd_range = (-5, 85)
    ratio_range = (0.8, 2.5)

    base_cbar_kwargs = {"orientation": "horizontal", "pad": 0.4}
    tc_cbar_kwargs = base_cbar_kwargs
    pd_cbar_kwargs = base_cbar_kwargs
    ratio_cbar_kwargs = dict(base_cbar_kwargs, **{"label": "Brightness Temperature Ratio [1]"})

    # Plot features
    for var, ds_var, columns_to_plot, col_wrap, cbar_kwargs, var_range in [
        (VARIABLE_TC, ds_tc, tc_columns_to_plot, 3, tc_cbar_kwargs, tc_range),
        ("PD", ds_pd, pd_columns_to_plot, 2, pd_cbar_kwargs, pd_range),
        ("Ratio", ds_ratio, ratio_columns_to_plot, 2, ratio_cbar_kwargs, ratio_range)
    ]:
        n_cols = col_wrap
        n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols
        ds_var = ds_var[columns_to_plot]
        fc = ds_var.to_array(dim=var, name=var).gpm.plot_map(
            col=var,
            col_wrap=col_wrap,
            vmin=var_range[0],
            vmax=var_range[1],
            fig_kwargs={"figsize": (n_cols * 3 + 1, n_rows * 3 + 1)},
            cbar_kwargs=cbar_kwargs,
            axes_pad=(0.1, 0.4),
        )
        fc.remove_title_dimension_prefix()
        fc.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.1, wspace=0.1, hspace=0.1)
        # TODO: why is `ds.gpm.crop(extent)` not enough?
        fc.set_extent(extent)
        for axis in fc.axs.flatten():
            axis.scatter(point[0], point[1], marker="x", c="black")

        if SAVEFIG_FLAG:
            plt.savefig(images_dir / f"{var}_{suffix}.png")
        plt.show()

    # Convert Xarray Dataset to Pandas DataFrame
    df_pd = ds_pd.reset_coords(names=COLUMN_TIME).reset_coords(drop=True).to_dataframe()
    # TODO: why does PD_165 have NaN? should rows with NaNs be processed differently?
    df_pd_cut, feature_cols = segment_features_into_bins(df_pd)

    df_pd[feature_cols].hist(bins=100, figsize=(6, 6))
    plt.tight_layout()
    plt.show()

    df_pd_cut[feature_cols].hist(bins=100, figsize=(6, 6))
    plt.tight_layout()
    plt.show()

    df_pd_cut[["count"]].hist(bins=20)
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
