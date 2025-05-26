"""
Example of feature retrieval.
"""
import pathlib

import gpm
import xarray as xr
from gpm.retrievals.retrieval_1b_c_pmw import retrieve_PD, retrieve_PR, retrieve_PCT, retrieve_rgb_composites
from matplotlib import pyplot as plt

from pmw_analysis.constants import (
    DIM_ALONG_TRACK, COLUMN_TIME, SAVEFIG_FLAG, SAVEFIG_DIR, PRODUCT_1C_GMI_R,
    PRODUCT_TYPE_RS, VERSION, STORAGE_GES_DISC, VARIABLE_TC, DIM_PMW,
)
from pmw_analysis.quantization.dataframe_pandas import segment_features_into_bins
from pmw_analysis.retrievals.retrieval_1b_c_pmw import retrieve_FD

# Define analysis time period
START_TIME = "2015-09-05 08:00:00"
END_TIME = "2015-09-05 09:00:00"

# Define product
PRODUCT = PRODUCT_1C_GMI_R
PRODUCT_TYPE = PRODUCT_TYPE_RS

# Get available products
gpm.available_products(product_levels="1C", start_time=START_TIME, end_time=END_TIME)

# Download data
gpm.download(PRODUCT, START_TIME, END_TIME, PRODUCT_TYPE, VERSION, STORAGE_GES_DISC)

# Open data
# TODO: chunks?
dt = gpm.open_datatree(PRODUCT, START_TIME, END_TIME, VARIABLE_TC, product_type=PRODUCT_TYPE, chunks={})
ds = dt.gpm.regrid_pmw_l1(scan_mode_reference="S1")

# TODO: why does it work differently?
# ds = gpm.open_dataset(product, start_time, end_time, variable, product_type=product_type, chunks={})

# Load data over region of interest
extent = [12, 16, 39, 42]
list_isel_dict = ds.gpm.get_crop_slices_by_extent(extent)
ds = xr.concat([ds.isel(isel_dict) for isel_dict in list_isel_dict], dim=DIM_ALONG_TRACK)
ds = ds.compute()

da = ds[VARIABLE_TC]

# Retrieve dataset of brightness temperature
ds_tc = da.gpm.unstack_dimension(dim=DIM_PMW, suffix="_")

# Compute polarization difference
ds_pd = retrieve_PD(ds)  # ds.gpm.retrieve("polarization_difference") # ds.gpm.retrieve("PD")

# Compute frequency difference
ds_fd = retrieve_FD(ds)

# Compute polarization ratio
ds_pr = retrieve_PR(ds)  # ds.gpm.retrieve("polarization_ratio") # ds.gpm.retrieve("PR")

# Compute polarization correct temperature
# - Remove surface water signature on land brightness temperature
ds_pct = retrieve_PCT(ds)  # ds.gpm.retrieve("polarization_corrected_temperature")   # ds.gpm.retrieve("PCT")

# Retrieve RGB composites
ds_rgb = retrieve_rgb_composites(ds)  # ds.gpm.retrieve("rgb_composites")

# Plot features
for var, ds_var in [(VARIABLE_TC, ds_tc), ("PD", ds_pd), ("FD", ds_fd)]:
    cbar_kwargs = {"orientation": "horizontal"}
    plot_kwargs = {"axes_pad": (0.1, 1)}
    fc = ds_var.to_array(dim=var, name=var).gpm.plot_map(
        col=var,
        col_wrap=4,
        # vmax=20,
        # vmin=-2,
        cbar_kwargs=cbar_kwargs,
        axes_pad=(0.1, 0.5),
    )
    fc.remove_title_dimension_prefix()
    # TODO: why is `ds.gpm.crop(extent)` not enough?
    fc.set_extent(extent)
    if SAVEFIG_FLAG:
        plt.savefig(pathlib.Path(SAVEFIG_DIR) / f"hurricane_{var}.png")
    plt.show()

# Convert xarray Dataset to pandas DataFrame
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
