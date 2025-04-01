"""
Example of UMAP dimension reduction.
"""
import datetime
import pathlib

import gpm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import umap
import xarray as xr
from matplotlib.colors import LogNorm
from sklearn.preprocessing import StandardScaler

from pmw_analysis.constants import (
    SAVEFIG_FLAG, SAVEFIG_DIR, COLUMN_TIME, PRODUCT_1C_GMI_R, PRODUCT_TYPE_RS, VERSION,
    STORAGE_GES_DISC, VARIABLE_TC, DIM_PMW, DIM_ALONG_TRACK, PRODUCT_2A_GMI, VARIABLE_SURFACE_TYPE_INDEX,
)
from pmw_analysis.copypaste.utils.dataframe import compute_2d_histogram
from pmw_analysis.preprocessing import remove_missing_values
from pmw_analysis.utils.pyplot import get_surface_type_cmap

# Define product
PRODUCT_TYPE = PRODUCT_TYPE_RS

extent = [25, 35, 55, 65]
# extent = None

dates = [
    datetime.date(2015, 3, 20),
    datetime.date(2015, 6, 21),
    datetime.date(2015, 9, 23),
    datetime.date(2015, 12, 22),
]

ds_1C_list = []
ds_2A_list = []

fig_kwargs = {"figsize": (10, 4), "dpi": 300}
for idx, date in enumerate(dates):
    start_time = datetime.datetime(date.year, date.month, date.day, hour=0, minute=0, second=0)
    end_time = start_time + datetime.timedelta(days=1)

    ds_dict = {}
    for product, variable in [(PRODUCT_1C_GMI_R, VARIABLE_TC), (PRODUCT_2A_GMI, VARIABLE_SURFACE_TYPE_INDEX)]:
        # TODO: check_integrity
        gpm.download(product, start_time, end_time, PRODUCT_TYPE, VERSION, STORAGE_GES_DISC, check_integrity=False)

        dt = gpm.open_datatree(product, start_time, end_time, variable, product_type=PRODUCT_TYPE, chunks={})
        ds = dt.gpm.regrid_pmw_l1(scan_mode_reference="S1")

        if extent is not None:
            list_isel_dict = ds.gpm.get_crop_slices_by_extent(extent)
            ds = xr.concat([ds.isel(isel_dict) for isel_dict in list_isel_dict], dim=DIM_ALONG_TRACK)
        ds = ds.compute()
        ds_dict[product] = ds

    # 1C
    ds = ds_dict[PRODUCT_1C_GMI_R]
    da = ds[VARIABLE_TC]
    ds_tc = da.gpm.unstack_dimension(dim=DIM_PMW, suffix="_")
    ds_1C_list.append(ds_tc)

    vars_to_plot = ["Tc_183V3"]
    for freq in ["19", "37", "89"]:
        v = ds_tc[f"Tc_{freq}V"]
        h = ds_tc[f"Tc_{freq}H"]
        vars_to_plot.append(f"Tc_{freq}V")

        ds_tc = ds_tc.assign({f"PR_{freq}": v / h})
        vars_to_plot.append(f"PR_{freq}")

        ds_tc = ds_tc.assign({f"I_{freq}": v ** 2 + h ** 2})
        vars_to_plot.append(f"I_{freq}")

    for var_to_plot in vars_to_plot:
        da = ds_tc[var_to_plot]
        da.gpm.plot_map(add_swath_lines=False, fig_kwargs=fig_kwargs, cmap="Spectral_r")

        if SAVEFIG_FLAG:
            plt.savefig(pathlib.Path(SAVEFIG_DIR) / f"{dates[idx].year}_{dates[idx].month}_{var_to_plot}.png")
        plt.show()

    # 2A
    ds = ds_dict[PRODUCT_2A_GMI]
    ds_2A_list.append(ds)
    # TODO: where are missing values located on map?
    da: xr.DataArray = ds[VARIABLE_SURFACE_TYPE_INDEX]
    cmap, norm = get_surface_type_cmap(ds[VARIABLE_SURFACE_TYPE_INDEX].attrs["flag_meanings"])
    cbar_kwargs = {"cmap": cmap, "norm": norm}
    da.gpm.plot_map(add_swath_lines=False, cbar_kwargs=cbar_kwargs, fig_kwargs=fig_kwargs)

    if SAVEFIG_FLAG:
        plt.savefig(pathlib.Path(SAVEFIG_DIR) / f"{dates[idx].year}_{dates[idx].month}_surface_type.png")
    plt.show()

ds_list = []
for idx in range(len(dates)):
    ds = xr.merge([ds_1C_list[idx], ds_2A_list[idx]])
    ds_list.append(ds)

df_list = []
for idx, ds in enumerate(ds_list):
    df = ds.reset_coords(names=COLUMN_TIME).reset_coords(drop=True).to_dataframe()
    df_list.append(df)

df = pd.concat(df_list)

# df.to_pickle("df.pkl")
# df = pd.read_pickle("df.pkl")

# 1. Check correlation and remove highly correlated features.
columns_low = [col for col in df.columns if any(freq in col for freq in ["10", "19", "23", "37"])]
columns_high = [col for col in df.columns if any(freq in col for freq in ["89", "165", "183"])]

corr_mtx = df.drop(columns=["time"]).corr()

sns.heatmap(corr_mtx[columns_low].loc[columns_low].abs(), annot=True, vmin=0, vmax=1)
plt.show()

sns.heatmap(corr_mtx[columns_low].loc[columns_high].abs(), annot=True, vmin=0, vmax=1)
plt.show()

sns.heatmap(corr_mtx[columns_high].loc[columns_high].abs(), annot=True, vmin=0, vmax=1)
plt.show()

CORR_THR = 0.9
feature_cols_to_use = [col for idx, col in enumerate(corr_mtx.columns) if
                       not (corr_mtx[col][idx + 1:].abs() >= CORR_THR).any()]

# 2. Plot pairs of features.
df_umap = remove_missing_values(df)
for freq in ["19", "89", "165"]:
    df_umap[f"Tc_{freq}PR"] = df_umap[f"Tc_{freq}V"] / df_umap[f"Tc_{freq}H"]
    df_umap[f"Tc_{freq}I"] = df_umap[f"Tc_{freq}V"] ** 2 + df_umap[f"Tc_{freq}H"] ** 2

xy_list = []
for freq in ["19", "89", "165"]:
    X = f"Tc_{freq}PR"
    Y = f"Tc_{freq}V"
    xy_list.append((X, Y))

for var_type in ["V", "H", "PR", "I"]:
    X = f"Tc_19{var_type}"
    Y = f"Tc_89{var_type}"
    xy_list.append((X, Y))

for xfreq in ["19", "89"]:
    X = f"Tc_{xfreq}V"
    Y = "Tc_183V3"
    xy_list.append((X, Y))

surface_types = ["NaN"] + ds_2A_list[0][VARIABLE_SURFACE_TYPE_INDEX].attrs["flag_meanings"]
cmap_st, norm_st = get_surface_type_cmap(surface_types)

for x, y in xy_list:
    hist = compute_2d_histogram(df_umap, x=x, y=y, var="surfaceTypeIndex", x_bins=200, y_bins=200)

    for var_stat, cmap, norm, tick_labels, cbar_kwargs in [
        ("mode", cmap_st, norm_st, surface_types, {"ticks": range(1, len(surface_types) + 1)}),
        ("count", "rocket_r", LogNorm(), None, {}),
    ]:
        mtx = hist[f"surfaceTypeIndex_{var_stat}"].to_pandas()
        mtx.fillna(0, inplace=True)

        plt.figure(figsize=(8, 5))
        ax = sns.heatmap(mtx, cmap=cmap, norm=norm, cbar_kws=cbar_kwargs)

        if tick_labels is not None:
            cbar = ax.collections[0].colorbar
            cbar.set_ticklabels(surface_types)

        ax.invert_yaxis()
        ax.set_aspect('equal')

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        ax.set_xticklabels([f"{mtx.columns[int(i)]:.1f}" for i in xticks if 0 <= int(i) < len(mtx.columns)])
        ax.set_yticklabels([f"{mtx.index[int(i)]:.1f}" for i in yticks if 0 <= int(i) < len(mtx.index)])

        plt.tight_layout()
        if SAVEFIG_FLAG:
            plt.savefig(pathlib.Path(SAVEFIG_DIR) / "pairplots" / f"{dates[0].year}_{x}_{y}_{var_stat}.png")
        plt.show()

# 3. Run UMAP on sparse data.
df_umap_sparse = df_umap.iloc[::600]
print(len(df_umap_sparse))

columns_low_v = [col for col in columns_low if col.endswith("V")]
columns_high_v = [col for col in columns_high if col.endswith("V") or col.endswith("V3") or col.endswith("V7")]

for cols, suffix in [(columns_low_v, "low"), (columns_high_v, "high")]:
    features = df_umap_sparse[cols].values
    reducer = umap.UMAP()
    features_scaled = StandardScaler().fit_transform(features)
    embedding = reducer.fit_transform(features_scaled)

    # with open("embedding_5_30.pkl", "wb") as f:
    #     pickle.dump(reducer, f)

    embedding = embedding[::50]

    COLOR_COL = "Tc_19H"
    plt.scatter(embedding[:, 0], embedding[:, 1], c=df_umap_sparse[COLOR_COL][::50], s=3)
    plt.colorbar(label=COLOR_COL)
    if SAVEFIG_FLAG:
        plt.savefig(pathlib.Path(SAVEFIG_DIR) / f"2015_umap_sparse_{suffix}.png")
    plt.show()

df_umap.to_csv("df_umap.csv")
