import pathlib

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from pmw_analysis.constants import PMW_ANALYSIS_DIR, COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX, STRUCT_FIELD_COUNT
from pmw_analysis.decomposition import WPCA
from pmw_analysis.preprocessing_polars import filter_surface_type
from pmw_analysis.utils.pyplot import get_surface_type_cmap

def plot_histogram2d(data, ax, vmin=None, vmax=None, cmap="rocket_r", norm=None, alpha=1.0, bins=200, title=None):
    hist = np.histogram2d(data[:, 0], data[:, 1], bins=bins)[0]
    sns.heatmap(hist, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, annot=False, ax=ax, alpha=alpha, cbar=False)
    ax.set_title(title)

SUFFIX = ""

df_merged: pl.DataFrame = pl.read_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "merged" / "final.parquet")

# 0. Plot histogram of counts.
df_count = df_merged[COLUMN_COUNT].log()
hist: pl.DataFrame = df_count.hist(bin_count=50)

plt.stairs(hist["count"][1:], np.exp(hist["breakpoint"]), fill=True)
plt.xscale("log")
plt.show()

tc_cols = ['Tc_10H','Tc_10V','Tc_19H','Tc_19V','Tc_23V','Tc_37H','Tc_37V','Tc_89H','Tc_89V','Tc_165H','Tc_165V','Tc_183V3','Tc_183V7']

df = df_merged[tc_cols + [VARIABLE_SURFACE_TYPE_INDEX, COLUMN_COUNT]]
df = df
# TODO: should we process NaNs differently?
df = df.drop_nans()

# Fit PCA and WPCA on all points.
fig_count, axes_count = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

weight = df[COLUMN_COUNT]
features = df[tc_cols]

# PCA
scaler = StandardScaler()
fs_scaled = scaler.fit_transform(features)
pca = PCA(n_components=2, svd_solver="covariance_eigh")
fs_pca_reduced = pca.fit_transform(fs_scaled)

# WPCA
weighted_scaler = StandardScaler()
fs_weighted_scaled = weighted_scaler.fit_transform(features, sample_weight=weight)
wpca = WPCA(n_components=2)
fs_wpca_reduced = wpca.fit_transform(fs_weighted_scaled, sample_weight=weight.to_numpy())

plot_histogram2d(fs_pca_reduced, axes_count[0])
plot_histogram2d(fs_wpca_reduced, axes_count[1])
axes_count[0].invert_xaxis()

surface_types = ['Ocean',
                'Sea-Ice',
                'High vegetation',
                'Medium vegetation',
                'Low vegetation',
                'Sparse vegetation',
                'Desert',
                'Elevated snow cover',
                'High snow cover',
                'Moderate snow cover',
                'Light snow cover',
                'Standing Water',
                'Ocean or water Coast',
                'Mixed land/ocean or water coast',
                'Land coast',
                'Sea-ice edge',
                'Mountain rain',
                'Mountain snow']

cmap_st, _ = get_surface_type_cmap(['NaN'] + surface_types)

vmin = None
vmax = None
norm = None

count_all = df[COLUMN_COUNT].cast(pl.Int64).sum()
counts = np.array([filter_surface_type(df, idx + 1)[COLUMN_COUNT].cast(pl.Int64).sum() for idx, st in enumerate(surface_types)])
alphas = counts / counts.sum()
alphas = alphas * 0.8 / alphas.max()

ncols = 6
nrows = (len(surface_types) + ncols - 1) // ncols

fig_st, axes_st = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4 * ncols, 4 * nrows))
fig_st_combined, axes_st_combined = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

for idx_st, surface_type in tqdm(enumerate(surface_types)):
    flag_value = idx_st + 1
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", (cmap_st.colors[flag_value])])

    df_to_use = filter_surface_type(df, flag_value)
    df_to_use = df_to_use.filter(df_to_use[COLUMN_COUNT].is_not_null())

    weight = df_to_use[COLUMN_COUNT]
    features = df_to_use[tc_cols]

    fs_pca_reduced = pca.transform(scaler.transform(features))
    fs_wpca_reduced = wpca.transform(weighted_scaler.transform(features))

    # plot separate
    plot_histogram2d(fs_wpca_reduced, axes_st[idx_st // ncols][idx_st % ncols], vmin, vmax, cmap, norm, title=surface_type)

    # plot combined
    plot_histogram2d(fs_pca_reduced, axes_st_combined[0], vmin, vmax, cmap, norm, alphas[idx_st])
    plot_histogram2d(fs_wpca_reduced, axes_st_combined[1], vmin, vmax, cmap, norm, alphas[idx_st])
    axes_st_combined[0].invert_xaxis()

fig_st.tight_layout()

fig_count.savefig(pathlib.Path("images") / "pca" / f"count{SUFFIX}.png")
fig_st_combined.savefig(pathlib.Path("images") / "pca" / f"st_combined{SUFFIX}.png")
fig_st.savefig(pathlib.Path("images") / "pca" / f"st{SUFFIX}.png")

fig_count.show()
fig_st_combined.show()
fig_st.show()
