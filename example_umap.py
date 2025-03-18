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
from sklearn.preprocessing import StandardScaler

from pmw_analysis.constants import (
    SAVEFIG_FLAG, SAVEFIG_DIR, COLUMN_TIME, PRODUCT_1C_GMI_R, PRODUCT_TYPE_RS, VERSION,
    STORAGE_GES_DISC, VARIABLE_TC, DIM_PMW,
)
from pmw_analysis.preprocessing import remove_missing_values

# Define product
PRODUCT = PRODUCT_1C_GMI_R
PRODUCT_TYPE = PRODUCT_TYPE_RS

dates = [
    datetime.date(2015, 3, 20),
    datetime.date(2015, 6, 21),
    datetime.date(2015, 9, 23),
    datetime.date(2015, 12, 22),
]

ds_list = []

fig_kwargs = {"figsize": (10, 10), "dpi": 300}
for idx, date in enumerate(dates):
    start_time = datetime.datetime(date.year, date.month, date.day, hour=0, minute=0, second=0)
    end_time = start_time + datetime.timedelta(days=1)

    # TODO: check_integrity
    gpm.download(PRODUCT, start_time, end_time, PRODUCT_TYPE, VERSION, STORAGE_GES_DISC, check_integrity=False)

    dt = gpm.open_datatree(PRODUCT, start_time, end_time, VARIABLE_TC, product_type=PRODUCT_TYPE, chunks={})
    ds = dt.gpm.regrid_pmw_l1(scan_mode_reference="S1")
    ds = ds.compute()

    # TODO: where are missing values located on map?
    # da: xr.DataArray = ds[VARIABLE_TC]
    # da.sel(pmw_frequency="19H").gpm.plot_map(add_swath_lines=False, vmin=70, vmax=350, fig_kwargs=fig_kwargs)
    # if SAVEFIG_FLAG:
    #     plt.savefig(pathlib.Path(SAVEFIG_PATH) / f"{dates[0].year}_{idx}.png")
    # plt.show()

    ds_list.append(ds)

df_list = []
for idx, ds in enumerate(ds_list):
    da = ds[VARIABLE_TC]
    ds = da.gpm.unstack_dimension(dim=DIM_PMW, suffix="_")
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
df_umap_sparse = df_umap.iloc[::150]
print(len(df_umap_sparse))

columns_low_v = [col for col in columns_low if col.endswith("V")]
columns_high_v = [col for col in columns_high if col.endswith("V") or col.endswith("V3") or col.endswith("V7")]

sns.pairplot(df_umap_sparse[columns_low_v])
plt.show()

sns.pairplot(df_umap_sparse[columns_high_v])
plt.show()

g = sns.PairGrid(df_umap_sparse, x_vars=columns_low_v, y_vars=columns_high_v)
g.map(sns.scatterplot)
plt.show()

df_umap_sparse = df_umap.iloc[::600]
print(len(df_umap_sparse))

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
