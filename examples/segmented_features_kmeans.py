import polars as pl
import pathlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pmw_analysis.constants import PMW_ANALYSIS_DIR, COLUMN_COUNT
from pmw_analysis.decomposition import WPCA

SUFFIX = ""

df_merged: pl.DataFrame = pl.read_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "merged" / "final.parquet")

tc_cols = ['Tc_10H','Tc_10V','Tc_19H','Tc_19V','Tc_23V','Tc_37H','Tc_37V','Tc_89H','Tc_89V','Tc_165H','Tc_165V','Tc_183V3','Tc_183V7']
df = df_merged[tc_cols + [COLUMN_COUNT]]
df = df.drop_nans()

df = df[:100000]

weight = df[COLUMN_COUNT]
features = df[tc_cols]

fs_scaled = StandardScaler().fit_transform(features, sample_weight=weight)

kmeans = KMeans(n_clusters=3)
labels = kmeans.fit_predict(fs_scaled)

wpca = WPCA(n_components=2)
fs_wpca_reduced = wpca.fit_transform(fs_scaled, sample_weight=weight.to_numpy())

plt.scatter(fs_wpca_reduced[:, 0], fs_wpca_reduced[:, 1], c=labels)
plt.show()

