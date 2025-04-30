import numpy as np
import polars as pl
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from tqdm import tqdm

from pmw_analysis.constants import PMW_ANALYSIS_DIR, COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX, STRUCT_FIELD_COUNT
from pmw_analysis.preprocessing_polars import filter_surface_type
from pmw_analysis.utils.pyplot import get_surface_type_cmap


def _sorted_not_null_unique_values(df: pl.DataFrame, col: str) -> pl.Series:
    return df[col].unique().drop_nulls().sort()

def get_histogram(df_counts, col1, col2) -> pl.DataFrame:
    unique1 = df_counts.select(col1).unique().filter(pl.col(col1).is_not_null()).sort(col1)
    unique2 = df_counts.select(col2).unique().filter(pl.col(col2).is_not_null()).sort(col2)
    all_pairs = unique1.join(unique2, how="cross")

    hist = all_pairs.join(df_counts, on=[col1, col2], how="left").fill_null(0)

    pivoted = hist.pivot(
        values="count",
        index=col1,
        on=col2,
        aggregate_function="first"  # just one value per pair
    ).fill_null(0)

    return pivoted

def get_diag_histogram(df_counts, col) -> pl.DataFrame:
    diag = df_counts.filter(pl.col(col).is_not_null()).sort(col)

    data = {
        str(val): [
            count if i == j else 0
            for i in range(len(diag))
        ]
        for j, (val, count) in enumerate(zip(diag[col], diag[COLUMN_COUNT]))
    }

    diag_df = pl.DataFrame(data)
    diag_df = diag_df.with_columns(pl.Series(col, diag[col])).select([col, *diag_df.columns])

    return diag_df

df_merged = pl.read_parquet(pathlib.Path(PMW_ANALYSIS_DIR) / "merged" / "final.parquet")
tc_cols = ['Tc_10H','Tc_10V','Tc_19H','Tc_19V','Tc_23V','Tc_37H','Tc_37V','Tc_89H','Tc_89V','Tc_165H','Tc_165V','Tc_183V3','Tc_183V7']
df = df_merged[tc_cols + ["surfaceTypeIndex", "count"]]

# cast Tc values to pl.Enum for faster processing
for col in tc_cols:
    unique_count = df.n_unique(col)

    # enum_type = pl.Enum(_sorted_not_null_unique_values(df, col).round().cast(pl.UInt16).cast(pl.Utf8))
    # df = df.with_columns(pl.col(col).round().cast(pl.UInt16).cast(pl.Utf8).cast(enum_type))
    df = df.with_columns(pl.col(col).round().cast(pl.UInt16).cast(pl.Utf8).str.zfill(3).cast(pl.Categorical("lexical")))

    assert unique_count == df.n_unique(col)
_sorted_not_null_unique_values(df, "Tc_10H")

for tc_col in tc_cols:
    value_count = df[[tc_col, COLUMN_COUNT]].group_by(tc_col).sum()
    null_count = value_count.filter(pl.col(tc_col).is_null())[COLUMN_COUNT][0]
    print(f"{tc_col} has {len(value_count.filter(pl.col(tc_col).is_not_null()))} unique non-null values.")
    print(f"{tc_col} has {null_count} missing values.")


def set_xticks(ax, x_ticks, col):
    ax.set_xticks(np.arange(len(x_ticks)))
    ax.set_xticklabels(x_ticks, rotation="vertical", fontsize=5)
    ax.set_xlabel(col, fontsize=15)

def set_yticks(ax, y_ticks, col):
    ax.set_yticks(np.arange(len(y_ticks)))
    ax.set_yticklabels(y_ticks, rotation="horizontal", fontsize=5)
    ax.set_ylabel(col, fontsize=15)


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

k = 13

vmin = None
vmax = None
norm = None

for idx_st, surface_type in tqdm(enumerate([None])):# + surface_types):
    fig, axes = plt.subplots(k, k, figsize=(20, 20))
    df_to_use = df
    if surface_type is None:
        flag_value = None
        cmap = "rocket_r"
    else:
        flag_value = idx_st

        df_to_use = df_to_use[tc_cols + [VARIABLE_SURFACE_TYPE_INDEX]]
        df_to_use = filter_surface_type(df_to_use, flag_value)

        cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", (cmap_st.colors[flag_value])])

    for idx1, tc_col1 in tqdm(enumerate(tc_cols[:k])):
        for idx2, tc_col2 in enumerate(tc_cols[:idx1 + 1]):
            if idx1 == idx2:
                cols = [tc_col1]
            else:
                cols = [tc_col1, tc_col2]

            counts = df_to_use.select(cols + [COLUMN_COUNT]).group_by(cols, maintain_order=False).sum()

            if norm is None:
                vmin = 1
                vmax = counts[COLUMN_COUNT].max()
                norm = LogNorm(vmin=vmin, vmax=vmax)

            if idx1 != idx2:
                histogram = get_histogram(counts, tc_col1, tc_col2)
            else:
                histogram = get_diag_histogram(counts, tc_col1)

            x_ticks = [round(float(tick)) for tick in histogram.columns[1:]]
            y_ticks = [round(float(tick)) for tick in histogram[tc_col1]]

            assert x_ticks == sorted(x_ticks)
            assert y_ticks == sorted(y_ticks)

            sns.heatmap(histogram[:, 1:], vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, annot=False, ax=axes[idx1, idx2], cbar=False)
            sns.heatmap(histogram[:, 1:].transpose(), vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, annot=False, ax=axes[idx2, idx1], cbar=False)

            for j, (i1, i2) in enumerate([(idx1, idx2), (idx2, idx1)]):
                if i1 == i2 and j == 1:
                    continue
                ax = axes[i1, i2]

                ax.invert_yaxis()
                if i1 != k - 1:
                    ax.set_xticklabels([])
                    ax.set_xlabel("")
                else:
                    set_xticks(ax, x_ticks, tc_col2)
                if i2 != 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel("")
                else:
                    set_yticks(ax, y_ticks, tc_col1)

    # sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    #
    # fig.subplots_adjust(right=0.9)
    # cbar_ax = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    # cbar = fig.colorbar(sm, cax=cbar_ax)
    # cbar.ax.tick_params(labelsize=26)

    if surface_type is not None:
        fig.suptitle(surface_type, fontsize=30)
        fig.savefig(pathlib.Path("images") / f"testing_count_{surface_type.replace("/", "_")}.png")
    else:
        fig.suptitle("All Surfaces", fontsize=30)
        fig.savefig(pathlib.Path("images") / f"testing_count_{surface_type}.png")
    fig.show()
