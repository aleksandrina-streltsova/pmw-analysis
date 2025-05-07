import argparse

import numpy as np
import polars as pl
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from tqdm import tqdm

from pmw_analysis.constants import PMW_ANALYSIS_DIR, COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX, STRUCT_FIELD_COUNT, \
    TC_COLUMNS, ST_COLUMNS, COLUMN_OCCURRENCE, COLUMN_TIME
from pmw_analysis.preprocessing_polars import filter_surface_type
from pmw_analysis.utils.pyplot import get_surface_type_cmap


def _sorted_not_null_unique_values(df: pl.DataFrame, col: str) -> pl.Series:
    return df[col].unique().drop_nulls().sort()

def _get_hist(df_counts, col1, col2) -> pl.DataFrame:
    unique1 = df_counts.select(col1).unique().filter(pl.col(col1).is_not_null()).sort(col1)
    unique2 = df_counts.select(col2).unique().filter(pl.col(col2).is_not_null()).sort(col2)
    all_pairs = unique1.join(unique2, how="cross")

    hist = all_pairs.join(df_counts, on=[col1, col2], how="left").fill_null(0)

    pivoted= hist.pivot(
        values="count",
        index=col1,
        on=col2,
        aggregate_function="first"  # just one value per pair
    ).fill_null(0)

    pivoted = pivoted[::-1]

    return pivoted

def _get_diag_hist(df_counts, col) -> pl.DataFrame:
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


def plot_point_accumulation(df: pl.DataFrame):
    df_count = df.with_columns(
        pl.col(COLUMN_OCCURRENCE).str.split("|").list.get(0).str.to_datetime("%Y-%m-%d %H:%M:%S.%9f").alias(COLUMN_TIME),
    ).select([COLUMN_TIME, COLUMN_COUNT])

    df_count = df_count.with_columns(
        pl.col(COLUMN_TIME).dt.round("1d")
    )

    df_count = df_count.group_by(COLUMN_TIME).sum()
    df_count = df_count.sort(COLUMN_TIME)
    df_count = df_count.with_columns(
        pl.col(COLUMN_COUNT).cum_sum()
    )

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(df_count[COLUMN_TIME], df_count[COLUMN_COUNT])
    fig.show()


def analyze(df_merged: pl.DataFrame):
    path = pathlib.Path(PMW_ANALYSIS_DIR) / "default" / "0.parquet"
    df_merged = pl.read_parquet(path)

    df = df_merged[TC_COLUMNS + ["surfaceTypeIndex", "count"]]

    # cast Tc values to pl.Enum for faster processing
    for col in TC_COLUMNS:
        unique_count = df.n_unique(col)

        # enum_type = pl.Enum(_sorted_not_null_unique_values(df, col).round().cast(pl.UInt16).cast(pl.Utf8))
        # df = df.with_columns(pl.col(col).round().cast(pl.UInt16).cast(pl.Utf8).cast(enum_type))
        df = df.with_columns(pl.col(col).round().cast(pl.UInt16).cast(pl.Utf8).str.zfill(3).cast(pl.Categorical("lexical")))

        assert unique_count == df.n_unique(col)
    _sorted_not_null_unique_values(df, "Tc_10H")

    for tc_col in TC_COLUMNS:
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

    cmap_st, _ = get_surface_type_cmap(['NaN'] + ST_COLUMNS)

    k = 13

    vmin = None
    vmax = None
    norm = None

    for idx_st, surface_type in tqdm(enumerate([None])):# + ST_COLUMNS):
        fig, axes = plt.subplots(k, k, figsize=(20, 20))
        df_to_use = df
        if surface_type is None:
            flag_value = None
            cmap = "rocket_r"
        else:
            flag_value = idx_st

            df_to_use = df_to_use[TC_COLUMNS + [VARIABLE_SURFACE_TYPE_INDEX]]
            df_to_use = filter_surface_type(df_to_use, flag_value)

            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", (cmap_st.colors[flag_value])])

        hists_mtx = []
        x_ticks_all = []
        y_ticks_all = []

        for idx1, tc_col1 in tqdm(enumerate(TC_COLUMNS[:k])):
            hists_row = []
            for idx2, tc_col2 in enumerate(TC_COLUMNS[:idx1 + 1]):
                cols = [tc_col1] if idx1 == idx2 else [tc_col1, tc_col2]
                counts = df_to_use.select(cols + [COLUMN_COUNT]).group_by(cols, maintain_order=False).sum()

                hist = _get_hist(counts, tc_col1, tc_col2) if idx1 != idx2 else _get_diag_hist(counts, tc_col1)
                hists_row.append(hist)

                x_ticks = [round(float(tick)) for tick in hist.columns[1:]]
                y_ticks = [round(float(tick)) for tick in hist[tc_col1]]

                assert x_ticks == sorted(x_ticks)
                assert y_ticks == sorted(y_ticks)[::-1]

                x_ticks_all += x_ticks
                y_ticks_all += y_ticks

            hists_mtx.append(hists_row)

            for idx1, tc_col1 in tqdm(enumerate(TC_COLUMNS[:k])):
                hists_row = []
                for idx2, tc_col2 in enumerate(TC_COLUMNS[:idx1 + 1]):
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
                        hist = _get_hist(counts, tc_col1, tc_col2)
                    else:
                        hist = _get_diag_hist(counts, tc_col1)

                    x_ticks = [round(float(tick)) for tick in hist.columns[1:]]
                    y_ticks = [round(float(tick)) for tick in hist[tc_col1]]

                    assert x_ticks == sorted(x_ticks)
                    assert y_ticks == sorted(y_ticks)

                    sns.heatmap(hist[:, 1:], vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, annot=False,
                                ax=axes[idx1, idx2], cbar=False)
                    sns.heatmap(hist[:, 1:].transpose(), vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, annot=False,
                                ax=axes[idx2, idx1], cbar=False)

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



def main():
    parser = argparse.ArgumentParser(description="Analyse quantized PMW features")

    parser.add_argument("file", help="Path to the data file")

    args = parser.parse_args()

    path = pathlib.Path(args.file)
    path = pathlib.Path(PMW_ANALYSIS_DIR) / "default" / "0.parquet"

    df_merged = pl.read_parquet(path)



if __name__ == '__main__':
