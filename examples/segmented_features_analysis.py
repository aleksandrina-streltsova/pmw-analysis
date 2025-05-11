import argparse
import pathlib
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.colors import LogNorm
from tqdm import tqdm

from pmw_analysis.constants import PMW_ANALYSIS_DIR, COLUMN_COUNT, VARIABLE_SURFACE_TYPE_INDEX, TC_COLUMNS, ST_COLUMNS, \
    COLUMN_ACCUM_UNIQUE, COLUMN_ACCUM_ALL, COLUMN_OCCURRENCE_TIME
from pmw_analysis.preprocessing_polars import filter_surface_type, expand_occurrence_column
from pmw_analysis.utils.pyplot import get_surface_type_cmap


def _sorted_not_null_unique_values(df: pl.DataFrame, col: str) -> pl.Series:
    return df[col].unique().drop_nulls().sort()


def _cum_sum_over_time(df: pl.DataFrame, var: Optional[str] = None, descending: bool = False) -> pl.DataFrame:
    if var is None:
        df = df.group_by(COLUMN_OCCURRENCE_TIME).agg(
            pl.len().alias(COLUMN_ACCUM_UNIQUE),
            pl.col(COLUMN_COUNT).cast(pl.UInt64).sum().alias(COLUMN_ACCUM_ALL)
        )
    else:
        df = df.group_by(COLUMN_OCCURRENCE_TIME).agg(
            pl.col(var).sum().alias(COLUMN_ACCUM_UNIQUE),
            pl.col(var).mul(pl.col(COLUMN_COUNT)).sum().alias(COLUMN_ACCUM_ALL)
        )

    df = df.sort(COLUMN_OCCURRENCE_TIME, descending=descending)
    df = df.with_columns(
        pl.col(COLUMN_ACCUM_UNIQUE).cum_sum(),
        pl.col(COLUMN_ACCUM_ALL).cum_sum(),
    )
    if descending:
        df = df.reverse()
    return df


def plot_point_accumulation(path: pathlib.Path, var: Optional[str]):
    df = pl.read_parquet(path)

    columns = [COLUMN_OCCURRENCE_TIME, COLUMN_COUNT] + ([] if var is None else [var, f"{var}_count"])
    df = expand_occurrence_column(df).select(columns)
    df = df.with_columns(pl.col(COLUMN_OCCURRENCE_TIME).dt.round("1d"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax1, ax2 = axes

    df = df.with_columns(pl.col(COLUMN_COUNT).cast(pl.UInt64))
    df_count = _cum_sum_over_time(df)

    ax1.plot(df_count[COLUMN_OCCURRENCE_TIME], df_count[COLUMN_ACCUM_ALL], color="b", label="All")
    ax2.plot(df_count[COLUMN_OCCURRENCE_TIME], df_count[COLUMN_ACCUM_UNIQUE], color="b", label="All")

    ax1.set_title("Signatures first seen before this time")
    ax2.set_title("Unique signatures seen before this time")

    ax1.set_xlabel("Time")
    ax2.set_xlabel("Time")

    ax1.set_ylabel("Signatures count")
    ax2.set_ylabel("Signatures count")

    if var is not None:
        var_count = f"{var}_{COLUMN_COUNT}"
        df_var_not_null = df.filter(pl.col(var_count).ne(0).and_(pl.col(var).is_not_nan())) # TODO: fix
        df_var_not_null = df_var_not_null.drop(COLUMN_COUNT).rename({var_count: COLUMN_COUNT})
        df_var_not_null = df_var_not_null.with_columns(pl.col(COLUMN_COUNT).cast(pl.UInt64))

        df_var_sum_desc = _cum_sum_over_time(df_var_not_null, var=var, descending=True)
        df_var_count_desc = _cum_sum_over_time(df_var_not_null, descending=True)

        df_var_mean_desc = df_var_sum_desc.with_columns(
            df_var_sum_desc[COLUMN_ACCUM_ALL] / df_var_count_desc[COLUMN_ACCUM_ALL],
            df_var_sum_desc[COLUMN_ACCUM_UNIQUE] / df_var_count_desc[COLUMN_ACCUM_UNIQUE],
        )

        count_total_all = df_var_not_null[COLUMN_COUNT].sum()
        count_total_unique = len(df_var_not_null)

        ax1.plot(df_var_count_desc[COLUMN_OCCURRENCE_TIME], count_total_all - df_var_count_desc[COLUMN_ACCUM_ALL], color="g", label=f"Not-null '{var}'")
        ax2.plot(df_var_count_desc[COLUMN_OCCURRENCE_TIME], count_total_unique - df_var_count_desc[COLUMN_ACCUM_UNIQUE], color="g", label=f"Not-null '{var}'")

        ax1_var, ax2_var = (ax1.twinx(), ax2.twinx())

        ax1_var.plot(df_var_mean_desc[COLUMN_OCCURRENCE_TIME], df_var_mean_desc[COLUMN_ACCUM_ALL], color="r", label=var)
        ax2_var.plot(df_var_mean_desc[COLUMN_OCCURRENCE_TIME], df_var_mean_desc[COLUMN_ACCUM_UNIQUE], color="r", label=var)

        ax1_var.set_ylabel(f"Mean of '{var}' (from this time to end)")
        ax2_var.set_ylabel(f"Mean of '{var}' (from this time to end)")

    fig.legend()
    fig.tight_layout()

    images_path = pathlib.Path("images") / path.parent.name
    images_path.mkdir(exist_ok=True)

    suffix = "" if var is None else f"_{var}"
    fig.savefig(images_path / f"count_over_time{suffix}.png")

    fig.show()


def analyze(path: pathlib.Path, var: Optional[str]):
    df_merged = pl.read_parquet(path)

    df = df_merged[TC_COLUMNS + ["surfaceTypeIndex", "count"]]
    df = df

    n_bins = []
    for tc_col in TC_COLUMNS:
        value_count = df[[tc_col, COLUMN_COUNT]].group_by(tc_col).sum()
        null_count = value_count.filter(pl.col(tc_col).is_null())[COLUMN_COUNT][0]
        unique_value_count = len(value_count.filter(pl.col(tc_col).is_not_null()))

        print(f"{tc_col} has {unique_value_count} unique non-null values.")
        print(f"{tc_col} has {null_count} missing values.")

        n_bins.append(unique_value_count)

    n_bins = np.array(n_bins)
    n_bins_cs = np.cumsum(n_bins)
    n_bins_cs = np.insert(n_bins_cs, 0, 0)

    cmap_st, _ = get_surface_type_cmap(['NaN'] + ST_COLUMNS)

    min_vals = np.array(df.select([pl.col(col).min() for col in TC_COLUMNS]).row(0))
    max_vals = np.array(df.select([pl.col(col).max() for col in TC_COLUMNS]).row(0))

    x_ticks = np.linspace(0, n_bins.sum() - 1, n_bins.sum())
    y_ticks = np.linspace(0, n_bins.sum() - 1, n_bins.sum())
    x_tick_labels = np.concatenate(
        [np.round(np.linspace(min_vals[i], max_vals[i], n_bins[i])).astype(int) for i in range(len(TC_COLUMNS))])
    y_tick_labels = np.concatenate(
        [np.round(np.linspace(min_vals[i], max_vals[i], n_bins[i])).astype(int)[::-1] for i in range(len(TC_COLUMNS))])

    for idx_st, surface_type in enumerate([None] + ST_COLUMNS):
        df_to_use = df

        if surface_type is None:
            cmap = "rocket_r"
        else:
            flag_value = idx_st

            df_to_use = df_to_use[TC_COLUMNS + [VARIABLE_SURFACE_TYPE_INDEX]]
            df_to_use = filter_surface_type(df_to_use, flag_value)

            cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", ["white", (cmap_st.colors[flag_value])])

        if var is not None:
            cmap = "viridis"

        hists_mtx = np.zeros((n_bins_cs[-1], n_bins_cs[-1]))

        for idx1, tc_col1 in tqdm(enumerate(TC_COLUMNS)):
            for idx2, tc_col2 in enumerate(TC_COLUMNS[:idx1 + 1]):
                cols = [tc_col1] if idx1 == idx2 else [tc_col1, tc_col2]

                counts = df_to_use.select(cols + [COLUMN_COUNT]).group_by(cols, maintain_order=False).sum()
                hist_range = ((min_vals[idx1], max_vals[idx1]), (min_vals[idx2], max_vals[idx2]))
                hist = np.histogram2d(counts[tc_col1], counts[tc_col2],
                                      range=hist_range, weights=counts[COLUMN_COUNT],
                                      bins=(n_bins[idx1], n_bins[idx2]))[0]

                if var is not None:
                    counts_var = df_to_use.with_columns(pl.col(var).mul(COLUMN_COUNT).alias("tmp")).select(
                        cols + ["tmp"]).group_by(cols, maintain_order=False).sum()
                    hist_var = np.histogram2d(counts_var[tc_col1], counts_var[tc_col2],
                                              range=hist_range, weights=counts_var["tmp"],
                                              bins=(n_bins[idx1], n_bins[idx2]))[0]
                    hist = hist_var / np.where(np.isclose(hist, 0), 1, hist)
                if idx1 == idx2:
                    continue
                hist = hist[::-1]
                hists_mtx[n_bins_cs[idx1]: n_bins_cs[idx1 + 1], n_bins_cs[idx2]: n_bins_cs[idx2 + 1]] = hist
                hists_mtx[n_bins_cs[idx2]: n_bins_cs[idx2 + 1], n_bins_cs[idx1]: n_bins_cs[idx1 + 1]] = hist.T

        if var is None:
            vmin = max(hists_mtx.min(), 1)
            vmax = max(hists_mtx.max(), 1)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            vmin = None
            vmax = np.percentile(hists_mtx, 99)  # TODO
            norm = None

        fig, ax = plt.subplots(1, 1, figsize=(2 * len(TC_COLUMNS) + 4, 2 * len(TC_COLUMNS)))
        sns.heatmap(hists_mtx, vmin=vmin, vmax=vmax, cmap=cmap, norm=norm, annot=False, ax=ax)
        ax.set_xticks(x_ticks, x_tick_labels, fontsize=5)
        ax.set_yticks(y_ticks, y_tick_labels, fontsize=5)
        ax.set_xlabel("           ".join(TC_COLUMNS), fontsize=16)
        ax.set_ylabel("           ".join(TC_COLUMNS[::-1]), fontsize=16)
        # fig.show()

        images_path = pathlib.Path("images") / path.parent.name
        images_path.mkdir(exist_ok=True)

        if var is not None:
            images_path = images_path / var
            images_path.mkdir(exist_ok=True)

        title_suffix = "" if var is None else f" ({var})"

        if surface_type is not None:
            fig.suptitle(surface_type + title_suffix, fontsize=30)
            fig.savefig(images_path / f"count_{surface_type.replace("/", "_")}_{len(TC_COLUMNS)}.png")
        else:
            fig.suptitle("All Surfaces" + title_suffix, fontsize=30)
            fig.savefig(images_path / f"count_{surface_type}_{len(TC_COLUMNS)}.png")
        # fig.show()


def main():
    parser = argparse.ArgumentParser(description="Analyse quantized PMW features")
    subparsers = parser.add_subparsers(dest="analysis", required=True, help="Analysis to perform")
    parser.add_argument("--transform", "-t", default="default", choices=["default", "pd", "ratio"],
                        help="Type of transformation performed on data")

    pairplot_parser = subparsers.add_parser("pairplot", help="")
    pairplot_parser.add_argument("--var", "-v", help="Variable to color by in pair plots")

    accum_parser = subparsers.add_parser("accum", help="")
    accum_parser.add_argument("--var", "-v", help="Variable to ")

    args = parser.parse_args()

    path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform / "final.parquet"

    if args.analysis == "accum":
        plot_point_accumulation(path, args.var)
    elif args.analysis == "pairplot":
        analyze(path, args.var)


if __name__ == '__main__':
    main()
