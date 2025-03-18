"""
Example of detecting and plotting seasonal cycles in time series data.
"""
import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pmw_analysis.constants import (
    COLUMN_TIME_FRACTION,
    ATTR_NAME,
    SAVEFIG_FLAG,
    SAVEFIG_DIR,
    BUCKET_DIR,
)
from pmw_analysis.cycle_detection import detect_cycle, plot_cycle, plot_periodogram
from pmw_analysis.preprocessing import read_time_series_and_drop_nan
from pmw_analysis.retrievals.retrieval_1b_c_pmw import retrieve_PD
from pmw_analysis.utils.pandas import timestamp_to_fraction
from pmw_analysis.utils.pyplot import scatter_na

plt.rcParams["image.cmap"] = "viridis"

# City: Saint Petersburg
# Define point with RFI
point_city = (30.36, 59.93)
# Define point without RFI
point_outskirts = (31.05, 58.95)

DISTANCE = 5000

ts_list = []
feature_cols = [f"Tc_{freq}{pol}" for freq in ["19", "37"] for pol in ["H", "V"]]
for point, name in [(point_city, "city"), (point_outskirts, "outskirts")]:
    ts = read_time_series_and_drop_nan(BUCKET_DIR, point, DISTANCE, name, feature_cols)
    ts = retrieve_PD(ts)
    ts[COLUMN_TIME_FRACTION] = ts["time"].apply(timestamp_to_fraction)

    ts_list.append(ts)
feature_cols += [f"PD_{freq}" for freq in ["19", "37"]]

# 1. Colorize time series by precipitation flags
for color_col in ["surfacePrecipitation", "Tc_165V", "mostLikelyPrecipitation", "precipitationYesNoFlag",
                  "probabilityOfPrecip"]:
    n = len(feature_cols)
    fig, axes = plt.subplots(n, len(ts_list), figsize=(6 * len(ts_list), 2 * n), dpi=300)
    for idx_col, ts in enumerate(ts_list):
        name = ts.attrs[ATTR_NAME]
        for idx_row, feature_col in enumerate(feature_cols):
            # TODO: think about removing this assertion
            assert ts[feature_col].isna().sum() == 0

            plt.sca(axes[idx_row, idx_col])
            scatter_na(x=ts[COLUMN_TIME_FRACTION], y=ts[feature_col], c=ts[color_col], color_label=color_col)
            plt.ylabel("[K]")
            plt.title(f"{feature_col} ({name})")
    plt.tight_layout()
    if SAVEFIG_FLAG:
        plt.savefig(pathlib.Path(SAVEFIG_DIR) / f"{color_col}.png")
    plt.show()

# 2. Calculate correlation matrix.
for ts in ts_list:
    name = ts.attrs[ATTR_NAME]
    corr_mtx = ts[feature_cols].corr()
    sns.heatmap(corr_mtx, annot=True)
    if SAVEFIG_FLAG:
        plt.savefig(pathlib.Path(SAVEFIG_DIR) / f"{name}_corr.png")
    plt.show()

# 3. Detect cycles.
for idx, ts in enumerate(ts_list):
    ts_list[idx] = detect_cycle(ts, feature_cols)

for freq in ["19", "37"]:
    cols = [col for col in feature_cols if freq in col]
    plot_cycle(ts_list, cols, suffix="all")
    plot_periodogram(ts_list, cols, suffix="all")


# 4. Divide the time series into periods, keeping only the median value for each period, and then detect cycles.
def _get_medians(ts_noisy: pd.DataFrame, column: str, time_period: float) -> pd.DataFrame:
    ts_noisy = ts_noisy.copy()
    ts_noisy["period"] = ts_noisy[COLUMN_TIME_FRACTION] // time_period

    # Function to find the row closest to the median of 'A' in each period
    def median_row(sub_ts):
        median = sub_ts[column].median()
        return sub_ts.iloc[(sub_ts[column] - median).abs().argmin()]

    # Apply function to each period
    median_rows = ts_noisy.groupby("period").apply(median_row, include_groups=False).reset_index(drop=True)
    median_rows.attrs[ATTR_NAME] = ts_noisy.attrs[ATTR_NAME]
    print(f"{len(median_rows)}/{len(ts_noisy)} rows")
    return median_rows


TIME_PERIOD = 20
for freq in ["19", "37"]:
    cols = [col for col in feature_cols if freq in col]
    ts_medians_list = []
    for idx, ts in enumerate(ts_list):
        ts_medians = _get_medians(ts, f"Tc_{freq}H", TIME_PERIOD)
        ts_medians = detect_cycle(ts_medians, cols)
        ts_medians_list.append(ts_medians)

    plot_cycle(ts_medians_list, cols, suffix=f"median_{TIME_PERIOD}")
    plot_periodogram(ts_medians_list, cols, suffix=f"median_{TIME_PERIOD}")
