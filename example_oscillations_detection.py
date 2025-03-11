"""
Example of detecting and plotting seasonal cycles in time series data.
"""

from pmw_analysis.constants import DOMINANT_CYCLE_COLUMN_SUFFIX, TIME_FRACTION_COLUMN
from pmw_analysis.cycle_detection import detect_cycle, plot_cycle, plot_periodogram
from pmw_analysis.preprocessing import read_time_series_and_drop_nan
from pmw_analysis.retrievals.retrieval_1b_c_pmw import retrieve_PD
from pmw_analysis.utils.pandas import timestamp_to_fraction

# Define bucket directory
BUCKET_DIR = "/home/rina/Desktop/data_spb"

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

    # only dry beams
    ts_dry = ts[~(ts["surfacePrecipitation"] > 0)]

    ts_dry = retrieve_PD(ts_dry)
    ts_dry[TIME_FRACTION_COLUMN] = ts_dry["time"].apply(timestamp_to_fraction)

    ts_list.append(ts_dry)
feature_cols += [f"PD_{freq}" for freq in ["19", "37"]]

for idx, ts in enumerate(ts_list):
    ts_list[idx] = detect_cycle(ts, feature_cols)

for freq in ["19", "37"]:
    cols = [col for col in feature_cols if freq in col]
    plot_cycle(ts_list, cols)
    plot_periodogram(ts_list, cols)

dom_cycle_cols = [col for col in ts_list[0].columns if DOMINANT_CYCLE_COLUMN_SUFFIX in col]

ts_decycled_list = []
for idx, ts in enumerate(ts_list):
    ts_decycled = ts[[TIME_FRACTION_COLUMN]].copy()
    for dom_cycle_col in dom_cycle_cols:
        feature_col = dom_cycle_col.removesuffix(DOMINANT_CYCLE_COLUMN_SUFFIX)
        ts_decycled[feature_col] = ts[feature_col] - ts[dom_cycle_col]

    ts_decycled[TIME_FRACTION_COLUMN] = ts_decycled[TIME_FRACTION_COLUMN] % 1
    ts_decycled.sort_values(by=[TIME_FRACTION_COLUMN], inplace=True)

    ts_decycled_list.append(ts_decycled)

for idx, ts in enumerate(ts_decycled_list):
    ts_decycled_list[idx] = detect_cycle(ts, feature_cols)

for freq in ["19", "37"]:
    cols = [col for col in feature_cols if freq in col]
    plot_cycle(ts_decycled_list, cols)
    plot_periodogram(ts_decycled_list, cols)
