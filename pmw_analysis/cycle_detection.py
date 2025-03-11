"""
This module contains cycle detection and visualization methods.
"""
import warnings
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lombscargle

from pmw_analysis.constants import TIME_FRACTION_COLUMN, DOMINANT_CYCLE_COLUMN_SUFFIX, PERIODOGRAM_DICT_ATTR, NAME_ATTR


def _fit_sine_wave(t: np.ndarray, y: np.ndarray, freq) -> np.ndarray:
    sin_fit = np.sin(freq * t)
    cos_fit = np.cos(freq * t)

    # Solve for amplitude and phase
    a, b = np.linalg.lstsq(np.column_stack([sin_fit, cos_fit]), y - y.mean(), rcond=None)[0]
    sine_wave = a * sin_fit + b * cos_fit + y.mean()

    return sine_wave


def detect_cycle(time_series: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """
    Detects the dominant cycle in a time series using the Lomb-Scargle periodogram.

    Returns
    -------
        ts : pandas.DataFrame
            Time series dataframe with new columns corresponding to the detected dominant cycles,
            with suffixes '_dom_cycle', and attribute dictionary with feature column keys and
            their corresponding periodogram values.
    """
    n = len(time_series)

    duration = time_series[TIME_FRACTION_COLUMN].values.ptp()
    freqs = np.linspace(1 / duration, n / duration, 5 * n)

    periodogram_dict = {}

    for feature_col in feature_columns:
        periodogram = lombscargle(time_series[TIME_FRACTION_COLUMN].values, time_series[feature_col].values, freqs,
                                  floating_mean=True)
        dom_freq = freqs[periodogram.argmax()]
        sine_wave = _fit_sine_wave(t=time_series[TIME_FRACTION_COLUMN].values,
                                   y=time_series[feature_col].values,
                                   freq=dom_freq)
        dom_cycle_col = feature_col + DOMINANT_CYCLE_COLUMN_SUFFIX
        time_series[dom_cycle_col] = sine_wave

        periodogram_dict[feature_col] = (freqs, periodogram)

    time_series.attrs[PERIODOGRAM_DICT_ATTR] = periodogram_dict

    return time_series


def plot_cycle(time_series: Union[pd.DataFrame, List[pd.DataFrame]], feature_cols: List[str]):
    """
    Plot the time series along with the detected dominant cycle.
    Also, plot the time series with the dominant cycle removed.
    """
    if isinstance(time_series, pd.DataFrame):
        ts_list = [time_series]
    else:
        ts_list = time_series

    n = len(feature_cols)
    _, axes = plt.subplots(n, 2 * len(ts_list), figsize=(6 * n, 6), dpi=300)

    for idx_col, ts in enumerate(ts_list):
        name = ts.attrs[NAME_ATTR]

        for idx_row, feature_col in enumerate(feature_cols):
            # TODO: think about removing this assertion
            assert ts[feature_col].isna().sum() == 0

            dom_cycle_col = feature_col + DOMINANT_CYCLE_COLUMN_SUFFIX
            if dom_cycle_col not in ts.columns:
                warnings.warn(f"Time series {name} does not contain {dom_cycle_col}. Skipping {feature_col}."
                              f"Call detect_cycle to detect the dominant cycle.")
                continue

            ax = axes[idx_row, 2 * idx_col]
            ts.plot.scatter(x=TIME_FRACTION_COLUMN, y=feature_col, s=1, ax=ax)
            ts.plot(x=TIME_FRACTION_COLUMN, y=dom_cycle_col, ax=ax, c="r")
            ax.set_ylabel("[K]")
            ax.set_title(f"{feature_col} ({name})")

            # Remove the dominant cycle
            ax = axes[idx_row, 2 * idx_col + 1]
            ax.scatter(ts[TIME_FRACTION_COLUMN], ts[feature_col] - ts[dom_cycle_col], s=1)
            ax.set_ylabel("[K]")
            ax.set_title(f"{feature_col} with cycle removed ({name})")
    plt.tight_layout()
    plt.show()


def plot_periodogram(time_series: Union[pd.DataFrame, List[pd.DataFrame]], feature_cols: List[str]):
    """
    Plot the Lomb-Scargle periodogram with a vertical line indicating the dominant frequency.
    """
    if isinstance(time_series, pd.DataFrame):
        ts_list = [time_series]
    else:
        ts_list = time_series

    n = len(feature_cols)
    _, axes = plt.subplots(n, len(ts_list), figsize=(3 * n, 6), dpi=300)

    for idx_col, ts in enumerate(ts_list):
        periodogram_dict = ts.attrs[PERIODOGRAM_DICT_ATTR]

        for idx_row, feature_col in enumerate(feature_cols):
            ax = axes[idx_row, idx_col]
            freqs, periodogram = periodogram_dict[feature_col]

            dominant_freq = freqs[periodogram.argmax()]
            period = 2 * np.pi / dominant_freq

            ax.plot(freqs, np.sqrt(4 * periodogram / (5 * n)))
            ax.axvline(dominant_freq, color='r', alpha=0.25)
            ax.set_xlabel("Frequency (rad/s)")
            ax.set_title(
                f"{feature_col} ({ts.attrs[NAME_ATTR]}), frequency = {dominant_freq:.3f}, period = {period:.3f}")
            ax.grid()
    plt.tight_layout()
    plt.show()
