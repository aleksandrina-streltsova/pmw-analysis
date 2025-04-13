"""
This module contains cycle detection and visualization methods.
"""
import pathlib
import warnings
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lombscargle

from pmw_analysis.constants import (
    COLUMN_TIME_FRACTION, COLUMN_SUFFIX_DOMINANT_CYCLE,
    ATTR_PERIODOGRAM_DICT, ATTR_NAME,
    SAVEFIG_DIR, SAVEFIG_FLAG
)
from pmw_analysis.utils.pyplot import subplots


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
    n_freqs = 500

    duration = time_series[COLUMN_TIME_FRACTION].values.ptp()
    year_freq = 2 * np.pi / 365
    freqs = np.linspace(1 / duration, year_freq * 12, n_freqs)

    periodogram_dict = {}

    for feature_col in feature_columns:
        periodogram = lombscargle(time_series[COLUMN_TIME_FRACTION].values, time_series[feature_col].values, freqs,
                                  floating_mean=True)
        dom_freq = freqs[periodogram.argmax()]
        sine_wave = _fit_sine_wave(t=time_series[COLUMN_TIME_FRACTION].values,
                                   y=time_series[feature_col].values,
                                   freq=dom_freq)
        dom_cycle_col = feature_col + COLUMN_SUFFIX_DOMINANT_CYCLE
        time_series[dom_cycle_col] = sine_wave

        periodogram_dict[feature_col] = (freqs, periodogram)

    time_series.attrs[ATTR_PERIODOGRAM_DICT] = periodogram_dict

    return time_series


def plot_cycle(time_series: Union[pd.DataFrame, List[pd.DataFrame]], feature_cols: List[str], suffix: str = ""):
    """
    Plot the time series along with the detected dominant cycle.
    Also, plot the time series with the dominant cycle removed.
    """
    if isinstance(time_series, pd.DataFrame):
        ts_list = [time_series]
    else:
        ts_list = time_series

    _, axes = subplots(len(feature_cols), 2 * len(ts_list), xscale=3, yscale=1)

    for idx_col, ts in enumerate(ts_list):
        name = ts.attrs[ATTR_NAME]

        for idx_row, feature_col in enumerate(feature_cols):
            # TODO: think about removing this assertion
            assert ts[feature_col].isna().sum() == 0

            dom_cycle_col = feature_col + COLUMN_SUFFIX_DOMINANT_CYCLE
            if dom_cycle_col not in ts.columns:
                warnings.warn(f"Time series {name} does not contain {dom_cycle_col}. Skipping {feature_col}."
                              f"Call detect_cycle to detect the dominant cycle.")
                continue

            ax = axes[idx_row, 2 * idx_col]
            ts.plot.scatter(x=COLUMN_TIME_FRACTION, y=feature_col, s=1, ax=ax)
            ts.plot(x=COLUMN_TIME_FRACTION, y=dom_cycle_col, ax=ax, c="r")
            ax.set_ylabel("[K]")
            ax.set_title(f"{feature_col} ({name})")

            # Remove the dominant cycle
            ax = axes[idx_row, 2 * idx_col + 1]
            ax.scatter(ts[COLUMN_TIME_FRACTION], ts[feature_col] - ts[dom_cycle_col], s=1)
            ax.set_ylabel("[K]")
            ax.set_title(f"{feature_col} with cycle removed ({name})")
    plt.tight_layout()
    if SAVEFIG_FLAG:
        plt.savefig(pathlib.Path(SAVEFIG_DIR) / f"cycle_{feature_cols[-1]}_{suffix}.png")
    plt.show()


def plot_periodogram(time_series: Union[pd.DataFrame, List[pd.DataFrame]], feature_cols: List[str], suffix: str = ""):
    """
    Plot the Lomb-Scargle periodogram with a vertical line indicating the dominant frequency.
    """
    if isinstance(time_series, pd.DataFrame):
        ts_list = [time_series]
    else:
        ts_list = time_series

    _, axes = subplots(len(feature_cols), len(ts_list), xscale=3, yscale=1)

    for idx_col, ts in enumerate(ts_list):
        periodogram_dict = ts.attrs[ATTR_PERIODOGRAM_DICT]

        for idx_row, feature_col in enumerate(feature_cols):
            ax = axes[idx_row, idx_col]
            freqs, periodogram = periodogram_dict[feature_col]

            dominant_freq = freqs[periodogram.argmax()]
            period = 2 * np.pi / dominant_freq

            ax.plot(freqs, periodogram)
            ax.axvline(dominant_freq, color='r', alpha=0.25)
            ax.set_xlabel("Frequency (rad/s)")
            ax.set_title(
                f"{feature_col} ({ts.attrs[ATTR_NAME]}), frequency = {dominant_freq:.3f}, period = {period:.3f}")
            ax.grid()
    plt.tight_layout()
    if SAVEFIG_FLAG:
        plt.savefig(pathlib.Path(SAVEFIG_DIR) / f"periodogram_{feature_cols[-1]}_{suffix}.png")
    plt.show()
