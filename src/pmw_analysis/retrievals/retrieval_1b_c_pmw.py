"""
This module contains GPM PMW 1B and 1C products community-based retrievals.
"""
from typing import Union, Callable, List, Dict, Tuple

import gpm.utils.pmw
import pandas as pd
import xarray as xr
import polars as pl
from gpm.utils.pmw import (
    PMWFrequency,
)
from gpm.utils.xarray import (
    get_xarray_variable,
)

import pmw_analysis.utils.pandas
import pmw_analysis.utils.pmw
from pmw_analysis.constants import COLUMN_L1C_QUALITY_FLAG


def _retrieve_frequency_difference_xr(
        ds: xr.Dataset,
        variable: str | None,
        find_pairs: Callable[[List[PMWFrequency]], Dict[str, Tuple[PMWFrequency, PMWFrequency]]],
        prefix: str,
        name: str,
) -> xr.Dataset:
    # Retrieve DataArray with brightness temperatures
    if variable is None:
        variable = gpm.utils.xarray.get_default_variable(ds, possible_variables=["Tb", "Tc"])
    da = get_xarray_variable(ds, variable=variable)

    # Retrieve available frequencies
    pmw_frequencies = [PMWFrequency.from_string(freq) for freq in da["pmw_frequency"].data]

    # Retrieve pairs
    dict_pairs = find_pairs(pmw_frequencies)

    # If no combo, raise error
    if len(dict_pairs) == 0:
        pmw_frequencies = [freq.title() for freq in pmw_frequencies]
        raise ValueError(f"Impossible to compute {name} with channels: {pmw_frequencies}. No pairs.")

    # Compute differences
    ds_t = da.gpm.unstack_dimension(dim="pmw_frequency", prefix="", suffix="")
    dict_diff = {}
    for pair_str, (freq_curr, freq_next) in dict_pairs.items():
        diff_name = f"{prefix}_{pair_str}"
        # TODO: why no "_" ?
        dict_diff[diff_name] = ds_t[f"{variable}{freq_curr.to_string()}"] - ds_t[f"{variable}{freq_next.to_string()}"]

    # Create dataset
    ds_diff = xr.Dataset(dict_diff)
    return ds_diff


def _retrieve_frequency_difference_pd(
        df: pd.DataFrame,
        variable: str | None,
        find_pairs: Callable[[List[PMWFrequency]], Dict[str, Tuple[PMWFrequency, PMWFrequency]]],
        prefix: str,
        name: str,
) -> pd.DataFrame:
    # Retrieve Series with brightness temperatures
    if variable is None:
        variable = pmw_analysis.utils.pandas.get_default_variable(df, possible_variables=["Tb", "Tc"])

    # Retrieve available frequencies
    variable_prefix = f"{variable}_"
    pmw_frequencies = [PMWFrequency.from_string(str(col).removeprefix(variable_prefix)) for col in df.columns if
                       col.startswith(variable_prefix)]

    # Retrieve pairs
    dict_pairs = find_pairs(pmw_frequencies)

    # If no combo, raise error
    if len(dict_pairs) == 0:
        pmw_frequencies = [freq.title() for freq in pmw_frequencies]
        raise ValueError(f"Impossible to compute {name} with channels: {pmw_frequencies}. No pairs.")

    # Compute differences
    for pair_str, (freq_curr, freq_next) in dict_pairs.items():
        diff_name = f"{prefix}_{pair_str}"
        # TODO: inplace
        df[diff_name] = df[f"{variable}_{freq_curr.to_string()}"] - df[f"{variable}_{freq_next.to_string()}"]

    return df


def _retrieve_difference(
        data: Union[xr.Dataset, pd.DataFrame],
        variable: str | None,
        find_pairs: Callable[[List[PMWFrequency]], Dict[str, Tuple[PMWFrequency, PMWFrequency]]],
        prefix: str,
        name: str,
) -> Union[xr.Dataset, pd.DataFrame]:
    if isinstance(data, xr.Dataset):
        return _retrieve_frequency_difference_xr(data, variable, find_pairs, prefix, name)
    if isinstance(data, pd.DataFrame):
        return _retrieve_frequency_difference_pd(data, variable, find_pairs, prefix, name)
    raise TypeError(f"Unsupported data type: {type(data)}")


def retrieve_polarization_difference(
        data: Union[xr.Dataset, pd.DataFrame],
        variable: str | None = None,
) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Retrieve PMW Channels Polarized Difference (PD).
    """
    return _retrieve_difference(data, variable, gpm.utils.pmw.find_polarization_pairs, "PD", "polarized difference")


def retrieve_frequency_difference(
        data: Union[xr.Dataset, pd.DataFrame],
        variable: str | None = None,
) -> Union[xr.Dataset, pd.DataFrame]:
    """
    Retrieve PMW Channels Frequency Difference (FD).
    """
    return _retrieve_difference(data, variable, pmw_analysis.utils.pmw.find_frequency_pairs, "FD",
                                "frequency difference")


def retrieve_possible_sun_glint(data: pl.DataFrame) -> Tuple[pl.DataFrame, str]:
    sun_glint_column = f"{COLUMN_L1C_QUALITY_FLAG}_SunGlint"
    return data.with_columns(pl.col(COLUMN_L1C_QUALITY_FLAG).eq(1).alias(sun_glint_column)), sun_glint_column


#### ALIAS
retrieve_PD = retrieve_polarization_difference
retrieve_FD = retrieve_frequency_difference
