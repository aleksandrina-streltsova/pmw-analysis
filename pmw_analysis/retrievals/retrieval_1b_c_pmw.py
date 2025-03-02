"""This module contains GPM PMW 1B and 1C products community-based retrievals."""
from collections import defaultdict

import xarray as xr
from gpm.utils.pmw import (
    PMWFrequency,
)
from gpm.utils.xarray import (
    get_default_variable,
    get_xarray_variable,
)


def retrieve_frequency_difference(ds: xr.Dataset, variable: str = None) -> xr.Dataset:
    """Retrieve PMW Channels Frequency Difference (FD)."""
    # Retrieve DataArray with brightness temperatures
    if variable is None:
        variable = get_default_variable(ds, possible_variables=["Tb", "Tc"])
    da = get_xarray_variable(ds, variable=variable)

    # Retrieve available frequencies
    pmw_frequencies = [PMWFrequency.from_string(freq) for freq in da["pmw_frequency"].data]

    # Retrieve frequencies grouped by polarization
    dict_frequency_groups = defaultdict(list)
    for freq in pmw_frequencies:
        dict_frequency_groups[freq.polarization].append(freq)
    dict_frequency_groups = {pol: freqs for pol, freqs in dict_frequency_groups.items() if len(freqs) > 1}

    # If no combo, raise error
    if len(dict_frequency_groups) == 0:
        pmw_frequencies = [freq.title() for freq in pmw_frequencies]
        raise ValueError(f"Impossible to compute frequency difference with channels: {pmw_frequencies}. No pairs.")

    # Compute FDs
    ds_t = da.gpm.unstack_dimension(dim="pmw_frequency", prefix="", suffix="")
    dict_fd = {}
    for pol, freqs in dict_frequency_groups.items():
        for freq_curr, freq_next in zip(freqs, freqs[1:]):
            fd_name = f"FD_{freq_curr.to_string()}_{freq_next.to_string()}"
            dict_fd[fd_name] = ds_t[f"{variable}{freq_curr.to_string()}"] - ds_t[f"{variable}{freq_next.to_string()}"]

    # Create dataset
    ds_fd = xr.Dataset(dict_fd)
    return ds_fd


#### ALIAS
retrieve_FD = retrieve_frequency_difference
