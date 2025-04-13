"""
This module provides PMW utilities.
"""
from collections import defaultdict
from typing import List, Tuple, Dict

from gpm.utils.pmw import PMWFrequency


def find_frequency_pairs(pmw_frequencies: List[PMWFrequency]) -> Dict[str, Tuple[PMWFrequency, PMWFrequency]]:
    """
    Identify center frequency pairs of PMWFrequency objects.

    The PMWFrequency objects must share the same polarization
    but differ in center frequency (e.g., 10 GHz vs 19 GHz).

    This function iterates through each PMWFrequency in the input list and
    attempts to match it with another PMWFrequency that:
      1. Has the same polarization.
      2. Has the nearest higher center frequency (e.g., 10 GHz vs 19 GHz).

    Once a valid pair is found, it is stored in a dictionary with a key
    formed by concatenating the pair's frequencies, separated by an underscore.
    For consistent ordering of pairs, lower frequency is placed first in the tuple.

    Parameters
    ----------
    pmw_frequencies : list of PMWFrequency
        A list of PMWFrequency objects to be examined for pairs.

    Returns
    -------
    dict
        A dictionary where keys are concatenations of the pair's frequencies,
        separated by an underscore, and values are 2-tuples of PMWFrequency objects
        in (lower frequency, higher frequency) order. If no match is found for a given
        frequency, that frequency is not included in the dictionary.
    """
    pmw_frequencies.sort()

    dict_frequency_groups = defaultdict(list)
    for freq in pmw_frequencies:
        dict_frequency_groups[freq.polarization].append(freq)

    frequency_groups = [freqs for freqs in dict_frequency_groups.values() if len(freqs) > 1]
    frequency_pairs = {}
    for freqs in frequency_groups:
        for freq_curr, freq_next in zip(freqs, freqs[1:]):
            frequency_pairs[f"{freq_curr.to_string()}_{freq_next.to_string()}"] = (freq_curr, freq_next)

    return frequency_pairs
