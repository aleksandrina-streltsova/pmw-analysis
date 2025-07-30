"""
This module provides utilities for working with data stored in Pandas format.
"""
import pandas as pd


def get_default_variable(df: pd.DataFrame, possible_variables) -> str:
    """
    Return one of the possible default variables.

    Check if one of the variables in 'possible_variables' is present in the pandas.DataFrame.
    If neither variable is present, raise an error.
    If both are present, raise an error.
    Return the name of the single available variable in the pandas.DataFrame

    Parameters
    ----------
    df : pandas.DataFrame
        The Pandas DataFrame to inspect.
    possible_variables : list of str
        The variable names to look for.

    Returns
    -------
    str
        The name of the variable found in the pandas.DataFrame
.
    """
    if isinstance(possible_variables, str):
        possible_variables = [possible_variables]
    found_vars = [v for v in possible_variables if any(col.startswith(f"{v}_") for col in df.columns)]
    if len(found_vars) == 0:
        raise ValueError(f"None of {possible_variables} variables were found in the dataset.")
    if len(found_vars) > 1:
        raise ValueError(f"Multiple variables found: {found_vars}. Please specify which to use.")
    return found_vars[0]


def timestamp_to_fraction(ts: pd.Timestamp) -> float:
    """
    Convert timestamp to fraction.
    """
    return ts.toordinal() + ts.hour / 24 + ts.minute / (24 * 60) + ts.second / (24 * 60 * 60)
