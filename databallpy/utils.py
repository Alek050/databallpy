from typing import Union

import numpy as np


def _to_int(value) -> int:
    """Function to make a integer of the value if possible, else -999

    Args:
        value (): a variable value

    Returns:
       int: integer if value can be changed to integer, else -999
    """
    try:
        value = _to_float(value)
        return int(value)
    except (TypeError, ValueError):
        return -999


def _to_float(value) -> Union[float, int]:
    """Function to make a float of the value if possible, else np.nan

    Args:
        value (): a variable value

    Returns:
        Union[float, int]: integer if value can be changed to integer, else np.nan
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan
