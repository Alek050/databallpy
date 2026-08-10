import warnings

import numpy as np
from scipy.signal import savgol_filter


def _filter_data(
    array: np.ndarray,
    filter_type: str = "savitzky_golay",
    window_length: int = 7,
    polyorder: int = 2,
) -> np.ndarray:
    """Function to filter data

    Args:
        array (np.ndarray): _description_
        filter_type (str, optional): type of filter to use. Defaults to
            "savitzky_golay". Options: {"moving_average", "savitzky_golay"}
        window_length (int, optional): Window length of the filter. Defaults to 7.
        polyorder (int, optional): polyorder to use when the savitzky_golay filter
            is selected. Defaults to 2.

    Returns:
        np.ndarray: filtered data
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("array should be of type np.ndarray")

    if filter_type not in ["moving_average", "savitzky_golay"]:
        raise ValueError(
            "filter_type should be one of: 'moving_average', 'savitzky_golay'"
            f", got: {filter_type}"
        )

    if not isinstance(window_length, int):
        raise TypeError(
            f"window_length should be of type int, not {type(window_length)}"
        )

    if not isinstance(polyorder, int):
        raise TypeError(f"polyorder should be of type int not {type(polyorder)}")

    if not len(array) > window_length:
        raise ValueError("length of data should be greater than the window length")

    if filter_type == "savitzky_golay":
        try:
            return _savgol_with_nan_compat(
                array, window_length=window_length, polyorder=polyorder, mode="interp"
            )
        except Exception as e:
            warnings.warn(
                "An unexpected error occurred while filtering "
                f"the data: {e}. /nReturning the original data."
            )
            return array

    elif filter_type == "moving_average":
        try:
            return np.convolve(
                array, np.ones(window_length) / window_length, mode="same"
            )
        except Exception as e:
            warnings.warn(
                "An unexpected error occurred while filtering "
                f"the data: {e}. /nReturning the original data."
            )
            return array


def _savgol_with_nan_compat(
    array: np.ndarray,
    window_length: int,
    polyorder: int,
    mode: str = "interp",
) -> np.ndarray:
    arr = np.asarray(array, dtype=float)

    mask = ~np.isfinite(arr)

    if not mask.any():
        return savgol_filter(arr, window_length, polyorder, mode=mode).round(2)

    valid_count = (~mask).sum()
    if valid_count < max(window_length, polyorder + 1):
        warnings.warn(
            "Not enough finite samples to apply Savitzky–Golay filter; "
            "returning original data for backward compatibility."
        )
        return arr

    x = np.arange(arr.size)
    arr_filled = arr.copy()
    arr_filled[mask] = np.interp(x[mask], x[~mask], arr[~mask])

    filtered = savgol_filter(arr_filled, window_length, polyorder, mode=mode).round(2)

    filtered[mask] = np.nan
    return filtered
