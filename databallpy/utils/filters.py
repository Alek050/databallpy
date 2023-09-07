import numpy as np
from scipy.signal import savgol_filter


def filter_data(
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
        return savgol_filter(
            array, axis=0, window_length=window_length, polyorder=polyorder
        )

    elif filter_type == "moving_average":
        return np.convolve(array, np.ones(window_length) / window_length, mode="same")
