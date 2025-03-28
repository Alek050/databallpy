import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from databallpy.features.feature_utils import _check_column_ids
from databallpy.utils.warnings import deprecated


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
            return savgol_filter(
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


@deprecated(
    "The filter_tracking_data function is deprecated and will removed in version 0.8.0. Please use Game.TrackingData.filter_tracking_data() instead."
)
def filter_tracking_data(
    tracking_data: pd.DataFrame,
    column_ids: str | list[str],
    filter_type: str = "savitzky_golay",
    window_length: int = 7,
    polyorder: int = 2,
    inplace: bool = False,
) -> pd.DataFrame:
    """Function to filter tracking data in specified DataFrame columns.

    Args:
        tracking_data (pd.DataFrame): DataFrame containing tracking data.
        column_ids (str| list[str]): List of column IDs to apply the filter to.
        filter_type (str, optional): Type of filter to use. Defaults to
            "savitzky_golay". Options: {"moving_average", "savitzky_golay"}.
        window_length (int, optional): Window length of the filter. Defaults to 7.
        polyorder (int, optional): Polyorder to use when the savitzky_golay filter
            is selected. Defaults to 2.
        inplace (bool, optional): If True, modifies the DataFrame in place.
            Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with filtered data. Returns None if inplace=True.
    """
    if not isinstance(tracking_data, pd.DataFrame):
        raise TypeError(f"df should be of type pd.DataFrame, not {type(tracking_data)}")
    if isinstance(column_ids, str):
        column_ids = [column_ids]
    _check_column_ids(tracking_data, column_ids)
    if not isinstance(inplace, bool):
        raise TypeError(f"inplace should be of type bool, not {type(inplace)}")
    if not isinstance(window_length, int):
        raise TypeError(
            f"window_length should be of type int, not {type(window_length)}"
        )
    if not isinstance(polyorder, int):
        raise TypeError(f"polyorder should be of type int, not {type(polyorder)}")
    if filter_type not in ["moving_average", "savitzky_golay"]:
        raise ValueError(
            "filter_type should be one of: 'moving_average', 'savitzky_golay'"
            f", got: {filter_type}"
        )

    if not inplace:
        tracking_data = tracking_data.copy()

    xy_columns = [
        col
        for col in tracking_data.columns
        if "".join(col.split("_")[:-1]) in column_ids and col[-1] in ["x", "y"]
    ]
    for col in xy_columns:
        if filter_type == "savitzky_golay":
            tracking_data[col] = savgol_filter(
                tracking_data[col].values,
                window_length=window_length,
                polyorder=polyorder,
                mode="interp",
            )
        elif filter_type == "moving_average":
            tracking_data[col] = np.convolve(
                tracking_data[col], np.ones(window_length) / window_length, mode="same"
            )

    if not inplace:
        return tracking_data
